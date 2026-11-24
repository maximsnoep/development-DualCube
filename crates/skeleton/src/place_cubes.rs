//! Signed-axis coordinates, node cubes, and stretched edge cuboids.

use crate::{
    Cuboid, CuboidID, CuboidLayout, CuboidOwner, LabeledSkeleton, RealizedSkeleton, SkeletonNodeID,
    SkeletonResult,
};
use dualcube_types::prelude::*;
use slotmap::SecondaryMap;
use std::collections::{HashMap, HashSet};

mod validate;

/// Recompute target coordinates from labels alone. Input-space positions are unchanged.
/// Two units per level leave a positive gap between unit node cubes.
pub fn coordinates(
    labeled: &LabeledSkeleton,
) -> SkeletonResult<SecondaryMap<SkeletonNodeID, Vector3D>> {
    if labeled.skeleton.nodes.is_empty() {
        return Err("Cannot place an empty skeleton".into());
    }
    labeled.validate_labels()?;
    let levels =
        crate::label::constraints::coordinate_levels(&labeled.skeleton, &labeled.labels, &[], true)
            .ok_or("Labels imply contradictory coordinate orders")?;
    // TODO: Optimize lengths for geometric fidelity beyond the two-pass level compaction.
    Ok(levels
        .iter()
        .map(|(id, p)| {
            (
                id,
                Vector3D::new(2. * p[0] as f64, 2. * p[1] as f64, 2. * p[2] as f64),
            )
        })
        .collect())
}

/// Place the nodes of a labeled skeleton whose surface is valid (as labeling leaves it); the
/// labels are checked. Can be called again after relabeling.
pub fn run(mesh: &Mesh<INPUT>, labeled: &LabeledSkeleton) -> SkeletonResult<RealizedSkeleton> {
    labeled.skeleton.debug_validate(mesh);
    realize(labeled)
}

fn realize(labeled: &LabeledSkeleton) -> SkeletonResult<RealizedSkeleton> {
    let result = RealizedSkeleton {
        labeled: labeled.clone(),
        node_positions: coordinates(labeled)?,
    };
    result.validate_geometry()?;
    tracing::info!(
        "Cube placement: {} node cubes, {} edge cuboids",
        result.node_positions.len(),
        result.labeled.skeleton.edges.len()
    );
    Ok(result)
}

impl RealizedSkeleton {
    /// Unit node cubes, then one cuboid per edge filling the gap between its endpoint cubes,
    /// each connected to both.
    pub fn cuboids(&self) -> CuboidLayout {
        let mut cuboids = CuboidLayout::default();
        let mut nodes = SecondaryMap::new();
        for id in self.labeled.skeleton.nodes.keys() {
            nodes.insert(
                id,
                cuboids.cuboids.insert(Cuboid {
                    center: self.node_positions[id],
                    size: Vector3D::repeat(1.),
                    owner: CuboidOwner::Node(id),
                }),
            );
        }
        for (edge, data) in &self.labeled.skeleton.edges {
            let [a, b] = data.nodes;
            let delta = self.node_positions[b] - self.node_positions[a];
            let axis = crate::label::Port::from_label(self.labeled.labels[edge]).axis();
            let mut size = Vector3D::repeat(1.);
            size[axis] = delta[axis].abs() - 1.;
            let connection = cuboids.cuboids.insert(Cuboid {
                center: (self.node_positions[a] + self.node_positions[b]) / 2.,
                size,
                owner: CuboidOwner::Edge(edge),
            });
            cuboids
                .connections
                .extend([[nodes[a], connection], [connection, nodes[b]]]);
        }
        cuboids
    }

    /// The solids whose free faces the loop drawing's template is read from (see
    /// `draw_loops::template`): the cubes of the nodes that are not leaves, and the edge
    /// cuboids, each stretched over the cube of a leaf end. Every free face holds one crossing
    /// of two template loops.
    pub fn template_solids(&self) -> Vec<TemplateSolid> {
        let skeleton = &self.labeled.skeleton;
        let degree = skeleton.degrees();
        let cube = |node: SkeletonNodeID| {
            let center = self.node_positions[node];
            (
                center - Vector3D::repeat(0.5),
                center + Vector3D::repeat(0.5),
            )
        };
        let mut solids: Vec<_> = skeleton
            .nodes
            .keys()
            .filter(|&n| degree[n] != 1)
            .map(|n| {
                let (lo, hi) = cube(n);
                TemplateSolid {
                    lo,
                    hi,
                    edge_axis: None,
                    hidden: [[false; 2]; 3],
                }
            })
            .collect();
        let index: HashMap<_, _> = skeleton
            .nodes
            .keys()
            .filter(|&n| degree[n] != 1)
            .zip(0..)
            .collect();
        for (edge, data) in &skeleton.edges {
            let [a, b] = data.nodes;
            let axis = crate::label::Port::from_label(self.labeled.labels[edge]).axis();
            let (pa, pb) = (self.node_positions[a], self.node_positions[b]);
            // Between the cubes, then over a leaf's cube.
            let (mut lo, mut hi) = (pa.inf(&pb), pa.sup(&pb));
            for k in 0..3 {
                let inset = if k == axis { 0.5 } else { -0.5 };
                lo[k] += inset;
                hi[k] -= inset;
            }
            let mut hidden = [[false; 2]; 3];
            for node in [a, b] {
                // The side of the cuboid toward the node.
                let positive = self.node_positions[node][axis] > (pa[axis] + pb[axis]) / 2.;
                match index.get(&node) {
                    Some(&solid) => {
                        hidden[axis][usize::from(positive)] = true;
                        solids[solid].hidden[axis][usize::from(!positive)] = true;
                    }
                    None => {
                        let (cube_lo, cube_hi) = cube(node);
                        lo = lo.inf(&cube_lo);
                        hi = hi.sup(&cube_hi);
                    }
                }
            }
            solids.push(TemplateSolid {
                lo,
                hi,
                edge_axis: Some(axis),
                hidden,
            });
        }
        solids
    }
}

/// An axis-aligned solid of the template's cube surface (see
/// [`RealizedSkeleton::template_solids`]).
#[derive(Clone, Debug, PartialEq)]
pub struct TemplateSolid {
    pub lo: Vector3D,
    pub hi: Vector3D,
    /// The axis of an edge cuboid.
    pub edge_axis: Option<usize>,
    /// The faces glued to another solid, per axis and side (negative, positive).
    pub hidden: [[bool; 2]; 3],
}

/// A template loop's crossing of a free face: the face's midline across the loop's axis.
#[derive(Clone, Debug, PartialEq)]
pub struct TemplateSection {
    /// The loop's axis.
    pub axis: usize,
    /// The face: the axis of its normal and its side.
    pub normal: usize,
    pub positive: bool,
    pub ends: [Vector3D; 2],
    /// Whether the loop is the patch boundary of an edge (around its cuboid, across its axis).
    pub boundary: bool,
}

impl TemplateSolid {
    /// The loop sections on its free faces.
    pub fn sections(&self) -> Vec<TemplateSection> {
        let center = (self.lo + self.hi) / 2.;
        let mut sections = vec![];
        for normal in 0..3 {
            for positive in [false, true] {
                if self.hidden[normal][usize::from(positive)] {
                    continue;
                }
                let plane = if positive { self.hi } else { self.lo }[normal];
                for axis in (0..3).filter(|&a| a != normal) {
                    // Along the third axis, at the middle of the loop's axis.
                    let along = 3 - normal - axis;
                    let mut ends = [center; 2];
                    for (end, value) in ends.iter_mut().zip([self.lo[along], self.hi[along]]) {
                        end[normal] = plane;
                        end[along] = value;
                    }
                    sections.push(TemplateSection {
                        axis,
                        normal,
                        positive,
                        ends,
                        boundary: self.edge_axis == Some(axis),
                    });
                }
            }
        }
        sections
    }
}

impl CuboidLayout {
    /// Faces left out where connected cuboids meet, as (cuboid, axis, positive side): the end
    /// faces of edge cuboids, so that the node cube's square face shows around them.
    pub fn hidden_faces(&self) -> HashSet<(CuboidID, usize, bool)> {
        let edge = |id: CuboidID| matches!(self.cuboids[id].owner, CuboidOwner::Edge(_));
        let mut hidden = HashSet::new();
        for &[a, b] in &self.connections {
            let delta = self.cuboids[b].center - self.cuboids[a].center;
            let Some(axis) = (0..3).find(|&i| delta[i] != 0.) else {
                continue;
            };
            let positive = delta[axis] > 0.;
            if edge(a) || !edge(b) {
                hidden.insert((a, axis, positive));
            }
            if edge(b) || !edge(a) {
                hidden.insert((b, axis, !positive));
            }
        }
        hidden
    }
}

#[cfg(test)]
mod tests;
