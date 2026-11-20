//! Signed-axis coordinates, node cubes, and stretched edge cuboids.

use crate::{
    Cuboid, CuboidID, CuboidLayout, CuboidOwner, LabeledSkeleton, RealizedSkeleton, SkeletonNodeID,
    SkeletonResult,
};
use dualcube_types::prelude::*;
use slotmap::SecondaryMap;
use std::collections::HashSet;

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
