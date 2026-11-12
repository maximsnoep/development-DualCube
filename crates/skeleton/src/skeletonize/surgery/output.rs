//! Surgery result as a skeleton: one edge per closed boundary, then subdivided until the
//! graph is simple.

use super::Surgery;
use crate::surface::edit::parallel;
use crate::{PatchBoundary, Skeleton, SkeletonEdge, SkeletonNode};
use dualcube_types::prelude::*;

impl Surgery<'_> {
    /// The patches and closed boundaries; may contain self-loops and parallel edges.
    pub(super) fn skeleton(&self) -> Skeleton {
        let mut skeleton = Skeleton {
            surface: self.surface.clone(),
            ..Skeleton::default()
        };
        let mut nodes = HashMap::new();
        for (patch, p) in self.patches.iter().enumerate() {
            if p.is_some() {
                let position = self.input_position(patch);
                nodes.insert(patch, skeleton.nodes.insert(SkeletonNode { position }));
            }
        }
        for arc in self.arcs.values() {
            skeleton.edges.insert(SkeletonEdge {
                nodes: arc.sides.map(|p| nodes[&p]),
                boundary: PatchBoundary {
                    crossings: arc.crossings.iter().copied().collect(),
                },
            });
        }
        skeleton
    }
}

/// Subdivide with parallel boundaries: a self-loop gets one on each side (a triangle), and
/// each further edge between the same two nodes gets one (a path of two).
pub(super) fn simplify(mesh: &Mesh<INPUT>, skeleton: &mut Skeleton) {
    let loops: Vec<_> = skeleton
        .edges
        .iter()
        .filter(|(_, e)| e.nodes[0] == e.nodes[1])
        .map(|(id, _)| id)
        .collect();
    for edge in loops {
        let node = skeleton.edges[edge].nodes[0];
        let boundary = skeleton.edges[edge].boundary.clone();
        let p = skeleton.nodes[node].position;
        let c = centroid(mesh, skeleton, &boundary);
        let [left, right] = [(p + 2. * c) / 3., (2. * p + c) / 3.]
            .map(|position| skeleton.nodes.insert(SkeletonNode { position }));
        let left_copy = parallel(mesh, skeleton, &boundary, 0);
        let right_copy = parallel(mesh, skeleton, &boundary, 1);
        skeleton.edges[edge].nodes = [left, right];
        skeleton.edges.insert(SkeletonEdge {
            nodes: [node, left],
            boundary: left_copy,
        });
        skeleton.edges.insert(SkeletonEdge {
            nodes: [right, node],
            boundary: right_copy,
        });
    }
    let mut seen = HashSet::new();
    let edges: Vec<_> = skeleton.edges.keys().collect();
    for edge in edges {
        let [a, b] = skeleton.edges[edge].nodes;
        if seen.insert(if a < b { [a, b] } else { [b, a] }) {
            continue;
        }
        let boundary = skeleton.edges[edge].boundary.clone();
        let position = (skeleton.nodes[a].position + skeleton.nodes[b].position) / 2.;
        let middle = skeleton.nodes.insert(SkeletonNode { position });
        let copy = parallel(mesh, skeleton, &boundary, 0);
        skeleton.edges[edge].nodes = [middle, b];
        skeleton.edges.insert(SkeletonEdge {
            nodes: [a, middle],
            boundary: copy,
        });
    }
}

fn centroid(mesh: &Mesh<INPUT>, skeleton: &Skeleton, boundary: &PatchBoundary) -> Vector3D {
    let sum: Vector3D = boundary
        .crossings
        .iter()
        .map(|&c| {
            let exit = skeleton.surface.crossings[c].exit;
            (mesh.position(mesh.root(exit)) + mesh.position(mesh.toor(exit))) / 2.
        })
        .sum();
    sum / boundary.crossings.len() as f64
}
