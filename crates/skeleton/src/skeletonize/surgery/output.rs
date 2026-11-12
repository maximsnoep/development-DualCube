//! Surgery result as a skeleton: one edge per closed boundary, then subdivided until the
//! graph is simple.

use super::Surgery;
use crate::surface::canonical;
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

/// Remove turns back across one mesh edge (into a triangle and out through the same edge),
/// which flips leave beside junctions. The innermost turn's two crossings are neighbors in
/// edge order; removing it pushes the curve out of the triangle, an isotopy that keeps the
/// boundaries disjoint, and may expose the next turn.
pub(super) fn untangle(mesh: &Mesh<INPUT>, skeleton: &mut Skeleton) {
    let edges: Vec<_> = skeleton.edges.keys().collect();
    let mut removed = 0;
    // A turn of one boundary may wait for a turn of another inside it.
    let mut last = usize::MAX;
    while removed != last {
        last = removed;
        for &edge in &edges {
            removed += untangle_boundary(mesh, skeleton, edge);
        }
    }
    if removed > 0 {
        tracing::info!("Connectivity surgery: removed {removed} turns back across a mesh edge");
    }
}

/// Remove the innermost turns of one boundary; the number removed.
fn untangle_boundary(
    mesh: &Mesh<INPUT>,
    skeleton: &mut Skeleton,
    edge: crate::SkeletonEdgeID,
) -> usize {
    let mut removed = 0;
    let mut crossings = std::mem::take(&mut skeleton.edges[edge].boundary.crossings);
    let mut i = 0;
    while crossings.len() > 2 && i < crossings.len() {
        let j = (i + 1) % crossings.len();
        let (c, d) = (crossings[i], crossings[j]);
        let (x, y) = (
            skeleton.surface.crossings[c].exit,
            skeleton.surface.crossings[d].exit,
        );
        let order = &skeleton.surface.edge_order[&canonical(mesh, x)];
        let at = |id| order.iter().position(|&o| o == id).unwrap();
        if y != mesh.twin(x) || at(c).abs_diff(at(d)) != 1 {
            i += 1;
            continue;
        }
        let order = skeleton
            .surface
            .edge_order
            .get_mut(&canonical(mesh, x))
            .unwrap();
        order.retain(|&o| o != c && o != d);
        if order.is_empty() {
            skeleton.surface.edge_order.remove(&canonical(mesh, x));
        }
        skeleton.surface.crossings.remove(c);
        skeleton.surface.crossings.remove(d);
        if j == 0 {
            crossings.remove(i);
            crossings.remove(0);
        } else {
            crossings.drain(i..=j);
        }
        removed += 1;
        // The neighbors now meet across the removed triangle and may turn back too.
        i = i.saturating_sub(1).min(crossings.len().saturating_sub(1));
    }
    skeleton.edges[edge].boundary.crossings = crossings;
    removed
}
