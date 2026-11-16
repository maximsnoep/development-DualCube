//! Surgery result as a skeleton: one edge per closed boundary, then subdivided until the
//! graph is simple.

use super::Surgery;
use crate::surface::edit::parallel;
use crate::{PatchBoundary, Skeleton, SkeletonEdge, SkeletonEdgeID, SkeletonNode};
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
/// each further edge between the same two nodes gets one (a path of two). A copy says nothing
/// about where a boundary belongs, so the boundaries of every such group are then spread: each
/// in turn moves to the middle between its neighbours.
pub(super) fn simplify(mesh: &Mesh<INPUT>, skeleton: &mut Skeleton) {
    let mut groups: Vec<Vec<SkeletonEdgeID>> = Vec::new();
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
        let c = boundary.centroid(mesh, &skeleton.surface);
        let [left, right] = [(p + 2. * c) / 3., (2. * p + c) / 3.]
            .map(|position| skeleton.nodes.insert(SkeletonNode { position }));
        let left_copy = parallel(mesh, skeleton, &boundary, 0);
        let right_copy = parallel(mesh, skeleton, &boundary, 1);
        skeleton.edges[edge].nodes = [left, right];
        let before = skeleton.edges.insert(SkeletonEdge {
            nodes: [node, left],
            boundary: left_copy,
        });
        let after = skeleton.edges.insert(SkeletonEdge {
            nodes: [right, node],
            boundary: right_copy,
        });
        groups.push(vec![before, edge, after]);
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
        let added = skeleton.edges.insert(SkeletonEdge {
            nodes: [a, middle],
            boundary: copy,
        });
        groups.push(vec![added, edge]);
    }
    for _ in 0..3 {
        for &edge in groups.iter().flatten() {
            if let Some(spread) = crate::label::subdivide::recentre(mesh, skeleton, edge) {
                *skeleton = spread;
            }
        }
    }
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
    edge: SkeletonEdgeID,
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
        let (rank_c, rank_d) = (
            skeleton.surface.rank(mesh, c).1,
            skeleton.surface.rank(mesh, d).1,
        );
        if y != mesh.twin(x) || rank_c.abs_diff(rank_d) != 1 {
            i += 1;
            continue;
        }
        skeleton.surface.remove(mesh, c);
        skeleton.surface.remove(mesh, d);
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
