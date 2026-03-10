use std::collections::{HashMap, HashSet};

use log::warn;
use mehsh::prelude::{HasPosition, Mesh};
use ordered_float::OrderedFloat;
use petgraph::graph::EdgeIndex;

use super::cutting_plan::{reconstruct_path, restricted_dijkstra_with_pred};
use super::{CuttingPlan, MIN_CUT_SEPARATION};
use crate::prelude::{VertID, INPUT};
use crate::skeleton::boundary_loop::BoundaryLoop;

/// Ordered boundary vertices on one side of a boundary loop, with arc-length data.
pub(super) struct OrderedBoundary {
    /// Vertices in traversal order around the boundary, on our side of the loop.
    pub vertices: Vec<VertID>,
    /// `cumulative[i]` = arc-length from `vertices[0]` to `vertices[i]` along the boundary.
    /// `cumulative[0] = 0.0`.
    pub cumulative: Vec<f64>,
    /// Total perimeter (including the closing edge from last back to first).
    pub total_length: f64,
    /// Inverse: vertex -> index in `vertices`.
    pub vert_index: HashMap<VertID, usize>,
}

impl OrderedBoundary {
    /// Normalized [0, 1) arc-length position of a vertex on this boundary.
    pub fn normalized_position(&self, v: VertID) -> f64 {
        let idx = self.vert_index[&v];
        if self.total_length > 0.0 {
            self.cumulative[idx] / self.total_length
        } else {
            0.0
        }
    }

    /// Walk from vertex `from` to vertex `to` in the forward direction (increasing index,
    /// wrapping around). Returns the sequence of vertices including both endpoints.
    /// When `from == to`, returns the full cycle (all `n` vertices, without repeating
    /// the start).
    pub fn arc_between(&self, from: VertID, to: VertID) -> Vec<VertID> {
        let from_idx = self.vert_index[&from];
        let to_idx = self.vert_index[&to];
        let n = self.vertices.len();
        let mut result = Vec::new();
        if from_idx == to_idx {
            // Full cycle: emit every vertex once.
            for k in 0..n {
                result.push(self.vertices[(from_idx + k) % n]);
            }
        } else {
            let mut i = from_idx;
            loop {
                result.push(self.vertices[i]);
                if i == to_idx {
                    break;
                }
                i = (i + 1) % n;
            }
        }
        result
    }
}

/// A shortest path through the region interior connecting two boundary loops.
pub(super) struct CutPath {
    pub boundary_a: EdgeIndex,
    pub boundary_b: EdgeIndex,
    /// Vertex sequence from `endpoint_a` on boundary_a to `endpoint_b` on boundary_b.
    pub path: Vec<VertID>,
}

impl CutPath {
    pub fn endpoint_a(&self) -> VertID {
        *self.path.first().unwrap()
    }
    pub fn endpoint_b(&self) -> VertID {
        *self.path.last().unwrap()
    }
}

/// Which side of a cut a free vertex is on (determined by rotational neighbor order).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CutSide {
    /// The side encountered first when rotating from prev->next around the cut vertex.
    Forward,
    /// The other side.
    Backward,
}

/// Finds actual cut paths on the mesh for each entry in the cutting plan.
///
/// Enforces `MIN_CUT_SEPARATION`: when multiple cuts land on the same boundary,
/// their endpoints must be at least `MIN_CUT_SEPARATION` apart in normalized [0,1]
/// arc-length. If the Dijkstra-optimal endpoint is too close, we pick the nearest
/// allowed vertex instead.
pub(super) fn find_cut_paths(
    boundaries: &HashMap<EdgeIndex, OrderedBoundary>,
    _patch_set: &HashSet<VertID>,
    patch_verts: &[VertID],
    mesh: &Mesh<INPUT>,
    cutting_plan: &CuttingPlan,
) -> Vec<CutPath> {
    let vert_to_idx: HashMap<VertID, usize> = patch_verts
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();

    // All boundary vertices (on any loop), used to prevent cuts from passing through them.
    let all_boundary_verts: HashSet<VertID> = boundaries
        .values()
        .flat_map(|b| b.vertices.iter().copied())
        .collect();

    // Track committed endpoint positions per boundary for separation checks.
    let mut committed: HashMap<EdgeIndex, Vec<f64>> = HashMap::new();
    let mut cut_paths = Vec::new();

    for &(edge_a, edge_b) in &cutting_plan.cuts {
        let boundary_a = &boundaries[&edge_a];
        let boundary_b = &boundaries[&edge_b];

        // Run Dijkstra from all vertices on boundary_a, restricted to region interior
        // (excluding vertices on other boundaries, except the two we're connecting).
        let sources: HashSet<VertID> = boundary_a.vertices.iter().copied().collect();
        let exclude: HashSet<VertID> = all_boundary_verts
            .iter()
            .copied()
            .filter(|v| !sources.contains(v) && !boundary_b.vert_index.contains_key(v))
            .collect();

        let (dist, pred) =
            restricted_dijkstra_with_pred(&sources, patch_verts, &vert_to_idx, &exclude, mesh);

        // Pick endpoint on boundary_b: closest vertex that satisfies separation.
        let committed_b_snapshot: Vec<f64> = committed.get(&edge_b).cloned().unwrap_or_default();
        let endpoint_b =
            pick_separated_endpoint(boundary_b, &dist, &vert_to_idx, &committed_b_snapshot);

        let Some(endpoint_b) = endpoint_b else {
            warn!(
                "Could not find valid endpoint on boundary {:?} for cut from {:?}",
                edge_b, edge_a
            );
            continue;
        };

        // Reconstruct path from boundary_a to endpoint_b.
        let path = reconstruct_path(&pred, patch_verts, &vert_to_idx, endpoint_b, &sources);

        if path.is_empty() {
            warn!(
                "Empty path for cut {:?} -> {:?}, boundaries may be unreachable",
                edge_a, edge_b
            );
            continue;
        }

        let endpoint_a = path[0];

        // Record committed endpoints.
        committed
            .entry(edge_a)
            .or_default()
            .push(boundary_a.normalized_position(endpoint_a));
        committed
            .entry(edge_b)
            .or_default()
            .push(boundary_b.normalized_position(endpoint_b));

        cut_paths.push(CutPath {
            boundary_a: edge_a,
            boundary_b: edge_b,
            path,
        });
    }

    cut_paths
}

/// Picks the best endpoint on a boundary: closest by Dijkstra distance
/// that is at least `MIN_CUT_SEPARATION` away from all committed endpoints.
fn pick_separated_endpoint(
    boundary: &OrderedBoundary,
    dist: &[f64],
    vert_to_idx: &HashMap<VertID, usize>,
    committed: &[f64],
) -> Option<VertID> {
    // Sort boundary vertices by distance.
    let mut candidates: Vec<(f64, VertID)> = boundary
        .vertices
        .iter()
        .filter_map(|&v| vert_to_idx.get(&v).map(|&idx| (dist[idx], v)))
        .filter(|(d, _)| d.is_finite())
        .collect();
    candidates.sort_by_key(|(d, _)| OrderedFloat(*d));

    for (_, v) in &candidates {
        let pos = boundary.normalized_position(*v);
        let ok = committed.iter().all(|&c| {
            let diff = (pos - c).abs();
            let circular_diff = diff.min(1.0 - diff);
            circular_diff >= MIN_CUT_SEPARATION
        });
        if ok {
            return Some(*v);
        }
    }
    // Fallback: just pick the closest vertex, ignoring separation.
    candidates.first().map(|(_, v)| *v)
}

/// Assembles the disk boundary by walking a spanning tree (Euler tour) over
/// the cut-connected boundary loops.
///
/// For degree d, there are exactly 4(d-1) segments.
pub(super) fn build_disk_boundary(
    boundaries: &HashMap<EdgeIndex, OrderedBoundary>,
    cut_paths: &[CutPath],
) -> Vec<Vec<VertID>> {
    // Build tree adjacency: for each boundary, which cuts attach to it?
    // Each cut appears as two directed entries (one per endpoint boundary).
    let mut adj: HashMap<EdgeIndex, Vec<CutNeighbor>> = HashMap::new();
    for boundary_eid in boundaries.keys() {
        adj.entry(*boundary_eid).or_default();
    }

    for (cut_idx, cut) in cut_paths.iter().enumerate() {
        let pos_a = boundaries[&cut.boundary_a].normalized_position(cut.endpoint_a());
        adj.entry(cut.boundary_a).or_default().push(CutNeighbor {
            cut_idx,
            endpoint_here: cut.endpoint_a(),
            target_boundary: cut.boundary_b,
            arc_position: pos_a,
        });

        let pos_b = boundaries[&cut.boundary_b].normalized_position(cut.endpoint_b());
        adj.entry(cut.boundary_b).or_default().push(CutNeighbor {
            cut_idx,
            endpoint_here: cut.endpoint_b(),
            target_boundary: cut.boundary_a,
            arc_position: pos_b,
        });
    }

    // Sort children at each boundary by arc position.
    for children in adj.values_mut() {
        children.sort_by_key(|c| OrderedFloat(c.arc_position));
    }

    // Pick an arbitrary root (first boundary in the cut tree).
    let root = if let Some(first_cut) = cut_paths.first() {
        first_cut.boundary_a
    } else {
        // No cuts ⇒ shouldn't happen for degree ≥ 2; return single boundary as fallback.
        let (&_eid, boundary) = boundaries.iter().next().unwrap();
        return vec![boundary.vertices.clone()];
    };

    let mut segments: Vec<Vec<VertID>> = Vec::new();
    let mut visited_cuts: HashSet<usize> = HashSet::new();
    euler_tour(
        root,
        None, // no parent cut
        boundaries,
        cut_paths,
        &adj,
        &mut visited_cuts,
        &mut segments,
    );
    segments
}

struct CutNeighbor {
    cut_idx: usize,
    endpoint_here: VertID,
    target_boundary: EdgeIndex,
    arc_position: f64,
}

/// Recursive Euler tour.  At each boundary, walks between cut endpoints,
/// emitting boundary arcs and cut path traversals.
fn euler_tour(
    boundary_eid: EdgeIndex,
    parent_cut: Option<usize>,
    boundaries: &HashMap<EdgeIndex, OrderedBoundary>,
    cut_paths: &[CutPath],
    adj: &HashMap<EdgeIndex, Vec<CutNeighbor>>,
    visited_cuts: &mut HashSet<usize>,
    segments: &mut Vec<Vec<VertID>>,
) {
    let boundary = &boundaries[&boundary_eid];
    let children: Vec<&CutNeighbor> = adj[&boundary_eid]
        .iter()
        .filter(|cn| Some(cn.cut_idx) != parent_cut && !visited_cuts.contains(&cn.cut_idx))
        .collect();

    // Where we enter this boundary (the parent cut's endpoint, or first child for root).
    let entry_vertex = if let Some(parent_idx) = parent_cut {
        let cut = &cut_paths[parent_idx];
        if cut.boundary_a == boundary_eid {
            cut.endpoint_a()
        } else {
            cut.endpoint_b()
        }
    } else if let Some(first_child) = children.first() {
        first_child.endpoint_here
    } else {
        // Leaf root with no children — entire boundary is one segment.
        segments.push(boundary.vertices.clone());
        return;
    };

    // Sort children by cyclic arc-length after entry_vertex.
    let entry_pos = boundary.normalized_position(entry_vertex);
    let mut sorted_children: Vec<&CutNeighbor> = children;
    sorted_children.sort_by_key(|cn| {
        let diff = cn.arc_position - entry_pos;
        OrderedFloat(if diff < 0.0 { diff + 1.0 } else { diff })
    });

    let mut current_vertex = entry_vertex;
    for child in &sorted_children {
        // Boundary arc from current_vertex to this child's endpoint.
        // Skip when they coincide (happens at root where entry = first child's endpoint).
        if current_vertex != child.endpoint_here {
            let arc = boundary.arc_between(current_vertex, child.endpoint_here);
            segments.push(arc);
        }

        // Forward cut path.
        visited_cuts.insert(child.cut_idx);
        let cut = &cut_paths[child.cut_idx];
        let forward = if cut.boundary_a == boundary_eid {
            cut.path.clone()
        } else {
            let mut rev = cut.path.clone();
            rev.reverse();
            rev
        };
        segments.push(forward);

        // Recurse into child subtree.
        euler_tour(
            child.target_boundary,
            Some(child.cut_idx),
            boundaries,
            cut_paths,
            adj,
            visited_cuts,
            segments,
        );

        // Backward cut path (return).
        let backward = if cut.boundary_a == boundary_eid {
            let mut rev = cut.path.clone();
            rev.reverse();
            rev
        } else {
            cut.path.clone()
        };
        segments.push(backward);

        current_vertex = child.endpoint_here;
    }

    // Final arc: from last child's endpoint back to entry_vertex.
    let closing_arc = boundary.arc_between(current_vertex, entry_vertex);
    // For the root, this closes the loop. For non-root, this returns to parent entry.
    segments.push(closing_arc);
}

/// Extracts ordered boundary vertices on *our* side of a boundary loop.
///
/// For each half-edge in the loop, takes the endpoint in `patch_set`.
/// Deduplicates consecutive duplicates and forms a cycle.
pub(super) fn ordered_boundary_on_our_side(
    boundary_loop: &BoundaryLoop,
    patch_set: &HashSet<VertID>,
    mesh: &Mesh<INPUT>,
) -> OrderedBoundary {
    let mut vertices = Vec::new();
    for &he in &boundary_loop.edge_midpoints {
        let r = mesh.root(he);
        let t = mesh.toor(he);
        let ours = if patch_set.contains(&r) { r } else { t };
        if vertices.last() != Some(&ours) {
            vertices.push(ours);
        }
    }
    // Remove closing duplicate (cycle).
    if vertices.len() > 1 && vertices.first() == vertices.last() {
        vertices.pop();
    }

    let n = vertices.len();
    let mut cumulative = vec![0.0_f64; n];
    for i in 1..n {
        cumulative[i] = cumulative[i - 1]
            + (mesh.position(vertices[i]) - mesh.position(vertices[i - 1])).norm();
    }
    let closing = if n > 1 {
        (mesh.position(vertices[0]) - mesh.position(vertices[n - 1])).norm()
    } else {
        0.0
    };
    let total_length = if n > 0 {
        cumulative[n - 1] + closing
    } else {
        0.0
    };

    let vert_index: HashMap<VertID, usize> =
        vertices.iter().enumerate().map(|(i, &v)| (v, i)).collect();

    OrderedBoundary {
        vertices,
        cumulative,
        total_length,
        vert_index,
    }
}
