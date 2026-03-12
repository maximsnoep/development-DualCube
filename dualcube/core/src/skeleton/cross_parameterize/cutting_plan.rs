use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};

use log::error;
use mehsh::prelude::{HasNeighbors, HasPosition, Mesh};
use ordered_float::OrderedFloat;
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;

use crate::prelude::{VertID, INPUT};
use crate::skeleton::boundary_loop::BoundaryLoop;
use crate::skeleton::orthogonalize::LabeledCurveSkeleton;
use super::CuttingPlan;

/// Computes a shared cutting plan for a region, combining geodesic distances from
/// both the input and polycube sides to select an MST over the boundary loops.
///
/// For degree < 2, returns an empty plan (no cuts needed).
/// For degree >= 2, returns d-1 cuts forming a minimum spanning tree.
pub fn compute_cutting_plan(
    node_idx: NodeIndex,
    input_skeleton: &LabeledCurveSkeleton,
    polycube_skeleton: &LabeledCurveSkeleton,
    input_mesh: &Mesh<INPUT>,
    polycube_mesh: &Mesh<INPUT>,
) -> CuttingPlan {
    let degree = input_skeleton.edges(node_idx).count();

    if degree < 2 {
        return CuttingPlan { cuts: Vec::new() };
    }

    // Compute pairwise shortest-path distances between boundary loops on each side.
    let input_distances = pairwise_boundary_distances(node_idx, input_skeleton, input_mesh);
    let polycube_distances = pairwise_boundary_distances(node_idx, polycube_skeleton, polycube_mesh);

    // Normalize each side by its maximum finite distance, so both contribute on [0, 1].
    let input_max = input_distances
        .values()
        .copied()
        .filter(|d| d.is_finite())
        .fold(0.0, f64::max);
    let polycube_max = polycube_distances
        .values()
        .copied()
        .filter(|d| d.is_finite())
        .fold(0.0, f64::max);

    // Build combined cost for every boundary pair.
    let mut weighted_edges: Vec<(EdgeIndex, EdgeIndex, f64)> = Vec::new();
    for (&(a, b), &d_input) in &input_distances {
        let d_polycube = polycube_distances.get(&(a, b)).copied().unwrap_or(f64::INFINITY);

        let norm_input = if input_max > 0.0 {
            d_input / input_max
        } else {
            0.0
        };
        let norm_polycube = if polycube_max > 0.0 {
            d_polycube / polycube_max
        } else {
            0.0
        };

        weighted_edges.push((a, b, norm_input + norm_polycube));
    }

    let mst = kruskal_mst(&weighted_edges);

    if mst.len() != degree - 1 {
        error!(
            "Cutting plan for node {:?}: expected {} cuts but MST has {}.",
            node_idx,
            degree - 1,
            mst.len()
        );
    }

    CuttingPlan { cuts: mst }
}

/// Returns the set of vertices *within this region* that lie on a specific boundary loop.
///
/// Each half-edge in the boundary loop crosses between two patches; we keep the
/// endpoint that belongs to our patch.
fn boundary_vertices_of_region(
    boundary_loop: &BoundaryLoop,
    patch_set: &HashSet<VertID>,
    mesh: &Mesh<INPUT>,
) -> HashSet<VertID> {
    let mut result = HashSet::new();
    for &e in &boundary_loop.edge_midpoints {
        let r = mesh.root(e);
        let t = mesh.toor(e);
        if patch_set.contains(&r) {
            result.insert(r);
        }
        if patch_set.contains(&t) {
            result.insert(t);
        }
    }
    result
}

/// Computes pairwise shortest geodesic distances between all boundary loops of a
/// region on one mesh side.
///
/// Returns a map from canonically-ordered `(EdgeIndex, EdgeIndex)` pairs (smaller
/// index first) to the minimum vertex-to-vertex distance through the region interior.
fn pairwise_boundary_distances(
    node_idx: NodeIndex,
    skeleton: &LabeledCurveSkeleton,
    mesh: &Mesh<INPUT>,
) -> HashMap<(EdgeIndex, EdgeIndex), f64> {
    let patch_verts = &skeleton[node_idx].skeleton_node.patch_vertices;
    let patch_set: HashSet<VertID> = patch_verts.iter().copied().collect();

    // Build a vertex-to-index map for the region (used by Dijkstra).
    let vert_to_idx: HashMap<VertID, usize> = patch_verts
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();

    // Collect each incident edge's boundary vertices.
    let mut boundary_info: Vec<(EdgeIndex, HashSet<VertID>)> = Vec::new();
    for edge_ref in skeleton.edges(node_idx) {
        let bverts =
            boundary_vertices_of_region(&edge_ref.weight().boundary_loop, &patch_set, mesh);
        boundary_info.push((edge_ref.id(), bverts));
    }

    // Run one multi-source Dijkstra per boundary.
    let all_dists: Vec<Vec<f64>> = boundary_info
        .iter()
        .map(|(_, bverts)| restricted_dijkstra(bverts, patch_verts, &vert_to_idx, mesh))
        .collect();

    // Extract pairwise minimum distances.
    let mut result = HashMap::new();
    for i in 0..boundary_info.len() {
        for j in (i + 1)..boundary_info.len() {
            let (eid_i, _) = &boundary_info[i];
            let (eid_j, bverts_j) = &boundary_info[j];

            let min_dist = bverts_j
                .iter()
                .filter_map(|v| vert_to_idx.get(v).map(|&idx| all_dists[i][idx]))
                .fold(f64::INFINITY, f64::min);

            // Store with canonical order (smaller EdgeIndex first).
            let key = if *eid_i < *eid_j {
                (*eid_i, *eid_j)
            } else {
                (*eid_j, *eid_i)
            };
            result.insert(key, min_dist);
        }
    }

    result
}

/// Multi-source Dijkstra restricted to vertices within the region.
///
/// Returns a distance vector indexed by position in `region_verts`.
/// Unreachable vertices keep their initial inf value, should not happen.
fn restricted_dijkstra(
    sources: &HashSet<VertID>,
    region_verts: &[VertID],
    vert_to_idx: &HashMap<VertID, usize>,
    mesh: &Mesh<INPUT>,
) -> Vec<f64> {
    let n = region_verts.len();
    let mut dist = vec![f64::INFINITY; n];
    let mut heap: BinaryHeap<Reverse<(OrderedFloat<f64>, usize)>> = BinaryHeap::new();

    for &s in sources {
        if let Some(&idx) = vert_to_idx.get(&s) {
            dist[idx] = 0.0;
            heap.push(Reverse((OrderedFloat(0.0), idx)));
        }
    }

    while let Some(Reverse((OrderedFloat(d), u_idx))) = heap.pop() {
        if d > dist[u_idx] {
            continue; // stale entry
        }
        let u = region_verts[u_idx];
        for nbr in mesh.neighbors(u) {
            // Only traverse edges within the region.
            let Some(&nbr_idx) = vert_to_idx.get(&nbr) else {
                continue;
            };
            let edge_len = (mesh.position(u) - mesh.position(nbr)).norm();
            let new_dist = d + edge_len;
            if new_dist < dist[nbr_idx] {
                dist[nbr_idx] = new_dist;
                heap.push(Reverse((OrderedFloat(new_dist), nbr_idx)));
            }
        }
    }

    dist
}

/// Kruskal's MST on a small complete graph of boundary loops.
///
/// Takes a list of `(node_a, node_b, cost)` triples (where "nodes" are skeleton
/// `EdgeIndex` values identifying boundary loops) and returns the MST edges.
fn kruskal_mst(weighted_edges: &[(EdgeIndex, EdgeIndex, f64)]) -> Vec<(EdgeIndex, EdgeIndex)> {
    let mut sorted: Vec<_> = weighted_edges.to_vec();
    sorted.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));

    // Union-find with path compression.
    let mut parent: HashMap<EdgeIndex, EdgeIndex> = HashMap::new();

    fn find(parent: &mut HashMap<EdgeIndex, EdgeIndex>, x: EdgeIndex) -> EdgeIndex {
        let p = *parent.entry(x).or_insert(x);
        if p == x {
            return x;
        }
        let root = find(parent, p);
        parent.insert(x, root);
        root
    }

    let mut result = Vec::new();
    for &(a, b, cost) in &sorted {
        if !cost.is_finite() {
            continue;
        }
        let ra = find(&mut parent, a);
        let rb = find(&mut parent, b);
        if ra != rb {
            parent.insert(ra, rb);
            result.push((a, b));
        }
    }

    result
}
