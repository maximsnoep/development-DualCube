use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};

use mehsh::prelude::{HasNeighbors, HasPosition, Mesh, Vector3D};
use ordered_float::OrderedFloat;
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;

use crate::prelude::{EdgeID, VertID, INPUT};
use crate::skeleton::boundary_loop::BoundaryLoop;
use crate::skeleton::orthogonalize::LabeledCurveSkeleton;

use super::{
    BoundaryParameterization, CutPath, CuttingPlan, SurfacePoint, MIN_CUT_SEPARATION,
    geodesic::straighten_vertex_path,
};

/// Computes cutting plans for both the input and polycube sides of a region.
///
/// The cut *topology* (which boundary loops to connect) is shared and computed
/// from combined geodesic distances. Boundary parameterizations and the actual
/// cut paths are computed independently per side.
///
/// Returns `(input_plan, polycube_plan)`.
pub fn compute_cutting_plans(
    node_idx: NodeIndex,
    input_skeleton: &LabeledCurveSkeleton,
    polycube_skeleton: &LabeledCurveSkeleton,
    input_mesh: &Mesh<INPUT>,
    polycube_mesh: &Mesh<INPUT>,
) -> (CuttingPlan, CuttingPlan) {
    let degree = input_skeleton.edges(node_idx).count();

    // Step 1: Parameterize all boundaries on both sides.
    let input_params = parameterize_all_boundaries(node_idx, input_skeleton, input_mesh);
    let polycube_params = parameterize_all_boundaries(node_idx, polycube_skeleton, polycube_mesh);

    if degree < 2 {
        return (
            CuttingPlan {
                boundary_params: input_params,
                cuts: Vec::new(),
            },
            CuttingPlan {
                boundary_params: polycube_params,
                cuts: Vec::new(),
            },
        );
    }

    // Step 2: Compute shared cut topology (which boundaries to connect).
    let cut_topology = compute_cut_topology(
        node_idx,
        degree,
        input_skeleton,
        polycube_skeleton,
        input_mesh,
        polycube_mesh,
    );

    // Step 3: Find the actual cut paths on each side.
    let input_cuts = compute_cut_paths(
        node_idx,
        &cut_topology,
        input_skeleton,
        input_mesh,
        &input_params,
    );
    let polycube_cuts = compute_cut_paths(
        node_idx,
        &cut_topology,
        polycube_skeleton,
        polycube_mesh,
        &polycube_params,
    );

    (
        CuttingPlan {
            boundary_params: input_params,
            cuts: input_cuts,
        },
        CuttingPlan {
            boundary_params: polycube_params,
            cuts: polycube_cuts,
        },
    )
}

/// Computes arc-length parameterizations for every boundary loop incident to a region node.
fn parameterize_all_boundaries(
    node_idx: NodeIndex,
    skeleton: &LabeledCurveSkeleton,
    mesh: &Mesh<INPUT>,
) -> HashMap<EdgeIndex, BoundaryParameterization> {
    let mut params = HashMap::new();
    for edge_ref in skeleton.edges(node_idx) {
        let boundary = &edge_ref.weight().boundary_loop;
        let param = parameterize_boundary(boundary, mesh);
        params.insert(edge_ref.id(), param);
    }
    params
}

/// Computes the arc-length parameterization of a single boundary loop.
///
/// The basis point (`t = 0`) is the edge midpoint with the greatest x coordinate,
/// breaking ties with y, then z. Parameters are assigned by cumulative arc length
/// normalised to `[0, 1)`.
fn parameterize_boundary(boundary: &BoundaryLoop, mesh: &Mesh<INPUT>) -> BoundaryParameterization {
    let n = boundary.edge_midpoints.len();
    assert!(n > 0, "Boundary loop must have at least one edge");

    // Compute midpoint 3D positions.
    let midpoints: Vec<Vector3D> = boundary
        .edge_midpoints
        .iter()
        .map(|&e| {
            let p0 = mesh.position(mesh.root(e));
            let p1 = mesh.position(mesh.toor(e));
            (p0 + p1) * 0.5
        })
        .collect();

    // Find basis index: midpoint with greatest x, then y, then z.
    let basis_index = (0..n)
        .max_by(|&a, &b| {
            midpoints[a]
                .x
                .partial_cmp(&midpoints[b].x)
                .unwrap_or(Ordering::Equal)
                .then(
                    midpoints[a]
                        .y
                        .partial_cmp(&midpoints[b].y)
                        .unwrap_or(Ordering::Equal),
                )
                .then(
                    midpoints[a]
                        .z
                        .partial_cmp(&midpoints[b].z)
                        .unwrap_or(Ordering::Equal),
                )
        })
        .unwrap_or(0);

    // Compute segment lengths between consecutive midpoints.
    let segment_lengths: Vec<f64> = (0..n)
        .map(|i| {
            let j = (i + 1) % n;
            (midpoints[j] - midpoints[i]).norm()
        })
        .collect();

    let total_length: f64 = segment_lengths.iter().sum();

    // Assign t-values starting from the basis index.
    let mut t_values = vec![0.0; n];
    if total_length > 0.0 {
        let mut cumulative = 0.0;
        for k in 0..n {
            let idx = (basis_index + k) % n;
            t_values[idx] = cumulative / total_length;
            cumulative += segment_lengths[idx];
        }
    }

    BoundaryParameterization {
        t_values,
        total_length,
        basis_index,
    }
}

/// Determines which boundary loops to connect with cuts, using combined
/// geodesic distances. Returns `d − 1` pairs forming a minimum spanning tree.
fn compute_cut_topology(
    node_idx: NodeIndex,
    degree: usize,
    input_skeleton: &LabeledCurveSkeleton,
    polycube_skeleton: &LabeledCurveSkeleton,
    input_mesh: &Mesh<INPUT>,
    polycube_mesh: &Mesh<INPUT>,
) -> Vec<(EdgeIndex, EdgeIndex)> {
    let input_distances = pairwise_boundary_distances(node_idx, input_skeleton, input_mesh);
    let polycube_distances =
        pairwise_boundary_distances(node_idx, polycube_skeleton, polycube_mesh);

    // Normalize each side to [0, 1].
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

    let mut weighted_edges: Vec<(EdgeIndex, EdgeIndex, f64)> = Vec::new();
    for (&(a, b), &d_input) in &input_distances {
        let d_polycube = polycube_distances
            .get(&(a, b))
            .copied()
            .unwrap_or(f64::INFINITY);
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

    assert_eq!(
        mst.len(),
        degree - 1,
        "Cut topology for node {:?}: expected {} cuts but MST has {}. \
         The region's boundary loops may not be fully connected.",
        node_idx,
        degree - 1,
        mst.len()
    );

    mst
}

/// Computes actual cut paths on one mesh side for the given cut topology.
fn compute_cut_paths(
    node_idx: NodeIndex,
    cut_topology: &[(EdgeIndex, EdgeIndex)],
    skeleton: &LabeledCurveSkeleton,
    mesh: &Mesh<INPUT>,
    boundary_params: &HashMap<EdgeIndex, BoundaryParameterization>,
) -> Vec<CutPath> {
    let patch_verts = &skeleton[node_idx].skeleton_node.patch_vertices;
    let patch_set: HashSet<VertID> = patch_verts.iter().copied().collect();
    let vert_to_idx: HashMap<VertID, usize> = patch_verts
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();

    // Track which t-values on each boundary are already used, for MIN_CUT_SEPARATION.
    let mut used_t: HashMap<EdgeIndex, Vec<f64>> = HashMap::new();

    let mut cuts = Vec::new();
    for &(edge_a, edge_b) in cut_topology {
        let loop_a = &skeleton
            .edge_weight(edge_a)
            .expect("skeleton edge_a missing")
            .boundary_loop;
        let loop_b = &skeleton
            .edge_weight(edge_b)
            .expect("skeleton edge_b missing")
            .boundary_loop;

        let cut = find_cut_path(
            edge_a,
            edge_b,
            loop_a,
            loop_b,
            patch_verts,
            &patch_set,
            &vert_to_idx,
            mesh,
            boundary_params,
            &used_t,
        );

        used_t
            .entry(cut.start_boundary)
            .or_default()
            .push(cut.start_t);
        used_t.entry(cut.end_boundary).or_default().push(cut.end_t);
        cuts.push(cut);
    }

    cuts
}

/// Finds one cut path between two boundary loops.
///
/// Uses multi-source Dijkstra from boundary-A region-vertices to find the closest
/// boundary-B region-vertex, reconstructs the vertex path, then wraps it as a
/// [`SurfacePath`] whose first and last points are at boundary-edge midpoints
/// (i.e. `SurfacePoint::OnEdge { .., t: 0.5 }`).
fn find_cut_path(
    edge_a: EdgeIndex,
    edge_b: EdgeIndex,
    loop_a: &BoundaryLoop,
    loop_b: &BoundaryLoop,
    patch_verts: &[VertID],
    patch_set: &HashSet<VertID>,
    vert_to_idx: &HashMap<VertID, usize>,
    mesh: &Mesh<INPUT>,
    boundary_params: &HashMap<EdgeIndex, BoundaryParameterization>,
    used_t: &HashMap<EdgeIndex, Vec<f64>>,
) -> CutPath {
    let bverts_a = boundary_vertices_of_region(loop_a, patch_set, mesh);
    let bverts_b = boundary_vertices_of_region(loop_b, patch_set, mesh);

    // Run Dijkstra from boundary-A vertices with predecessor tracking.
    let (dist, pred) = dijkstra_with_predecessors(&bverts_a, patch_verts, vert_to_idx, mesh);

    // Find the closest boundary-B vertex.
    let mut best_b_idx: Option<usize> = None;
    let mut best_dist = f64::INFINITY;
    for &v in &bverts_b {
        if let Some(&idx) = vert_to_idx.get(&v) {
            if dist[idx] < best_dist {
                best_dist = dist[idx];
                best_b_idx = Some(idx);
            }
        }
    }

    let best_b_idx = best_b_idx.unwrap_or_else(|| {
        panic!(
            "No path found from boundary {:?} to {:?}: Dijkstra could not reach any \
             boundary-B vertex from boundary-A within the region",
            edge_a, edge_b
        )
    });

    // Reconstruct vertex path (from source in A to target in B).
    let mut path_indices = Vec::new();
    let mut current = best_b_idx;
    let max_iters = patch_verts.len() + 1;
    for _ in 0..max_iters {
        path_indices.push(current);
        match pred[current] {
            Some(prev) => current = prev,
            None => break,
        }
    }
    path_indices.reverse();

    let vertex_path: Vec<VertID> = path_indices.iter().map(|&i| patch_verts[i]).collect();

    let start_vertex = vertex_path[0];
    let end_vertex = *vertex_path.last().unwrap();

    // Find which boundary midpoint is adjacent to start / end.
    let start_midpoint_idx = find_incident_boundary_midpoint(start_vertex, loop_a, mesh);
    let end_midpoint_idx = find_incident_boundary_midpoint(end_vertex, loop_b, mesh);

    // Pick t-values, respecting MIN_CUT_SEPARATION from already-used endpoints.
    let param_a = &boundary_params[&edge_a];
    let param_b = &boundary_params[&edge_b];

    let start_t = pick_t_with_separation(start_midpoint_idx, param_a, loop_a, used_t.get(&edge_a));
    let end_t = pick_t_with_separation(end_midpoint_idx, param_b, loop_b, used_t.get(&edge_b));

    // Build the surface path: Dijkstra gives a vertex path; geodesic straightening
    // produces a path that crosses triangle edges at optimal positions.
    let start_edge = loop_a.edge_midpoints[start_midpoint_idx];
    let end_edge = loop_b.edge_midpoints[end_midpoint_idx];

    let prefix = SurfacePoint::OnEdge { edge: start_edge, t: 0.5 };
    let suffix = SurfacePoint::OnEdge { edge: end_edge, t: 0.5 };

    let path = straighten_vertex_path(
        &vertex_path,
        mesh,
        patch_set,
        Some(prefix),
        Some(suffix),
    );

    CutPath {
        start_boundary: edge_a,
        start_t,
        end_boundary: edge_b,
        end_t,
        path,
    }
}

/// Returns the t-value for a boundary midpoint, attempting to respect
/// [`MIN_CUT_SEPARATION`] from already-used t-values on the same boundary.
/// Falls back to the natural t-value if no well-separated alternative exists.
fn pick_t_with_separation(
    preferred_idx: usize,
    param: &BoundaryParameterization,
    boundary: &BoundaryLoop,
    used: Option<&Vec<f64>>,
) -> f64 {
    let t = param.t_values[preferred_idx];

    let Some(used_ts) = used else { return t };
    if used_ts.is_empty() {
        return t;
    }

    let too_close = used_ts.iter().any(|&ut| {
        let diff = (t - ut).abs();
        diff.min(1.0 - diff) < MIN_CUT_SEPARATION
    });

    if !too_close {
        return t;
    }

    // Try adjacent midpoints, alternating directions.
    let n = boundary.edge_midpoints.len();
    for offset in 1..n {
        for &dir in &[-1i32, 1] {
            let idx = ((preferred_idx as i32 + dir * offset as i32).rem_euclid(n as i32)) as usize;
            let candidate_t = param.t_values[idx];
            let ok = used_ts.iter().all(|&ut| {
                let diff = (candidate_t - ut).abs();
                diff.min(1.0 - diff) >= MIN_CUT_SEPARATION
            });
            if ok {
                return candidate_t;
            }
        }
    }

    // All midpoints are too close to an existing endpoint. Use the original
    // t-value — with very few boundary edges this can legitimately happen.
    t
}

/// Returns the index (into `boundary.edge_midpoints`) of the midpoint that has
/// `vertex` as one of its edge endpoints. If no exact incidence is found, picks
/// the closest midpoint by Euclidean distance.
fn find_incident_boundary_midpoint(
    vertex: VertID,
    boundary: &BoundaryLoop,
    mesh: &Mesh<INPUT>,
) -> usize {
    // Prefer an edge that is directly incident on this vertex.
    for (i, &e) in boundary.edge_midpoints.iter().enumerate() {
        if mesh.root(e) == vertex || mesh.toor(e) == vertex {
            return i;
        }
    }

    // Fallback: closest midpoint by Euclidean distance.
    let pos = mesh.position(vertex);
    (0..boundary.edge_midpoints.len())
        .min_by(|&a, &b| {
            let mid_a = edge_midpoint_position(boundary.edge_midpoints[a], mesh);
            let mid_b = edge_midpoint_position(boundary.edge_midpoints[b], mesh);
            let da = (mid_a - pos).norm();
            let db = (mid_b - pos).norm();
            da.partial_cmp(&db).unwrap_or(Ordering::Equal)
        })
        .unwrap_or(0)
}

/// 3D midpoint of a mesh edge.
fn edge_midpoint_position(edge: EdgeID, mesh: &Mesh<INPUT>) -> Vector3D {
    let p0 = mesh.position(mesh.root(edge));
    let p1 = mesh.position(mesh.toor(edge));
    (p0 + p1) * 0.5
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

/// Multi-source Dijkstra restricted to region vertices, with predecessor tracking.
///
/// Returns `(distances, predecessors)` where each predecessor is an index into
/// `region_verts`, or `None` for source vertices.
fn dijkstra_with_predecessors(
    sources: &HashSet<VertID>,
    region_verts: &[VertID],
    vert_to_idx: &HashMap<VertID, usize>,
    mesh: &Mesh<INPUT>,
) -> (Vec<f64>, Vec<Option<usize>>) {
    let n = region_verts.len();
    let mut dist = vec![f64::INFINITY; n];
    let mut pred: Vec<Option<usize>> = vec![None; n];
    let mut heap: BinaryHeap<Reverse<(OrderedFloat<f64>, usize)>> = BinaryHeap::new();

    for &s in sources {
        if let Some(&idx) = vert_to_idx.get(&s) {
            dist[idx] = 0.0;
            heap.push(Reverse((OrderedFloat(0.0), idx)));
        }
    }

    while let Some(Reverse((OrderedFloat(d), u_idx))) = heap.pop() {
        if d > dist[u_idx] {
            continue;
        }
        let u = region_verts[u_idx];
        for nbr in mesh.neighbors(u) {
            let Some(&nbr_idx) = vert_to_idx.get(&nbr) else {
                continue;
            };
            let edge_len = (mesh.position(u) - mesh.position(nbr)).norm();
            let new_dist = d + edge_len;
            if new_dist < dist[nbr_idx] {
                dist[nbr_idx] = new_dist;
                pred[nbr_idx] = Some(u_idx);
                heap.push(Reverse((OrderedFloat(new_dist), nbr_idx)));
            }
        }
    }

    (dist, pred)
}

/// Multi-source Dijkstra without predecessor tracking (distance-only).
fn restricted_dijkstra(
    sources: &HashSet<VertID>,
    region_verts: &[VertID],
    vert_to_idx: &HashMap<VertID, usize>,
    mesh: &Mesh<INPUT>,
) -> Vec<f64> {
    dijkstra_with_predecessors(sources, region_verts, vert_to_idx, mesh).0
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
