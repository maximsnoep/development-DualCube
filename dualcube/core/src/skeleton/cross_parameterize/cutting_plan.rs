use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};

use log::{error, info, warn};
use mehsh::prelude::{HasNeighbors, HasPosition, Mesh, Vector3D};
use ordered_float::OrderedFloat;
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;

use crate::prelude::{EdgeID, VertID, INPUT};
use crate::skeleton::boundary_loop::BoundaryLoop;
use crate::skeleton::orthogonalize::LabeledCurveSkeleton;

use super::{CutPath, CuttingPlan, SurfacePath, MIN_CUT_BOUNDARY_PROPORTION};

/// Computes cutting plans for both the input and polycube sides of a region.
///
/// 1. Compute cut topology (MST over combined geodesic distances).
/// 2. Find cut paths independently on each side (shortest vertex-disjoint
///    Dijkstra paths).
/// 3. Assign shared `t`-values to cut endpoints based on input-side
///    arc-length proportions.
/// 4. Build boundary parameterizations: natural arc-length for the input
///    side, constrained (warped) for the polycube side so that cut
///    endpoints match.
///
/// Cut paths follow mesh edges exclusively. The only face-traversing
/// segments are the first and last: boundary-midpoint to the nearest
/// (internal) patch vertex.
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

    if degree < 2 {
        return (
            CuttingPlan { cuts: Vec::new() },
            CuttingPlan { cuts: Vec::new() },
        );
    }

    // Compute shared cut topology (which boundaries to connect).
    let cut_topology = compute_cut_topology(
        node_idx,
        degree,
        input_skeleton,
        polycube_skeleton,
        input_mesh,
        polycube_mesh,
    );

    // Find cut paths on both sides independently.
    let mut input_cuts =
        compute_side_cut_paths(node_idx, &cut_topology, input_skeleton, input_mesh);
    let mut polycube_cuts =
        compute_side_cut_paths(node_idx, &cut_topology, polycube_skeleton, polycube_mesh);

    // Ensure cut endpoints on the same boundary are not adjacent (<=1 midpoint
    // index apart). Adjacent endpoints produce an empty polygon side in UV.
    // ensure_endpoint_separation(&mut input_cuts, input_skeleton, input_mesh, node_idx);
    // ensure_endpoint_separation(
    //     &mut polycube_cuts,
    //     polycube_skeleton,
    //     polycube_mesh,
    //     node_idx,
    // );
    warn!("TODO: what to do with endpoints close to eachother?");

    (
        CuttingPlan { cuts: input_cuts },
        CuttingPlan {
            cuts: polycube_cuts,
        },
    )
}

/// Computes cut paths for one side (input or polycube) independently.
///
/// Finds shortest vertex-disjoint Dijkstra paths between boundary loops
/// according to the given cut topology. The `start_t` and `end_t` fields
/// of the returned [`CutPath`]s are initialised to `0.0` and must be
/// assigned later by [`assign_shared_t_values_and_parameterize`].
fn compute_side_cut_paths(
    node_idx: NodeIndex,
    cut_topology: &[(EdgeIndex, EdgeIndex)],
    skeleton: &LabeledCurveSkeleton,
    mesh: &Mesh<INPUT>,
) -> Vec<CutPath> {
    warn!("TODO: do we even wanna use this?!");
    let patch_verts = &skeleton[node_idx].skeleton_node.patch_vertices;
    let patch_set: HashSet<VertID> = patch_verts.iter().copied().collect();
    let vert_to_idx: HashMap<VertID, usize> = patch_verts
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();

    let mut used_verts: HashSet<VertID> = HashSet::new();
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

        let cut = find_shortest_cut_path(
            edge_a,
            edge_b,
            loop_a,
            loop_b,
            patch_verts,
            &patch_set,
            &vert_to_idx,
            mesh,
            &used_verts,
        );

        // Record used vertices for disjointness.
        for pt in &cut.path.interior_verts {
            used_verts.insert(*pt);
        }
        cuts.push(cut);
    }

    verify_cuts_disjoint(&cuts);
    cuts
}

/// Finds the shortest cut path between two boundary loops.
///
/// Uses multi-source Dijkstra from boundary-A region-vertices to find the
/// closest boundary-B region-vertex, reconstructs the vertex path, then
/// wraps it as a [`SurfacePath`] whose first and last points are at
/// boundary-edge midpoints (`SurfacePoint::OnEdge { .., t: 0.5 }`).
///
/// The `start_t` and `end_t` fields are left as `0.0`; they are assigned
/// later once shared boundary parameterizations are computed.
fn find_shortest_cut_path(
    edge_a: EdgeIndex,
    edge_b: EdgeIndex,
    loop_a: &BoundaryLoop,
    loop_b: &BoundaryLoop,
    patch_verts: &[VertID],
    patch_set: &HashSet<VertID>,
    vert_to_idx: &HashMap<VertID, usize>,
    mesh: &Mesh<INPUT>,
    forbidden_verts: &HashSet<VertID>,
) -> CutPath {
    let bverts_a = boundary_vertices_of_region(loop_a, patch_set, mesh);
    let bverts_b = boundary_vertices_of_region(loop_b, patch_set, mesh);

    // Run Dijkstra from boundary-A vertices, avoiding forbidden interior vertices.
    let sources: HashSet<VertID> = bverts_a
        .iter()
        .copied()
        .filter(|v| !forbidden_verts.contains(v))
        .collect();
    assert!(
        !sources.is_empty(),
        "All boundary-A vertices for {:?} are forbidden — cannot route cut",
        edge_a
    );

    let (dist, pred) =
        dijkstra_with_predecessors(&sources, patch_verts, vert_to_idx, mesh, forbidden_verts);

    // Find the closest reachable boundary-B vertex.
    let mut best_b_idx: Option<usize> = None;
    let mut best_dist = f64::INFINITY;
    for &v in &bverts_b {
        if forbidden_verts.contains(&v) {
            continue;
        }
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
    assert!(
        !vertex_path.is_empty(),
        "Cut path between {:?} and {:?} has no vertices",
        edge_a,
        edge_b
    );

    let start_vertex = vertex_path[0];
    let end_vertex = *vertex_path.last().unwrap();

    // Find which boundary midpoint is adjacent to start / end.
    let start_midpoint_idx = find_incident_boundary_midpoint(start_vertex, loop_a, mesh);
    let end_midpoint_idx = find_incident_boundary_midpoint(end_vertex, loop_b, mesh);

    // Build the surface path: [boundary_midpoint, v0, v1, …, vn, boundary_midpoint].
    let start_edge = loop_a.edge_midpoints[start_midpoint_idx];
    let end_edge = loop_b.edge_midpoints[end_midpoint_idx];

    let path = build_vertex_surface_path(start_edge, &vertex_path, end_edge);

    CutPath {
        start_boundary: edge_a,
        end_boundary: edge_b,
        path,
    }
}

/// Computes 3D midpoint positions for all edges in a boundary loop.
fn compute_midpoint_positions(boundary: &BoundaryLoop, mesh: &Mesh<INPUT>) -> Vec<Vector3D> {
    boundary
        .edge_midpoints
        .iter()
        .map(|&e| {
            let p0 = mesh.position(mesh.root(e));
            let p1 = mesh.position(mesh.toor(e));
            (p0 + p1) * 0.5
        })
        .collect()
}

/// Computes segment lengths between consecutive midpoints (cyclic).
fn compute_segment_lengths(midpoints: &[Vector3D]) -> Vec<f64> {
    let n = midpoints.len();
    (0..n)
        .map(|i| (midpoints[(i + 1) % n] - midpoints[i]).norm())
        .collect()
}

/// Finds the basis index: midpoint with greatest x, then y, then z.
fn find_basis_index(midpoints: &[Vector3D]) -> usize {
    (0..midpoints.len())
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
        .expect("Boundary loop has no midpoints")
}

fn build_vertex_surface_path(
    start_edge: EdgeID,
    vertex_path: &[VertID],
    end_edge: EdgeID,
) -> SurfacePath {
    SurfacePath {
        start: start_edge,
        interior_verts: vertex_path.to_vec(),
        end: end_edge,
    }
}

/// Circular distance on [0, 1).
fn circular_dist(a: f64, b: f64) -> f64 {
    let d = (a - b).abs();
    d.min(1.0 - d)
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

/// Asserts that no two cut paths share any mesh vertex.
fn verify_cuts_disjoint(cuts: &[CutPath]) {
    let mut all_verts: HashSet<VertID> = HashSet::new();
    for (i, cut) in cuts.iter().enumerate() {
        for vertex in &cut.path.interior_verts {
            assert!(
                all_verts.insert(*vertex),
                "Cut paths are not disjoint: vertex {:?} appears in cut {} and a previous cut",
                vertex,
                i
            );
        }
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

/// Computes pairwise shortest vertex-to-vertex distances between all boundary
/// loops of a region on one mesh side.
fn pairwise_boundary_distances(
    node_idx: NodeIndex,
    skeleton: &LabeledCurveSkeleton,
    mesh: &Mesh<INPUT>,
) -> HashMap<(EdgeIndex, EdgeIndex), f64> {
    let patch_verts = &skeleton[node_idx].skeleton_node.patch_vertices;
    let patch_set: HashSet<VertID> = patch_verts.iter().copied().collect();

    let vert_to_idx: HashMap<VertID, usize> = patch_verts
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();

    let mut boundary_info: Vec<(EdgeIndex, HashSet<VertID>)> = Vec::new();
    for edge_ref in skeleton.edges(node_idx) {
        let bverts =
            boundary_vertices_of_region(&edge_ref.weight().boundary_loop, &patch_set, mesh);
        boundary_info.push((edge_ref.id(), bverts));
    }

    let no_forbidden = HashSet::new();
    let all_dists: Vec<Vec<f64>> = boundary_info
        .iter()
        .map(|(_, bverts)| {
            dijkstra_with_predecessors(bverts, patch_verts, &vert_to_idx, mesh, &no_forbidden).0
        })
        .collect();

    let mut result = HashMap::new();
    for i in 0..boundary_info.len() {
        for j in (i + 1)..boundary_info.len() {
            let (eid_i, _) = &boundary_info[i];
            let (eid_j, bverts_j) = &boundary_info[j];

            let min_dist = bverts_j
                .iter()
                .filter_map(|v| vert_to_idx.get(v).map(|&idx| all_dists[i][idx]))
                .fold(f64::INFINITY, f64::min);

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
/// Vertices in `forbidden` are not traversed (but sources are always included).
///
/// Returns `(distances, predecessors)` where each predecessor is an index into
/// `region_verts`, or `None` for source vertices.
fn dijkstra_with_predecessors(
    sources: &HashSet<VertID>,
    region_verts: &[VertID],
    vert_to_idx: &HashMap<VertID, usize>,
    mesh: &Mesh<INPUT>,
    forbidden: &HashSet<VertID>,
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
            if forbidden.contains(&nbr) {
                continue;
            }
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

/// Kruskal's MST on a small complete graph of boundary loops.
fn kruskal_mst(weighted_edges: &[(EdgeIndex, EdgeIndex, f64)]) -> Vec<(EdgeIndex, EdgeIndex)> {
    let mut sorted: Vec<_> = weighted_edges.to_vec();
    sorted.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));

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
