use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};

use log::warn;
use mehsh::prelude::{HasNeighbors, HasPosition, Mesh, Vector3D};
use ordered_float::OrderedFloat;
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;

use crate::prelude::{EdgeID, VertID, INPUT};
use crate::skeleton::boundary_loop::BoundaryLoop;
use crate::skeleton::orthogonalize::LabeledCurveSkeleton;

use super::{
    BoundaryParameterization, CutPath, CuttingPlan, SurfacePath, SurfacePoint,
    MIN_CUT_BOUNDARY_PROPORTION,
};

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
            CuttingPlan {
                cuts: Vec::new(),
                boundary_params: HashMap::new(),
            },
            CuttingPlan {
                cuts: Vec::new(),
                boundary_params: HashMap::new(),
            },
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

    // Assign shared t-values and build boundary parameterizations.
    let (input_boundary_params, polycube_boundary_params) = assign_shared_t_values_and_parameterize(
        node_idx,
        &mut input_cuts,
        &mut polycube_cuts,
        input_skeleton,
        polycube_skeleton,
        input_mesh,
        polycube_mesh,
    );

    (
        CuttingPlan {
            cuts: input_cuts,
            boundary_params: input_boundary_params,
        },
        CuttingPlan {
            cuts: polycube_cuts,
            boundary_params: polycube_boundary_params,
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
        for pt in &cut.path.points {
            if let SurfacePoint::OnVertex { vertex } = pt {
                used_verts.insert(*vertex);
            }
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
        start_midpoint_idx,
        start_t: 0.0, // assigned later
        end_boundary: edge_b,
        end_midpoint_idx,
        end_t: 0.0, // assigned later
        path,
    }
}

/// Assigns shared `t`-values to cut endpoints and builds boundary
/// parameterizations for both sides.
///
/// The input side's arc-length parameterization is the authority: each cut
/// endpoint's `t`-value is taken from the input boundary's natural
/// arc-length proportion at the midpoint where the cut touches.
///
/// The polycube side's boundary parameterizations are then *constrained* so
/// that the same `t`-values appear at the polycube's cut endpoint
/// midpoints, with the remaining midpoints distributed by arc-length
/// between constraints.
///
/// Panics if any two cut endpoints on the same boundary are closer than
/// [`MIN_CUT_BOUNDARY_PROPORTION`].
fn assign_shared_t_values_and_parameterize(
    node_idx: NodeIndex,
    input_cuts: &mut [CutPath],
    polycube_cuts: &mut [CutPath],
    input_skeleton: &LabeledCurveSkeleton,
    polycube_skeleton: &LabeledCurveSkeleton,
    input_mesh: &Mesh<INPUT>,
    polycube_mesh: &Mesh<INPUT>,
) -> (
    HashMap<EdgeIndex, BoundaryParameterization>,
    HashMap<EdgeIndex, BoundaryParameterization>,
) {
    // Collect which cut endpoints land on each boundary.
    // Each entry is (cut_index, is_start_endpoint).
    let mut boundary_cut_endpoints: HashMap<EdgeIndex, Vec<(usize, bool)>> = HashMap::new();
    for (i, cut) in input_cuts.iter().enumerate() {
        boundary_cut_endpoints
            .entry(cut.start_boundary)
            .or_default()
            .push((i, true));
        boundary_cut_endpoints
            .entry(cut.end_boundary)
            .or_default()
            .push((i, false));
    }

    // Compute input-side natural arc-length parameterizations (used to determine
    // the authoritative t-values for cut endpoints).
    let input_natural_params = parameterize_all_boundaries(node_idx, input_skeleton, input_mesh);

    // Assign t-values to cuts and collect constraints per boundary for polycube side.
    // constraints maps EdgeIndex -> Vec<(polycube_midpoint_idx, required_t)>
    let mut polycube_constraints: HashMap<EdgeIndex, Vec<(usize, f64)>> = HashMap::new();

    for (&edge_idx, endpoints) in &boundary_cut_endpoints {
        let input_param = &input_natural_params[&edge_idx];

        let mut ts_on_boundary = Vec::new();

        for &(cut_idx, is_start) in endpoints {
            // Get the input-side midpoint index and its natural t-value.
            let input_midpoint_idx = if is_start {
                input_cuts[cut_idx].start_midpoint_idx
            } else {
                input_cuts[cut_idx].end_midpoint_idx
            };
            let t = input_param.t_values[input_midpoint_idx];

            // Assign to both sides.
            if is_start {
                input_cuts[cut_idx].start_t = t;
                polycube_cuts[cut_idx].start_t = t;
            } else {
                input_cuts[cut_idx].end_t = t;
                polycube_cuts[cut_idx].end_t = t;
            }

            // Record the polycube-side constraint.
            let polycube_midpoint_idx = if is_start {
                polycube_cuts[cut_idx].start_midpoint_idx
            } else {
                polycube_cuts[cut_idx].end_midpoint_idx
            };
            polycube_constraints
                .entry(edge_idx)
                .or_default()
                .push((polycube_midpoint_idx, t));

            ts_on_boundary.push(t);
        }

        // Check minimum separation between endpoints on this boundary.
        for i in 0..ts_on_boundary.len() {
            for j in (i + 1)..ts_on_boundary.len() {
                let sep = circular_dist(ts_on_boundary[i], ts_on_boundary[j]);
                if sep < MIN_CUT_BOUNDARY_PROPORTION {
                    warn!(
                        "Cut endpoints on boundary {:?} are close: \
                         t={:.4} and t={:.4} (separation {:.4} < {:.4}). \
                         Parameterization quality may be reduced.",
                        edge_idx,
                        ts_on_boundary[i],
                        ts_on_boundary[j],
                        sep,
                        MIN_CUT_BOUNDARY_PROPORTION,
                    );
                }
            }
        }
    }

    // Build boundary parameterizations for both sides.
    let mut input_boundary_params = HashMap::new();
    let mut polycube_boundary_params = HashMap::new();

    for edge_ref in input_skeleton.edges(node_idx) {
        let edge_idx = edge_ref.id();

        // Input side: natural arc-length parameterization (already computed).
        input_boundary_params.insert(edge_idx, input_natural_params[&edge_idx].clone());

        // Polycube side: constrained parameterization.
        let polycube_boundary = &polycube_skeleton
            .edge_weight(edge_idx)
            .expect("polycube skeleton edge missing")
            .boundary_loop;
        let constraints = polycube_constraints.get(&edge_idx);
        let polycube_param = match constraints {
            Some(c) if !c.is_empty() => {
                parameterize_boundary_constrained(polycube_boundary, polycube_mesh, c)
            }
            _ => {
                // No cuts touch this boundary — use natural arc-length.
                parameterize_boundary(polycube_boundary, polycube_mesh)
            }
        };
        polycube_boundary_params.insert(edge_idx, polycube_param);
    }

    (input_boundary_params, polycube_boundary_params)
}

/// Computes arc-length parameterizations for every boundary loop incident to
/// a region node.
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
/// Sets a basis point (`t = 0`) at the midpoint with the greatest x
/// coordinate (breaking ties with y, then z). Parameters are assigned by
/// cumulative arc length along the loop, normalised to `[0, 1)`.
fn parameterize_boundary(boundary: &BoundaryLoop, mesh: &Mesh<INPUT>) -> BoundaryParameterization {
    let n = boundary.edge_midpoints.len();
    assert!(n > 0, "Boundary loop must have at least one edge");

    let midpoints = compute_midpoint_positions(boundary, mesh);

    // Find basis index: midpoint with greatest x, then y, then z.
    let basis_index = find_basis_index(&midpoints);

    let segment_lengths = compute_segment_lengths(&midpoints);
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

/// Builds a boundary parameterization where certain midpoints are
/// constrained to specific `t`-values, and the remaining midpoints are
/// distributed by arc-length proportionally between consecutive
/// constraints.
///
/// This is used for the polycube side so that cut endpoint `t`-values
/// match the input side's values exactly.
///
/// `constraints` is a slice of `(midpoint_idx, required_t)`.
fn parameterize_boundary_constrained(
    boundary: &BoundaryLoop,
    mesh: &Mesh<INPUT>,
    constraints: &[(usize, f64)],
) -> BoundaryParameterization {
    let n = boundary.edge_midpoints.len();
    assert!(n > 0, "Boundary loop must have at least one edge");
    assert!(
        !constraints.is_empty(),
        "parameterize_boundary_constrained called with no constraints"
    );

    let midpoints = compute_midpoint_positions(boundary, mesh);
    let segment_lengths = compute_segment_lengths(&midpoints);
    let total_length: f64 = segment_lengths.iter().sum();

    // Compute cumulative arc-length from index 0 for ordering purposes.
    let mut cumul_from_0 = vec![0.0; n];
    for i in 1..n {
        cumul_from_0[i] = cumul_from_0[i - 1] + segment_lengths[i - 1];
    }

    // Sort constraints by their position around the loop (arc-length from index 0).
    let mut sorted: Vec<(usize, f64, f64)> = constraints
        .iter()
        .map(|&(idx, t)| (idx, t, cumul_from_0[idx]))
        .collect();
    sorted.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));

    let nc = sorted.len();
    let mut t_values = vec![0.0; n];

    // Assign constrained midpoints.
    for &(idx, t, _) in &sorted {
        t_values[idx] = t;
    }

    // For each segment between consecutive constraints, distribute the
    // midpoints in between by arc-length proportion.
    for c in 0..nc {
        let (idx_start, t_start, _) = sorted[c];
        let (idx_end, t_end, _) = sorted[(c + 1) % nc];

        // Walk forward from idx_start to idx_end, collecting intermediate
        // midpoint indices and their cumulative arc-lengths from idx_start.
        let mut intermediates: Vec<(usize, f64)> = Vec::new(); // (midpoint_idx, arc_from_start)
        let mut arc_from_start = 0.0;
        let mut i = idx_start;
        loop {
            arc_from_start += segment_lengths[i];
            let next = (i + 1) % n;
            if next == idx_end {
                break;
            }
            intermediates.push((next, arc_from_start));
            i = next;
        }
        let total_arc = arc_from_start;

        if total_arc > 0.0 && !intermediates.is_empty() {
            // Compute the t-range for this segment (circular).
            let t_range = if t_end > t_start {
                t_end - t_start
            } else {
                t_end + 1.0 - t_start
            };

            for &(idx, arc) in &intermediates {
                let frac = arc / total_arc;
                let t = t_start + frac * t_range;
                t_values[idx] = if t >= 1.0 { t - 1.0 } else { t };
            }
        }
    }

    // Use the first constraint's midpoint as the nominal basis.
    let basis_index = sorted[0].0;

    BoundaryParameterization {
        t_values,
        total_length,
        basis_index,
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
        .unwrap_or(0)
}

/// Builds a [`SurfacePath`] from a boundary midpoint, a vertex path, and an
/// ending boundary midpoint. Interior points are all `OnVertex`.
fn build_vertex_surface_path(
    start_edge: EdgeID,
    vertex_path: &[VertID],
    end_edge: EdgeID,
) -> SurfacePath {
    let mut points = Vec::with_capacity(vertex_path.len() + 2);
    points.push(SurfacePoint::OnEdge {
        edge: start_edge,
        t: 0.5,
    });
    for &v in vertex_path {
        points.push(SurfacePoint::OnVertex { vertex: v });
    }
    points.push(SurfacePoint::OnEdge {
        edge: end_edge,
        t: 0.5,
    });
    SurfacePath { points }
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
        for pt in &cut.path.points {
            if let SurfacePoint::OnVertex { vertex } = pt {
                assert!(
                    all_verts.insert(*vertex),
                    "Cut paths are not disjoint: vertex {:?} appears in cut {} and a previous cut",
                    vertex,
                    i
                );
            }
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
