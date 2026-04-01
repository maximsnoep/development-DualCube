use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};

use log::warn;
use mehsh::prelude::{HasNeighbors, HasPosition, Mesh, Vector3D};
use ordered_float::OrderedFloat;
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;

use crate::prelude::{EdgeID, VertID, INPUT};
use crate::skeleton::boundary_loop::BoundaryLoop;
use crate::skeleton::cross_parameterize::boundary_walk::calculate_boundary_loop_reversal_flags;
use crate::skeleton::orthogonalize::LabeledCurveSkeleton;

use super::coordination::{
    BoundaryFrame, CutCycleOrder, CutEndpointSpec, PolycubeCandidate, RegionCoordination,
    path_length,
};
use super::{CutPath, CuttingPlan, SurfacePath};

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
    let input_cuts = compute_side_cut_paths(node_idx, &cut_topology, input_skeleton, input_mesh);
    let polycube_cuts =
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

// ─────────────────────────────────────────────────────────────────────────────
// Phase B: boundary frames + constrained cut paths
// ─────────────────────────────────────────────────────────────────────────────

/// Builds an oriented `BoundaryFrame` for every boundary loop of a region.
///
/// Orientation is determined by `calculate_boundary_loop_reversal_flags`: if a
/// boundary's raw `edge_midpoints` order places the patch on the right, we
/// reverse + twin so the patch is on the left (CCW convention).
pub fn compute_boundary_frames(
    node_idx: NodeIndex,
    skeleton: &LabeledCurveSkeleton,
    mesh: &Mesh<INPUT>,
) -> HashMap<EdgeIndex, BoundaryFrame> {
    let reverse_flags = calculate_boundary_loop_reversal_flags(node_idx, skeleton, mesh);
    let mut frames = HashMap::new();

    for edge_ref in skeleton.edges(node_idx) {
        let skel_edge = edge_ref.id();
        let boundary = &edge_ref.weight().boundary_loop;
        let reversed = *reverse_flags.get(&skel_edge).unwrap_or(&false);

        // Slots are always the canonical edge IDs from edge_midpoints (the ones stored in the
        // VFG's edge_midpoint_ids_to_node_indices map). Reversal only changes the ORDER,
        // not the edge IDs — taking twins would produce IDs absent from the VFG map.
        let slots: Vec<EdgeID> = if reversed {
            boundary.edge_midpoints.iter().copied().rev().collect()
        } else {
            boundary.edge_midpoints.clone()
        };

        assert!(
            slots.len() >= 2,
            "Boundary loop on skeleton edge {:?} has only {} slots (need >= 2)",
            skel_edge,
            slots.len()
        );

        frames.insert(
            skel_edge,
            BoundaryFrame {
                skeleton_edge: skel_edge,
                orientation_reversed: reversed,
                slots,
            },
        );
    }

    frames
}

/// Finds the shortest cut path from a specific start-slot edge to a specific
/// end-slot edge, with constrained source/target vertex sets.
///
/// Sources = patch vertices incident to `start_slot_edge`.
/// Targets = patch vertices incident to `end_slot_edge`.
/// Returns `None` when Dijkstra cannot reach any target (caller rejects candidate).
fn find_constrained_cut_path(
    edge_a: EdgeIndex,
    edge_b: EdgeIndex,
    start_slot_edge: EdgeID,
    end_slot_edge: EdgeID,
    patch_verts: &[VertID],
    patch_set: &HashSet<VertID>,
    vert_to_idx: &HashMap<VertID, usize>,
    mesh: &Mesh<INPUT>,
    forbidden_verts: &HashSet<VertID>,
) -> Option<CutPath> {
    // Vertices incident to the start slot that are inside the patch.
    let sources: HashSet<VertID> = [mesh.root(start_slot_edge), mesh.toor(start_slot_edge)]
        .into_iter()
        .filter(|v| patch_set.contains(v) && !forbidden_verts.contains(v))
        .collect();

    if sources.is_empty() {
        return None;
    }

    // Vertices incident to the end slot that are inside the patch.
    let target_verts: HashSet<VertID> = [mesh.root(end_slot_edge), mesh.toor(end_slot_edge)]
        .into_iter()
        .filter(|v| patch_set.contains(v))
        .collect();

    if target_verts.is_empty() {
        return None;
    }

    let (dist, pred) =
        dijkstra_with_predecessors(&sources, patch_verts, vert_to_idx, mesh, forbidden_verts);

    // Pick the closest reachable target.
    let best_b_idx = target_verts
        .iter()
        .filter(|v| !forbidden_verts.contains(v))
        .filter_map(|v| vert_to_idx.get(v).copied())
        .filter(|&idx| dist[idx].is_finite())
        .min_by(|&a, &b| dist[a].partial_cmp(&dist[b]).unwrap_or(Ordering::Equal))?;

    // Reconstruct vertex path.
    let mut path_indices = Vec::new();
    let mut current = best_b_idx;
    for _ in 0..=patch_verts.len() {
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
        "Constrained cut path between {:?} and {:?} reconstructed empty",
        edge_a,
        edge_b
    );

    let path = build_vertex_surface_path(start_slot_edge, &vertex_path, end_slot_edge);
    Some(CutPath {
        start_boundary: edge_a,
        end_boundary: edge_b,
        path,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase C: polycube candidate enumeration + input tie-break scoring
// ─────────────────────────────────────────────────────────────────────────────

/// Enumerates all feasible polycube slot assignments for the given cut topology.
///
/// Feasibility constraints per candidate:
/// 1. No two endpoints on the same boundary occupy the same slot.
/// 2. Two endpoints on the same boundary must not be adjacent (distance 1 mod N).
/// 3. Every constrained Dijkstra path must be realizable.
///
/// Panics if no feasible candidate exists.
pub fn enumerate_polycube_candidates(
    node_idx: NodeIndex,
    cut_topology: &[(EdgeIndex, EdgeIndex)],
    polycube_frames: &HashMap<EdgeIndex, BoundaryFrame>,
    polycube_skeleton: &LabeledCurveSkeleton,
    polycube_mesh: &Mesh<INPUT>,
) -> Vec<PolycubeCandidate> {
    let patch_verts = &polycube_skeleton[node_idx].skeleton_node.patch_vertices;
    let patch_set: HashSet<VertID> = patch_verts.iter().copied().collect();
    let vert_to_idx: HashMap<VertID, usize> = patch_verts
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();

    let mut results = Vec::new();

    enumerate_candidates_recursive(
        cut_topology,
        0,
        polycube_frames,
        patch_verts,
        &patch_set,
        &vert_to_idx,
        polycube_mesh,
        &mut HashMap::new(),
        &mut HashSet::new(),
        &mut Vec::new(),
        &mut Vec::new(),
        &mut results,
    );

    assert!(
        !results.is_empty(),
        "No feasible polycube candidate found for region {:?} with {} cuts",
        node_idx,
        cut_topology.len()
    );

    results
}

#[allow(clippy::too_many_arguments)]
fn enumerate_candidates_recursive(
    cut_topology: &[(EdgeIndex, EdgeIndex)],
    cut_idx: usize,
    polycube_frames: &HashMap<EdgeIndex, BoundaryFrame>,
    patch_verts: &[VertID],
    patch_set: &HashSet<VertID>,
    vert_to_idx: &HashMap<VertID, usize>,
    mesh: &Mesh<INPUT>,
    boundary_used_slots: &mut HashMap<EdgeIndex, HashSet<usize>>,
    used_interior_verts: &mut HashSet<VertID>,
    partial_assignments: &mut Vec<(EdgeIndex, usize, EdgeIndex, usize)>,
    partial_cuts: &mut Vec<CutPath>,
    results: &mut Vec<PolycubeCandidate>,
) {
    if cut_idx == cut_topology.len() {
        let primary_score: f64 = partial_cuts
            .iter()
            .map(|c| path_length(&c.path, mesh))
            .sum();
        results.push(PolycubeCandidate {
            assignments: partial_assignments.clone(),
            cuts: partial_cuts.clone(),
            primary_score,
        });
        return;
    }

    let (edge_a, edge_b) = cut_topology[cut_idx];
    let frame_a = &polycube_frames[&edge_a];
    let frame_b = &polycube_frames[&edge_b];

    let slots_a: Vec<usize> = (0..frame_a.num_slots())
        .filter(|s| {
            !boundary_used_slots
                .get(&edge_a)
                .map(|u| u.contains(s))
                .unwrap_or(false)
        })
        .collect();

    let slots_b: Vec<usize> = (0..frame_b.num_slots())
        .filter(|s| {
            !boundary_used_slots
                .get(&edge_b)
                .map(|u| u.contains(s))
                .unwrap_or(false)
        })
        .collect();

    for &sa in &slots_a {
        for &sb in &slots_b {
            // Same boundary: reject if adjacent.
            if edge_a == edge_b && (sa == sb || frame_a.slots_adjacent(sa, sb)) {
                continue;
            }

            let start_slot_edge = frame_a.slot_edge(sa);
            let end_slot_edge = frame_b.slot_edge(sb);

            let Some(cut) = find_constrained_cut_path(
                edge_a,
                edge_b,
                start_slot_edge,
                end_slot_edge,
                patch_verts,
                patch_set,
                vert_to_idx,
                mesh,
                used_interior_verts,
            ) else {
                continue;
            };

            let new_verts: Vec<VertID> = cut.path.interior_verts.clone();

            boundary_used_slots.entry(edge_a).or_default().insert(sa);
            boundary_used_slots.entry(edge_b).or_default().insert(sb);
            for &v in &new_verts {
                used_interior_verts.insert(v);
            }
            partial_assignments.push((edge_a, sa, edge_b, sb));
            partial_cuts.push(cut);

            enumerate_candidates_recursive(
                cut_topology,
                cut_idx + 1,
                polycube_frames,
                patch_verts,
                patch_set,
                vert_to_idx,
                mesh,
                boundary_used_slots,
                used_interior_verts,
                partial_assignments,
                partial_cuts,
                results,
            );

            partial_cuts.pop();
            partial_assignments.pop();
            for &v in &new_verts {
                used_interior_verts.remove(&v);
            }
            boundary_used_slots.entry(edge_a).or_default().remove(&sa);
            boundary_used_slots.entry(edge_b).or_default().remove(&sb);
        }
    }
}

/// Maps a polycube slot index to the proportionally equivalent input slot index.
///
/// Polycube slot `pc_slot` out of `pc_total` represents arc fraction `pc_slot / pc_total`.
/// The input slot is the nearest index to that same arc fraction within `in_total` slots.
fn map_polycube_slot_to_input(pc_slot: usize, pc_total: usize, in_total: usize) -> usize {
    let pos = pc_slot as f64 / pc_total as f64;
    (pos * in_total as f64).round() as usize % in_total
}

/// Scores a polycube candidate on the input side (tie-break).
///
/// Proportionally maps polycube slot indices to input slot indices, then runs
/// constrained Dijkstra. Returns `None` if any cut is unrealizable.
pub fn score_input_candidate(
    candidate: &PolycubeCandidate,
    node_idx: NodeIndex,
    polycube_frames: &HashMap<EdgeIndex, BoundaryFrame>,
    input_frames: &HashMap<EdgeIndex, BoundaryFrame>,
    input_skeleton: &LabeledCurveSkeleton,
    input_mesh: &Mesh<INPUT>,
) -> Option<(f64, Vec<CutPath>)> {
    let patch_verts = &input_skeleton[node_idx].skeleton_node.patch_vertices;
    let patch_set: HashSet<VertID> = patch_verts.iter().copied().collect();
    let vert_to_idx: HashMap<VertID, usize> = patch_verts
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();

    let mut used_verts: HashSet<VertID> = HashSet::new();
    let mut input_cuts = Vec::new();
    let mut total_len = 0.0;

    for &(edge_a, sa, edge_b, sb) in &candidate.assignments {
        let pc_frame_a = polycube_frames.get(&edge_a)?;
        let pc_frame_b = polycube_frames.get(&edge_b)?;
        let input_frame_a = input_frames.get(&edge_a)?;
        let input_frame_b = input_frames.get(&edge_b)?;

        // Proportional mapping: polycube slot at arc fraction sa/pc_total → nearest input slot.
        let sa_input =
            map_polycube_slot_to_input(sa, pc_frame_a.num_slots(), input_frame_a.num_slots());
        let sb_input =
            map_polycube_slot_to_input(sb, pc_frame_b.num_slots(), input_frame_b.num_slots());

        let start_slot_edge = input_frame_a.slot_edge(sa_input);
        let end_slot_edge = input_frame_b.slot_edge(sb_input);

        let cut = find_constrained_cut_path(
            edge_a,
            edge_b,
            start_slot_edge,
            end_slot_edge,
            patch_verts,
            &patch_set,
            &vert_to_idx,
            input_mesh,
            &used_verts,
        )?;

        total_len += path_length(&cut.path, input_mesh);
        for &v in &cut.path.interior_verts {
            used_verts.insert(v);
        }
        input_cuts.push(cut);
    }

    Some((total_len, input_cuts))
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase D: candidate selection + CutCycleOrder + compute_region_coordination
// ─────────────────────────────────────────────────────────────────────────────

/// Builds the canonical `CutCycleOrder` from the selected endpoint assignments.
///
/// Walks every boundary frame's slot list in deterministic order (sorted by
/// `EdgeIndex`). The first event visited becomes `events[0]` (phase anchor).
fn build_cut_cycle_order(
    endpoint_specs: &[CutEndpointSpec],
    polycube_frames: &HashMap<EdgeIndex, BoundaryFrame>,
) -> CutCycleOrder {
    let spec_map: HashMap<(EdgeIndex, usize), &CutEndpointSpec> = endpoint_specs
        .iter()
        .map(|s| ((s.boundary, s.slot_id), s))
        .collect();

    let mut boundary_order: Vec<EdgeIndex> = polycube_frames.keys().copied().collect();
    boundary_order.sort_by_key(|e| e.index());

    let mut events: Vec<CutEndpointSpec> = Vec::new();
    for boundary in boundary_order {
        let frame = &polycube_frames[&boundary];
        for slot_id in 0..frame.num_slots() {
            if let Some(&spec) = spec_map.get(&(boundary, slot_id)) {
                events.push(spec.clone());
            }
        }
    }

    assert_eq!(
        events.len(),
        endpoint_specs.len(),
        "CutCycleOrder event count mismatch: expected {}, got {}",
        endpoint_specs.len(),
        events.len()
    );

    CutCycleOrder { events }
}

/// Lexicographically selects the best candidate:
/// primary = polycube path length, secondary = input path length, tertiary = canonical assignment key.
///
/// Panics if all candidates fail input-side realization.
fn select_candidate(
    mut polycube_candidates: Vec<PolycubeCandidate>,
    node_idx: NodeIndex,
    polycube_frames: &HashMap<EdgeIndex, BoundaryFrame>,
    input_frames: &HashMap<EdgeIndex, BoundaryFrame>,
    input_skeleton: &LabeledCurveSkeleton,
    input_mesh: &Mesh<INPUT>,
) -> (Vec<CutEndpointSpec>, Vec<CutPath>, Vec<CutPath>) {
    polycube_candidates.sort_by(|a, b| {
        a.primary_score
            .partial_cmp(&b.primary_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                canonical_assignment_key(&a.assignments)
                    .cmp(&canonical_assignment_key(&b.assignments))
            })
    });

    // Lex pass: iterate in primary-score order, tracking the best (primary, secondary) seen.
    let mut best_primary = f64::INFINITY;
    let mut best_secondary = f64::INFINITY;
    let mut best: Option<(Vec<CutEndpointSpec>, Vec<CutPath>, Vec<CutPath>)> = None;

    for candidate in &polycube_candidates {
        // Once primary score strictly exceeds the best already accepted, we can stop.
        if candidate.primary_score > best_primary + 1e-12 {
            break;
        }

        let Some((input_score, input_cuts)) = score_input_candidate(
            candidate,
            node_idx,
            polycube_frames,
            input_frames,
            input_skeleton,
            input_mesh,
        ) else {
            continue;
        };

        let primary = candidate.primary_score;
        let is_better = best.is_none()
            || primary < best_primary - 1e-12
            || (primary <= best_primary + 1e-12 && input_score < best_secondary - 1e-12);

        if is_better {
            best_primary = primary;
            best_secondary = input_score;

            let endpoint_specs: Vec<CutEndpointSpec> = candidate
                .assignments
                .iter()
                .enumerate()
                .flat_map(|(cut_id, &(edge_a, sa, edge_b, sb))| {
                    [
                        CutEndpointSpec {
                            cut_id,
                            boundary: edge_a,
                            slot_id: sa,
                            is_start: true,
                        },
                        CutEndpointSpec {
                            cut_id,
                            boundary: edge_b,
                            slot_id: sb,
                            is_start: false,
                        },
                    ]
                })
                .collect();

            best = Some((endpoint_specs, candidate.cuts.clone(), input_cuts));
        }
    }

    let (endpoint_specs, polycube_cuts, input_cuts) = best.unwrap_or_else(|| {
        panic!(
            "No candidate for region {:?} survived input-side realization",
            node_idx
        )
    });

    (endpoint_specs, polycube_cuts, input_cuts)
}

/// Stable canonical key for deterministic tie-breaking.
fn canonical_assignment_key(
    assignments: &[(EdgeIndex, usize, EdgeIndex, usize)],
) -> Vec<(usize, usize, usize, usize)> {
    let mut key: Vec<(usize, usize, usize, usize)> = assignments
        .iter()
        .map(|&(ea, sa, eb, sb)| (ea.index(), sa, eb.index(), sb))
        .collect();
    key.sort();
    key
}

/// Main entry point: computes the fully coordinated region artifact (replaces `compute_cutting_plans`).
///
/// For degree < 2 returns an empty coordination (no cuts needed).
pub fn compute_region_coordination(
    node_idx: NodeIndex,
    input_skeleton: &LabeledCurveSkeleton,
    polycube_skeleton: &LabeledCurveSkeleton,
    input_mesh: &Mesh<INPUT>,
    polycube_mesh: &Mesh<INPUT>,
) -> RegionCoordination {
    let degree = input_skeleton.edges(node_idx).count();

    let polycube_frames = compute_boundary_frames(node_idx, polycube_skeleton, polycube_mesh);
    let input_frames = compute_boundary_frames(node_idx, input_skeleton, input_mesh);

    if degree < 2 {
        return RegionCoordination {
            polycube_frames,
            input_frames,
            endpoint_specs: Vec::new(),
            cycle_order: CutCycleOrder { events: Vec::new() },
            polycube_cuts: Vec::new(),
            input_cuts: Vec::new(),
        };
    }

    let cut_topology = compute_cut_topology(
        node_idx,
        degree,
        input_skeleton,
        polycube_skeleton,
        input_mesh,
        polycube_mesh,
    );

    let polycube_candidates = enumerate_polycube_candidates(
        node_idx,
        &cut_topology,
        &polycube_frames,
        polycube_skeleton,
        polycube_mesh,
    );

    let (endpoint_specs, polycube_cuts, input_cuts) = select_candidate(
        polycube_candidates,
        node_idx,
        &polycube_frames,
        &input_frames,
        input_skeleton,
        input_mesh,
    );

    let cycle_order = build_cut_cycle_order(&endpoint_specs, &polycube_frames);

    RegionCoordination {
        polycube_frames,
        input_frames,
        endpoint_specs,
        cycle_order,
        polycube_cuts,
        input_cuts,
    }
}
