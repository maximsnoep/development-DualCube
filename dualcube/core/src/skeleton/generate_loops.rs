use std::{
    collections::{BinaryHeap, HashMap, HashSet},
    f64::consts::PI,
};

use bimap::BiHashMap;
use ordered_float::OrderedFloat;
use log::{error, warn};
use mehsh::prelude::{HasFaces, HasPosition, HasVertices, Mesh, Vector3D};
use petgraph::{
    graph::{EdgeIndex, NodeIndex},
    visit::{EdgeRef, IntoEdgeReferences, IntoNodeReferences},
};
use slotmap::SlotMap;

use crate::{
    prelude::{EdgeID, FaceID, PrincipalDirection, VertID, INPUT},
    skeleton::{
        boundary_loop::BoundaryLoop,
        orthogonalize::{AxisSign, LabeledCurveSkeleton, LabeledSkeletonSignExt},
        SkeletonData,
    },
    solutions::{Loop, LoopID},
};

const ALL_DIRS: [PrincipalDirection; 3] = [
    PrincipalDirection::X,
    PrincipalDirection::Y,
    PrincipalDirection::Z,
];
const ALL_SIGNS: [AxisSign; 2] = [AxisSign::Positive, AxisSign::Negative];

/// Per boundary loop, the crossing points for each orthogonal (direction, sign).
pub type CrossingMap = HashMap<LoopID, HashMap<(PrincipalDirection, AxisSign), EdgeID>>;

/// Per node, a face point for each (direction, sign) slot that has no neighboring patch.
/// Each face point is an interior mesh edge midpoint on the patch surface.
pub type FacePointMap = HashMap<NodeIndex, HashMap<(PrincipalDirection, AxisSign), EdgeID>>;

pub enum LoopGenerationError {
    MissingLabeledSkeleton,
    // TODO other error variants
}

/// Generates surface-embedded loops from a polycube and polycube map.
pub fn generate_loops(
    skeleton_data: &SkeletonData,
    mesh: &Mesh<INPUT>,
) -> Result<(SlotMap<LoopID, Loop>, CrossingMap, FacePointMap), LoopGenerationError> {
    let mut map: SlotMap<LoopID, Loop> = SlotMap::with_key();

    let skeleton: &LabeledCurveSkeleton = skeleton_data
        .labeled_skeleton
        .as_ref()
        .ok_or_else(|| LoopGenerationError::MissingLabeledSkeleton)?;
    let (boundary_map, crossings) = get_boundaries_and_crossing_points(skeleton, mesh, &mut map);
    let face_points = compute_face_points(skeleton, mesh);

    // Trace paths between boundary points and face points to create the loops
    pathing_for_loops(
        boundary_map,
        crossings.clone(), // TODO: later we can simply consume as we no longer need to return it
        face_points.clone(), // TODO: same here
        skeleton,
        mesh,
        &mut map,
    );

    Ok((map, crossings, face_points))
}

/// Calculates for each patch-patch boundary the appropriate loop and crossing points for the other two loop types.
///
/// For each boundary loop, places 4 crossings by:
/// 1. Projecting the boundary points onto the loop's plane (perpendicular to the skeleton edge).
/// 2. Computing angles from the centroid in that plane.
/// 3. For each orthogonal (direction, sign), projecting the axis direction onto the plane to
///    get a target angle, then picking the boundary point closest to that target angle.
///
/// This ensures crossings are naturally spread around the loop (~90° apart for axis-aligned geometry)
/// without needing direction propagation between nodes.
fn get_boundaries_and_crossing_points(
    skeleton: &LabeledCurveSkeleton,
    mesh: &Mesh<INPUT>,
    map: &mut SlotMap<LoopID, Loop>,
) -> (BiHashMap<EdgeIndex, LoopID>, CrossingMap) {
    let mut crossings: CrossingMap = HashMap::new();
    let mut boundary_map = BiHashMap::new();

    for edge in skeleton.edge_references() {
        let weight = edge.weight();
        let direction = weight.direction;
        let boundary = weight.boundary_loop.clone();

        // Create loop and save its ID
        let loop_id = map.insert(get_loop(boundary.clone(), direction));
        boundary_map.insert(edge.id(), loop_id);

        // Compute centroid of boundary loop
        let n = boundary.edge_midpoints.len() as f64;
        let centroid: Vector3D = boundary
            .edge_midpoints
            .iter()
            .fold(Vector3D::zeros(), |acc, &e| acc + mesh.position(e))
            / n;

        // Loop plane normal from skeleton edge geometry
        let source_pos = skeleton[edge.source()].skeleton_node.position;
        let target_pos = skeleton[edge.target()].skeleton_node.position;
        let normal = (target_pos - source_pos).normalize();

        // Build orthonormal basis (u, v) on the plane perpendicular to the normal
        let arbitrary = if normal.x.abs() < 0.9 {
            Vector3D::new(1.0, 0.0, 0.0)
        } else {
            Vector3D::new(0.0, 1.0, 0.0)
        };
        let u = normal.cross(&arbitrary).normalize();
        let v = normal.cross(&u); // already unit length

        // Compute angle of each boundary point relative to centroid in the loop plane
        let point_angles: Vec<(usize, f64)> = boundary
            .edge_midpoints
            .iter()
            .enumerate()
            .map(|(i, &e)| {
                let offset = mesh.position(e) - centroid;
                let proj_u = offset.dot(&u);
                let proj_v = offset.dot(&v);
                (i, proj_v.atan2(proj_u))
            })
            .collect();

        // For each orthogonal (direction, sign), project the axis direction onto the loop plane
        // to get a target angle, then pick the boundary point closest in angle.
        let ortho: Vec<_> = ALL_DIRS
            .iter()
            .copied()
            .filter(|&d| d != direction)
            .collect();
        let mut loop_crossings = HashMap::new();

        for &dir in &ortho {
            // A `dir`-type loop crosses this boundary at positions spread along the
            // THIRD direction T = third(dir, boundary_dir).  Keying by (T, sign)
            // means crossing (T,+) is at the +T side and (T,-) at the -T side,
            // matching the face-point sign convention on adjacent nodes.
            let t_dir = third(dir, direction);
            for sign in ALL_SIGNS {
                let axis_vec = match sign {
                    AxisSign::Positive => Vector3D::from(t_dir),
                    AxisSign::Negative => -Vector3D::from(t_dir),
                };
                let target_angle = axis_vec.dot(&v).atan2(axis_vec.dot(&u));

                let &(best_idx, _) = point_angles
                    .iter()
                    .min_by(|(_, a), (_, b)| {
                        let diff_a = angle_distance(*a, target_angle);
                        let diff_b = angle_distance(*b, target_angle);
                        diff_a
                            .partial_cmp(&diff_b)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .expect("boundary loop should not be empty");

                loop_crossings.insert((t_dir, sign), boundary.edge_midpoints[best_idx]);
            }
        }

        crossings.insert(loop_id, loop_crossings);
    }

    (boundary_map, crossings)
}

/// For each skeleton node, finds an interior mesh edge midpoint for every (direction, sign)
/// slot that does not already have a neighboring patch (skeleton edge).
///
/// Boundary centroids from existing edges are used internally as direction constraints:
/// if the opposite sign of the same direction has a boundary, the face point is placed
/// directly opposite it. Otherwise falls back to the global axis direction.
pub fn compute_face_points(skeleton: &LabeledCurveSkeleton, mesh: &Mesh<INPUT>) -> FacePointMap {
    let mut result: FacePointMap = HashMap::new();

    for (node_idx, node_weight) in skeleton.node_references() {
        let node_pos = node_weight.skeleton_node.position;
        let patch_set: HashSet<VertID> = node_weight
            .skeleton_node
            .patch_vertices
            .iter()
            .copied()
            .collect();

        // Record boundary centroids and their directions.
        // These are NOT stored in the output, only used to guide placement of missing slots.
        let mut occupied: HashSet<(PrincipalDirection, AxisSign)> = HashSet::new();
        let mut boundary_centroids: HashMap<(PrincipalDirection, AxisSign), Vector3D> =
            HashMap::new();
        let mut known_dirs: Vec<Vector3D> = Vec::new();

        let degree = skeleton.edges(node_idx).count();

        for edge_ref in skeleton.edges(node_idx) {
            let ew = edge_ref.weight();
            let dir = ew.direction;
            // edge_ref.source() on StableUnGraph returns the stored first endpoint,
            // NOT necessarily the querying node. Use edge_endpoints to check correctly.
            let (stored_a, _) = skeleton
                .edge_endpoints(edge_ref.id())
                .expect("edge must exist");
            let sign = if stored_a == node_idx {
                ew.sign
            } else {
                ew.sign.flipped()
            };

            if occupied.contains(&(dir, sign)) {
                warn!(
                    "Node {:?} (degree {}): duplicate slot ({:?}, {:?}) — edge {:?} \
                     (stored sign {:?}, stored_source={:?}, this node={:?})",
                    node_idx,
                    degree,
                    dir,
                    sign,
                    edge_ref.id(),
                    ew.sign,
                    stored_a,
                    node_idx
                );
            }

            let boundary = &ew.boundary_loop;
            let n = boundary.edge_midpoints.len() as f64;
            let centroid: Vector3D = boundary
                .edge_midpoints
                .iter()
                .fold(Vector3D::zeros(), |acc, &e| acc + mesh.position(e))
                / n;

            let dir_vec = (centroid - node_pos).normalize();
            known_dirs.push(dir_vec);
            occupied.insert((dir, sign));
            boundary_centroids.insert((dir, sign), centroid);
        }

        // Collect candidate interior edge midpoints
        // Exclude edges that lie on any boundary loop (those are near patch borders).
        let mut boundary_edges: HashSet<(VertID, VertID)> = HashSet::new();
        for edge_ref in skeleton.edges(node_idx) {
            for &e in &edge_ref.weight().boundary_loop.edge_midpoints {
                let a = mesh.root(e);
                let b = mesh.toor(e);
                let key = if a < b { (a, b) } else { (b, a) };
                boundary_edges.insert(key);
            }
        }

        let mut seen: HashSet<(VertID, VertID)> = HashSet::new();
        let mut candidates: Vec<EdgeID> = Vec::new();

        for &v in &node_weight.skeleton_node.patch_vertices {
            for face in mesh.faces(v) {
                let verts: Vec<VertID> = mesh.vertices(face).collect();
                for i in 0..verts.len() {
                    let a = verts[i];
                    let b = verts[(i + 1) % verts.len()];
                    if !patch_set.contains(&a) || !patch_set.contains(&b) {
                        continue;
                    }
                    let key = if a < b { (a, b) } else { (b, a) };
                    if boundary_edges.contains(&key) {
                        continue;
                    }
                    if seen.insert(key) {
                        if let Some((e, _)) = mesh.edge_between_verts(a, b) {
                            candidates.push(e);
                        }
                    }
                }
            }
        }

        if candidates.is_empty() {
            warn!(
                "Node {:?} has no interior edge candidates for missing face points.",
                node_idx
            );
        }

        // Fill missing slots (only directions without a skeleton edge)
        let mut interior_points: HashMap<(PrincipalDirection, AxisSign), EdgeID> = HashMap::new();

        for dir in ALL_DIRS {
            for sign in ALL_SIGNS {
                if occupied.contains(&(dir, sign)) {
                    continue;
                }
                if candidates.is_empty() {
                    continue;
                }

                let opposite_sign = sign.flipped();
                let target_dir =
                    if let Some(&centroid) = boundary_centroids.get(&(dir, opposite_sign)) {
                        // Opposite side has a boundary: place directly opposite its centroid
                        -(centroid - node_pos).normalize()
                    } else if let Some(&edge) = interior_points.get(&(dir, opposite_sign)) {
                        // Opposite side was already placed as an interior point: go opposite
                        -(edge_midpoint_pos(edge, mesh) - node_pos).normalize()
                    } else {
                        // No opposite exists yet: use global axis direction
                        match sign {
                            AxisSign::Positive => Vector3D::from(dir),
                            AxisSign::Negative => -Vector3D::from(dir),
                        }
                    };

                let best = *candidates
                    .iter()
                    .max_by(|&&e1, &&e2| {
                        let v1 = (edge_midpoint_pos(e1, mesh) - node_pos).normalize();
                        let v2 = (edge_midpoint_pos(e2, mesh) - node_pos).normalize();
                        let dot1 = v1.dot(&target_dir);
                        let dot2 = v2.dot(&target_dir);
                        dot1.partial_cmp(&dot2).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .expect("candidates is non-empty");

                let best_pos = edge_midpoint_pos(best, mesh);
                known_dirs.push((best_pos - node_pos).normalize());
                interior_points.insert((dir, sign), best);
            }
        }

        assert_eq!(
            occupied.len(),
            degree,
            "Node {:?} (degree {}): only {} unique (dir, sign) slots — \
             upstream orthogonalization assigned duplicate slots. Occupied: {:?}",
            node_idx,
            degree,
            occupied.len(),
            occupied
        );
        debug_assert_eq!(
            degree + interior_points.len(),
            6,
            "Node {:?}: degree ({}) + face_points ({}) != 6",
            node_idx,
            degree,
            interior_points.len()
        );

        result.insert(node_idx, interior_points);
    }

    result
}

/// Position of an edge's midpoint.
fn edge_midpoint_pos(e: EdgeID, mesh: &Mesh<INPUT>) -> Vector3D {
    let a = mesh.position(mesh.root(e));
    let b = mesh.position(mesh.toor(e));
    (a + b) * 0.5
}

/// Minimum angle (radians) between `v` and any vector in `others`. Returns π if `others` is empty.
fn min_angle_to(v: &Vector3D, others: &[Vector3D]) -> f64 {
    others
        .iter()
        .map(|d| d.dot(v).clamp(-1.0, 1.0).acos())
        .fold(PI, f64::min)
}

/// Shortest angular distance between two angles in radians.
fn angle_distance(a: f64, b: f64) -> f64 {
    let mut d = (a - b) % (2.0 * PI);
    if d > PI {
        d -= 2.0 * PI;
    } else if d < -PI {
        d += 2.0 * PI;
    }
    d.abs()
}

fn get_loop(boundary: BoundaryLoop, direction: PrincipalDirection) -> Loop {
    Loop {
        edges: boundary.edge_midpoints,
        direction,
    }
}

/// Cost multiplier applied when entering or leaving a face that has at least one blocked edge.
/// Applied symmetrically on both sides, so two adjacent-to-loop faces incur 4x total.
const SHARED_EDGE_MULTIPLIER: f64 = 2.0;

/// Dijkstra's on the mesh dual graph to find the shortest intermediate path between two
/// control-point edges, using geodesic (face-centroid -> edge-midpoint -> face-centroid) cost.
///
/// Returns the mesh edges crossed between `source` and `target`, **exclusive** of both.
/// The caller is responsible for pushing `source` before and `target` after this list.
///
/// Blocked edges (other loops) act as hard walls: the path may not cross them.
/// Faces adjacent to blocked edges incur a `SHARED_EDGE_MULTIPLIER` cost penalty when
/// entering or leaving, encouraging paths to stay away from existing loops.
/// Returns `None` if no path exists.
fn surface_path_intermediates(
    source: EdgeID,
    target: EdgeID,
    blocked: &HashSet<EdgeID>,
    mesh: &Mesh<INPUT>,
) -> Option<Vec<EdgeID>> {
    let faces_of = |e: EdgeID| -> Vec<FaceID> {
        let a = mesh.root(e);
        let b = mesh.toor(e);
        let set_a: HashSet<FaceID> = mesh.faces(a).collect();
        mesh.faces(b).filter(|f| set_a.contains(f)).collect()
    };

    let face_centroid = |f: FaceID| -> Vector3D {
        let verts: Vec<VertID> = mesh.vertices(f).collect();
        let n = verts.len() as f64;
        verts.iter().fold(Vector3D::zeros(), |acc, &v| acc + mesh.position(v)) / n
    };

    // Returns true if the face has at least one edge in `blocked` (other than `except`).
    let face_touches_blocked = |f: FaceID, except: EdgeID| -> bool {
        let verts: Vec<VertID> = mesh.vertices(f).collect();
        let n = verts.len();
        for i in 0..n {
            let a = verts[i];
            let b = verts[(i + 1) % n];
            if let Some((e, _)) = mesh.edge_between_verts(a, b) {
                if e != except && blocked.contains(&e) {
                    return true;
                }
            }
        }
        false
    };

    let source_faces = faces_of(source);
    let target_face_set: HashSet<FaceID> = faces_of(target).into_iter().collect();

    // If source and target already share a face, no intermediate edges are needed.
    if source_faces.iter().any(|f| target_face_set.contains(f)) {
        return Some(vec![]);
    }

    // Dijkstra. dist: best known cost to reach a face.
    // prev: face -> (parent_face, edge_used_to_reach_it).
    // Heap entries: (Reverse(cost), face).
    let mut dist: HashMap<FaceID, f64> = HashMap::new();
    let mut prev: HashMap<FaceID, (FaceID, Option<EdgeID>)> = HashMap::new();
    let mut heap: BinaryHeap<(std::cmp::Reverse<OrderedFloat<f64>>, FaceID)> = BinaryHeap::new();

    for &sf in &source_faces {
        dist.insert(sf, 0.0);
        prev.insert(sf, (sf, None));
        heap.push((std::cmp::Reverse(OrderedFloat(0.0)), sf));
    }

    let mut found: Option<FaceID> = None;

    'dijkstra: while let Some((std::cmp::Reverse(OrderedFloat(cost)), face)) = heap.pop() {
        if dist.get(&face).copied().unwrap_or(f64::INFINITY) < cost {
            continue; // stale entry
        }

        let face_verts: Vec<VertID> = mesh.vertices(face).collect();
        let n = face_verts.len();
        let centroid_a = face_centroid(face);

        for i in 0..n {
            let a = face_verts[i];
            let b = face_verts[(i + 1) % n];
            let Some((edge, _)) = mesh.edge_between_verts(a, b) else { continue };

            if blocked.contains(&edge) {
                continue;
            }

            // The adjacent face across this edge.
            let set_a: HashSet<FaceID> = mesh.faces(a).collect();
            let Some(next_face) = mesh.faces(b).find(|&f| set_a.contains(&f) && f != face)
            else {
                continue;
            };

            let edge_mid = edge_midpoint_pos(edge, mesh);
            let centroid_b = face_centroid(next_face);
            let mut step_cost = (centroid_a - edge_mid).norm() + (edge_mid - centroid_b).norm();

            // Penalize leaving a face adjacent to blocked edges.
            if face_touches_blocked(face, edge) {
                step_cost *= SHARED_EDGE_MULTIPLIER;
            }
            // Penalize entering a face adjacent to blocked edges.
            if face_touches_blocked(next_face, edge) {
                step_cost *= SHARED_EDGE_MULTIPLIER;
            }

            let new_cost = cost + step_cost;
            if new_cost < dist.get(&next_face).copied().unwrap_or(f64::INFINITY) {
                dist.insert(next_face, new_cost);
                prev.insert(next_face, (face, Some(edge)));

                if target_face_set.contains(&next_face) {
                    found = Some(next_face);
                    break 'dijkstra;
                }

                heap.push((std::cmp::Reverse(OrderedFloat(new_cost)), next_face));
            }
        }
    }

    // Reconstruct intermediate edges (source and target excluded).
    let end_face = found?;
    let mut path: Vec<EdgeID> = Vec::new();
    let mut current = end_face;
    loop {
        let (parent, edge_opt) = prev[&current];
        match edge_opt {
            None => break, // reached a start face
            Some(edge) => path.push(edge),
        }
        current = parent;
    }
    path.reverse();
    Some(path)
}

fn pathing_for_loops(
    boundary_map: BiHashMap<EdgeIndex, LoopID>,
    crossings: CrossingMap,
    face_points: FacePointMap,
    skeleton: &LabeledCurveSkeleton,
    mesh: &Mesh<INPUT>,
    map: &mut SlotMap<LoopID, Loop>,
) {
    for &loop_axis in &ALL_DIRS {
        // Crossings visited by this loop axis: a crossing on a boundary with direction D and
        // dir_sign (A, s) is visited by loops with axis third(D, A), so filter by that.
        let mut unvisited_crossings: HashSet<(LoopID, (PrincipalDirection, AxisSign))> = crossings
            .iter()
            .flat_map(|(&loop_id, dir_sign_map)| {
                let &edge_idx = boundary_map
                    .get_by_right(&loop_id)
                    .expect("every loop has a boundary edge");
                let boundary_dir = skeleton
                    .edge_weight(edge_idx)
                    .expect("edge must exist")
                    .direction;
                dir_sign_map
                    .keys()
                    .filter(|(a, _)| third(boundary_dir, *a) == loop_axis)
                    .map(|&ds| (loop_id, ds))
                    .collect::<Vec<_>>()
            })
            .collect();

        // Face points visited by this loop axis: any face point whose dir_sign direction ≠ loop_axis.
        // (Each face point is visited once per orthogonal loop type, so we rebuild per axis.)
        let mut unvisited_face_points: HashSet<(NodeIndex, (PrincipalDirection, AxisSign))> =
            face_points
                .iter()
                .flat_map(|(&node, dir_sign_map)| {
                    dir_sign_map
                        .keys()
                        .filter(|(dir, _)| *dir != loop_axis)
                        .map(|&ds| (node, ds))
                        .collect::<Vec<_>>()
                })
                .collect();

        // Blocked edges: all edges in loops established before this direction.
        // These act as walls in the dual-graph Dijkstra (other loops must not be crossed).
        let blocked: HashSet<EdgeID> = map.values()
            .flat_map(|l| l.edges.iter().copied())
            .collect();

        // Repeatedly pick any unvisited point and trace the full loop it belongs to.
        while !unvisited_crossings.is_empty() || !unvisited_face_points.is_empty() {
            let start = if let Some(&(loop_id, dir_sign)) = unvisited_crossings.iter().next() {
                NextPoint::Crossing { loop_id, dir_sign }
            } else {
                let &(patch, dir_sign) = unvisited_face_points.iter().next().unwrap();
                NextPoint::FacePoint { patch, dir_sign }
            };

            // First pass: collect control-point edges in order.
            let mut current = start;
            let mut control_points: Vec<EdgeID> = Vec::new();
            loop {
                let edge_id = match current {
                    NextPoint::Crossing { loop_id, dir_sign } => crossings[&loop_id][&dir_sign],
                    NextPoint::FacePoint { patch, dir_sign } => face_points[&patch][&dir_sign],
                };
                match current {
                    NextPoint::Crossing { loop_id, dir_sign } => {
                        unvisited_crossings.remove(&(loop_id, dir_sign));
                    }
                    NextPoint::FacePoint { patch, dir_sign } => {
                        unvisited_face_points.remove(&(patch, dir_sign));
                    }
                }
                control_points.push(edge_id);
                current = next_point(current, loop_axis, skeleton, &boundary_map);
                if current == start {
                    break;
                }
            }

            // Second pass: connect consecutive control points via surface path.
            let n = control_points.len();
            let mut loop_edges = Vec::new();
            let mut path_ok = true;
            for i in 0..n {
                let src = control_points[i];
                let tgt = control_points[(i + 1) % n];
                loop_edges.push(src);
                match surface_path_intermediates(src, tgt, &blocked, mesh) {
                    Some(inter) => loop_edges.extend(inter),
                    None => {
                        error!(
                            "No surface path from {:?} to {:?} for {:?}-loop (control point {}/{})",
                            src, tgt, loop_axis, i + 1, n
                        );
                        path_ok = false;
                    }
                }
            }

            if path_ok {
                map.insert(Loop { edges: loop_edges, direction: loop_axis });
            }
        }
    }
}

/// A point on the surface that lies on a loop.
///
/// - `Crossing`: on a boundary between two patches.
///   `dir_sign = (A, s)` is the CrossingMap key, where `A = third(loop_axis, boundary_dir)`
///   and `s` is the sign of the slot the loop was at *before* crossing this boundary.
///   An L-loop only visits crossings whose `dir_sign` direction ≠ L.
/// - `FacePoint`: on a node patch. `dir_sign = (A, s)` is the key into `FacePointMap[patch]`,
///   representing the face the loop is currently sitting on.
///   An L-loop only visits face points whose `dir_sign` direction ≠ L.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum NextPoint {
    Crossing { loop_id: LoopID, dir_sign: (PrincipalDirection, AxisSign) },
    FacePoint { patch: NodeIndex, dir_sign: (PrincipalDirection, AxisSign) },
}

/// Returns the next `(direction, sign)` slot in CCW order for an `L`-loop currently at `(dir, sign)`.
///
/// CCW is defined as: when viewed from the `+L` direction, rotation goes from one orthogonal
/// axis to the next via the right-hand cross product: `L × (dir * sign)`.
/// This always maps between the two non-L directions, cycling through 4 slots.
fn ccw_next(
    loop_axis: PrincipalDirection,
    dir: PrincipalDirection,
    sign: AxisSign,
) -> (PrincipalDirection, AxisSign) {
    use PrincipalDirection::{X, Y, Z};
    // Cross product table (positive results): X×Y=+Z, Y×Z=+X, Z×X=+Y.
    // Swapping operands negates: X×Z=-Y, Y×X=-Z, Z×Y=-X.
    let cross_positive = matches!((loop_axis, dir), (X, Y) | (Y, Z) | (Z, X));
    let res_dir = third(loop_axis, dir);
    let res_sign = if cross_positive { sign } else { sign.flipped() };
    (res_dir, res_sign)
}

/// Core traversal step: given a node and the slot `(A, s)` the loop is currently at,
/// returns the next `NextPoint` for `loop_axis`.
///
/// - Computes `next_slot = ccw_next(loop_axis, A, s)`.
/// - If the node has an edge in the `next_slot` direction+sign: we cross that boundary ->
///   returns `Crossing` with `dir_sign = (A, s)` (the slot we departed from).
/// - Otherwise: we stay on the same patch -> returns `FacePoint` with `dir_sign = next_slot`.
fn next_from_node_slot(
    node: NodeIndex,
    slot: (PrincipalDirection, AxisSign),
    loop_axis: PrincipalDirection,
    skeleton: &LabeledCurveSkeleton,
    boundary_map: &bimap::BiHashMap<EdgeIndex, LoopID>,
) -> NextPoint {
    let (a, s) = slot;
    let next_slot = ccw_next(loop_axis, a, s);
    let (next_dir, next_sign) = next_slot;

    // Check whether this node has a skeleton edge in the next_slot direction.
    let edge_in_next_dir: Option<EdgeIndex> = skeleton
        .edges(node)
        .find(|e| {
            let sign_from_node = skeleton
                .edge_sign_from(e.id(), node)
                .expect("node must be endpoint");
            e.weight().direction == next_dir && sign_from_node == next_sign
        })
        .map(|e| e.id());

    match edge_in_next_dir {
        Some(edge_idx) => {
            // Cross the boundary: crossing key = slot we departed from.
            let &loop_id = boundary_map.get_by_left(&edge_idx).expect("edge must have a loop");
            NextPoint::Crossing { loop_id, dir_sign: slot }
        }
        None => {
            // Rotate within the same patch.
            NextPoint::FacePoint { patch: node, dir_sign: next_slot }
        }
    }
}

/// Given a `current` surface point and the `loop_axis`, returns the unique next point
/// in CCW traversal order.
///
/// Two cases:
///
/// **FacePoint** — delegate directly to `next_from_node_slot`.
///
/// **Crossing** — determine which endpoint node we enter (via `ccw_next` on the crossing slot,
/// which gives the direction+sign we move through the boundary), compute the incoming slot
/// on that node (boundary direction with flipped sign), then delegate to `next_from_node_slot`.
fn next_point(
    current: NextPoint,
    loop_axis: PrincipalDirection,
    skeleton: &LabeledCurveSkeleton,
    boundary_map: &bimap::BiHashMap<EdgeIndex, LoopID>,
) -> NextPoint {
    match current {
        NextPoint::FacePoint { patch, dir_sign } => {
            next_from_node_slot(patch, dir_sign, loop_axis, skeleton, boundary_map)
        }

        NextPoint::Crossing { loop_id, dir_sign: (a, s) } => {
            // The boundary direction D = third(loop_axis, a).
            // ccw_next tells us which direction+sign we move through the boundary.
            let (move_dir, move_sign) = ccw_next(loop_axis, a, s);
            // move_dir == third(loop_axis, a) == the boundary's direction.

            // Find the boundary edge and determine which endpoint node we enter.
            let &edge_idx = boundary_map.get_by_right(&loop_id).expect("loop must have an edge");
            let (src, tgt) = skeleton.edge_endpoints(edge_idx).expect("edge must exist");

            // The edge is stored with a sign from src->tgt. Compare to move_sign to pick the node.
            let edge_sign_from_src = skeleton
                .edge_sign_from(edge_idx, src)
                .expect("src is an endpoint");
            let entered_node = if move_sign == edge_sign_from_src {
                tgt  // moving in the same direction as src->tgt means we enter tgt
            } else {
                src
            };

            // We entered through the face opposite to move_sign.
            let incoming_slot = (move_dir, move_sign.flipped());
            next_from_node_slot(entered_node, incoming_slot, loop_axis, skeleton, boundary_map)
        }
    }
}

/// Returns the unique direction that is neither `a` nor `b`.
fn third(a: PrincipalDirection, b: PrincipalDirection) -> PrincipalDirection {
    match (a, b) {
        (PrincipalDirection::X, PrincipalDirection::Y)
        | (PrincipalDirection::Y, PrincipalDirection::X) => PrincipalDirection::Z,
        (PrincipalDirection::X, PrincipalDirection::Z)
        | (PrincipalDirection::Z, PrincipalDirection::X) => PrincipalDirection::Y,
        (PrincipalDirection::Y, PrincipalDirection::Z)
        | (PrincipalDirection::Z, PrincipalDirection::Y) => PrincipalDirection::X,
        _ => panic!("directions must be different"),
    }
}
