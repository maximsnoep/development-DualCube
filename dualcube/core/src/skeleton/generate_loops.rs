use std::{
    collections::{HashMap, HashSet},
    f64::consts::PI,
};

use bimap::BiHashMap;
use log::warn;
use mehsh::prelude::{HasFaces, HasPosition, HasVertices, Mesh, Vector3D};
use petgraph::{
    graph::{EdgeIndex, NodeIndex},
    visit::{EdgeRef, IntoEdgeReferences, IntoNodeReferences},
};
use slotmap::SlotMap;

use crate::{
    prelude::{EdgeID, PrincipalDirection, VertID, INPUT},
    skeleton::{
        boundary_loop::BoundaryLoop,
        orthogonalize::{AxisSign, IVector3D, LabeledCurveSkeleton},
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

    // Build a coordinate → node lookup from the grid positions assigned during orthogonalization.
    let grid: HashMap<IVector3D, NodeIndex> = skeleton
        .node_references()
        .map(|(idx, w)| (w.grid_position, idx))
        .collect();

    // Trace paths between boundary points and face points to create the loops
    pathing_for_loops(
        boundary_map,
        crossings.clone(), // TODO: later we can simply consume as we no longer need to return it
        face_points.clone(), // TODO: same here
        skeleton,
        &grid,
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

fn pathing_for_loops(
    _boundary_map: BiHashMap<EdgeIndex, LoopID>,
    _crossings: CrossingMap,
    _face_points: FacePointMap,
    _skeleton: &LabeledCurveSkeleton,
    _grid: &HashMap<IVector3D, NodeIndex>,
    _mesh: &Mesh<INPUT>,
    _map: &mut SlotMap<LoopID, Loop>,
) {
    // TODO: implement coordinate-based loop tracing using the grid map.

    // Print all coordinates
    for (coord, node_idx) in _grid {
        println!("Node {:?} at grid position {:?}", node_idx, coord);
    }

    // Account for 'edges' !!! Grid positions do not account for cubes. Maybe scale manually first...
}

/// A point on the surface that lies on a loop.
///
/// - `Crossing`: on a boundary between two patches. `dir_sign = (loop_dir, sign)` is the key
///   into `CrossingMap[loop_id]` that identifies which of the two crossings of this loop type
///   on this boundary we are at.
/// - `FacePoint`: on a node patch. `dir_sign = (face_dir, sign)` is the key into `FacePointMap[patch]`.
#[derive(Copy, Clone, Debug)]
enum NextPoint {
    Crossing { loop_id: LoopID, dir_sign: (PrincipalDirection, AxisSign) },
    FacePoint { patch: NodeIndex, dir_sign: (PrincipalDirection, AxisSign) },
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
