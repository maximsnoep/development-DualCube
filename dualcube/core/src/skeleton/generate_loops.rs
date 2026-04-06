use std::collections::HashMap;

use mehsh::prelude::{HasPosition, Mesh, Vector3D};
use petgraph::{
    graph::EdgeIndex,
    visit::{EdgeRef, IntoEdgeReferences},
};
use slotmap::SlotMap;

use crate::{
    prelude::{EdgeID, PrincipalDirection, INPUT},
    skeleton::{
        boundary_loop::BoundaryLoop,
        orthogonalize::{AxisSign, LabeledCurveSkeleton},
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

// custom error
pub enum LoopGenerationError {
    MissingLabeledSkeleton,
    // todo other error variants
}

/// Generates surface-embedded loops from a polycube and polycube map.
pub fn generate_loops(
    skeleton_data: &SkeletonData,
    mesh: &Mesh<INPUT>,
) -> Result<(SlotMap<LoopID, Loop>, CrossingMap), LoopGenerationError> {
    let mut map: SlotMap<LoopID, Loop> = SlotMap::with_key();

    // Use ortho-skeleton, for each patch boundary assign 4 points that will host loop (paths).
    // throw warn and return if not there
    let skeleton: &LabeledCurveSkeleton = skeleton_data
        .labeled_skeleton
        .as_ref()
        .ok_or_else(|| LoopGenerationError::MissingLabeledSkeleton)?;
    let (boundary_map, crossings) = get_boundaries_and_crossing_points(skeleton, mesh, &mut map);

    // For each patch, for each side that does not correspond to a boundary,
    // find a point on the surface that represents the center of that face.
    // TODO: from the skeleton node, we will have 6 vectors that are ideally all equally spaced angle-wise. For each direction that does not have a boundary, we can find an ideal direction, then find a point far in that direction on the surface.

    // Trace paths between boundaries and points to create the loops
    // TODO: restricted Dijkstra's or something. Can be somewhat smart about ordering and having the second loop of each pair be as far as possible from the first to nicely divide the surface.

    Ok((map, crossings))
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
) -> (HashMap<EdgeIndex, LoopID>, CrossingMap) {
    let mut crossings: CrossingMap = HashMap::new();
    let mut boundary_map = HashMap::new();

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
        let ortho: Vec<_> = ALL_DIRS.iter().copied().filter(|&d| d != direction).collect();
        let mut loop_crossings = HashMap::new();

        for &dir in &ortho {
            for sign in ALL_SIGNS {
                let axis_vec = match sign {
                    AxisSign::Positive => Vector3D::from(dir),
                    AxisSign::Negative => -Vector3D::from(dir),
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

                loop_crossings.insert((dir, sign), boundary.edge_midpoints[best_idx]);
            }
        }

        crossings.insert(loop_id, loop_crossings);
    }

    (boundary_map, crossings)
}

/// Shortest angular distance between two angles in radians.
fn angle_distance(a: f64, b: f64) -> f64 {
    let mut d = (a - b) % (2.0 * std::f64::consts::PI);
    if d > std::f64::consts::PI {
        d -= 2.0 * std::f64::consts::PI;
    } else if d < -std::f64::consts::PI {
        d += 2.0 * std::f64::consts::PI;
    }
    d.abs()
}

fn get_loop(boundary: BoundaryLoop, direction: PrincipalDirection) -> Loop {
    Loop {
        edges: boundary.edge_midpoints,
        direction,
    }
}
