use std::collections::HashMap;

use log::warn;
use mehsh::prelude::{HasPosition, Mesh, Vector3D};
use petgraph::{
    graph::{EdgeIndex, NodeIndex},
    visit::{EdgeRef, IntoEdgeReferences, IntoNodeReferences},
};
use slotmap::SlotMap;

use crate::{
    prelude::{EdgeID, PrincipalDirection, INPUT},
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

/// Per-node resolved direction vectors for each (PrincipalDirection, AxisSign).
type NodeDirectionMap = HashMap<NodeIndex, HashMap<(PrincipalDirection, AxisSign), Vector3D>>;

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
fn get_boundaries_and_crossing_points(
    skeleton: &LabeledCurveSkeleton,
    mesh: &Mesh<INPUT>,
    map: &mut SlotMap<LoopID, Loop>,
) -> (HashMap<EdgeIndex, LoopID>, CrossingMap) {
    let mut crossings: CrossingMap = HashMap::new();
    let mut boundary_map = HashMap::new();

    // Precompute direction vectors per node, starting from high-degree nodes and propagating to neighbors.
    let node_directions = compute_node_directions(skeleton);

    // Find each patch-patch boundary (which corresponds to skeleton edge)
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

        // For each orthogonal direction and sign, find the boundary edge midpoint maximal in that direction
        // relative to the centroid. Always 4 crossings: 2 per other loop type
        // (e.g. for an X loop there are 2 Y crossings and 2 Z crossings).
        let source_dirs = &node_directions[&edge.source()];
        let target_dirs = &node_directions[&edge.target()];

        let mut loop_crossings = HashMap::new();
        for &ortho_dir in ALL_DIRS.iter().filter(|&&d| d != direction) {
            for sign in ALL_SIGNS {
                // Average direction vectors from both endpoint nodes
                let dir_src = source_dirs[&(ortho_dir, sign)];
                let dir_tgt = target_dirs[&(ortho_dir, sign)];
                let dir_vec = ((dir_src + dir_tgt) / 2.0).normalize();

                let &best_edge = boundary
                    .edge_midpoints
                    .iter()
                    .max_by(|&&a, &&b| {
                        let dot_a = (mesh.position(a) - centroid).dot(&dir_vec);
                        let dot_b = (mesh.position(b) - centroid).dot(&dir_vec);
                        dot_a
                            .partial_cmp(&dot_b)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .expect("boundary loop should not be empty");

                loop_crossings.insert((ortho_dir, sign), best_edge);
            }
        }
        crossings.insert(loop_id, loop_crossings);
    }

    (boundary_map, crossings)
}

/// Computes per-node direction vectors by propagating from high-degree nodes to low-degree ones.
///
/// For each node, resolves what the actual 3D direction is for each (PrincipalDirection, AxisSign):
/// 1. From edges: use the displacement to the neighbor as the direction vector.
/// 2. Extrapolate: if only one sign of a direction is known, negate it for the other.
/// 3. Propagate: inherit missing directions from already-resolved neighbors.
/// 4. Fallback: use global axis directions.
fn compute_node_directions(skeleton: &LabeledCurveSkeleton) -> NodeDirectionMap {
    let mut node_directions: NodeDirectionMap = HashMap::new();

    // Sort nodes by degree descending: high-degree nodes have more edge constraints
    let mut nodes_by_degree: Vec<_> = skeleton.node_references().collect();
    nodes_by_degree.sort_by_key(|n| std::cmp::Reverse(skeleton.neighbors(n.0).count()));

    for (node_idx, node_weight) in &nodes_by_degree {
        let node_idx = *node_idx;
        let node_pos = node_weight.skeleton_node.position;
        let mut directions: HashMap<(PrincipalDirection, AxisSign), Vector3D> = HashMap::new();

        // Step 1: From edges incident to this node, compute actual displacement direction vectors
        for edge_ref in skeleton.edges(node_idx) {
            let edge_weight = edge_ref.weight();
            let dir = edge_weight.direction;
            let sign = skeleton
                .edge_sign_from(edge_ref.id(), node_idx)
                .expect("node should be endpoint of its own edge");

            let other = if edge_ref.source() == node_idx {
                edge_ref.target()
            } else {
                edge_ref.source()
            };
            let other_pos = skeleton[other].skeleton_node.position;
            let displacement = (other_pos - node_pos).normalize();

            directions.insert((dir, sign), displacement);
        }

        // Step 2: For directions with one sign but not the other, extrapolate by negating
        for dir in ALL_DIRS {
            let has_pos = directions.contains_key(&(dir, AxisSign::Positive));
            let has_neg = directions.contains_key(&(dir, AxisSign::Negative));
            if has_pos && !has_neg {
                let v = directions[&(dir, AxisSign::Positive)];
                directions.insert((dir, AxisSign::Negative), -v);
            } else if has_neg && !has_pos {
                let v = directions[&(dir, AxisSign::Negative)];
                directions.insert((dir, AxisSign::Positive), -v);
            }
        }

        // Step 3: Check already-resolved neighbors for missing directions (propagation)
        for dir in ALL_DIRS {
            for sign in ALL_SIGNS {
                if directions.contains_key(&(dir, sign)) {
                    continue;
                }
                for edge_ref in skeleton.edges(node_idx) {
                    let other = if edge_ref.source() == node_idx {
                        edge_ref.target()
                    } else {
                        edge_ref.source()
                    };
                    if let Some(neighbor_dirs) = node_directions.get(&other) {
                        if let Some(&v) = neighbor_dirs.get(&(dir, sign)) {
                            directions.insert((dir, sign), v);
                            break;
                        }
                    }
                }
            }
        }

        // Step 4: Fallback to global axis directions
        for dir in ALL_DIRS {
            for sign in ALL_SIGNS {
                directions.entry((dir, sign)).or_insert_with(|| {
                    let v = Vector3D::from(dir);
                    match sign {
                        AxisSign::Positive => v,
                        AxisSign::Negative => -v,
                    }
                });
            }
        }

        node_directions.insert(node_idx, directions);
    }

    node_directions
}

fn get_loop(boundary: BoundaryLoop, direction: PrincipalDirection) -> Loop {
    Loop {
        edges: boundary.edge_midpoints,
        direction,
    }
}
