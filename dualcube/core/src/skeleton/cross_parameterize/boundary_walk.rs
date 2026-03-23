use core::panic;
use std::{
    collections::{HashMap, HashSet},
    f64::consts::PI,
};

use log::{error, warn};
use mehsh::prelude::{HasVertices, Mesh};
use petgraph::{
    graph::{EdgeIndex, NodeIndex},
    prelude::StableUnGraph,
    visit::EdgeRef,
};

use crate::{
    prelude::{EdgeID, VertID, INPUT},
    skeleton::{
        cross_parameterize::{
            virtual_mesh::{
                EdgemidpointToVirtual, VertexToVirtual, VirtualEdgeWeight, VirtualNode,
                VirtualNodeOrigin,
            },
            CuttingPlan,
        },
        orthogonalize::LabeledCurveSkeleton,
    },
};

/// Returns, for each boundary-loop edge around this patch, whether its current
/// midpoint order must be reversed so traversal keeps the patch on the left.
pub fn calculate_boundary_loop_reversal_flags(
    patch_node_idx: NodeIndex,
    skeleton: &LabeledCurveSkeleton,
    mesh: &Mesh<INPUT>,
) -> HashMap<EdgeIndex, bool> {
    let patch_vertices: HashSet<VertID> = skeleton
        .node_weight(patch_node_idx)
        .unwrap()
        .skeleton_node
        .patch_vertices
        .iter()
        .copied()
        .collect();

    let mut reverse_flags = HashMap::new();

    for skeleton_edge in skeleton.edges(patch_node_idx) {
        let edge_id = skeleton_edge.id();
        let boundary = &skeleton_edge.weight().boundary_loop;

        let mut left_votes = 0usize;
        let mut right_votes = 0usize;

        for &oriented_boundary_edge in &boundary.edge_midpoints {
            let face_id = mesh.face(oriented_boundary_edge);
            let face_vertices: Vec<VertID> = mesh.vertices(face_id).collect();
            if face_vertices.len() < 3 {
                panic!(
                    "Face with fewer than 3 vertices encountered while classifying boundary orientation."
                );
            }

            // Oriented symbolic embedding of this face cycle onto a regular n-gon.
            // Works for triangles and quads (and any n >= 3).
            let coord = |v: VertID| -> (f64, f64) {
                let Some(i) = face_vertices.iter().position(|&x| x == v) else {
                    unreachable!("Face coordinate lookup used vertex outside current face")
                };
                symbolic_face_coord(i, face_vertices.len())
            };

            let current_u = mesh.root(oriented_boundary_edge);
            let current_v = mesh.toor(oriented_boundary_edge);
            if !face_vertices.contains(&current_u) || !face_vertices.contains(&current_v) {
                panic!("Boundary edge endpoints are not both vertices of its incident face.");
            }

            let current_key = if current_u < current_v {
                (current_u, current_v)
            } else {
                (current_v, current_u)
            };

            let mut crossing_edges = Vec::new();
            for i in 0..face_vertices.len() {
                let a = face_vertices[i];
                let b = face_vertices[(i + 1) % face_vertices.len()];
                if patch_vertices.contains(&a) ^ patch_vertices.contains(&b) {
                    let key = if a < b { (a, b) } else { (b, a) };
                    crossing_edges.push(key);
                }
            }

            if crossing_edges.len() != 2 {
                panic!(
                    "Boundary face has {} patch-crossing edges; expected exactly 2.",
                    crossing_edges.len()
                );
            }

            let other_key = if crossing_edges[0] == current_key {
                crossing_edges[1]
            } else if crossing_edges[1] == current_key {
                crossing_edges[0]
            } else {
                panic!("Current boundary edge is not one of the two crossing edges in its face.");
            };

            let patch_face_vertices: Vec<VertID> = face_vertices
                .iter()
                .copied()
                .filter(|v| patch_vertices.contains(v))
                .collect();
            if patch_face_vertices.is_empty() || patch_face_vertices.len() == face_vertices.len() {
                panic!(
                    "Boundary face has invalid patch membership for orientation classification."
                );
            }

            // Representative point of the patch-side within this face.
            let patch_point = {
                let mut sx = 0.0;
                let mut sy = 0.0;
                for v in &patch_face_vertices {
                    let c = coord(*v);
                    sx += c.0;
                    sy += c.1;
                }
                let denom = patch_face_vertices.len() as f64;
                (sx / denom, sy / denom)
            };

            let current_mid = {
                let a = coord(current_key.0);
                let b = coord(current_key.1);
                ((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5)
            };
            let other_mid = {
                let a = coord(other_key.0);
                let b = coord(other_key.1);
                ((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5)
            };

            // Side-of-segment test: does patch representative lie left/right of
            // the directed midpoint segment current -> other?
            let dir = (other_mid.0 - current_mid.0, other_mid.1 - current_mid.1);
            let rel = (patch_point.0 - current_mid.0, patch_point.1 - current_mid.1);
            let orient = dir.0 * rel.1 - dir.1 * rel.0;

            if orient > 0.0 {
                left_votes += 1;
            } else if orient < 0.0 {
                right_votes += 1;
            }
        }

        if left_votes > 0 && right_votes > 0 {
            warn!("Conflicting votes for boundary loop orientation on skeleton edge {:?}: {} left vs {} right. This may indicate a non-manifold boundary or other irregularity. Defaulting to no reversal.", edge_id, left_votes, right_votes);
        } else if left_votes == 0 && right_votes == 0 {
            panic!(
                "Could not classify orientation for boundary loop {:?}: no non-degenerate votes.",
                edge_id
            );
        }

        // Reverse if current order places the patch on the right overall.
        reverse_flags.insert(edge_id, right_votes > left_votes);
    }

    reverse_flags
}

fn symbolic_face_coord(index: usize, n: usize) -> (f64, f64) {
    let angle = 2.0 * PI * (index as f64) / (n as f64);
    (angle.cos(), angle.sin())
}

/// Calculates the single boundary loop of the virtual mesh.
/// The resulting loop is a simple cycle of virtual node indices, following all boundaries in CCW.
/// Adds all edges for nodes along the boundary (properly dealing with duplicates and cuts.)
///
/// Note that for the tri-mesh, quads are accepted around boundaries.
pub fn calculate_boundary_loop(
    patch_node_idx: NodeIndex,
    skeleton: &LabeledCurveSkeleton,
    mesh: &Mesh<INPUT>,
    graph: &mut StableUnGraph<VirtualNode, VirtualEdgeWeight>,
    vert_to_nodes: &HashMap<VertID, VertexToVirtual>,
    edge_midpoint_ids_to_node_indices: &HashMap<EdgeID, EdgemidpointToVirtual>,
    cutting_plan: &CuttingPlan,
    reverse_flags: &HashMap<EdgeIndex, bool>,
    is_tri_mesh: bool,
) -> Vec<NodeIndex> {
    let mut succ: HashMap<NodeIndex, NodeIndex> = HashMap::new();

    let resolve_midpoint_edge_id = |edge_id: EdgeID| -> EdgeID {
        if edge_midpoint_ids_to_node_indices.contains_key(&edge_id) {
            edge_id
        } else {
            let twin = mesh.twin(edge_id);
            if edge_midpoint_ids_to_node_indices.contains_key(&twin) {
                twin
            } else {
                panic!(
                    "Boundary edge {:?} not found in midpoint map, nor its twin {:?}",
                    edge_id, twin
                );
            }
        }
    };

    // edge midpoint -> (cut_index, is_start)
    let mut cut_endpoint_role: HashMap<EdgeID, (usize, bool)> = HashMap::new();
    for (cut_index, cut) in cutting_plan.cuts.iter().enumerate() {
        let start_key = resolve_midpoint_edge_id(cut.path.start);
        let old = cut_endpoint_role.insert(start_key, (cut_index, true));
        if old.is_some() {
            panic!(
                "Cut endpoint midpoint {:?} reused across cuts",
                cut.path.start
            );
        }
        let end_key = resolve_midpoint_edge_id(cut.path.end);
        let old = cut_endpoint_role.insert(end_key, (cut_index, false));
        if old.is_some() {
            panic!(
                "Cut endpoint midpoint {:?} reused across cuts",
                cut.path.end
            );
        }
    }

    let boundary_node_in = |edge_id: EdgeID| -> NodeIndex {
        let key = resolve_midpoint_edge_id(edge_id);
        match edge_midpoint_ids_to_node_indices.get(&key) {
            Some(EdgemidpointToVirtual::Unique(n)) => *n,
            Some(EdgemidpointToVirtual::CutEndpointPair { left, right }) => {
                let Some((_, is_start)) = cut_endpoint_role.get(&key) else {
                    panic!(
                        "CutEndpointPair found without cut role for midpoint {:?}",
                        edge_id
                    );
                };
                if *is_start {
                    *left
                } else {
                    *right
                }
            }
            None => panic!(
                "Boundary midpoint {:?} missing from virtual-node map",
                edge_id
            ),
        }
    };

    let boundary_node_out = |edge_id: EdgeID| -> NodeIndex {
        let key = resolve_midpoint_edge_id(edge_id);
        match edge_midpoint_ids_to_node_indices.get(&key) {
            Some(EdgemidpointToVirtual::Unique(n)) => *n,
            Some(EdgemidpointToVirtual::CutEndpointPair { left, right }) => {
                let Some((_, is_start)) = cut_endpoint_role.get(&key) else {
                    panic!(
                        "CutEndpointPair found without cut role for midpoint {:?}",
                        edge_id
                    );
                };
                if *is_start {
                    *right
                } else {
                    *left
                }
            }
            None => panic!(
                "Boundary midpoint {:?} missing from virtual-node map",
                edge_id
            ),
        }
    };

    // Boundary successor edges (CCW for this patch).
    for skeleton_edge in skeleton.edges(patch_node_idx) {
        let edge_id = skeleton_edge.id();
        let boundary = &skeleton_edge.weight().boundary_loop;
        if boundary.edge_midpoints.is_empty() {
            continue;
        }

        let reversed = *reverse_flags.get(&edge_id).unwrap_or(&false);
        let ordered: Vec<EdgeID> = if reversed {
            boundary
                .edge_midpoints
                .iter()
                .copied()
                .rev()
                .map(|e| mesh.twin(e))
                .collect()
        } else {
            boundary.edge_midpoints.clone()
        };

        for i in 0..ordered.len() {
            let a = ordered[i];
            let b = ordered[(i + 1) % ordered.len()];
            let src = boundary_node_out(a);
            let dst = boundary_node_in(b);
            if let Some(prev) = succ.insert(src, dst) {
                if prev != dst {
                    panic!(
                        "Boundary successor conflict at node {:?}: {:?} vs {:?}",
                        src, prev, dst
                    );
                }
            }
        }
    }

    // Cut-side successor edges.
    for cut in &cutting_plan.cuts {
        let start_key = resolve_midpoint_edge_id(cut.path.start);
        let end_key = resolve_midpoint_edge_id(cut.path.end);

        let Some(EdgemidpointToVirtual::CutEndpointPair {
            left: start_left,
            right: start_right,
        }) = edge_midpoint_ids_to_node_indices.get(&start_key)
        else {
            panic!(
                "Cut start midpoint {:?} (resolved {:?}) does not map to a duplicated endpoint",
                cut.path.start, start_key
            );
        };
        let Some(EdgemidpointToVirtual::CutEndpointPair {
            left: end_left,
            right: end_right,
        }) = edge_midpoint_ids_to_node_indices.get(&end_key)
        else {
            panic!(
                "Cut end midpoint {:?} (resolved {:?}) does not map to a duplicated endpoint",
                cut.path.end, end_key
            );
        };

        let mut left_chain = Vec::new();
        left_chain.push(*start_left);
        for v in &cut.path.interior_verts {
            match vert_to_nodes.get(v) {
                Some(VertexToVirtual::CutPair { left, .. }) => left_chain.push(*left),
                Some(VertexToVirtual::Unique(_)) => {
                    panic!("Cut interior vertex {:?} is not duplicated", v)
                }
                None => panic!("Cut interior vertex {:?} missing in virtual map", v),
            }
        }
        left_chain.push(*end_left);
        for pair in left_chain.windows(2) {
            if let [a, b] = pair {
                if let Some(prev) = succ.insert(*a, *b) {
                    if prev != *b {
                        panic!("Cut-left successor conflict at node {:?}", a);
                    }
                }
            }
        }

        let mut right_chain = Vec::new();
        right_chain.push(*end_right);
        for v in cut.path.interior_verts.iter().rev() {
            match vert_to_nodes.get(v) {
                Some(VertexToVirtual::CutPair { right, .. }) => right_chain.push(*right),
                Some(VertexToVirtual::Unique(_)) => {
                    panic!("Cut interior vertex {:?} is not duplicated", v)
                }
                None => panic!("Cut interior vertex {:?} missing in virtual map", v),
            }
        }
        right_chain.push(*start_right);
        for pair in right_chain.windows(2) {
            if let [a, b] = pair {
                if let Some(prev) = succ.insert(*a, *b) {
                    if prev != *b {
                        panic!("Cut-right successor conflict at node {:?}", a);
                    }
                }
            }
        }
    }

    // Start from an arbitrary cut endpoint left copy when possible.
    let start = if let Some(first_cut) = cutting_plan.cuts.first() {
        let start_key = resolve_midpoint_edge_id(first_cut.path.start);
        match edge_midpoint_ids_to_node_indices.get(&start_key) {
            Some(EdgemidpointToVirtual::CutEndpointPair { left, .. }) => *left,
            _ => panic!("First cut start endpoint is not duplicated as expected"),
        }
    } else {
        // Degree-1 fallback: start at any boundary midpoint node.
        let Some(any_boundary_edge) = skeleton
            .edges(patch_node_idx)
            .next()
            .map(|e| e.weight().boundary_loop.edge_midpoints.first().copied())
            .flatten()
        else {
            unreachable!("Degree-1 patch with no boundary midpoints found.")
        };
        boundary_node_in(any_boundary_edge)
    };

    // Simple directed walk of the boundary cycle.
    let mut boundary_loop = Vec::new();
    let mut seen: HashSet<NodeIndex> = HashSet::new();
    let mut current = start;

    for _ in 0..(succ.len() + 2) {
        if !seen.insert(current) {
            break;
        }
        boundary_loop.push(current);

        let Some(next) = succ.get(&current).copied() else {
            panic!(
                "Boundary traversal got stuck at node {:?} (no successor)",
                current
            );
        };

        // Check correctness
        if let VirtualNodeOrigin::MeshVertex(v) = graph[next].origin {
            panic!(
                "Boundary traversal entered non-boundary mesh vertex {:?} (node {:?})",
                v, next
            )
        }

        // When adding boundary edges, the target to connect to might be a duplicated one, making it unclear always which to connect to.
        // Then, the first time we see this edge, we add what the correct other side is, so the next time we can add the edge properly.
        let mut boundary_lookback: HashMap<EdgeID, NodeIndex> = HashMap::new();
        // Add edges around boundaries.
        if is_tri_mesh {
            tri_mesh_boundary_edges(graph, current, next, &mut boundary_lookback);
        } else {
            quad_mesh_boundary_edges(graph, current, next, &mut boundary_lookback);
        }
        if !boundary_lookback.is_empty() {
            // TODO SHOULD PANIC WHEN EVERYTHING IS FULLY WORKING, NOW JUST ERROR
            error!("Boundary edge lookback produced non-empty result, indicating some boundary edge cases are not fully handled yet. Lookback: {:?}", boundary_lookback);
        }

        // Add the traced disk boundary as actual VFG edges.
        if graph.find_edge(current, next).is_none() {
            let length = (graph[current].position - graph[next].position).norm();
            graph.add_edge(current, next, VirtualEdgeWeight { length });
        }

        current = next;
        if current == start {
            break;
        }
    }

    if boundary_loop.is_empty() {
        panic!("Boundary traversal produced an empty loop");
    }

    boundary_loop
}

enum AddingEdgeInput {
    // Simple // Not done in this part of the code... All these are added in add_internal_edges after the boundary loop is fully calculated.
    //
    /// The direction is clear from the current vertex. There is no other side that will check back so no need to worry about lookback or duplicates.
    DuplicateToSingular {
        source: NodeIndex, // Side of duplicate is clear!
        target: NodeIndex, // Other side has no choice!
    },

    /// Any type of duplicate to any other type of duplicate.
    DuplicateToDuplicate {
        source: NodeIndex, 
        edge: EdgeID, // for lookback
    },
}

/// Smartly adds an edge to the VFG. Accounts for looking back for duplicates and not adding multiple parallel edges.
fn add_edge(
    graph: &mut StableUnGraph<VirtualNode, VirtualEdgeWeight>,
    lookback: &mut HashMap<EdgeID, NodeIndex>,
    input: AddingEdgeInput,
) {
    if let AddingEdgeInput::DuplicateToSingular { source, target } = input {
        // Only add if not already present, to avoid parallel edges.
        if graph.find_edge(source, target).is_none() { 
            let length = (graph[source].position - graph[target].position).norm();
            graph.add_edge(source, target, VirtualEdgeWeight { length });
        }
    } else if let AddingEdgeInput::DuplicateToDuplicate { source, edge } = input {
        if let Some(prev_target) = lookback.get(&edge) {
            // Add edge
            if graph.find_edge(source, *prev_target).is_none() { 
                let length = (graph[source].position - graph[*prev_target].position).norm();
                graph.add_edge(source, *prev_target, VirtualEdgeWeight { length });
            } else {
                unreachable!("Duplicate to duplicate case cannot cause parallel edges.");
            }

            // Clear lookback
            lookback.remove(&edge);
        } else {
            lookback.insert(edge, source);
            // Edge will be added when we encounter the other duplicate.
        }
    } else {
        unreachable!();
    }
}
// For mostly tri-meshes, only quads accepted around boundaries.
fn tri_mesh_boundary_edges(
    graph: &mut StableUnGraph<VirtualNode, VirtualEdgeWeight>,
    current: NodeIndex,
    next: NodeIndex,
    boundary_lookback: &mut HashMap<EdgeID, NodeIndex>,
) {
    let current_type = &graph[current].origin;
    let next_type = &graph[next].origin;
    match (current_type, next_type) {
        (
            VirtualNodeOrigin::BoundaryMidpoint {
                edge: current_edge,
                boundary_edge: current_boundary_edge,
            },
            VirtualNodeOrigin::BoundaryMidpoint {
                edge: next_edge,
                boundary_edge: next_boundary_edge,
            },
        ) => {
            // Simplest case: for both edges, add edge from patch vertex to its connecting boundary midpoint
            // TODO
        }
        (
            VirtualNodeOrigin::BoundaryMidpoint { .. },
            VirtualNodeOrigin::CutEndpointMidpointDuplicate { .. },
        )
        | (
            VirtualNodeOrigin::CutEndpointMidpointDuplicate { .. },
            VirtualNodeOrigin::BoundaryMidpoint { .. },
        ) => {
            // Get triangle spanned by the edges that host the midpoints. (likely not explicitly necessary)
            // Get opposite node of cut endpoint in this triangle, i.e. next_nodes - current_nodes (they always share 1 node to be a triangle)
            // TODO..
            // Add edge from cut endpoint to the opposite node.
            // Add edge from opposite node to boundary midpoint.
        }

        (VirtualNodeOrigin::CutDuplicate { .. }, VirtualNodeOrigin::CutDuplicate { .. }) => {
            // TODO, something with sides?! Main difficult case.
        }

        (
            VirtualNodeOrigin::CutEndpointMidpointDuplicate { .. },
            VirtualNodeOrigin::CutDuplicate { .. },
        )
        | (
            VirtualNodeOrigin::CutDuplicate { .. },
            VirtualNodeOrigin::CutEndpointMidpointDuplicate { .. },
        ) => {
            // TODO: something...
            // The corner edge is already taken care of in Midpoint-Endpoint cases.
            // TODO: something with adding edges from the right side...
        }

        (
            VirtualNodeOrigin::CutEndpointMidpointDuplicate { .. },
            VirtualNodeOrigin::CutEndpointMidpointDuplicate { .. },
        ) => {
            // Impossible to resolve in tri-mesh by just adding edges..
            // The two edges necessarily go out on the two-legged side of the triangle.
            // We want to connect another edge to the corner to make it degree 3 but we cannot do this for both at the same time...
            panic!("Boundary traversal encountered adjacent cut endpoints at nodes {:?} and {:?}. TODO: filter this out earlier!.",
                        current, next);
            // ... could be resolved actually by adding a vertex in the middle of the quad resulting from the midpoints
        }

        //
        _ => {
            panic!(
                        "Boundary traversal found unexpected pair: from node {:?} to {:?} of types {:?} to {:?}.",
                        current, next, current_type, next_type
                    );
        }
    }
}

// Strict quad mesh.
fn quad_mesh_boundary_edges(
    graph: &mut StableUnGraph<VirtualNode, VirtualEdgeWeight>,
    current: NodeIndex,
    next: NodeIndex,
    boundary_lookback: &mut HashMap<EdgeID, NodeIndex>,
) {
    let current_type = &graph[current].origin;
    let next_type = &graph[next].origin;
    match (current_type, next_type) {
        (
            VirtualNodeOrigin::BoundaryMidpoint {
                edge: current_edge,
                boundary_edge: current_boundary_edge,
            },
            VirtualNodeOrigin::BoundaryMidpoint {
                edge: next_edge,
                boundary_edge: next_boundary_edge,
            },
        ) => {
            // Simplest case: for both edges, add edge from patch vertex to its connecting boundary midpoint
            // TODO: check
        }
        (
            VirtualNodeOrigin::BoundaryMidpoint { .. },
            VirtualNodeOrigin::CutEndpointMidpointDuplicate { .. },
        )
        | (
            VirtualNodeOrigin::CutEndpointMidpointDuplicate { .. },
            VirtualNodeOrigin::BoundaryMidpoint { .. },
        ) => {
            // ???
        }

        (VirtualNodeOrigin::CutDuplicate { .. }, VirtualNodeOrigin::CutDuplicate { .. }) => {
            // TODO, something with sides?! Main difficult case.
        }

        (
            VirtualNodeOrigin::CutEndpointMidpointDuplicate { .. },
            VirtualNodeOrigin::CutDuplicate { .. },
        )
        | (
            VirtualNodeOrigin::CutDuplicate { .. },
            VirtualNodeOrigin::CutEndpointMidpointDuplicate { .. },
        ) => {
            // TODO: something...
            // The corner edge is already taken care of in Midpoint-Endpoint cases.
            // TODO: something with adding edges from the right side...
        }

        (
            VirtualNodeOrigin::CutEndpointMidpointDuplicate { .. },
            VirtualNodeOrigin::CutEndpointMidpointDuplicate { .. },
        ) => {
            // Can be solved in quad mesh...?
            // How do we make the corner degree 3? Degree 2 is free from boundary edges but there is no obvious third edge to add...

            // If we are okay with the strict quad mesh no longer being strict,
            // we can just add a vertex in the middle of the quad an connect it to all. No crossings, easy third edge. No sides worry.
        }

        //
        _ => {
            panic!(
                        "Boundary traversal found unexpected pair: from node {:?} to {:?} of types {:?} to {:?}.",
                        current, next, current_type, next_type
                    );
        }
    }
}
