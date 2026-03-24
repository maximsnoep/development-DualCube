use core::panic;
use std::{
    collections::{HashMap, HashSet, VecDeque},
    f64::consts::PI,
};

use itertools::Itertools;
use log::{error, info, warn};
use mehsh::{
    prelude::{HasEdges, HasFaces, HasVertices, Mesh, Vector3D},
};
use petgraph::{
    graph::{EdgeIndex, NodeIndex},
    prelude::StableUnGraph,
    visit::EdgeRef,
};

use crate::{
    prelude::{EdgeID, FaceID, VertID, INPUT},
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

fn ordered_vert_pair(a: VertID, b: VertID) -> (VertID, VertID) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Collects all mesh-vertex edges traversed by cut interior paths.
fn collect_cut_edges(
    mesh: &Mesh<INPUT>,
    cutting_plan: &CuttingPlan,
) -> (HashSet<EdgeID>, HashMap<(VertID, VertID), (VertID, VertID)>) {
    let mut cut_edges = HashSet::new();
    let mut cut_edge_canonical_direction: HashMap<(VertID, VertID), (VertID, VertID)> =
        HashMap::new();

    for cut in &cutting_plan.cuts {
        for pair in cut.path.interior_verts.windows(2) {
            if let [a, b] = pair {
                let key = ordered_vert_pair(*a, *b);
                let edges = mesh
                    .edge_between_verts(*a, *b)
                    .expect("Cut interior vertices are not connected by an edge.");
                cut_edges.insert(edges.0);
                cut_edges.insert(edges.1);

                if let Some(prev) = cut_edge_canonical_direction.insert(key, (*a, *b)) {
                    if prev != (*a, *b) {
                        panic!(
                            "Conflicting canonical direction for cut edge {:?}: {:?} vs {:?}",
                            key,
                            prev,
                            (*a, *b)
                        );
                    }
                }
            }
        }
    }

    (cut_edges, cut_edge_canonical_direction)
}

enum CutSideFaceVertices {
    Tri {
        face: FaceID,
        other: VertID,
    },
    Quad {
        face: FaceID,
        other_a: VertID,
        other_b: VertID,
    },
}

/// Returns the face on the chosen side of directed cut segment `u -> v`, plus
/// the non-cut face vertices (1 for tri, 2 for quad).
/// Can only be called for u,v not endpoints.
///
/// Note that endpoints could be outside of the patch in case face is cut.
fn find_cut_side_face_and_vertex(
    mesh: &Mesh<INPUT>,
    cut_edge_canonical_direction: &HashMap<(VertID, VertID), (VertID, VertID)>,
    u: VertID,
    v: VertID,
    side: bool,
) -> CutSideFaceVertices {
    let undirected = ordered_vert_pair(u, v);
    let (from, to) = *cut_edge_canonical_direction
        .get(&undirected)
        .expect("No canonical cut direction stored for cut edge");

    let (e0, e1) = mesh
        .edge_between_verts(from, to)
        .expect("Expected cut interior vertices to be connected by an edge");
    let edge_uv = if mesh.root(e0) == from && mesh.toor(e0) == to {
        e0
    } else if mesh.root(e1) == from && mesh.toor(e1) == to {
        e1
    } else {
        panic!(
            "Expected to find oriented edge {:?}->{:?}, but neither halfedge matched.",
            from, to
        );
    };

    let side_face = if side {
        mesh.face(mesh.twin(edge_uv))
    } else {
        mesh.face(edge_uv)
    };

    let face_vertices: Vec<VertID> = mesh.vertices(side_face).collect();
    let others: Vec<VertID> = face_vertices
        .iter()
        .copied()
        .filter(|w| *w != u && *w != v)
        .collect();

    if others.is_empty() {
        panic!("Face only consists of two cut vertices. This is not possible.")
    }

    match others.len() {
        1 => CutSideFaceVertices::Tri {
            face: side_face,
            other: others[0],
        },
        2 => CutSideFaceVertices::Quad {
            face: side_face,
            other_a: others[0],
            other_b: others[1],
        },
        _ => panic!("Unexpected number of non-cut vertices found."),
    }
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
    patch_vertices: &HashSet<VertID>,
) -> Vec<NodeIndex> {
    let mut succ: HashMap<NodeIndex, NodeIndex> = HashMap::new();
    let (cut_edges, cut_edge_canonical_direction) = collect_cut_edges(mesh, cutting_plan);

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

    // When adding boundary edges, the target to connect to might be a duplicated one, making it unclear always which to connect to.
    // Then, the first time we see this edge, we add what the correct other side is, so the next time we can add the edge properly.
    let mut boundary_lookback: HashMap<EdgeID, NodeIndex> = HashMap::new();

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

        // Add edges around boundaries.
        mesh_boundary_edges(
            graph,
            current,
            next,
            &mut boundary_lookback,
            &cut_edges,
            &cut_edge_canonical_direction,
            vert_to_nodes,
            edge_midpoint_ids_to_node_indices,
            mesh,
            patch_vertices,
        );

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

    if !boundary_lookback.is_empty() {
        // TODO SHOULD PANIC WHEN EVERYTHING IS FULLY WORKING, NOW JUST ERROR
        error!("Boundary edge lookback produced non-empty result, indicating some boundary edge cases are not fully handled yet. Lookback:");
        // Print types in detail for debugging.
        for (edge_id, node_idx) in &boundary_lookback {
            let node = &graph[*node_idx];
            let edge_verts = mesh.vertices(*edge_id).collect_vec();
            let [v1, v2] = edge_verts.as_slice() else {
                unreachable!("Edge always has 2 vertices");
            };
            let type_v1 = vert_to_nodes
                .get(&v1)
                .map(|x| match x {
                    VertexToVirtual::Unique(_) => "Unique",
                    VertexToVirtual::CutPair { .. } => "CutPair",
                })
                .unwrap_or("OUTSIDE PATCH");
            let type_v2 = vert_to_nodes
                .get(&v2)
                .map(|x| match x {
                    VertexToVirtual::Unique(_) => "Unique",
                    VertexToVirtual::CutPair { .. } => "CutPair",
                })
                .unwrap_or("OUTSIDE PATCH");

            error!(
                    "Lookback entry: edge {:?} (verts {:?} with types {:?}) -> node {:?} with lookback origin {:?}",
                    edge_id, [v1, v2], [type_v1, type_v2], node_idx, node.origin
                );
        }
    }

    if boundary_loop.is_empty() {
        panic!("Boundary traversal produced an empty loop");
    }

    boundary_loop
}

enum AddingEdgeInput {
    /// Add an edge along an existing mesh edge.
    AlongEdge {
        source: NodeIndex, // Side of duplicate is clear!
        edge: EdgeID,      // Other side is determined by lookback and duplicate status.
    },
}

/// Extracts the cut index and side information from a cut-related node origin.
/// Panics if the origin is not a cut duplicate or cut endpoint midpoint duplicate.
fn get_cut_info(origin: &VirtualNodeOrigin) -> (usize, bool) {
    match origin {
        VirtualNodeOrigin::CutDuplicate {
            cut_index, side, ..
        } => (*cut_index, *side),
        VirtualNodeOrigin::CutEndpointMidpointDuplicate {
            cut_index, side, ..
        } => (*cut_index, *side),
        _ => panic!("get_cut_info called on non-cut node origin: {:?}", origin),
    }
}

/// Smartly adds an edge to the VFG. Accounts for looking back for duplicates and not adding multiple parallel edges.
fn add_edge(
    graph: &mut StableUnGraph<VirtualNode, VirtualEdgeWeight>,
    lookback: &mut HashMap<EdgeID, NodeIndex>,
    vert_to_nodes: &HashMap<VertID, VertexToVirtual>,
    edge_midpoint_ids_to_node_indices: &HashMap<EdgeID, EdgemidpointToVirtual>,
    input: AddingEdgeInput,
    mesh: &Mesh<INPUT>,
    patch_vertices: &HashSet<VertID>,
) {
    #[allow(irrefutable_let_patterns)]
    // TODO: remove this if it really turns out to be not necessary
    if let AddingEdgeInput::AlongEdge { source, edge } = input {
        let midpoint_edge = if edge_midpoint_ids_to_node_indices.contains_key(&edge) {
            Some(edge)
        } else {
            let twin = mesh.twin(edge);
            if edge_midpoint_ids_to_node_indices.contains_key(&twin) {
                Some(twin)
            } else {
                None
            }
        };

        // First check if edge corresponds to a midpoint.
        if let Some(mid_edge) = midpoint_edge {
            match edge_midpoint_ids_to_node_indices.get(&mid_edge).unwrap() {
                EdgemidpointToVirtual::Unique(target) => {
                    if *target == source {
                        // We care about the vertex side, look in vertex side.
                    } else {
                        // Simple case, just check for parallel edge.
                        if graph.find_edge(source, *target).is_none() {
                            let length = (graph[source].position - graph[*target].position).norm();
                            graph.add_edge(source, *target, VirtualEdgeWeight { length });
                        }
                        lookback.remove(&mid_edge); // Clear lookback in case it was set from a previous duplicate.
                        return;
                    }
                }
                EdgemidpointToVirtual::CutEndpointPair { left, right } => {
                    // Check if source is either
                    if *left == source || *right == source {
                        // If this is the case, we care about the other side.
                    } else {
                        // If source and target are from the same cut, we can connect directly by matching sides.
                        let source_cut = match graph[source].origin {
                            VirtualNodeOrigin::CutDuplicate { .. }
                            | VirtualNodeOrigin::CutEndpointMidpointDuplicate { .. } => {
                                Some(get_cut_info(&graph[source].origin))
                            }
                            _ => None,
                        };

                        let mut connected_directly = false;
                        if let Some((s_cut, s_side)) = source_cut {
                            let (l_cut, l_side) = get_cut_info(&graph[*left].origin);
                            let (_r_cut, r_side) = get_cut_info(&graph[*right].origin);

                            if s_cut == l_cut {
                                let target = if s_side == l_side { *left } else { *right };
                                if graph.find_edge(source, target).is_none() {
                                    let length = (graph[source].position - graph[target].position).norm();
                                    graph.add_edge(source, target, VirtualEdgeWeight { length });
                                }
                                connected_directly = true;
                            }
                        }

                        if connected_directly {
                            return;
                        }

                        // The target is actually a duplicated pair. We need to use lookback
                        if lookback.contains_key(&mid_edge) {
                            let target = lookback.get(&mid_edge).unwrap();
                            if graph.find_edge(source, *target).is_none() {
                                let length =
                                    (graph[source].position - graph[*target].position).norm();
                                graph.add_edge(source, *target, VirtualEdgeWeight { length });
                            } else {
                                unreachable!("Lookback case should not be able to cause parallel edges under current assumptions?");
                            }
                            lookback.remove(&mid_edge); // Clear lookback after using.
                            return;
                        } else {
                            // Lookback not set yet, set it and wait for the other duplicate to be processed.
                            lookback.insert(mid_edge, source);
                            return;
                        }
                    }
                }
            };
        }

        // For non-midpoint edges, project to a stable orientation (root->toor)
        // so duplicate lookback keys match independent of half-edge direction.
        let mut edge = edge;
        if mesh.root(edge) > mesh.toor(edge) {
            edge = mesh.twin(edge);
        }

        // Only reach here when if fails (edge has no midpoint), or source is actually a midpoint (so we want to find vertex).
        {
            // Get vertices of this edge
            let Some([v1, v2]) = mesh.vertices(edge).collect_array::<2>() else {
                unreachable!("Edge always has 2 vertices");
            };

            // Check if either vertex corresponds to the source node
            let is_v1_source = match vert_to_nodes.get(&v1) {
                Some(VertexToVirtual::Unique(n)) => *n == source,
                Some(VertexToVirtual::CutPair { left, right }) => {
                    *left == source || *right == source
                }
                _ => false,
            };
            let is_v2_source = match vert_to_nodes.get(&v2) {
                Some(VertexToVirtual::Unique(n)) => *n == source,
                Some(VertexToVirtual::CutPair { left, right }) => {
                    *left == source || *right == source
                }
                _ => false,
            };

            let other_vert = if is_v1_source {
                v2
            } else if is_v2_source {
                v1
            } else {
                // This happens when the source is a midpoint. We pick the vertex that is part of the patch.
                if patch_vertices.contains(&v1) && !patch_vertices.contains(&v2) {
                    v1
                } else if patch_vertices.contains(&v2) && !patch_vertices.contains(&v1) {
                    v2
                } else {
                    unreachable!(
                        "Source node {:?} does not correspond to either vertex of edge {:?}.",
                        source, edge
                    );
                }
            };

            // Check if other_vert is duplicated, if so we need to use lookback to find the correct duplicate to connect to.
            match vert_to_nodes.get(&other_vert) {
                Some(VertexToVirtual::CutPair { left, right }) => {
                    // Unique nodes (like BoundaryMidpoint) do not need to use lookback. 
                    // The CutDuplicate on the other side will process this edge and connect directly to us.
                    let is_source_duplicate = matches!(
                        graph[source].origin,
                        VirtualNodeOrigin::CutDuplicate { .. } | VirtualNodeOrigin::CutEndpointMidpointDuplicate { .. }
                    );
                    if !is_source_duplicate {
                        return;
                    }

                    // If source and target are from the same cut, we can connect directly by matching sides.
                    let source_cut = match graph[source].origin {
                        VirtualNodeOrigin::CutDuplicate { .. }
                        | VirtualNodeOrigin::CutEndpointMidpointDuplicate { .. } => {
                            Some(get_cut_info(&graph[source].origin))
                        }
                        _ => None,
                    };

                    if let Some((s_cut, s_side)) = source_cut {
                        let (l_cut, l_side) = get_cut_info(&graph[*left].origin);
                        if s_cut == l_cut {
                            let target = if s_side == l_side { *left } else { *right };
                            if graph.find_edge(source, target).is_none() {
                                let length = (graph[source].position - graph[target].position).norm();
                                graph.add_edge(source, target, VirtualEdgeWeight { length });
                            }
                            return;
                        }
                    }

                    // This vertex is duplicated, we need to use lookback to find the correct duplicate to connect to.
                    if lookback.contains_key(&edge) {
                        let target = lookback.get(&edge).unwrap();
                        if graph.find_edge(source, *target).is_none() {
                            let length = (graph[source].position - graph[*target].position).norm();
                            graph.add_edge(source, *target, VirtualEdgeWeight { length });
                        } else {
                            unreachable!("Lookback case should not be able to cause parallel edges under current assumptions?");
                        }
                        lookback.remove(&edge); // Clear lookback after using.
                    } else {
                        // Lookback not set yet, set it and wait for the other duplicate to be processed.
                        lookback.insert(edge, source);
                    }
                }
                Some(VertexToVirtual::Unique(target)) => {
                    // This vertex is not duplicated, we can directly connect to the unique node. Just check for parallel edge.
                    if graph.find_edge(source, *target).is_none() {
                        let length = (graph[source].position - graph[*target].position).norm();
                        graph.add_edge(source, *target, VirtualEdgeWeight { length });
                    }
                    lookback.remove(&edge); // Clear lookback in case it was set from a previous duplicate.
                }
                None => unreachable!("Other vertex missing from virtual map: {:?}", other_vert),
            }
        }
    } else {
        unreachable!();
    }

    // if let AddingEdgeInput::DuplicateToSingular { source, target } = input {
    //     // Only add if not already present, to avoid parallel edges.
    //     if graph.find_edge(source, target).is_none() {
    //         let length = (graph[source].position - graph[target].position).norm();
    //         graph.add_edge(source, target, VirtualEdgeWeight { length });
    //     }
    // } else if let AddingEdgeInput::DuplicateToDuplicate { source, edge } = input {
    //     if let Some(prev_target) = lookback.get(&edge) {
    //         // Add edge
    //         if graph.find_edge(source, *prev_target).is_none() {
    //             let length = (graph[source].position - graph[*prev_target].position).norm();
    //             graph.add_edge(source, *prev_target, VirtualEdgeWeight { length });
    //         } else {
    //             unreachable!("Duplicate to duplicate case cannot cause parallel edges.");
    //         }

    //         // Clear lookback
    //         lookback.remove(&edge);
    //     } else {
    //         lookback.insert(edge, source);
    //         // Edge will be added when we encounter the other duplicate.
    //     }
    // } else {
    //     unreachable!();
    // }
}

pub fn fill_faces_for_cut_endpoint(graph: &mut StableUnGraph<VirtualNode, VirtualEdgeWeight>, is_tri_mesh: bool) {
    // To make sure we do not add vertices twice for faces with 2 cut endpoints
    let mut done: HashSet<NodeIndex> = HashSet::new();

    // Save what nodes to add (can't edit graph while iterating).
    // Saved is per node its position, and the neigbors
    let mut nodes_to_add: Vec<(Vector3D, Vec<NodeIndex>)> = Vec::new();

    // Find cut endpoint
    for node_idx in graph.node_indices() {
        if let VirtualNodeOrigin::CutEndpointMidpointDuplicate { .. } = graph[node_idx].origin {
            if done.contains(&node_idx) {
                continue;
            }

            // Get the 2 direct neighbors (cutendpoints do not get more edges added until now)
            let neighbors: Vec<NodeIndex> = graph.neighbors(node_idx).collect();
            if neighbors.len() != 2 {
                panic!(
                    "Cut endpoint midpoint duplicate node {:?} has degree {}, expected 2.",
                    node_idx,
                    neighbors.len()
                );
            }

            // If they share an edge, we are in the tri case
            let nodes;
            if graph.find_edge(neighbors[0], neighbors[1]).is_some() {
                // Tri case

                nodes = vec![node_idx, neighbors[0], neighbors[1]];

                // Mark all 3 nodes as done
                done.insert(node_idx);
                done.insert(neighbors[0]);
                done.insert(neighbors[1]);
            } else {
                // Face should be a quad now, we need to find the 4th vertex and connect to it.

                let neighbors_0 = graph
                    .neighbors(neighbors[0])
                    .filter(|n| *n != node_idx)
                    .collect_vec();
                let neighbors_1 = graph
                    .neighbors(neighbors[1])
                    .filter(|n| *n != node_idx)
                    .collect_vec();
                let shared = neighbors_0
                    .iter()
                    .filter(|n| neighbors_1.contains(n))
                    .copied()
                    .collect_vec();
                if shared.len() != 1 {
                    // panic!(
                    //     "Cut endpoint midpoint duplicate node {:?} neighbors do not share exactly one other neighbor as expected for quad face: {:?} and {:?} with shared {:?}",
                    //     node_idx, neighbors_0, neighbors_1, shared
                    // ); FIX THIS!!
                    error!(
                        "is_tri: {:?}, Cut endpoint midpoint duplicate node {:?} neighbors do not share exactly one other neighbor as expected for quad face: {:?} and {:?} with shared {:?}",
                        is_tri_mesh, node_idx, neighbors_0, neighbors_1, shared
                    );
                    continue;
                }

                nodes = vec![node_idx, neighbors[0], neighbors[1], shared[0]];

                // Mark all 4 nodes as done
                done.insert(node_idx);
                done.insert(neighbors[0]);
                done.insert(neighbors[1]);
                done.insert(shared[0]);
            }

            match nodes.len() {
                3 => info!("is_tri: {:?}, Tri face found for cut endpoint {:?} with nodes {:?}", is_tri_mesh, node_idx, nodes),
                4 => info!("is_tri: {:?}, Quad face found for cut endpoint {:?} with nodes {:?}", is_tri_mesh, node_idx, nodes),
                _ => unreachable!(),
            }

            // Add vertex in middle
            let zero = Vector3D::new(0.0, 0.0, 0.0);
            let mid_pos = nodes
                .iter()
                .map(|n| graph[*n].position)
                .fold(zero, |acc, p| acc + p)
                / (nodes.len() as f64);

            nodes_to_add.push((mid_pos, nodes));
        }
    }

    // Add all nodes we need to add
    for (pos, neighbors) in &nodes_to_add {
        let mid_idx = graph.add_node(VirtualNode {
            position: *pos,
            origin: match neighbors.len() {
                3 => VirtualNodeOrigin::ArtificialInTri,
                4 => VirtualNodeOrigin::ArtificialInQuad,
                _ => unreachable!(),
            },
        });

        for n in neighbors {
            let length = (graph[mid_idx].position - graph[*n].position).norm();
            graph.add_edge(mid_idx, *n, VirtualEdgeWeight { length });
        }
    }
}

/// Handles both tri-meshes and quad meshes (and mixes).
/// For input, we assume input is a mix always and look at the face when it matters.
fn mesh_boundary_edges(
    graph: &mut StableUnGraph<VirtualNode, VirtualEdgeWeight>,
    current: NodeIndex,
    next: NodeIndex,
    boundary_lookback: &mut HashMap<EdgeID, NodeIndex>,
    cut_edges: &HashSet<EdgeID>,
    cut_edge_canonical_direction: &HashMap<(VertID, VertID), (VertID, VertID)>,
    vert_to_nodes: &HashMap<VertID, VertexToVirtual>,
    edge_midpoint_ids_to_node_indices: &HashMap<EdgeID, EdgemidpointToVirtual>,
    mesh: &Mesh<INPUT>,
    patch_vertices: &HashSet<VertID>,
) {
    let current_type = &graph[current].origin.clone();
    let next_type = &graph[next].origin.clone();
    match (current_type, next_type) {
        (
            VirtualNodeOrigin::BoundaryMidpoint {
                edge: current_edge, ..
            },
            VirtualNodeOrigin::BoundaryMidpoint {
                edge: next_edge, ..
            },
        ) => {
            // Simplest case: for both edges, add edge from patch vertex to its connecting boundary midpoint
            add_edge(
                graph,
                boundary_lookback,
                vert_to_nodes,
                edge_midpoint_ids_to_node_indices,
                AddingEdgeInput::AlongEdge {
                    source: current,
                    edge: *current_edge,
                },
                mesh,
                patch_vertices,
            );
            add_edge(
                graph,
                boundary_lookback,
                vert_to_nodes,
                edge_midpoint_ids_to_node_indices,
                AddingEdgeInput::AlongEdge {
                    source: next,
                    edge: *next_edge,
                },
                mesh,
                patch_vertices,
            );
        }
        (
            VirtualNodeOrigin::BoundaryMidpoint { .. },
            VirtualNodeOrigin::CutEndpointMidpointDuplicate { .. },
        )
        | (
            VirtualNodeOrigin::CutEndpointMidpointDuplicate { .. },
            VirtualNodeOrigin::BoundaryMidpoint { .. },
        ) => {
            // Walk face (quad or tri), add vertex in the middle and connect to all nodes.
            // This fixes degrees in both current side majority triangle and minority triangle case (1 or 2 nodes inside patch).
            // Also works for quads.

            // Note that walking the face is difficult. The VFG might be partial if we came from the boundary side, but purely relying on original mesh is difficult as well because of the midpoints...

            // We do this in a second pass, after all CutDuplicates have their edges so we can reliably walk faces using only the VFG (no longer using mesh).
        }

        (
            VirtualNodeOrigin::CutEndpointMidpointDuplicate { .. },
            VirtualNodeOrigin::CutEndpointMidpointDuplicate { .. }, 
        ) => {
            // This is kinda 2 cases:
            // - Different cut, then actually there are no other edges to connect. Necessarily, the face is only this edge, the two different cuts going out and then an edge connecting those (quad).
                // We could connect those non-endpoint vertices together but I think the other cases should cover this..
            // - Same cut, then the cut has no internal vertices, there may or may not be edges to add? This is unclear to me now. TODO...
        }

        (
            VirtualNodeOrigin::CutDuplicate {
                original: left_id,
                side,
                ..
            },
            VirtualNodeOrigin::CutDuplicate {
                original: right_id, ..
            },
        ) => {
            // Find vertices of face on our side.
            let our_side_face_vertices = find_cut_side_face_and_vertex(
                mesh,
                cut_edge_canonical_direction,
                *left_id,
                *right_id,
                *side,
            );
            let mut edges_to_add = HashSet::new();
            let starting_face = match our_side_face_vertices {
                CutSideFaceVertices::Tri { face, .. } => face,
                CutSideFaceVertices::Quad { face, .. } => face,
            };

            // Do BFS over faces to find all incident edges on our side of the cut.
            // We start with the face on the correct side we found.
            // We walk the ring of faces around the cut vertex, staying on our side by not crossing cut edges or boundary edges.
            let seen_faces: &mut HashSet<FaceID> = &mut HashSet::new();
            seen_faces.insert(starting_face);
            let mut queue: VecDeque<FaceID> = VecDeque::new();
            queue.push_back(starting_face);

            // Add initial edges from starting face
            let vertices: Vec<VertID> = mesh.vertices(starting_face).collect();
            for vertex in vertices {
                if vertex == *left_id || vertex == *right_id {
                    continue;
                }

                if let Some(edges) = mesh.edge_between_verts(*left_id, vertex) {
                    if !cut_edges.contains(&edges.0) && !cut_edges.contains(&edges.1) {
                        edges_to_add.insert((current, edges));
                    }
                }

                if let Some(edges) = mesh.edge_between_verts(*right_id, vertex) {
                    if !cut_edges.contains(&edges.0) && !cut_edges.contains(&edges.1) {
                        edges_to_add.insert((next, edges));
                    }
                }
            }

            let bfs_step =
                |face_id: FaceID,
                 queue: &mut VecDeque<FaceID>,
                 seen: &mut HashSet<FaceID>,
                 edges_to_add: &mut HashSet<(NodeIndex, (EdgeID, EdgeID))>| {
                    // Every edge is part of exactly 2 faces, one of which will be the one we are looking at.
                    for edge in mesh.edges(face_id) {
                        // Do not traverse over cut edges
                        if cut_edges.contains(&edge) {
                            continue;
                        }

                        // Do not traverse over boundary edges to stay within the patch side.
                        // Boundary edges have midpoints in our map.
                        let twin = mesh.twin(edge);
                        if edge_midpoint_ids_to_node_indices.contains_key(&edge)
                            || edge_midpoint_ids_to_node_indices.contains_key(&twin)
                        {
                            continue;
                        }

                        // Get the other face on this edge
                        let faces = mesh.faces(edge).collect_vec();
                        let [face1, face2] = faces.as_slice() else {
                            panic!("Watertight mesh is expected to have exactly 2 incident faces for each edge, but Edge {:?} has {:?}",
                                            edge, faces
                                        );
                        };
                        let other_face = if *face1 == face_id {
                            *face2
                        } else {
                            *face1
                        };

                        if seen.contains(&other_face) {
                            continue;
                        }
                        seen.insert(other_face);

                        // Stay incident to the cut vertices to explore the "wedge" of faces on this side of the cut.
                        let vertices: Vec<VertID> = mesh.vertices(other_face).collect();
                        if !vertices.contains(left_id) && !vertices.contains(right_id) {
                            continue;
                        }

                        queue.push_back(other_face);

                        // Add all edges from this face to current and next, except along cuts.
                        for vertex in vertices {
                            if vertex == *left_id || vertex == *right_id {
                                continue;
                            }

                            if let Some(edges) = mesh.edge_between_verts(*left_id, vertex) {
                                if !cut_edges.contains(&edges.0) && !cut_edges.contains(&edges.1) {
                                    edges_to_add.insert((current, edges));
                                }
                            }

                            if let Some(edges) = mesh.edge_between_verts(*right_id, vertex) {
                                if !cut_edges.contains(&edges.0) && !cut_edges.contains(&edges.1) {
                                    edges_to_add.insert((next, edges));
                                }
                            }
                        }
                    }
                };

            while !queue.is_empty() {
                bfs_step(
                    queue.pop_front().unwrap(),
                    &mut queue,
                    seen_faces,
                    &mut edges_to_add,
                );
            }

            // For all edges we found, add them to the graph
            for (source, edges) in edges_to_add {
                // I don't think it matters which of the half-edges we use
                let edge = edges.0;

                add_edge(
                    graph,
                    boundary_lookback,
                    vert_to_nodes,
                    edge_midpoint_ids_to_node_indices,
                    AddingEdgeInput::AlongEdge { source, edge },
                    mesh,
                    patch_vertices,
                );
            }
        }

        (
            VirtualNodeOrigin::CutEndpointMidpointDuplicate { .. },
            VirtualNodeOrigin::CutDuplicate { .. },
        )
        | (
            VirtualNodeOrigin::CutDuplicate { .. },
            VirtualNodeOrigin::CutEndpointMidpointDuplicate { .. },
        ) => {
            // We already add all necessary edges for the CutDuplicate, and the corner case is handled in Endpoint-Midpoint case.
            // Nothing to do!
        }

        //
        _ => {
            panic!("Boundary traversal found unexpected pair: from node {:?} to {:?} of types {:?} to {:?}.",
                        current, next, current_type, next_type);
        }
    }
}
