use std::collections::{HashMap, HashSet};

use itertools::Itertools;
use log::{info, warn};
use mehsh::prelude::{HasEdges, HasFaces, HasVertices, Mesh};
use petgraph::graph::NodeIndex;

use crate::{
    prelude::{EdgeID, FaceID, VertID, INPUT},
    skeleton::cross_parameterize::virtual_mesh::{
        EdgemidpointToVirtual, VertexToVirtual, VirtualEdgeWeight, VirtualFlatGeometry,
        VirtualNodeOrigin,
    },
};

/// Helper: returns a short string describing a node's origin (cut_index, side, degree).
fn describe_node(vfg: &VirtualFlatGeometry, n: NodeIndex) -> String {
    let deg = vfg.graph.edges(n).count();
    match &vfg.graph[n].origin {
        VirtualNodeOrigin::CutDuplicate {
            original,
            cut_index,
            side,
            ..
        } => format!(
            "CutDup(orig={:?}, cut={}, side={}, deg={})",
            original,
            cut_index,
            if *side { "R" } else { "L" },
            deg
        ),
        VirtualNodeOrigin::CutEndpointMidpointDuplicate {
            cut_index, side, ..
        } => format!(
            "CutEndpointMid(cut={}, side={}, deg={})",
            cut_index,
            if *side { "R" } else { "L" },
            deg
        ),
        VirtualNodeOrigin::BoundaryMidpoint { edge, .. } => {
            format!("BoundaryMid(edge={:?}, deg={})", edge, deg)
        }
        VirtualNodeOrigin::MeshVertex(v) => format!("MeshVertex({:?}, deg={})", v, deg),
        other => format!("{:?}(deg={})", other, deg),
    }
}

/// Adds all edges to the VFG based on mesh connectivity. Does not touch boundaries or cuts.
pub fn add_internal_edges(
    vfg: &mut VirtualFlatGeometry,
    mesh: &Mesh<INPUT>,
    patch_vertices: &[VertID],
    vert_to_nodes: HashMap<VertID, VertexToVirtual>,
    edge_midpoint_ids_to_node_indices: HashMap<EdgeID, EdgemidpointToVirtual>,
) {
    // Loop over all patch vertices, then their edges in the original mesh.
    for vert in patch_vertices {
        for edge in mesh.edges(*vert) {
            let edge_vertices = mesh.vertices(edge).collect::<Vec<_>>();
            if edge_vertices.len() != 2 {
                unreachable!(
                    "Mesh edge with {:?} vertices encountered while building VFG",
                    edge_vertices.len()
                );
            }
            let other_vert = if edge_vertices[0] == *vert {
                edge_vertices[1]
            } else if edge_vertices[1] == *vert {
                edge_vertices[0]
            } else {
                unreachable!(
                    "Current vertex {:?} not found among its own edge vertices",
                    vert
                );
            };

            let self_node_origin = vert_to_nodes
                .get(vert)
                .expect("Patch vertex missing from virtual node map");

            if patch_vertices.contains(&other_vert) {
                // Only do work for one direction (only necessary in this case, as we do not search from midpoints)
                if other_vert < *vert {
                    continue;
                }

                let other_node_origin = vert_to_nodes
                    .get(&other_vert)
                    .expect("Patch vertex missing from virtual node map");

                match (self_node_origin, other_node_origin) {
                    // Both unique
                    (VertexToVirtual::Unique(self_node), VertexToVirtual::Unique(other_node)) => {
                        // Both endpoints are regular vertices, just add edge
                        let self_pos = vfg.graph[*self_node].position;
                        let other_pos = vfg.graph[*other_node].position;
                        let length = (self_pos - other_pos).norm();
                        vfg.graph
                            .add_edge(*self_node, *other_node, VirtualEdgeWeight { length });
                    }
                    // All other cases are handled in wiring the boundary. These are just correctness checks.
                    // One cut, one unique
                    (
                        VertexToVirtual::CutPair {
                            left: dup_left,
                            right: dup_right,
                        },
                        VertexToVirtual::Unique(unique),
                    )
                    | (
                        VertexToVirtual::Unique(unique),
                        VertexToVirtual::CutPair {
                            left: dup_left,
                            right: dup_right,
                        },
                    ) => {
                        let has_left = vfg.graph.find_edge(*dup_left, *unique).is_some();
                        let has_right = vfg.graph.find_edge(*dup_right, *unique).is_some();
                        if !has_left && !has_right {
                            panic!(
                                "internal_edges check: MISSING CutPair<->Unique (or Unique<->CutPair). \
                                 self={:?} L=[{:?} {}] R=[{:?} {}] other={:?} [{:?} {}]. \
                                 Mesh edge {:?}<->{:?}",
                                vert,
                                dup_left, describe_node(&vfg, *dup_left),
                                dup_right, describe_node(&vfg, *dup_right),
                                other_vert, unique, describe_node(&vfg, *unique),
                                vert, other_vert
                            );
                        }
                    }
                    // Both cuts
                    (
                        VertexToVirtual::CutPair {
                            left: self_left,
                            right: self_right,
                        },
                        VertexToVirtual::CutPair {
                            left: other_left,
                            right: other_right,
                        },
                    ) => {
                        let has_any = vfg.graph.find_edge(*self_left, *other_left).is_some()
                            || vfg.graph.find_edge(*self_left, *other_right).is_some()
                            || vfg.graph.find_edge(*self_right, *other_left).is_some()
                            || vfg.graph.find_edge(*self_right, *other_right).is_some();
                        if !has_any {
                            panic!(
                                "internal_edges check: MISSING CutPair<->CutPair. \
                                 self={:?} L=[{:?} {}] R=[{:?} {}] \
                                 other={:?} L=[{:?} {}] R=[{:?} {}]. \
                                 Mesh edge {:?}<->{:?}",
                                vert,
                                self_left,
                                describe_node(&vfg, *self_left),
                                self_right,
                                describe_node(&vfg, *self_right),
                                other_vert,
                                other_left,
                                describe_node(&vfg, *other_left),
                                other_right,
                                describe_node(&vfg, *other_right),
                                vert,
                                other_vert
                            );
                        }
                    }
                }
            } else {
                // Other side of edge lies outside patch, so other vertex is not in VFG.
                // Edge should go to the midpoint of this edge instead.

                // Resolve the boundary edge ID against the orientation used by
                // stored boundary loops (either half-edge can appear in mesh.edges()).
                let (edge_ab, edge_ba) = mesh
                    .edge_between_verts(*vert, other_vert)
                    .expect("Expected edge between neighboring vertices to exist");
                let midpoint_edge = if edge_midpoint_ids_to_node_indices.contains_key(&edge_ab) {
                    edge_ab
                } else if edge_midpoint_ids_to_node_indices.contains_key(&edge_ba) {
                    edge_ba
                } else {
                    panic!(
                        "Boundary midpoint missing for edge between {:?} and {:?}. \
                         Neither orientation ({:?}, {:?}) exists in midpoint map.",
                        vert, other_vert, edge_ab, edge_ba
                    );
                };

                // Get node origin
                let midpoint_node_origin = edge_midpoint_ids_to_node_indices
                    .get(&midpoint_edge)
                    .expect("Boundary midpoint missing from virtual node map. Edge goes from inside patch to outside, boundary loop must cross this to be a cycle.");

                // Match based on duplicate/unique status of self and other
                // All cases are handled in wiring the boundary. These are just correctness checks.
                match (self_node_origin, midpoint_node_origin) {
                    // Both unique
                    (
                        VertexToVirtual::Unique(self_node),
                        EdgemidpointToVirtual::Unique(mid_node),
                    ) => {
                        if vfg.graph.find_edge(*self_node, *mid_node).is_none() {
                            panic!(
                                "internal_edges check: MISSING Unique<->UniqueMid. \
                                 self={:?} [{:?} {}] mid=[{:?} {}]. \
                                 Mesh edge {:?}<->{:?} (midpoint edge {:?})",
                                vert,
                                self_node,
                                describe_node(&vfg, *self_node),
                                mid_node,
                                describe_node(&vfg, *mid_node),
                                vert,
                                other_vert,
                                midpoint_edge
                            );
                        }
                    }
                    // Vertex is duplicated, midpoint unique
                    (
                        VertexToVirtual::CutPair {
                            left: self_left,
                            right: self_right,
                        },
                        EdgemidpointToVirtual::Unique(mid_node),
                    ) => {
                        let has_left = vfg.graph.find_edge(*self_left, *mid_node).is_some();
                        let has_right = vfg.graph.find_edge(*self_right, *mid_node).is_some();
                        if !has_left && !has_right {
                            warn!(
                                "internal_edges check: MISSING CutPair<->UniqueMid. \
                                 self={:?} L=[{:?} {}] R=[{:?} {}] mid=[{:?} {}]. \
                                 Mesh edge {:?}<->{:?} (midpoint edge {:?}). Falling back on repair. \
                                 This indicates an underlying issue in boundary wriring logic that should be fixed.",
                                vert,
                                self_left, describe_node(&vfg, *self_left),
                                self_right, describe_node(&vfg, *self_right),
                                mid_node, describe_node(&vfg, *mid_node),
                                vert, other_vert, midpoint_edge
                            );

                            let corresponding_mesh_vertex = match &vfg.graph[*self_left].origin {
                                VirtualNodeOrigin::CutDuplicate { original, .. } => original,
                                other => {
                                    panic!(
                                        "Expected CutPair node to be CutDuplicate, found {:?}",
                                        other
                                    );
                                }
                            };

                            // We look at the 2 faces adjacent to this edge and try to determine the correct target from there.
                            let faces: [FaceID; 2] = mesh
                                .faces(edge)
                                .collect_array()
                                .expect("Expected exactly two faces adjacent to internal edge");

                            // Collect all bounding both faces, filter to ones that have self_left or self_right
                            let bounding_edges = faces
                                .iter()
                                .flat_map(|&face| mesh.edges(face))
                                .filter(|&e| {
                                    let [u, v] = mesh
                                        .vertices(e)
                                        .collect_array()
                                        .expect("Expected edge to have exactly two vertices");
                                    u == *corresponding_mesh_vertex
                                        || v == *corresponding_mesh_vertex
                                })
                                .collect::<HashSet<_>>();

                            // Filter for twins
                            let mut seen_edges = HashSet::new();
                            let mut to_remove = Vec::new();
                            for &e in &bounding_edges {
                                let twin = mesh.twin(e);
                                if seen_edges.contains(&twin) {
                                    // remove twin from bounding_edges to avoid double counting ?
                                    to_remove.push(twin);
                                }
                                seen_edges.insert(e);
                                seen_edges.insert(twin);
                            }
                            for e in to_remove {
                                if bounding_edges.contains(&e) {
                                    seen_edges.remove(&e);
                                }
                            }

                            // Vote which of self_left or self_right is correct
                            let mut left_votes = 0;
                            let mut right_votes = 0;
                            let twin = mesh.twin(edge);
                            for &e in &bounding_edges {
                                if e == edge || e == twin {
                                    continue;
                                }
                                let [u, v] = mesh
                                    .vertices(e)
                                    .collect_array()
                                    .expect("Expected edge to have exactly two vertices");

                                // DEBUG: print the endpoints of the edge we are looking at and `edge`
                                info!(
                                    "Voting on edge {:?}<->{:?} for vertex {:?} (corresponding to {:?}). \
                                     Currently have left_votes={}, right_votes={}. \
                                     Looking at bounding edge {:?} with vertices {:?} and {:?}.",
                                    vert, other_vert, vert, corresponding_mesh_vertex,
                                    left_votes, right_votes,
                                    e, u, v
                                );

                                let other_endpoint = if u == *corresponding_mesh_vertex {
                                    v
                                } else if v == *corresponding_mesh_vertex {
                                    u
                                } else {
                                    continue;
                                };

                                if patch_vertices.contains(&other_endpoint) {
                                    let other_nodes = vert_to_nodes
                                        .get(&other_endpoint)
                                        .expect("Patch vertex missing from virtual node map");
                                    match other_nodes {
                                        VertexToVirtual::Unique(other_node) => {
                                            if vfg.graph.find_edge(*self_left, *other_node).is_some() {
                                                left_votes += 1;
                                                info!(
                                                    "Vote for LEFT because of edge {:?}<->{:?} connecting self_left to {:?} (corresponding to {:?})",
                                                    vert, other_endpoint, other_node, other_endpoint
                                                );
                                            }
                                            if vfg.graph.find_edge(*self_right, *other_node).is_some() {
                                                right_votes += 1;
                                                info!(
                                                    "Vote for RIGHT because of edge {:?}<->{:?} connecting self_right to {:?} (corresponding to {:?})",
                                                    vert, other_endpoint, other_node, other_endpoint
                                                );
                                            }
                                        }
                                        VertexToVirtual::CutPair {
                                            left: other_left,
                                            right: other_right,
                                        } => {
                                            if vfg.graph.find_edge(*self_left, *other_left).is_some()
                                                || vfg.graph.find_edge(*self_left, *other_right).is_some()
                                            {
                                                left_votes += 1;
                                                info!(
                                                    "Vote for LEFT because of edge {:?}<->{:?} connecting self_left to {:?} (corresponding to {:?})",
                                                    vert, other_endpoint, other_left, other_endpoint
                                                );
                                            }
                                            if vfg.graph.find_edge(*self_right, *other_left).is_some()
                                                || vfg.graph.find_edge(*self_right, *other_right).is_some()
                                            {
                                                right_votes += 1;
                                                info!(
                                                    "Vote for RIGHT because of edge {:?}<->{:?} connecting self_right to {:?} (corresponding to {:?})",
                                                    vert, other_endpoint, other_right, other_endpoint
                                                );
                                            }
                                        }
                                    }
                                } else {
                                    // Neighbor is outside patch, so it should correspond to a boundary midpoint.
                                    let midpoint_node_origin = if let Some(m) =
                                        edge_midpoint_ids_to_node_indices.get(&e)
                                    {
                                        Some(m)
                                    } else {
                                        let twin = mesh.twin(e);
                                        edge_midpoint_ids_to_node_indices.get(&twin)
                                    };

                                    let Some(midpoint_node_origin) = midpoint_node_origin else {
                                        continue;
                                    };

                                    match midpoint_node_origin {
                                        EdgemidpointToVirtual::Unique(other_mid) => {
                                            if vfg.graph.find_edge(*self_left, *other_mid).is_some() {
                                                left_votes += 1;
                                                info!(
                                                    "Vote for LEFT because of edge {:?}<->{:?} connecting self_left to midpoint node {:?} (corresponding to edge {:?})",
                                                    vert, other_endpoint, other_mid, e
                                                );
                                            }
                                            if vfg.graph.find_edge(*self_right, *other_mid).is_some() {
                                                right_votes += 1;
                                                info!(
                                                    "Vote for RIGHT because of edge {:?}<->{:?} connecting self_right to midpoint node {:?} (corresponding to edge {:?})",
                                                    vert, other_endpoint, other_mid, e
                                                );
                                            }
                                        }
                                        EdgemidpointToVirtual::CutEndpointPair {
                                            left: mid_left,
                                            right: mid_right,
                                        } => {
                                            if vfg.graph.find_edge(*self_left, *mid_left).is_some()
                                                || vfg.graph.find_edge(*self_left, *mid_right).is_some()
                                            {
                                                left_votes += 1;
                                                info!(
                                                    "Vote for LEFT because of edge {:?}<->{:?} connecting self_left to midpoint node {:?} (corresponding to edge {:?})",
                                                    vert, other_endpoint, mid_left, e
                                                );
                                            }
                                            if vfg.graph.find_edge(*self_right, *mid_left).is_some()
                                                || vfg.graph.find_edge(*self_right, *mid_right).is_some()
                                            {
                                                right_votes += 1;
                                                info!(
                                                    "Vote for RIGHT because of edge {:?}<->{:?} connecting self_right to midpoint node {:?} (corresponding to edge {:?})",
                                                    vert, other_endpoint, mid_right, e
                                                );
                                            }
                                        }
                                    }
                                }
                            }

                            let chosen_side = if left_votes > right_votes {
                                Some(*self_left)
                            } else if right_votes > left_votes {
                                Some(*self_right)
                            } else {
                                None
                            };

                            if let Some(chosen_side) = chosen_side {
                                if left_votes > 0 && right_votes > 0 {
                                    warn!(
                                        "internal_edges fallback: both sides received votes for CutPair<->UniqueMid; choosing majority. \
                                         self={:?} L=[{:?} {}] R=[{:?} {}] mid=[{:?} {}]. \
                                         Mesh edge {:?}<->{:?} (midpoint edge {:?}). Votes: left={}, right={}",
                                        vert,
                                        self_left,
                                        describe_node(&vfg, *self_left),
                                        self_right,
                                        describe_node(&vfg, *self_right),
                                        mid_node,
                                        describe_node(&vfg, *mid_node),
                                        vert,
                                        other_vert,
                                        midpoint_edge,
                                        left_votes,
                                        right_votes
                                    );
                                }

                                let length =
                                    (vfg.graph[chosen_side].position - vfg.graph[*mid_node].position)
                                        .norm();
                                vfg.graph
                                    .add_edge(chosen_side, *mid_node, VirtualEdgeWeight { length });
                            } else if left_votes == 0 && right_votes == 0 {
                                panic!(
                                    "internal_edges check: NO VOTES for CutPair<->UniqueMid. \
                                     self={:?} L=[{:?} {}] R=[{:?} {}] mid=[{:?} {}]. \
                                     Mesh edge {:?}<->{:?} (midpoint edge {:?}). \
                                     Votes: left={}, right={}. Falling back on repair. \
                                     This indicates an underlying issue in boundary wriring logic that should be fixed.",
                                    vert,
                                    self_left, describe_node(&vfg, *self_left),
                                    self_right, describe_node(&vfg, *self_right),
                                    mid_node, describe_node(&vfg, *mid_node),
                                    vert, other_vert, midpoint_edge,
                                    left_votes, right_votes
                                );
                            } else {
                                panic!(
                                    "internal_edges check: TIE while voting CutPair<->UniqueMid. \
                                     self={:?} L=[{:?} {}] R=[{:?} {}] mid=[{:?} {}]. \
                                     Mesh edge {:?}<->{:?} (midpoint edge {:?}). \
                                     Votes: left={}, right={}. Falling back on repair. \
                                     This indicates an underlying issue in boundary wriring logic that should be fixed.",
                                    vert,
                                    self_left, describe_node(&vfg, *self_left),
                                    self_right, describe_node(&vfg, *self_right),
                                    mid_node, describe_node(&vfg, *mid_node),
                                    vert, other_vert, midpoint_edge,
                                    left_votes, right_votes
                                );
                            }
                        }
                    }
                    // Vertex unique, midpoint duplicated
                    (
                        VertexToVirtual::Unique(self_node),
                        EdgemidpointToVirtual::CutEndpointPair {
                            left: mid_left,
                            right: mid_right,
                        },
                    ) => {
                        let has_left = vfg.graph.find_edge(*self_node, *mid_left).is_some();
                        let has_right = vfg.graph.find_edge(*self_node, *mid_right).is_some();
                        if !has_left && !has_right {
                            panic!(
                                "internal_edges check: MISSING Unique<->CutEndpointMid. \
                                 self={:?} [{:?} {}] mid_L=[{:?} {}] mid_R=[{:?} {}]. \
                                 Mesh edge {:?}<->{:?} (midpoint edge {:?})",
                                vert,
                                self_node,
                                describe_node(&vfg, *self_node),
                                mid_left,
                                describe_node(&vfg, *mid_left),
                                mid_right,
                                describe_node(&vfg, *mid_right),
                                vert,
                                other_vert,
                                midpoint_edge
                            );
                        }
                    }
                    // Both duplicated
                    (
                        VertexToVirtual::CutPair {
                            left: self_left,
                            right: self_right,
                        },
                        EdgemidpointToVirtual::CutEndpointPair {
                            left: mid_left,
                            right: mid_right,
                        },
                    ) => {
                        let has_any = vfg.graph.find_edge(*self_left, *mid_left).is_some()
                            || vfg.graph.find_edge(*self_left, *mid_right).is_some()
                            || vfg.graph.find_edge(*self_right, *mid_left).is_some()
                            || vfg.graph.find_edge(*self_right, *mid_right).is_some();
                        if !has_any {
                            panic!(
                                "internal_edges check: MISSING CutPair<->CutEndpointMid. \
                                 self={:?} L=[{:?} {}] R=[{:?} {}] \
                                 mid_L=[{:?} {}] mid_R=[{:?} {}]. \
                                 Mesh edge {:?}<->{:?} (midpoint edge {:?})",
                                vert,
                                self_left,
                                describe_node(&vfg, *self_left),
                                self_right,
                                describe_node(&vfg, *self_right),
                                mid_left,
                                describe_node(&vfg, *mid_left),
                                mid_right,
                                describe_node(&vfg, *mid_right),
                                vert,
                                other_vert,
                                midpoint_edge
                            );
                        }
                    }
                }
            }
        }
    }
}
