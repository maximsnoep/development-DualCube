use std::collections::HashMap;

use mehsh::prelude::{HasEdges, HasVertices, Mesh};

use crate::{
    prelude::{EdgeID, VertID, INPUT},
    skeleton::cross_parameterize::virtual_mesh::{
        EdgemidpointToVirtual, VertexToVirtual, VirtualEdgeWeight, VirtualFlatGeometry,
    },
};

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
                            left: self_left,
                            right: self_right,
                        },
                        VertexToVirtual::Unique(other_node),
                    ) => {
                        // Only check whether connection was covered to one of the duplicates before.
                        // TODO : correctness check: edge exists to one of the duplicates
                    }
                    (
                        VertexToVirtual::Unique(self_node),
                        VertexToVirtual::CutPair {
                            left: other_left,
                            right: other_right,
                        },
                    ) => {
                        // Only check whether connection was covered to one of the duplicates before.
                        // TODO : correctness check: edge exists to one of the duplicates before.
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
                        // Only check whether connection was covered to one of the duplicates before.
                        // TODO : edge exist between any of the duplicates of other sides
                    }

                    _ => unreachable!("Unexpected origin pair encountered."),
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
                        // let self_pos = vfg.graph[*self_node].position;
                        // let mid_pos = vfg.graph[*mid_node].position;
                        // let length = (self_pos - mid_pos).norm();
                        // vfg.graph
                        //     .add_edge(*self_node, *mid_node, VirtualEdgeWeight { length });
                        // TODO: correctness check: edge exists
                    }
                    // Vertex is duplicated, midpoint unique
                    (
                        VertexToVirtual::CutPair {
                            left: self_left,
                            right: self_right,
                        },
                        EdgemidpointToVirtual::Unique(mid_node),
                    ) => {
                        // Only check whether connection was covered to one of the duplicates before.
                        // TODO: correctness check: edge exists to one of the duplicates
                    }
                    // Vertex unique, midpoint duplicated
                    (
                        VertexToVirtual::Unique(self_node),
                        EdgemidpointToVirtual::CutEndpointPair {
                            left: mid_left,
                            right: mid_right,
                        },
                    ) => {
                        // Only check whether connection was covered to one of the duplicates before.
                        // TODO : correctness check: edge exists to one of the duplicates
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
                        // Only check whether connection was covered to one of the duplicates before.
                        // TODO : correctness check: edge exist between any of the duplicates of other sides
                    }
                }
            }
        }
    }
}
