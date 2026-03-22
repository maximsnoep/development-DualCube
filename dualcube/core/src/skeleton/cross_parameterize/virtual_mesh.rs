use std::collections::{HashMap, HashSet};

use itertools::Itertools;
use log::{self, error, warn};
use mehsh::prelude::{HasEdges, HasNeighbors, HasPosition, HasVertices, Mesh, Vector3D};
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::prelude::StableGraph;
use petgraph::stable_graph::StableUnGraph;
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};

use crate::prelude::{EdgeID, VertID, INPUT};
use crate::skeleton::boundary_loop::{self, BoundaryLoop};
use crate::skeleton::cross_parameterize::edge_id_to_midpoint_pos;
use crate::skeleton::orthogonalize::LabeledCurveSkeleton;

use super::{CutPath, CuttingPlan};

/// Tracks where a virtual node came from, so that we can map results back to the
/// original mesh and relate duplicated cut nodes to each other.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VirtualNodeOrigin {
    /// A regular mesh vertex that is not on any cut and not a boundary midpoint.
    MeshVertex(VertID),

    /// A boundary-loop midpoint introduced as a real vertex.
    /// Stores the boundary edge whose midpoint this is and the skeleton edge it
    /// belongs to.
    BoundaryMidpoint {
        edge: EdgeID,
        boundary_edge: EdgeIndex,
    },

    /// A vertex on a cut that has been duplicated. Each side of the cut gets its
    /// own copy. `peer` points to the node index of the other copy in the same
    /// `VirtualFlatGeometry` graph.
    ///
    /// `original` is the underlying surface point (always a mesh vertex for
    /// edge-following cuts).
    CutDuplicate {
        original: VertID,
        /// The index of the other copy (the "peer" on the opposite side of the cut).
        /// Set to `None` during construction and filled in once both copies exist.
        peer: Option<NodeIndex>,
        /// Which cut this came from (index into `CuttingPlan::cuts`).
        cut_index: usize,
        /// Which side of the cut: `false` = left (the side containing the face
        /// to the left of the cut direction start→end), `true` = right.
        side: bool,
    },

    /// A boundary midpoint that sits at the attachment point of a cut.
    /// Like `CutDuplicate`, it is duplicated: each side of the cut gets its own
    /// copy so that they can receive distinct UV positions.
    CutEndpointMidpointDuplicate {
        /// The boundary edge whose midpoint this is.
        edge: EdgeID,
        /// The skeleton edge (boundary loop) this belongs to.
        boundary: EdgeIndex,
        /// The other copy of this midpoint.
        peer: Option<NodeIndex>,
        /// Which cut this is an endpoint of.
        cut_index: usize,
        /// Which side of the cut: `false` = left, `true` = right.
        side: bool,
    },
}

/// Weight on virtual-mesh edges. Currently stores the Euclidean length so that
/// Laplacian weights are easy to compute.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualEdgeWeight {
    pub length: f64,
}

/// A graph-based mesh representation of a single region after cutting.
///
/// The surface is "opened up" along every cut so that each side of a cut has its
/// own copy of the cut vertices. Boundary midpoints from the boundary loops are
/// introduced as explicit vertices. The result is a topological disk with a
/// single boundary loop.
///
/// Cuts follow mesh edges exclusively — no face-interior crossing points exist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualFlatGeometry {
    /// The mesh-like adjacency graph. Each node carries a 3D position plus its
    /// origin information; each edge carries a length.
    pub graph: StableUnGraph<VirtualNode, VirtualEdgeWeight>,

    /// original mesh vertex -> virtual node(s).
    /// Interior vertices map to exactly one node; cut vertices map to two.
    pub vert_to_nodes: HashMap<VertID, VertexToVirtual>,

    /// The single boundary loop of this virtual mesh, as an ordered sequence of
    /// node indices. After all cuts are applied the topology is a disk, so the
    /// boundary is one simple cycle. Every node appears at most once.
    pub boundary_loop: Vec<NodeIndex>,
}

/// For a given original mesh vertex, which virtual node(s) correspond to it in the VFG.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VertexToVirtual {
    Unique(NodeIndex),
    CutPair { left: NodeIndex, right: NodeIndex },
}

pub enum EdgemidpointToVirtual {
    Unique(NodeIndex),
    CutEndpointPair { left: NodeIndex, right: NodeIndex },
}

/// Per-node payload in the virtual graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualNode {
    pub position: Vector3D,
    pub origin: VirtualNodeOrigin,
}

impl Default for VirtualFlatGeometry {
    // TODO: remove this once all degree-0 regions are implemented.
    fn default() -> Self {
        Self::empty()
    }
}

impl VirtualFlatGeometry {
    /// Returns an empty `VirtualFlatGeometry` with no nodes, edges, or boundary.
    /// Used as a placeholder for degree-0 regions that are TODO for now.
    pub fn empty() -> Self {
        VirtualFlatGeometry {
            graph: StableUnGraph::default(),
            vert_to_nodes: HashMap::new(),
            boundary_loop: Vec::new(),
        }
    }

    /// Builds the virtual flat geometry for one side of a region, given the
    /// skeleton, the mesh, and the cutting plan that was already computed.
    pub fn build(
        patch_node_idx: NodeIndex,
        skeleton: &LabeledCurveSkeleton,
        mesh: &Mesh<INPUT>,
        cutting_plan: &CuttingPlan,
    ) -> Self {
        // Initialize empty structures and fill at each step
        let mut graph: StableUnGraph<VirtualNode, VirtualEdgeWeight> = StableUnGraph::default();
        let mut vert_to_nodes: HashMap<VertID, VertexToVirtual> = HashMap::new();

        // Step 1:  add all vertex nodes
        let patch_vertices = &skeleton
            .node_weight(patch_node_idx)
            .unwrap()
            .skeleton_node
            .patch_vertices;
        for vert_id in patch_vertices {
            let vert_pos = mesh.position(*vert_id);
            let virtual_node_idx = graph.add_node(VirtualNode {
                position: vert_pos,
                origin: VirtualNodeOrigin::MeshVertex(*vert_id),
            });
            vert_to_nodes.insert(*vert_id, VertexToVirtual::Unique(virtual_node_idx));
        }

        // Step 2:  add all boundary midpoints
        let boundary_edges = skeleton.edges(patch_node_idx);
        let mut edge_midpoint_ids_to_node_indices: HashMap<EdgeID, EdgemidpointToVirtual> =
            HashMap::new();
        let mut skeleton_edge_to_boundary_midpoint: HashMap<EdgeIndex, Vec<EdgeID>> =
            HashMap::new();
        for boundary_edge in boundary_edges {
            let edge_weight = boundary_edge.weight();
            let midpoint_loop = &edge_weight.boundary_loop;
            skeleton_edge_to_boundary_midpoint
                .insert(boundary_edge.id(), midpoint_loop.edge_midpoints.clone());

            // Add a midpoint vertex for each midpoint in the boundary loop
            for edge_idx in &midpoint_loop.edge_midpoints {
                // Get the two vertices of this boundary edge
                let pos = edge_id_to_midpoint_pos(*edge_idx, mesh);

                // Add the midpoint as a virtual node
                let midpoint_node_idx = graph.add_node(VirtualNode {
                    position: pos,
                    origin: VirtualNodeOrigin::BoundaryMidpoint {
                        edge: *edge_idx,
                        boundary_edge: boundary_edge.id(),
                    },
                });
                edge_midpoint_ids_to_node_indices
                    .insert(*edge_idx, EdgemidpointToVirtual::Unique(midpoint_node_idx));
            }
        }

        // Step 3:  account for cut duplicates in both nodes and edges.
        //          Replace all previously added nodes with their duplcated versions
        for (cut_index, cut_path) in cutting_plan.cuts.iter().enumerate() {
            let path = &cut_path.path;

            // Start cut endpoint duplicate
            let start_midpoint = path.start;
            duplicate_cut_endpoint(
                &mut graph,
                &mut edge_midpoint_ids_to_node_indices,
                cut_index,
                cut_path,
                start_midpoint,
            );

            // Interior vertex duplicates
            let interior_verts = &path.interior_verts;
            for vert in interior_verts {
                duplicate_cut_vertex(&mut graph, &mut vert_to_nodes, cut_index, *vert);
            }

            // End cut endpoint duplicate
            let end_midpoint = path.end;
            duplicate_cut_endpoint(
                &mut graph,
                &mut edge_midpoint_ids_to_node_indices,
                cut_index,
                cut_path,
                end_midpoint,
            );
        }

        // Step 4:  trace boundary loop, add edges as we go
        let boundary_loop = calculate_boundary_loop(patch_node_idx, skeleton, mesh);

        // Step 5:  add all other edges using original mesh connectivity (being careful about duplicates)

        let vfg = VirtualFlatGeometry {
            graph,
            vert_to_nodes,
            boundary_loop,
        };

        // check_invariants(&vfg);

        vfg
    }
}

/// Calculates the single boundary loop of the virtual mesh.
/// The resulting loop is a simple cycle of virtual node indices in CCW order.
fn calculate_boundary_loop(
    patch_node_idx: NodeIndex,
    skeleton: &LabeledCurveSkeleton,
    mesh: &Mesh<INPUT>,
) -> Vec<NodeIndex> {
    let mut boundary_loop = Vec::new();

    let patch_vertices: HashSet<VertID> = skeleton
        .node_weight(patch_node_idx)
        .unwrap()
        .skeleton_node
        .patch_vertices
        .iter()
        .copied()
        .collect();

    // Collect all vertices that have an edge to another patch.
    let mut boundary_vertices = HashSet::new();
    for skeleton_edge in skeleton.edges(patch_node_idx) {
        // Traverse boundary, get edges, add our side vertex to boundary_vertices
        let edge_weight = skeleton_edge.weight();
        let midpoint_loop = &edge_weight.boundary_loop;
        for edge_idx in &midpoint_loop.edge_midpoints {
            let Some([v1, v2]) = mesh.vertices(*edge_idx).collect_array::<2>() else {
                panic!("Edge with not exactly 2 vertices found while calculating boundary loop.");
            };
            if patch_vertices.contains(&v1) && !patch_vertices.contains(&v2) {
                boundary_vertices.insert(v1);
            } else if patch_vertices.contains(&v2) && !patch_vertices.contains(&v1) {
                boundary_vertices.insert(v2);
            } else {
                panic!("Boundary edge does not have exactly one vertex in the patch.");
            }
        }
    }

    // TODO: calculate whether each boundaryloop is in CCW or CW order.


    // TODO: trace boundary using ccw info
    boundary_loop
}

fn duplicate_cut_endpoint(
    graph: &mut StableGraph<VirtualNode, VirtualEdgeWeight, petgraph::Undirected>,
    edge_midpoint_ids_to_node_indices: &mut HashMap<EdgeID, EdgemidpointToVirtual>,
    cut_index: usize,
    cut_path: &CutPath,
    midpoint: EdgeID,
) {
    // Remove original from graph
    let midpoint_node_idx = match edge_midpoint_ids_to_node_indices.get(&midpoint) {
        Some(EdgemidpointToVirtual::Unique(idx)) => *idx,
        Some(EdgemidpointToVirtual::CutEndpointPair { .. }) => panic!(
            "Cut endpoint {:?} is shared by multiple cuts, which is not supported.",
            midpoint
        ),
        None => unreachable!(
            "Cut endpoint {:?} does not correspond to any known boundary midpoint",
            midpoint
        ),
    };
    let midpoint_pos = graph.node_weight(midpoint_node_idx).unwrap().position;
    if graph.remove_node(midpoint_node_idx).is_none() {
        panic!(
            "Cut endpoint is reused for multiple cuts, leaving no space between to parameterize."
        );
    };
    edge_midpoint_ids_to_node_indices.remove(&midpoint);

    // Insert two duplicates, one for each side of the cut
    let dup_virtual_1 = VirtualNode {
        position: midpoint_pos,
        origin: VirtualNodeOrigin::CutEndpointMidpointDuplicate {
            edge: midpoint,
            boundary: cut_path.start_boundary,
            peer: None, // to be filled in after both copies are created
            cut_index: cut_index,
            side: false,
        },
    };
    let dup_virtual_2 = VirtualNode {
        position: midpoint_pos,
        origin: VirtualNodeOrigin::CutEndpointMidpointDuplicate {
            edge: midpoint,
            boundary: cut_path.start_boundary,
            peer: None, // to be filled in after both copies are created
            cut_index: cut_index,
            side: true,
        },
    };
    let dup_node_idx_1 = graph.add_node(dup_virtual_1);
    let dup_node_idx_2 = graph.add_node(dup_virtual_2);

    // Update peer references
    if let VirtualNodeOrigin::CutEndpointMidpointDuplicate { peer, .. } =
        &mut graph[dup_node_idx_1].origin
    {
        *peer = Some(dup_node_idx_2);
    } else {
        unreachable!();
    }
    if let VirtualNodeOrigin::CutEndpointMidpointDuplicate { peer, .. } =
        &mut graph[dup_node_idx_2].origin
    {
        *peer = Some(dup_node_idx_1);
    } else {
        unreachable!();
    }

    // Save as duplicate in map
    edge_midpoint_ids_to_node_indices.insert(
        midpoint,
        EdgemidpointToVirtual::CutEndpointPair {
            left: dup_node_idx_1,
            right: dup_node_idx_2,
        },
    );
}

fn duplicate_cut_vertex(
    graph: &mut StableGraph<VirtualNode, VirtualEdgeWeight, petgraph::Undirected>,
    vert_to_nodes: &mut HashMap<VertID, VertexToVirtual>,
    cut_index: usize,
    vertex: VertID,
) {
    // Remove original from graph
    let node_idx = match vert_to_nodes.get(&vertex) {
        Some(VertexToVirtual::Unique(idx)) => *idx,
        Some(VertexToVirtual::CutPair { .. }) => panic!(
            "Cut vertex {:?} is shared by multiple cuts, which is not supported.",
            vertex
        ),
        None => unreachable!(
            "Cut vertex {:?} does not correspond to any known mesh vertex",
            vertex
        ),
    };
    let pos = graph.node_weight(node_idx).unwrap().position;
    if graph.remove_node(node_idx).is_none() {
        panic!("Cut vertex is reused for multiple cuts, leaving no space between to parameterize.");
    };
    vert_to_nodes.remove(&vertex);

    // Insert two duplicates, one for each side of the cut
    let dup_virtual_1 = VirtualNode {
        position: pos,
        origin: VirtualNodeOrigin::CutDuplicate {
            original: vertex,
            peer: None,
            cut_index,
            side: false,
        },
    };
    let dup_virtual_2 = VirtualNode {
        position: pos,
        origin: VirtualNodeOrigin::CutDuplicate {
            original: vertex,
            peer: None,
            cut_index,
            side: true,
        },
    };
    let dup_node_idx_1 = graph.add_node(dup_virtual_1);
    let dup_node_idx_2 = graph.add_node(dup_virtual_2);

    // Update peer references
    if let VirtualNodeOrigin::CutDuplicate { peer, .. } = &mut graph[dup_node_idx_1].origin {
        *peer = Some(dup_node_idx_2);
    } else {
        unreachable!();
    }
    if let VirtualNodeOrigin::CutDuplicate { peer, .. } = &mut graph[dup_node_idx_2].origin {
        *peer = Some(dup_node_idx_1);
    } else {
        unreachable!();
    }

    // Save as duplicate in map
    vert_to_nodes.insert(
        vertex,
        VertexToVirtual::CutPair {
            left: dup_node_idx_1,
            right: dup_node_idx_2,
        },
    );
}
// /// Checks structural invariants on the completed VFG.
// fn check_invariants(vfg: &VirtualFlatGeometry) {
//     let n_nodes = vfg.graph.node_count();

//     // 1. Boundary loop is non-empty.
//     assert!(!vfg.boundary_loop.is_empty(), "VFG boundary loop is empty");

//     // 2. Boundary loop is a simple cycle (no repeated nodes).
//     let boundary_set: HashSet<NodeIndex> = vfg.boundary_loop.iter().copied().collect();
//     if boundary_set.len() != vfg.boundary_loop.len() {
//         // Failure! Now find which nodes appear multiple times.
//         let mut counts: HashMap<NodeIndex, usize> = HashMap::new();
//         for &n in &vfg.boundary_loop {
//             *counts.entry(n).or_insert(0) += 1;
//         }
//         for (&node, &count) in &counts {
//             if count > 1 {
//                 log::error!(
//                     "VFG boundary duplicate: node {:?} appears {} times, origin: {:?}",
//                     node,
//                     count,
//                     vfg.graph[node].origin,
//                 );
//                 // Show positions in the loop
//                 let positions: Vec<usize> = vfg
//                     .boundary_loop
//                     .iter()
//                     .enumerate()
//                     .filter(|(_, &n)| n == node)
//                     .map(|(i, _)| i)
//                     .collect();
//                 log::error!("  at loop positions: {:?}", positions);
//             }
//         }
//         panic!(
//             "VFG boundary loop has repeated nodes: {} unique out of {} total",
//             boundary_set.len(),
//             vfg.boundary_loop.len(),
//         );
//     }

//     // 3. Every boundary node exists in the graph.
//     for &node in &vfg.boundary_loop {
//         assert!(
//             vfg.graph.node_weight(node).is_some(),
//             "VFG boundary references non-existent node {:?}",
//             node,
//         );
//     }

//     // 4. Degree checks.
//     for node in vfg.graph.node_indices() {
//         let degree = vfg.graph.edges(node).count();

//         assert!(
//             degree >= 3,
//             "VFG invariant violated: node {:?} ({:?}) has {} neighbours, expected >= 3",
//             node,
//             vfg.graph[node].origin,
//             degree
//         );
//     }

//     // 5. All duplicated pairs (CutDuplicate and CutEndpointMidpoint) have matching peers.
//     for node in vfg.graph.node_indices() {
//         let peer_opt = match vfg.graph[node].origin {
//             VirtualNodeOrigin::CutDuplicate {
//                 peer: Some(peer), ..
//             } => Some(peer),
//             VirtualNodeOrigin::CutEndpointMidpoint {
//                 peer: Some(peer), ..
//             } => Some(peer),
//             _ => None,
//         };
//         if let Some(peer) = peer_opt {
//             assert!(
//                 vfg.graph.node_weight(peer).is_some(),
//                 "Duplicate {:?} has peer {:?} that doesn't exist",
//                 node,
//                 peer,
//             );
//             let peer_of_peer = match vfg.graph[peer].origin {
//                 VirtualNodeOrigin::CutDuplicate { peer: Some(p), .. } => Some(p),
//                 VirtualNodeOrigin::CutEndpointMidpoint { peer: Some(p), .. } => Some(p),
//                 _ => None,
//             };
//             assert_eq!(
//                 peer_of_peer,
//                 Some(node),
//                 "Duplicate peer mismatch: {:?} -> {:?} -> {:?}",
//                 node,
//                 peer,
//                 peer_of_peer,
//             );
//         }
//     }

//     // 7. vert_to_nodes consistency.
//     // for (vert, nodes) in &vfg.vert_to_nodes {
//     //     assert!(!nodes.is_empty(), "vert_to_nodes[{:?}] is empty", vert);
//     //     assert!(
//     //         nodes.len() <= 2,
//     //         "vert_to_nodes[{:?}] has {} entries (expected 1 or 2)",
//     //         vert,
//     //         nodes.len()
//     //     );
//     //     for &ni in nodes {
//     //         assert!(
//     //             vfg.graph.node_weight(ni).is_some(),
//     //             "vert_to_nodes[{:?}] references non-existent node {:?}",
//     //             vert,
//     //             ni,
//     //         );
//     //     }
//     // }

//     // 8. All graph nodes are accounted for in vert_to_nodes, midpoints, or cut-endpoint midpoints.
//     // let tracked_nodes: HashSet<NodeIndex> = vfg
//     //     .vert_to_nodes
//     //     .values()
//     //     .flat_map(|v| v.iter().copied())
//     //     .collect();
//     // let midpoint_count = vfg
//     //     .graph
//     //     .node_indices()
//     //     .filter(|n| {
//     //         matches!(
//     //             vfg.graph[*n].origin,
//     //             VirtualNodeOrigin::BoundaryMidpoint { .. }
//     //         )
//     //     })
//     //     .count();
//     // let cut_endpoint_mid_count = vfg
//     //     .graph
//     //     .node_indices()
//     //     .filter(|n| {
//     //         matches!(
//     //             vfg.graph[*n].origin,
//     //             VirtualNodeOrigin::CutEndpointMidpoint { .. }
//     //         )
//     //     })
//     //     .count();
//     // let vertex_node_count = tracked_nodes.len();
//     // assert_eq!(
//     //     vertex_node_count + midpoint_count + cut_endpoint_mid_count,
//     //     n_nodes,
//     //     "Node accounting mismatch: {} vertex + {} midpoint + {} cut-endpoint-midpoint != {} total",
//     //     vertex_node_count,
//     //     midpoint_count,
//     //     cut_endpoint_mid_count,
//     //     n_nodes,
//     // );

//     // 9. All duplicated pairs (CutDuplicate and CutEndpointMidpoint) have
//     //    fully disjoint VFG neighbor sets.
//     for node in vfg.graph.node_indices() {
//         let is_left_copy = match vfg.graph[node].origin {
//             VirtualNodeOrigin::CutDuplicate {
//                 side: false,
//                 peer: Some(_),
//                 ..
//             } => true,
//             VirtualNodeOrigin::CutEndpointMidpoint {
//                 side: false,
//                 peer: Some(_),
//                 ..
//             } => true,
//             _ => false,
//         };
//         if !is_left_copy {
//             continue;
//         }
//         let peer = match vfg.graph[node].origin {
//             VirtualNodeOrigin::CutDuplicate { peer: Some(p), .. } => p,
//             VirtualNodeOrigin::CutEndpointMidpoint { peer: Some(p), .. } => p,
//             _ => unreachable!(),
//         };
//         let nbrs_left: HashSet<NodeIndex> = vfg.graph.neighbors(node).collect();
//         let nbrs_right: HashSet<NodeIndex> = vfg.graph.neighbors(peer).collect();
//         let shared: Vec<NodeIndex> = nbrs_left.intersection(&nbrs_right).copied().collect();
//         assert!(
//             shared.is_empty(),
//             "Duplicated pair {:?} and {:?} share neighbors: {:?}",
//             node,
//             peer,
//             shared
//                 .iter()
//                 .map(|&s| format!("{:?} ({:?})", s, vfg.graph[s].origin))
//                 .collect::<Vec<_>>(),
//         );
//     }

//     // 10. No parallel edges (multi-edges between the same pair of nodes).
//     {
//         let mut edge_set: HashSet<(usize, usize)> = HashSet::new();
//         for node in vfg.graph.node_indices() {
//             for edge in vfg.graph.edges(node) {
//                 let a = edge.source().index();
//                 let b = edge.target().index();
//                 let key = if a < b { (a, b) } else { (b, a) };
//                 // Each undirected edge appears twice (once from each endpoint),
//                 // so we only insert from the smaller side.
//                 if a <= b {
//                     assert!(
//                         edge_set.insert(key),
//                         "VFG has parallel edges between nodes {} and {}",
//                         key.0,
//                         key.1,
//                     );
//                 }
//             }
//         }
//     }

//     // 11. Planarity necessary condition: E ≤ 3V − 6 for simple planar graphs (V ≥ 3).
//     let n_edges = vfg.graph.edge_count();
//     if n_nodes >= 3 {
//         assert!(
//             n_edges <= 3 * n_nodes - 6,
//             "VFG edge count {} exceeds planar bound 3V-6 = {} (V={}). \
//              The graph cannot be planar.",
//             n_edges,
//             3 * n_nodes - 6,
//             n_nodes,
//         );
//     }
// }
