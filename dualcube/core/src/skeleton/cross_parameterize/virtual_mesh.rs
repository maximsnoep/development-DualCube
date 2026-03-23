use core::panic;
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;

use log::{info, warn};
use mehsh::prelude::{HasEdges, HasPosition, HasVertices, Mesh, Vector3D};
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::prelude::StableGraph;
use petgraph::stable_graph::StableUnGraph;
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};

use crate::prelude::{EdgeID, VertID, INPUT};
use crate::skeleton::cross_parameterize::edge_id_to_midpoint_pos;
use crate::skeleton::orthogonalize::LabeledCurveSkeleton;

use super::CuttingPlan;

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
        /// to the left of the cut direction start->end), `true` = right.
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
/// Cuts follow mesh edges exclusively - no face-interior crossing points exist.
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

    /// For every skeleton boundary loop around this patch: whether its stored
    /// midpoint order must be reversed so traversal keeps this patch on the left.
    #[serde(default)]
    pub boundary_loop_reverse: HashMap<EdgeIndex, bool>,
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
            boundary_loop_reverse: HashMap::new(),
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
                cut_path.start_boundary,
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
                cut_path.end_boundary,
                end_midpoint,
            );
        }

        // Step 4: classify boundary-loop orientation for this patch.
        let boundary_loop_reverse =
            calculate_boundary_loop_reversal_flags(patch_node_idx, skeleton, mesh);

        // Step 5:  trace boundary loop, add edges as we go, both between boundary nodes and to other mesh vertices.
        let boundary_loop = calculate_boundary_loop(
            patch_node_idx,
            skeleton,
            mesh,
            &mut graph,
            &vert_to_nodes,
            &edge_midpoint_ids_to_node_indices,
            cutting_plan,
            &boundary_loop_reverse,
        );

        let mut vfg = VirtualFlatGeometry {
            graph,
            vert_to_nodes: vert_to_nodes.clone(),
            boundary_loop,
            boundary_loop_reverse,
        };

        // Step 6:  add all internal edges using original mesh connectivity
        add_internal_edges(
            &mut vfg,
            mesh,
            patch_vertices,
            vert_to_nodes,
            edge_midpoint_ids_to_node_indices,
        );

        check_invariants(&vfg);

        vfg
    }
}

/// Adds all edges to the VFG based on mesh connectivity. Does not touch boundaries or cuts.
fn add_internal_edges(
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

/// Returns, for each boundary-loop edge around this patch, whether its current
/// midpoint order must be reversed so traversal keeps the patch on the left.
fn calculate_boundary_loop_reversal_flags(
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
fn calculate_boundary_loop(
    patch_node_idx: NodeIndex,
    skeleton: &LabeledCurveSkeleton,
    mesh: &Mesh<INPUT>,
    graph: &mut StableUnGraph<VirtualNode, VirtualEdgeWeight>,
    vert_to_nodes: &HashMap<VertID, VertexToVirtual>,
    edge_midpoint_ids_to_node_indices: &HashMap<EdgeID, EdgemidpointToVirtual>,
    cutting_plan: &CuttingPlan,
    reverse_flags: &HashMap<EdgeIndex, bool>,
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
            unreachable!(
                "Degree-1 patch with no boundary midpoints found."
            )
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

        // TODO: add edges based on the triangle spanned by current and next.

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

fn duplicate_cut_endpoint(
    graph: &mut StableGraph<VirtualNode, VirtualEdgeWeight, petgraph::Undirected>,
    edge_midpoint_ids_to_node_indices: &mut HashMap<EdgeID, EdgemidpointToVirtual>,
    cut_index: usize,
    boundary_edge: EdgeIndex,
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
            boundary: boundary_edge,
            peer: None, // to be filled in after both copies are created
            cut_index: cut_index,
            side: false,
        },
    };
    let dup_virtual_2 = VirtualNode {
        position: midpoint_pos,
        origin: VirtualNodeOrigin::CutEndpointMidpointDuplicate {
            edge: midpoint,
            boundary: boundary_edge,
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

/// Checks structural invariants on the completed VFG.
fn check_invariants(vfg: &VirtualFlatGeometry) {
    // 1. Boundary loop is non-empty.
    assert!(!vfg.boundary_loop.is_empty(), "VFG boundary loop is empty");

    // 2. Boundary loop is a simple cycle (no repeated nodes).
    let boundary_set: HashSet<NodeIndex> = vfg.boundary_loop.iter().copied().collect();
    if boundary_set.len() != vfg.boundary_loop.len() {
        // Failure! Now find which nodes appear multiple times.
        let mut counts: HashMap<NodeIndex, usize> = HashMap::new();
        for &n in &vfg.boundary_loop {
            *counts.entry(n).or_insert(0) += 1;
        }
        for (&node, &count) in &counts {
            if count > 1 {
                log::error!(
                    "VFG boundary duplicate: node {:?} appears {} times, origin: {:?}",
                    node,
                    count,
                    vfg.graph[node].origin,
                );
                // Show positions in the loop
                let positions: Vec<usize> = vfg
                    .boundary_loop
                    .iter()
                    .enumerate()
                    .filter(|(_, &n)| n == node)
                    .map(|(i, _)| i)
                    .collect();
                log::error!("  at loop positions: {:?}", positions);
            }
        }
        panic!(
            "VFG boundary loop has repeated nodes: {} unique out of {} total",
            boundary_set.len(),
            vfg.boundary_loop.len(),
        );
    }

    // 3. Every boundary node exists in the graph.
    for &node in &vfg.boundary_loop {
        assert!(
            vfg.graph.node_weight(node).is_some(),
            "VFG boundary references non-existent node {:?}",
            node,
        );
    }

    // 4. Degree checks.
    for node in vfg.graph.node_indices() {
        let degree = vfg.graph.edges(node).count();

        // TODO: switch back to strong invariant when actually working edge adding...
        // if degree < 3 {
        //     log::error!(
        //         "VFG degree invariant violated: node {:?} ({:?}) has degree {}, expected >= 3",
        //         node,
        //         vfg.graph[node].origin,
        //         degree
        //     );
        // }

        // assert!(
        //     degree >= 3,
        //     "VFG invariant violated: node {:?} ({:?}) has {} neighbours, expected >= 3",
        //     node,
        //     vfg.graph[node].origin,
        //     degree
        // );
    }

    // 5. All duplicated pairs (CutDuplicate and CutEndpointMidpoint) have matching peers.
    for node in vfg.graph.node_indices() {
        let peer_opt = match vfg.graph[node].origin {
            VirtualNodeOrigin::CutDuplicate {
                peer: Some(peer), ..
            } => Some(peer),
            VirtualNodeOrigin::CutEndpointMidpointDuplicate {
                peer: Some(peer), ..
            } => Some(peer),
            _ => None,
        };
        if let Some(peer) = peer_opt {
            assert!(
                vfg.graph.node_weight(peer).is_some(),
                "Duplicate {:?} has peer {:?} that doesn't exist",
                node,
                peer,
            );
            let peer_of_peer = match vfg.graph[peer].origin {
                VirtualNodeOrigin::CutDuplicate { peer: Some(p), .. } => Some(p),
                VirtualNodeOrigin::CutEndpointMidpointDuplicate { peer: Some(p), .. } => Some(p),
                _ => None,
            };
            assert_eq!(
                peer_of_peer,
                Some(node),
                "Duplicate peer mismatch: {:?} -> {:?} -> {:?}",
                node,
                peer,
                peer_of_peer,
            );
        }
    }

    // 6. All duplicated vertices are part of the boundary.
    let boundary_set = vfg.boundary_loop.iter().copied().collect::<HashSet<_>>();
    for node in vfg.graph.node_indices() {
        if let VirtualNodeOrigin::CutDuplicate { original, .. } = vfg.graph[node].origin {
            if !boundary_set.contains(&node) {
                panic!(
                    "Cut duplicate node {:?} (original vertex {:?}) is not part of the boundary loop",
                    node, original
                );
            }
        } else if let VirtualNodeOrigin::CutEndpointMidpointDuplicate { edge, .. } =
            vfg.graph[node].origin
        {
            if !boundary_set.contains(&node) {
                panic!(
                    "Cut endpoint midpoint duplicate node {:?} (original edge {:?}) is not part of the boundary loop",
                    node, edge
                );
            }
        }
    }

    // 7. Local cut-endpoint topology on the boundary cycle. For each duplicated cut endpoint, exactly one
    // of its two boundary neighbors must continue along the same cut side.
    let n = vfg.boundary_loop.len();
    let cut_side_of = |idx: NodeIndex| -> Option<(usize, bool)> {
        match vfg.graph[idx].origin {
            VirtualNodeOrigin::CutDuplicate {
                cut_index, side, ..
            }
            | VirtualNodeOrigin::CutEndpointMidpointDuplicate {
                cut_index, side, ..
            } => Some((cut_index, side)),
            _ => None,
        }
    };
    for (i, &node) in vfg.boundary_loop.iter().enumerate() {
        if let VirtualNodeOrigin::CutEndpointMidpointDuplicate {
            edge,
            cut_index,
            side,
            peer,
            ..
        } = vfg.graph[node].origin
        {
            let peer =
                peer.expect("cut endpoint midpoint duplicate peer should be set by invariant 5");
            let prev = vfg.boundary_loop[(i + n - 1) % n];
            let next = vfg.boundary_loop[(i + 1) % n];

            let prev_same_side = cut_side_of(prev) == Some((cut_index, side));
            let next_same_side = cut_side_of(next) == Some((cut_index, side));
            let same_side_count = usize::from(prev_same_side) + usize::from(next_same_side);

            info!(
                "Checking cut endpoint {:?} (edge {:?}, cut {}, side {}, peer {:?}) prev {:?} next {:?}",
                node,
                edge,
                cut_index,
                if side { "right" } else { "left" },
                peer,
                prev,
                next,
            );

            assert!(
                same_side_count == 1,
                "Cut endpoint {:?} (edge {:?}, cut {}, side {}) has invalid boundary neighborhood: prev={:?} next={:?}; expected exactly one same-side cut neighbor",
                node,
                edge,
                cut_index,
                if side { "right" } else { "left" },
                prev,
                next,
            );
        }
    }

    // // 9. All graph nodes are accounted for in vert_to_nodes, midpoints, or cut-endpoint midpoints.
    // // TODO: double check this
    // let tracked_nodes: HashSet<NodeIndex> = vfg
    //     .vert_to_nodes
    //     .values()
    //     .flat_map(|v| v.iter().copied())
    //     .collect();
    // let midpoint_count = vfg
    //     .graph
    //     .node_indices()
    //     .filter(|n| {
    //         matches!(
    //             vfg.graph[*n].origin,
    //             VirtualNodeOrigin::BoundaryMidpoint { .. }
    //         )
    //     })
    //     .count();
    // let cut_endpoint_mid_count = vfg
    //     .graph
    //     .node_indices()
    //     .filter(|n| {
    //         matches!(
    //             vfg.graph[*n].origin,
    //             VirtualNodeOrigin::CutEndpointMidpoint { .. }
    //         )
    //     })
    //     .count();
    // let vertex_node_count = tracked_nodes.len();
    // assert_eq!(
    //     vertex_node_count + midpoint_count + cut_endpoint_mid_count,
    //     n_nodes,
    //     "Node accounting mismatch: {} vertex + {} midpoint + {} cut-endpoint-midpoint != {} total",
    //     vertex_node_count,
    //     midpoint_count,
    //     cut_endpoint_mid_count,
    //     n_nodes,
    // );

    // // 10. All duplicated pairs (CutDuplicate and CutEndpointMidpoint) have
    // //    fully disjoint VFG neighbor sets.
    // // TODO: this might not necessarily hold for really small boundary loops.. (if there is only one vertex between cut endpoints, special case though that is detectable)
    // for node in vfg.graph.node_indices() {
    //     let is_left_copy = match vfg.graph[node].origin {
    //         VirtualNodeOrigin::CutDuplicate {
    //             side: false,
    //             peer: Some(_),
    //             ..
    //         } => true,
    //         VirtualNodeOrigin::CutEndpointMidpoint {
    //             side: false,
    //             peer: Some(_),
    //             ..
    //         } => true,
    //         _ => false,
    //     };
    //     if !is_left_copy {
    //         continue;
    //     }
    //     let peer = match vfg.graph[node].origin {
    //         VirtualNodeOrigin::CutDuplicate { peer: Some(p), .. } => p,
    //         VirtualNodeOrigin::CutEndpointMidpoint { peer: Some(p), .. } => p,
    //         _ => unreachable!(),
    //     };
    //     let nbrs_left: HashSet<NodeIndex> = vfg.graph.neighbors(node).collect();
    //     let nbrs_right: HashSet<NodeIndex> = vfg.graph.neighbors(peer).collect();
    //     let shared: Vec<NodeIndex> = nbrs_left.intersection(&nbrs_right).copied().collect();
    //     assert!(
    //         shared.is_empty(),
    //         "Duplicated pair {:?} and {:?} share neighbors: {:?}",
    //         node,
    //         peer,
    //         shared
    //             .iter()
    //             .map(|&s| format!("{:?} ({:?})", s, vfg.graph[s].origin))
    //             .collect::<Vec<_>>(),
    //     );
    // }

    // // 11. No parallel edges (multi-edges between the same pair of nodes).
    // {
    //     let mut edge_set: HashSet<(usize, usize)> = HashSet::new();
    //     for node in vfg.graph.node_indices() {
    //         for edge in vfg.graph.edges(node) {
    //             let a = edge.source().index();
    //             let b = edge.target().index();
    //             let key = if a < b { (a, b) } else { (b, a) };
    //             // Each undirected edge appears twice (once from each endpoint),
    //             // so we only insert from the smaller side.
    //             if a <= b {
    //                 assert!(
    //                     edge_set.insert(key),
    //                     "VFG has parallel edges between nodes {} and {}",
    //                     key.0,
    //                     key.1,
    //                 );
    //             }
    //         }
    //     }
    // }
}
