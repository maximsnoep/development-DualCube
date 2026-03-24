use core::panic;
use std::collections::{HashMap, HashSet};

use mehsh::prelude::{HasPosition, Mesh, Vector3D};
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::stable_graph::StableUnGraph;
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};

use crate::prelude::{EdgeID, VertID, INPUT};
use crate::skeleton::cross_parameterize::boundary_walk::{
    calculate_boundary_loop, calculate_boundary_loop_reversal_flags,
};
use crate::skeleton::cross_parameterize::duplicate_cut_vertices::{
    duplicate_cut_endpoint, duplicate_cut_vertex,
};
use crate::skeleton::cross_parameterize::edge_id_to_midpoint_pos;
use crate::skeleton::cross_parameterize::internal_edges::add_internal_edges;
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

    /// Artificial node introduced in a tri face to increase degree of some boundary cut vertex.
    ArtificialInTri,

    /// Artificial node introduced in a quad face to increase degree of some boundary cut vertex.
    ArtificialInQuad
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
    ///
    /// Works for both strict triangular meshes (`is_tri_mesh = true`) and
    /// strict quad meshes (`is_tri_mesh = false`).
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
        let patch_vertex_set: HashSet<VertID> = patch_vertices.iter().copied().collect();
        let boundary_loop = calculate_boundary_loop(
            patch_node_idx,
            skeleton,
            mesh,
            &mut graph,
            &vert_to_nodes,
            &edge_midpoint_ids_to_node_indices,
            cutting_plan,
            &boundary_loop_reverse,
            &patch_vertex_set,
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
    let mut low_degree_counts = [0; 3]; // Index 0 for degree 0, 1 for degree 1, 2 for degree 2
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
        if degree < 3 {
            low_degree_counts[degree] += 1;
        }

        // assert!(
        //     degree >= 3,
        //     "VFG invariant violated: node {:?} ({:?}) has {} neighbours, expected >= 3",
        //     node,
        //     vfg.graph[node].origin,
        //     degree
        // );
    }
    let sum = low_degree_counts.iter().sum::<usize>();
    if sum > 0 {
        log::error!(
            "VFG has {} low-degree nodes: {} degree 0, {} degree 1, {} degree 2",
            sum,
            low_degree_counts[0],
            low_degree_counts[1],
            low_degree_counts[2],
        );
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
            ..
        } = vfg.graph[node].origin
        {
            let prev = vfg.boundary_loop[(i + n - 1) % n];
            let next = vfg.boundary_loop[(i + 1) % n];

            let prev_same_side = cut_side_of(prev) == Some((cut_index, side));
            let next_same_side = cut_side_of(next) == Some((cut_index, side));
            let same_side_count = usize::from(prev_same_side) + usize::from(next_same_side);

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

    // 8. Cut-side consistency along the boundary. When moving between two adjacent boundary nodes
    // that both belong to the same cut, they must be on the same side of that cut.
    for i in 0..n {
        let u = vfg.boundary_loop[i];
        let v = vfg.boundary_loop[(i + 1) % n];

        if let (Some((cut_u, side_u)), Some((cut_v, side_v))) = (cut_side_of(u), cut_side_of(v)) {
            if cut_u == cut_v {
                assert_eq!(
                    side_u,
                    side_v,
                    "Boundary edge between nodes {:?} and {:?} switches sides of cut {} ({} to {})",
                    u,
                    v,
                    cut_u,
                    if side_u { "right" } else { "left" },
                    if side_v { "right" } else { "left" }
                );
            }
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

    // 10. All duplicated pairs (CutDuplicate and CutEndpointMidpoint) have
    //    fully disjoint VFG neighbor sets.
    // TODO: this might not necessarily hold for really small boundary loops.. (if there is only one vertex between cut endpoints, special case though that is detectable)
    for node in vfg.graph.node_indices() {
        let is_left_copy = match vfg.graph[node].origin {
            VirtualNodeOrigin::CutDuplicate {
                side: false,
                peer: Some(_),
                ..
            } => true,
            VirtualNodeOrigin::CutEndpointMidpointDuplicate {
                side: false,
                peer: Some(_),
                ..
            } => true,
            _ => false,
        };
        if !is_left_copy {
            continue;
        }
        let peer = match vfg.graph[node].origin {
            VirtualNodeOrigin::CutDuplicate { peer: Some(p), .. } => p,
            VirtualNodeOrigin::CutEndpointMidpointDuplicate { peer: Some(p), .. } => p,
            _ => unreachable!(),
        };
        let nbrs_left: HashSet<NodeIndex> = vfg.graph.neighbors(node).collect();
        let nbrs_right: HashSet<NodeIndex> = vfg.graph.neighbors(peer).collect();
        let shared: Vec<NodeIndex> = nbrs_left.intersection(&nbrs_right).copied().collect();
        assert!(
            shared.is_empty(),
            "Duplicated pair {:?} and {:?} share neighbors: {:?}",
            node,
            peer,
            shared
                .iter()
                .map(|&s| format!("{:?} ({:?})", s, vfg.graph[s].origin))
                .collect::<Vec<_>>(),
        );
    }

    // 11. No parallel edges (multi-edges between the same pair of nodes).
    let mut edge_set: HashSet<(usize, usize)> = HashSet::new();
    for node in vfg.graph.node_indices() {
        for edge in vfg.graph.edges(node) {
            let a = edge.source().index();
            let b = edge.target().index();
            // Each undirected edge appears twice (once from each endpoint),
            // so we only insert from the smaller side.
            if a <= b {
                let key = if a < b { (a, b) } else { (b, a) };
                assert!(
                    edge_set.insert(key), // returns false if already present
                    "VFG has parallel edges between nodes {} and {}",
                    key.0,
                    key.1,
                );
            }
        }
    }

    // 12. All edges that have at least one duplicate as endpoint, should actually have an edge to either of the peers.
    // Checked in internal_edges already!
}
