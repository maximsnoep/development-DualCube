use std::collections::{HashMap, HashSet};

use log::error;
use mehsh::prelude::{HasPosition, HasVertices, Mesh, Vector3D};
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::stable_graph::StableUnGraph;
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};

use crate::prelude::{EdgeID, FaceID, VertID, INPUT};
use crate::skeleton::boundary_loop::BoundaryLoop;
use crate::skeleton::orthogonalize::LabeledCurveSkeleton;

use super::{CutPath, CuttingPlan, SurfacePoint};

/// Tracks where a virtual node came from, so that we can map results back to the
/// original mesh and relate duplicated cut nodes to each other.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VirtualNodeOrigin {
    /// A regular mesh vertex that is not on any cut and not a boundary midpoint.
    MeshVertex(VertID),

    /// A boundary-loop midpoint introduced as a real vertex.
    /// Stores the boundary edge whose midpoint this is and the skeleton edge it
    /// belongs to.
    BoundaryMidpoint { edge: EdgeID, boundary: EdgeIndex },

    /// A vertex on a cut that has been duplicated. Each side of the cut gets its
    /// own copy. `peer` points to the node index of the other copy in the same
    /// `VirtualFlatGeometry` graph.
    ///
    /// `original` is the underlying surface point (always a mesh vertex for
    /// edge-following cuts).
    CutDuplicate {
        original: SurfacePoint,
        /// The index of the other copy (the "peer" on the opposite side of the cut).
        /// Set to `None` during construction and filled in once both copies exist.
        peer: Option<NodeIndex>,
        /// Which cut this came from (index into `CuttingPlan::cuts`).
        cut_index: usize,
        /// Which side of the cut: TODO what this means? Likely need this.
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
    pub vert_to_nodes: HashMap<VertID, Vec<NodeIndex>>,

    /// The single boundary loop of this virtual mesh, as an ordered sequence of
    /// node indices. After all cuts are applied the topology is a disk, so the
    /// boundary is one simple cycle. Every node appears at most once.
    pub boundary_loop: Vec<NodeIndex>,
}

/// Per-node payload in the virtual graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualNode {
    pub position: Vector3D,
    pub origin: VirtualNodeOrigin,
}

impl Default for VirtualFlatGeometry {
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
        node_idx: NodeIndex,
        skeleton: &LabeledCurveSkeleton,
        mesh: &Mesh<INPUT>,
        cutting_plan: &CuttingPlan,
    ) -> Self {
        let patch_verts = &skeleton[node_idx].skeleton_node.patch_vertices;
        let patch_set: HashSet<VertID> = patch_verts.iter().copied().collect();

        // Get all vertices of the patch

        // All all boundary midpoints as virtual nodes

        // Determine cut point

        // Wire up edges between virtual nodes according to the boundary (taking care of the cut point, which should not be connected)

        // Wire up internal vertices, simply keep all connections from original mesh (no edge crosses cut anyways)

        // Do cut duplication and wiring based on mesh

        // let vfg = VirtualFlatGeometry {
        //     graph: builder.graph,
        //     vert_to_nodes: builder.vert_to_nodes,
        //     boundary_loop,
        // };

        // Invariant checks.
        // check_invariants(&vfg);

        // vfg

        // TODO:
        VirtualFlatGeometry {
            graph: StableUnGraph::default(),
            vert_to_nodes: HashMap::new(),
            boundary_loop: Vec::new(),
        }
    }
}

/// Checks structural invariants on the completed VFG.
fn check_invariants(vfg: &VirtualFlatGeometry) {
    let boundary_set: HashSet<NodeIndex> = vfg.boundary_loop.iter().copied().collect();

    for node in vfg.graph.node_indices() {
        let degree = vfg.graph.edges(node).count();
        let is_boundary = boundary_set.contains(&node);

        if is_boundary {
            // Boundary nodes need at least 3 neighbour, i.e. the one that has the edge that spawned it, and the two neighbors along the boundary loop.
            assert!(
                degree >= 1,
                "VFG invariant violated: boundary node {:?} ({:?}) has 0 neighbours",
                node,
                vfg.graph[node].origin,
            );
        } else {
            // Interior nodes need at least 3 neighbours (triangle mesh).
            assert!(
                degree >= 3,
                "VFG invariant violated: interior node {:?} ({:?}) has {} neighbours, expected >= 3",
                node,
                vfg.graph[node].origin,
                degree
            );
        }
    }

    // Boundary loop is a simple cycle with at least 3 nodes. Only cut nodes can be duplicated
    // TODO ..


    // Cut vertices connect to 1 boundary node and one on the cut/another boundary
    // TODO...
}
