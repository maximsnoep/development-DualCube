//! Skeleton data models and contracts for the initialization pipeline.

use dualcube_dual::{Loop, LoopID};
use dualcube_types::prelude::{Direction, EdgeID, Sign, Vector3D, VertID};
use slotmap::{SecondaryMap, SlotMap};

slotmap::new_key_type! {
    /// A node in one skeleton. Keys are local to that skeleton and its clones.
    pub struct SkeletonNodeID;
    /// An edge in one skeleton. Keys are local to that skeleton and its clones.
    pub struct SkeletonEdgeID;
    /// A cube in one cube layout. Keys are local to that layout and its clones.
    pub struct CubeID;
}

/// A curve skeleton together with its induced partition of the input surface.
///
/// A successful generation stage returns a nonempty connected graph. Every
/// input vertex belongs to exactly one node's patch, and every edge joins two
/// existing, distinct nodes. An edge's boundary is the ordered simple closed
/// boundary between those nodes' patches. A single-node skeleton is valid.
#[derive(Clone, Debug, Default)]
pub struct Skeleton {
    pub nodes: SlotMap<SkeletonNodeID, SkeletonNode>,
    pub edges: SlotMap<SkeletonEdgeID, SkeletonEdge>,
}

#[derive(Clone, Debug)]
pub struct SkeletonNode {
    /// Embedded position in the coordinate system of the original input mesh.
    pub position: Vector3D,
    /// Original input vertices owned by this node.
    pub patch_vertices: Vec<VertID>,
}

#[derive(Clone, Debug)]
pub struct SkeletonEdge {
    /// Endpoints; their order defines the orientation used by [`EdgeLabel`].
    pub nodes: [SkeletonNodeID; 2],
    pub boundary: PatchBoundary,
}

/// A patch boundary represented by crossings of input-mesh edges.
#[derive(Clone, Debug)]
pub struct PatchBoundary {
    /// One half-edge per crossed geometric edge, in cyclic traversal order;
    /// the first entry is not repeated at the end. The crossing lies at the
    /// edge midpoint. This is not the doubled half-edge format of a dual loop.
    /// Winding is not prescribed; consumers must account for surface geometry.
    pub edge_midpoints: Vec<EdgeID>,
}

/// Signed principal axis from `SkeletonEdge::nodes[0]` to `nodes[1]` in the
/// target cube layout, not necessarily in the input-space embedding.
#[derive(Clone, Copy, Debug)]
pub struct EdgeLabel {
    pub direction: Direction,
    pub sign: Sign,
}

/// The skeleton element represented by a cube. Multiple cubes may represent
/// the same element; edge cubes represent the connection between node patches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CubeOwner {
    Node(SkeletonNodeID),
    Edge(SkeletonEdgeID),
}

/// An axis-aligned cube in target space. Target-space units need not match the
/// input mesh. Placement and size are chosen by the labeling/optimization method.
#[derive(Clone, Debug)]
pub struct Cube {
    /// Finite target-space coordinates.
    pub center: Vector3D,
    /// Finite, strictly positive side length.
    pub size: f64,
    /// Links target geometry to the skeleton's input-surface patches.
    pub owner: CubeOwner,
}

/// Explicit cubes and their intended face adjacencies, available to optimization.
///
/// Connections join existing, distinct cubes sharing a complete face, without
/// duplicate pairs. Incidental geometric contact does not create a connection.
/// Cube interiors must not overlap. Every owner refers to the accompanying
/// skeleton; every skeleton node has exactly one cube, 
/// and every skeleton edge has at least one cube. 
#[derive(Clone, Debug, Default)]
pub struct CubeLayout {
    pub cubes: SlotMap<CubeID, Cube>,
    pub connections: Vec<[CubeID; 2]>,
}

/// Complete input/output of optimization, including the target geometry it uses.
#[derive(Clone, Debug)]
pub struct LabeledSkeleton {
    pub skeleton: Skeleton,
    pub labels: SecondaryMap<SkeletonEdgeID, EdgeLabel>,
    pub cubes: CubeLayout, // TODO: maybe separate into own type?
}

/// Dual loops drawn on the surface, able to be primalized into a polycube segmentation.
pub type DualLoops = SlotMap<LoopID, Loop>;
