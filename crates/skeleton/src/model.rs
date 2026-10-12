//! Skeleton data models and contracts for the initialization pipeline.

use dualcube_dual::{Loop, LoopID};
use dualcube_types::prelude::{Direction, EdgeID, Sign, Vector3D};
use slotmap::{SecondaryMap, SlotMap};
use std::collections::HashMap;

slotmap::new_key_type! {
    /// A node in one skeleton. Keys are local to that skeleton and its clones.
    pub struct SkeletonNodeID;
    /// An edge in one skeleton. Keys are local to that skeleton and its clones.
    pub struct SkeletonEdgeID;
    /// One occurrence of a boundary crossing an input-mesh edge.
    pub struct BoundaryCrossingID;
    /// A cube in one cube layout. Keys are local to that layout and its clones.
    pub struct CubeID;
}

/// A curve skeleton whose patches are surface regions bounded by its edge loops.
///
/// Successful extraction returns connected genus-zero patches with disjoint boundaries.
/// Regions may contain no original vertices. Input mesh topology remains immutable.
/// Full validation after later edits awaits boundary-region reconstruction.
#[derive(Clone, Debug, Default)]
pub struct Skeleton {
    pub nodes: SlotMap<SkeletonNodeID, SkeletonNode>,
    pub edges: SlotMap<SkeletonEdgeID, SkeletonEdge>,
    pub surface: BoundarySurface,
}

#[derive(Clone, Debug)]
pub struct SkeletonNode {
    /// Embedded position in the coordinate system of the original input mesh.
    pub position: Vector3D,
}

#[derive(Clone, Debug)]
pub struct SkeletonEdge {
    /// Left and right surface regions along the oriented boundary loop.
    /// Their order also defines the orientation used by [`EdgeLabel`].
    pub nodes: [SkeletonNodeID; 2],
    pub boundary: PatchBoundary,
}

/// An oriented, simple closed curve on the input surface; no axis is prescribed.
#[derive(Clone, Debug, Default)]
pub struct PatchBoundary {
    /// Cyclic traversal, without repeating the first occurrence at the end.
    /// An input edge may occur repeatedly, with distinct crossing IDs.
    pub crossings: Vec<BoundaryCrossingID>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundaryCrossing {
    /// Half-edge of the triangle the traversal exits; it enters through its twin.
    /// The arc to the next crossing lies in the triangle of that twin.
    pub exit: EdgeID,
}

/// Shared combinatorial ordering of all boundary crossings on the immutable input.
#[derive(Clone, Debug, Default)]
pub struct BoundarySurface {
    pub crossings: SlotMap<BoundaryCrossingID, BoundaryCrossing>,
    /// Root-to-tip order on the smaller of each pair of twin half-edge IDs.
    /// Each occurrence belongs to one loop and appears exactly once in these orders.
    // TODO(step 5): Derive uniform offsets (rank + 1)/(count + 1); later optimize geometry.
    pub edge_order: HashMap<EdgeID, Vec<BoundaryCrossingID>>,
}

/// Signed principal axis from `SkeletonEdge::nodes[0]` to `nodes[1]` in the
/// target cube layout, not necessarily in the input-space embedding.
#[derive(Clone, Copy, Debug)]
pub struct EdgeLabel {
    pub direction: Direction,
    pub sign: Sign,
}

/// Correspondence between target cubes and input-surface patches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CubeOwner {
    /// A node cube, entirely owned by that node's patch.
    Node(SkeletonNodeID),
    /// An edge cube entirely owned by one endpoint's patch.
    Edge {
        edge: SkeletonEdgeID,
        /// Must be an endpoint of `edge`.
        patch: SkeletonNodeID,
    },
    /// Split at the cube center, perpendicular to the edge's signed axis.
    /// The positive side belongs to `nodes[1]`, the other side to `nodes[0]`.
    Midpoint(SkeletonEdgeID),
}

/// An axis-aligned cube in target space. Target-space units need not match the
/// input mesh. Placement and size are chosen by cube placement/optimization.
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
/// Global intersections and coincident cubes are allowed. Owners refer to the
/// accompanying skeleton; every node has exactly one node cube. Each edge has one midpoint
/// cube and any additional edge cubes belong to the endpoint on their side.
#[derive(Clone, Debug, Default)]
pub struct CubeLayout {
    pub cubes: SlotMap<CubeID, Cube>,
    pub connections: Vec<[CubeID; 2]>,
}

/// Combinatorial labeling, before cube placement. Every edge has a signed axis.
/// Incident outward ports are distinct, and positive edge lengths admit consistent
/// coordinates. Global intersections and coincident nodes are allowed.
#[derive(Clone, Debug)]
pub struct LabeledSkeleton {
    pub skeleton: Skeleton,
    pub labels: SecondaryMap<SkeletonEdgeID, EdgeLabel>,
}

/// Labeled skeleton with target geometry, used by optimization and loop drawing.
/// Topology edits must keep labels, patch boundaries, and cube ownership consistent.
#[derive(Clone, Debug)]
pub struct RealizedSkeleton {
    pub labeled: LabeledSkeleton,
    pub cubes: CubeLayout,
}

/// Dual loops drawn on the surface, able to be primalized into a polycube segmentation.
/// Uses main's doubled half-edge format with crossing offsets in (0, 1),
/// complementary on twins and strictly ordered along shared edges.
pub type DualLoops = SlotMap<LoopID, Loop>;
