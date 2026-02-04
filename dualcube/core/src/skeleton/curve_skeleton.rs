use mehsh::prelude::{Vector3D, VertKey};
use petgraph::graph::{EdgeIndex, NodeIndex, UnGraph};

use crate::prelude::INPUT;

/// Nodes store their 3D position and a list of original mesh vertex keys
/// that represent the induced surface patch.
pub type SkeletonNode = (Vector3D, Vec<VertKey<INPUT>>);

/// The extracted 1D skeleton, embedded in 3D space.
pub type CurveSkeleton = UnGraph<SkeletonNode, ()>;

/// Methods for manipulating curve skeletons, along with their induced surface patches.
pub trait CurveSkeletonManipulation {
    /// Smooths the boundaries between surface patches induced by the skeleton, by
    /// shifting which surface vertices are assigned which skeleton node along the boundary.
    fn smooth_boundaries(&mut self);

    /// Subdivides the given edge by inserting a new node at its midpoint,
    /// geometrically, this means inserting a region where the old boundary was.
    /// For an edge X-Y, we insert Z such that Z only connects to X and Y,
    /// X and Y no longer connect directly, and that X maintains its original connections
    /// (except Y) and Y maintains its original connections (except X).
    fn subdivide_edge(&mut self, edge_index: EdgeIndex);

    /// Given a node with degree higher than 3, splits it into two nodes,
    /// where each node has at least 2 of the original connections.
    /// Geometrically, this means splitting a region into two regions.
    fn split_high_degree_node(&mut self, node_index: NodeIndex);
}