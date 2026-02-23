use mehsh::prelude::{Mesh, Vector3D};
use petgraph::graph::{EdgeIndex, NodeIndex, UnGraph};
use serde::{Deserialize, Serialize};

use crate::{
    prelude::{VertID, INPUT},
    skeleton::boundary_loop::BoundaryLoop,
};

/// Nodes store their 3D position and a list of original mesh vertex keys
/// that represent the induced surface patch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkeletonNode {
    /// The embedded 3D position of this node in space, ideally inside the volume of the mesh.
    pub position: Vector3D,

    /// The list of original mesh vertex keys that represent the induced surface patch for this node.
    pub patch_vertices: Vec<VertID>,
}

/// The extracted 1D skeleton, embedded in 3D space.
pub type CurveSkeleton = UnGraph<SkeletonNode, BoundaryLoop>;

/// Methods for manipulating curve skeletons, along with their induced surface patches.
pub trait CurveSkeletonManipulation {
    /// Removes a node from the skeleton by reassigning its induced surface patch
    /// to its neighboring nodes, and removing the node and its edges.
    ///
    /// Only works for nodes with degree 2, neighbors also cannot be connected directly yet.
    fn dissolve_subdivision(&mut self, node_index: NodeIndex, mesh: &Mesh<INPUT>);

    /// Smooths the boundaries between surface patches induced by the skeleton, by
    /// shifting which surface vertices are assigned which skeleton node along the boundary.
    ///
    /// Uses harmonic fields to re-partition vertices between adjacent regions while
    /// preserving original region sizes and connectivity.
    /// // TODO: rebalance region sizes to make loops close to circles (or regions close to cuboid shaped)
    fn smooth_boundaries(&mut self, mesh: &Mesh<INPUT>);

    /// Subdivides the given edge by inserting a new node at its midpoint.
    /// Geometrically, this means inserting a region where the old boundary was.
    ///
    /// For an edge X-Y, we insert Z such that Z only connects to X and Y,
    /// X and Y no longer connect directly, and that X maintains its original connections
    /// (except Y) and Y maintains its original connections (except X).
    fn subdivide_edge(&mut self, edge_index: EdgeIndex, mesh: &Mesh<INPUT>) -> bool;

    /// Given a node with degree higher than 3, splits it into two nodes,
    /// where each node has at least 2 of the original connections.
    /// Geometrically, this means splitting a region into two regions.
    ///
    /// Uses PCA to cluster neighbors spatially, then a harmonic field to partition
    /// the mesh region. Returns `false` if the node has degree < 4 or partitioning fails.
    fn split_high_degree_node(&mut self, node_index: NodeIndex, mesh: &Mesh<INPUT>) -> bool;

    /// Merges two nodes connected by an edge. Use `merge_behavior` to determine how to assign patch vertices and node position after merging.
    /// This is differen from `dissolve_subdivision` as it can be applied more generally, but does not redraw boundaries.
    /// 
    /// Note that this can move edges 'outside' the mesh!!!
    /// 
    /// This can fail if the source and target share a neighbor.
    fn merge_nodes(&mut self, source: NodeIndex, target: NodeIndex, merge_behavior: MergeBehavior) -> bool;
}

pub enum MergeBehavior {
    /// When merging, assign all patch vertices to the target node. Keep the target node position as is.
    SourceIntoTarget,
    
    /// When merging, assign all patch vertices to the source node. Keep the source node position as is.
    TargetIntoSource,

    /// When merging assign patch vertices to one of the two nodes, set the position of the merged node to the midpoint between the original two nodes.
    Midpoint,
}

/// Methods for calculating geometric properties of curve skeletons and their induced surface patches.
pub trait CurveSkeletonSpatial {
    /// Calculates the volume of the surface patch induced by this node. Note that this is an approximation,
    /// as the patch is likely not watertight, so volume is ill-defined.
    fn patch_volume(&self, node_index: NodeIndex, mesh: &Mesh<INPUT>) -> f64;

    /// Computes the volume of the convex hull of the surface patch induced by this node.
    fn patch_hull_volume(&self, node_index: NodeIndex, mesh: &Mesh<INPUT>) -> f64;

    /// Returns a value in [0,1] representing how convex the patch is, where 1 means perfectly convex and 0 means very non-convex.
    fn patch_convexity_score(&self, node_index: NodeIndex, mesh: &Mesh<INPUT>) -> f64;

    /// Calculates the volume of the union of connected surface patches.
    fn patches_volume(&self, node_indices: &[NodeIndex], mesh: &Mesh<INPUT>) -> f64;

    /// Computes the convex-hull volume of the union of connected patches.
    fn patches_hull_volume(&self, node_indices: &[NodeIndex], mesh: &Mesh<INPUT>) -> f64;

    /// Convexity score for a group of connected patches, in [0,1].
    fn patches_convexity_score(&self, node_indices: &[NodeIndex], mesh: &Mesh<INPUT>) -> f64;

    // TODO: move patch centroid into here maybe
}