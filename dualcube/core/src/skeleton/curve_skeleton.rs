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
    /// Removes a node from the skeleton by reassigning its induced surface patch
    /// to its neighboring nodes, and removing the node and its edges.
    /// 
    /// Only works for nodes with degree 2, neighbors also cannot be connected directly yet.
    fn dissolve_subdivision(&mut self, node_index: NodeIndex);

    /// Smooths the boundaries between surface patches induced by the skeleton, by
    /// shifting which surface vertices are assigned which skeleton node along the boundary.
    fn smooth_boundaries(&mut self);

    /// Subdivides the given edge by inserting a new node at its midpoint.
    /// Geometrically, this means inserting a region where the old boundary was.
    /// 
    /// For an edge X-Y, we insert Z such that Z only connects to X and Y,
    /// X and Y no longer connect directly, and that X maintains its original connections
    /// (except Y) and Y maintains its original connections (except X).
    fn subdivide_edge(&mut self, edge_index: EdgeIndex);

    /// Given a node with degree higher than 3, splits it into two nodes,
    /// where each node has at least 2 of the original connections.
    /// Geometrically, this means splitting a region into two regions.
    fn split_high_degree_node(&mut self, node_index: NodeIndex);
}

impl CurveSkeletonManipulation for CurveSkeleton {
    fn dissolve_subdivision(&mut self, node_index: NodeIndex) {
        let neighbors: Vec<NodeIndex> = self.neighbors(node_index).collect();
        assert!(neighbors.len() == 2, "Node must have degree 2 to be dissolved.");
        assert!(self.find_edge(neighbors[0], neighbors[1]).is_none(), "Neighbors must not be directly connected.");

        // We merge the surface patch vertices from this node into one of its neighbors
        // TODO: investigate splitting it more evenly?
        // Prioritise lower degree neighbor, else smaller patch size neighbor
        let degree_a = self.neighbors(neighbors[0]).count();
        let degree_b = self.neighbors(neighbors[1]).count();

        let target_neighbor = if degree_a < degree_b {
            neighbors[0]
        } else if degree_b < degree_a {
            neighbors[1]
        } else {
            // Equal degree, pick smaller patch
            let patch_size_a = self.node_weight(neighbors[0]).unwrap().1.len();
            let patch_size_b = self.node_weight(neighbors[1]).unwrap().1.len();
            if patch_size_a <= patch_size_b {
                neighbors[0]
            } else {
                neighbors[1]
            }
        };

        // Move patch data to target neighbor
        let mut data_to_move = std::mem::take(&mut self.node_weight_mut(node_index).unwrap().1);
        self.node_weight_mut(target_neighbor).unwrap().1.append(&mut data_to_move);

        // Rewire edges
        self.add_edge(neighbors[0], neighbors[1], ());

        // Remove the node itself (including its edges)
        self.remove_node(node_index);
    }

    fn smooth_boundaries(&mut self) {
        todo!()
    }

    fn subdivide_edge(&mut self, edge_index: EdgeIndex) {
        todo!()
    }

    fn split_high_degree_node(&mut self, node_index: NodeIndex) {
        todo!()
    }
}