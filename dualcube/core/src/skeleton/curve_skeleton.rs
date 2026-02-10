use mehsh::prelude::{EdgeKey, HasFaces, HasPosition, HasSize, HasVertices, Mesh, Vector3D};
use petgraph::graph::{EdgeIndex, NodeIndex, UnGraph};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::prelude::{INPUT, VertID};


/// Nodes store their 3D position and a list of original mesh vertex keys
/// that represent the induced surface patch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkeletonNode {
    /// The embedded 3D position of this node in space, ideally inside the volume of the mesh.
    pub position: Vector3D,

    /// The list of original mesh vertex keys that represent the induced surface patch for this node.
    pub patch_vertices: Vec<VertID>,
}

/// Edges of the graph geometrically represent boundary loops between patches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryLoop {
    /// The list of surface edges that form this boundary loop, in traversal order. 
    /// This is always a simple cycle.
    pub vertices: Vec<EdgeKey<INPUT>>,
}

/// The extracted 1D skeleton, embedded in 3D space.
pub type CurveSkeleton = UnGraph<SkeletonNode, BoundaryLoop>;

/// Methods for manipulating curve skeletons, along with their induced surface patches.
pub trait CurveSkeletonManipulation {
    /// Removes a node from the skeleton by reassigning its induced surface patch
    /// to its neighboring nodes, and removing the node and its edges.
    ///
    /// Only works for nodes with degree 2, neighbors also cannot be connected directly yet.
    fn dissolve_subdivision(&mut self, node_index: NodeIndex);

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
}

/// Computes the centroid position of a set of mesh vertices.
/// Weighs each vertex by the one-ring area inside the patch to approximate true surface centroid.
/// Note that this throws away area from shared faces... (TODO: figure out simple construction to do weigh these properly)
pub fn patch_centroid(vertices: &[VertID], mesh: &Mesh<INPUT>) -> Vector3D {
    if vertices.is_empty() {
        return Vector3D::zeros();
    }

    let vert_set: HashSet<VertID> = vertices.iter().copied().collect();
    let mut weighted_sum = Vector3D::zeros();
    let mut total_area = 0.0;

    for &v in vertices {
        // compute area of faces adjacent to v, weighted by how much of each face is in the patch
        let mut area_v = 0.0;
        for face in mesh.faces(v) {
            let face_verts: Vec<_> = mesh.vertices(face).collect();
            let count_in = face_verts
                .iter()
                .filter(|&&fv| vert_set.contains(&fv))
                .count();
            if count_in == 0 {
                continue;
            }
            let face_area = mesh.size(face);
            area_v += face_area * (count_in as f64) / (face_verts.len() as f64);
        }

        weighted_sum += mesh.position(v) * area_v;
        total_area += area_v;
    }

    weighted_sum / total_area
}
