use std::collections::{HashMap, HashSet};

use faer::{Mat, Side, prelude::Solve, sparse::{SparseColMat, Triplet, linalg::solvers::{Llt, SymbolicLlt}}};
use mehsh::prelude::{HasNeighbors, Mesh, Vector3D, VertKey};
use petgraph::graph::{EdgeIndex, NodeIndex, UnGraph};

use crate::prelude::INPUT;

/// Alias for vertex keys from the input mesh, used in skeleton node patches.
pub type VKey = VertKey<INPUT>;

/// Nodes store their 3D position and a list of original mesh vertex keys
/// that represent the induced surface patch.
pub type SkeletonNode = (Vector3D, Vec<VKey>);

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
        assert!(
            neighbors.len() == 2,
            "Node must have degree 2 to be dissolved."
        );
        assert!(
            self.find_edge(neighbors[0], neighbors[1]).is_none(),
            "Neighbors must not be directly connected."
        );

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
        self.node_weight_mut(target_neighbor)
            .unwrap()
            .1
            .append(&mut data_to_move);

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

enum HarmonicFieldError {
    /// No free vertices to solve for (e.g. extremely coarse mesh).
    NoFreeVertices,

    /// Failed to construct sparse matrix from triplets.
    FailedToBuildSparseMatrix,

    /// The matrix is not Symmetric Positive Definite (e.g. disconnected free components).
    NonSpdMatrix,

    /// Symbolic factorization failed (e.g. zero-dimension matrix).
    SymbolicFactorizationFailed,
}

/// Solves the Laplace equation `L * x = 0` subject to Dirichlet boundary conditions.
///
/// Returns a map of `vertex key -> calculated_value` for all 'free' vertices,
/// i.e. vertices not present in `fixed_0` or `fixed_1`.
///
/// # Arguments
/// - `all_candidate_vertices` - The set of vertices to include in the solve.
/// - `fixed_0` - Boundary vertices fixed to value 0.0.
/// - `fixed_1` - Boundary vertices fixed to value 1.0.
/// - `mesh` - The mesh providing vertex adjacency information.
///
/// # Returns
/// - `Ok(HashMap<VKey, f64>)` mapping free vertex keys to values in the range (0.0, 1.0).
///     Note that this range is exclusive, as only the boundaries are fixed to 0.0 and 1.0.
/// - `Err(HarmonicFieldError)` if the solve could not be performed.
fn solve_harmonic_scalar_field(
    all_candidate_vertices: &[VKey],
    fixed_0: &HashSet<VKey>,
    fixed_1: &HashSet<VKey>,
    mesh: &Mesh<INPUT>,
) -> Result<HashMap<VKey, f64>, HarmonicFieldError> {
    // Map vertices to dense keys for solver.
    let mut free_mapping: HashMap<VKey, usize> = HashMap::new();
    let mut free_vertices: Vec<VKey> = Vec::new();

    // Get all the free vertices.
    for &v_idx in all_candidate_vertices {
        if !fixed_0.contains(&v_idx) && !fixed_1.contains(&v_idx) {
            free_mapping.insert(v_idx, free_vertices.len());
            free_vertices.push(v_idx);
        }
    }

    let n_free = free_vertices.len();

    // Not enough vertices to perform a solve (e.g. extremely coarse mesh).
    if n_free == 0 {
        return Err(HarmonicFieldError::NoFreeVertices);
    }

    let mut triplets = Vec::new();
    // RHS vector b (n_free x 1).
    let mut rhs = Mat::<f64>::zeros(n_free, 1);

    for (row_idx, &mesh_key) in free_vertices.iter().enumerate() {
        let neighbors: Vec<VKey> = mesh.neighbors(mesh_key).collect();
        let degree = neighbors.len() as f64;

        // Graph Laplacian Diagonal: L_ii = degree
        triplets.push(Triplet::new(row_idx, row_idx, degree));

        for nbr in &neighbors {
            if let Some(&col_idx) = free_mapping.get(nbr) {
                // Free-Free connection: L_ij = -1
                triplets.push(Triplet::new(row_idx, col_idx, -1.0));
            } else if fixed_1.contains(nbr) {
                // Connection to Sink (Fixed value 1.0).
                // In the equation L*x = 0, this term is (-1 * x_nbr).
                // Moved to RHS, it becomes (+1 * 1.0).
                rhs[(row_idx, 0)] += 1.0;
            } else if fixed_0.contains(nbr) {
                // Connection to Source (Fixed value 0.0).
                // Contribution to RHS is (+1 * 0.0) = 0.0.
            }
        }
    }

    // The Graph Laplacian for a connected component with Dirichlet boundaries is
    // Symmetric Positive Definite (SPD). We use Simplicial LLT (Cholesky).

    // Build Sparse Matrix from Triplets
    let lhs = match SparseColMat::<usize, f64>::try_new_from_triplets(n_free, n_free, &triplets) {
        Ok(mat) => mat,
        Err(_) => return Err(HarmonicFieldError::FailedToBuildSparseMatrix),
    };

    // Symbolic Analysis (Permutation ordering to minimize fill-in)
    let symbolic = match SymbolicLlt::try_new(lhs.symbolic(), Side::Lower) {
        Ok(sym) => sym,
        Err(_) => return  Err(HarmonicFieldError::SymbolicFactorizationFailed),
    };

    // Numeric Factorization
    let llt = match Llt::try_new_with_symbolic(symbolic, lhs.as_ref(), Side::Lower) {
        Ok(f) => f,
        Err(_) => return Err(HarmonicFieldError::NonSpdMatrix),
    };

    // Solve for x
    let x_free = llt.solve(rhs.as_ref());

    // Collect results
    let mut result = HashMap::new();
    for (i, &global_v) in free_vertices.iter().enumerate() {
        result.insert(global_v, x_free[(i, 0)]);
    }

    Ok(result)
}
