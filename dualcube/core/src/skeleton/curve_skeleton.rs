use std::collections::{HashMap, HashSet, VecDeque};

use faer::{Mat, Side, prelude::Solve, sparse::{SparseColMat, Triplet, linalg::solvers::{Llt, SymbolicLlt}}};
use log::{error, warn};
use mehsh::prelude::{HasNeighbors, HasPosition, Mesh, Vector3D, VertKey};
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

    fn smooth_boundaries(&mut self, mesh: &Mesh<INPUT>) {
        // Collect edge indices upfront since we don't change graph topology (only node weights).
        let edges: Vec<_> = self.edge_indices().collect();

        // We might update a region multiple times (if it has multiple neighbors).
        for edge_idx in edges {
            let (node_a, node_b) = match self.edge_endpoints(edge_idx) {
                Some(ep) => ep,
                None => continue,
            };

            // Snapshot current sizes to preserve vertex amounts // TODO: area/volume might be better?
            let size_a = self.node_weight(node_a).unwrap().1.len();
            let size_b = self.node_weight(node_b).unwrap().1.len();
            let target_total = size_a + size_b;

            // If regions are tiny, skip
            if target_total < 4 {
                continue;
            }

            // Identify Boundaries (Anchors)
            let vertex_map = get_vertex_map(self);

            let mut fixed_a: HashSet<VKey> = HashSet::new();
            let mut fixed_b: HashSet<VKey> = HashSet::new();

            // Check external connections for Node A
            let vertices_a = self.node_weight(node_a).unwrap().1.clone();
            for &v in &vertices_a {
                for nbr in mesh.neighbors(v) {
                    if let Some(&nbr_region) = vertex_map.get(&nbr) {
                        if nbr_region != node_a && nbr_region != node_b {
                            fixed_a.insert(v);
                            break;
                        }
                    }
                }
            }

            // Check external connections for Node B
            let vertices_b = self.node_weight(node_b).unwrap().1.clone();
            for &v in &vertices_b {
                for nbr in mesh.neighbors(v) {
                    if let Some(&nbr_region) = vertex_map.get(&nbr) {
                        if nbr_region != node_a && nbr_region != node_b {
                            fixed_b.insert(v);
                            break;
                        }
                    }
                }
            }

            // If a region is a leaf, it has no external boundary. We must pin its "tip".
            // Find the vertex furthest from the neighbor's skeleton position, which will
            // be at the extremity pointing away from the rest of the skeleton.
            if fixed_a.is_empty() {
                let neighbor_pos = self.node_weight(node_b).unwrap().0;
                if let Some(v) = find_furthest_from_point(&vertices_a, neighbor_pos, mesh) {
                    fixed_a.insert(v);
                }
            }
            if fixed_b.is_empty() {
                let neighbor_pos = self.node_weight(node_a).unwrap().0;
                if let Some(v) = find_furthest_from_point(&vertices_b, neighbor_pos, mesh) {
                    fixed_b.insert(v);
                }
            }

            // Safety checks
            if fixed_a.is_empty() || fixed_b.is_empty() {
                continue;
            }
            if !fixed_a.is_disjoint(&fixed_b) {
                continue;
            }

            // Solve Harmonic Field
            let mut all_vertices = Vec::with_capacity(target_total);
            all_vertices.extend_from_slice(&vertices_a);
            all_vertices.extend_from_slice(&vertices_b);

            let computed_values = match solve_harmonic_scalar_field(
                &all_vertices,
                &fixed_a,
                &fixed_b,
                mesh,
            ) {
                Ok(vals) => vals,
                Err(e) => {
                    error!("Harmonic field solve failed for edge {:?}: {:?}", edge_idx, e);
                    continue;
                }
            };

            // Re-assign Vertices based on Volume Fraction
            let mut valued_vertices: Vec<(f64, VKey)> = Vec::with_capacity(target_total);

            // Add computed free vertices
            for (v, val) in computed_values {
                valued_vertices.push((val, v));
            }
            // Add fixed vertices (0.0 for A, 1.0 for B)
            for &v in &fixed_a {
                valued_vertices.push((0.0, v));
            }
            for &v in &fixed_b {
                valued_vertices.push((1.0, v));
            }

            // Sort by scalar field
            valued_vertices
                .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            // Split at the index that restores the original size of Region A.
            // Clamp split index to ensure neither region becomes empty.
            let min_size = 1;
            let max_idx = valued_vertices.len().saturating_sub(min_size);
            let split_idx = size_a.clamp(min_size, max_idx);

            let (new_a_slice, new_b_slice) = valued_vertices.split_at(split_idx);

            let new_verts_a: Vec<VKey> = new_a_slice.iter().map(|&(_, v)| v).collect();
            let new_verts_b: Vec<VKey> = new_b_slice.iter().map(|&(_, v)| v).collect();

            // If the new assignment creates islands, revert this refinement.
            if is_connected(&new_verts_a, mesh) && is_connected(&new_verts_b, mesh) {
                // Commit changes
                self.node_weight_mut(node_a).unwrap().1 = new_verts_a;
                self.node_weight_mut(node_b).unwrap().1 = new_verts_b;
            } else {
                warn!(
                    "Boundary refinement failed connectivity check for edge {:?}, skipping.",
                    edge_idx
                );
            }
        }
    }

    fn subdivide_edge(&mut self, edge_index: EdgeIndex) {
        todo!()
    }

    fn split_high_degree_node(&mut self, node_index: NodeIndex) {
        todo!()
    }
}

#[derive(Debug)]
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

/// Inverts the mapping from skeleton nodes to mesh vertices.
///
/// Returns a map from mesh vertex key to the skeleton node that owns it.
fn get_vertex_map(skeleton: &CurveSkeleton) -> HashMap<VKey, NodeIndex> {
    let mut vertex_map = HashMap::new();

    for node_idx in skeleton.node_indices() {
        if let Some((_, vertices)) = skeleton.node_weight(node_idx) {
            for &vertex in vertices {
                vertex_map.insert(vertex, node_idx);
            }
        }
    }

    vertex_map
}

/// BFS connectivity check for a subset of mesh vertices.
///
/// Returns true if all vertices in the slice are connected via mesh edges
/// that stay within the vertex set.
fn is_connected(vertices: &[VKey], mesh: &Mesh<INPUT>) -> bool {
    if vertices.is_empty() {
        return true;
    }

    let vert_set: HashSet<VKey> = vertices.iter().copied().collect();
    let start = vertices[0];

    let mut queue = VecDeque::new();
    queue.push_back(start);

    let mut visited = HashSet::new();
    visited.insert(start);

    while let Some(curr) = queue.pop_front() {
        for nbr in mesh.neighbors(curr) {
            if vert_set.contains(&nbr) && !visited.contains(&nbr) {
                visited.insert(nbr);
                queue.push_back(nbr);
            }
        }
    }

    visited.len() == vertices.len()
}

/// Finds the vertex in `region_verts` that is furthest (Euclidean distance) from `point`.
///
/// Used to find the "tip" of a leaf region by finding the vertex furthest from
/// the neighbor's skeleton position.
fn find_furthest_from_point(
    region_verts: &[VKey],
    point: Vector3D,
    mesh: &Mesh<INPUT>,
) -> Option<VKey> {
    region_verts
        .iter()
        .max_by(|&&a, &&b| {
            let dist_a = mesh.position(a).metric_distance(&point);
            let dist_b = mesh.position(b).metric_distance(&point);
            dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .copied()
}
