use std::collections::{HashMap, HashSet, VecDeque};

use faer::{
    prelude::Solve,
    sparse::{
        linalg::solvers::{Llt, SymbolicLlt},
        SparseColMat, Triplet,
    },
    Mat, Side,
};
use log::{error, warn};
use mehsh::prelude::{
    HasNeighbors, HasPosition,  Mesh, Vector3D,
};
use nalgebra::Matrix3;
use petgraph::graph::{EdgeIndex, NodeIndex};

use crate::{
    prelude::{CurveSkeleton, INPUT, VertID},
    skeleton::curve_skeleton::{CurveSkeletonManipulation, SkeletonNode, patch_centroid},
};

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
            let patch_size_a = self.node_weight(neighbors[0]).unwrap().patch_vertices.len();
            let patch_size_b = self.node_weight(neighbors[1]).unwrap().patch_vertices.len();
            if patch_size_a <= patch_size_b {
                neighbors[0]
            } else {
                neighbors[1]
            }
        };

        // Move patch data to target neighbor
        let mut data_to_move =
            std::mem::take(&mut self.node_weight_mut(node_index).unwrap().patch_vertices);
        self.node_weight_mut(target_neighbor)
            .unwrap()
            .patch_vertices
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
            let size_a = self.node_weight(node_a).unwrap().patch_vertices.len();
            let size_b = self.node_weight(node_b).unwrap().patch_vertices.len();
            let target_total = size_a + size_b;

            // If regions are tiny, skip
            if target_total < 4 {
                continue;
            }

            // Identify Boundaries (Anchors)
            let vertex_map = get_vertex_map(self);

            let mut fixed_a: HashSet<VertID> = HashSet::new();
            let mut fixed_b: HashSet<VertID> = HashSet::new();

            // Check external connections for Node A
            let vertices_a = self.node_weight(node_a).unwrap().patch_vertices.clone();
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
            let vertices_b = self.node_weight(node_b).unwrap().patch_vertices.clone();
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
                let neighbor_pos = self.node_weight(node_b).unwrap().position;
                if let Some(v) = find_furthest_from_point(&vertices_a, neighbor_pos, mesh) {
                    fixed_a.insert(v);
                }
            }
            if fixed_b.is_empty() {
                let neighbor_pos = self.node_weight(node_a).unwrap().position;
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

            let computed_values =
                match solve_harmonic_scalar_field(&all_vertices, &fixed_a, &fixed_b, mesh) {
                    Ok(vals) => vals,
                    Err(e) => {
                        error!(
                            "Harmonic field solve failed for edge {:?}: {:?}",
                            edge_idx, e
                        );
                        continue;
                    }
                };

            // Re-assign Vertices based on Scalar Field
            let mut valued_vertices: Vec<(f64, VertID)> = Vec::with_capacity(target_total);

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

            let new_verts_a: Vec<VertID> = new_a_slice.iter().map(|&(_, v)| v).collect();
            let new_verts_b: Vec<VertID> = new_b_slice.iter().map(|&(_, v)| v).collect();

            // If the new assignment creates islands, revert this refinement.
            if is_connected(&new_verts_a, mesh) && is_connected(&new_verts_b, mesh) {
                // Commit changes
                self.node_weight_mut(node_a).unwrap().patch_vertices = new_verts_a;
                self.node_weight_mut(node_b).unwrap().patch_vertices = new_verts_b;
            } else {
                warn!(
                    "Boundary refinement failed connectivity check for edge {:?}, skipping.",
                    edge_idx
                );
            }
        }
    }

    fn subdivide_edge(&mut self, edge_index: EdgeIndex, mesh: &Mesh<INPUT>) -> bool {
        let (left_index, right_index) = match self.edge_endpoints(edge_index) {
            Some(ep) => ep,
            None => return false,
        };

        let vertex_map = get_vertex_map(self);

        let mut left_boundary: HashSet<VertID> = HashSet::new();
        let mut right_boundary: HashSet<VertID> = HashSet::new();

        // Find boundary vertices: vertices adjacent to an external region.
        let vertices_left = self.node_weight(left_index).unwrap().patch_vertices.clone();
        for &v in &vertices_left {
            for nbr in mesh.neighbors(v) {
                if let Some(&nbr_region) = vertex_map.get(&nbr) {
                    if nbr_region != left_index && nbr_region != right_index {
                        left_boundary.insert(v);
                        break;
                    }
                }
            }
        }

        let vertices_right = self
            .node_weight(right_index)
            .unwrap()
            .patch_vertices
            .clone();
        for &v in &vertices_right {
            for nbr in mesh.neighbors(v) {
                if let Some(&nbr_region) = vertex_map.get(&nbr) {
                    if nbr_region != left_index && nbr_region != right_index {
                        right_boundary.insert(v);
                        break;
                    }
                }
            }
        }

        // If left and right boundaries overlap, they touch externally — abort.
        if !left_boundary.is_disjoint(&right_boundary) {
            return false;
        }

        // For leaf regions with no external boundary, pin the vertex furthest from the interface.
        if left_boundary.is_empty() {
            let neighbor_pos = self.node_weight(right_index).unwrap().position;
            if let Some(v) = find_furthest_from_point(&vertices_left, neighbor_pos, mesh) {
                left_boundary.insert(v);
            }
        }
        if right_boundary.is_empty() {
            let neighbor_pos = self.node_weight(left_index).unwrap().position;
            if let Some(v) = find_furthest_from_point(&vertices_right, neighbor_pos, mesh) {
                right_boundary.insert(v);
            }
        }

        if left_boundary.is_empty() || right_boundary.is_empty() {
            return false;
        }

        // Solve harmonic field over the combined region.
        let mut all_vertices = Vec::with_capacity(vertices_left.len() + vertices_right.len());
        all_vertices.extend_from_slice(&vertices_left);
        all_vertices.extend_from_slice(&vertices_right);

        let computed_values =
            match solve_harmonic_scalar_field(&all_vertices, &left_boundary, &right_boundary, mesh)
            {
                Ok(vals) => vals,
                Err(e) => {
                    warn!(
                        "Harmonic field solve failed for edge {:?}: {:?}",
                        edge_index, e
                    );
                    return false;
                }
            };

        // Sort vertices by scalar field value.
        let mut valued_vertices: Vec<(f64, VertID)> = Vec::with_capacity(all_vertices.len());
        for (v, val) in computed_values {
            valued_vertices.push((val, v));
        }
        for &v in &left_boundary {
            valued_vertices.push((0.0, v));
        }
        for &v in &right_boundary {
            valued_vertices.push((1.0, v));
        }
        valued_vertices.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Split into 3 roughly equal chunks.
        let total = valued_vertices.len();
        let split_1 = total / 3;
        let split_2 = (total * 2) / 3;

        let new_left: Vec<VertID> = valued_vertices[..split_1].iter().map(|&(_, v)| v).collect();
        let new_mid: Vec<VertID> = valued_vertices[split_1..split_2]
            .iter()
            .map(|&(_, v)| v)
            .collect();
        let new_right: Vec<VertID> = valued_vertices[split_2..].iter().map(|&(_, v)| v).collect();

        if new_left.is_empty() || new_mid.is_empty() || new_right.is_empty() {
            return false;
        }

        // Verify new left and right don't share a direct mesh edge.
        let right_set: HashSet<VertID> = new_right.iter().copied().collect();
        for &v in &new_left {
            for nbr in mesh.neighbors(v) {
                if right_set.contains(&nbr) {
                    return false;
                }
            }
        }

        // Place the new node at the centroid of its assigned patch.
        let p_mid = patch_centroid(&new_mid, mesh);

        self.node_weight_mut(left_index).unwrap().patch_vertices = new_left;
        self.node_weight_mut(right_index).unwrap().patch_vertices = new_right;

        let mid_index = self.add_node(SkeletonNode {
            position: p_mid,
            patch_vertices: new_mid,
        });

        // Replace L <-> R with L <-> M <-> R.ß
        self.remove_edge(edge_index);
        self.add_edge(left_index, mid_index, ());
        self.add_edge(mid_index, right_index, ());

        true
    }

    fn split_high_degree_node(&mut self, node_index: NodeIndex, mesh: &Mesh<INPUT>) -> bool {
        let neighbors: Vec<NodeIndex> = self.neighbors(node_index).collect();
        let degree = neighbors.len();

        // Need at least 4 neighbors to split into two groups of >= 2.
        if degree < 4 {
            return false;
        }

        let node_pos = self.node_weight(node_index).unwrap().position;

        // Find the axis of maximum variance among neighbor directions (PCA).
        let mut cov = Matrix3::<f64>::zeros();
        for &nbr in &neighbors {
            let nbr_pos = self.node_weight(nbr).unwrap().position;
            let dir = nbr_pos - node_pos;
            let len = dir.norm();
            let dir = if len > 1e-12 {
                dir / len
            } else {
                Vector3D::zeros()
            };
            for i in 0..3 {
                for j in 0..3 {
                    cov[(i, j)] += dir[i] * dir[j];
                }
            }
        }

        let eigen = cov.symmetric_eigen();
        let max_idx = eigen.eigenvalues.imax();
        let split_axis = eigen.eigenvectors.column(max_idx).into_owned();

        // Project neighbors onto the split axis and sort.
        let mut nbr_proj: Vec<(f64, NodeIndex)> = neighbors
            .iter()
            .map(|&nbr| {
                let nbr_pos = self.node_weight(nbr).unwrap().position;
                let dir = nbr_pos - node_pos;
                let len = dir.norm();
                let dir = if len > 1e-12 {
                    dir / len
                } else {
                    Vector3D::zeros()
                };
                (dir.dot(&split_axis), nbr)
            })
            .collect();

        nbr_proj.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Find the largest gap in projections, ensuring at least 2 neighbors per group.
        let min_group_size = 2;
        let max_split_idx = degree - min_group_size;
        let mut best_split_idx = degree / 2;
        let mut max_gap = -1.0;

        for i in min_group_size..=max_split_idx {
            let gap = nbr_proj[i].0 - nbr_proj[i - 1].0;
            if gap > max_gap {
                max_gap = gap;
                best_split_idx = i;
            }
        }

        let (group_a_proj, group_b_proj) = nbr_proj.split_at(best_split_idx);
        let group_a_nodes: HashSet<NodeIndex> = group_a_proj.iter().map(|(_, n)| *n).collect();
        let group_b_nodes: HashSet<NodeIndex> = group_b_proj.iter().map(|(_, n)| *n).collect();

        // Identify mesh boundary vertices between this node's patch and its neighbor groups.
        let vertex_map = get_vertex_map(self);
        let node_vertices = self.node_weight(node_index).unwrap().patch_vertices.clone();

        let mut fixed_0: HashSet<VertID> = HashSet::new();
        let mut fixed_1: HashSet<VertID> = HashSet::new();

        for &v in &node_vertices {
            for nbr in mesh.neighbors(v) {
                if let Some(&nbr_region) = vertex_map.get(&nbr) {
                    if nbr_region != node_index {
                        if group_a_nodes.contains(&nbr_region) {
                            fixed_0.insert(v);
                        } else if group_b_nodes.contains(&nbr_region) {
                            fixed_1.insert(v);
                        }
                        break;
                    }
                }
            }
        }

        if fixed_0.is_empty() || fixed_1.is_empty() {
            return false;
        }
        if !fixed_0.is_disjoint(&fixed_1) {
            return false;
        }

        // Solve harmonic field to partition the patch.
        let computed_values =
            match solve_harmonic_scalar_field(&node_vertices, &fixed_0, &fixed_1, mesh) {
                Ok(vals) => vals,
                Err(e) => {
                    warn!(
                        "Harmonic field solve failed for node {:?}: {:?}",
                        node_index, e
                    );
                    return false;
                }
            };

        // Threshold at 0.5 to split free vertices.
        let mut verts_a = Vec::new();
        let mut verts_b = Vec::new();

        for (v, val) in computed_values {
            if val < 0.5 {
                verts_a.push(v);
            } else {
                verts_b.push(v);
            }
        }

        verts_a.extend(fixed_0.iter());
        verts_b.extend(fixed_1.iter());

        if !is_connected(&verts_a, mesh) || !is_connected(&verts_b, mesh) {
            error!("Split created disconnected regions, aborting.");
            return false;
        }

        // Place new nodes at the centroids of their assigned patches.
        let pos_a = patch_centroid(&verts_a, mesh);
        let pos_b = patch_centroid(&verts_b, mesh);

        let idx_a = self.add_node(SkeletonNode {
            position: pos_a,
            patch_vertices: verts_a,
        });
        let idx_b = self.add_node(SkeletonNode {
            position: pos_b,
            patch_vertices: verts_b,
        });

        self.add_edge(idx_a, idx_b, ());

        // Reconnect neighbors to the appropriate new node.
        for &nbr in &neighbors {
            if group_a_nodes.contains(&nbr) {
                self.add_edge(nbr, idx_a, ());
            } else {
                self.add_edge(nbr, idx_b, ());
            }
        }

        // Remove original node (also removes its old edges).
        self.remove_node(node_index);

        true
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
/// - `Ok(HashMap<VertID, f64>)` mapping free vertex keys to values in the range (0.0, 1.0).
///     Note that this range is exclusive, as only the boundaries are fixed to 0.0 and 1.0.
/// - `Err(HarmonicFieldError)` if the solve could not be performed.
fn solve_harmonic_scalar_field(
    all_candidate_vertices: &[VertID],
    fixed_0: &HashSet<VertID>,
    fixed_1: &HashSet<VertID>,
    mesh: &Mesh<INPUT>,
) -> Result<HashMap<VertID, f64>, HarmonicFieldError> {
    // Map vertices to dense keys for solver.
    let mut free_mapping: HashMap<VertID, usize> = HashMap::new();
    let mut free_vertices: Vec<VertID> = Vec::new();

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
        let neighbors: Vec<VertID> = mesh.neighbors(mesh_key).collect();
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
        Err(_) => return Err(HarmonicFieldError::SymbolicFactorizationFailed),
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
fn get_vertex_map(skeleton: &CurveSkeleton) -> HashMap<VertID, NodeIndex> {
    let mut vertex_map = HashMap::new();

    for node_idx in skeleton.node_indices() {
        if let Some(node) = skeleton.node_weight(node_idx) {
            for &vertex in &node.patch_vertices {
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
fn is_connected(vertices: &[VertID], mesh: &Mesh<INPUT>) -> bool {
    if vertices.is_empty() {
        return true;
    }

    let vert_set: HashSet<VertID> = vertices.iter().copied().collect();
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
    region_verts: &[VertID],
    point: Vector3D,
    mesh: &Mesh<INPUT>,
) -> Option<VertID> {
    region_verts
        .iter()
        .max_by(|&&a, &&b| {
            let dist_a = mesh.position(a).metric_distance(&point);
            let dist_b = mesh.position(b).metric_distance(&point);
            dist_a
                .partial_cmp(&dist_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .copied()
}
