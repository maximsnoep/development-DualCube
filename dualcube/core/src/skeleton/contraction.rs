use std::collections::HashMap;

use faer::{
    linalg::solvers::Solve,
    sparse::{
        linalg::solvers::{Llt, SymbolicLlt},
        SparseColMat, Triplet,
    },
    Mat, Side,
};
use log::{error, info, warn};
use mehsh::prelude::{
    HasFaces, HasNeighbors, HasPosition, HasSize, HasVertices, Mesh, SetPosition, Tag, VertKey,
};
use serde::{Deserialize, Serialize};

/// Tag type for contraction-view of skeleton.
#[derive(Debug, Clone, Default, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CONTRACTION;

/// Epsilon value for volume-based stopping condition.
pub const VOLUME_EPSILON: f64 = 1e-6;

/// Holds the persistent state required across contraction iterations.
struct ContractionState {
    /// We need integer indices for sparse matrix assembly (row/column indices),
    /// so we maintain a stable vertex ordering throughout contraction.
    pub vert_ids: Vec<VertKey<CONTRACTION>>,

    /// Reverse lookup: VertKey -> matrix index.
    /// Used for efficient O(1) index lookups during matrix assembly.
    pub vert_to_idx: HashMap<VertKey<CONTRACTION>, usize>,

    /// Global contraction weight scalar (s_L * W_L).
    /// Starts as 10^-3 * sqrt(AvgArea). Increases by s_L (2.0) each step.
    pub wl: f64,

    /// Per-vertex attraction weights (W_H).
    /// Starts as 1.0. Updates based on ratio of current area to original area.
    pub wh: HashMap<VertKey<CONTRACTION>, f64>,

    /// Original One-Ring Area of each vertex (A^0).
    /// Used to calculate the update for W_H.
    pub original_areas: HashMap<VertKey<CONTRACTION>, f64>,

    /// Original mesh volume. Used for stopping condition.
    pub original_volume: f64,

    /// Whether the mesh cannot be optimized further due to instability.
    pub unstable: bool,

    /// Cache symbolic factorization since topology is static in this step.
    pub symbolic: Option<SymbolicLlt<usize>>,
}

impl ContractionState {
    /// Computes the one-ring area for a vertex (sum of areas of adjacent faces).
    fn one_ring_area(mesh: &Mesh<CONTRACTION>, vertex: VertKey<CONTRACTION>) -> f64 {
        mesh.faces(vertex).map(|face| mesh.size(face)).sum()
    }

    /// Creates a new `ContractionState` from a mesh.
    ///
    /// Initializes:
    /// - `vert_ids` and `vert_to_idx` for matrix index mapping
    /// - `wl` as 10^-3 * sqrt(average one-ring area)
    /// - `wh` as 1.0 for all vertices
    /// - `original_areas` as the one-ring area for each vertex
    /// - `original_volume` from the mesh volume
    pub fn new(mesh: &Mesh<CONTRACTION>) -> Self {
        let vert_ids = mesh.vert_ids();
        let n = vert_ids.len();

        // Build reverse lookup for efficient index queries
        let vert_to_idx: HashMap<VertKey<CONTRACTION>, usize> = vert_ids
            .iter()
            .enumerate()
            .map(|(idx, &v)| (v, idx))
            .collect();

        // Compute one-ring areas and initialize weights for all vertices
        let mut original_areas = HashMap::with_capacity(n);
        let mut wh = HashMap::with_capacity(n);
        let mut total_area = 0.0;

        for &v in &vert_ids {
            let area = Self::one_ring_area(mesh, v);
            original_areas.insert(v, area);
            wh.insert(v, 1.0);
            total_area += area;
        }

        // Initial wl = 10^-3 * sqrt(avg_area)
        let avg_area = total_area / n as f64;
        let wl = 1e-3 * avg_area.sqrt();

        Self {
            vert_ids,
            vert_to_idx,
            wl,
            wh,
            original_areas,
            original_volume: mesh.volume(),
            unstable: false,
            symbolic: None,
        }
    }
}

/// Implements geometry contraction from 'Skeleton Extraction by Mesh Contraction' by Au et al. (2008).
/// This pulls vertices approximately towards the medial axis locally of the shape, using a
/// number of iterations until convergence/instability.
/// See paper Section 4 for more details.
///
/// # Parameters
/// - `mesh`: The input mesh to be contracted.
/// - `max_iterations`: Maximum number of contraction iterations to perform, in case convergence is not reached.
///
/// # Returns
/// - The contracted mesh.
pub fn contract_mesh<M: Tag>(mesh: &Mesh<M>, max_iterations: usize) -> Mesh<CONTRACTION> {
    // Make a copy of the mesh to use for contraction
    let mut contraction_mesh: Mesh<CONTRACTION> =
        mehsh::prelude::Mesh::<CONTRACTION>::convert(mesh);

    // Setup bookkeeping for contraction
    let mut state = ContractionState::new(&contraction_mesh);

    // Contract until convergence or max iterations reached
    for i in 0..max_iterations {
        if state.unstable {
            warn!(
                "Contraction halted due to numerical instability at iteration {}.",
                i - 1
            );
            break;
        }

        let current_volume = contraction_mesh.volume();
        let vol_ratio = current_volume / state.original_volume;

        info!(
            "Geometry contraction iteration {}: Volume Ratio = {:.8}",
            i, vol_ratio
        );

        if vol_ratio < VOLUME_EPSILON {
            info!("Geometry contraction converged at iteration {}.", i);
            break;
        }

        contract_once(&mut contraction_mesh, &mut state);
    }

    contraction_mesh
}

/// Performs a single iteration of geometry contraction.
///
/// # Steps
/// 1. Construct Linear System (W_L^2 * L^T * L + W_H^2) V' = W_H^2 V
/// 2. Solve for new positions V'
/// 3. Update weights W_L and W_H
fn contract_once(mesh: &mut Mesh<CONTRACTION>, state: &mut ContractionState) {
    let num_verts = state.vert_ids.len();

    // Setup Laplacian L triplets for sparse matrix assembly
    let l_triplets = get_laplacian_triplets(mesh, state);

    // Assemble System Matrix A = (W_L * L)^T (W_L * L) + W_H^2,
    // which simplifies to: A = W_L^2 * L^2 + W_H^2 (since L is symmetric).
    // We compute L^2 sparse triplets manually via adjacency.
    let mut a_entries: HashMap<(usize, usize), f64> = HashMap::new();

    // Build efficient adjacency for L to compute L*L
    let mut adj_l: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_verts];
    for (r, c, v) in &l_triplets {
        adj_l[*r].push((*c, *v));
    }

    let wl_sq = state.wl * state.wl;

    // A += W_L^2 * (L * L)
    for i in 0..num_verts {
        for (k, val_ik) in &adj_l[i] {
            // L_ik exists. Now find L_kj (neighbors of k).
            for (j, val_kj) in &adj_l[*k] {
                let val = wl_sq * val_ik * val_kj;
                *a_entries.entry((i, *j)).or_default() += val;
            }
        }
    }

    // A += W_H^2 (Diagonal)
    for (i, &v) in state.vert_ids.iter().enumerate() {
        let wh = state.wh[&v];
        let wh_sq = wh * wh;
        *a_entries.entry((i, i)).or_default() += wh_sq;
    }

    // Convert Map to faer Triplets
    let triplets_a: Vec<Triplet<usize, usize, f64>> = a_entries
        .into_iter()
        .map(|((r, c), val)| Triplet::new(r, c, val))
        .collect();

    // Create Sparse Matrix A
    let mat_a = SparseColMat::try_new_from_triplets(num_verts, num_verts, &triplets_a)
        .expect("Failed to create sparse matrix from triplets");

    // Assemble RHS Matrix B = W_H^2 * V_old (dense N x 3 matrix)
    let mat_b = Mat::from_fn(num_verts, 3, |r, c| {
        let v = state.vert_ids[r];
        let wh = state.wh[&v];
        let wh_sq = wh * wh;
        let pos = mesh.position(v);
        match c {
            0 => wh_sq * pos.x,
            1 => wh_sq * pos.y,
            _ => wh_sq * pos.z,
        }
    });

    // Solve AX = B
    // A is Symmetric Positive Definite (SPD) -> Use Cholesky (LLT)
    // Split into symbolic and numeric passes for efficiency
    if state.symbolic.is_none() {
        match SymbolicLlt::try_new(mat_a.symbolic(), Side::Lower) {
            Ok(sym) => state.symbolic = Some(sym),
            Err(error) => {
                warn!("Symbolic factorization failed: {}", error);
                state.unstable = true;
                return;
            }
        }
    }
    // Guaranteed to exist now
    let symbolic = state.symbolic.clone().unwrap();

    let llt = match Llt::try_new_with_symbolic(symbolic, mat_a.as_ref(), Side::Lower) {
        Ok(factorization) => factorization,
        Err(error) => {
            warn!("Numeric factorization failed: {}", error);
            state.unstable = true;
            return;
        }
    };
    let solution = llt.solve(mat_b.as_ref());

    // Update vertex positions from solution
    for (i, &v) in state.vert_ids.iter().enumerate() {
        let new_pos = mehsh::prelude::Vector3D::new(solution[(i, 0)], solution[(i, 1)], solution[(i, 2)]);
        mesh.set_position(v, new_pos);
    }

    // Update Weights for next iteration
    const SL: f64 = 2.0;
    state.wl *= SL;
    const MIN_VERTEX_AREA: f64 = 1e-12;

    for &v in &state.vert_ids {
        let current_area = ContractionState::one_ring_area(mesh, v);
        let safe_area = current_area.max(MIN_VERTEX_AREA);
        // Eq: W_H_new = W_H_old * sqrt(A_0 / A_new).
        // Since we only store current W_H and W_H^0=1, this simplifies to sqrt(A_0 / A_new).
        let original_area = state.original_areas[&v];
        state.wh.insert(v, (original_area / safe_area).sqrt());
    }
}

/// Computes the cotangent weight for a specific edge between v_i and v_j.
/// This corresponds to (cot alpha + cot beta) in Eq 1.
///
/// Returns 0.0 if edge doesn't exist or for degenerate triangles to avoid NaNs.
fn cotangent_weight(
    mesh: &Mesh<CONTRACTION>,
    v_i: VertKey<CONTRACTION>,
    v_j: VertKey<CONTRACTION>,
) -> f64 {
    // Find the edge between the two vertices
    let Some((edge_ij, _twin)) = mesh.edge_between_verts(v_i, v_j) else {
        return 0.0;
    };

    let p_i = mesh.position(v_i);
    let p_j = mesh.position(v_j);

    // Cap the cotangent value to prevent numerical instability.
    const MAX_COTAN: f64 = 1e12;
    // Cap crossproduct length to avoid division by near-zero
    const MIN_CROSSPROD_LEN: f64 = 1e-12;

    let mut sum_cot = 0.0;

    // Each edge has exactly 2 adjacent faces in a manifold mesh
    for face in mesh.faces(edge_ij) {
        // Find the 3rd vertex (v_k) that is OPPOSITE to the edge (i, j)
        let v_k = mesh.vertices(face).find(|&v| v != v_i && v != v_j);

        if let Some(v_k) = v_k {
            let p_k = mesh.position(v_k);

            // Vectors from p_k to the edge vertices
            let u = p_i - p_k;
            let v = p_j - p_k;

            // Cot(theta) = cos(theta) / sin(theta)
            // dot(u, v) = |u||v| cos(theta)
            // |cross(u, v)| = |u||v| sin(theta)
            // => dot / |cross| = cot(theta)
            let cross_len = u.cross(&v).magnitude();

            // Handle degenerate triangles (area ~ 0) safely
            if cross_len > MIN_CROSSPROD_LEN {
                let cot = u.dot(&v) / cross_len;
                // Clamp to range [-MAX, MAX]
                sum_cot += cot.clamp(-MAX_COTAN, MAX_COTAN);
            }
        }
    }

    sum_cot
}

/// Generates the triplet form (row, col, value) of the Laplacian Matrix L.
/// L_ij = cot_weight (if adjacent)
/// L_ii = -sum(cot_weights)
/// See Eq (1).
fn get_laplacian_triplets(
    mesh: &Mesh<CONTRACTION>,
    state: &ContractionState,
) -> Vec<(usize, usize, f64)> {
    let num_verts = state.vert_ids.len();
    // Estimate capacity: ~6 neighbors per vertex + diagonal = 7 * N
    let mut triplets = Vec::with_capacity(num_verts * 7);

    for (i, &v_i) in state.vert_ids.iter().enumerate() {
        let mut sum_weights = 0.0;

        // Iterate over 1-ring neighbors
        for v_j in mesh.neighbors(v_i) {
            let j = state.vert_to_idx[&v_j];

            // Compute cotangent weight for edge (i, j)
            let weight = cotangent_weight(mesh, v_i, v_j);

            // Only add if the weight is valid (filters severe degeneracy)
            if weight.is_finite() {
                // Off-diagonal L_ij = weight
                triplets.push((i, j, weight));
                sum_weights += weight;
            } else {
                error!("Non-finite cotangent weight for edge ({:?}, {:?})", v_i, v_j);
            }
        }

        // Diagonal L_ii = -sum(weights)
        triplets.push((i, i, -sum_weights));
    }

    triplets
}
