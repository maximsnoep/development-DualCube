use std::collections::HashMap;

use faer::{
    Mat, prelude::SolveLstsq, sparse::{
        SparseColMat, Triplet, linalg::solvers::{Qr, SymbolicQr}
    }
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
    /// Used for O(1) index lookups during matrix assembly.
    pub vert_to_idx: HashMap<VertKey<CONTRACTION>, usize>,

    /// Global contraction weight scalar (s_L * W_L).
    /// Starts as 10^-3 * sqrt(AvgArea). Increases by s_L each step.
    pub wl: f64,

    /// Per-vertex attraction weights (W_H).
    /// Starts as 1.0. Updates based on ratio of current area to original area.
    pub wh: HashMap<VertKey<CONTRACTION>, f64>,

    /// Original one-ring area of each vertex (A^0).
    /// Used to calculate the update for W_H.
    pub original_areas: HashMap<VertKey<CONTRACTION>, f64>,

    /// Original mesh volume. Used for stopping condition.
    pub original_volume: f64,

    /// Whether the mesh cannot be optimized further due to instability.
    pub unstable: bool,

    /// Cache symbolic factorization since topology is static in this step.
    pub symbolic: Option<SymbolicQr<usize>>,
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
    /// - `wl` as 10^-3 * sqrt(average face area)
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
        for &v in &vert_ids {
            let area = Self::one_ring_area(mesh, v);
            original_areas.insert(v, area);
            wh.insert(v, 1.0);
        }

        // Initial wl = 10^-3 * sqrt(avg_area)
        let mut total_area = 0.0;
        let face_ids = mesh.face_ids();
        let num_faces = face_ids.len();
        for f in face_ids {
            total_area += mesh.size(f);
        }
        let avg_area = total_area / num_faces as f64;
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

// TODO: better stability

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
/// 1. Construct overdetermined stacked system M V' = B
///    where M = [ W_L * L ] and B = [   0   ]
///              [   W_H   ]         [ W_H V ]
/// 2. Solve for new positions V' in the least-squares sense using Sparse QR
/// 3. Update weights W_L and W_H
fn contract_once(mesh: &mut Mesh<CONTRACTION>, state: &mut ContractionState) {
    let num_verts = state.vert_ids.len();

    // Setup Laplacian L triplets for sparse matrix assembly
    let l_triplets = get_laplacian_triplets(mesh, state);

    // Assemble System Matrix M (2N x N)
    let mut triplets_m = Vec::with_capacity(l_triplets.len() + num_verts);

    // Top block: W_L * L
    for (r, c, val) in l_triplets {
        triplets_m.push(Triplet::new(r, c, state.wl * val));
    }

    // Bottom block: W_H (Diagonal)
    for (i, &v) in state.vert_ids.iter().enumerate() {
        let wh = state.wh[&v];
        triplets_m.push(Triplet::new(num_verts + i, i, wh));
    }

    // Create Sparse Matrix M
    let mat_m = SparseColMat::try_new_from_triplets(2 * num_verts, num_verts, &triplets_m)
        .expect("Failed to create sparse matrix from triplets");

    // Assemble RHS Matrix B (2N x 3)
    let mat_b = Mat::from_fn(2 * num_verts, 3, |r, c| {
        if r < num_verts {
            0.0
        } else {
            let i = r - num_verts;
            let v = state.vert_ids[i];
            let wh = state.wh[&v];
            let pos = mesh.position(v);
            match c {
                0 => wh * pos.x,
                1 => wh * pos.y,
                2 => wh * pos.z,
                _ => unreachable!(),
            }
        }
    });

    // Solve M X = B in the least-squares sense
    if state.symbolic.is_none() {
        match SymbolicQr::try_new(mat_m.symbolic()) {
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

    let qr = match Qr::try_new_with_symbolic(symbolic, mat_m.as_ref()) {
        Ok(factorization) => factorization,
        Err(error) => {
            warn!("Numeric factorization failed: {}", error);
            state.unstable = true;
            return;
        }
    };
    let solution = qr.solve_lstsq(mat_b.as_ref());

    // Update vertex positions from solution
    for (i, &v) in state.vert_ids.iter().enumerate() {
        let new_pos =
            mehsh::prelude::Vector3D::new(solution[(i, 0)], solution[(i, 1)], solution[(i, 2)]);
        mesh.set_position(v, new_pos);
    }

    // Update Weights for next iteration
    const SL: f64 = 4.0; // 2.0 in the paper // TODO: find out why 10 iterations is enough in the paper but we need 30ish sometimes
    state.wl *= SL;
    const MIN_VERTEX_AREA: f64 = 1e-16;

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
/// Returns 0.0 when the edge does not exist.
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

    // Cap crossproduct length to avoid division by zero
    const MIN_CROSSPROD_LEN: f64 = 1e-8;

    let mut sum_cot = 0.0;
    // Each edge has exactly 2 adjacent faces in a watertight 2-manifold mesh
    for face in mesh.faces(edge_ij) {
        // Find the third vertex making up the face
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
                sum_cot += cot
            }
        } else {
            unreachable!(
                "Expected to find a third vertex for face {:?} adjacent to edge ({:?}, {:?})",
                face, v_i, v_j
            );
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
            if weight.is_finite() {
                // Off-diagonal L_ij = weight
                triplets.push((i, j, weight));
                sum_weights += weight;
            } else {
                error!(
                    "Non-finite cotangent weight for edge ({:?}, {:?})",
                    v_i, v_j
                );
            }
        }

        // Diagonal L_ii = -sum(weights)
        triplets.push((i, i, -sum_weights));
    }

    triplets
}
