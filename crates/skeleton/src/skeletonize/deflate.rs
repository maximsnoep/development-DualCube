use std::collections::HashMap;

use mehsh::prelude::{
    HasFaces, HasNeighbors, HasPosition, HasVertices, Mesh, SetPosition, Tag, Vector3D, VertKey,
};
use serde::{Deserialize, Serialize};
use skeleton_numerics::contraction::{ContractionSolver, Step, Triangles};
use tracing::{info, warn};

/// Tag type for contraction-view of skeleton.
#[derive(Debug, Clone, Default, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CONTRACTION;

/// Epsilon value for volume-based stopping condition.
pub const VOLUME_EPSILON: f64 = 1e-6;

/// Holds the persistent state required across contraction iterations.
/// Positions live in a flat array during contraction; the mesh is written once at the end.
struct ContractionState {
    /// Stable vertex ordering; indexes `positions` and the topology.
    pub vert_ids: Vec<VertKey<CONTRACTION>>,

    /// Static topology, in the same order as the mesh iterators.
    triangles: Triangles,

    /// Current vertex positions.
    positions: Vec<Vector3D>,

    /// Global contraction weight scalar (`s_L` * `W_L`).
    /// Starts as 10^-3 * sqrt(AvgArea). Increases by `s_L` each step.
    pub wl: f64,

    /// Per-vertex attraction weights (`W_H`).
    /// Starts as 1.0. Updates based on ratio of current area to original area.
    pub wh: Vec<f64>,

    /// Original one-ring area of each vertex (A^0).
    /// Used to calculate the update for `W_H`.
    pub original_areas: Vec<f64>,

    /// Original mesh volume. Used for stopping condition.
    pub original_volume: f64,

    /// Whether the mesh cannot be optimized further due to instability.
    pub unstable: bool,

    /// Caches the symbolic factorization since topology is static in this step.
    solver: ContractionSolver,
}

impl ContractionState {
    // Cache topology once; contraction only changes positions.
    pub fn new(mesh: &Mesh<CONTRACTION>) -> Self {
        let vert_ids = mesh.vert_ids();
        let n = vert_ids.len();

        // Build reverse lookup for efficient index queries
        let vert_to_idx: HashMap<VertKey<CONTRACTION>, usize> = vert_ids
            .iter()
            .enumerate()
            .map(|(idx, &v)| (v, idx))
            .collect();

        let face_ids = mesh.face_ids();
        let face_indices: HashMap<_, _> =
            face_ids.iter().enumerate().map(|(i, &f)| (f, i)).collect();
        let faces = face_ids
            .iter()
            .map(|&f| {
                let corners: Vec<_> = mesh.vertices(f).map(|v| vert_to_idx[&v]).collect();
                [corners[0], corners[1], corners[2]]
            })
            .collect();
        let vertex_faces = vert_ids
            .iter()
            .map(|&v| mesh.faces(v).map(|f| face_indices[&f]).collect())
            .collect();
        let neighbors = vert_ids
            .iter()
            .map(|&a| {
                mesh.neighbors(a)
                    .map(|b| {
                        let (edge, _) = mesh.edge_between_verts(a, b).unwrap();
                        let opposite: Vec<_> = mesh
                            .faces(edge)
                            .map(|face| mesh.vertices(face).find(|&v| v != a && v != b).unwrap())
                            .map(|v| vert_to_idx[&v])
                            .collect();
                        (vert_to_idx[&b], [opposite[0], opposite[1]])
                    })
                    .collect()
            })
            .collect();
        let triangles = Triangles {
            faces,
            vertex_faces,
            neighbors,
        };
        let positions: Vec<_> = vert_ids.iter().map(|&v| mesh.position(v)).collect();

        // Initial wl = 10^-3 * sqrt(avg_area)
        let face_areas = triangles.face_areas(&positions);
        let total_area = face_areas.iter().fold(0.0, |total, area| total + area);
        let avg_area = total_area / face_areas.len() as f64;
        let wl = 1e-3 * avg_area.sqrt();

        Self {
            original_areas: triangles.one_ring_areas(&positions),
            original_volume: triangles.volume(&positions),
            vert_ids,
            triangles,
            positions,
            wl,
            wh: vec![1.; n],
            unstable: false,
            solver: ContractionSolver::default(),
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
/// - The last finite contracted mesh; numerical stalls stop contraction.
pub fn run<M: Tag>(
    mesh: &Mesh<M>,
    max_iterations: usize,
) -> crate::SkeletonResult<Mesh<CONTRACTION>> {
    let start = std::time::Instant::now();
    crate::validate::input_genus(mesh)?;

    // Make a copy of the mesh to use for contraction
    let mut contraction_mesh: Mesh<CONTRACTION> = Mesh::<CONTRACTION>::convert(mesh);

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

        let current_volume = state.triangles.volume(&state.positions);
        let vol_ratio = current_volume / state.original_volume;

        info!(
            "Geometry contraction: Volume ratio = {:.8} (elapsed {:.2}s)",
            vol_ratio,
            start.elapsed().as_secs_f64()
        );

        if vol_ratio < VOLUME_EPSILON {
            info!(
                "Geometry contraction: converged (elapsed {:.2}s)",
                start.elapsed().as_secs_f64()
            );
            break;
        }

        contract_once(&mut state)?;
    }

    for (&v, &position) in state.vert_ids.iter().zip(&state.positions) {
        contraction_mesh.set_position(v, position);
    }
    Ok(contraction_mesh)
}

/// Solve the stacked least-squares system, then update positions and weights.
fn contract_once(state: &mut ContractionState) -> crate::SkeletonResult<()> {
    let step = state
        .solver
        .step(&state.triangles, state.wl, &state.wh, &mut state.positions)?;
    let problem = match step {
        Step::Solved => None,
        Step::NonFiniteLaplacian => Some("Non-finite Laplacian"),
        Step::NonFiniteWeights => Some("Non-finite contraction weights"),
        Step::NonFiniteRightHandSide => Some("Non-finite contraction right-hand side"),
        Step::NonFiniteSolution => Some("Contraction produced non-finite positions"),
    };
    if let Some(problem) = problem {
        warn!("{problem}; keeping the previous iterate");
        state.unstable = true;
        return Ok(());
    }

    // Update Weights for next iteration
    const SL: f64 = 4.0; // 2.0 in the paper // TODO: find out why 10 iterations is enough in the paper but we need 30ish sometimes
    state.wl *= SL;
    const MIN_VERTEX_AREA: f64 = 1e-16;

    for (i, current_area) in state
        .triangles
        .one_ring_areas(&state.positions)
        .into_iter()
        .enumerate()
    {
        let safe_area = current_area.max(MIN_VERTEX_AREA);
        // Attraction weight relative to the original area, with initial weight 1.
        let original_area = state.original_areas[i];
        state.wh[i] = (original_area / safe_area).sqrt();
    }
    Ok(())
}

#[cfg(test)]
mod tests;
