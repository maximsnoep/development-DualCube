//! The solution pipeline: loops → dual → corners → layout → polycube → quad.
//!
//! Each stage clones the solution, applies one operation, and reports back as
//! a [`JobResult::StageCompleted`]; [`Stage::next_job`] then decides whether
//! the pipeline continues or stops with a refresh of the renders.

use super::{Job, JobResult};
use crate::render;
use crate::resources::Configuration;
use bevy::prelude::*;
use dualcube::prelude::*;
#[cfg(feature = "hex")]
use dualcube_hex::HexExt;
use mehsh::prelude::VertKey;

/// A completed stage of the pipeline.
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub(super) enum Stage {
    Graph,
    Loops,
    Dual,
    Corners,
    Layout,
    Polycube,
    Quad,
    Hex,
}

impl Stage {
    /// The job that follows this stage (the pipeline runs to the end, then refreshes the renders).
    pub(super) fn next_job(self, solution: Solution, configuration: Configuration) -> Job {
        match self {
            Self::Graph => Job::refresh(solution, configuration),
            Self::Loops => Job::compute_dual(solution, configuration),
            Self::Dual => Job::place_corners(solution, configuration),
            Self::Corners => Job::place_paths(solution, configuration),
            Self::Layout => Job::compute_polycube(solution, configuration),
            // The quad mesh (and the polycube map) complete the pipeline.
            Self::Polycube => Job::compute_quad(solution, configuration),
            Self::Quad => Job::refresh(solution, configuration),
            Self::Hex => Job::refresh(solution, configuration),
        }
    }
}

/// Clones the solution and applies `operation` to it.
/// Logs a warning and returns `None` if the operation fails.
fn try_step<E: std::fmt::Debug>(
    solution: &Solution,
    operation: &str,
    f: impl FnOnce(&mut Solution) -> Result<(), E>,
) -> Option<Solution> {
    let mut modified = solution.clone();
    match f(&mut modified) {
        Ok(()) => Some(modified),
        Err(err) => {
            warn!("Failed to {operation}: {err:?}");
            None
        }
    }
}

// A new solution from scratch (see `Solution::initialize`); only the flow fields and graphs are kept (they only depend
// on the mesh).
fn initialized(solution: &Solution, configuration: &Configuration) -> Solution {
    let mut initialized = Solution::new(solution.mesh_ref.clone());
    initialized.fields = solution.fields.clone();
    initialized.flow_graphs = solution.flow_graphs.clone();
    initialized.quality = configuration.quality;
    initialized.initialize();
    initialized.smooth_loops();
    initialized
}

fn completed(stage: Stage, solution: Solution) -> Option<JobResult> {
    Some(JobResult::StageCompleted { stage, solution })
}

#[allow(dead_code)]
impl Job {
    pub fn initialize_loops(solution: Solution, configuration: Configuration) -> Self {
        Self::new("initializing loops", move || {
            completed(Stage::Loops, initialized(&solution, &configuration))
        })
        // A new start: replaces a running optimization.
        .preempting()
    }

    /// Optimize the loops and the layout (see `Solution::optimize`), reporting to the monitor (see `OptimizationLive`);
    /// supervised: it runs until stopped. A solution without a layout is initialized first.
    pub fn optimize(
        solution: Solution,
        configuration: Configuration,
        monitor: OptimizationMonitor,
    ) -> Self {
        Self::new("optimizing", move || {
            let mut solution = if solution.layout.is_none() || solution.loops.len() < 3 {
                initialized(&solution, &configuration)
            } else {
                solution.clone()
            };
            if solution.layout.is_none() {
                warn!("Failed to optimize the solution: the initialization failed");
                monitor.finish();
                return completed(Stage::Loops, solution);
            }
            solution.quality = configuration.quality;
            let params = CoupledParams {
                loops: configuration.anneal,
                layout: configuration.layout_evolution,
                layout_generations: configuration.layout_generations,
                max_cycles: usize::MAX,
                patience: configuration.patience,
            };
            match solution.optimize(&params, &monitor) {
                // The result is embedded (dual, layout, polycube): only the polycube and quad mesh follow.
                // (With smooth loops, as the best solution shown while optimizing, see `poll_optimization`.)
                Ok(optimized) => completed(Stage::Layout, optimized.with_smooth_loops()),
                Err(e) => {
                    warn!("Failed to optimize the solution. {e}");
                    None
                }
            }
        })
    }

    /// A solution without loops (and so without layout, polycube and quad mesh): only the input mesh, with its flow
    /// fields and graphs (they only depend on the mesh).
    pub fn clear(solution: Solution, configuration: Configuration) -> Self {
        Self::new("clearing the solution", move || {
            let mut cleared = Solution::new(solution.mesh_ref.clone());
            cleared.fields = solution.fields.clone();
            cleared.flow_graphs = solution.flow_graphs.clone();
            cleared.quality = configuration.quality;
            completed(Stage::Graph, cleared)
        })
    }

    pub fn prepare_flow(solution: Solution, _configuration: Configuration) -> Self {
        Self::new("computing flow fields and graphs", move || {
            let mut solution = solution.clone();
            solution.prepare_flow();
            completed(Stage::Graph, solution)
        })
    }

    pub fn compute_dual(solution: Solution, configuration: Configuration) -> Self {
        Self::new("computing dual", move || {
            match try_step(&solution, "construct dual and polycube", |s| {
                s.construct_dual_and_polycube()
            }) {
                Some(modified) => completed(Stage::Dual, modified),
                // On failure still refresh, so the user sees the (unchanged) solution.
                None => Some(JobResult::Refreshed(render::refresh(
                    &solution,
                    &configuration,
                ))),
            }
        })
    }

    pub fn place_corners(solution: Solution, _configuration: Configuration) -> Self {
        Self::new("placing corners", move || {
            let modified = try_step(&solution, "place corners and paths", |s| s.place_corners())?;
            completed(Stage::Corners, modified)
        })
    }

    pub fn move_corner(
        solution: Solution,
        _configuration: Configuration,
        corner: VertKey<POLYCUBE>,
        new_vertex: VertID,
    ) -> Self {
        Self::new("moving corner", move || {
            let modified = try_step(&solution, "move corner", |s| {
                s.move_corner_to(corner, new_vertex)
            })?;
            completed(Stage::Layout, modified)
        })
    }

    pub fn place_paths(solution: Solution, _configuration: Configuration) -> Self {
        Self::new("placing paths", move || {
            let modified = try_step(&solution, "place paths", |s| s.place_paths())?;
            completed(Stage::Layout, modified)
        })
    }

    /// Post-processing: smooth the paths of the layout (see `Solution::smooth_layout`).
    pub fn smooth_layout(solution: Solution, _configuration: Configuration) -> Self {
        Self::new("smoothing paths", move || {
            let modified = try_step(&solution, "smooth paths", |s| s.smooth_layout())?;
            completed(Stage::Layout, modified)
        })
    }

    pub fn compute_polycube(solution: Solution, _configuration: Configuration) -> Self {
        Self::new("computing polycube", move || {
            let modified = try_step(&solution, "resize polycube", |s| {
                s.resize_polycube(false)
            })?;
            completed(Stage::Polycube, modified)
        })
    }

    /// The quad mesh, and the polycube map (the mesh mapped onto the polycube).
    pub fn compute_quad(solution: Solution, configuration: Configuration) -> Self {
        Self::new("computing quad mesh", move || {
            let modified = try_step(&solution, "construct quad", |s| {
                s.construct_quad(QuadDensity::Fixed(configuration.omega))
            })?;
            completed(Stage::Quad, modified)
        })
    }

    #[allow(unused_variables)]
    pub fn compute_hex(solution: Solution, configuration: Configuration) -> Self {
        #[cfg(feature = "hex")]
        {
            Self::new("computing hex", move || {
                let modified = try_step(&solution, "construct hex", |s| s.construct_hex())?;
                completed(Stage::Hex, modified)
            })
        }
        #[cfg(not(feature = "hex"))]
        {
            Self::new("computing hex", move || {
                warn!(
                    "Hex computation is disabled. Rebuild gui with `--features hex` to enable it."
                );
                None
            })
        }
    }

    pub fn refresh(solution: Solution, configuration: Configuration) -> Self {
        Self::new("refreshing", move || {
            Some(JobResult::Refreshed(render::refresh(
                &solution,
                &configuration,
            )))
        })
    }
}
