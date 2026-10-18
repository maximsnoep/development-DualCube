//! Coupled optimization of the loops and the layout (see `Solution::optimize`): cycles of
//!
//! 1. some steps of the simulated annealing of the loops (`Solution::anneal`), from the best solution so far (every
//!    cycle starts hot again);
//! 2. some generations of the optimization of the layout (`Solution::optimize_layout`) of the result;
//! 3. every loop moved to the middle of the patches of that layout (see `Solution::medial_loops`), with the layout
//!    embedded again on the same corners;
//!
//! and the result replaces the best solution if it is better. The loops are thus compared after their layout was
//! optimized (a new loop structure starts with a default layout, which is rarely better than an optimized one). If the
//! loops did not change, the layout of the best solution is optimized further. All phases report to one monitor (one
//! history, the statistics of all mutations, and the adaptive choice of the mutations per kind).

use crate::prelude::*;
use serde::{Deserialize, Serialize};

/// Parameters of the coupled optimization.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CoupledParams {
    /// The annealing of the loops (its steps per cycle, 0 skips it).
    pub loops: AnnealParams,
    pub layout: LayoutEvolutionParams,
    /// Generations of the layout per cycle (0 skips it).
    pub layout_generations: usize,
    /// Maximum number of cycles.
    pub max_cycles: usize,
    /// Stop after this many cycles in a row without a new best solution (0: never). A cycle is one annealing of the
    /// loops and one optimization of the layout (`loops.steps` and `layout_generations` generations).
    #[serde(default = "default_patience")]
    pub patience: usize,
}

const fn default_patience() -> usize {
    5
}

impl Default for CoupledParams {
    fn default() -> Self {
        Self {
            loops: AnnealParams::default(),
            layout: LayoutEvolutionParams::default(),
            // (16 children per generation.)
            layout_generations: 300,
            max_cycles: usize::MAX,
            patience: default_patience(),
        }
    }
}

impl Solution {
    /// The coupled optimization of the loops and the layout (see the module documentation), reporting to (and
    /// stoppable through) the given monitor. Never worse than the input.
    pub fn optimize(
        &self,
        params: &CoupledParams,
        monitor: &OptimizationMonitor,
    ) -> Result<Self, SolutionError> {
        let result = self.optimize_inner(params, monitor);
        monitor.finish();
        result
    }

    fn optimize_inner(
        &self,
        params: &CoupledParams,
        monitor: &OptimizationMonitor,
    ) -> Result<Self, SolutionError> {
        // The loops of the input in the middle of its patches too (as those of every result; the layout, and so the
        // quality, stay the same), such that a result is never one with the loops as they were added.
        let mut best = self.medial_loops().unwrap_or_else(|| self.clone());
        let mut best_quality = best.get_quality().ok_or(SolutionError::NoPrimal)?;
        monitor.note_best(best_quality);
        // Cycles in a row without a new best solution.
        let mut stale = 0;
        for cycle in 1..=params.max_cycles {
            if monitor.stopping() {
                break;
            }
            let (steps, layout_generations, patience) = monitor.settings().map_or(
                (params.loops.steps, params.layout_generations, params.patience),
                |settings| (settings.steps, settings.layout_generations, settings.patience),
            );
            if patience > 0 && stale >= patience {
                info!("optimize: stopped after {stale} cycles without a new best solution");
                break;
            }
            if steps == 0 && layout_generations == 0 {
                break;
            }

            // The loops.
            let timer = std::time::Instant::now();
            let mut candidate = best.clone();
            if steps > 0 {
                let loops = AnnealParams {
                    steps,
                    ..params.loops
                };
                if let Ok(annealed) = best.anneal_phase(&loops, monitor)
                    && annealed.loop_signature() != best.loop_signature()
                {
                    candidate = annealed;
                }
            }
            let loops_time = timer.elapsed();
            let timer = std::time::Instant::now();

            // The layout (of the new loops, or further of the best solution).
            if layout_generations > 0 && !monitor.stopping() {
                let layout = LayoutEvolutionParams {
                    max_generations: layout_generations,
                    patience: usize::MAX,
                    ..params.layout
                };
                monitor.set_layout_active(true);
                if let Ok(optimized) = candidate.optimize_layout_phase(&layout, monitor) {
                    candidate = optimized;
                }
                monitor.set_layout_active(false);
            }

            let layout_time = timer.elapsed();
            let timer = std::time::Instant::now();

            // The loops through the middle of the patches of the layout (also when stopping: it is quick, and every
            // result has its loops in the middle of its patches).
            match candidate.medial_loops() {
                Some(medial) => candidate = medial,
                None => warn!("optimize: cycle {cycle}: no medial loops"),
            }
            let medial_time = timer.elapsed();
            let quality = candidate.get_quality().unwrap_or(f64::NEG_INFINITY);
            info!(
                "optimize: cycle {cycle}: {} loops, quality {quality} (best {best_quality}); loops {loops_time:?}, layout {layout_time:?}, medial loops {medial_time:?}",
                candidate.loops.len()
            );
            if quality > best_quality {
                monitor.show_result(quality, || candidate.clone());
                best = candidate;
                best_quality = quality;
                stale = 0;
            } else {
                stale += 1;
            }
        }
        Ok(best)
    }
}
