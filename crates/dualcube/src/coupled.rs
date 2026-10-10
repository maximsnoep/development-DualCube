//! Coupled optimization of the loops and the layout (see `Solution::optimize`): cycles of
//!
//! 1. some generations of the evolution of the loops (`Solution::evolve`, in the chosen phase, see `LoopPhase`), from
//!    the best solution so far;
//! 2. some generations of the evolution of the layout (`Solution::optimize_layout`) of the result;
//! 3. every loop moved to the middle of the patches of that layout (see `Solution::medial_loops`), with the layout
//!    embedded again on the same corners;
//!
//! and the result replaces the best solution if it is better. The loops are thus compared after their layout was
//! optimized (a new loop structure starts with a default layout, which is rarely better than an optimized one). If the
//! loops did not change, the layout of the best solution is optimized further. All phases report to one monitor (one
//! history, the statistics of all mutations, and the adaptive choice of the mutations per kind).
//!
//! In the initialization phase, a cycle instead evolves `INITIAL_CANDIDATES` new initial loop structures (see
//! `Solution::initial_candidates`) with additions only, for the generations of the loops each (without the layout),
//! and the best replaces the best solution if it is better.

use crate::prelude::*;
use serde::{Deserialize, Serialize};

/// Parameters of the coupled optimization.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CoupledParams {
    pub loops: EvolutionParams,
    pub layout: LayoutEvolutionParams,
    /// Generations of the loops, and of the layout, per cycle (0 skips that phase).
    pub loop_generations: usize,
    pub layout_generations: usize,
    /// Maximum number of cycles.
    pub max_cycles: usize,
}

impl Default for CoupledParams {
    fn default() -> Self {
        Self {
            loops: EvolutionParams::default(),
            layout: LayoutEvolutionParams::default(),
            loop_generations: 10,
            layout_generations: 40,
            max_cycles: usize::MAX,
        }
    }
}

// The number of new initial loop structures per cycle in the initialization phase.
const INITIAL_CANDIDATES: usize = 4;

impl Solution {
    /// The coupled optimization of the loops and the layout (see the module documentation), reporting to (and
    /// stoppable through) the given monitor. Never worse than the input.
    pub fn optimize(
        &self,
        params: &CoupledParams,
        monitor: &EvolutionMonitor,
    ) -> Result<Self, SolutionError> {
        let result = self.optimize_inner(params, monitor);
        monitor.finish();
        result
    }

    fn optimize_inner(
        &self,
        params: &CoupledParams,
        monitor: &EvolutionMonitor,
    ) -> Result<Self, SolutionError> {
        let mut best = self.clone();
        let mut best_quality = best.get_quality().ok_or(SolutionError::NoPrimal)?;
        monitor.note_best(best_quality);
        for cycle in 1..=params.max_cycles {
            if monitor.stopping() {
                break;
            }
            let (loop_generations, layout_generations, phase) = match monitor.settings() {
                Some(settings) => (
                    settings.loop_generations,
                    settings.layout_generations,
                    monitor.advance_phase(settings.phase, settings.advance_after),
                ),
                None => (
                    params.loop_generations,
                    params.layout_generations,
                    params.loops.phase,
                ),
            };
            if loop_generations == 0 && layout_generations == 0 {
                break;
            }

            // Initialization: new initial structures, each with some generations of additions.
            if phase == LoopPhase::Initialization {
                monitor.set_activity(format!("cycle {cycle}: initialization"));
                let loops = EvolutionParams {
                    max_generations: loop_generations,
                    patience: usize::MAX,
                    phase,
                    ..params.loops
                };
                // (The best solution so far stays shown until a new structure is better.)
                monitor.show_now(best_quality, || best.clone());
                let candidates = best.initial_candidates(INITIAL_CANDIDATES);
                let count = candidates.len();
                for (i, candidate) in candidates.into_iter().enumerate() {
                    monitor.set_activity(format!(
                        "cycle {cycle}: initialization (structure {} of {count})",
                        i + 1
                    ));
                    let switched = monitor.settings().is_some_and(|settings| {
                        monitor.advance_phase(settings.phase, settings.advance_after)
                            != LoopPhase::Initialization
                    });
                    if monitor.stopping() || switched {
                        break;
                    }
                    let Ok(evolved) = candidate.evolve_phase(&loops, monitor) else {
                        continue;
                    };
                    let quality = evolved.get_quality().unwrap_or(f64::NEG_INFINITY);
                    info!(
                        "optimize: cycle {cycle}: initial structure with {} loops, quality {quality} (best {best_quality})",
                        evolved.loops.len()
                    );
                    if quality > best_quality {
                        best = evolved;
                        best_quality = quality;
                        monitor.note_best(best_quality);
                    }
                }
                // The next cycle continues from the best solution: show it.
                monitor.show_now(best_quality, || best.clone());
                continue;
            }

            // The loops.
            let timer = std::time::Instant::now();
            let mut candidate = best.clone();
            if loop_generations > 0 {
                let current = if monitor.pruning_state().0 {
                    LoopPhase::Pruning
                } else {
                    phase
                };
                monitor.set_activity(format!("cycle {cycle}: {}", current.name()));
                let loops = EvolutionParams {
                    max_generations: loop_generations,
                    patience: usize::MAX,
                    ..params.loops
                };
                if let Ok(evolved) = best.evolve_phase(&loops, monitor)
                    && evolved.loop_signature() != best.loop_signature()
                {
                    candidate = evolved;
                }
            }
            let loops_time = timer.elapsed();
            let timer = std::time::Instant::now();

            // The layout (of the new loops, or further of the best solution).
            if layout_generations > 0 && !monitor.stopping() {
                monitor.set_activity(format!("cycle {cycle}: layout"));
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

            // The loops through the middle of the patches of the layout.
            if !monitor.stopping() {
                monitor.set_activity(format!("cycle {cycle}: medial loops"));
                match candidate.medial_loops() {
                    Some(medial) => {
                        candidate = medial;
                        if let Some(quality) = candidate.get_quality() {
                            monitor.show_now(quality, || candidate.clone());
                        }
                    }
                    None => warn!("optimize: cycle {cycle}: no medial loops"),
                }
            }
            let medial_time = timer.elapsed();
            let quality = candidate.get_quality().unwrap_or(f64::NEG_INFINITY);
            info!(
                "optimize: cycle {cycle}: {} loops, quality {quality} (best {best_quality}); loops {loops_time:?}, layout {layout_time:?}, medial loops {medial_time:?}",
                candidate.loops.len()
            );
            if quality > best_quality {
                best = candidate;
                best_quality = quality;
                monitor.note_best(best_quality);
            } else {
                // The next cycle continues from the best solution: show it.
                monitor.show_now(best_quality, || best.clone());
            }
        }
        Ok(best)
    }
}
