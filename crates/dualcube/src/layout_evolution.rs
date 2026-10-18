//! Evolutionary optimization of the layout (the primal): the positions of the corners and the paths between them,
//! for a fixed loop structure and polycube (the combinatorics of the layout never change).
//!
//! One layout is improved with the mutations (each re-places only the paths it affects, see
//! `Layout::move_corner`, unless noted otherwise). Mutations keep the layout valid, but are bounded by its patches, not
//! by the loop regions (see `Layout::begin_mutation`): a corner may move anywhere inside the patches around it, a path
//! anywhere inside its two patches. The loops follow the layout afterwards (see `Solution::medial_loops`).
//!
//! - `corner`: move a corner to a candidate position inside the patches around it (near the intersection of its fitted zone
//!   planes, near the position aligned with its neighbors, or a random nearby vertex; see `Layout::corner_candidates`),
//! - `path`: remove one path and compute it again, with all other paths in place (see `Layout::reroute_path`).
//!
//! Every generation, `offspring` children of the layout are made (in parallel), and the best replaces it if it is better:
//! a (1+lambda) evolution strategy (in tests, larger populations were no better, as their layouts hardly differ). The
//! quality is that of the solution (`Solution::get_quality`). The result is never worse than the input. Smoothing the paths is a separate
//! (visual) post-processing step (see `Solution::smooth_layout`).

use crate::prelude::*;
use rand::seq::IteratorRandom;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Parameters of the layout evolution.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayoutEvolutionParams {
    /// Number of offspring per generation (lambda).
    pub offspring: usize,
    /// Maximum number of generations.
    pub max_generations: usize,
    /// Stop after this many generations without improvement (tau).
    pub patience: usize,
    /// Relative frequencies of the mutations (0 disables a mutation).
    pub corner: f64,
    pub path: f64,
}

impl Default for LayoutEvolutionParams {
    fn default() -> Self {
        Self {
            // One child per thread (of 16). In tests on 13 models (same number of offspring), one layout (1+8) was as
            // good as or better than populations (2+8, 4+16, 8+32).
            offspring: 16,
            max_generations: 120,
            patience: 20,
            corner: 0.4,
            path: 0.15,
        }
    }
}

impl LayoutEvolutionParams {
    // The prior weights of the mutations (the fixed probabilities).
    fn mutation_prior(&self) -> Vec<(&'static str, f64)> {
        vec![("corner", self.corner), ("path", self.path)]
    }

    /// These parameters without the given mutations (by their names in the statistics).
    #[must_use]
    pub fn without(mut self, disabled: &std::collections::BTreeSet<String>) -> Self {
        if disabled.contains("corner") {
            self.corner = 0.;
        }
        if disabled.contains("path") {
            self.path = 0.;
        }
        self
    }

    // The parameters (without the mutations switched off while running, if any) and the number of offspring.
    fn live(&self, monitor: &OptimizationMonitor) -> (Self, usize) {
        let params = monitor
            .settings()
            .map_or(*self, |settings| self.without(&settings.layout_disabled));
        (params, params.offspring.max(1))
    }
}

impl Solution {
    /// Optimize the layout (see the module documentation), reporting to (and stoppable through) the given monitor.
    pub fn optimize_layout(
        &self,
        params: &LayoutEvolutionParams,
        monitor: &OptimizationMonitor,
    ) -> Result<Self, SolutionError> {
        let result = self.optimize_layout_phase(params, monitor);
        monitor.finish();
        result
    }

    // See `optimize_layout` (without finishing the monitor).
    pub(crate) fn optimize_layout_phase(
        &self,
        params: &LayoutEvolutionParams,
        monitor: &OptimizationMonitor,
    ) -> Result<Self, SolutionError> {
        let started_at = Instant::now();
        let (_, lambda) = params.live(monitor);
        let input_quality = self.get_quality().ok_or(SolutionError::NoPrimal)?;
        let mut seed = self.layout.clone().ok_or(SolutionError::NoPrimal)?;
        // Corner moves need paths restricted to loop regions (which smoothed paths may not be): re-embed if needed.
        if !(seed.is_complete() && seed.regions.is_some()) {
            seed.place_all_corners();
            seed.place_paths_best(LayoutParams::default().candidates)?;
        }

        // The score of a layout: the quality of the solution with that layout.
        let weights = self.quality.weights;
        let terms = QualityTerms::needed(&weights);
        let (loop_count, quality_params) = (self.loops.len(), self.quality);
        let score = |layout: &Layout| -> Option<f64> {
            QualityReport::compute(loop_count, layout, &quality_params, terms).score(&weights)
        };

        let mut seed_quality = score(&seed).ok_or(SolutionError::NoPrimal)?;
        // Every path computed again once (the worst first, by their penalties), kept where the score improves: when a
        // layout is built, every path is placed before the paths after it, and is then often far from where the score
        // wants it (on CAD models, computing a path again in the final layout improves the score for most paths).
        let penalties = element_penalties(&seed, &weights);
        let structure = &seed.polycube_ref.structure;
        let mut edges = structure
            .edge_ids()
            .into_iter()
            .filter(|&edge| edge < structure.twin(edge))
            .map(|edge| (edge, penalties.paths.get(&edge).copied().unwrap_or(0.)))
            .collect_vec();
        edges.sort_by(|(_, a), (_, b)| b.total_cmp(a));
        let replaced_at = Instant::now();
        let mut replaced = 0;
        for (edge, _) in edges {
            let mut child = seed.clone();
            child.begin_mutation();
            let placed = child.reroute_path(edge).is_ok();
            child.end_mutation();
            if placed
                && child.is_complete()
                && let Some(quality) = score(&child)
                && quality > seed_quality
            {
                seed = child;
                seed_quality = quality;
                replaced += 1;
            }
        }
        info!(
            "optimize_layout: {replaced} paths computed again (quality {seed_quality}) in {:?}",
            replaced_at.elapsed()
        );
        let corners = seed.vert_to_corner.len();
        let mut current = (Member::new(seed), seed_quality);
        let mut stats = monitor.mutation_stats_map();
        let mut best = seed_quality;
        let mut stale = 0;
        let mut selection = monitor.take_selection("layout");
        monitor.record(layout_stats(0, current.1, corners, started_at));
        // Show the layout of this phase (from now on).
        monitor.note_best(input_quality);
        monitor.offer_best(seed_quality, || self.with_layout(current.0.layout.clone()));
        info!("optimize_layout: start lambda={lambda} quality={seed_quality}");

        for generation in 0..params.max_generations {
            let (live, lambda) = params.live(monitor);
            let chances = selection.chances(&live.mutation_prior());
            monitor.record_chances(&selection.report(&chances));
            // Every try: the chosen mutation, and the scored child (if the mutation succeeded).
            let attempts = (0..lambda)
                .collect_vec()
                .into_par()
                .with_pool(pool())
                .filter_map(|_| {
                    let (parent, parent_quality) = &current;
                    let op = choose_mutation(&chances)?;
                    let timer = Instant::now();
                    let mutated = mutate_layout(
                        &parent.layout,
                        parent.levels(),
                        parent.penalties(&weights),
                        op,
                    );
                    monitor.record_mutation_times([(op, timer.elapsed())]);
                    let child = mutated.and_then(|(child, op)| {
                        let quality = score(&child)?;
                        Some((child, quality, op, *parent_quality))
                    });
                    Some((op, child))
                })
                .collect::<Vec<_>>();
            // Tries and improvements of the best per mutation (see `MutationSelection`); failed tries count too.
            let mut tries: HashMap<&'static str, (usize, usize)> = HashMap::default();
            for (op, _) in &attempts {
                tries.entry(op).or_default().0 += 1;
            }
            let children = attempts
                .into_iter()
                .filter_map(|(_, child)| child)
                .collect_vec();
            for (_, quality, op, parent_quality) in &children {
                let entry = stats.entry(op).or_default();
                entry[0] += 1;
                entry[1] += 1;
                if quality > parent_quality {
                    entry[2] += 1;
                }
                if *quality > best + 1e-12 {
                    entry[3] += 1;
                    tries.entry(op).or_default().1 += 1;
                }
            }
            selection.record(tries);
            // The best child replaces the layout if it is better (the others are dropped on other threads).
            let mut children = children;
            if let Some(index) = children
                .iter()
                .position_max_by(|(_, a, _, _), (_, b, _, _)| a.total_cmp(b))
            {
                let (child, quality, _, _) = children.swap_remove(index);
                if quality > current.1 + 1e-12 {
                    drop_later(std::mem::replace(&mut current, (Member::new(child), quality)));
                } else {
                    drop_later(child);
                }
            }
            drop_later(children);

            monitor.record_mutations(&stats);
            monitor.record(layout_stats(generation + 1, current.1, corners, started_at));
            if current.1 > best + 1e-12 {
                best = current.1;
                stale = 0;
                monitor.offer_best(best, || self.with_layout(current.0.layout.clone()));
            } else {
                stale += 1;
            }
            info!(
                "optimize_layout: generation {} best={best} ({:?})",
                generation + 1,
                started_at.elapsed()
            );
            if stale >= params.patience || monitor.stopping() {
                break;
            }
        }

        monitor.put_selection("layout", selection);
        let mut result = self.with_layout(current.0.layout.clone());
        if result.resize_polycube(false).is_err() {
            warn!("optimize_layout: resizing the polycube failed");
        }
        let output_quality = result.get_quality();
        info!(
            "optimize_layout: quality {input_quality} -> {output_quality:?} in {:?}; mutations (generated, evaluated, better than parent, new best): {stats:?}",
            started_at.elapsed()
        );
        // Never worse than the input.
        match output_quality {
            Some(quality) if quality >= input_quality => Ok(result),
            _ => Ok(self.clone()),
        }
    }

    /// Post-processing of the layout (visual): smooth (straighten) its paths. Not part of the optimization (the smoothed
    /// paths may leave the loop regions that its mutations need).
    pub fn smooth_layout(&mut self) -> Result<(), SolutionError> {
        let layout = self.layout.as_mut().ok_or(SolutionError::NoPrimal)?;
        layout.straighten_paths()?;
        // The quad mesh belongs to the old paths.
        self.quad = None;
        Ok(())
    }

    // This solution with the given layout.
    fn with_layout(&self, layout: Layout) -> Self {
        let mut solution = self.clone();
        solution.layout = Some(layout);
        // The quad mesh belongs to the old layout (its elements refer to the old refined mesh).
        solution.quad = None;
        solution
    }
}

// The given mutation of a layout (see the module documentation); `None` if it does not apply or fails.
fn mutate_layout(
    parent: &Layout,
    levels: &HashMap<ZoneID, f64>,
    penalties: &ElementPenalties,
    op: &'static str,
) -> Option<(Layout, &'static str)> {
    let mut child = parent.clone();
    // Bounded by the patches, not by the loop regions (see `Layout::begin_mutation`).
    child.begin_mutation();
    let structure = &parent.polycube_ref.structure;
    // Corners and paths are chosen by their penalties (see `pick_weighted`).
    let random_corner = || {
        pick_weighted(parent.vert_to_corner.left_values().map(|&corner| {
            (
                corner,
                penalties.corners.get(&corner).copied().unwrap_or(0.),
            )
        }))
    };
    let random_edge = || {
        pick_weighted(
            structure
                .edge_ids()
                .into_iter()
                .map(|edge| (edge, penalties.paths.get(&edge).copied().unwrap_or(0.))),
        )
    };
    // Move a corner to one of its candidate positions (near its targets).
    let move_corner = |layout: &mut Layout, corner: VertKey<POLYCUBE>| {
        let targets = layout.corner_targets(corner, levels);
        let candidate = layout
            .corner_candidates(corner, &targets, 2, 3)
            .into_iter()
            .choose(&mut rand::rng())?;
        layout.move_corner(corner, candidate).ok()
    };
    match op {
        "corner" => move_corner(&mut child, random_corner()?)?,
        _ => child.reroute_path(random_edge()?).ok()?,
    }
    child.end_mutation();
    child.is_complete().then_some((child, op))
}

// A layout, with the levels of its zones (computed once, when it is first mutated).
struct Member {
    layout: Layout,
    levels: std::sync::OnceLock<HashMap<ZoneID, f64>>,
    // The penalty of every corner and path (see `element_penalties`), computed once, when it is first mutated: the
    // mutations target the elements that need it.
    penalties: std::sync::OnceLock<ElementPenalties>,
}

impl Member {
    fn new(layout: Layout) -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self {
            layout,
            levels: std::sync::OnceLock::new(),
            penalties: std::sync::OnceLock::new(),
        })
    }

    fn levels(&self) -> &HashMap<ZoneID, f64> {
        self.levels.get_or_init(|| self.layout.fit_zone_levels())
    }

    fn penalties(&self, weights: &QualityWeights) -> &ElementPenalties {
        self.penalties
            .get_or_init(|| element_penalties(&self.layout, weights))
    }
}

fn layout_stats(generation: usize, quality: f64, corners: usize, started_at: Instant) -> GenerationStats {
    GenerationStats {
        generation,
        best: quality,
        current: quality,
        loops: corners,
        layout: true,
        seconds: started_at.elapsed().as_secs_f64(),
        temperature: None,
        chains: [None; SHOWN_CHAINS],
    }
}
