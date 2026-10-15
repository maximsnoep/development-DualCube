//! Evolutionary optimization of the layout (the primal): the positions of the corners and the paths between them,
//! for a fixed loop structure and polycube (the combinatorics of the layout never change).
//!
//! A population of layouts evolves with the mutations (each re-places only the paths it affects, see
//! `Layout::move_corner`, unless noted otherwise). Mutations keep the layout valid, but are bounded by its patches, not
//! by the loop regions (see `Layout::begin_mutation`): a corner may move anywhere inside the patches around it, a path
//! anywhere inside its two patches. The loops follow the layout afterwards (see `Solution::medial_loops`).
//!
//! - `corner`: move a corner to a candidate position inside the patches around it (near the intersection of its fitted zone
//!   planes, near the position aligned with its neighbors, or a random nearby vertex; see `Layout::corner_candidates`),
//! - `path`: remove one path and compute it again, with all other paths in place (see `Layout::reroute_path`).
//!
//! Selection is as in the evolution of the loops (rank-based parents, plus-selection). The quality is that of the
//! solution (`Solution::get_quality`). The result is never worse than the input. Smoothing the paths is a separate
//! (visual) post-processing step (see `Solution::smooth_layout`).

use crate::prelude::*;
use rand::seq::IteratorRandom;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Parameters of the layout evolution.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayoutEvolutionParams {
    /// Population size (mu).
    pub population: usize,
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
            // A (1+8) strategy: in tests on 13 models (same number of offspring) it was as good as or better than
            // larger populations (2+8, 4+16, 8+32), as the layouts of a population hardly differ.
            population: 1,
            offspring: 8,
            max_generations: 240,
            patience: 40,
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

    // The parameters, population size, and number of offspring, with the changes made while running (if any).
    fn live(&self, monitor: &EvolutionMonitor) -> (Self, usize, usize) {
        let params = match monitor.settings() {
            Some(settings) => Self {
                population: settings.layout.population,
                offspring: settings.layout.offspring,
                ..self.without(&settings.layout.disabled)
            },
            None => *self,
        };
        (params, params.population.max(1), params.offspring.max(1))
    }
}

impl Solution {
    /// Optimize the layout (see the module documentation), reporting to (and stoppable through) the given monitor.
    pub fn optimize_layout(
        &self,
        params: &LayoutEvolutionParams,
        monitor: &EvolutionMonitor,
    ) -> Result<Self, SolutionError> {
        let result = self.optimize_layout_phase(params, monitor);
        monitor.finish();
        result
    }

    // See `optimize_layout` (without finishing the monitor).
    pub(crate) fn optimize_layout_phase(
        &self,
        params: &LayoutEvolutionParams,
        monitor: &EvolutionMonitor,
    ) -> Result<Self, SolutionError> {
        let started_at = Instant::now();
        let (_, mu, lambda) = params.live(monitor);
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
        let mut population = vec![(Member::new(seed), seed_quality)];
        let mut stats = monitor.mutation_stats_map();
        let mut best = seed_quality;
        let mut stale = 0;
        let mut selection = monitor.take_selection("layout");
        monitor.record(layout_stats(0, &population, corners, started_at));
        // Show the best of the population of this phase (from now on).
        monitor.note_best(input_quality);
        monitor.begin_phase(seed_quality, || {
            self.with_layout(population[0].0.layout.clone())
        });
        info!("optimize_layout: start mu={mu} lambda={lambda} quality={seed_quality}");

        for generation in 0..params.max_generations {
            let (live, mu, lambda) = params.live(monitor);
            let chances = selection.chances(&live.mutation_prior());
            monitor.record_chances(&selection.report(&chances));
            // Parents among the best (see `PARENTS`).
            let n = population.len().min(PARENTS);
            // Every try: the chosen mutation, and the scored child (if the mutation succeeded).
            let attempts = (0..lambda)
                .collect_vec()
                .into_par()
                .filter_map(|_| {
                    let (parent, parent_quality) = &population[rank_pick(n)];
                    let op = choose_mutation(&chances)?;
                    let child = mutate_layout(
                        &parent.layout,
                        parent.levels(),
                        parent.penalties(&weights),
                        op,
                    )
                    .and_then(|(child, op)| {
                        let quality = score(&child)?;
                        Some((child, quality, op, *parent_quality))
                    });
                    Some((op, child))
                })
                .collect::<Vec<_>>();
            // Tries and improvements of the best per mutation (see `MutationSelection`); failed tries count too.
            let mut tries: HashMap<&'static str, (usize, usize)> = HashMap::new();
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
            // Plus-selection: the best mu of parents and offspring (distinct qualities, as layouts are hard to compare).
            population.extend(
                children
                    .into_iter()
                    .map(|(child, q, _, _)| (Member::new(child), q)),
            );
            population.sort_by(|(_, a), (_, b)| b.total_cmp(a));
            population.dedup_by(|(_, a), (_, b)| (*a - *b).abs() < 1e-12);
            population.truncate(mu);

            monitor.record_mutations(&stats);
            monitor.record(layout_stats(
                generation + 1,
                &population,
                corners,
                started_at,
            ));
            if population[0].1 > best + 1e-12 {
                best = population[0].1;
                stale = 0;
                monitor.offer_best(best, || self.with_layout(population[0].0.layout.clone()));
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
        let mut result = self.with_layout(population.swap_remove(0).0.layout.clone());
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

// A layout of the population, with the levels of its zones (computed once, when it is first mutated).
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

fn layout_stats(
    generation: usize,
    population: &[(std::sync::Arc<Member>, f64)],
    corners: usize,
    started_at: Instant,
) -> GenerationStats {
    let qualities = population.iter().map(|(_, q)| *q).collect_vec();
    let n = qualities.len().max(1) as f64;
    let mean = qualities.iter().sum::<f64>() / n;
    GenerationStats {
        generation,
        best: qualities.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        mean,
        loops: corners,
        phase: LoopPhase::default(),
        layout: true,
        seconds: started_at.elapsed().as_secs_f64(),
    }
}

// Rank-based selection: the i-th best of n is chosen with probability proportional to n - i.
fn rank_pick(n: usize) -> usize {
    let total = n * (n + 1) / 2;
    let mut pick = rand::random_range(0..total.max(1));
    for i in 0..n {
        if pick < n - i {
            return i;
        }
        pick -= n - i;
    }
    0
}
