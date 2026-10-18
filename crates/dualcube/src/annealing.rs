//! Simulated annealing of the loop structure (see `Solution::anneal`).
//!
//! Several chains anneal side by side from the same start (see `AnnealParams::chains`; the best of several chains varies
//! less than one chain). Every step, a batch of children of the current solution of every chain is made with the
//! mutations (see below; all in parallel, screened and embedded). Per chain, its best child replaces its current
//! solution if it is at least as good, or else with probability exp(difference / temperature); a chain that did not
//! find a new best solution for `restart` steps goes back to the best. The temperature falls linearly from its start to
//! 0, and the mix of the mutations follows it: from that of growth (hot: mostly additions) to that of shaping and
//! pruning (cold: moving, merging and removing loops), with an adaptive choice of the mutations (see
//! `MutationSelection`). The starting temperature follows from the worsenings of the first step (see
//! `AnnealParams::acceptance`), such that it suits the scale of the quality of the model (the quality of different
//! models differs by an order of magnitude). The result is the best solution seen, embedded (see below), and never
//! worse than the input.

//!
//! The mutations of the loops:
//!
//! - added loops are valid by construction: they follow a topological structure of the paper's filtered graph G^V
//!   (see `Solution::sample_valid_loop`), as in Section 4.1,
//! - `add routed` first picks a route through the current loop structure (a sequence of regions), and adds the
//!   cheapest loop along it; it adds one loop per axis at most, and is targeted: with probability `targeted`, a loop is
//!   added through a badly aligned triangle (sampled proportionally to its area times its flattening distortion), along
//!   one of the two axes that can create a patch with the label that the triangle needs, or (with probability
//!   `interest`) through a point of interest (a concentration of Gaussian curvature, such as the tip of a protrusion),
//! - `add free` adds the cheapest valid loop over all routes (sequences of regions), following the flow as long as it
//!   is cheap (e.g., around the outline of a flat shape, over all its protrusions, where `add routed` would cut
//!   across), from a few starts, the cheapest per length,
//! - a `feature` mutation adds intersecting loops of two or three axes through the same point at once,
//! - a `replace` mutation searches the cheapest path for a loop that crosses the same loops in the same order (it
//!   keeps the loop structure, and only moves the loop),
//! - a `merge` mutation replaces two neighboring loops of the same axis by one loop that crosses everything that they
//!   cross (the union of their crossings), or the crossings of one of them, preferably for loops that run alongside
//!   each other (crossing the same loops, close together),
//! - removals only pick loops that keep the loop structure valid (checked locally, see `Dual::check_removal`).
//!
//! Children are screened with a cheap estimate of their quality (from the dual structure only), and only the best
//! fraction `screen` is embedded (layout) and scored; duplicates (also across steps) are evaluated once.
//!
//! The annealing only works on the base solution: loops on the input mesh, scored with a quick embedding (no
//! smoothing). The result is embedded without any optimization (smoothing, corner placement); those are applied
//! afterwards, on top of it.

use crate::prelude::*;
use rand::seq::{IteratorRandom, SliceRandom};
use serde::{Deserialize, Serialize};
use slotmap::Key as _;
use std::sync::Arc;
use std::time::Instant;

/// The mutations of the loops, by their names in the statistics (see the module documentation).
pub const LOOP_MUTATIONS: [&str; 6] =
    ["add routed", "add free", "feature", "replace", "merge", "remove"];

// The prior weights of the mutations when hot (mostly additions, and moving loops) and when cold (moving, merging and
// removing loops); in between, a mix (see `AnnealParams`).
const HOT: [(&str, f64); 5] = [
    ("add routed", 0.35),
    ("add free", 0.1),
    ("feature", 0.1),
    ("replace", 0.2),
    ("merge", 0.05),
];
const COLD: [(&str, f64); 3] = [("replace", 0.175), ("merge", 0.275), ("remove", 0.375)];

// The prior weights of all mutations at the given heat (1: hot, 0: cold), without those that are switched off.
fn mutation_prior(hot: f64, disabled: &std::collections::BTreeSet<String>) -> Vec<(&'static str, f64)> {
    let weight = |table: &[(&str, f64)], name: &str| {
        table
            .iter()
            .find(|(mutation, _)| *mutation == name)
            .map_or(0., |&(_, weight)| weight)
    };
    LOOP_MUTATIONS
        .iter()
        .map(|&name| {
            let prior = hot * weight(&HOT, name) + (1. - hot) * weight(&COLD, name);
            (name, if disabled.contains(name) { 0. } else { prior })
        })
        .collect()
}

// The distance (relative to their size) below which two loops count as close for a merge (see `Solution::merge`).
const MERGE_DISTANCE: f64 = 0.05;

// The mean distance from the points of one loop to the nearest point of the other (both ways), over the mean radius of
// the loops (their length over 2 pi).
fn relative_distance(a: &[Vector3D], b: &[Vector3D]) -> f64 {
    let nearest = |from: &[Vector3D], to: &[Vector3D]| {
        from.iter()
            .map(|p| {
                to.iter()
                    .map(|q| (p - q).norm())
                    .fold(f64::INFINITY, f64::min)
            })
            .sum::<f64>()
            / from.len().max(1) as f64
    };
    let length = |points: &[Vector3D]| {
        points
            .iter()
            .circular_tuple_windows()
            .map(|(p, q)| (p - q).norm())
            .sum::<f64>()
    };
    let radius = (length(a) + length(b)) / 2. / std::f64::consts::TAU;
    if a.is_empty() || b.is_empty() || radius <= 0. {
        return f64::INFINITY;
    }
    (nearest(a, b) + nearest(b, a)) / 2. / radius
}

// The union of the crossings of two loops that are merged (see `Solution::merge`): the shortest sequence that contains
// both orders of crossings (as subsequences), over the rotations of the second, in either direction; it starts like
// the first.
fn union_order(first: &[LoopID], second: &[LoopID]) -> Vec<LoopID> {
    // The shortest common supersequence (from the longest common subsequence).
    let supersequence = |a: &[LoopID], b: &[LoopID]| {
        let (n, m) = (a.len(), b.len());
        let mut lcs = vec![vec![0usize; m + 1]; n + 1];
        for i in (0..n).rev() {
            for j in (0..m).rev() {
                lcs[i][j] = if a[i] == b[j] {
                    lcs[i + 1][j + 1] + 1
                } else {
                    lcs[i + 1][j].max(lcs[i][j + 1])
                };
            }
        }
        let (mut i, mut j, mut out) = (0, 0, vec![]);
        while i < n || j < m {
            if i < n && j < m && a[i] == b[j] {
                out.push(a[i]);
                i += 1;
                j += 1;
            } else if j == m || (i < n && lcs[i + 1][j] >= lcs[i][j + 1]) {
                out.push(a[i]);
                i += 1;
            } else {
                out.push(b[j]);
                j += 1;
            }
        }
        out
    };
    let reversed = second.iter().rev().copied().collect_vec();
    [second.to_vec(), reversed]
        .iter()
        .flat_map(|b| {
            (0..b.len().max(1)).map(move |r| {
                let rotated = b
                    .iter()
                    .cycle()
                    .skip(r)
                    .take(b.len())
                    .copied()
                    .collect_vec();
                supersequence(first, &rotated)
            })
        })
        .min_by_key(Vec::len)
        .unwrap_or_else(|| first.to_vec())
}

// Report the chances of all mutations to the monitor (those not used, with chance 0).
fn report_chances(
    monitor: &OptimizationMonitor,
    selection: &MutationSelection,
    chances: &[(&'static str, f64)],
) {
    let reported = LOOP_MUTATIONS
        .iter()
        .map(|&name| {
            let chance = chances
                .iter()
                .find(|(mutation, _)| *mutation == name)
                .map_or(0., |&(_, chance)| chance);
            (name, chance)
        })
        .collect_vec();
    monitor.record_chances(&selection.report(&reported));
}



/// Parameters of the simulated annealing of the loops.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnnealParams {
    /// Children per step, of all chains together (made and evaluated in parallel; per chain, the best is the
    /// candidate).
    pub offspring: usize,
    /// Number of steps.
    pub steps: usize,
    /// Number of chains: independent annealings from the same start, each with its share of the children; the result
    /// is the best of all (less variance than one chain with all the children).
    #[serde(default = "default_chains")]
    pub chains: usize,
    /// A chain that did not find a new best solution for this many steps goes back to the best solution (0: never).
    #[serde(default = "default_restart")]
    pub restart: usize,
    /// The chance that the median worsening among the children of the first step is accepted at the start: sets the
    /// starting temperature (a difference of quality, which depends on the model).
    pub acceptance: f64,
    /// Probability that an added loop is targeted at a badly aligned triangle or a point of interest.
    pub targeted: f64,
    /// Probability that a targeted loop starts at a point of interest instead of at a badly aligned triangle.
    pub interest: f64,
    /// Fraction of the children (by estimated quality) that is embedded and scored.
    pub screen: f64,
}

const fn default_chains() -> usize {
    1
}

const fn default_restart() -> usize {
    0
}

impl Default for AnnealParams {
    fn default() -> Self {
        Self {
            // (Half of them are scored, see `screen`: 16, one per thread. In the optimization (90 s, 3 models, 2 runs),
            // 32 children did best: 0.614 against 0.544 with 16, and 0.603 with 16 all scored.)
            offspring: 32,
            steps: 100,
            chains: default_chains(),
            // (In the optimization (5 minutes, 4 models), one chain without going back to the best did best: 0.647 against
            // 0.633 with 2 chains, 0.593 with 4, and 0.588 with one chain going back after 40 steps.)
            restart: default_restart(),
            // As good as the best fixed starting temperature (0.03, of 0.01, 0.03 and 0.1) in tests.
            acceptance: 0.5,
            targeted: 0.5,
            interest: 0.3,
            screen: 0.5,
        }
    }
}

// Children whose acceptance is less likely than this are not accepted (and not prepared to be parents).
const MIN_ACCEPTANCE: f64 = 0.01;

impl Solution {
    /// Simulated annealing of the loop structure (see the module documentation), reporting to (and stoppable through)
    /// the given monitor.
    pub fn anneal(
        &self,
        params: &AnnealParams,
        monitor: &OptimizationMonitor,
    ) -> Result<Self, SolutionError> {
        let result = self
            .anneal_phase(params, monitor)
            .map(|result| self.better_of(result));
        monitor.finish();
        result
    }

    // The annealing (see `anneal`): its best solution, embedded, even if it is worse than the input.
    pub(crate) fn anneal_phase(
        &self,
        params: &AnnealParams,
        monitor: &OptimizationMonitor,
    ) -> Result<Self, SolutionError> {
        let initial_quality = self.get_quality().ok_or(SolutionError::NoPrimal)?;
        let started_at = Instant::now();
        profile::reset();
        monitor.offer_best(initial_quality, || self.clone());
        let (seed, seed_quality) = self.search_seed(initial_quality);
        let context = MutationContext::new(&seed);
        let mut seen = HashSet::from_iter([seed.loop_signature()]);
        let mut selection = monitor.take_selection("loops");
        let mut stats = monitor.mutation_stats_map();
        let mut best = (seed, seed_quality);
        // Every chain: its current solution, and the steps since it found a new best.
        let chains = params.chains.clamp(1, params.offspring.max(1));
        let mut currents = vec![(best.clone(), 0); chains];
        // The best quality that every chain found (for the statistics).
        let mut chain_bests = vec![best.1; chains];
        let per_chain = (params.offspring / chains).max(1);
        // The starting temperature (set after the first step; until then, every child is prepared to be a parent).
        let mut start = None;

        for step in 0..params.steps {
            let hot = 1. - step as f64 / params.steps as f64;
            // The mix of the mutations follows the temperature (those switched off can change while running).
            let disabled = monitor.settings().map(|s| s.loops_disabled).unwrap_or_default();
            let chances = selection.chances(&mutation_prior(hot, &disabled));
            report_chances(monitor, &selection, &chances);

            // The children of all chains, in parallel.
            let attempts = (0..chains * per_chain)
                .collect_vec()
                .into_par()
                .with_pool(pool())
                .filter_map(|i| {
                    let chain = i / per_chain;
                    let op = choose_mutation(&chances)?;
                    let timer = Instant::now();
                    let child =
                        profile::timed(3, || currents[chain].0.0.mutate(params, &context, op));
                    monitor.record_mutation_times([(op, timer.elapsed())]);
                    Some((op, child.map(|(child, op)| (child, (chain, op)))))
                })
                .collect::<Vec<_>>();
            // Tries and improvements of the best per mutation (see `MutationSelection`); failed tries count too.
            let mut tries: HashMap<&'static str, (usize, usize)> = HashMap::default();
            for (op, child) in &attempts {
                tries.entry(child.as_ref().map_or(*op, |(_, (_, op))| *op)).or_default().0 += 1;
            }
            let children = attempts.into_iter().filter_map(|(_, child)| child).collect_vec();
            for (_, (_, op)) in &children {
                stats.entry(op).or_default()[0] += 1;
            }
            // (Children that no chain would accept are not prepared to be parents.)
            let worst = currents.iter().map(|((_, q), _)| *q).fold(f64::INFINITY, f64::min);
            let threshold =
                start.map_or(f64::NEG_INFINITY, |start| worst + start * hot * MIN_ACCEPTANCE.ln());
            let offspring = Self::evaluate_offspring(children, params, &mut seen, threshold, best.1);
            // The starting temperature: the median worsening is accepted with chance `acceptance`.
            let start = *start.get_or_insert_with(|| {
                let mut worse = offspring
                    .iter()
                    .map(|(_, quality, (chain, _))| currents[*chain].0.1 - quality)
                    .filter(|&d| d > 0.)
                    .collect_vec();
                worse.sort_by(f64::total_cmp);
                let median = worse.get(worse.len() / 2).copied().unwrap_or(0.01 * best.1);
                let start = median / (1. / params.acceptance.clamp(1e-6, 1. - 1e-6)).ln();
                info!("anneal: starting temperature {start} (median worsening {median})");
                start
            });
            let temperature = start * hot;
            for (_, quality, (chain, op)) in &offspring {
                let entry = stats.entry(op).or_default();
                entry[1] += 1;
                if *quality > currents[*chain].0.1 {
                    entry[2] += 1;
                }
                if *quality > best.1 + 1e-12 {
                    entry[3] += 1;
                    tries.entry(op).or_default().1 += 1;
                }
            }
            selection.record(tries);

            // Every chain: its best child replaces its current solution if it is at least as good, or else with
            // probability exp(difference / temperature).
            // (The solutions that are not kept are dropped on other threads.)
            let mut candidates: Vec<Option<(Self, f64)>> = vec![None; chains];
            let mut discarded = vec![];
            for (child, quality, (chain, _)) in offspring {
                if candidates[chain].as_ref().is_none_or(|(_, q)| quality > *q) {
                    discarded.extend(candidates[chain].replace((child, quality)));
                } else {
                    discarded.push((child, quality));
                }
            }
            for (chain, candidate) in candidates.into_iter().enumerate() {
                let (current, stale) = &mut currents[chain];
                let threshold = current.1 + temperature * MIN_ACCEPTANCE.ln();
                if let Some((child, quality)) = candidate {
                    if quality >= threshold
                        && (quality >= current.1
                            || rand::random::<f64>() < ((quality - current.1) / temperature).exp())
                    {
                        discarded.push(std::mem::replace(current, (child, quality)));
                    } else {
                        discarded.push((child, quality));
                    }
                }
                chain_bests[chain] = chain_bests[chain].max(current.1);
                if current.1 > best.1 + 1e-12 {
                    best = current.clone();
                    *stale = 0;
                    monitor.offer_best(best.1, || self.with_embedding_of(&best.0));
                } else {
                    *stale += 1;
                }
            }
            drop_later(discarded);
            // A chain without a new best for a while goes back to the best.
            for (current, stale) in &mut currents {
                if params.restart > 0 && *stale >= params.restart {
                    *current = best.clone();
                    *stale = 0;
                }
            }

            let (current, _) = currents
                .iter()
                .max_by(|((_, a), _), ((_, b), _)| a.total_cmp(b))
                .expect("at least one chain");
            monitor.record_mutations(&stats);
            monitor.record(GenerationStats {
                generation: step + 1,
                best: current.1,
                current: current.1,
                loops: current.0.loops.len(),
                layout: false,
                seconds: started_at.elapsed().as_secs_f64(),
                temperature: Some(temperature),
                chains: std::array::from_fn(|chain| chain_bests.get(chain).copied()),
            });
            info!(
                "anneal: step {} temperature={temperature:.4} current={} best={} loops={}",
                step + 1,
                current.1,
                best.1,
                current.0.loops.len()
            );
            if monitor.stopping() {
                break;
            }
        }
        monitor.put_selection("loops", selection);
        info!("anneal: mutation statistics (generated, evaluated, better than current, new best): {stats:?}");
        info!(
            "anneal: best solution with quality {} and {} loops after {:?}",
            best.1,
            best.0.loops.len(),
            started_at.elapsed()
        );
        info!("anneal: CPU time per phase: {}", profile::report());
        // The best solution as it was scored (else, e.g. if the seed could not be embedded, embedded again).
        if best.0.layout.is_some() {
            Ok(self.with_embedding_of(&best.0))
        } else {
            Ok(self.embed_result([best.0]))
        }
    }
}

// CPU time (summed over all threads) per part of the annealing, reported at the end of a run (debug log).
mod profile {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    pub const PHASES: [&str; 8] = [
        "search loop",
        "place loop",
        "dual structure",
        "mutate (other)",
        "estimate",
        "embed",
        "quality",
        "compact",
    ];
    static NANOS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

    pub fn timed<T>(phase: usize, f: impl FnOnce() -> T) -> T {
        let timer = Instant::now();
        let result = f();
        NANOS[phase].fetch_add(timer.elapsed().as_nanos() as u64, Ordering::Relaxed);
        result
    }

    pub fn reset() {
        for nanos in &NANOS {
            nanos.store(0, Ordering::Relaxed);
        }
    }

    pub fn report() -> String {
        PHASES
            .iter()
            .zip(&NANOS)
            .map(|(name, nanos)| {
                format!("{name}={:.1}s", nanos.load(Ordering::Relaxed) as f64 * 1e-9)
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

// Data shared by all mutations of one run.
struct MutationContext {
    // Faces weighted by the (smoothed) absolute Gaussian curvature around them (points of interest).
    interest: Vec<(FaceID, f64)>,
    interest_total: f64,
}

impl MutationContext {
    fn new(solution: &Solution) -> Self {
        let mesh = &solution.mesh_ref;
        let verts = mesh.vert_ids();
        let index: HashMap<VertID, usize> =
            verts.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        let mut curvature = verts.iter().map(|&v| mesh.defect(v)).collect_vec();
        // Smooth (mass-preserving), such that noise cancels out and concentrations remain.
        for _ in 0..5 {
            let mut next = vec![0.; curvature.len()];
            for (i, &v) in verts.iter().enumerate() {
                let neighbors = mesh
                    .neighbors(v)
                    .filter_map(|n| index.get(&n).copied())
                    .collect_vec();
                let share = curvature[i] / (neighbors.len() + 1) as f64;
                next[i] += share;
                for j in neighbors {
                    next[j] += share;
                }
            }
            curvature = next;
        }
        // Only pronounced concentrations: the faces around the vertices with the largest absolute curvature.
        let mut magnitudes = curvature.iter().map(|k| k.abs()).collect_vec();
        magnitudes.sort_by(f64::total_cmp);
        let threshold = magnitudes
            .get(magnitudes.len().saturating_sub(1 + magnitudes.len() / 50))
            .copied()
            .unwrap_or(0.);
        let interest = mesh
            .face_ids()
            .into_iter()
            .map(|face| {
                let weight = mesh
                    .vertices(face)
                    .map(|v| curvature[index[&v]].abs())
                    .filter(|&k| k >= threshold)
                    .sum::<f64>();
                (face, weight)
            })
            .filter(|&(_, w)| w > 0.)
            .collect_vec();
        let interest_total = interest.iter().map(|(_, w)| w).sum();
        Self {
            interest,
            interest_total,
        }
    }

    fn sample_interest(&self) -> Option<FaceID> {
        if self.interest_total <= 0. {
            return None;
        }
        let mut pick = rand::random::<f64>() * self.interest_total;
        self.interest
            .iter()
            .find(|(_, w)| {
                pick -= w;
                pick <= 0.
            })
            .or(self.interest.last())
            .map(|(face, _)| *face)
    }
}

/// A canonical signature of a loop structure: loops (with their direction) rotated to start at their smallest edge,
/// sorted.
pub type LoopSignature = Vec<(usize, Vec<u64>)>;

impl Solution {
    // This solution with the loops, dual structure, polycube and layout of an embedded candidate (see `search_seed` and
    // `evaluate_offspring`), without the data of the search.
    fn with_embedding_of(&self, candidate: &Self) -> Self {
        let mut result = self.clone();
        result.loops = candidate.loops.clone();
        result.occupied = candidate.occupied.clone();
        result.dual = candidate.dual.clone();
        result.polycube = candidate.polycube.clone();
        result.layout = candidate.layout.clone();
        result.quad = None;
        result.corner_hint = None;
        result.targets = None;
        result.loop_penalties = None;
        result
    }

    // The start of an annealing: this solution with the given quality, scored like the offspring: with a quick
    // embedding of its own, not near the corners of its layout (which may be optimized: its quality would then be
    // close to that optimized one, which the quick embeddings of other loop structures cannot compare with). The
    // annealing thus compares loop structures by their quick embeddings only; its children start from these corners.
    // With its embedding (it may be the result).
    fn search_seed(&self, quality: f64) -> (Self, f64) {
        let mut seed = self.clone();
        seed.prepare_flow();
        let mut seed_quality = quality;
        let mut scored_seed = seed.clone();
        if scored_seed
            .reconstruct_near(true, LayoutParams::fast(), None)
            .is_ok()
            && let Some(quality) = scored_seed.get_quality()
        {
            seed = scored_seed;
            seed_quality = quality;
        }
        seed.prepare_as_parent();
        (seed, seed_quality)
    }

    // The first of the candidates (from the best) that can be embedded (without optimizations), with the loops of the
    // candidate; this solution if none can. The full embedding may fail where the fast one succeeded; then fall back
    // to more attempts, the fast embedding, and finally the next candidate.
    fn embed_result(&self, candidates: impl IntoIterator<Item = Self>) -> Self {
        for candidate in candidates {
            let mut result = self.clone();
            result.loops = candidate.loops;
            result.occupied = candidate.occupied;
            let hint = candidate.corner_hint;
            let fallbacks = [
                LayoutParams::default(),
                LayoutParams {
                    attempts: 10,
                    ..LayoutParams::default()
                },
                LayoutParams::fast(),
            ];
            for layout_params in fallbacks {
                if result
                    .reconstruct_near(false, layout_params, hint.as_deref())
                    .is_ok()
                    && result.get_quality().is_some()
                {
                    return result;
                }
                warn!("anneal: embedding the result failed with {layout_params:?}; retrying");
            }
        }
        self.clone()
    }

    // The result of an optimization, unless it scores worse than this (input) solution.
    fn better_of(&self, result: Self) -> Self {
        match (self.get_quality(), result.get_quality()) {
            (Some(input), Some(output)) if output < input => {
                info!(
                    "anneal: result ({output}) is worse than the input ({input}); keeping the input"
                );
                self.clone()
            }
            (Some(_), None) => self.clone(),
            _ => result,
        }
    }

    // Embed and score the (new) offspring: duplicates are skipped, the rest is screened with an estimate first.
    // Children scoring below `threshold` cannot enter the population, so their targets (for targeted mutations, see
    // `compact_for_search`) are not computed.
    fn evaluate_offspring<T: Send + Sync + Copy>(
        children: Vec<(Self, T)>,
        params: &AnnealParams,
        seen: &mut HashSet<LoopSignature>,
        threshold: f64,
        best: f64,
    ) -> Vec<(Self, f64, T)> {
        let children = children
            .into_iter()
            .filter(|(child, _)| seen.insert(child.loop_signature()))
            .collect_vec();
        let children = if params.screen < 1. {
            let estimated = children
                .into_par()
                .with_pool(pool())
                .filter_map(|(mut child, tag)| {
                    // The dual structure is needed for the estimate and the embedding: build it once.
                    if !child.dual_is_current() {
                        child.dual =
                            profile::timed(2, || Dual::from(child.mesh_ref.clone(), &child.loops));
                    }
                    profile::timed(4, || child.estimate_quality()).map(|q| (child, tag, q))
                })
                .collect::<Vec<_>>();
            let keep = ((estimated.len() as f64 * params.screen).ceil() as usize).max(1);
            let mut estimated = estimated;
            estimated.sort_by(|(_, _, a), (_, _, b)| b.total_cmp(a));
            let screened_out = estimated.split_off(keep.min(estimated.len()));
            drop_later(screened_out.into_iter().map(|(child, ..)| child).collect_vec());
            estimated.into_iter().map(|(child, tag, _)| (child, tag)).collect_vec()
        } else {
            children
        };
        children
            .into_par()
            .with_pool(pool())
            .filter_map(|(mut child, tag)| {
                let timer = Instant::now();
                // Near the corners of the parent (see `CornerHint`).
                let hint = child.corner_hint.clone();
                let result = profile::timed(5, || {
                    child.reconstruct_near(true, LayoutParams::fast(), hint.as_deref())
                });
                if timer.elapsed().as_secs_f64() > 5. {
                    warn!(
                        "anneal: slow evaluation ({:?}) of a child with {} loops: {result:?}",
                        timer.elapsed(),
                        child.loops.len()
                    );
                }
                result.ok()?;
                let quality = profile::timed(6, || child.get_quality())?;
                profile::timed(7, || {
                    if quality >= threshold {
                        child.prepare_as_parent();
                    }
                    // (A new best keeps its embedding: it may be the result.)
                    if quality <= best {
                        child.compact();
                    }
                });
                Some((child, quality, tag))
            })
            .collect()
    }

    // A mutation (see the module documentation): replace, feature, removal (with the given probability), or addition.
    // The given mutation (see `MutationSelection`); an addition if it does not apply (e.g., too few loops).
    fn mutate(
        &self,
        params: &AnnealParams,
        context: &MutationContext,
        op: &'static str,
    ) -> Option<(Self, &'static str)> {
        let mut child = self.clone_loop_state();
        let enough = self.loops.len() > 3;

        if self.loops.len() > 4 && op == "merge" {
            child.dual = self.dual.clone();
            return child.merge().map(|()| (child, "merge"));
        }
        if enough && op == "replace" {
            child.dual = self.dual.clone();
            return child.replace(self).map(|()| (child, "replace"));
        }

        // The dual structure of the parent is up to date for the child (until loops are added or removed).
        child.dual = self.dual.clone();

        if op == "feature" {
            // Feature: intersecting loops of several axes through the same point: a point of interest, or a badly
            // aligned triangle (a region with a bad score).
            let face = if rand::random::<bool>() {
                context.sample_interest()?
            } else {
                self.targeted_face(self, None)?.0
            };
            let mut axes = DIRECTIONS;
            axes.shuffle(&mut rand::rng());
            let count = rand::random_range(2..=3);
            let mut added = 0;
            for &axis in axes.iter().take(count) {
                let start = child.best_anchor_in_face(face, axis).map(|[e1, _]| e1);
                if child.add_valid_loop(axis, start, false) {
                    added += 1;
                }
            }
            debug!("mutation: operation=feature added={added}");
            return (added > 0).then_some((child, "feature"));
        }

        if op == "add free" {
            // A loop of a random axis over any route (see the module documentation).
            let axis = DIRECTIONS[rand::random_range(0..3)];
            let targeted = rand::random::<f64>() < params.targeted;
            return child
                .add_new_loop(self, axis, targeted, true, params, context)
                .then_some((child, "add free"));
        }

        if enough && op == "remove" {
            child.remove_valid_loop(self)?;
            debug!("mutation: operation=remove");
            return Some((child, "remove"));
        }

        if op != "add routed" {
            return None;
        }
        // Along a route: zero or one loop per axis (at least one loop in total).
        let mut counts = [0; 3].map(|_: usize| rand::random_range(0..=1));
        if counts.iter().sum::<usize>() == 0 {
            counts[rand::random_range(0..3)] = 1;
        }
        let mut added = 0;
        for (axis, count) in DIRECTIONS.into_iter().zip(counts) {
            for _ in 0..count {
                let targeted = rand::random::<f64>() < params.targeted;
                if child.add_new_loop(self, axis, targeted, false, params, context) {
                    added += 1;
                }
            }
        }
        debug!("mutation: operation=add routed counts={counts:?} added={added}");
        (added > 0).then_some((child, "add routed"))
    }

    // Add a valid loop of the given axis (from the given edge, if any), and update the dual structure. Returns whether
    // a loop was added. Free: over all sequences of regions.
    fn add_valid_loop(&mut self, axis: Direction, start: Option<EdgeID>, free: bool) -> bool {
        let found = profile::timed(0, || {
            if free {
                self.free_valid_loop(axis, start, 2)
            } else {
                self.sample_valid_loop(axis, start, 2)
            }
        });
        let Some((edges, _, gaps)) = found else {
            return false;
        };
        let loop_id = profile::timed(1, || self.add_loop_in_gaps(Loop::new(edges, axis), gaps));
        match profile::timed(2, || Dual::from(self.mesh_ref.clone(), &self.loops)) {
            Ok(dual) => {
                self.dual = Ok(dual);
                true
            }
            Err(err) => {
                debug!("add_valid_loop: constructed loop is invalid: {err:?}");
                self.del_loop(loop_id);
                false
            }
        }
    }

    // Remove a loop that keeps the loop structure valid (using the dual structure of the parent, which is up to date).
    fn remove_valid_loop(&mut self, parent: &Self) -> Option<()> {
        // (Bad loops are more likely removed; see `loop_penalty`.)
        let loop_id = pick_weighted(
            parent
                .dual
                .as_ref()
                .ok()?
                .removable_loops()
                .into_iter()
                .filter(|loop_id| self.loops.contains_key(*loop_id))
                .map(|loop_id| (loop_id, parent.loop_penalty(loop_id))),
        )?;
        self.del_loop(loop_id);
        Some(())
    }

    // Merge: replace two loops of the same axis that bound a common loop region (neighbors without loops of their axis
    // between them) by one loop. Loops that run alongside each other (crossing the same loops, close together) are
    // preferred. The new loop crosses everything that the two cross (the union of their crossings, see `union_order`),
    // or else the same loops as one of the two (searched like a replace); if that fails, it is the cheapest valid loop
    // from a region between them (over all sequences of regions).
    fn merge(&mut self) -> Option<()> {
        // Candidates: two loops of the same axis that bound a common region, with the regions between them (bounded
        // by both) and their crossings.
        let (pairs, crossings) = {
            let dual = self.dual.as_ref().ok()?;
            let structure = &dual.loop_structure;
            let mut pairs: HashMap<(LoopID, LoopID), Vec<LoopRegionID>> = HashMap::default();
            for region in structure.face_ids() {
                let loops = structure
                    .edges(region)
                    .map(|segment| dual.segment_to_loop(segment))
                    .unique()
                    .sorted()
                    .collect_vec();
                for (i, &a) in loops.iter().enumerate() {
                    for &b in &loops[i + 1..] {
                        if self.loops[a].direction == self.loops[b].direction {
                            pairs.entry((a, b)).or_default().push(region);
                        }
                    }
                }
            }
            let crossings: HashMap<LoopID, Vec<(usize, LoopID)>> = pairs
                .keys()
                .flat_map(|&(a, b)| [a, b])
                .unique()
                .map(|loop_id| (loop_id, self.crossings_along(dual, loop_id)))
                .collect();
            (pairs, crossings)
        };
        if pairs.is_empty() {
            return None;
        }

        // Pairs that run alongside each other are preferred: the weight is the square of the fraction of their
        // crossings that have a region between them (1 for two loops that cross the same loops in the same order),
        // over their distance relative to their size (see `MERGE_DISTANCE`).
        let mut positions: HashMap<LoopID, Vec<Vector3D>> = HashMap::default();
        let mut weighted = vec![];
        for (&(a, b), regions) in &pairs {
            for loop_id in [a, b] {
                positions
                    .entry(loop_id)
                    .or_insert_with(|| self.get_coordinates_of_loop(loop_id));
            }
            let most = crossings[&a].len().max(crossings[&b].len()).max(1);
            let shared = (regions.len() as f64 / most as f64).min(1.);
            let distance = relative_distance(&positions[&a], &positions[&b]);
            weighted.push(((a, b), shared * shared / (distance + MERGE_DISTANCE)));
        }
        let (a, b) = pick_weighted(weighted)?;
        let axis = self.loops[a].direction;

        // One loop that crosses everything that the two cross (the union of their crossings, see `union_order`), or
        // (in random order) the crossings of one of the two: searched like a replace, from where the first of the two
        // was. Otherwise, the cheapest valid loop from a region between them.
        let order = if rand::random::<bool>() {
            [a, b]
        } else {
            [b, a]
        };
        let sequence_of = |keep: LoopID| {
            let list = &crossings[&keep];
            (1..=list.len())
                .map(|i| list[i % list.len()].1)
                .collect_vec()
        };
        let union = union_order(&sequence_of(order[0]), &sequence_of(order[1]));
        let attempts = [
            (order[0], union),
            (order[0], sequence_of(order[0])),
            (order[1], sequence_of(order[1])),
        ];
        for (keep, sequence) in attempts {
            let old = self.loops[keep].clone();
            let list = &crossings[&keep];
            if list.len() < 2 || sequence.len() < 2 {
                continue;
            }
            let (from, to) = (list[0].0, list[1].0);
            let len = old.edges.len();
            let middle = if to > from {
                (from + to) / 2
            } else {
                (from + (to + len)) / 2 % len
            };
            let mut trial = self.clone_loop_state();
            trial.del_loop(a);
            trial.del_loop(b);
            let Some((edges, _, gaps)) =
                trial.construct_crossing_loop(axis, old.edges[middle], &sequence)
            else {
                continue;
            };
            trial.add_loop_in_gaps(Loop::new(edges, axis), gaps);
            if let Ok(dual) = Dual::from(trial.mesh_ref.clone(), &trial.loops) {
                trial.dual = Ok(dual);
                *self = trial;
                debug!(
                    "mutation: operation=merge axis={axis:?} crossings={}",
                    sequence.len()
                );
                return Some(());
            }
        }
        let start = {
            let dual = self.dual.as_ref().ok()?;
            let region = *pairs[&(a, b)].iter().choose(&mut rand::rng())?;
            let vert = dual
                .region_to_verts(region)
                .into_iter()
                .choose(&mut rand::rng())?;
            self.mesh_ref.edges(vert).next()?
        };
        let mut trial = self.clone();
        trial.del_loop(a);
        trial.del_loop(b);
        trial.dual = Ok(Dual::from(trial.mesh_ref.clone(), &trial.loops).ok()?);
        if !trial.add_valid_loop(axis, Some(start), true) {
            return None;
        }
        *self = trial;
        debug!("mutation: operation=merge axis={axis:?} kept=free");
        Some(())
    }

    // The crossings of a loop in the order along it: the position on the loop, and the other loop.
    fn crossings_along(&self, dual: &Dual, loop_id: LoopID) -> Vec<(usize, LoopID)> {
        let position: HashMap<FaceID, usize> = self.loops[loop_id]
            .edges
            .iter()
            .enumerate()
            .rev()
            .map(|(i, &e)| (self.mesh_ref.face(e), i))
            .collect();
        dual.intersections_of(loop_id)
            .into_iter()
            .filter_map(|(face, other)| position.get(&face).map(|&i| (i, other)))
            .sorted()
            .collect()
    }

    // Replace: a loop (chosen by its penalty, see `loop_penalty`) is searched again as the cheapest loop of its axis that
    // crosses the same loops in the same order (from halfway between two of its crossings), which keeps the loop
    // structure and only moves the loop.
    fn replace(&mut self, parent: &Self) -> Option<()> {
        let dual = parent.dual.as_ref().ok()?;
        let loop_id = parent.badly_aligned_loop(&parent.loops.keys().collect_vec())?;
        let old = parent.loops[loop_id].clone();
        // The crossings in the order along the loop (by the position of their face on the loop).
        let position: HashMap<FaceID, usize> = old
            .edges
            .iter()
            .enumerate()
            .rev()
            .map(|(i, &e)| (self.mesh_ref.face(e), i))
            .collect();
        let crossings = dual
            .intersections_of(loop_id)
            .into_iter()
            .filter_map(|(face, other)| position.get(&face).map(|&i| (i, other)))
            .sorted()
            .collect_vec();
        if crossings.len() < 2 {
            return None;
        }
        // Start halfway between two consecutive crossings, and list the crossings from there on.
        let k = rand::random_range(0..crossings.len());
        let (from, to) = (crossings[k].0, crossings[(k + 1) % crossings.len()].0);
        let len = old.edges.len();
        let middle = if to > from {
            (from + to) / 2
        } else {
            (from + (to + len)) / 2 % len
        };
        let order = (0..crossings.len())
            .map(|i| crossings[(k + 1 + i) % crossings.len()].1)
            .collect_vec();
        self.del_loop(loop_id);
        let (edges, _, gaps) =
            self.construct_crossing_loop(old.direction, old.edges[middle], &order)?;
        if edges == old.edges {
            return None;
        }
        let new_id = self.add_loop_in_gaps(Loop::new(edges, old.direction), gaps);
        match Dual::from(self.mesh_ref.clone(), &self.loops) {
            Ok(dual) => {
                self.dual = Ok(dual);
                debug!("mutation: operation=replace axis={:?}", old.direction);
                Some(())
            }
            Err(_) => {
                self.del_loop(new_id);
                None
            }
        }
    }

    // The penalty of a loop (see `compute_loop_penalties`; the mean of the others for a new loop, 1 if unknown).
    fn loop_penalty(&self, loop_id: LoopID) -> f64 {
        let Some(penalties) = &self.loop_penalties else {
            return 1.;
        };
        penalties
            .get(&loop_id)
            .copied()
            .unwrap_or_else(|| penalties.values().sum::<f64>() / penalties.len().max(1) as f64)
    }

    // The penalty of every loop of the layout: the mean penalty of the polycube edges that its segments cross (see
    // `element_penalties`; every segment of a loop is dual to a polycube edge, whose penalty includes its path and half
    // of its two patches).
    fn compute_loop_penalties(&self) -> Option<HashMap<LoopID, f64>> {
        let dual = self.dual.as_ref().ok()?;
        let layout = self.layout.as_ref()?;
        let polycube = &layout.polycube_ref;
        let penalties = element_penalties(layout, &self.quality.weights);
        let structure = &dual.loop_structure;
        let mut sums: HashMap<LoopID, (f64, usize)> = HashMap::default();
        for segment in structure.edge_ids() {
            let regions = [
                structure.face(segment),
                structure.face(structure.twin(segment)),
            ];
            let Some([u, v]) = regions
                .iter()
                .map(|region| polycube.region_to_vertex.get_by_left(region).copied())
                .collect::<Option<Vec<_>>>()
                .and_then(|corners| corners.into_iter().collect_array::<2>())
            else {
                continue;
            };
            let Some((edge, _)) = polycube.structure.edge_between_verts(u, v) else {
                continue;
            };
            let entry = sums.entry(dual.segment_to_loop(segment)).or_default();
            entry.0 += penalties.paths.get(&edge).copied().unwrap_or(0.);
            entry.1 += 1;
        }
        Some(
            sums.into_iter()
                .map(|(loop_id, (sum, count))| (loop_id, sum / count.max(1) as f64))
                .collect(),
        )
    }

    // A loop among the given ones, chosen by its penalty (see `loop_penalty`) if known, else with probability
    // proportional to its misalignment (flow cost per length).
    fn badly_aligned_loop(&self, loops: &[LoopID]) -> Option<LoopID> {
        if self.loop_penalties.is_some() {
            return pick_weighted(
                loops
                    .iter()
                    .map(|&loop_id| (loop_id, self.loop_penalty(loop_id))),
            );
        }
        let graphs = self.flow_graphs.as_ref()?;
        let mesh = &self.mesh_ref;
        let weighted = loops
            .iter()
            .map(|&loop_id| {
                let lewp = &self.loops[loop_id];
                let graph = &graphs[lewp.direction as usize];
                let (mut cost, mut length) = (0., 0.);
                for [a, b] in Self::cycled_windows(&lewp.edges) {
                    if let Some(w) = graph.get_directed_weight(a, b) {
                        cost += w;
                    }
                    length += (mesh.position(b) - mesh.position(a)).norm();
                }
                (loop_id, (cost / length.max(1e-12)).max(1e-9))
            })
            .collect_vec();
        let total: f64 = weighted.iter().map(|(_, w)| w).sum();
        if total <= 0. {
            return None;
        }
        let mut pick = rand::random::<f64>() * total;
        weighted
            .iter()
            .find(|(_, w)| {
                pick -= w;
                pick <= 0.
            })
            .or(weighted.last())
            .map(|(loop_id, _)| *loop_id)
    }

    // Add a valid loop of the given axis, returns whether a loop was added. Targeted: through a badly aligned triangle
    // of the parent's layout (see `targeted_face`) or a point of interest (then possibly along another axis).
    fn add_new_loop(
        &mut self,
        parent: &Self,
        axis: Direction,
        targeted: bool,
        free: bool,
        params: &AnnealParams,
        context: &MutationContext,
    ) -> bool {
        for _ in 0..3 {
            let (axis, start) = if targeted {
                let target = if rand::random::<f64>() < params.interest {
                    context.sample_interest().map(|face| (face, axis))
                } else {
                    self.targeted_face(parent, Some(axis))
                };
                let Some((face, axis)) = target else {
                    continue;
                };
                (axis, self.best_anchor_in_face(face, axis).map(|[e1, _]| e1))
            } else {
                (axis, None)
            };
            if self.add_valid_loop(axis, start, free) {
                return true;
            }
        }
        false
    }

    // A badly aligned triangle of the parent's layout (sampled proportionally to its area times its flattening
    // distortion), with an axis for a loop through it. A patch with a label along axis `a` lies at the intersection
    // of loops of the two other axes, so the loop gets one of the two axes other than the axis of the triangle's best
    // label (the given axis, if it is one of these).
    fn targeted_face(&self, parent: &Self, axis: Option<Direction>) -> Option<(FaceID, Direction)> {
        let mesh = &self.mesh_ref;
        let computed;
        let weighted: &[(FaceID, f64)] = match (&parent.targets, &parent.layout) {
            (Some(targets), _) => targets,
            (None, Some(layout)) => {
                computed = penalty_per_input_face(layout, &parent.quality.weights);
                &computed
            }
            (None, None) => return None,
        };
        let total: f64 = weighted.iter().map(|(_, w)| w).sum();
        if total <= 0. {
            return None;
        }
        let mut pick = rand::random::<f64>() * total;
        let face = weighted
            .iter()
            .find(|(_, w)| {
                pick -= w;
                pick <= 0.
            })
            .map_or(weighted[weighted.len() - 1].0, |(face, _)| *face);
        let (wanted, _) = to_principal_direction(mesh.normal(face));
        let candidates = DIRECTIONS
            .into_iter()
            .filter(|&d| d != wanted)
            .collect_vec();
        let axis = match axis {
            Some(axis) if candidates.contains(&axis) => axis,
            _ => candidates[rand::random_range(0..candidates.len())],
        };
        Some((face, axis))
    }

    // The move inside the face that is best aligned with the flow of the given axis.
    fn best_anchor_in_face(&self, face: FaceID, axis: Direction) -> Option<[EdgeID; 2]> {
        let graph = &self.flow_graphs.as_ref()?[axis as usize];
        let mesh = &self.mesh_ref;
        let edges = mesh.edges(face).collect_vec();
        edges
            .iter()
            .copied()
            .cartesian_product(edges.clone())
            .filter(|(a, b)| a != b)
            .filter_map(|(a, b)| {
                let weight = graph.get_directed_weight(a, b)?;
                let length = (mesh.position(b) - mesh.position(a)).norm().max(1e-12);
                Some(([a, b], weight / length))
            })
            .min_by_key(|&(_, rank)| OrderedFloat(rank))
            .map(|(anchor, _)| anchor)
    }

    // Prepare an evaluated solution to be a parent: the misalignment per input face and the penalty per loop (for
    // targeted mutations), and the corners of its layout (for those of its children).
    fn prepare_as_parent(&mut self) {
        if let Some(layout) = &self.layout {
            self.targets = Some(Arc::new(penalty_per_input_face(
                layout,
                &self.quality.weights,
            )));
            self.loop_penalties = self.compute_loop_penalties().map(Arc::new);
            // Its children start their layouts from its corners.
            self.corner_hint = CornerHint::of(self).map(Arc::new);
        }
    }

    // Drop the embedding (the layout, polycube, and refined mesh of the dual structure) of an evaluated solution.
    fn compact(&mut self) {
        self.layout = None;
        self.polycube = None;
        self.quad = None;
        self.fields = None;
        if let Ok(dual) = &mut self.dual {
            dual.release_refined_mesh();
        }
    }

    pub(crate) fn clone_loop_state(&self) -> Self {
        // The mutations change only loop bookkeeping. Avoid cloning any derived
        // representations; they are rebuilt once the candidate needs scoring.
        // `flow_graphs` is Arc-backed, so this only bumps a refcount and lets
        // candidates keep sampling loops in later iterations.
        Self {
            mesh_ref: self.mesh_ref.clone(),
            loops: self.loops.clone(),
            occupied: self.occupied.clone(),
            dual: Err(PropertyViolationError::default()),
            polycube: None,
            layout: None,
            quad: None,
            flow_graphs: self.flow_graphs.clone(),
            fields: None,
            quality: self.quality,
            targets: None,
            corner_hint: self.corner_hint.clone(),
            loop_penalties: self.loop_penalties.clone(),
        }
    }

    /// A canonical signature of the loop structure (independent of the order of the loops and of where each loop's
    /// edge sequence starts).
    #[must_use]
    pub fn loop_signature(&self) -> LoopSignature {
        self.loops
            .values()
            .map(|lewp| {
                let raw = lewp
                    .edges
                    .iter()
                    .map(|e| e.raw().data().as_ffi())
                    .collect_vec();
                let start = raw.iter().position_min().unwrap_or(0);
                let mut rotated = raw[start..].to_vec();
                rotated.extend_from_slice(&raw[..start]);
                (lewp.direction as usize, rotated)
            })
            .sorted()
            .collect()
    }
}

#[cfg(test)]
mod mutation_tests {
    use crate::prelude::*;
    use std::sync::Arc;

    // A merge replaces two loops by one, and prefers loops that run alongside each other.
    #[test]
    fn merge_replaces_two_loops_by_one() {
        let path = std::env::var("DUALCUBE_TEST_MESH").map_or_else(
            |_| {
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../mehsh/assets/blub001k.obj")
            },
            std::path::PathBuf::from,
        );
        let mesh = Arc::new(Mesh::<INPUT>::from_obj(&path).unwrap().0);
        let mut solution = Solution::new(mesh);
        solution.initialize();
        let params = AnnealParams {
            steps: 15,
            ..AnnealParams::default()
        };
        let annealed = solution.anneal(&params, &OptimizationMonitor::default()).unwrap();
        let (mut tried, mut succeeded) = (0, 0);
        let timer = std::time::Instant::now();
        for _ in 0..30 {
            let mut child = annealed.clone_loop_state();
            child.dual = annealed.dual.clone();
            tried += 1;
            if child.merge().is_some() {
                succeeded += 1;
                assert_eq!(child.loops.len(), annealed.loops.len() - 1);
            }
        }
        println!(
            "merge: {succeeded}/{tried} valid ({} loops) in {:?}",
            annealed.loops.len(),
            timer.elapsed()
        );
    }
}
