//! Evolutionary optimization of the loop structure.
//!
//! The evolutionary algorithm of the paper (Section 4.3, Algorithm 1) with the following changes (each was found to
//! improve the results in ablation studies):
//!
//! - added loops are valid by construction: they follow a topological structure of the paper's filtered graph G^V
//!   (see `Solution::sample_valid_loop`), as in Section 4.1,
//! - parents are selected by rank among the best `PARENTS` solutions only (the i-th best of n with probability
//!   proportional to n - i + 1, as in Evocube), such that the population stays coherent,
//! - additions add one loop per axis at most, and are targeted: with probability `targeted`, a loop is added through
//!   a badly aligned triangle (sampled proportionally to its area times its flattening distortion), along one of the
//!   two axes that can create a patch with the label that the triangle needs, or (with probability `interest`)
//!   through a point of interest (a concentration of Gaussian curvature, such as the tip of a protrusion),
//! - a `feature` mutation adds intersecting loops of two or three axes through the same point at once,
//! - a `free loops` mutation adds the cheapest valid loop over all sequences of regions (see `EvolutionParams`),
//! - a `replace` mutation searches the cheapest path for a loop that crosses the same loops in the same order (it
//!   keeps the loop structure, and only moves the loop),
//! - a `reroute` mutation routes the worse half of a loop again, freely (crossing any loops of other axes, in any
//!   order; the loop structure may change),
//! - a `merge` mutation replaces two neighboring loops of the same axis by one loop that crosses everything that they
//!   cross (the union of their crossings), or the crossings of one of them, preferably for loops that run alongside
//!   each other (crossing the same loops, close together),
//! - removals only pick loops that keep the loop structure valid (checked locally, see `Dual::check_removal`),
//! - the mutations follow the phase (see `LoopPhase`): initialization (additions), growth (mostly additions, and moving
//!   loops), or optimization (moving, merging, and removing loops); when growth or optimization did not improve for
//!   `phase_patience` generations, pruning (removals and merges) takes over for `prune_generations` generations, and
//!   so on,
//! - offspring are screened with a cheap estimate of their quality (from the dual structure only), and only the
//!   best fraction `screen` is embedded (layout) and scored; duplicates (also across generations) are evaluated once,
//! - plus-selection: the next generation is the best mu distinct structures among parents and offspring.
//!
//! A running evolution can be followed and stopped from another thread with an [`EvolutionMonitor`] (see
//! `Solution::evolve_monitored`): statistics of every generation, the best solution so far, and a request to stop
//! after the current generation.
//!
//! The evolution only works on the base solution: loops on the input mesh, scored with a fast embedding (no
//! smoothing). The result is embedded without any optimization (smoothing, corner placement); those are applied
//! afterwards, on top of it. The result is never worse than the input: the input is scored with the same (fast)
//! embedding as the offspring, and if the final embedding of the best solution scores below the input, the input is
//! returned.

use crate::prelude::*;
use rand::seq::{IteratorRandom, SliceRandom};
use serde::{Deserialize, Serialize};
use slotmap::Key as _;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

/// The phases of the evolution of the loops: each uses its own mix of the mutations (see `LoopPhase::prior`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LoopPhase {
    /// Many new initial loop structures, each with a few generations of additions; the best is kept (see
    /// `Solution::optimize`).
    Initialization,
    /// Mostly additions, and moving loops (replace, reroute); sometimes merges.
    #[default]
    Growth,
    /// Moving loops (replace, reroute), merges, and removals; no additions.
    Optimization,
    /// Mostly removals, and merges. Not chosen, but an interlude of growth and optimization: it starts by itself when
    /// they did not improve for `EvolutionParams::phase_patience` generations, and lasts
    /// `EvolutionParams::prune_generations` generations.
    Pruning,
}

impl LoopPhase {
    pub const ALL: [Self; 4] = [
        Self::Initialization,
        Self::Growth,
        Self::Optimization,
        Self::Pruning,
    ];

    /// The phases that can be chosen (pruning starts by itself).
    pub const CHOSEN: [Self; 3] = [Self::Initialization, Self::Growth, Self::Optimization];

    /// The phase after this one when it stalls (see `LiveSettings::advance_after`): initialization, growth,
    /// optimization, and then growth and optimization in turn.
    #[must_use]
    pub const fn next(self) -> Self {
        match self {
            Self::Initialization | Self::Optimization => Self::Growth,
            Self::Growth | Self::Pruning => Self::Optimization,
        }
    }

    /// The index of the phase (in `ALL`).
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// The mutations switched on in this phase by default: those of its prior (a bit per mutation, by its index in
    /// `LOOP_MUTATIONS`).
    #[must_use]
    pub fn default_enabled(self) -> u16 {
        LOOP_MUTATIONS
            .iter()
            .enumerate()
            .filter(|(_, name)| self.prior().iter().any(|(mutation, _)| mutation == *name))
            .fold(0, |bits, (index, _)| bits | (1 << index))
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Initialization => "seeding",
            Self::Growth => "growth",
            Self::Optimization => "shaping",
            Self::Pruning => "pruning",
        }
    }

    // The prior weights of the mutations in this phase (the others are not used in it).
    const fn prior(self) -> &'static [(&'static str, f64)] {
        match self {
            Self::Initialization => &[("add", 0.6), ("feature", 0.2), ("free loops", 0.2)],
            Self::Growth => &[
                ("add", 0.35),
                ("feature", 0.1),
                ("free loops", 0.1),
                ("replace", 0.2),
                ("reroute", 0.2),
                ("merge", 0.05),
            ],
            Self::Optimization => &[
                ("replace", 0.35),
                ("reroute", 0.35),
                ("merge", 0.15),
                ("remove", 0.15),
            ],
            Self::Pruning => &[("remove", 0.6), ("merge", 0.4)],
        }
    }
}

/// The mutations of the loops, by their names in the statistics (see the module documentation).
pub const LOOP_MUTATIONS: [&str; 7] = [
    "add",
    "remove",
    "replace",
    "reroute",
    "merge",
    "feature",
    "free loops",
];

// The prior weight of a mutation that is switched on in a phase that does not use it by default.
const OFF_PHASE_WEIGHT: f64 = 0.1;

/// Parents are chosen among this many best solutions of the population (see `rank_select`).
pub(crate) const PARENTS: usize = 3;

/// Parameters of the evolutionary algorithm.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvolutionParams {
    /// Population size (mu).
    pub population: usize,
    /// Number of offspring per generation (lambda).
    pub offspring: usize,
    /// Maximum number of generations.
    pub max_generations: usize,
    /// Stop after this many generations without improvement (tau).
    pub patience: usize,
    /// Probability that an added loop is targeted at a badly aligned triangle or a point of interest.
    pub targeted: f64,
    /// Probability that a targeted loop starts at a point of interest instead of at a badly aligned triangle.
    pub interest: f64,
    /// The phase (see `LoopPhase`; pruning starts by itself).
    pub phase: LoopPhase,
    /// Switch from growth or optimization to pruning after this many generations without improvement.
    pub phase_patience: usize,
    /// Prune for this many generations, then go back (0: never prune).
    #[serde(default = "default_prune_generations")]
    pub prune_generations: usize,
    /// Fraction of the offspring (by estimated quality) that is embedded and scored.
    pub screen: f64,
    /// Per phase (by its index, see `LoopPhase::index`): the mutations that are switched on (a bit per mutation, by its
    /// index in `LOOP_MUTATIONS`). By default those of the phase (see `LoopPhase::default_enabled`).
    pub enabled: [u16; 4],
}

impl Default for EvolutionParams {
    fn default() -> Self {
        Self {
            population: 20,
            offspring: 60,
            max_generations: 150,
            // Progress comes in small steps with plateaus in between.
            patience: 20,
            targeted: 0.5,
            interest: 0.3,
            phase: LoopPhase::Growth,
            phase_patience: 2,
            prune_generations: default_prune_generations(),
            screen: 0.5,
            enabled: LoopPhase::ALL.map(LoopPhase::default_enabled),
        }
    }
}

const fn default_prune_generations() -> usize {
    3
}

/// The number of past generations whose improvements determine the chances of the mutations.
pub const SELECTION_WINDOW: usize = 30;
/// The smallest chance of an enabled mutation.
pub const MIN_CHANCE: f64 = 0.01;

/// Pseudo-tries (at the mean success rate) per mutation: mutations with few tries get about the mean success rate.
const SELECTION_PSEUDO_TRIES: f64 = 10.;

/// Adaptive choice of the mutations: the chance of a mutation is proportional to its prior weight (e.g., of the phase)
/// times its success rate (the fraction of its tries that improved the best solution) in the last `SELECTION_WINDOW`
/// generations, but at least `MIN_CHANCE`. The rate does not depend on how often a mutation was tried; mutations with
/// few tries are pulled towards the mean rate (`SELECTION_PSEUDO_TRIES`). Without improvements in the window, the
/// chances follow the prior weights.
#[derive(Default)]
pub struct MutationSelection {
    // Per generation: the tries and the improvements of the best solution per mutation.
    window: VecDeque<HashMap<&'static str, (usize, usize)>>,
}

impl MutationSelection {
    /// Record the tries and improvements of the best solution per mutation in a generation.
    pub fn record(&mut self, generation: HashMap<&'static str, (usize, usize)>) {
        self.window.push_back(generation);
        while self.window.len() > SELECTION_WINDOW {
            self.window.pop_front();
        }
    }

    /// The chances of the given mutations, with their tries and improvements in the window (for reporting).
    #[must_use]
    pub fn report(&self, chances: &[(&'static str, f64)]) -> Vec<(&'static str, MutationChance)> {
        chances
            .iter()
            .map(|&(name, chance)| {
                let (tries, successes) = self
                    .window
                    .iter()
                    .filter_map(|generation| generation.get(name))
                    .fold((0, 0), |(t, s), &(tries, successes)| {
                        (t + tries, s + successes)
                    });
                (
                    name,
                    MutationChance {
                        chance,
                        tries,
                        successes,
                    },
                )
            })
            .collect()
    }

    /// The chances of the mutations with a positive prior weight (the others are switched off).
    #[must_use]
    pub fn chances(&self, prior: &[(&'static str, f64)]) -> Vec<(&'static str, f64)> {
        let enabled = prior
            .iter()
            .filter(|(_, weight)| *weight > 0.)
            .copied()
            .collect_vec();
        if enabled.is_empty() {
            return vec![];
        }
        let counts = enabled
            .iter()
            .map(|&(name, _)| {
                self.window
                    .iter()
                    .filter_map(|generation| generation.get(name))
                    .fold((0., 0.), |(t, s), &(tries, successes)| {
                        (t + tries as f64, s + successes as f64)
                    })
            })
            .collect_vec();
        let (tries, successes) = counts.iter().fold((0., 0.), |(t, s), &(tries, successes)| {
            (t + tries, s + successes)
        });
        let weights = if successes > 0. {
            let mean = successes / tries;
            enabled
                .iter()
                .zip(&counts)
                .map(|(&(_, prior), &(t, s))| {
                    prior * (s + SELECTION_PSEUDO_TRIES * mean) / (t + SELECTION_PSEUDO_TRIES)
                })
                .collect_vec()
        } else {
            enabled.iter().map(|&(_, weight)| weight).collect_vec()
        };
        let total: f64 = weights.iter().sum();
        // At least `MIN_CHANCE` each (if that fits), the rest in proportion to the weights.
        let floor = MIN_CHANCE.min(1. / enabled.len() as f64);
        let rest = 1. - floor * enabled.len() as f64;
        enabled
            .iter()
            .zip(weights)
            .map(|(&(name, _), weight)| (name, floor + rest * weight / total))
            .collect()
    }
}

/// The chance of a mutation to be chosen, and its tries and improvements of the best in the last `SELECTION_WINDOW`
/// generations (on which the chance is based, see `MutationSelection`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MutationChance {
    pub chance: f64,
    pub tries: usize,
    pub successes: usize,
}

/// A random element, with a chance proportional to its weight (its penalty), plus a small floor
/// (`ELEMENT_FLOOR` of the mean weight) so that every element can still be chosen, but good ones hardly ever are.
// The distance (relative to their size) below which two loops count as close for a merge (see `Solution::merge`).
const MERGE_DISTANCE: f64 = 0.05;

// The mean distance from the points of one loop to the nearest point of the other (both ways), over the mean radius of
// the loops (their length over 2 pi).
fn relative_distance(a: &[Vector3D], b: &[Vector3D]) -> f64 {
    let nearest = |from: &[Vector3D], to: &[Vector3D]| {
        from.iter()
            .map(|p| to.iter().map(|q| (p - q).norm()).fold(f64::INFINITY, f64::min))
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
                let rotated = b.iter().cycle().skip(r).take(b.len()).copied().collect_vec();
                supersequence(first, &rotated)
            })
        })
        .min_by_key(Vec::len)
        .unwrap_or_else(|| first.to_vec())
}

pub(crate) fn pick_weighted<T: Copy>(items: impl IntoIterator<Item = (T, f64)>) -> Option<T> {
    const ELEMENT_FLOOR: f64 = 0.02;
    let items = items
        .into_iter()
        .map(|(item, weight)| {
            (
                item,
                if weight.is_finite() {
                    weight.max(0.)
                } else {
                    0.
                },
            )
        })
        .collect_vec();
    if items.is_empty() {
        return None;
    }
    let mean = items.iter().map(|(_, w)| w).sum::<f64>() / items.len() as f64;
    let floor = if mean > 0. { ELEMENT_FLOOR * mean } else { 1. };
    let total: f64 = items.iter().map(|(_, w)| w + floor).sum();
    let mut pick = rand::random::<f64>() * total;
    for &(item, weight) in &items {
        pick -= weight + floor;
        if pick <= 0. {
            return Some(item);
        }
    }
    items.last().map(|(item, _)| *item)
}

/// A random mutation with the given chances.
#[must_use]
pub fn choose_mutation(chances: &[(&'static str, f64)]) -> Option<&'static str> {
    let total: f64 = chances.iter().map(|(_, c)| c).sum();
    let mut pick = rand::random::<f64>() * total;
    for &(name, chance) in chances {
        pick -= chance;
        if pick <= 0. {
            return Some(name);
        }
    }
    chances.last().map(|(name, _)| *name)
}

impl EvolutionParams {
    /// Whether the given mutation is switched on in the given phase.
    #[must_use]
    pub fn is_enabled(&self, phase: LoopPhase, name: &str) -> bool {
        LOOP_MUTATIONS
            .iter()
            .position(|&mutation| mutation == name)
            .is_some_and(|index| self.enabled[phase.index()] & (1 << index) != 0)
    }

    /// Switch the given mutation on or off in the given phase.
    pub fn set_enabled(&mut self, phase: LoopPhase, name: &str, on: bool) {
        if let Some(index) = LOOP_MUTATIONS.iter().position(|&mutation| mutation == name) {
            if on {
                self.enabled[phase.index()] |= 1 << index;
            } else {
                self.enabled[phase.index()] &= !(1 << index);
            }
        }
    }

    // The prior weights of all mutations in the given phase: as in the phase (see `LoopPhase::prior`), or
    // `OFF_PHASE_WEIGHT` for one that is switched on but not used by the phase; 0 if switched off.
    fn mutation_prior(&self, phase: LoopPhase) -> Vec<(&'static str, f64)> {
        LOOP_MUTATIONS
            .iter()
            .map(|&name| {
                let weight = phase
                    .prior()
                    .iter()
                    .find(|(mutation, _)| *mutation == name)
                    .map_or(OFF_PHASE_WEIGHT, |&(_, weight)| weight);
                (name, if self.is_enabled(phase, name) { weight } else { 0. })
            })
            .collect()
    }

    // The parameters, population size, and number of offspring, with the changes made while running (if any).
    pub(crate) fn live(&self, monitor: &EvolutionMonitor) -> (Self, usize, usize) {
        let params = match monitor.settings() {
            Some(settings) => Self {
                population: settings.loops.population,
                offspring: settings.loops.offspring,
                phase: monitor.phase().unwrap_or(settings.phase),
                phase_patience: if settings.phase_patience > 0 {
                    settings.phase_patience
                } else {
                    self.phase_patience
                },
                prune_generations: settings.prune_generations,
                enabled: settings.loop_enabled.unwrap_or(self.enabled),
                ..*self
            },
            None => *self,
        };
        (params, params.population.max(1), params.offspring.max(1))
    }
}

/// Statistics of the population of one generation (after selection).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenerationStats {
    /// 0 for the initial population.
    pub generation: usize,
    pub best: f64,
    pub mean: f64,
    /// Number of loops of the best solution.
    pub loops: usize,
    /// The phase of the loops in which the generation was produced (for generations of the loops).
    pub phase: LoopPhase,
    /// Whether the generation is one of the layout optimization (else of the loops).
    pub layout: bool,
    /// Time since the start of the evolution.
    pub seconds: f64,
}

/// Settings of one kind of evolution (of the loops, or of the layout) that can be changed while it runs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PhaseSettings {
    /// Population size (mu).
    pub population: usize,
    /// Number of offspring per generation (lambda).
    pub offspring: usize,
    /// The mutations that are switched off, by their names in the statistics (see `EvolutionMonitor::mutation_stats`).
    pub disabled: std::collections::BTreeSet<String>,
}

/// Settings that can be changed while an optimization runs (see `EvolutionMonitor::set_settings`); they apply from the
/// next generation (the numbers of generations from the next cycle, see `Solution::optimize`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LiveSettings {
    pub loops: PhaseSettings,
    pub layout: PhaseSettings,
    /// Generations of the loops, and of the layout, per cycle.
    pub loop_generations: usize,
    pub layout_generations: usize,
    /// Switch from growth or optimization to pruning after this many generations without improvement (see
    /// `EvolutionParams::phase_patience`; 0: unchanged).
    pub phase_patience: usize,
    /// Prune for this many generations (see `EvolutionParams::prune_generations`).
    pub prune_generations: usize,
    /// The phase of the loops (see `LoopPhase`).
    pub phase: LoopPhase,
    /// Per phase: the mutations of the loops that are switched on (see `EvolutionParams::enabled`; `None`: unchanged).
    pub loop_enabled: Option<[u16; 4]>,
    /// Move on to the next phase (see `LoopPhase::next`) after this many generations (of the loops or the layout)
    /// without improvement of the best solution (0: never). The phase that is running is `EvolutionMonitor::phase`.
    pub advance_after: usize,
}

/// Follows and controls a running evolution from another thread (e.g., a GUI): the statistics of every generation,
/// the best solution so far (if `live`), changes of its settings, and a request to stop after the current generation.
/// Cheap to clone (shared).
#[derive(Clone, Default)]
pub struct EvolutionMonitor {
    inner: Arc<MonitorInner>,
}

#[derive(Default)]
struct MonitorInner {
    live: bool,
    stop: AtomicBool,
    state: Mutex<MonitorState>,
    settings: Mutex<Option<LiveSettings>>,
}

#[derive(Default)]
struct MonitorState {
    history: Vec<GenerationStats>,
    // Per mutation: generated, evaluated (screened and embedded), better than its parent, better than the best.
    mutations: std::collections::BTreeMap<&'static str, [usize; 4]>,
    // The current chance of every mutation to be chosen (see `MutationSelection`).
    // Per mutation: its current chance to be chosen, and its tries and improvements of the best in the window of the
    // adaptive selection (see `MutationSelection`).
    chances: std::collections::BTreeMap<&'static str, MutationChance>,
    // The solution shown: the best of the population of the current phase (embedded), with a version that increases
    // when it changes, and its quality.
    best: Option<(usize, Solution)>,
    shown_quality: Option<f64>,
    // The quality of the best solution so far (over all phases).
    best_quality: Option<f64>,
    // The adaptive selection of the mutations, per kind of evolution (kept over the phases of an optimization).
    selections: HashMap<&'static str, MutationSelection>,
    // When the first generation was recorded.
    started: Option<Instant>,
    // What the optimization is doing (e.g., the phase), for the user.
    activity: String,
    // Whether the loops are pruning, and the generations without improvement in that (or the other) phase: kept over
    // the evolutions of the loops of an optimization.
    pruning: (bool, usize),
    // Whether the layout is being optimized (else the loops).
    layout: bool,
    // The phase of the loops that is running, and the phase last chosen in the settings (see `advance_phase`).
    phase: Option<LoopPhase>,
    chosen: Option<LoopPhase>,
    // Generations (of the loops or the layout) since the best solution last improved.
    stale: usize,
    finished: bool,
}

impl EvolutionMonitor {
    /// A monitor that also keeps the best solution so far, embedded (for live views; costs an embedding per
    /// improvement).
    #[must_use]
    pub fn live() -> Self {
        Self {
            inner: Arc::new(MonitorInner {
                live: true,
                ..MonitorInner::default()
            }),
        }
    }

    /// Request the evolution to stop after the current generation (the best solution is then embedded and returned
    /// as usual).
    pub fn stop(&self) {
        self.inner.stop.store(true, Ordering::Relaxed);
    }

    #[must_use]
    pub fn stopping(&self) -> bool {
        self.inner.stop.load(Ordering::Relaxed)
    }

    /// Change the settings of the evolution (from its next generation; the parameters it was started with otherwise).
    pub fn set_settings(&self, settings: LiveSettings) {
        *self
            .inner
            .settings
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(settings);
    }

    pub(crate) fn settings(&self) -> Option<LiveSettings> {
        self.inner
            .settings
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Whether the evolution has finished (including the final embedding).
    #[must_use]
    pub fn finished(&self) -> bool {
        self.state().finished
    }

    /// The statistics of all generations so far.
    #[must_use]
    pub fn history(&self) -> Vec<GenerationStats> {
        self.state().history.clone()
    }

    /// Per mutation: how many were generated, evaluated (screened and embedded), better than their parent, and better
    /// than the best solution so far.
    #[must_use]
    pub fn mutation_stats(&self) -> Vec<(&'static str, [usize; 4])> {
        self.state()
            .mutations
            .iter()
            .map(|(&name, &counts)| (name, counts))
            .collect()
    }

    pub(crate) fn record_mutations(
        &self,
        stats: &std::collections::BTreeMap<&'static str, [usize; 4]>,
    ) {
        self.state().mutations = stats.clone();
    }

    // The chances of the mutations of one kind (those of the other kind are kept).
    pub(crate) fn record_chances(&self, chances: &[(&'static str, MutationChance)]) {
        self.state().chances.extend(chances.iter().copied());
    }

    /// The current chance of every mutation to be chosen (see `MutationSelection`).
    #[must_use]
    pub fn mutation_chances(&self) -> Vec<(&'static str, MutationChance)> {
        self.state()
            .chances
            .iter()
            .map(|(&name, &chance)| (name, chance))
            .collect()
    }

    /// The best solution so far and its version, if its version is newer than `version`.
    #[must_use]
    pub fn best_since(&self, version: usize) -> Option<(usize, Solution)> {
        self.state()
            .best
            .as_ref()
            .filter(|(v, _)| *v > version)
            .map(|(v, solution)| (*v, solution.clone()))
    }

    fn state(&self) -> MutexGuard<'_, MonitorState> {
        self.inner
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    // Record the statistics of a generation: numbered and timed over all phases of an optimization.
    pub(crate) fn record(&self, mut stats: GenerationStats) {
        let mut state = self.state();
        state.stale += 1;
        // In initialization, every new initial structure starts from scratch: only the best so far is reported.
        if stats.phase == LoopPhase::Initialization
            && !stats.layout
            && let Some(best) = state.best_quality
        {
            stats.best = stats.best.max(best);
            stats.mean = stats.best;
        }
        stats.generation = state.history.len();
        stats.seconds = state
            .started
            .get_or_insert_with(Instant::now)
            .elapsed()
            .as_secs_f64();
        state.history.push(stats);
    }

    // The statistics per mutation so far (an evolution continues counting).
    pub(crate) fn mutation_stats_map(
        &self,
    ) -> std::collections::BTreeMap<&'static str, [usize; 4]> {
        self.state().mutations.clone()
    }

    // A phase starts: show the best of its population (if live), and from then on every better solution of the phase
    // (see `offer_best`). The solution is only made if it is shown.
    pub(crate) fn begin_phase(&self, quality: f64, solution: impl FnOnce() -> Solution) {
        self.note_best(quality);
        if self.inner.live {
            self.show(quality, solution());
        }
    }

    // Show the given solution (if live) if it is better than the solution shown (the best of the population of the
    // current phase). The solution is only made then.
    pub(crate) fn offer_best(&self, quality: f64, solution: impl FnOnce() -> Solution) {
        self.note_best(quality);
        if !self.inner.live
            || self
                .state()
                .shown_quality
                .is_some_and(|shown| quality <= shown)
        {
            return;
        }
        self.show(quality, solution());
    }

    // Show the given solution (if live), whatever its quality: the solution that the optimization continues from.
    pub(crate) fn show_now(&self, quality: f64, solution: impl FnOnce() -> Solution) {
        self.note_best(quality);
        if self.inner.live {
            self.show(quality, solution());
        }
    }

    fn show(&self, quality: f64, solution: Solution) {
        let mut state = self.state();
        state.shown_quality = Some(quality);
        let version = state.best.as_ref().map_or(0, |(v, _)| *v) + 1;
        state.best = Some((version, solution));
    }

    // A solution of the given quality was found (the best quality so far is reported).
    pub(crate) fn note_best(&self, quality: f64) {
        let mut state = self.state();
        if state.best_quality.is_none_or(|best| quality > best + 1e-12) {
            state.stale = 0;
        }
        state.best_quality = Some(state.best_quality.map_or(quality, |best| best.max(quality)));
    }

    /// The phase of the loops that is running (see `LiveSettings::advance_after`), if it was set.
    #[must_use]
    pub fn phase(&self) -> Option<LoopPhase> {
        self.state().phase
    }

    // The phase to run (at the start of a cycle of an optimization): the chosen one if it changed, else the next phase
    // if the best solution did not improve for `advance_after` generations, else the running one.
    pub(crate) fn advance_phase(&self, chosen: LoopPhase, advance_after: usize) -> LoopPhase {
        let mut state = self.state();
        let next = match state.phase {
            Some(phase) if state.chosen == Some(chosen) => {
                if advance_after > 0 && state.stale >= advance_after {
                    info!("optimize: no improvement for {} generations: next phase", state.stale);
                    phase.next()
                } else {
                    phase
                }
            }
            _ => chosen,
        };
        if state.phase != Some(next) {
            state.stale = 0;
            state.pruning = (next == LoopPhase::Pruning, 0);
        }
        state.phase = Some(next);
        state.chosen = Some(chosen);
        next
    }

    /// The quality of the best solution so far (over all phases).
    #[must_use]
    pub fn best_quality(&self) -> Option<f64> {
        self.state().best_quality
    }

    pub(crate) fn take_selection(&self, kind: &'static str) -> MutationSelection {
        self.state().selections.remove(kind).unwrap_or_default()
    }

    pub(crate) fn put_selection(&self, kind: &'static str, selection: MutationSelection) {
        self.state().selections.insert(kind, selection);
    }

    /// What the optimization is doing (shown with its progress).
    pub fn set_activity(&self, activity: String) {
        self.state().activity = activity;
    }

    /// What the optimization is doing (e.g., its phase and cycle).
    #[must_use]
    pub fn activity(&self) -> String {
        self.state().activity.clone()
    }

    // Whether the loops are pruning, and the generations without improvement since the phase started (see
    // `MonitorState::pruning`).
    pub(crate) fn pruning_state(&self) -> (bool, usize) {
        self.state().pruning
    }

    pub(crate) fn set_pruning_state(&self, pruning: bool, stale: usize) {
        self.state().pruning = (pruning, stale);
    }

    /// Whether the loops are in the pruning phase (also when it started by itself, see `LoopPhase::Pruning`).
    #[must_use]
    pub fn pruning(&self) -> bool {
        self.state().pruning.0
    }

    /// Whether the optimization is working on the layout (else on the loops).
    #[must_use]
    pub fn layout_active(&self) -> bool {
        self.state().layout
    }

    pub(crate) fn set_layout_active(&self, layout: bool) {
        self.state().layout = layout;
    }

    /// Mark the optimization as finished (it is no longer running).
    pub fn finish(&self) {
        self.state().finished = true;
    }
}

fn generation_stats(
    generation: usize,
    population: &[(Solution, f64)],
    phase: LoopPhase,
    started_at: Instant,
) -> GenerationStats {
    let qualities = population.iter().map(|(_, q)| *q).collect_vec();
    let n = qualities.len().max(1) as f64;
    let mean = qualities.iter().sum::<f64>() / n;
    GenerationStats {
        generation,
        best: qualities.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        mean,
        loops: population.first().map_or(0, |(s, _)| s.loops.len()),
        phase,
        layout: false,
        seconds: started_at.elapsed().as_secs_f64(),
    }
}

// CPU time (summed over all threads) per phase of the evolution, reported at the end of a run (debug log).
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
    /// Evolve the loop structure (see the module documentation). The solution is scored with its quality criterion
    /// (`self.quality`).
    pub fn evolve(&self, params: &EvolutionParams) -> Result<Self, SolutionError> {
        self.evolve_monitored(params, &EvolutionMonitor::default())
    }

    /// Evolve the loop structure, reporting to (and stoppable through) the given monitor.
    pub fn evolve_monitored(
        &self,
        params: &EvolutionParams,
        monitor: &EvolutionMonitor,
    ) -> Result<Self, SolutionError> {
        let result = self
            .evolve_phase(params, monitor)
            .map(|result| self.better_of(result));
        monitor.finish();
        result
    }

    // An embedded copy of this solution with the loops of `candidate` (fast layout, for live views).
    fn embedded_copy(&self, candidate: &Self) -> Self {
        let mut copy = self.clone();
        copy.loops = candidate.loops.clone();
        copy.occupied = candidate.occupied.clone();
        if copy
            .reconstruct_near(true, LayoutParams::fast(), candidate.corner_hint.as_deref())
            .is_err()
        {
            warn!("evolve: embedding the best solution for the live view failed");
        }
        copy
    }

    // The evolution of the loops (see `evolve`): its best solution, embedded, even if it is worse than the input.
    pub(crate) fn evolve_phase(
        &self,
        params: &EvolutionParams,
        monitor: &EvolutionMonitor,
    ) -> Result<Self, SolutionError> {
        let (mut live, mut mu, mut lambda) = params.live(monitor);
        let Some(initial_quality) = self.get_quality() else {
            return Err(SolutionError::NoPrimal);
        };
        let started_at = Instant::now();
        profile::reset();
        // Show the input (with its layout) until a better solution is found; building the initial population takes a
        // while. (In initialization, the input is a new initial structure: shown only if it is the best so far.)
        if live.phase == LoopPhase::Initialization {
            monitor.offer_best(initial_quality, || self.clone());
        } else {
            monitor.begin_phase(initial_quality, || self.clone());
        }
        let mut seed = self.clone();
        seed.prepare_flow();
        // Score the seed like the offspring (fast embedding near the corners of its layout), such that they are
        // compared fairly; its layout (e.g., optimized) is largely kept.
        let mut seed_quality = initial_quality;
        let mut fast_seed = seed.clone();
        let hint = CornerHint::of(&seed);
        if fast_seed
            .reconstruct_near(true, LayoutParams::fast(), hint.as_ref())
            .is_ok()
            && let Some(quality) = fast_seed.get_quality()
        {
            seed = fast_seed;
            seed_quality = quality;
        }

        seed.compact_for_evolution(true);
        let context = MutationContext::new(&seed);
        let seed_signature = seed.loop_signature();
        let mut seen: HashSet<LoopSignature> = HashSet::from([seed_signature.clone()]);
        let mut population = vec![(seed, seed_quality)];
        let mut selection = monitor.take_selection("loops");
        monitor.note_best(initial_quality);

        // The phase: as chosen, or pruning (kept over the evolutions of an optimization).
        let (mut pruning, mut phase_stale) = monitor.pruning_state();
        if live.phase == LoopPhase::Initialization {
            pruning = false;
        }
        let switches = |live: &EvolutionParams| {
            matches!(live.phase, LoopPhase::Growth | LoopPhase::Optimization)
        };
        let phase_of = |live: &EvolutionParams, pruning: bool| {
            if pruning {
                LoopPhase::Pruning
            } else {
                live.phase
            }
        };

        // Initial population: the seed and mutations of it.
        let chances = selection.chances(&live.mutation_prior(phase_of(&live, pruning)));
        for _ in 0..3 {
            if population.len() >= mu {
                break;
            }
            let parent = &population[0].0;
            let children = (0..(mu - population.len()) * 2)
                .collect_vec()
                .into_par()
                .filter_map(|_| {
                    parent
                        .mutate(&live, &context, choose_mutation(&chances)?)
                        .map(|(child, op)| (child, (op, f64::NEG_INFINITY)))
                })
                .collect::<Vec<_>>();
            let evaluated =
                Self::evaluate_offspring(children, params, &mut seen, f64::NEG_INFINITY);
            population.extend(
                evaluated
                    .into_iter()
                    .map(|(child, q, _)| (child, q))
                    .take(mu - population.len()),
            );
        }
        population.sort_by(|(_, a), (_, b)| b.total_cmp(a));

        let mut best = population[0].1;
        let mut stale = 0;
        // Per operation: generated, evaluated (screened and embedded), better than its parent, better than the best.
        let mut stats = monitor.mutation_stats_map();
        info!(
            "evolve: start mu={mu} lambda={lambda} population={} initial_quality={initial_quality}",
            population.len()
        );
        monitor.record(generation_stats(0, &population, phase_of(&live, pruning), started_at));
        // Show the best of the population if it is better than the input (scored with its own layout).
        monitor.offer_best(population[0].1, || self.embedded_copy(&population[0].0));

        for generation in 0..params.max_generations {
            let timer = Instant::now();
            (live, mu, lambda) = params.live(monitor);
            if !switches(&live) {
                pruning = live.phase == LoopPhase::Pruning;
            }
            let phase = phase_of(&live, pruning);
            // Parents among the best (see `PARENTS`).
            let n = population.len().min(PARENTS);
            let chances = selection.chances(&live.mutation_prior(phase));
            // (All mutations are reported: those not used in this phase with chance 0.)
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
            let mutation_timer = Instant::now();
            // Every try: the chosen mutation, and the child (if the mutation succeeded).
            let attempts = (0..lambda)
                .collect_vec()
                .into_par()
                .filter_map(|_| {
                    let (parent, parent_quality) = &population[rank_select(n)];
                    let op = choose_mutation(&chances)?;
                    let child = profile::timed(3, || parent.mutate(&live, &context, op))
                        .map(|(child, op)| (child, (op, *parent_quality)));
                    Some((op, child))
                })
                .collect::<Vec<_>>();
            // Tries and improvements of the best per mutation (see `MutationSelection`); failed tries count too.
            let mut tries: HashMap<&'static str, (usize, usize)> = HashMap::new();
            for (op, child) in &attempts {
                let op = child.as_ref().map_or(*op, |(_, (op, _))| *op);
                tries.entry(op).or_default().0 += 1;
            }
            let children = attempts
                .into_iter()
                .filter_map(|(_, child)| child)
                .collect_vec();
            let mutation_time = mutation_timer.elapsed();
            let generated = children.len();
            for (_, (op, _)) in &children {
                stats.entry(op).or_default()[0] += 1;
            }
            // Plus-selection keeps the best mu: a child below the worst of a full population does not enter it.
            let threshold = if population.len() >= mu {
                population.last().map_or(f64::NEG_INFINITY, |(_, q)| *q)
            } else {
                f64::NEG_INFINITY
            };
            let offspring = Self::evaluate_offspring(children, params, &mut seen, threshold);
            for (_, quality, (op, parent_quality)) in &offspring {
                let entry = stats.entry(op).or_default();
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
            let evaluated = offspring.len();
            let best_child = offspring.iter().map(|(_, q, _)| *q).max_by(f64::total_cmp);

            // Plus-selection: the best mu (distinct) of parents and offspring.
            population.extend(offspring.into_iter().map(|(child, q, _)| (child, q)));
            population.sort_by(|(_, a), (_, b)| b.total_cmp(a));
            population.truncate(mu);

            monitor.record_mutations(&stats);
            monitor.record(generation_stats(
                generation + 1,
                &population,
                phase,
                started_at,
            ));
            let current = population[0].1;
            // While pruning, `phase_stale` counts its generations; otherwise those without improvement.
            if current > best + 1e-12 {
                best = current;
                stale = 0;
                phase_stale = if pruning { phase_stale + 1 } else { 0 };
                monitor.offer_best(current, || self.embedded_copy(&population[0].0));
            } else {
                stale += 1;
                phase_stale += 1;
            }
            let limit = if pruning {
                live.prune_generations
            } else {
                live.phase_patience
            };
            if switches(&live)
                && (pruning || live.prune_generations > 0)
                && phase_stale >= limit.max(1)
            {
                pruning = !pruning;
                phase_stale = 0;
                let next = phase_of(&live, pruning);
                info!("evolve: switching to the {} phase", next.name());
                monitor.set_activity(monitor.activity().replace(phase.name(), next.name()));
            }
            info!(
                "evolve: generation {} generated={generated} evaluated={evaluated} best_child={best_child:?} best={best} loops={} stale={stale} elapsed={:?} (mutations {mutation_time:?})",
                generation + 1,
                population[0].0.loops.len(),
                timer.elapsed()
            );
            if stale >= params.patience {
                break;
            }
            if monitor.stopping() {
                info!("evolve: stopped after generation {}", generation + 1);
                break;
            }
        }

        monitor.put_selection("loops", selection);
        // (Only growth and optimization switch to pruning by themselves.)
        if switches(&live) {
            monitor.set_pruning_state(pruning, phase_stale);
        } else {
            monitor.set_pruning_state(live.phase == LoopPhase::Pruning, 0);
        }
        info!(
            "evolve: mutation statistics (generated, evaluated, better than parent, new best): {stats:?}"
        );
        // ("mutate (other)" includes the search, placement, and dual structure of the added loops.)
        info!("evolve: CPU time per phase: {}", profile::report());
        info!(
            "evolve: picked best solution with quality {} and {} loops after {:?}",
            population[0].1,
            population[0].0.loops.len(),
            started_at.elapsed()
        );
        // Embed the best solution (without optimizations). The full embedding may fail where the fast one succeeded;
        // then fall back to more attempts, the fast embedding, and finally the next best solution.
        for (candidate, _) in population {
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
                    return Ok(result);
                }
                warn!("evolve: embedding the result failed with {layout_params:?}; retrying");
            }
        }
        Ok(self.clone())
    }

    // The result of an optimization, unless it scores worse than this (input) solution.
    fn better_of(&self, result: Self) -> Self {
        match (self.get_quality(), result.get_quality()) {
            (Some(input), Some(output)) if output < input => {
                info!(
                    "evolve: result ({output}) is worse than the input ({input}); keeping the input"
                );
                self.clone()
            }
            (Some(_), None) => self.clone(),
            _ => result,
        }
    }

    // Embed and score the (new) offspring: duplicates are skipped, the rest is screened with an estimate first.
    // Children scoring below `threshold` cannot enter the population, so their targets (for targeted mutations, see
    // `compact_for_evolution`) are not computed.
    fn evaluate_offspring<T: Send + Sync + Copy>(
        children: Vec<(Self, T)>,
        params: &EvolutionParams,
        seen: &mut HashSet<LoopSignature>,
        threshold: f64,
    ) -> Vec<(Self, f64, T)> {
        let children = children
            .into_iter()
            .filter(|(child, _)| seen.insert(child.loop_signature()))
            .collect_vec();
        let children = if params.screen < 1. {
            let estimated = children
                .into_par()
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
            estimated
                .into_iter()
                .sorted_by(|(_, _, a), (_, _, b)| b.total_cmp(a))
                .take(keep)
                .map(|(child, tag, _)| (child, tag))
                .collect_vec()
        } else {
            children
        };
        children
            .into_par()
            .filter_map(|(mut child, tag)| {
                let timer = Instant::now();
                // Near the corners of the parent (see `CornerHint`).
                let hint = child.corner_hint.clone();
                let result = profile::timed(5, || {
                    child.reconstruct_near(true, LayoutParams::fast(), hint.as_deref())
                });
                if timer.elapsed().as_secs_f64() > 5. {
                    warn!(
                        "evolve: slow evaluation ({:?}) of a child with {} loops: {result:?}",
                        timer.elapsed(),
                        child.loops.len()
                    );
                }
                result.ok()?;
                let quality = profile::timed(6, || child.get_quality())?;
                profile::timed(7, || child.compact_for_evolution(quality >= threshold));
                Some((child, quality, tag))
            })
            .collect()
    }

    // A mutation (see the module documentation): replace, feature, removal (with the given probability), or addition.
    // The given mutation (see `MutationSelection`); an addition if it does not apply (e.g., too few loops).
    fn mutate(
        &self,
        params: &EvolutionParams,
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
        if enough && op == "reroute" {
            child.dual = self.dual.clone();
            return child.reroute(self).map(|()| (child, "reroute"));
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

        if op == "free loops" {
            // A free loop (see `EvolutionParams::free`) of a random axis.
            let axis = DIRECTIONS[rand::random_range(0..3)];
            let targeted = rand::random::<f64>() < params.targeted;
            return child
                .add_new_loop(self, axis, targeted, true, params, context)
                .then_some((child, "free loops"));
        }

        if enough && op == "remove" {
            child.remove_valid_loop(self)?;
            debug!("mutation: operation=remove");
            return Some((child, "remove"));
        }

        if op != "add" {
            return None;
        }
        // Addition: zero or one loop per axis (at least one loop in total).
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
        debug!("mutation: operation=add counts={counts:?} added={added}");
        (added > 0).then_some((child, "add"))
    }

    // Add a valid loop of the given axis (from the given edge, if any), and update the dual structure. Returns whether
    // a loop was added. Free: over all sequences of regions (see `EvolutionParams::free`).
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
            let mut pairs: HashMap<(LoopID, LoopID), Vec<LoopRegionID>> = HashMap::new();
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
        let mut positions: HashMap<LoopID, Vec<Vector3D>> = HashMap::new();
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
        let order = if rand::random::<bool>() { [a, b] } else { [b, a] };
        let sequence_of = |keep: LoopID| {
            let list = &crossings[&keep];
            (1..=list.len()).map(|i| list[i % list.len()].1).collect_vec()
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
                debug!("mutation: operation=merge axis={axis:?} crossings={}", sequence.len());
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

    // Reroute: route the worse half of a loop again, freely. A loop is chosen by its penalty (see `loop_penalty`), and cut
    // in two halves (by its faces); the half to route again is chosen with probability proportional to its penalty
    // (by the penalties of its faces, see `compact_for_evolution`). The cheapest path of the loop's axis between the ends
    // of the other half replaces it: it crosses any loops of other axes in any order, and never runs through the faces
    // of the kept half (see `Solution::construct_open_path`). The loop structure may change; the result is checked.
    fn reroute(&mut self, parent: &Self) -> Option<()> {
        let loop_id = parent.badly_aligned_loop(&parent.loops.keys().collect_vec())?;
        let old = parent.loops[loop_id].clone();
        let mesh = self.mesh_ref.clone();
        let len = old.edges.len();
        if old.offsets.len() != len {
            return None;
        }
        // The half-edges through which the loop enters a face (one per face).
        let entries = (0..len)
            .filter(|&i| mesh.twin(old.edges[(i + len - 1) % len]) == old.edges[i])
            .collect_vec();
        let m = entries.len();
        if m < 6 {
            return None;
        }
        let half = m / 2;
        let penalties: HashMap<FaceID, f64> = parent
            .targets
            .as_ref()
            .map(|targets| targets.iter().copied().collect())
            .unwrap_or_default();
        let face_penalty = entries
            .iter()
            .map(|&i| penalties.get(&mesh.face(old.edges[i])).copied().unwrap_or(0.))
            .collect_vec();
        // The halves: the faces from the k-th on, and their penalties (a sliding window).
        let mut window: f64 = face_penalty[..half].iter().sum();
        let mut halves = Vec::with_capacity(m);
        for k in 0..m {
            halves.push((k, window));
            window += face_penalty[(k + half) % m] - face_penalty[k];
        }
        let k = pick_weighted(halves)?;
        // The loop runs again from the first face of the cut half (`cut`) to the first face of the kept half (`kept`).
        let (cut, kept) = (entries[k], entries[(k + half) % m]);
        let kept_indices = (0..(cut + len - kept) % len)
            .map(|i| (kept + i) % len)
            .collect_vec();
        self.del_loop(loop_id);
        // The gap of the old loop along a half-edge, among the other loops.
        let gap = |solution: &Self, index: usize| {
            let edge = old.edges[index];
            solution
                .loops_on_edge(edge)
                .iter()
                .filter(|&&other| {
                    solution.loops[other]
                        .offset(edge)
                        .is_some_and(|offset| offset < old.offsets[index])
                })
                .count()
        };
        let from = (old.edges[cut], gap(self, cut));
        let to = (old.edges[kept], gap(self, kept));
        let avoid: HashSet<FaceID> = kept_indices
            .iter()
            .map(|&i| mesh.face(old.edges[i]))
            .collect();
        let (path, _, path_gaps) = self.construct_open_path(old.direction, from, to, &avoid)?;
        let cut_edges = (0..(kept + len - cut) % len)
            .map(|i| old.edges[(cut + i) % len])
            .collect_vec();
        if path == cut_edges {
            return None;
        }
        // The kept half (with its gaps), then the new path.
        let mut edges = kept_indices.iter().map(|&i| old.edges[i]).collect_vec();
        let mut gaps = kept_indices
            .iter()
            .enumerate()
            .filter(|&(_, &i)| mesh.twin(old.edges[i]) == old.edges[(i + 1) % len])
            .map(|(position, &i)| (position, gap(self, i)))
            .collect_vec();
        let offset = edges.len();
        edges.extend(path);
        gaps.extend(path_gaps.into_iter().map(|(i, g)| (i + offset, g)));
        self.check_loop(&edges).ok()?;
        let new_id = self.add_loop_in_gaps(Loop::new(edges, old.direction), gaps);
        match Dual::from(self.mesh_ref.clone(), &self.loops) {
            Ok(dual) => {
                self.dual = Ok(dual);
                debug!("mutation: operation=reroute axis={:?}", old.direction);
                Some(())
            }
            Err(_) => {
                self.del_loop(new_id);
                None
            }
        }
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
        let mut sums: HashMap<LoopID, (f64, usize)> = HashMap::new();
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
        params: &EvolutionParams,
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

    // Keep only what the evolution needs of an evaluated solution: its loops, dual structure (without the cached refined
    // mesh), and the misalignment per input face (for targeted mutations). The layout is dropped to save memory.
    fn compact_for_evolution(&mut self, targets: bool) {
        if targets && let Some(layout) = &self.layout {
            self.targets = Some(Arc::new(penalty_per_input_face(
                layout,
                &self.quality.weights,
            )));
            self.loop_penalties = self.compute_loop_penalties().map(Arc::new);
            // Its children start their layouts from its corners.
            self.corner_hint = CornerHint::of(self).map(Arc::new);
        }
        self.layout = None;
        self.polycube = None;
        self.quad = None;
        self.fields = None;
        if let Ok(dual) = &mut self.dual {
            dual.release_refined_mesh();
        }
    }

    pub(crate) fn clone_loop_state(&self) -> Self {
        // Evolution mutates only loop bookkeeping. Avoid cloning any derived
        // representations; they are rebuilt once the candidate needs scoring.
        // `flow_graphs` is Arc-backed, so this only bumps a refcount and lets
        // evolved candidates keep sampling loops in later iterations.
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

// Rank-based selection (as in Evocube): the i-th best of n is chosen with probability proportional to n - i.
fn rank_select(n: usize) -> usize {
    let total = n * (n + 1) / 2;
    let mut pick = rand::random_range(0..total.max(1));
    for i in 0..n {
        let weight = n - i;
        if pick < weight {
            return i;
        }
        pick -= weight;
    }
    0
}

#[cfg(test)]
mod selection_tests {
    use super::{MIN_CHANCE, MutationSelection};
    use std::collections::HashMap;

    #[test]
    fn chances_follow_success_rates_with_a_floor() {
        let prior = [("a", 0.5), ("b", 0.3), ("c", 0.2), ("off", 0.)];
        let mut selection = MutationSelection::default();
        // Without improvements: the prior.
        let chances = selection.chances(&prior);
        assert_eq!(chances.len(), 3);
        assert!(chances[0].1 > chances[1].1 && chances[1].1 > chances[2].1);
        // `a` tried often with few successes, `b` rarely but with a high rate, `c` never succeeds.
        for _ in 0..30 {
            selection.record(HashMap::from([
                ("a", (20, 1)),
                ("b", (2, 1)),
                ("c", (5, 0)),
            ]));
        }
        let chances: HashMap<_, _> = selection.chances(&prior).into_iter().collect();
        assert!(chances["b"] > chances["a"], "{chances:?}");
        assert!(chances["c"] >= MIN_CHANCE - 1e-12, "{chances:?}");
        assert!((chances.values().sum::<f64>() - 1.).abs() < 1e-9);
        assert!(!chances.contains_key("off"));
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
            |_| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../mehsh/assets/blub001k.obj"),
            std::path::PathBuf::from,
        );
        let mesh = Arc::new(Mesh::<INPUT>::from_obj(&path).unwrap().0);
        let mut solution = Solution::new(mesh);
        solution.initialize();
        let params = EvolutionParams {
            max_generations: 15,
            ..EvolutionParams::default()
        };
        let evolved = solution.evolve(&params).unwrap();
        let (mut tried, mut succeeded) = (0, 0);
        let timer = std::time::Instant::now();
        for _ in 0..30 {
            let mut child = evolved.clone_loop_state();
            child.dual = evolved.dual.clone();
            tried += 1;
            if child.merge().is_some() {
                succeeded += 1;
                assert_eq!(child.loops.len(), evolved.loops.len() - 1);
            }
        }
        println!(
            "merge: {succeeded}/{tried} valid ({} loops) in {:?}",
            evolved.loops.len(),
            timer.elapsed()
        );
    }

    // A reroute keeps the number of loops, and keeps the structure valid (checked by `Dual::from`).
    #[test]
    fn reroute_routes_part_of_a_loop_again() {
        let path = std::env::var("DUALCUBE_TEST_MESH").map_or_else(
            |_| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../mehsh/assets/blub001k.obj"),
            std::path::PathBuf::from,
        );
        let mesh = Arc::new(Mesh::<INPUT>::from_obj(&path).unwrap().0);
        let mut solution = Solution::new(mesh);
        solution.initialize();
        let params = EvolutionParams {
            max_generations: 15,
            ..EvolutionParams::default()
        };
        let evolved = solution.evolve(&params).unwrap();
        let (mut tried, mut succeeded) = (0, 0);
        let timer = std::time::Instant::now();
        for _ in 0..40 {
            let mut child = evolved.clone_loop_state();
            child.dual = evolved.dual.clone();
            tried += 1;
            if child.reroute(&evolved).is_some() {
                succeeded += 1;
                assert_eq!(child.loops.len(), evolved.loops.len());
                assert!(child.dual.is_ok());
            }
        }
        println!(
            "reroute: {succeeded}/{tried} valid ({} loops) in {:?}",
            evolved.loops.len(),
            timer.elapsed()
        );
    }
}
