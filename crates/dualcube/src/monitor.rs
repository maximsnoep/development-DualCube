//! Following and controlling a running optimization (see `Solution::optimize`) from another thread.

use crate::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

/// The number of chains of the annealing whose best quality is recorded (see `GenerationStats::chains`).
pub const SHOWN_CHAINS: usize = 4;

/// Statistics of one generation (of the layout) or step (of the annealing of the loops).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenerationStats {
    /// 0 for the start.
    pub generation: usize,
    /// The best quality so far (of the optimization, see `OptimizationMonitor::best_quality`; set when recorded), and
    /// that of the current solution.
    pub best: f64,
    pub current: f64,
    /// Number of loops (or of corners, for the layout) of the current solution.
    pub loops: usize,
    /// Whether the generation is one of the layout optimization (else of the loops).
    pub layout: bool,
    /// Time since the start of the optimization.
    pub seconds: f64,
    /// The temperature of a step of the simulated annealing (see `Solution::anneal`; else none).
    pub temperature: Option<f64>,
    /// The best quality that every chain of the annealing found so far in it (the first `SHOWN_CHAINS` chains; else
    /// none).
    pub chains: [Option<f64>; SHOWN_CHAINS],
}

/// Settings that can be changed while an optimization runs (see `OptimizationMonitor::set_settings`); they apply from
/// the next step or generation (the numbers of them from the next cycle, see `Solution::optimize`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LiveSettings {
    /// The mutations of the loops, and of the layout, that are switched off, by their names in the statistics (see
    /// `OptimizationMonitor::mutation_stats`).
    pub loops_disabled: std::collections::BTreeSet<String>,
    pub layout_disabled: std::collections::BTreeSet<String>,
    /// Steps of the annealing of the loops, and generations of the layout, per cycle.
    pub steps: usize,
    pub layout_generations: usize,
    /// Cycles without a new best solution after which the optimization stops (0: never, see `CoupledParams`).
    pub patience: usize,
}

/// Follows and controls a running optimization from another thread (e.g., a GUI): the statistics of every step and
/// generation, the best solution so far (if `live`), changes of its settings, and a request to stop after the current
/// step or generation.
/// Cheap to clone (shared).
#[derive(Clone, Default)]
pub struct OptimizationMonitor {
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
    // Per mutation: the total time of its tries (making the child, without scoring it), and their number.
    mutation_times: std::collections::BTreeMap<&'static str, (f64, usize)>,
    // Per mutation: its current chance to be chosen, and its tries and improvements of the best in the window of the
    // adaptive selection (see `MutationSelection`).
    chances: std::collections::BTreeMap<&'static str, MutationChance>,
    // The solution shown: the best so far (embedded), with a version that increases when it changes, and its quality.
    best: Option<(usize, Solution)>,
    shown_quality: Option<f64>,
    // The quality of the best solution so far.
    best_quality: Option<f64>,
    // The adaptive selection of the mutations, per kind (loops or layout; kept over the cycles of an optimization).
    selections: HashMap<&'static str, MutationSelection>,
    // When the first generation was recorded.
    started: Option<Instant>,
    // Whether the layout is being optimized (else the loops).
    layout: bool,
    finished: bool,
}

impl OptimizationMonitor {
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

    /// Request the optimization to stop after the current step or generation (the best solution is then embedded and
    /// returned as usual).
    pub fn stop(&self) {
        self.inner.stop.store(true, Ordering::Relaxed);
    }

    #[must_use]
    pub fn stopping(&self) -> bool {
        self.inner.stop.load(Ordering::Relaxed)
    }

    /// Change the settings of the optimization (from its next step; the parameters it was started with otherwise).
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

    /// Whether the optimization has finished (including the final embedding).
    #[must_use]
    pub fn finished(&self) -> bool {
        self.state().finished
    }

    /// The statistics of all steps and generations so far.
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

    /// The mean time of a try of every mutation so far (making the child, without scoring it), in milliseconds.
    #[must_use]
    pub fn mutation_times(&self) -> Vec<(&'static str, f64)> {
        self.state()
            .mutation_times
            .iter()
            .map(|(&name, &(seconds, count))| (name, 1e3 * seconds / count.max(1) as f64))
            .collect()
    }

    // Add the times of tries of mutations (see `mutation_times`).
    pub(crate) fn record_mutation_times(
        &self,
        times: impl IntoIterator<Item = (&'static str, std::time::Duration)>,
    ) {
        let mut state = self.state();
        for (name, time) in times {
            let entry = state.mutation_times.entry(name).or_default();
            entry.0 += time.as_secs_f64();
            entry.1 += 1;
        }
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

    // Record the statistics of a step or generation: numbered and timed over all cycles of an optimization.
    pub(crate) fn record(&self, mut stats: GenerationStats) {
        let mut state = self.state();
        stats.generation = state.history.len();
        stats.best = state.best_quality.unwrap_or(stats.current);
        stats.seconds = state
            .started
            .get_or_insert_with(Instant::now)
            .elapsed()
            .as_secs_f64();
        state.history.push(stats);
    }

    // The statistics per mutation so far (every cycle continues counting).
    pub(crate) fn mutation_stats_map(
        &self,
    ) -> std::collections::BTreeMap<&'static str, [usize; 4]> {
        self.state().mutations.clone()
    }

    // Show the given solution (if live) if it is better than the solution shown, the best so far (so the live view only
    // changes when the best solution improves). The solution is only made then.
    pub(crate) fn offer_best(&self, quality: f64, solution: impl FnOnce() -> Solution) {
        self.offer(quality, false, solution);
    }

    // Show the new best result of the optimization (if live), also if it is as good as the solution shown (e.g., the
    // same layout with its loops moved, see `Solution::medial_loops`): the live view shows what stopping returns.
    pub(crate) fn show_result(&self, quality: f64, solution: impl FnOnce() -> Solution) {
        self.offer(quality, true, solution);
    }

    fn offer(&self, quality: f64, equal: bool, solution: impl FnOnce() -> Solution) {
        self.note_best(quality);
        let shown = self.state().shown_quality;
        let better = shown.is_none_or(|shown| quality > shown || (equal && quality >= shown));
        if self.inner.live && better {
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
        state.best_quality = Some(state.best_quality.map_or(quality, |best| best.max(quality)));
    }

    /// The quality of the best solution so far.
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
