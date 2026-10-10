//! Live view of a running evolution: the best solution so far is rendered while the evolution job runs (outside the
//! single job slot, see [`poll_evolution`]; at most every `RENDER_INTERVAL`), and the statistics per generation are
//! shown in the evolution window.

use crate::render;
use crate::render::store::RenderObjectStore;
use crate::resources::{Configuration, SolutionResource};
use bevy::prelude::*;
use bevy::tasks::futures_lite::future;
use bevy::tasks::{AsyncComputeTaskPool, Task};
use dualcube::prelude::*;
use std::time::{Duration, Instant};

/// The live view fetches the best solution of a running evolution this long after its last render finished (the
/// first one at once): rendering may take longer than a generation.
const RENDER_INTERVAL: Duration = Duration::from_secs(5);

/// The monitor of the current (or last) evolution, and the rendering of its best solution so far.
#[derive(Resource, Default)]
pub struct EvolutionLive {
    pub monitor: Option<EvolutionMonitor>,
    /// The statistics of the earlier evolutions (shown before the current one in the chart).
    pub past: Vec<Vec<GenerationStats>>,
    /// Whether the evolution window is shown.
    pub open: bool,
    // The version of the best solution that was last rendered, and the polycube lengths (unit or geometric) it was
    // rendered with.
    seen: usize,
    seen_unit: Option<bool>,
    refresh: Option<Task<(Solution, RenderObjectStore)>>,
    // When the last render finished.
    rendered_at: Option<Instant>,
    /// The phase of the loops that the running optimization was last seen in (the phase chosen in the window follows
    /// it when it changes).
    pub seen_phase: Option<LoopPhase>,
}

impl EvolutionLive {
    /// Start following a new evolution.
    pub fn start(&mut self) -> EvolutionMonitor {
        if let Some(previous) = &self.monitor {
            let history = previous.history();
            if !history.is_empty() {
                self.past.push(history);
            }
        }
        let monitor = EvolutionMonitor::live();
        self.monitor = Some(monitor.clone());
        self.seen = 0;
        self.refresh = None;
        self.rendered_at = None;
        self.seen_phase = None;
        monitor
    }

    /// Forget the evolutions so far (e.g., for a new model); the window stays open if it was.
    pub fn reset(&mut self) {
        self.stop();
        *self = Self {
            open: self.open,
            ..Self::default()
        };
    }

    /// Stop the running evolution (if any).
    pub fn stop(&self) {
        if let Some(monitor) = &self.monitor {
            monitor.stop();
        }
    }

    /// Whether an evolution is running.
    pub fn running(&self) -> bool {
        self.monitor.as_ref().is_some_and(|m| !m.finished())
    }
}

/// Renders the best solution of the running optimization when it changed, at most every `RENDER_INTERVAL` (after the
/// last render finished).
pub(super) fn poll_evolution(
    mut live: ResMut<'_, EvolutionLive>,
    mut solution_resource: ResMut<'_, SolutionResource>,
    mut render_object_store: ResMut<'_, RenderObjectStore>,
    configuration: Res<'_, Configuration>,
) {
    // The resources are passed as `ResMut` (not dereferenced): only an actual change may mark them as changed, as
    // a changed render store respawns all renders.
    poll(
        live.bypass_change_detection(),
        &mut solution_resource,
        &mut render_object_store,
        &configuration,
    );
}

// See `poll_evolution`. Results that arrive after the evolution finished are dropped (the final result replaces them).
fn poll(
    live: &mut EvolutionLive,
    solution_resource: &mut ResMut<'_, SolutionResource>,
    render_object_store: &mut ResMut<'_, RenderObjectStore>,
    configuration: &Configuration,
) {
    let running = live.running();
    if let Some(task) = &mut live.refresh
        && let Some((solution, store)) = future::block_on(future::poll_once(task))
    {
        live.refresh = None;
        live.rendered_at = Some(Instant::now());
        if running {
            solution_resource.current_solution = solution;
            **render_object_store = store;
        }
    }
    if !running || live.refresh.is_some() {
        return;
    }
    // Render the best solution again when the polycube lengths were switched (at once), else at most every
    // `RENDER_INTERVAL`.
    if live.seen_unit != Some(configuration.unit) {
        live.seen = 0;
    } else if live
        .rendered_at
        .is_some_and(|rendered| rendered.elapsed() < RENDER_INTERVAL)
    {
        return;
    }
    let Some((version, best)) = live
        .monitor
        .as_ref()
        .and_then(|monitor| monitor.best_since(live.seen))
    else {
        return;
    };
    live.seen = version;
    live.seen_unit = Some(configuration.unit);
    let configuration = configuration.clone();
    live.refresh = Some(AsyncComputeTaskPool::get().spawn(async move {
        // The polycube with the chosen lengths (the evolutions use unit lengths, or those of the input).
        let mut best = best;
        if best.resize_polycube(configuration.unit).is_err() {
            warn!("Failed to resize the polycube of the best solution");
        }
        let store = render::refresh(&best, &configuration);
        (best, store)
    }));
}
