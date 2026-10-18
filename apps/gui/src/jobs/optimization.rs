//! Live view of a running optimization: the best solution so far is rendered while the optimization job runs (outside the
//! single job slot, see [`poll_optimization`]; at most every `OptimizationLive::render_interval`), and the statistics per
//! generation are shown in the optimization window.

use crate::render;
use crate::render::store::RenderObjectStore;
use crate::resources::{Configuration, SolutionResource};
use bevy::prelude::*;
use bevy::tasks::futures_lite::future;
use bevy::tasks::{AsyncComputeTaskPool, Task};
use dualcube::prelude::*;
use std::time::{Duration, Instant};

/// The monitor of the current (or last) optimization, and the rendering of its best solution so far.
#[derive(Resource)]
pub struct OptimizationLive {
    pub monitor: Option<OptimizationMonitor>,
    /// The statistics of the earlier optimizations (shown before the current one in the chart).
    pub past: Vec<Vec<GenerationStats>>,
    /// Whether the optimization window is shown.
    pub open: bool,
    // The version of the best solution that was last rendered.
    seen: usize,
    refresh: Option<Task<(Solution, RenderObjectStore, Duration)>>,
    // When the last render finished.
    rendered_at: Option<Instant>,
    /// The live view renders the best solution of a running optimization this many seconds after its last render
    /// finished (the first one at once): rendering may take longer than a generation. A render that takes longer than
    /// this sets it to twice its time.
    pub render_interval: f64,
    // The time of the last render.
    render_time: Option<f64>,
    /// Whether the smoothed paths were switched off because smoothing was too slow (see `SLOW_SMOOTHING`; once: the
    /// user may switch them on again).
    pub smoothing_switched_off: bool,
    /// Whether a new best solution is announced with a sound (see `sound`), and the best quality that was announced
    /// (or that the optimization started with).
    pub ding: bool,
    pub dinged: Option<f64>,
}

impl Default for OptimizationLive {
    fn default() -> Self {
        Self {
            monitor: None,
            past: vec![],
            open: false,
            seen: 0,
            refresh: None,
            rendered_at: None,
            render_interval: 1.,
            render_time: None,
            smoothing_switched_off: false,
            ding: false,
            dinged: None,
        }
    }
}

impl OptimizationLive {
    /// Start following a new optimization.
    pub fn start(&mut self) -> OptimizationMonitor {
        if let Some(previous) = &self.monitor {
            let history = previous.history();
            if !history.is_empty() {
                self.past.push(history);
            }
        }
        let monitor = OptimizationMonitor::live();
        self.monitor = Some(monitor.clone());
        self.seen = 0;
        self.refresh = None;
        self.rendered_at = None;
        self.dinged = None;
        monitor
    }

    /// Forget the optimizations so far (e.g., for a new model); the window and the render interval stay.
    pub fn reset(&mut self) {
        self.stop();
        *self = Self {
            open: self.open,
            render_interval: self.render_interval,
            ding: self.ding,
            ..Self::default()
        };
    }

    /// The time of the last live render, in seconds.
    pub fn render_time(&self) -> Option<f64> {
        self.render_time
    }

    /// Stop the running optimization (if any).
    pub fn stop(&self) {
        if let Some(monitor) = &self.monitor {
            monitor.stop();
        }
    }

    /// Whether an optimization is running.
    pub fn running(&self) -> bool {
        self.monitor.as_ref().is_some_and(|m| !m.finished())
    }
}

/// Renders the best solution of the running optimization when it changed, at most every
/// `OptimizationLive::render_interval` (after the last render finished).
pub(super) fn poll_optimization(
    mut live: ResMut<'_, OptimizationLive>,
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

// See `poll_optimization`. Results that arrive after the optimization finished are dropped (the final result replaces them).
fn poll(
    live: &mut OptimizationLive,
    solution_resource: &mut ResMut<'_, SolutionResource>,
    render_object_store: &mut ResMut<'_, RenderObjectStore>,
    configuration: &Configuration,
) {
    let running = live.running();
    if let Some(task) = &mut live.refresh
        && let Some((solution, store, time)) = future::block_on(future::poll_once(task))
    {
        live.refresh = None;
        live.rendered_at = Some(Instant::now());
        let seconds = time.as_secs_f64();
        live.render_time = Some(seconds);
        if seconds > live.render_interval {
            live.render_interval = 2. * seconds;
        }
        if running {
            solution_resource.current_solution = solution;
            **render_object_store = store;
        }
    }
    if !running || live.refresh.is_some() {
        return;
    }
    // Render the best solution (when it changed) at most every `render_interval`.
    let interval = Duration::from_secs_f64(live.render_interval.max(0.));
    if live
        .rendered_at
        .is_some_and(|rendered| rendered.elapsed() < interval)
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
    let configuration = configuration.clone();
    live.refresh = Some(AsyncComputeTaskPool::get().spawn(async move {
        let timer = Instant::now();
        // The polycube with the lengths of the layout, and the loops smoothed (as the result, see `Job::optimize`).
        let mut best = best.with_smooth_loops();
        if best.resize_polycube(false).is_err() {
            warn!("Failed to resize the polycube of the best solution");
        }
        let store = render::refresh(&best, &configuration);
        (best, store, timer.elapsed())
    }));
}
