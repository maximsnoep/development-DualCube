//! The optimization window (see `Solution::optimize`): buttons to optimize, stop, and reset, the progress (step, best
//! and current quality), a chart of the quality per step and generation (over all runs so far) with the temperature
//! of the annealing, the steps of the annealing and the generations of the layout per cycle, the starting
//! temperature, the live renders, the weights of the quality criterion, and a table of the mutations (of the loops,
//! and of the layout): a toggle per mutation, its statistics, and its current chance to be chosen. The settings can be
//! changed while the optimization runs (from its next step or cycle). Styled like the menus (see `theme`, `widgets`).

use super::theme::{
    BG_COLOR, DIM_COLOR, OK_GREEN, OUTLINE_COLOR, TEXT_COLOR, TEXT_COLOR2, TEXT_SIZE, WARN_RED, text,
    window_frame,
};
use super::widgets::{
    bold_label, collapsed, label, sep, sleek_button, sleek_button_warn, space, window_title,
};
use crate::jobs::{Job, JobState, OptimizationLive};
use crate::resources::{Configuration, SolutionResource};
use bevy::prelude::*;
use bevy_egui::egui::{self, Color32, Grid, Pos2, Rect, RichText, Sense, Stroke, Vec2};
use dualcube::prelude::*;
use rustc_hash::{FxHashMap as HashMap};
use std::collections::BTreeSet;

// Smoothed paths are switched off (once) when a smoothing takes longer than this (in seconds).
const SLOW_SMOOTHING: f64 = 1.;

// The height of the chart (and of the thermometer next to it), and the width of the thermometer (with its value).
const CHART_HEIGHT: f32 = 130.;
const THERMOMETER_WIDTH: f32 = 48.;

/// The mutations of the layout, by their names in the statistics.
const LAYOUT_MUTATIONS: [&str; 2] = ["corner", "path"];

// The weights of the quality criterion (number fields), with a button to restore the defaults.
fn quality_weights(ui: &mut egui::Ui, weights: &mut QualityWeights) {
    let defaults = QualityWeights::default();
    Grid::new("quality weights")
        .num_columns(2)
        .spacing([12., 4.])
        .show(ui, |ui| {
            for (name, value) in [
                ("fidelity", &mut weights.fidelity),
                ("path regularity", &mut weights.path_regularity),
                ("non-degeneracy", &mut weights.non_degeneracy),
                ("correspondence", &mut weights.correspondence),
                ("complexity (per loop)", &mut weights.complexity),
            ] {
                label(ui, name, TEXT_COLOR2);
                ui.add(
                    egui::DragValue::new(value)
                        .range(0.0..=100.)
                        .speed(0.01)
                        .max_decimals(3),
                );
                ui.end_row();
            }
        });
    if *weights != defaults && sleek_button(ui, "defaults") {
        *weights = defaults;
    }
}

/// Shows the optimization window once it was opened (with the optimization button of the menu).
pub fn show(
    mut egui_ctx: bevy_egui::EguiContexts<'_, '_>,
    mut live: ResMut<'_, OptimizationLive>,
    mut conf: ResMut<'_, Configuration>,
    solution: Res<'_, SolutionResource>,
    job_state: Res<'_, JobState>,
    mut jobs: MessageWriter<'_, Job>,
) -> Result<(), BevyError> {
    if !live.open && live.monitor.is_none() {
        return Ok(());
    }
    // Without a layout, the optimization initializes first.
    let ready = job_state.request.is_none() && solution.current_solution.mesh_ref.nr_verts() > 0;
    let conf = &mut *conf;
    let monitor = live.monitor.clone();
    let history = monitor
        .as_ref()
        .map(OptimizationMonitor::history)
        .unwrap_or_default();
    let running = live.running();
    let (mut start, mut reset) = (false, false);

    let ctx = egui_ctx.ctx_mut()?;
    let is_collapsed = collapsed(ctx, "Optimization");
    egui::Window::new("optimization")
        .title_bar(false)
        .frame(window_frame())
        .default_width(480.)
        .min_width(430.)
        // At first at the top right of the main view (movable).
        .pivot(egui::Align2::RIGHT_TOP)
        .default_pos(super::main_area(ctx).right_top() + Vec2::new(-15., 45.))
        .resizable(!is_collapsed)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                // Clicking the title collapses the window to this row.
                window_title(ui, "Optimization");
                let (status, color) = match &monitor {
                    Some(m) if m.stopping() && running => {
                        ("stopping after this generation".to_owned(), WARN_RED)
                    }
                    Some(_) if running => ("running".to_owned(), TEXT_COLOR2),
                    Some(_) => ("finished".to_owned(), TEXT_COLOR2),
                    None => ("ready".to_owned(), TEXT_COLOR2),
                };
                label(ui, &status, color);
                // At the right: optimize (from the current solution) and reset (forget the optimizations so far, their
                // chart and statistics, and clear the solution: only the input mesh stays), or stop.
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if running {
                        if let Some(m) = &monitor
                            && !m.stopping()
                            && sleek_button_warn(ui, "stop")
                        {
                            m.stop();
                        }
                    } else {
                        // (Right to left.)
                        for (name, action) in [("reset", &mut reset), ("optimize", &mut start)] {
                            if ready && sleek_button(ui, name) {
                                *action = true;
                            } else if !ready {
                                label(ui, name, TEXT_COLOR2);
                            }
                        }
                    }
                });
            });
            if is_collapsed {
                return;
            }
            sep(ui);

            // The progress, and the chart with the temperature next to it (always shown, empty before the first
            // run, so the window keeps its size); the settings below them.
            progress(
                ui,
                &history,
                monitor.as_ref().and_then(OptimizationMonitor::best_quality),
            );
            space(ui);
            let mut runs = live.past.clone();
            runs.push(history.clone());
            ui.horizontal(|ui| {
                // (The chart leaves exactly the width of the thermometer, else the window would grow every frame.)
                chart(ui, &runs, THERMOMETER_WIDTH + ui.spacing().item_spacing.x);
                thermometer(ui, &history);
            });
            sep(ui);

            // Per cycle: the steps of the annealing of the loops and the generations of the layout (from the next
            // cycle). (The starting temperature follows from the model, see `AnnealParams::acceptance`.)
            ui.horizontal(|ui| {
                label(ui, "Dual generations", TEXT_COLOR2);
                ui.add(egui::DragValue::new(&mut conf.anneal.steps).range(0..=1000));
                space(ui);
                label(ui, "Primal generations", TEXT_COLOR2);
                ui.add(egui::DragValue::new(&mut conf.layout_generations).range(0..=1000));
            });
            // Stop when it stalls: after this many cycles in a row without a new best solution (0: never).
            ui.horizontal(|ui| {
                label(ui, "stop after", TEXT_COLOR2);
                ui.add(egui::DragValue::new(&mut conf.patience).range(0..=1000));
                label(ui, "cycles without improvement (0: never)", TEXT_COLOR2);
            });
            // The live renders of the best solution: the time between them (it grows to twice a render that takes
            // longer) and the mean time of a render; and smoothed paths (only of the solutions that are shown; the
            // solutions keep their paths) and the mean time of a smoothing.
            ui.horizontal(|ui| {
                label(ui, "render every", TEXT_COLOR2);
                ui.add(
                    egui::DragValue::new(&mut live.render_interval)
                        .range(0.0..=600.)
                        .speed(0.1)
                        .max_decimals(1)
                        .suffix(" s"),
                );
                if let Some(seconds) = live.render_time() {
                    label(ui, &format!("· {}", short_duration(seconds)), DIM_COLOR);
                }
                space(ui);
                if ui
                    .checkbox(&mut conf.smooth_paths, text("smooth paths"))
                    .changed()
                    && !running
                {
                    jobs.write(Job::refresh(
                        solution.current_solution.clone(),
                        conf.clone(),
                    ));
                }
                if let Some(seconds) = crate::render::smoothing_time() {
                    label(ui, &format!("· {}", short_duration(seconds)), DIM_COLOR);
                    // Smoothing that takes too long is switched off (once).
                    if seconds > SLOW_SMOOTHING && conf.smooth_paths && !live.smoothing_switched_off {
                        conf.smooth_paths = false;
                        live.smoothing_switched_off = true;
                    }
                }
                space(ui);
                // A sound on every new best solution.
                ui.checkbox(&mut live.ding, text("ding"));
            });
            sep(ui);

            let stats = monitor
                .as_ref()
                .map(OptimizationMonitor::mutation_stats)
                .unwrap_or_default()
                .into_iter()
                .collect::<HashMap<_, _>>();
            let chances = monitor
                .as_ref()
                .map(OptimizationMonitor::mutation_chances)
                .unwrap_or_default()
                .into_iter()
                .collect::<HashMap<_, _>>();
            let times = monitor
                .as_ref()
                .map(OptimizationMonitor::mutation_times)
                .unwrap_or_default()
                .into_iter()
                .collect::<HashMap<_, _>>();
            // The mutations of the loops (dual) and of the layout (primal).
            egui::CollapsingHeader::new(text("mutations"))
                .default_open(false)
                .show(ui, |ui| {
                    mutation_table(
                        ui,
                        [
                            (&LOOP_MUTATIONS[..], &mut conf.anneal_disabled),
                            (&LAYOUT_MUTATIONS[..], &mut conf.layout_disabled),
                        ],
                        &stats,
                        &chances,
                        &times,
                    );
                });
            // The weights of the quality criterion (see `QualityWeights`); used from the next start.
            egui::CollapsingHeader::new(text("quality weights"))
                .default_open(false)
                .show(ui, |ui| quality_weights(ui, &mut conf.quality.weights));
        });

    let settings = LiveSettings {
        loops_disabled: conf.anneal_disabled.clone(),
        layout_disabled: conf.layout_disabled.clone(),
        steps: conf.anneal.steps,
        layout_generations: conf.layout_generations,
        patience: conf.patience,
    };
    if running && let Some(m) = &monitor {
        // Applied from the next step (or cycle).
        m.set_settings(settings);
    } else if start {
        let monitor = live.start();
        monitor.set_settings(settings);
        jobs.write(Job::optimize(
            solution.current_solution.clone(),
            conf.clone(),
            monitor,
        ));
    } else if reset {
        // The optimizations so far, and the solution.
        live.reset();
        jobs.write(Job::clear(solution.current_solution.clone(), conf.clone()));
    }
    Ok(())
}

// The mutations in groups (e.g., of the loops and of the layout, one after the other; with the mutations that are
// switched off): per mutation its current chance to be chosen, a toggle, how many were tried, improved on their parent, and improved the best; the mean
// time of a try; the success (its rate of improving the best, normalized over the switched-on mutations of its group)
// over the whole run and in the last generations; and its chance to be chosen (based on the latter). Switched-off
// mutations are darkened.
fn mutation_table<const N: usize>(
    ui: &mut egui::Ui,
    groups: [(&[&str], &mut BTreeSet<String>); N],
    stats: &HashMap<&'static str, [usize; 4]>,
    chances: &HashMap<&'static str, MutationChance>,
    times: &HashMap<&'static str, f64>,
) {
    let rate =
        |successes: usize, tries: usize| (tries > 0).then(|| successes as f64 / tries as f64);
    let all_time = |name: &str| {
        let [tries, _, _, best] = stats.get(name).copied().unwrap_or_default();
        rate(best, tries)
    };
    let recent_rate = |name: &str| chances.get(name).and_then(|w| rate(w.successes, w.tries));
    // ("-": not tried; 0% when tried, also when no mutation of the group succeeded.)
    let normalized = |value: Option<f64>, total: f64| match value {
        Some(value) if total > 0. => format!("{:.1}%", 100. * value / total),
        Some(_) => "0.0%".to_owned(),
        None => "-".to_owned(),
    };
    Grid::new("mutations")
        .num_columns(8)
        .spacing([15., 2.])
        .show(ui, |ui| {
            let recent = format!("last {SELECTION_WINDOW}");
            for header in [
                "chance",
                "mutation",
                "tries",
                "improved",
                "new best",
                "ms",
                "success",
                recent.as_str(),
            ] {
                label(ui, header, TEXT_COLOR2);
            }
            ui.end_row();
            for (mutations, disabled) in groups {
                let enabled = mutations
                    .iter()
                    .copied()
                    .filter(|name| !disabled.contains(*name))
                    .collect::<Vec<_>>();
                let all_time_total: f64 = enabled.iter().filter_map(|name| all_time(name)).sum();
                let recent_total: f64 = enabled.iter().filter_map(|name| recent_rate(name)).sum();
                for &name in mutations {
                    let mut switched_on = !disabled.contains(name);
                    let color = if switched_on { TEXT_COLOR } else { DIM_COLOR };
                    label(
                        ui,
                        &chances
                            .get(name)
                            .filter(|_| switched_on)
                            .map_or_else(|| "-".to_owned(), |w| format!("{:.0}%", 100. * w.chance)),
                        color,
                    );
                    if ui
                        .checkbox(
                            &mut switched_on,
                            RichText::new(name).color(color).size(TEXT_SIZE),
                        )
                        .changed()
                    {
                        if switched_on {
                            disabled.remove(name);
                        } else {
                            disabled.insert(name.to_owned());
                        }
                    }
                    let [tries, _, improved, best] = stats.get(name).copied().unwrap_or_default();
                    label(ui, &tries.to_string(), color);
                    label(ui, &improved.to_string(), color);
                    label(ui, &best.to_string(), color);
                    label(
                        ui,
                        &times
                            .get(name)
                            .map_or_else(|| "-".to_owned(), |ms| format!("{ms:.1}")),
                        color,
                    );
                    label(
                        ui,
                        &normalized(all_time(name).filter(|_| switched_on), all_time_total),
                        color,
                    );
                    label(
                        ui,
                        &normalized(recent_rate(name).filter(|_| switched_on), recent_total),
                        color,
                    );
                    ui.end_row();
                }
            }
        });
}

// The progress: the generation, the best quality so far (of a solution the optimization can return, see
// `OptimizationMonitor::best_quality`), the quality of the current solution, and the time.
fn progress(ui: &mut egui::Ui, history: &[GenerationStats], best: Option<f64>) {
    let last = history.last();
    Grid::new("optimization progress")
        .num_columns(4)
        .spacing([24., 0.])
        .show(ui, |ui| {
            for caption in ["generation", "best", "current", "time"] {
                label(ui, caption, TEXT_COLOR2);
            }
            ui.end_row();
            if let Some(last) = last {
                label(ui, &last.generation.to_string(), TEXT_COLOR);
                bold_label(ui, &format!("{:.4}", best.unwrap_or(last.best)), OK_GREEN);
                label(ui, &format!("{:.4}", last.current), TEXT_COLOR);
                label(ui, &duration(last.seconds), TEXT_COLOR);
            } else {
                for placeholder in ["-", "-.----", "-.----", "-"] {
                    label(ui, placeholder, DIM_COLOR);
                }
            }
            ui.end_row();
        });
}

// The temperature of the annealing (of its last step) on a vertical scale from 0 (bottom) to the starting temperature
// (top), as high as the chart, with its value next to it.
fn thermometer(ui: &mut egui::Ui, history: &[GenerationStats]) {
    // The last steps of the annealing (of the last cycle): the first is at the starting temperature.
    let steps = history
        .iter()
        .rev()
        .skip_while(|s| s.temperature.is_none())
        .map_while(|s| s.temperature)
        .collect::<Vec<_>>();
    let temperature = steps.first().copied();
    let start = steps.last().copied().unwrap_or(0.);
    // A fixed size, painted (the bar at the left, its value at its top right).
    let (area, _) =
        ui.allocate_exact_size(Vec2::new(THERMOMETER_WIDTH, CHART_HEIGHT), Sense::hover());
    let painter = ui.painter_at(area);
    let bar = Rect::from_min_size(area.min, Vec2::new(12., CHART_HEIGHT));
    let fraction = temperature.map_or(0., |t| (t / start.max(1e-12)).clamp(0., 1.)) as f32;
    painter.rect_filled(
        Rect::from_min_max(
            Pos2::new(bar.left(), bar.bottom() - bar.height() * fraction),
            bar.max,
        ),
        0.,
        TEXT_COLOR2,
    );
    painter.rect_stroke(bar, 0., Stroke::new(1., OUTLINE_COLOR), egui::StrokeKind::Inside);
    let (value, color) = temperature.map_or_else(
        || ("-.---".to_owned(), DIM_COLOR),
        |t| (format!("{t:.3}"), TEXT_COLOR),
    );
    painter.text(
        Pos2::new(bar.right() + 4., bar.top()),
        egui::Align2::LEFT_TOP,
        value,
        egui::FontId::proportional(TEXT_SIZE),
        color,
    );
}

// A short duration: in milliseconds below a second, else in seconds.
fn short_duration(seconds: f64) -> String {
    if seconds < 1. {
        format!("{:.0} ms", 1e3 * seconds)
    } else {
        format!("{seconds:.1} s")
    }
}

// A duration: in seconds, in minutes after a minute, and in hours after an hour.
fn duration(seconds: f64) -> String {
    if seconds < 60. {
        format!("{seconds:.0} s")
    } else if seconds < 3600. {
        format!("{}:{:02} min", (seconds / 60.) as u64, seconds as u64 % 60)
    } else {
        format!(
            "{}:{:02} h",
            (seconds / 3600.) as u64,
            (seconds as u64 / 60) % 60
        )
    }
}

// The quality per generation, over all runs so far (separated by vertical lines), fitted to the values: of the current
// solution (thin line) and the best (thick line). It leaves `reserve` of the width (for the thermometer to its right).
fn chart(ui: &mut egui::Ui, runs: &[Vec<GenerationStats>], reserve: f32) {
    let size = Vec2::new((ui.available_width() - reserve).max(200.), CHART_HEIGHT);
    let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_stroke(
        rect,
        0.,
        Stroke::new(1., OUTLINE_COLOR),
        egui::StrokeKind::Inside,
    );
    let history = runs.iter().flatten().copied().collect::<Vec<_>>();
    if history.len() < 2 {
        return;
    }

    // Fitted to the values (leaving out the first quarter of the generations, where the qualities are far worse).
    let recent = &history[history.len() / 4..];
    let low = recent.iter().map(|s| s.current).fold(f64::INFINITY, f64::min);
    let high = history
        .iter()
        .map(|s| s.best)
        .fold(f64::NEG_INFINITY, f64::max);
    let (low, high) = if high - low < 1e-9 {
        (low - 0.01, high + 0.01)
    } else {
        let margin = 0.05 * (high - low);
        (low - margin, high + margin)
    };
    let last = history.len() - 1;
    let inner = Rect::from_min_max(rect.min + Vec2::new(8., 10.), rect.max - Vec2::new(8., 10.));
    let point = |index: usize, value: f64| {
        let x = inner.left() + inner.width() * index as f32 / last as f32;
        let t = ((value - low) / (high - low)).clamp(0., 1.) as f32;
        Pos2::new(x, inner.bottom() - t * inner.height())
    };

    // The phases of the annealing of the loops (dual) on a lighter background: their children are scored with a
    // layout that is not optimized, so their qualities are lower than those after the layout is optimized.
    // (Every phase: whether it optimizes the layout, its first and its last generation.)
    let mut phases: Vec<(bool, usize, usize)> = vec![];
    for (i, s) in history.iter().enumerate() {
        match phases.last_mut() {
            Some((layout, _, last)) if *layout == s.layout => *last = i,
            _ => phases.push((s.layout, i, i)),
        }
    }
    for &(layout, first, last) in &phases {
        if !layout {
            let (left, right) = (point(first, low).x, point(last, low).x);
            painter.rect_filled(
                Rect::from_min_max(Pos2::new(left, rect.top() + 1.), Pos2::new(right, rect.bottom() - 1.)),
                0.,
                BG_COLOR.lerp_to_gamma(OUTLINE_COLOR, 0.5),
            );
        }
    }

    // The start of every optimization after the first.
    let mut start = 0;
    for run in &runs[..runs.len() - 1] {
        start += run.len();
        if start > 0 && start < history.len() {
            let x = point(start, low).x;
            painter.line_segment(
                [Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())],
                Stroke::new(1., OUTLINE_COLOR),
            );
        }
    }

    let series = |value: fn(&GenerationStats) -> f64| {
        history
            .iter()
            .enumerate()
            .map(|(i, s)| point(i, value(s)))
            .collect::<Vec<_>>()
    };
    // The current quality, interrupted between the phases (they score differently).
    let current = series(|s| s.current);
    for &(_, first, last) in &phases {
        painter.add(egui::Shape::line(
            current[first..=last].to_vec(),
            Stroke::new(1., TEXT_COLOR2),
        ));
    }
    painter.add(egui::Shape::line(
        series(|s| s.best),
        Stroke::new(2., OK_GREEN),
    ));

    // The temperature of the annealing (on its own scale, from 0 at the bottom to the highest at the top): one line
    // over all generations, kept between the steps of the annealing (e.g., while the layout is optimized).
    let hottest = history
        .iter()
        .filter_map(|s| s.temperature)
        .fold(0., f64::max);
    if hottest > 0. {
        let y = |t: f64| inner.bottom() - (t / hottest) as f32 * inner.height();
        let mut current = None;
        let line = history
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                current = s.temperature.or(current);
                Some(Pos2::new(point(i, low).x, y(current?)))
            })
            .collect::<Vec<_>>();
        painter.add(egui::Shape::line(line, Stroke::new(1., DIM_COLOR)));
    }

    // The best quality of every chain of the annealing (a line per chain, in its shade of green; interrupted where
    // the layout is optimized).
    for (chain, color) in chain_colors().into_iter().enumerate() {
        let points = history
            .iter()
            .enumerate()
            .map(|(i, s)| s.chains[chain].map(|q| point(i, q)))
            .collect::<Vec<_>>();
        for run in points.split(Option::is_none) {
            let line = run.iter().flatten().copied().collect::<Vec<_>>();
            if line.len() > 1 {
                painter.add(egui::Shape::line(line, Stroke::new(1., color)));
            }
        }
    }
}

// The colors of the chains of the annealing in the chart (see `GenerationStats::chains`): shades of the green of the
// best quality, from darker to lighter.
fn chain_colors() -> [Color32; SHOWN_CHAINS] {
    [
        OK_GREEN.lerp_to_gamma(Color32::BLACK, 0.5),
        OK_GREEN.lerp_to_gamma(Color32::BLACK, 0.25),
        OK_GREEN.lerp_to_gamma(Color32::WHITE, 0.25),
        OK_GREEN.lerp_to_gamma(Color32::WHITE, 0.5),
    ]
}
