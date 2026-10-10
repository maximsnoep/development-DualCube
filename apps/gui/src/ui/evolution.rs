//! The optimization window (see `Solution::optimize`): the phase of the loops (see `LoopPhase`), the generations per
//! cycle of the loops and of the layout, the population sizes, buttons to start, stop, and continue the optimization, its progress (generation, best and mean
//! quality), a chart of the quality per generation (over all runs so far), and tables of the mutations (of the loops,
//! and of the layout): a toggle per mutation, its statistics, and its current chance to be chosen. The settings can be
//! changed while the optimization runs (from its next generation or cycle). Styled like the menus (see `theme`,
//! `widgets`).

use super::theme::{
    DIM_COLOR, OK_GREEN, OUTLINE_COLOR, TEXT_COLOR, TEXT_COLOR2, TEXT_SIZE, WARN_RED, text,
    text_format, window_frame,
};
use super::widgets::{
    bold_label, collapsed, label, sep, sleek_button, sleek_button_warn, space, window_title,
};
use crate::jobs::{EvolutionLive, Job, JobState};
use crate::resources::{Configuration, SolutionResource};
use bevy::prelude::*;
use bevy_egui::egui::{self, Color32, Grid, Pos2, Rect, RichText, Sense, Stroke, Vec2};
use dualcube::prelude::*;
use std::collections::{BTreeSet, HashMap};

/// The mutations of the layout, by their names in the statistics.
const LAYOUT_MUTATIONS: [&str; 8] = [
    "corner",
    "edge",
    "exchange",
    "path (ridge)",
    "path (shortest)",
    "path (axis)",
    "path (feature)",
    "reroute paths",
];

// Sliders for the weights of the quality criterion, with a button to restore the defaults.
fn quality_weights(ui: &mut egui::Ui, weights: &mut QualityWeights) {
    let defaults = QualityWeights::default();
    for (name, value, max) in [
        ("fidelity", &mut weights.fidelity, 5.),
        ("path regularity", &mut weights.path_regularity, 2.),
        ("coherence", &mut weights.coherence, 10.),
        ("rectangularity", &mut weights.rectangularity, 2.),
        ("corner regularity", &mut weights.corner_regularity, 5.),
        ("non-degeneracy", &mut weights.non_degeneracy, 5.),
        ("correspondence", &mut weights.correspondence, 2.),
        ("complexity (per loop)", &mut weights.complexity, 0.1),
    ] {
        ui.add(egui::Slider::new(value, 0.0..=max).text(text(name)));
    }
    if *weights != defaults && sleek_button(ui, "defaults") {
        *weights = defaults;
    }
}

/// Shows the optimization window once it was opened (with the optimize button of the DUAL stage).
pub fn show(
    mut egui_ctx: bevy_egui::EguiContexts<'_, '_>,
    mut live: ResMut<'_, EvolutionLive>,
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
        .map(EvolutionMonitor::history)
        .unwrap_or_default();
    let running = live.running();
    let mut start = false;

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
                    Some(m) if running => (m.activity(), TEXT_COLOR2),
                    Some(_) => ("finished".to_owned(), TEXT_COLOR2),
                    None => ("ready".to_owned(), TEXT_COLOR2),
                };
                label(ui, &status, color);
            });
            if is_collapsed {
                return;
            }
            sep(ui);

            // Start, stop, or continue (from the current result).
            ui.horizontal(|ui| {
                if running {
                    if let Some(m) = &monitor
                        && !m.stopping()
                        && sleek_button_warn(ui, "stop")
                    {
                        m.stop();
                    }
                } else {
                    let name = if history.is_empty() {
                        "start"
                    } else {
                        "continue"
                    };
                    if ready && sleek_button(ui, name) {
                        start = true;
                    } else if !ready {
                        label(ui, name, TEXT_COLOR2);
                    }
                }
            });
            space(ui);

            // The phase of the loops (pruning starts by itself, see `LoopPhase`); can be switched while running. It
            // follows the phase that runs (which moves on when it stalls, see `LiveSettings::advance_after`).
            if running
                && let Some(phase) = monitor.as_ref().and_then(EvolutionMonitor::phase)
                && live.seen_phase != Some(phase)
            {
                live.seen_phase = Some(phase);
                conf.evolution.phase = phase;
            }
            ui.horizontal(|ui| {
                label(ui, "phase", TEXT_COLOR2);
                for phase in LoopPhase::CHOSEN {
                    // The chosen phase in gray (not in the selection color).
                    let selected = conf.evolution.phase == phase;
                    let (color, fill) = if selected {
                        (TEXT_COLOR, Color32::from_gray(70))
                    } else {
                        (TEXT_COLOR2, Color32::TRANSPARENT)
                    };
                    let button =
                        egui::Button::new(RichText::new(phase.name()).color(color).size(TEXT_SIZE))
                            .fill(fill)
                            .stroke(Stroke::NONE);
                    if ui.add(button).clicked() {
                        conf.evolution.phase = phase;
                    }
                }
            });
            // When growth or shaping switch to pruning (mostly removals), and for how long (0: never).
            ui.horizontal(|ui| {
                label(ui, "prune after", TEXT_COLOR2);
                ui.add(egui::DragValue::new(&mut conf.evolution.phase_patience).range(1..=50));
                label(ui, "generations without improvement, for", TEXT_COLOR2);
                ui.add(egui::DragValue::new(&mut conf.evolution.prune_generations).range(0..=50));
            });
            // When the optimization moves on to the next phase (0: never).
            ui.horizontal(|ui| {
                label(ui, "next phase after", TEXT_COLOR2);
                ui.add(egui::DragValue::new(&mut conf.advance_after).range(0..=1000));
                label(ui, "generations without improvement", TEXT_COLOR2);
            });
            // The shown solutions (also while optimizing) with smoothed paths; the solutions keep their paths.
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
            sep(ui);

            // The settings of the evolutionary algorithm (each foldable).
            // Per cycle: the generations of the loops and of the layout; and the population sizes of both.
            egui::CollapsingHeader::new(text("generations and populations"))
                .default_open(false)
                .show(ui, |ui| {
                    Grid::new("optimization settings")
                        .num_columns(4)
                        .spacing([12., 4.])
                        .show(ui, |ui| {
                            for caption in ["", "generations", "population", "offspring"] {
                                label(ui, caption, TEXT_COLOR2);
                            }
                            ui.end_row();
                            label(ui, "loops", TEXT_COLOR);
                            ui.add(egui::DragValue::new(&mut conf.loop_generations).range(0..=200));
                            ui.add(
                                egui::DragValue::new(&mut conf.evolution.population).range(1..=30),
                            );
                            ui.add(
                                egui::DragValue::new(&mut conf.evolution.offspring).range(1..=100),
                            );
                            ui.end_row();
                            label(ui, "layout", TEXT_COLOR);
                            ui.add(
                                egui::DragValue::new(&mut conf.layout_generations).range(0..=200),
                            );
                            ui.add(
                                egui::DragValue::new(&mut conf.layout_evolution.population)
                                    .range(1..=30),
                            );
                            ui.add(
                                egui::DragValue::new(&mut conf.layout_evolution.offspring)
                                    .range(1..=100),
                            );
                            ui.end_row();
                        });
                });
            // The weights of the quality criterion (see `QualityWeights`); used from the next start.
            egui::CollapsingHeader::new(text("quality weights"))
                .default_open(false)
                .show(ui, |ui| quality_weights(ui, &mut conf.quality.weights));
            let stats = monitor
                .as_ref()
                .map(EvolutionMonitor::mutation_stats)
                .unwrap_or_default()
                .into_iter()
                .collect::<HashMap<_, _>>();
            let chances = monitor
                .as_ref()
                .map(EvolutionMonitor::mutation_chances)
                .unwrap_or_default()
                .into_iter()
                .collect::<HashMap<_, _>>();
            // Every phase switches its own mutations of the loops on and off (see `EvolutionParams::enabled`): the
            // table shows those of the phase that runs (pruning when it started by itself), else of the chosen phase.
            let selected = conf.evolution.phase;
            let shown = match &monitor {
                Some(m)
                    if running
                        && m.pruning()
                        && matches!(selected, LoopPhase::Growth | LoopPhase::Optimization) =>
                {
                    LoopPhase::Pruning
                }
                _ => selected,
            };
            egui::CollapsingHeader::new(text("loop mutations"))
                .default_open(false)
                .show(ui, |ui| {
                    label(
                        ui,
                        &format!("switched on in {} (every phase has its own)", shown.name()),
                        TEXT_COLOR2,
                    );
                    let mut disabled = LOOP_MUTATIONS
                        .iter()
                        .filter(|name| !conf.evolution.is_enabled(shown, name))
                        .map(|name| (*name).to_owned())
                        .collect::<BTreeSet<_>>();
                    mutation_table(
                        ui,
                        "loop mutations",
                        &LOOP_MUTATIONS,
                        &stats,
                        &chances,
                        &mut disabled,
                    );
                    for name in LOOP_MUTATIONS {
                        conf.evolution
                            .set_enabled(shown, name, !disabled.contains(name));
                    }
                });
            egui::CollapsingHeader::new(text("layout mutations"))
                .default_open(false)
                .show(ui, |ui| {
                    mutation_table(
                        ui,
                        "layout mutations",
                        &LAYOUT_MUTATIONS,
                        &stats,
                        &chances,
                        &mut conf.layout_disabled,
                    );
                });

            // The progress and the chart (always shown, empty before the first run, so the window keeps its size).
            sep(ui);
            progress(
                ui,
                history.last(),
                monitor.as_ref().and_then(EvolutionMonitor::best_quality),
            );
            space(ui);
            let mut runs = live.past.clone();
            runs.push(history.clone());
            chart(ui, &runs);
        });

    let settings = LiveSettings {
        loops: PhaseSettings {
            population: conf.evolution.population,
            offspring: conf.evolution.offspring,
            disabled: BTreeSet::new(),
        },
        layout: PhaseSettings {
            population: conf.layout_evolution.population,
            offspring: conf.layout_evolution.offspring,
            disabled: conf.layout_disabled.clone(),
        },
        loop_generations: conf.loop_generations,
        layout_generations: conf.layout_generations,
        phase_patience: conf.evolution.phase_patience,
        prune_generations: conf.evolution.prune_generations,
        phase: conf.evolution.phase,
        loop_enabled: Some(conf.evolution.enabled),
        advance_after: conf.advance_after,
    };
    if running && let Some(m) = &monitor {
        // Applied from the next generation (or cycle).
        m.set_settings(settings);
    } else if start {
        let monitor = live.start();
        monitor.set_settings(settings);
        jobs.write(Job::optimize(
            solution.current_solution.clone(),
            conf.clone(),
            monitor,
        ));
    }
    Ok(())
}

// Per mutation: a toggle, how many were tried, improved on their parent, and improved the best; the success (its rate of
// improving the best, normalized over the mutations of the table) over the whole run and in the last generations; and
// its chance to be chosen (based on the latter). Switched-off mutations are darkened.
fn mutation_table(
    ui: &mut egui::Ui,
    id: &str,
    mutations: &[&str],
    stats: &HashMap<&'static str, [usize; 4]>,
    chances: &HashMap<&'static str, MutationChance>,
    disabled: &mut BTreeSet<String>,
) {
    // The success rates (improvements of the best per try), normalized over the enabled mutations of the table (they sum
    // to 100%): over the whole run, and in the window of the adaptive selection.
    let rate =
        |successes: usize, tries: usize| (tries > 0).then(|| successes as f64 / tries as f64);
    let enabled = mutations
        .iter()
        .copied()
        .filter(|name| !disabled.contains(*name))
        .collect::<Vec<_>>();
    let all_time = |name: &str| {
        let [tries, _, _, best] = stats.get(name).copied().unwrap_or_default();
        rate(best, tries)
    };
    let recent_rate = |name: &str| chances.get(name).and_then(|w| rate(w.successes, w.tries));
    let normalized = |value: Option<f64>, total: f64| match value {
        Some(value) if total > 0. => format!("{:.1}%", 100. * value / total),
        _ => "-".to_owned(),
    };
    let all_time_total: f64 = enabled.iter().filter_map(|name| all_time(name)).sum();
    let recent_total: f64 = enabled.iter().filter_map(|name| recent_rate(name)).sum();
    Grid::new(id)
        .num_columns(7)
        .spacing([15., 2.])
        .show(ui, |ui| {
            let recent = format!("last {SELECTION_WINDOW}");
            for header in [
                "mutation",
                "tries",
                "improved",
                "new best",
                "success",
                recent.as_str(),
                "chance",
            ] {
                label(ui, header, TEXT_COLOR2);
            }
            ui.end_row();
            for &name in mutations {
                let mut enabled = !disabled.contains(name);
                let color = if enabled { TEXT_COLOR } else { DIM_COLOR };
                if ui
                    .checkbox(
                        &mut enabled,
                        RichText::new(name).color(color).size(TEXT_SIZE),
                    )
                    .changed()
                {
                    if enabled {
                        disabled.remove(name);
                    } else {
                        disabled.insert(name.to_owned());
                    }
                }
                let [tries, _, improved, best] = stats.get(name).copied().unwrap_or_default();
                label(ui, &tries.to_string(), color);
                label(ui, &improved.to_string(), color);
                label(ui, &best.to_string(), color);
                // Success (normalized, see above), over the whole run, and in the window of the adaptive selection
                // (on which the chance is based, see `MutationSelection`).
                let switched_on = !disabled.contains(name);
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
                label(
                    ui,
                    &chances
                        .get(name)
                        .filter(|_| switched_on)
                        .map_or_else(|| "-".to_owned(), |w| format!("{:.0}%", 100. * w.chance)),
                    color,
                );
                ui.end_row();
            }
        });
}

// The progress: the generation, the best quality of all time and of the generation, the mean quality of the
// generation, and the time.
fn progress(ui: &mut egui::Ui, last: Option<&GenerationStats>, best: Option<f64>) {
    Grid::new("evolution progress")
        .num_columns(5)
        .spacing([24., 0.])
        .show(ui, |ui| {
            for caption in [
                "generation",
                "all-time best",
                "generation best",
                "mean",
                "time",
            ] {
                label(ui, caption, TEXT_COLOR2);
            }
            ui.end_row();
            if let Some(last) = last {
                label(ui, &last.generation.to_string(), TEXT_COLOR);
                bold_label(ui, &format!("{:.4}", best.unwrap_or(last.best)), TEXT_COLOR);
                label(ui, &format!("{:.4}", last.best), OK_GREEN);
                label(ui, &format!("{:.4}", last.mean), TEXT_COLOR);
                label(ui, &duration(last.seconds), TEXT_COLOR);
            } else {
                for placeholder in ["-", "-.----", "-.----", "-.----", "-"] {
                    label(ui, placeholder, DIM_COLOR);
                }
            }
            ui.end_row();
        });
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

// The quality of the population per generation, over all runs so far (separated by vertical lines), fitted to the
// values: the mean (thin line) and the best (thick line).
fn chart(ui: &mut egui::Ui, runs: &[Vec<GenerationStats>]) {
    let size = Vec2::new(ui.available_width().max(200.), 130.);
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
    let low = recent.iter().map(|s| s.mean).fold(f64::INFINITY, f64::min);
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

    // The start of every evolution after the first.
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
    painter.add(egui::Shape::line(
        series(|s| s.mean),
        Stroke::new(1., TEXT_COLOR2),
    ));
    painter.add(egui::Shape::line(
        series(|s| s.best),
        Stroke::new(2., OK_GREEN),
    ));

    // Legend of the series (top right, in their colors).
    let mut legend = egui::text::LayoutJob::default();
    for (name, color) in [("best  ", OK_GREEN), ("mean", TEXT_COLOR2)] {
        legend.append(name, 0., text_format(color));
    }
    let galley = painter.layout_job(legend);
    painter.galley(
        rect.right_top() + Vec2::new(-4. - galley.size().x, 2.),
        galley,
        TEXT_COLOR2,
    );
}
