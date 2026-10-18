//! The benchmark window (see `io::benchmark`): the setup of a benchmark (the models, from folders or files, the runs
//! per model, the time per run, the parameters of the optimization, and the folder for the results), its progress, and
//! the results of a session (the running one, or one read from its folder): a table per model, sortable by every
//! column, with the figures of the best run; a model expands to its runs. Clicking a figure (or `open`) opens the
//! solution of that run in the GUI; hovering a figure shows it large.

use super::theme::{DIM_COLOR, OK_GREEN, TEXT_COLOR, TEXT_COLOR2, WARN_RED, window_frame};
use super::widgets::{collapsed, label, sep, sleek_button, sleek_button_warn, space, window_title};
use crate::jobs::{Job, figure_style};
use crate::resources::Configuration;
use bevy::prelude::*;
use bevy_egui::egui::{self, Grid, Sense, TextureHandle, TextureOptions, Vec2};
use dualcube::prelude::*;
use io::benchmark::{BenchmarkConfig, BenchmarkProgress, Session, models_in, read_png, run_benchmark};
use rustc_hash::FxHashMap as HashMap;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

// The size of the small figures in the table, and of a figure shown on hover (points).
const SMALL: f32 = 40.;
const LARGE: f32 = 360.;
// Figures read per frame (reading takes a few milliseconds each).
const READS_PER_FRAME: usize = 6;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Column {
    Model,
    Runs,
    Best,
    Mean,
    Worst,
    Loops,
    Seconds,
}

/// The state of the benchmark window.
#[derive(Resource)]
pub struct BenchmarkUi {
    pub open: bool,
    /// The models, and whether they are run.
    models: Vec<(PathBuf, bool)>,
    name: String,
    runs: usize,
    seconds: f64,
    output: PathBuf,
    /// The running (or last) benchmark.
    progress: Option<BenchmarkProgress>,
    /// The session shown (its folder and contents), if read from a folder.
    shown: Option<(PathBuf, Session)>,
    sort: (Column, bool),
    expanded: BTreeSet<String>,
    // The figures read (`None`: not readable).
    textures: HashMap<PathBuf, Option<TextureHandle>>,
}

impl Default for BenchmarkUi {
    fn default() -> Self {
        Self {
            open: false,
            models: vec![],
            name: "benchmark".to_owned(),
            runs: 3,
            seconds: 60.,
            output: std::env::current_dir().unwrap_or_default().join("benchmarks"),
            progress: None,
            shown: None,
            sort: (Column::Best, false),
            expanded: BTreeSet::new(),
            textures: HashMap::default(),
        }
    }
}

impl BenchmarkUi {
    fn running(&self) -> bool {
        self.progress.as_ref().is_some_and(|p| !p.state().finished)
    }
}

/// Shows the benchmark window once it was opened (with the benchmark button of the menu).
pub fn show(
    mut egui_ctx: bevy_egui::EguiContexts<'_, '_>,
    mut state: ResMut<'_, BenchmarkUi>,
    mut conf: ResMut<'_, Configuration>,
    mut jobs: MessageWriter<'_, Job>,
) -> Result<(), BevyError> {
    if !state.open {
        return Ok(());
    }
    let state = &mut *state;
    let ctx = egui_ctx.ctx_mut()?;
    let is_collapsed = collapsed(ctx, "Benchmark");
    let mut open = None;
    let mut reads = READS_PER_FRAME;
    egui::Window::new("benchmark")
        .title_bar(false)
        .frame(window_frame())
        .default_width(620.)
        .min_width(520.)
        .default_pos(super::main_area(ctx).left_top() + Vec2::new(15., 45.))
        .resizable(!is_collapsed)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                window_title(ui, "Benchmark");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if sleek_button(ui, "close") {
                        state.open = false;
                    }
                });
            });
            if is_collapsed {
                return;
            }
            sep(ui);
            setup(ui, state, &mut conf);
            sep(ui);
            results(ui, state, &mut open, &mut reads);
        });
    if let Some(path) = open {
        jobs.write(Job::import(path, conf.clone()));
    }
    Ok(())
}

// The setup: the models, the runs, the time, the parameters, the output folder, and start (or stop, and progress).
fn setup(ui: &mut egui::Ui, state: &mut BenchmarkUi, conf: &mut Configuration) {
    let running = state.running();
    ui.horizontal(|ui| {
        label(ui, "Models", TEXT_COLOR);
        label(
            ui,
            &format!(
                "{} of {} selected",
                state.models.iter().filter(|(_, on)| *on).count(),
                state.models.len()
            ),
            TEXT_COLOR2,
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if !state.models.is_empty() && sleek_button(ui, "clear") {
                state.models.clear();
            }
            if sleek_button(ui, "add files")
                && let Some(files) = rfd::FileDialog::new()
                    .add_filter("mesh", &["obj", "stl"])
                    .pick_files()
            {
                add_models(state, files);
            }
            if sleek_button(ui, "add folder")
                && let Some(dirs) = rfd::FileDialog::new().pick_folders()
            {
                for dir in dirs {
                    add_models(state, models_in(&dir));
                }
            }
        });
    });
    if !state.models.is_empty() {
        egui::ScrollArea::vertical()
            .id_salt("benchmark models")
            .max_height(120.)
            .show(ui, |ui| {
                let mut remove = None;
                for (i, (path, on)) in state.models.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        ui.checkbox(on, "");
                        let folder = path
                            .parent()
                            .and_then(Path::file_name)
                            .map_or_else(String::new, |f| f.to_string_lossy().into_owned());
                        let name = path
                            .file_name()
                            .map_or_else(String::new, |f| f.to_string_lossy().into_owned());
                        label(ui, &name, if *on { TEXT_COLOR } else { DIM_COLOR });
                        label(ui, &folder, DIM_COLOR);
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if sleek_button(ui, "x") {
                                remove = Some(i);
                            }
                        });
                    });
                }
                if let Some(i) = remove {
                    state.models.remove(i);
                }
            });
        ui.horizontal(|ui| {
            if sleek_button(ui, "all") {
                state.models.iter_mut().for_each(|(_, on)| *on = true);
            }
            if sleek_button(ui, "none") {
                state.models.iter_mut().for_each(|(_, on)| *on = false);
            }
        });
    }
    space(ui);

    // The runs and the optimization (its parameters are those of the optimization window).
    Grid::new("benchmark settings")
        .num_columns(4)
        .spacing([12., 4.])
        .show(ui, |ui| {
            label(ui, "name", TEXT_COLOR2);
            ui.add(egui::TextEdit::singleline(&mut state.name).desired_width(140.));
            label(ui, "runs per model", TEXT_COLOR2);
            ui.add(egui::DragValue::new(&mut state.runs).range(1..=100));
            ui.end_row();
            label(ui, "seconds per run", TEXT_COLOR2);
            ui.add(egui::DragValue::new(&mut state.seconds).range(1.0..=36_000.).speed(1.).suffix(" s"));
            label(ui, "children per step", TEXT_COLOR2);
            ui.add(egui::DragValue::new(&mut conf.anneal.offspring).range(1..=256));
            ui.end_row();
            label(ui, "dual generations", TEXT_COLOR2);
            ui.add(egui::DragValue::new(&mut conf.anneal.steps).range(0..=1000));
            label(ui, "primal generations", TEXT_COLOR2);
            ui.add(egui::DragValue::new(&mut conf.layout_generations).range(0..=1000));
            ui.end_row();
        });
    label(
        ui,
        "(The quality weights and the mutations switched on are those of the optimization window.)",
        DIM_COLOR,
    );
    ui.horizontal(|ui| {
        label(ui, "results in", TEXT_COLOR2);
        label(ui, &state.output.display().to_string(), TEXT_COLOR);
        if sleek_button(ui, "change")
            && let Some(dir) = rfd::FileDialog::new().set_directory(&state.output).pick_folder()
        {
            state.output = dir;
        }
    });
    space(ui);

    // Start, or stop and the progress.
    let selected = state.models.iter().filter(|(_, on)| *on).map(|(p, _)| p.clone()).collect::<Vec<_>>();
    ui.horizontal(|ui| {
        if running {
            if let Some(progress) = &state.progress {
                if progress.stopping() {
                    label(ui, "stopping after this run", WARN_RED);
                } else if sleek_button_warn(ui, "stop") {
                    progress.stop();
                }
                let s = progress.state();
                let current = s
                    .current
                    .as_ref()
                    .map_or_else(String::new, |(model, run)| format!(": {model}, run {run}"));
                label(ui, &format!("run {} of {}{current}", s.done + 1, s.total), TEXT_COLOR2);
            }
        } else if selected.is_empty() {
            label(ui, "start", DIM_COLOR);
            label(ui, "(select models first)", DIM_COLOR);
        } else if sleek_button(ui, "start") {
            let config = BenchmarkConfig {
                name: state.name.clone(),
                models: selected.clone(),
                runs: state.runs,
                seconds: state.seconds,
                params: CoupledParams {
                    loops: conf.anneal,
                    layout: conf.layout_evolution.without(&conf.layout_disabled),
                    layout_generations: conf.layout_generations,
                    max_cycles: usize::MAX,
                    patience: conf.patience,
                },
                quality: conf.quality,
            };
            let progress = BenchmarkProgress::default();
            let (output, p) = (state.output.clone(), progress.clone());
            std::thread::spawn(move || {
                if let Err(err) = run_benchmark(config, &output, &figure_style(), &p) {
                    warn!("benchmark failed: {err:#}");
                    let mut s = p.state();
                    s.error = Some(format!("{err:#}"));
                    s.finished = true;
                }
            });
            state.progress = Some(progress);
            state.shown = None;
            state.expanded.clear();
        } else {
            label(
                ui,
                &format!(
                    "{} runs, about {}",
                    selected.len() * state.runs,
                    duration(selected.len() as f64 * state.runs as f64 * (state.seconds + 2.))
                ),
                TEXT_COLOR2,
            );
        }
        if let Some(error) = state.progress.as_ref().and_then(|p| p.state().error.clone()) {
            label(ui, &error, WARN_RED);
        }
    });
}

// Adds models (not those already there), switched on.
fn add_models(state: &mut BenchmarkUi, paths: Vec<PathBuf>) {
    for path in paths {
        if !state.models.iter().any(|(p, _)| *p == path) {
            state.models.push((path, true));
        }
    }
}

fn duration(seconds: f64) -> String {
    if seconds < 90. {
        format!("{seconds:.0} s")
    } else if seconds < 5400. {
        format!("{:.0} min", seconds / 60.)
    } else {
        format!("{:.1} h", seconds / 3600.)
    }
}

// The results: of the running (or last) benchmark, or of a session read from its folder.
fn results(ui: &mut egui::Ui, state: &mut BenchmarkUi, open: &mut Option<PathBuf>, reads: &mut usize) {
    ui.horizontal(|ui| {
        label(ui, "Results", TEXT_COLOR);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if sleek_button(ui, "open results")
                && let Some(dir) = rfd::FileDialog::new().set_directory(&state.output).pick_folder()
            {
                match Session::load(&dir) {
                    Ok(session) => {
                        state.shown = Some((dir, session));
                        state.expanded.clear();
                    }
                    Err(err) => warn!("Failed to read the benchmark results: {err:#}"),
                }
            }
        });
    });
    let session = state.shown.clone().or_else(|| {
        let progress = state.progress.as_ref()?;
        let s = progress.state();
        Some((s.dir.clone()?, s.session.clone()?))
    });
    let Some((dir, session)) = session else {
        label(ui, "Run a benchmark, or open the results of one (its folder).", DIM_COLOR);
        return;
    };
    label(
        ui,
        &format!(
            "{} · {} · {} runs{}",
            dir.file_name().map_or_else(String::new, |f| f.to_string_lossy().into_owned()),
            session.started,
            session.runs.len(),
            if session.finished.is_some() { "" } else { " (not finished)" }
        ),
        TEXT_COLOR2,
    );
    space(ui);

    let mut summaries = session.summaries();
    let (column, ascending) = state.sort;
    let key = |s: &io::benchmark::ModelSummary| match column {
        Column::Model => 0.,
        Column::Runs => s.runs as f64,
        Column::Best => s.best.unwrap_or(f64::NEG_INFINITY),
        Column::Mean => s.mean.unwrap_or(f64::NEG_INFINITY),
        Column::Worst => s.worst.unwrap_or(f64::NEG_INFINITY),
        Column::Loops => s.mean_loops,
        Column::Seconds => s.mean_seconds,
    };
    summaries.sort_by(|a, b| {
        let order = if column == Column::Model {
            a.model.cmp(&b.model)
        } else {
            key(a).total_cmp(&key(b))
        };
        if ascending { order } else { order.reverse() }
    });

    egui::ScrollArea::vertical()
        .id_salt("benchmark results")
        .max_height(520.)
        .show(ui, |ui| {
            Grid::new("benchmark table")
                .num_columns(9)
                .spacing([14., 6.])
                .striped(true)
                .show(ui, |ui| {
                    label(ui, "", TEXT_COLOR2);
                    label(ui, "figures (best run)", TEXT_COLOR2);
                    for (name, c) in [
                        ("model", Column::Model),
                        ("runs", Column::Runs),
                        ("best", Column::Best),
                        ("mean", Column::Mean),
                        ("worst", Column::Worst),
                        ("loops", Column::Loops),
                        ("time", Column::Seconds),
                    ] {
                        let arrow = match state.sort {
                            (s, true) if s == c => " ^",
                            (s, false) if s == c => " v",
                            _ => "",
                        };
                        if ui
                            .add(egui::Label::new(egui::RichText::new(format!("{name}{arrow}")).color(TEXT_COLOR2)).sense(Sense::click()))
                            .on_hover_text("sort")
                            .clicked()
                        {
                            state.sort = if state.sort.0 == c {
                                (c, !state.sort.1)
                            } else {
                                // Names first A to Z, numbers first the highest.
                                (c, c == Column::Model)
                            };
                        }
                    }
                    ui.end_row();

                    let quality = |q: Option<f64>| q.map_or("-".to_owned(), |q| format!("{q:.4}"));
                    for s in &summaries {
                        let expanded = state.expanded.contains(&s.model);
                        if ui.selectable_label(expanded, if expanded { "v" } else { ">" }).clicked() {
                            if expanded {
                                state.expanded.remove(&s.model);
                            } else {
                                state.expanded.insert(s.model.clone());
                            }
                        }
                        let best = s.best_run.map(|i| &session.runs[i]);
                        ui.horizontal(|ui| {
                            if let Some(run) = best {
                                figures(ui, state, &dir, run, open, reads);
                            }
                        });
                        label(ui, &s.model, TEXT_COLOR);
                        label(
                            ui,
                            &format!("{}/{}", s.succeeded, s.runs),
                            if s.succeeded < s.runs { WARN_RED } else { TEXT_COLOR },
                        );
                        label(ui, &quality(s.best), OK_GREEN);
                        label(ui, &quality(s.mean), TEXT_COLOR);
                        label(ui, &quality(s.worst), TEXT_COLOR);
                        label(ui, &format!("{:.1}", s.mean_loops), TEXT_COLOR);
                        label(ui, &duration(s.mean_seconds), TEXT_COLOR);
                        ui.end_row();

                        if expanded {
                            for run in session.runs.iter().filter(|r| r.model == s.model) {
                                label(ui, "", TEXT_COLOR2);
                                ui.horizontal(|ui| figures(ui, state, &dir, run, open, reads));
                                ui.horizontal(|ui| {
                                    label(ui, &format!("run {}", run.run), TEXT_COLOR2);
                                    if let Some(solution) = &run.solution
                                        && sleek_button(ui, "open")
                                    {
                                        *open = Some(dir.join(solution));
                                    }
                                });
                                label(ui, "", TEXT_COLOR2);
                                label(ui, &quality(run.quality), TEXT_COLOR2);
                                match &run.error {
                                    Some(error) => label(ui, error, WARN_RED),
                                    None => label(ui, "", TEXT_COLOR2),
                                }
                                label(ui, "", TEXT_COLOR2);
                                label(ui, &run.loops.to_string(), TEXT_COLOR2);
                                label(
                                    ui,
                                    &duration(run.initialization_seconds + run.optimization_seconds),
                                    TEXT_COLOR2,
                                );
                                ui.end_row();
                            }
                        }
                    }
                });
        });
}

// The small figures of a run: hovering one shows it large, clicking one opens the run's solution.
fn figures(
    ui: &mut egui::Ui,
    state: &mut BenchmarkUi,
    dir: &Path,
    run: &io::benchmark::RunResult,
    open: &mut Option<PathBuf>,
    reads: &mut usize,
) {
    for (small, large) in run.small_figures.iter().zip(&run.figures) {
        let Some(small_texture) = texture(ui.ctx(), state, &dir.join(small), reads) else {
            ui.allocate_exact_size(Vec2::splat(SMALL), Sense::hover());
            continue;
        };
        let response = ui.add(egui::Image::new((small_texture.id(), Vec2::splat(SMALL))).sense(Sense::click()));
        // (The large figure is read when it is first hovered.)
        let large = if response.hovered() {
            let mut unlimited = usize::MAX;
            texture(ui.ctx(), state, &dir.join(large), &mut unlimited).map(|t| t.id())
        } else {
            None
        };
        let response = response.on_hover_ui(|ui| {
            if let Some(id) = large {
                ui.add(egui::Image::new((id, Vec2::splat(LARGE))));
            }
            label(ui, "click to open this run", DIM_COLOR);
        });
        if response.clicked()
            && let Some(solution) = &run.solution
        {
            *open = Some(dir.join(solution));
        }
    }
}

// The texture of a figure, read once (at most `reads` per frame; `None` until read, or if it cannot be read).
fn texture(
    ctx: &egui::Context,
    state: &mut BenchmarkUi,
    path: &Path,
    reads: &mut usize,
) -> Option<TextureHandle> {
    if let Some(texture) = state.textures.get(path) {
        return texture.clone();
    }
    if *reads == 0 {
        ctx.request_repaint();
        return None;
    }
    *reads -= 1;
    let texture = read_png(path).ok().map(|(width, height, pixels)| {
        ctx.load_texture(
            path.display().to_string(),
            egui::ColorImage::from_rgba_unmultiplied([width, height], &pixels),
            TextureOptions::LINEAR,
        )
    });
    state.textures.insert(path.to_path_buf(), texture.clone());
    texture
}
