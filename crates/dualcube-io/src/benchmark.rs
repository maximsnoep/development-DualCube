//! Benchmarks: the optimization (as in the GUI) of a set of models, a number of runs per model, each for a fixed
//! time, with the results in a folder per session:
//!
//! - `session.json`: the configuration and the result of every run (read back with [`Session::load`]),
//! - `results.md`: the results as tables (per model, and per run), `results.csv`: one row per run,
//! - `solutions/<model>_<run>.dc`: the result of every run (to open in the GUI),
//! - `figures/<model>_<run>_<figure>.png`: its figures (loops, segmentation, polycube), and small versions of them
//!   (`..._small.png`, for tables).
//!
//! The session files are written after every run, so a stopped (or crashed) session keeps the runs done so far.

use crate::figure::{self, Figure, FigureParams, FigureStyle};
use anyhow::Context;
use dualcube::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// The figures saved per run.
pub const FIGURES: [Figure; 3] = [Figure::LoopStructure, Figure::Segmentation, Figure::Polycube];
// The size of the figures and of their small versions (points; the PNGs have 4 pixels per point).
const FIGURE_SIZE: f64 = 200.;
const SMALL_SIZE: f64 = 40.;

/// What a benchmark runs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// A name for the session (part of its folder's name).
    pub name: String,
    /// The models (meshes: obj or stl).
    pub models: Vec<PathBuf>,
    /// Runs per model.
    pub runs: usize,
    /// The time of the optimization per run (after the initialization), in seconds.
    pub seconds: f64,
    pub params: CoupledParams,
    pub quality: QualityParams,
}

/// The result of one run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunResult {
    /// The name of the model (unique in the session).
    pub model: String,
    pub input: PathBuf,
    /// The run (from 1).
    pub run: usize,
    pub quality: Option<f64>,
    pub loops: usize,
    pub initialization_seconds: f64,
    pub optimization_seconds: f64,
    /// Steps of the annealing and generations of the layout done.
    pub steps: usize,
    pub error: Option<String>,
    /// The saved solution and figures (and their small versions), relative to the session's folder.
    pub solution: Option<PathBuf>,
    pub figures: Vec<PathBuf>,
    pub small_figures: Vec<PathBuf>,
}

/// A benchmark session: its configuration and the runs done.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Session {
    pub config: BenchmarkConfig,
    /// When it started and (if so) finished, as YYYY-MM-DD HH:MM:SS (UTC).
    pub started: String,
    pub finished: Option<String>,
    pub runs: Vec<RunResult>,
}

/// The results of a model over its runs.
#[derive(Clone, Debug)]
pub struct ModelSummary {
    pub model: String,
    pub runs: usize,
    /// Runs with a result.
    pub succeeded: usize,
    pub best: Option<f64>,
    pub mean: Option<f64>,
    pub worst: Option<f64>,
    /// The index (in `Session::runs`) of the best run.
    pub best_run: Option<usize>,
    pub mean_loops: f64,
    pub mean_seconds: f64,
}

impl Session {
    /// Reads a session from its folder.
    pub fn load(dir: &Path) -> anyhow::Result<Self> {
        let path = dir.join("session.json");
        let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }

    /// Writes the session files (see the module documentation) into its folder.
    pub fn save(&self, dir: &Path) -> anyhow::Result<()> {
        std::fs::write(dir.join("session.json"), serde_json::to_string_pretty(self)?)?;
        std::fs::write(dir.join("results.csv"), self.csv())?;
        std::fs::write(dir.join("results.md"), self.markdown())?;
        Ok(())
    }

    /// The results per model, in the order of the configuration.
    #[must_use]
    pub fn summaries(&self) -> Vec<ModelSummary> {
        let mut models: Vec<&str> = vec![];
        for run in &self.runs {
            if !models.contains(&run.model.as_str()) {
                models.push(&run.model);
            }
        }
        models
            .into_iter()
            .map(|model| {
                let runs = self
                    .runs
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| r.model == model)
                    .collect::<Vec<_>>();
                let qualities = runs.iter().filter_map(|(_, r)| r.quality).collect::<Vec<_>>();
                let best_run = runs
                    .iter()
                    .filter(|(_, r)| r.quality.is_some())
                    .max_by(|(_, a), (_, b)| a.quality.unwrap_or(f64::NEG_INFINITY).total_cmp(&b.quality.unwrap_or(f64::NEG_INFINITY)))
                    .map(|(i, _)| *i);
                let n = runs.len().max(1) as f64;
                ModelSummary {
                    model: model.to_owned(),
                    runs: runs.len(),
                    succeeded: qualities.len(),
                    best: qualities.iter().copied().reduce(f64::max),
                    mean: (!qualities.is_empty())
                        .then(|| qualities.iter().sum::<f64>() / qualities.len() as f64),
                    worst: qualities.iter().copied().reduce(f64::min),
                    best_run,
                    mean_loops: runs.iter().map(|(_, r)| r.loops as f64).sum::<f64>() / n,
                    mean_seconds: runs
                        .iter()
                        .map(|(_, r)| r.initialization_seconds + r.optimization_seconds)
                        .sum::<f64>()
                        / n,
                }
            })
            .collect()
    }

    fn csv(&self) -> String {
        let mut csv = String::from(
            "model,run,quality,loops,initialization_seconds,optimization_seconds,steps,error,solution,input\n",
        );
        for r in &self.runs {
            let _ = writeln!(
                csv,
                "{},{},{},{},{:.2},{:.2},{},{},{},{}",
                r.model,
                r.run,
                r.quality.map_or(String::new(), |q| format!("{q:.6}")),
                r.loops,
                r.initialization_seconds,
                r.optimization_seconds,
                r.steps,
                r.error.as_deref().unwrap_or("").replace(',', ";"),
                r.solution.as_ref().map_or(String::new(), |p| p.display().to_string()),
                r.input.display()
            );
        }
        csv
    }

    fn markdown(&self) -> String {
        let quality = |q: Option<f64>| q.map_or("-".to_owned(), |q| format!("{q:.4}"));
        let c = &self.config;
        let mut md = format!(
            "# Benchmark {}\n\nStarted {}{}. {} models, {} runs per model, {} s of optimization per run \
             ({} steps of the loops and {} generations of the layout per cycle, {} children per step).\n\n",
            c.name,
            self.started,
            self.finished.as_ref().map_or(" (not finished)".to_owned(), |f| format!(", finished {f}")),
            c.models.len(),
            c.runs,
            c.seconds,
            c.params.loops.steps,
            c.params.layout_generations,
            c.params.loops.offspring,
        );
        md.push_str("## Per model\n\n| model | runs | best | mean | worst | loops | seconds |\n|---|---:|---:|---:|---:|---:|---:|\n");
        for s in self.summaries() {
            let _ = writeln!(
                md,
                "| {} | {}/{} | {} | {} | {} | {:.1} | {:.0} |",
                s.model,
                s.succeeded,
                s.runs,
                quality(s.best),
                quality(s.mean),
                quality(s.worst),
                s.mean_loops,
                s.mean_seconds
            );
        }
        md.push_str("\n## Per run\n\n| model | run | quality | loops | seconds | figures | solution | error |\n|---|---:|---:|---:|---:|---|---|---|\n");
        for r in &self.runs {
            // (The small figures, linking to the large ones.)
            let figures = r
                .small_figures
                .iter()
                .zip(&r.figures)
                .map(|(small, large)| {
                    let link = |p: &PathBuf| p.display().to_string().replace('\\', "/");
                    format!("[![]({})]({})", link(small), link(large))
                })
                .collect::<Vec<_>>()
                .join(" ");
            let _ = writeln!(
                md,
                "| {} | {} | {} | {} | {:.0} | {figures} | {} | {} |",
                r.model,
                r.run,
                quality(r.quality),
                r.loops,
                r.initialization_seconds + r.optimization_seconds,
                r.solution.as_ref().map_or(String::new(), |p| p.display().to_string()),
                r.error.as_deref().unwrap_or("")
            );
        }
        md
    }
}

/// The state of a running benchmark, shared with whoever shows it (and can stop it).
#[derive(Clone, Default)]
pub struct BenchmarkProgress(Arc<Mutex<ProgressState>>);

#[derive(Default)]
pub struct ProgressState {
    /// The folder of the session, and the session so far.
    pub dir: Option<PathBuf>,
    pub session: Option<Session>,
    /// The run in progress (model, run), and the runs done of all.
    pub current: Option<(String, usize)>,
    pub done: usize,
    pub total: usize,
    pub finished: bool,
    pub error: Option<String>,
    stop: bool,
    monitor: Option<OptimizationMonitor>,
}

impl BenchmarkProgress {
    #[must_use]
    pub fn state(&self) -> MutexGuard<'_, ProgressState> {
        self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Stops the benchmark after the optimization of the current run (which keeps its result so far).
    pub fn stop(&self) {
        let mut state = self.state();
        state.stop = true;
        if let Some(monitor) = &state.monitor {
            monitor.stop();
        }
    }

    #[must_use]
    pub fn stopping(&self) -> bool {
        self.state().stop
    }
}

/// The date and time (UTC) as YYYY-MM-DD HH:MM:SS.
#[must_use]
pub fn now() -> String {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let day = seconds % 86_400;
    format!(
        "{} {:02}:{:02}:{:02}",
        figure::today(),
        day / 3600,
        day / 60 % 60,
        day % 60
    )
}

/// A name usable in file names (letters, digits, `-` and `_`).
fn file_name(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect()
}

/// The meshes (obj and stl) in a folder (not its subfolders), sorted by name; of a model in both formats, only the
/// obj.
#[must_use]
pub fn models_in(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return vec![];
    };
    let mut models = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("obj") || e.eq_ignore_ascii_case("stl"))
        })
        .collect::<Vec<_>>();
    models.sort();
    let objs = models
        .iter()
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("obj")))
        .map(|p| p.with_extension(""))
        .collect::<std::collections::BTreeSet<_>>();
    models.retain(|p| {
        p.extension().is_some_and(|e| e.eq_ignore_ascii_case("obj")) || !objs.contains(&p.with_extension(""))
    });
    models
}

/// Runs a benchmark: creates its session folder in `output` (named by its start time and name), and runs every model
/// the given number of times, reporting to (and stoppable through) `progress`. Returns the session's folder.
pub fn run_benchmark(
    config: BenchmarkConfig,
    output: &Path,
    style: &FigureStyle,
    progress: &BenchmarkProgress,
) -> anyhow::Result<PathBuf> {
    let started = now();
    let dir = output.join(format!(
        "{}_{}",
        started.replace([' ', ':'], "-"),
        file_name(&config.name)
    ));
    for sub in ["solutions", "figures"] {
        std::fs::create_dir_all(dir.join(sub)).with_context(|| format!("creating {}", dir.display()))?;
    }
    let session = Session {
        config,
        started,
        finished: None,
        runs: vec![],
    };
    run_session(session, dir, style, progress)
}

/// Continues a benchmark session in its folder (e.g., one that was stopped): runs the runs it does not have yet.
pub fn resume_benchmark(dir: &Path, style: &FigureStyle, progress: &BenchmarkProgress) -> anyhow::Result<PathBuf> {
    let mut session = Session::load(dir)?;
    session.finished = None;
    run_session(session, dir.to_path_buf(), style, progress)
}

// Runs the runs of the session that it does not have yet (see `run_benchmark`).
fn run_session(
    mut session: Session,
    dir: PathBuf,
    style: &FigureStyle,
    progress: &BenchmarkProgress,
) -> anyhow::Result<PathBuf> {
    let config = session.config.clone();
    // Unique names: the file name, with its folder if another model has the same file name.
    let stems = config
        .models
        .iter()
        .map(|p| p.file_stem().map_or_else(String::new, |s| s.to_string_lossy().into_owned()))
        .collect::<Vec<_>>();
    let names = config
        .models
        .iter()
        .zip(&stems)
        .map(|(path, stem)| {
            if stems.iter().filter(|s| *s == stem).count() > 1 {
                let parent = path
                    .parent()
                    .and_then(Path::file_name)
                    .map_or_else(String::new, |s| s.to_string_lossy().into_owned());
                format!("{parent}-{stem}")
            } else {
                stem.clone()
            }
        })
        .collect::<Vec<_>>();

    session.save(&dir)?;
    {
        let mut state = progress.state();
        state.dir = Some(dir.clone());
        state.session = Some(session.clone());
        state.total = config.models.len() * config.runs;
        state.done = session.runs.len();
    }
    'models: for (path, name) in config.models.iter().zip(&names) {
        for run in 1..=config.runs {
            // (Done already, when resumed.)
            if session.runs.iter().any(|r| r.model == *name && r.run == run) {
                continue;
            }
            if progress.stopping() {
                break 'models;
            }
            progress.state().current = Some((name.clone(), run));
            info!("benchmark: {name}, run {run}");
            let result = run_once(&config, path, name, run, &dir, style, progress);
            session.runs.push(result);
            session.save(&dir)?;
            let mut state = progress.state();
            state.session = Some(session.clone());
            state.done += 1;
        }
    }
    session.finished = Some(now());
    session.save(&dir)?;
    let mut state = progress.state();
    state.session = Some(session);
    state.current = None;
    state.finished = true;
    Ok(dir)
}

// One run: initialize, optimize for the configured time, and save the result and its figures.
fn run_once(
    config: &BenchmarkConfig,
    path: &Path,
    name: &str,
    run: usize,
    dir: &Path,
    style: &FigureStyle,
    progress: &BenchmarkProgress,
) -> RunResult {
    let mut result = RunResult {
        model: name.to_owned(),
        input: path.to_path_buf(),
        run,
        quality: None,
        loops: 0,
        initialization_seconds: 0.,
        optimization_seconds: 0.,
        steps: 0,
        error: None,
        solution: None,
        figures: vec![],
        small_figures: vec![],
    };
    let mut solution = match crate::import_solution(path) {
        Ok(solution) => solution,
        Err(err) => {
            result.error = Some(format!("reading the model: {err:#}"));
            return result;
        }
    };
    solution.quality = config.quality;
    let timer = Instant::now();
    solution.initialize();
    result.initialization_seconds = timer.elapsed().as_secs_f64();
    if solution.layout.is_none() {
        result.error = Some("the initialization failed".to_owned());
        return result;
    }

    // The optimization, on a thread of its own, stopped after the configured time.
    let monitor = OptimizationMonitor::default();
    progress.state().monitor = Some(monitor.clone());
    let timer = Instant::now();
    let (start, m, params) = (solution.clone(), monitor.clone(), config.params);
    let handle = std::thread::spawn(move || start.optimize(&params, &m));
    while timer.elapsed().as_secs_f64() < config.seconds && !handle.is_finished() && !monitor.stopping() {
        std::thread::sleep(Duration::from_millis(100));
    }
    monitor.stop();
    let optimized = handle.join();
    progress.state().monitor = None;
    result.optimization_seconds = timer.elapsed().as_secs_f64();
    result.steps = monitor.history().len();
    let optimized = match optimized {
        Ok(Ok(optimized)) => optimized.with_smooth_loops(),
        Ok(Err(err)) => {
            result.error = Some(format!("the optimization failed: {err}"));
            solution
        }
        Err(_) => {
            result.error = Some("the optimization panicked".to_owned());
            solution
        }
    };
    result.quality = optimized.get_quality();
    result.loops = optimized.loops.len();

    // The solution, and its figures (with smoothed paths, as the GUI shows them).
    let stem = format!("{}_{run}", file_name(name));
    let file = PathBuf::from("solutions").join(format!("{stem}.dc"));
    match crate::export_solution(&optimized, &dir.join(&file)) {
        Ok(()) => result.solution = Some(file),
        Err(err) => result.error = Some(format!("saving the solution: {err:#}")),
    }
    let mut shown = optimized.clone();
    if shown.smooth_layout().is_err() {
        shown = optimized;
    }
    for (size, small) in [(FIGURE_SIZE, false), (SMALL_SIZE, true)] {
        let params = FigureParams { size, ..FigureParams::default() };
        for figure in FIGURES {
            let Some(svg) = figure::figure_svg(&shown, figure, &params, style) else {
                continue;
            };
            let suffix = if small { "_small" } else { "" };
            let file = PathBuf::from("figures").join(format!("{stem}_{}{suffix}.png", figure.slug()));
            match figure::save_figure(&svg, &dir.join(&file)) {
                Ok(()) if small => result.small_figures.push(file),
                Ok(()) => result.figures.push(file),
                Err(err) => warn!("benchmark: saving {}: {err:#}", file.display()),
            }
        }
    }
    result
}

/// Reads a PNG (e.g., a figure) as straight (not premultiplied) RGBA pixels: (width, height, pixels).
pub fn read_png(path: &Path) -> anyhow::Result<(usize, usize, Vec<u8>)> {
    let pixmap = resvg::tiny_skia::Pixmap::load_png(path).with_context(|| format!("reading {}", path.display()))?;
    let (width, height) = (pixmap.width() as usize, pixmap.height() as usize);
    let pixels = pixmap
        .pixels()
        .iter()
        .flat_map(|p| {
            let c = p.demultiply();
            [c.red(), c.green(), c.blue(), c.alpha()]
        })
        .collect();
    Ok((width, height, pixels))
}
