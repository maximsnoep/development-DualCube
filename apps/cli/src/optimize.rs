//! `optimize`: run or sample the skeleton optimization on many meshes and compare the
//! restricted-label fidelity with the reconstructed polycube's quality (as drawn).

use dualcube::prelude::*;
use dualcube::skeleton::optimize::{self, Measure, Params, Tables};
use dualcube::skeleton::{LabeledSkeleton, Skeleton, certify, draw_loops, label, place_cubes};
use std::hash::{Hash, Hasher};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Optimize every mesh once and compare with the unoptimized start.
    Run,
    /// Random valid labelings around the start, for rank correlation.
    Sample,
}

#[derive(Clone, Debug)]
pub struct Options {
    pub inputs: Vec<PathBuf>,
    pub report: Option<PathBuf>,
    pub cache: PathBuf,
    pub mode: Mode,
    pub samples: usize,
    pub steps: usize,
    /// Score candidates by the primalized quality instead of restricted-label fidelity.
    pub primal: bool,
    pub params: Params,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            inputs: Vec::new(),
            report: None,
            cache: PathBuf::from("target/optimize-cache"),
            mode: Mode::Run,
            samples: 12,
            steps: 4,
            primal: true,
            params: Params::default(),
        }
    }
}

type Row = Vec<(String, String)>;

/// The certified extracted skeleton, from the cache when possible. Extraction is not
/// deterministic, so cached skeletons also keep experiments comparable.
fn skeleton(mesh: &Mesh<INPUT>, input: &Path, cache: &Path) -> anyhow::Result<Skeleton> {
    let bytes = std::fs::read(input)?;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut hasher);
    let file = cache.join(format!("{:016x}.skeleton", hasher.finish()));
    if let Ok(data) = std::fs::read(&file)
        && let Ok(skeleton) = rmp_serde::from_slice::<Skeleton>(&data)
    {
        return Ok(skeleton);
    }
    let skeleton = dualcube::skeleton::skeletonize::run(mesh)
        .map_err(|e| anyhow::anyhow!("extraction: {e}"))?;
    certify::run(mesh, &skeleton).map_err(|e| anyhow::anyhow!("certification: {e}"))?;
    std::fs::create_dir_all(cache)?;
    std::fs::write(&file, rmp_serde::to_vec(&skeleton)?)?;
    Ok(skeleton)
}

/// Quality of the polycube reconstructed from a labeling, as drawn.
fn primal(solution: &Solution, mesh: &Mesh<INPUT>, labeled: &LabeledSkeleton) -> Row {
    let result = catch_unwind(AssertUnwindSafe(
        || -> anyhow::Result<(QualityReport, f64, Option<QualityReport>)> {
            let placed = place_cubes::run(mesh, labeled).map_err(|e| anyhow::anyhow!("{e}"))?;
            let drawing = draw_loops::attempt(mesh, &placed).map_err(|e| anyhow::anyhow!("{e}"))?;
            if let Some(failure) = drawing.failure {
                anyhow::bail!("drawing: {failure}");
            }
            let mut solution = solution.clone();
            let loops = drawing.loops.len();
            // Corner placement does not enter this estimate: every triangle takes the best label
            // among the corners of its loop region.
            let estimate = drawing.dual.as_ref().map(|dual| {
                let polycube = Polycube::from_dual(dual);
                let weights = solution.quality.weights;
                QualityReport::estimate(
                    loops,
                    dual,
                    &polycube,
                    &solution.quality,
                    QualityTerms::needed(&weights),
                )
            });
            solution.replace_loops(drawing.loops);
            if let Some(dual) = drawing.dual {
                solution.dual = Ok(dual);
            }
            solution.reconstruct_solution(false)?;
            let report = solution
                .quality_report()
                .ok_or_else(|| anyhow::anyhow!("incomplete layout"))?;
            let score = solution.get_quality().unwrap_or(f64::NAN);
            Ok((report, score, estimate))
        },
    ));
    let format = |v: Option<f64>| v.map(|v| format!("{v:.6}")).unwrap_or_default();
    match result {
        Ok(Ok((report, score, estimate))) => vec![
            (
                "e_alignment".into(),
                format(estimate.as_ref().and_then(|e| e.alignment)),
            ),
            (
                "e_distortion".into(),
                format(estimate.as_ref().and_then(|e| e.distortion)),
            ),
            ("p_alignment".into(), format(report.alignment)),
            ("p_distortion".into(), format(report.distortion)),
            ("p_score".into(), format(Some(score))),
            ("p_loops".into(), report.loops.to_string()),
            ("p_error".into(), String::new()),
        ],
        Ok(Err(e)) => error_row(&e.to_string()),
        Err(_) => error_row("panic"),
    }
}

/// The quality score of the polycube reconstructed from a labeling, as drawn.
fn primal_score(solution: &Solution, mesh: &Mesh<INPUT>, labeled: &LabeledSkeleton) -> Option<f64> {
    catch_unwind(AssertUnwindSafe(|| {
        let placed = place_cubes::run(mesh, labeled).ok()?;
        let drawing = draw_loops::attempt(mesh, &placed).ok()?;
        if drawing.failure.is_some() {
            return None;
        }
        let mut solution = solution.clone();
        solution.replace_loops(drawing.loops);
        solution.dual = Ok(drawing.dual?);
        solution.reconstruct_solution(false).ok()?;
        solution.get_quality()
    }))
    .ok()
    .flatten()
}

fn error_row(error: &str) -> Row {
    ["p_alignment", "p_distortion", "p_score", "p_loops"]
        .into_iter()
        .map(|k| (k.to_owned(), String::new()))
        .chain([("p_error".to_owned(), error.replace(['\n', ','], " "))])
        .collect()
}

fn criterion(mesh: &Mesh<INPUT>, labeled: &LabeledSkeleton, prefix: &str) -> Row {
    match Tables::build(mesh, &labeled.skeleton) {
        Ok(tables) => {
            let f = tables.evaluate(&labeled.skeleton, &labeled.labels);
            vec![
                (
                    format!("{prefix}nodes"),
                    labeled.skeleton.nodes.len().to_string(),
                ),
                (format!("{prefix}alignment"), format!("{:.6}", f.alignment)),
                (
                    format!("{prefix}distortion"),
                    format!("{:.6}", f.distortion),
                ),
                (
                    format!("{prefix}naive"),
                    format!("{:.6}", tables.naive.alignment),
                ),
            ]
        }
        Err(e) => vec![(format!("{prefix}error"), e.to_string())],
    }
}

fn run_mesh(input: &Path, options: &Options) -> anyhow::Result<Vec<Row>> {
    let solution = io::import_solution(input)?;
    let mesh = solution.mesh_ref.clone();
    let skeleton = skeleton(&mesh, input, &options.cache)?;
    let labeled = label::run(&mesh, &skeleton).map_err(|e| anyhow::anyhow!("labeling: {e}"))?;
    let start = optimize::dissolve_straight(&mesh, &labeled).map_err(|e| anyhow::anyhow!("{e}"))?;
    let name = input
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let base = || {
        vec![
            ("mesh".to_owned(), name.clone()),
            ("faces".into(), mesh.nr_faces().to_string()),
        ]
    };
    let mut rows = Vec::new();
    match options.mode {
        Mode::Run => {
            let timer = Instant::now();
            let objective = |l: &LabeledSkeleton| primal_score(&solution, &mesh, l);
            let objective: Option<optimize::Objective<'_>> = options
                .primal
                .then_some(&objective as optimize::Objective<'_>);
            let (result, stats) =
                optimize::search::search(&mesh, &start, &options.params, objective)
                    .map_err(|e| anyhow::anyhow!("search: {e}"))?;
            let seconds = timer.elapsed().as_secs_f64();
            let mut row = base();
            row.extend(criterion(&mesh, &start, "start_"));
            row.extend(
                primal(&solution, &mesh, &start)
                    .into_iter()
                    .map(|(k, v)| (format!("start_{k}"), v)),
            );
            row.extend(criterion(&mesh, &result, "opt_"));
            row.extend(
                primal(&solution, &mesh, &result)
                    .into_iter()
                    .map(|(k, v)| (format!("opt_{k}"), v)),
            );
            row.push((
                "sweep_alignment".into(),
                format!("{:.6}", stats.after_sweep.alignment),
            ));
            row.push(("score_start".into(), format!("{:.6}", stats.score[0])));
            row.push((
                "score_relaxed".into(),
                stats.relaxed.map(|s| format!("{s:.6}")).unwrap_or_default(),
            ));
            row.push(("score_opt".into(), format!("{:.6}", stats.score[1])));
            row.push(("steps".into(), stats.steps.to_string()));
            row.push(("seconds".into(), format!("{seconds:.2}")));
            for (operator, [tried, valid, accepted]) in
                optimize::Operator::ALL.iter().zip(stats.operators)
            {
                row.push((
                    format!("{operator:?}"),
                    format!("{accepted}/{valid}/{tried}"),
                ));
            }
            rows.push(row);
        }
        Mode::Sample => {
            for sample in 0..=options.samples {
                let labeled = if sample == 0 {
                    start.clone()
                } else {
                    optimize::search::perturb(
                        &mesh,
                        &start,
                        options.steps,
                        &options.params,
                        sample as u64,
                    )
                    .map_err(|e| anyhow::anyhow!("{e}"))?
                };
                let mut row = base();
                row.push(("sample".into(), sample.to_string()));
                row.extend(criterion(&mesh, &labeled, ""));
                row.extend(primal(&solution, &mesh, &labeled));
                rows.push(row);
            }
        }
    }
    Ok(rows)
}

fn write(path: &Path, rows: &[Row]) -> anyhow::Result<()> {
    let mut header: Vec<String> = Vec::new();
    for row in rows {
        for (k, _) in row {
            if !header.contains(k) {
                header.push(k.clone());
            }
        }
    }
    let mut text = header.join(",");
    text.push('\n');
    for row in rows {
        let values: Vec<_> = header
            .iter()
            .map(|h| {
                row.iter()
                    .find(|(k, _)| k == h)
                    .map(|(_, v)| crate::stages::csv_field(v))
                    .unwrap_or_default()
            })
            .collect();
        text.push_str(&values.join(","));
        text.push('\n');
    }
    std::fs::write(path, text)?;
    Ok(())
}

pub fn run(options: &Options) -> anyhow::Result<()> {
    let mut paths = Vec::new();
    for input in &options.inputs {
        paths.extend(crate::stages::meshes(input)?);
    }
    let mut rows = Vec::new();
    for (i, path) in paths.iter().enumerate() {
        let timer = Instant::now();
        let result = catch_unwind(AssertUnwindSafe(|| run_mesh(path, options)));
        let outcome = match result {
            Ok(Ok(new)) => {
                let summary = new
                    .first()
                    .map(|r| {
                        r.iter()
                            .filter(|(k, _)| {
                                k.contains("alignment")
                                    || k.contains("distortion")
                                    || k.ends_with("nodes")
                            })
                            .map(|(k, v)| format!("{k}={v}"))
                            .collect::<Vec<_>>()
                            .join(" ")
                    })
                    .unwrap_or_default();
                rows.extend(new);
                summary
            }
            Ok(Err(e)) => format!("failed: {e}"),
            Err(_) => "panic".to_owned(),
        };
        println!(
            "[{}/{}] {} ({:.1}s) {outcome}",
            i + 1,
            paths.len(),
            path.display(),
            timer.elapsed().as_secs_f64()
        );
        if let Some(report) = &options.report {
            write(report, &rows)?;
        }
    }
    Ok(())
}

pub fn parse_measure(value: &str) -> anyhow::Result<Measure> {
    match value {
        "alignment" => Ok(Measure::Alignment),
        "distortion" => Ok(Measure::Distortion),
        other => anyhow::bail!("unknown measure `{other}` (alignment or distortion)"),
    }
}
