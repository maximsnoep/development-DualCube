//! `stages`: run the skeleton pipeline on many meshes up to one stage, one report row per mesh.

use dualcube::prelude::*;
use dualcube::skeleton::pipeline::{Progress, Stage};
use dualcube::skeleton::skeletonize::{self, deflate};
use dualcube::skeleton::validate;
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// A stage by its name in [`Stage::name`].
pub fn parse_stage(name: &str) -> anyhow::Result<Stage> {
    Stage::parse(name).ok_or_else(|| {
        let names: Vec<_> = Stage::ALL.iter().map(|s| s.name()).collect();
        anyhow::anyhow!("unknown stage `{name}` (one of {})", names.join(", "))
    })
}

/// The meshes named by an input: a mesh file, a folder of meshes, or a `.txt` set listing
/// mesh paths relative to itself (`#` starts a comment).
pub fn meshes(input: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let is_mesh = |p: &Path| {
        p.extension()
            .and_then(|x| x.to_str())
            .is_some_and(|x| matches!(x.to_ascii_lowercase().as_str(), "obj" | "stl"))
    };
    if input.is_dir() {
        let mut paths: Vec<_> = std::fs::read_dir(input)?
            .map(|entry| entry.map(|e| e.path()))
            .collect::<Result<_, _>>()?;
        paths.retain(|p| p.is_file() && is_mesh(p));
        paths.sort();
        return Ok(paths);
    }
    if input.extension().is_some_and(|x| x == "txt") {
        let base = input.parent().unwrap_or(Path::new("."));
        return Ok(std::fs::read_to_string(input)?
            .lines()
            .map(|line| line.split('#').next().unwrap_or("").trim())
            .filter(|line| !line.is_empty())
            .map(|line| base.join(line))
            .collect());
    }
    Ok(vec![input.to_owned()])
}

/// One mesh's outcome. Statistics stay empty for stages that did not run.
#[derive(Clone, Debug, Default)]
pub struct Row {
    pub input: PathBuf,
    pub genus: Option<usize>,
    pub faces: usize,
    /// Boundary vertices TetGen added to the input (`faces` counts the refined mesh).
    pub steiner: usize,
    /// The failing stage and its error or panic message.
    pub failure: Option<(Stage, String)>,
    pub seconds: [Option<f64>; Stage::ALL.len()],
    pub patches: Option<usize>,
    pub boundaries: Option<usize>,
    pub max_degree: Option<usize>,
    pub disks: Option<usize>,
    pub disk_misses: Option<usize>,
    pub junction_splits: Option<usize>,
    pub cycle_repairs: Option<usize>,
    pub optimized_patches: Option<usize>,
    pub optimized_boundaries: Option<usize>,
    pub loops_drawn: Option<usize>,
    pub loops_planned: Option<usize>,
    pub quality: Option<f64>,
    /// Peak resident memory of the process after the last stage run, in MB (Linux only).
    pub peak_mb: Option<u64>,
}

impl Row {
    pub fn status(&self) -> String {
        match &self.failure {
            None => "ok".to_owned(),
            Some((stage, _)) => format!("failed at {}", stage.name()),
        }
    }

    /// Report columns as (name, value) pairs; empty values for missing data.
    pub fn fields(&self) -> Vec<(&'static str, String)> {
        let count = |c: Option<usize>| c.map(|c| c.to_string()).unwrap_or_default();
        let mut fields = vec![
            ("input", self.input.display().to_string()),
            ("genus", count(self.genus)),
            ("faces", self.faces.to_string()),
            ("steiner", self.steiner.to_string()),
            ("status", self.status()),
            (
                "error",
                self.failure
                    .as_ref()
                    .map(|f| f.1.clone())
                    .unwrap_or_default(),
            ),
        ];
        const SECONDS: [&str; Stage::ALL.len()] = [
            "s_deflation",
            "s_extraction",
            "s_certification",
            "s_labeling",
            "s_placement",
            "s_optimization",
            "s_drawing",
            "s_reconstruction",
        ];
        for (name, seconds) in SECONDS.into_iter().zip(self.seconds) {
            fields.push((name, seconds.map(|s| format!("{s:.3}")).unwrap_or_default()));
        }
        fields.extend([
            ("patches", count(self.patches)),
            ("boundaries", count(self.boundaries)),
            ("max_degree", count(self.max_degree)),
            ("disks", count(self.disks)),
            ("disk_misses", count(self.disk_misses)),
            ("junction_splits", count(self.junction_splits)),
            ("cycle_repairs", count(self.cycle_repairs)),
            ("optimized_patches", count(self.optimized_patches)),
            ("optimized_boundaries", count(self.optimized_boundaries)),
            ("loops_drawn", count(self.loops_drawn)),
            ("loops_planned", count(self.loops_planned)),
            (
                "quality",
                self.quality.map(|q| format!("{q:.6}")).unwrap_or_default(),
            ),
            (
                "peak_mb",
                self.peak_mb.map(|m| m.to_string()).unwrap_or_default(),
            ),
        ]);
        fields
    }

    fn record(&mut self, stage: Stage, progress: &Progress, solution: &Solution) {
        match stage {
            Stage::Deflation | Stage::CubePlacement => {}
            Stage::Extraction => {
                if let Some(skeleton) = &progress.skeleton {
                    let mut degree = HashMap::new();
                    for edge in skeleton.edges.values() {
                        for node in edge.nodes {
                            *degree.entry(node).or_insert(0) += 1;
                        }
                    }
                    self.patches = Some(skeleton.nodes.len());
                    self.boundaries = Some(skeleton.edges.len());
                    self.max_degree = Some(degree.values().copied().max().unwrap_or(0));
                }
            }
            Stage::Certification => {
                if let Some(certificate) = &progress.certificate {
                    self.disks = Some(certificate.disks.len());
                    self.disk_misses = Some(certificate.misses.len());
                }
            }
            Stage::Labeling => {
                if let Some(labeling) = &progress.labeling {
                    self.junction_splits = Some(labeling.junction_splits.len());
                    self.cycle_repairs = Some(labeling.repairs.len());
                }
            }
            Stage::Optimization => {
                if let Some(optimized) = &progress.optimized {
                    self.optimized_patches = Some(optimized.labeled.skeleton.nodes.len());
                    self.optimized_boundaries = Some(optimized.labeled.skeleton.edges.len());
                }
            }
            Stage::LoopDrawing => {
                if let Some(drawing) = &progress.drawing {
                    self.loops_drawn = Some(drawing.loops.len());
                    self.loops_planned = Some(drawing.planned);
                }
            }
            Stage::Reconstruction => self.quality = solution.get_quality(),
        }
    }
}

/// Run one mesh through every stage up to `last`, stopping at the first failure or panic. With
/// `save`, a full run is saved into that folder (see [`save`]); failing to save fails the run.
pub fn run_mesh(input: &Path, last: Stage, certify_each_step: bool, save: Option<&Path>) -> Row {
    let mut row = Row {
        input: input.to_owned(),
        ..Row::default()
    };
    let mut solution = match io::import_mesh(input) {
        Ok((mesh, steiner)) => {
            row.steiner = steiner;
            Solution::new(std::sync::Arc::new(mesh))
        }
        Err(error) => {
            row.failure = Some((Stage::Deflation, format!("import: {error}")));
            return row;
        }
    };
    let mesh = solution.mesh_ref.clone();
    row.faces = mesh.nr_faces();
    row.genus = validate::input_genus(&mesh).ok();
    let mut progress = Progress::default();
    let mut contracted = None;
    // Kept only to be saved.
    let mut extraction = save.map(|_| None);
    let skipped = |s: Stage| certify_each_step && s == Stage::Certification;
    for stage in Stage::ALL.into_iter().filter(|&s| s <= last && !skipped(s)) {
        // Names the running stage in the log, for runs killed from outside (timeout, out of memory).
        tracing::info!("Stage {} started", stage.name());
        let start = Instant::now();
        let result = catch_unwind(AssertUnwindSafe(|| {
            step(
                stage,
                &mesh,
                &mut contracted,
                extraction.as_mut(),
                &mut progress,
                &mut solution,
                certify_each_step,
            )
        }));
        row.seconds[stage as usize] = Some(start.elapsed().as_secs_f64());
        row.peak_mb = peak_memory_mb();
        row.record(stage, &progress, &solution);
        let error = match result {
            Ok(Ok(())) => continue,
            Ok(Err(error)) => error.to_string(),
            Err(panic) => format!("panic: {}", panic_message(panic.as_ref())),
        };
        row.failure = Some((stage, error));
        break;
    }
    if let Some(folder) = save
        && row.failure.is_none()
        && last == Stage::Reconstruction
        && let Err(error) = self::save(folder, input, &solution, extraction.flatten(), progress)
    {
        row.failure = Some((Stage::Reconstruction, format!("save: {error}")));
    }
    row
}

/// Saves a full run of `input` into `folder` as `<stem>.dc` (with the skeleton stages, loadable in
/// the GUI; without a separate output, as the solution is that), and its labeling as `<stem>.flag`: one label (0..5 for +X, -X, +Y, -Y, +Z, -Z) per
/// triangle of `<stem>.obj`, the refined surface of the layout.
fn save(
    folder: &Path,
    input: &Path,
    solution: &Solution,
    extraction: Option<skeletonize::Extraction>,
    progress: Progress,
) -> anyhow::Result<()> {
    let layout = solution
        .layout
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("no layout"))?;
    let stem = input
        .file_stem()
        .ok_or_else(|| anyhow::anyhow!("no file name"))?
        .to_string_lossy();
    std::fs::create_dir_all(folder)?;
    let skeleton = io::SkeletonSave {
        extraction: extraction.map(io::SkeletonExtraction::Extracted),
        labeling: progress.labeling,
        labeled: progress.labeled,
        realized: progress.realized,
        optimized: progress.optimized,
        drawing: progress.drawing,
        loops: progress.loops,
        failure: None,
        // The solution itself: nothing evolved it (a copy would double the memory and the file).
        output: None,
    };
    let path = |extension: &str| folder.join(format!("{stem}.{extension}"));
    io::formats::dc::write(solution, Some(&skeleton), Some(input), &path("dc"))?;
    <io::Flag as io::Export>::export(solution, &path("flag"))?;
    write_obj(&layout.granulated_mesh, &path("obj"))
}

/// A triangle mesh as OBJ, positions in full precision, triangles in the order `Flag` labels them.
fn write_obj(mesh: &Mesh<INPUT>, path: &Path) -> anyhow::Result<()> {
    use std::io::Write;
    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut index = HashMap::new();
    for (i, vert) in mesh.vert_ids().into_iter().enumerate() {
        let p = mesh.position(vert);
        writeln!(file, "v {} {} {}", p.x, p.y, p.z)?;
        index.insert(vert, i + 1);
    }
    for face in mesh.face_ids() {
        let corners: Vec<_> = mesh.vertices(face).map(|v| index[&v].to_string()).collect();
        writeln!(file, "f {}", corners.join(" "))?;
    }
    file.flush()?;
    Ok(())
}

fn step(
    stage: Stage,
    mesh: &Mesh<INPUT>,
    contracted: &mut Option<Mesh<deflate::CONTRACTION>>,
    extraction: Option<&mut Option<skeletonize::Extraction>>,
    progress: &mut Progress,
    solution: &mut Solution,
    certify_each_step: bool,
) -> anyhow::Result<()> {
    let mut options = skeletonize::Options::default();
    options.surgery.certify_each_step = certify_each_step;
    match stage {
        Stage::Deflation => {
            *contracted = Some(
                deflate::run(mesh, options.max_iterations, options.contraction_step)
                    .map_err(|e| anyhow::anyhow!("{e}"))?,
            );
        }
        Stage::Extraction => {
            let deflated = contracted.take().expect("deflation runs first");
            let extracted = skeletonize::extract_from_contracted(mesh, deflated, options)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            *progress = Progress::from_skeleton(extracted.skeleton.clone());
            if let Some(kept) = extraction {
                *kept = Some(extracted);
            }
            if certify_each_step {
                // Every patch is contractible by construction; nothing left to certify.
                progress.certificate = Some(Default::default());
            }
        }
        Stage::Reconstruction => {
            let drawing = progress
                .drawing
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("no drawing to reconstruct"))?;
            solution.replace_loops(drawing.loops.clone());
            // The drawing's dual comes from its loop map; reconstruction reuses it.
            if let Some(dual) = drawing.dual.clone() {
                solution.dual = Ok(dual);
            }
            solution.reconstruct_solution(false)?;
            if !solution.layout.as_ref().is_some_and(Layout::is_complete) {
                anyhow::bail!("reconstruction left an incomplete layout");
            }
        }
        stage => progress.run_through(mesh, stage)?,
    }
    Ok(())
}

/// Peak resident memory of this process (`VmHWM`), in MB; `None` where `/proc` is unavailable.
fn peak_memory_mb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024)
}

fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    panic
        .downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_owned())
}

/// Quote a CSV field when it contains a separator, quote, or line break.
pub fn csv_field(value: &str) -> String {
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

pub fn write_report(path: &Path, rows: &[Row]) -> anyhow::Result<()> {
    let mut text = String::new();
    let header: Vec<_> = Row::default().fields().into_iter().map(|f| f.0).collect();
    text.push_str(&header.join(","));
    text.push('\n');
    for row in rows {
        let values: Vec<_> = row.fields().iter().map(|f| csv_field(&f.1)).collect();
        text.push_str(&values.join(","));
        text.push('\n');
    }
    std::fs::write(path, text)?;
    Ok(())
}

/// Run all meshes of all inputs, printing one line per mesh and a summary per outcome.
pub fn run(
    inputs: &[PathBuf],
    last: Stage,
    report: Option<&Path>,
    certify_each_step: bool,
    save: Option<&Path>,
) -> anyhow::Result<()> {
    let mut paths = Vec::new();
    for input in inputs {
        paths.extend(meshes(input)?);
    }
    if paths.is_empty() {
        anyhow::bail!("no meshes found");
    }
    let mut rows = Vec::new();
    for (i, path) in paths.iter().enumerate() {
        let row = run_mesh(path, last, certify_each_step, save);
        let seconds: f64 = row.seconds.iter().flatten().sum();
        println!(
            "[{}/{}] {}: {} ({seconds:.1}s){}",
            i + 1,
            paths.len(),
            path.display(),
            row.status(),
            row.failure
                .as_ref()
                .map(|f| format!(": {}", f.1))
                .unwrap_or_default()
        );
        rows.push(row);
        // Rewrite after every mesh, so an interrupted batch keeps its rows.
        if let Some(report) = report {
            write_report(report, &rows)?;
        }
    }
    let mut outcomes = BTreeMap::<String, usize>::new();
    for row in &rows {
        *outcomes.entry(row.status()).or_default() += 1;
    }
    println!("summary ({} meshes, through {}):", rows.len(), last.name());
    for (status, count) in outcomes {
        println!("  {status:<26} {count}");
    }
    if let Some(report) = report {
        println!("Wrote `{}`", report.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests;
