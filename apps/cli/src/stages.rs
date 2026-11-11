//! `stages`: run the skeleton pipeline on many meshes up to one stage, one report row per mesh.

use dualcube::prelude::*;
use dualcube::skeleton::{pipeline::Progress, skeletonize, validate};
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Pipeline stages in order. A run includes every stage up to and including its last one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage {
    Extraction,
    Certification,
    Labeling,
    Placement,
    Optimization,
    Drawing,
    Reconstruction,
}

impl Stage {
    pub const ALL: [Self; 7] = [
        Self::Extraction,
        Self::Certification,
        Self::Labeling,
        Self::Placement,
        Self::Optimization,
        Self::Drawing,
        Self::Reconstruction,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Extraction => "extraction",
            Self::Certification => "certification",
            Self::Labeling => "labeling",
            Self::Placement => "placement",
            Self::Optimization => "optimization",
            Self::Drawing => "drawing",
            Self::Reconstruction => "reconstruction",
        }
    }

    pub fn parse(name: &str) -> anyhow::Result<Self> {
        Self::ALL
            .into_iter()
            .find(|s| s.name() == name)
            .ok_or_else(|| {
                let names: Vec<_> = Self::ALL.iter().map(|s| s.name()).collect();
                anyhow::anyhow!("unknown stage `{name}` (one of {})", names.join(", "))
            })
    }
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
    /// The failing stage and its error or panic message.
    pub failure: Option<(Stage, String)>,
    pub seconds: [Option<f64>; 7],
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
    pub searches: Option<usize>,
    pub backtracks: Option<usize>,
    pub quality: Option<f64>,
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
            ("status", self.status()),
            (
                "error",
                self.failure
                    .as_ref()
                    .map(|f| f.1.clone())
                    .unwrap_or_default(),
            ),
        ];
        const SECONDS: [&str; 7] = [
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
            ("searches", count(self.searches)),
            ("backtracks", count(self.backtracks)),
            (
                "quality",
                self.quality.map(|q| format!("{q:.6}")).unwrap_or_default(),
            ),
        ]);
        fields
    }

    fn record(&mut self, stage: Stage, progress: &Progress, solution: &Solution) {
        match stage {
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
            Stage::Placement => {}
            Stage::Optimization => {
                if let Some(optimized) = &progress.optimized {
                    self.optimized_patches = Some(optimized.labeled.skeleton.nodes.len());
                    self.optimized_boundaries = Some(optimized.labeled.skeleton.edges.len());
                }
            }
            Stage::Drawing => {
                if let Some(drawing) = &progress.drawing {
                    self.loops_drawn = Some(drawing.loops.len());
                    self.loops_planned = Some(drawing.planned);
                    self.searches = Some(drawing.searches);
                    self.backtracks = Some(drawing.backtracks);
                }
            }
            Stage::Reconstruction => self.quality = solution.get_quality(),
        }
    }
}

/// Run one mesh through every stage up to `last`, stopping at the first failure or panic.
pub fn run_mesh(input: &Path, last: Stage) -> Row {
    let mut row = Row {
        input: input.to_owned(),
        ..Row::default()
    };
    let mut solution = match io::import_solution(input) {
        Ok(solution) => solution,
        Err(error) => {
            row.failure = Some((Stage::Extraction, format!("import: {error}")));
            return row;
        }
    };
    let mesh = solution.mesh_ref.clone();
    row.faces = mesh.nr_faces();
    row.genus = validate::input_genus(&mesh).ok();
    let mut progress = Progress::default();
    for stage in Stage::ALL.into_iter().filter(|&s| s <= last) {
        let start = Instant::now();
        let result = catch_unwind(AssertUnwindSafe(|| {
            step(stage, &mesh, &mut progress, &mut solution)
        }));
        row.seconds[stage as usize] = Some(start.elapsed().as_secs_f64());
        row.record(stage, &progress, &solution);
        let error = match result {
            Ok(Ok(())) => continue,
            Ok(Err(error)) => error.to_string(),
            Err(panic) => format!("panic: {}", panic_message(panic.as_ref())),
        };
        row.failure = Some((stage, error));
        break;
    }
    row
}

fn step(
    stage: Stage,
    mesh: &Mesh<INPUT>,
    progress: &mut Progress,
    solution: &mut Solution,
) -> anyhow::Result<()> {
    match stage {
        Stage::Extraction => {
            let skeleton = skeletonize::run(mesh).map_err(|e| anyhow::anyhow!("{e}"))?;
            *progress = Progress::from_skeleton(skeleton);
        }
        Stage::Certification => progress.certify(mesh)?,
        Stage::Labeling => progress.label(mesh)?,
        Stage::Placement => progress.place(mesh)?,
        Stage::Optimization => progress.optimize(mesh)?,
        Stage::Drawing => progress.finish(mesh)?,
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
    }
    Ok(())
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
pub fn run(inputs: &[PathBuf], last: Stage, report: Option<&Path>) -> anyhow::Result<()> {
    let mut paths = Vec::new();
    for input in inputs {
        paths.extend(meshes(input)?);
    }
    if paths.is_empty() {
        anyhow::bail!("no meshes found");
    }
    let mut rows = Vec::new();
    for (i, path) in paths.iter().enumerate() {
        let row = run_mesh(path, last);
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
