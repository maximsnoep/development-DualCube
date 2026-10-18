use crate::parser::{Args, Command};
use dualcube::prelude::*;
use io::Export;
use std::path::{Path, PathBuf};

pub fn run(args: anyhow::Result<Args>) -> anyhow::Result<()> {
    let args = args?;

    match args.command {
        Command::Help => {
            print_help();
        }
        Command::Benchmark {
            inputs,
            output,
            name,
            runs,
            seconds,
            quality,
            resume,
        } => {
            if let Some(dir) = resume {
                let progress = io::benchmark::BenchmarkProgress::default();
                io::benchmark::resume_benchmark(&dir, &io::figure::FigureStyle::default(), &progress)?;
                println!("Results in `{}`", dir.display());
                return Ok(());
            }
            // Folders stand for the models in them (see `models_in`).
            let models = inputs
                .iter()
                .flat_map(|input| {
                    if input.is_dir() {
                        io::benchmark::models_in(input)
                    } else {
                        vec![input.clone()]
                    }
                })
                .collect::<Vec<_>>();
            anyhow::ensure!(!models.is_empty(), "no models found");
            println!(
                "Benchmark `{name}`: {} models, {runs} runs each, {seconds} s per run",
                models.len()
            );
            let config = io::benchmark::BenchmarkConfig {
                name,
                models,
                runs,
                seconds,
                params: CoupledParams::default(),
                quality,
            };
            let progress = io::benchmark::BenchmarkProgress::default();
            let dir = io::benchmark::run_benchmark(
                config,
                &output,
                &io::figure::FigureStyle::default(),
                &progress,
            )?;
            println!("Results in `{}`", dir.display());
        }
        Command::Import { input } => {
            let solution = io::import_solution(&input)?;
            println!(
                "Imported `{}`: verts={}, faces={}, loops={}",
                input.display(),
                solution.mesh_ref.nr_verts(),
                solution.mesh_ref.nr_faces(),
                solution.loops.len()
            );
        }
        Command::Initialize {
            input,
            output,
            quality,
            reconstruct,
            unit,
        } => {
            let mut solution = io::import_solution(&input)?;
            solution.quality = quality;
            solution.initialize();

            if reconstruct {
                solution.reconstruct_solution(unit)?;
            }

            let output = output.unwrap_or_else(|| default_output(&input, "dc"));
            export_solution(&solution, &output)?;
            println!("Wrote `{}`", output.display());
            print_solution_summary("initialize", &input, &output, &solution);
        }
        Command::Anneal {
            input,
            output,
            steps,
            offspring,
            quality,
            reconstruct,
            unit,
        } => {
            let mut solution = io::import_solution(&input)?;
            solution.quality = quality;
            ensure_layout(&mut solution)?;

            let defaults = AnnealParams::default();
            let params = AnnealParams {
                steps: steps.unwrap_or(defaults.steps),
                offspring: offspring.unwrap_or(defaults.offspring),
                ..defaults
            };

            let Ok(mut annealed) = solution.anneal(&params, &OptimizationMonitor::default()) else {
                anyhow::bail!("annealing produced no valid solution");
            };

            if reconstruct {
                annealed.reconstruct_solution(unit)?;
            }

            let output = output.unwrap_or_else(|| default_output(&input, "dc"));
            export_solution(&annealed, &output)?;
            println!("Wrote `{}`", output.display());
            print_solution_summary("anneal", &input, &output, &annealed);
        }
        Command::Reconstruct {
            input,
            output,
            unit,
        } => {
            let mut solution = io::import_solution(&input)?;
            solution.reconstruct_solution(unit)?;

            let output = output.unwrap_or_else(|| default_output(&input, "dc"));
            export_solution(&solution, &output)?;
            println!("Wrote `{}`", output.display());
            print_solution_summary("reconstruct", &input, &output, &solution);
        }
        Command::Score {
            input,
            quality,
            csv,
        } => {
            let mut solution = io::import_solution(&input)?;
            solution.quality = quality;
            ensure_layout(&mut solution)?;
            let report = solution
                .quality_report()
                .ok_or_else(|| anyhow::anyhow!("solution has no complete layout"))?;
            let score = solution.get_quality();
            print_report(&report, score);
            if let Some(csv) = csv {
                append_csv(&csv, &input, &report, score)?;
                println!("Appended to `{}`", csv.display());
            }
        }
        Command::Export {
            input,
            output,
            format,
        } => {
            let solution = io::import_solution(&input)?;
            let output = output.unwrap_or_else(|| PathBuf::from("output"));
            export_solution_as(&solution, &output, &format)?;
            println!("Wrote `{}`", output.display());
            print_solution_summary("export", &input, &output, &solution);
        }
    }

    Ok(())
}

// Reconstruct the solution if it has no complete layout (e.g., loops only, or an input mesh).
fn ensure_layout(solution: &mut Solution) -> anyhow::Result<()> {
    if !solution.layout.as_ref().is_some_and(Layout::is_complete) {
        solution.reconstruct_solution(false)?;
    }
    if solution.layout.is_none() {
        anyhow::bail!("solution has no valid loop structure (initialize it first)");
    }
    Ok(())
}

fn format_option(value: Option<f64>) -> String {
    value.map_or_else(|| "none".to_owned(), |v| format!("{v:.6}"))
}

// The terms of a report, as (name, value) pairs.
fn report_terms(report: &QualityReport) -> Vec<(&'static str, String)> {
    let count = |c: Option<usize>| c.map_or_else(|| "none".to_owned(), |c| c.to_string());
    vec![
        ("alignment", format_option(report.alignment)),
        ("fidelity", format_option(report.fidelity)),
        ("path_backtrack", format_option(report.path_backtrack)),
        ("path_stretch", format_option(report.path_stretch)),
        ("non_degeneracy", format_option(report.non_degeneracy)),
        ("correspondence", format_option(report.correspondence)),
        ("loops", report.loops.to_string()),
        ("corners", count(report.corners)),
        ("irregular_corners", count(report.irregular_corners)),
    ]
}

fn print_report(report: &QualityReport, score: Option<f64>) {
    println!("quality terms:");
    for (name, value) in report_terms(report) {
        println!("  {name:<20} {value}");
    }
    println!("score: {}", format_option(score));
}

fn append_csv(
    path: &Path,
    input: &Path,
    report: &QualityReport,
    score: Option<f64>,
) -> anyhow::Result<()> {
    use std::io::Write;
    let terms = report_terms(report);
    let new_file = !path.exists();
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    if new_file {
        let header = std::iter::once("input".to_owned())
            .chain(terms.iter().map(|(name, _)| (*name).to_owned()))
            .chain(std::iter::once("score".to_owned()))
            .collect::<Vec<_>>()
            .join(",");
        writeln!(file, "{header}")?;
    }
    let row = std::iter::once(input.display().to_string())
        .chain(terms.into_iter().map(|(_, value)| value))
        .chain(std::iter::once(format_option(score)))
        .collect::<Vec<_>>()
        .join(",");
    writeln!(file, "{row}")?;
    Ok(())
}

fn default_output(input: &Path, extension: &str) -> PathBuf {
    input.with_extension(extension)
}

fn export_solution(solution: &Solution, output: &Path) -> anyhow::Result<()> {
    match output.extension().and_then(|x| x.to_str()) {
        Some(ext) => export_solution_as(solution, output, &ext.to_ascii_lowercase()),
        // No extension given: default to the native Dc format.
        None => export_solution_as(solution, &output.with_extension("dc"), "dc"),
    }
}

fn export_solution_as(solution: &Solution, output: &Path, format: &str) -> anyhow::Result<()> {
    match format {
        "dc" => io::Dc::export(solution, output),
        "obj" => io::OBJ::export(solution, output),
        "flag" => io::Flag::export(solution, output),
        "apg" => io::APG::export(solution, output),
        "loops" => io::Loops::export(solution, output),
        "nlr" => io::NLR::export(solution, output),
        "hex" | "hex.mesh" => io::HEX::export(solution, output),
        other => anyhow::bail!("unsupported export format: {other}"),
    }
}

fn print_solution_summary(command: &str, input: &Path, output: &Path, solution: &Solution) {
    println!(
        "summary command={} input={} output={} quality={} loops={} verts={} faces={} dual={} layout={} polycube={} quad={}",
        command,
        input.display(),
        output.display(),
        solution
            .get_quality()
            .map_or_else(|| "none".to_owned(), |q| format!("{q:.6}")),
        solution.loops.len(),
        solution.mesh_ref.nr_verts(),
        solution.mesh_ref.nr_faces(),
        solution.dual.is_ok(),
        solution
            .layout
            .as_ref()
            .map(|l| if l.is_complete() {
                "complete"
            } else {
                "incomplete"
            })
            .unwrap_or("none"),
        solution.polycube.is_some(),
        solution.quad.is_some()
    );
}

fn print_help() {
    println!(
        "\
DualCube CLI

Commands:
  help
      Show this help text.

  import <input>
      Import a mesh/solution and print a short summary.

  initialize --input <path> [--output <path>] [QUALITY] [--reconstruct] [--unit|--no-unit]
      Initialize the loop structure of an input mesh or solution: three loops that cross pairwise twice (a cube).
      Writes the result to --output, or defaults to changing the extension to .dc.

  anneal --input <path> [--output <path>] [--steps <n>] [--offspring <n>] [QUALITY] [--reconstruct] [--unit|--no-unit]
      Optimize the loops of an existing solution with simulated annealing. --steps: number of steps,
      --offspring: children per step.

  benchmark --input <model or folder> [--input ...] [--output <dir>] [--name <name>] [--runs <n>] [--seconds <s>] [QUALITY]
      Initialize and optimize every model (a folder: all obj/stl in it) for --seconds (default 60), --runs times
      (default 1). The results (tables, solutions, figures) go to a new folder in --output (default benchmarks);
      open it in the GUI's benchmark window. --resume <session folder> continues a stopped session (its own options).

  score --input <path> [QUALITY] [--csv <path>]
      Print all quality terms of a solution and its score.
      --csv appends a row (with a header for a new file), to compare many solutions.
      Note: .dc files do not store the layout, so loaded solutions are embedded again before scoring.

  reconstruct --input <path> [--output <path>] [--unit|--no-unit]
      Reconstruct dual/layout/polyclube/quad state from the current loops.

  export --input <path> --format <dc|obj|flag|apg|loops|nlr|hex> [--output <path>]
      Export a solution in the requested format.

QUALITY (the criterion used to compare solutions):
  score = 1 / (1 + fidelity + 3 path regularity + non-degeneracy + correspondence + beta * loops)
  --beta <f>        penalty per loop (default 0.05)

Examples:
  cli import bunny.obj
  cli initialize --input bunny.obj --output bunny.dc --reconstruct
  cli anneal --input bunny.dc --steps 60 --output annealed.dc
  cli anneal --input bunny.dc --beta 0.001 --output annealed.dc
  cli benchmark --input meshes/a.bouba_zero --seconds 1200 --name bouba
  cli score --input annealed.dc --csv scores.csv
  cli reconstruct --input loops.loops --output result.dc --unit
  cli export --input result.dc --format obj --output result.obj
"
    );
}
