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
        Command::Evolve {
            input,
            output,
            iterations,
            pool1,
            pool2,
            patience,
            quality,
            reconstruct,
            unit,
        } => {
            let mut solution = io::import_solution(&input)?;
            solution.quality = quality;
            ensure_layout(&mut solution)?;

            let defaults = EvolutionParams::default();
            let params = EvolutionParams {
                max_generations: iterations.unwrap_or(defaults.max_generations),
                population: pool1.unwrap_or(defaults.population),
                offspring: pool2.unwrap_or(defaults.offspring),
                patience: patience.unwrap_or(defaults.patience),
                ..defaults
            };

            let Ok(mut evolved) = solution.evolve(&params) else {
                anyhow::bail!("evolution produced no valid solution");
            };

            if reconstruct {
                evolved.reconstruct_solution(unit)?;
            }

            let output = output.unwrap_or_else(|| default_output(&input, "dc"));
            export_solution(&evolved, &output)?;
            println!("Wrote `{}`", output.display());
            print_solution_summary("evolve", &input, &output, &evolved);
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
        ("coherence", format_option(report.coherence)),
        ("rectangularity", format_option(report.rectangularity)),
        ("corner_regularity", format_option(report.corner_regularity)),
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

  evolve --input <path> [--output <path>] [--iterations <n>] [--pool1 <n>] [--pool2 <n>] [--patience <n>]
         [QUALITY] [--reconstruct] [--unit|--no-unit]
      Evolve an existing solution. --iterations: (maximum) number of generations, --pool1: population size,
      --pool2: offspring per generation, --patience: stop after this many generations without improvement.

  score --input <path> [QUALITY] [--csv <path>]
      Print all quality terms of a solution and its score.
      --csv appends a row (with a header for a new file), to compare many solutions.
      Note: .dc files do not store the layout, so loaded solutions are embedded again before scoring.

  reconstruct --input <path> [--output <path>] [--unit|--no-unit]
      Reconstruct dual/layout/polyclube/quad state from the current loops.

  export --input <path> --format <dc|obj|flag|apg|loops|nlr|hex> [--output <path>]
      Export a solution in the requested format.

QUALITY (the criterion used to compare solutions):
  score = -(distortion + 0.1 boundary + 0.05 curvature + 0.5 features) - beta * loops
  --beta <f>        penalty per loop (default 0.005)

Examples:
  cli import bunny.obj
  cli initialize --input bunny.obj --output bunny.dc --reconstruct
  cli evolve --input bunny.dc --iterations 10 --pool1 10 --pool2 30 --output evolved.dc
  cli evolve --input bunny.dc --beta 0.001 --output evolved.dc
  cli score --input evolved.dc --csv scores.csv
  cli reconstruct --input loops.loops --output result.dc --unit
  cli export --input result.dc --format obj --output result.obj
"
    );
}
