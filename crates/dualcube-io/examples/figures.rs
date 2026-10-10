//! Exports the figures of a model (see `io::figure`):
//! `cargo run --release -p io --example figures -- <mesh.obj|stl|dc> <output dir> [svg|pdf|png] [--label] [--evolve N] [--raw]`
//! (a mesh is initialized first, and evolved for N generations; the paths are smoothed unless `--raw`).
use dualcube::prelude::*;
use io::figure::{Annotation, FigureParams, FigureStyle, export_figures, today};
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = PathBuf::from(&args[1]);
    let dir = PathBuf::from(&args[2]);
    let extension = args.get(3).map_or("svg", String::as_str);
    let label = args.iter().any(|a| a == "--label");
    let generations = args
        .iter()
        .position(|a| a == "--evolve")
        .and_then(|i| args.get(i + 1)?.parse::<usize>().ok());

    let mut solution = io::import_solution(&path)?;
    if solution.layout.is_none() {
        if solution.loops.is_empty() {
            solution.initialize();
        }
        solution.reconstruct_solution(false)?;
    }
    if let Some(generations) = generations {
        solution = solution.evolve(&EvolutionParams {
            max_generations: generations,
            ..EvolutionParams::default()
        })?;
        solution.reconstruct_solution(false)?;
    }
    // The paths smoothed, as the GUI shows them (unless `--raw`).
    if !args.iter().any(|a| a == "--raw") && solution.smooth_layout().is_err() {
        eprintln!("Smoothing the paths failed; they are shown as they are");
    }
    let stem = path
        .file_stem()
        .map_or("model".into(), |s| s.to_string_lossy());
    let params = FigureParams {
        annotation: label.then(|| Annotation {
            input: stem.to_string(),
            score: solution.get_quality(),
            date: today(),
        }),
        ..FigureParams::default()
    };
    std::fs::create_dir_all(&dir)?;
    for saved in export_figures(
        &solution,
        &dir,
        &stem,
        extension,
        &params,
        &FigureStyle::default(),
    )? {
        println!("{}", saved.display());
    }
    Ok(())
}
