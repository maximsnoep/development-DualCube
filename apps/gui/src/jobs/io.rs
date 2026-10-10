//! Jobs that read and write solutions from/to disk.

use crate::colors;
use crate::jobs::{Job, JobResult};
use crate::render;
use crate::resources::Configuration;
use dualcube::prelude::*;
use io::figure::{Annotation, FigureParams, FigureStyle};
use std::path::PathBuf;

impl Job {
    /// Import a mesh or a solution; stops the running jobs.
    pub fn import(path: PathBuf, configuration: Configuration) -> Self {
        Self::new("importing", move || {
            info!("Importing solution from {}", path.display());
            match io::import_solution(&path) {
                Ok(solution) => Some(JobResult::Imported {
                    store: render::refresh(&solution, &configuration),
                    solution,
                    name: path.file_stem().map(|s| s.to_string_lossy().into_owned()),
                }),
                Err(err) => {
                    warn!("Failed to import {}: {err:#}", path.display());
                    None
                }
            }
        })
        .preempting()
    }

    /// The figures of the solution (see `io::figure`) in a directory, as `<name>_<figure>.<format>`, seen along the
    /// view of the camera (`view`: the direction it looks in, `up`: up on the screen). With smoothed paths if they are
    /// shown smoothed.
    pub fn export_figures(
        solution: Solution,
        configuration: Configuration,
        dir: PathBuf,
        name: String,
        view: Vector3D,
        up: Vector3D,
    ) -> Self {
        Self::new("exporting figures", move || {
            let smoothed = if configuration.smooth_paths {
                render::smoothed(&solution)
            } else {
                None
            };
            let solution = smoothed.as_ref().unwrap_or(&solution);
            let name = if name.is_empty() {
                "model".to_owned()
            } else {
                name.clone()
            };
            let params = FigureParams {
                view,
                up,
                loop_width: configuration.loop_width,
                // As on the model (see `render::objects::input_mesh::path_bands`).
                path_width: 0.6 * configuration.loop_width,
                annotation: configuration.figure_label.then(|| Annotation {
                    input: name.clone(),
                    score: solution.get_quality(),
                    date: io::figure::today(),
                }),
                ..FigureParams::default()
            };
            let axes = DIRECTIONS;
            let style = FigureStyle {
                primal: axes.map(|d| colors::from_direction(d, Some(Perspective::Primal), None)),
                light_mix: render::LIGHT_MIX,
                dual: axes.map(|d| colors::from_direction(d, Some(Perspective::Dual), None)),
                ..FigureStyle::default()
            };
            let extension = configuration.figure_format.extension();
            match io::figure::export_figures(solution, &dir, &name, extension, &params, &style) {
                Ok(saved) => {
                    for path in saved {
                        info!("Saved {}", path.display());
                    }
                }
                Err(err) => warn!("Failed to export the figures: {err:#}"),
            }
            None
        })
    }

    pub fn export(solution: Solution, path: PathBuf) -> Self {
        Self::new("exporting", move || {
            if solution.mesh_ref.vert_ids().is_empty() {
                warn!("Nothing to export: the mesh is empty");
                return None;
            }
            info!("Exporting solution to {}", path.display());
            if let Err(err) = io::export_solution(&solution, &path) {
                warn!("Failed to export {}: {err:#}", path.display());
            }
            None
        })
    }
}
