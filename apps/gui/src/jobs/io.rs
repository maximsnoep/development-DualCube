//! Jobs that read and write solutions from/to disk.

use crate::colors;
use crate::jobs::{Job, JobResult};
use crate::render;
use crate::resources::Configuration;
use crate::view_settings::{Report, ViewSettings};
use dualcube::prelude::*;
use io::figure::{Annotation, FigureParams, FigureStyle};
use std::path::{Path, PathBuf};

impl Job {
    /// Import a mesh or a solution; stops the running jobs.
    pub fn import(path: PathBuf, configuration: Configuration) -> Self {
        Self::new("importing", move || {
            info!("Importing solution from {}", path.display());
            match io::import_save(&path) {
                Ok(save) => Some(JobResult::Imported {
                    store: render::refresh(&save.solution, &configuration),
                    solution: save.solution,
                    skeleton: save.skeleton,
                    source: save
                        .source
                        .unwrap_or_else(|| std::fs::canonicalize(&path).unwrap_or(path.clone())),
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

    /// The figures of the solution (see `io::figure`) in a new folder `<name>_<timestamp>` of a directory, as
    /// `<name>_<figure>.<format>`, seen as `view`; with smoothed paths if it says so. Next to them `<name>.dc` (the
    /// solution and the skeleton stages) and `<name>_view.txt` (the model, the camera, and the settings, to make the
    /// figures again).
    pub fn export_figures(
        solution: Solution,
        configuration: Configuration,
        dir: PathBuf,
        name: String,
        view: ViewSettings,
        skeleton: Option<io::SkeletonSave>,
        source: Option<PathBuf>,
    ) -> Self {
        Self::new("exporting figures", move || {
            let name = if name.is_empty() {
                "model".to_owned()
            } else {
                name.clone()
            };
            let date = io::figure::today();
            let timestamp = io::figure::timestamp();
            let dir = dir.join(format!("{name}_{timestamp}"));
            if let Err(err) = std::fs::create_dir_all(&dir) {
                warn!("Failed to create {}: {err:#}", dir.display());
                return None;
            }
            info!("Exporting figures to {}", dir.display());
            let figures = figures(&solution, &dir, &name, &view, &date);
            let save = dir.join(format!("{name}.dc"));
            let save = match io::formats::dc::write(
                &solution,
                skeleton.as_ref(),
                source.as_deref(),
                &save,
            ) {
                Ok(()) => {
                    info!("Saved {}", save.display());
                    Some(save)
                }
                Err(err) => {
                    warn!("Failed to save {}: {err:#}", save.display());
                    None
                }
            };
            let text = view.report(&Report {
                name: &name,
                source: source.as_deref(),
                save: save.as_deref(),
                figures: &figures,
                vertices: solution.mesh_ref.nr_verts(),
                faces: solution.mesh_ref.nr_faces(),
                date: &timestamp,
                configuration: &configuration,
            });
            let path = dir.join(format!("{name}_view.txt"));
            match std::fs::write(&path, text) {
                Ok(()) => info!("Saved {}", path.display()),
                Err(err) => warn!("Failed to save {}: {err:#}", path.display()),
            }
            None
        })
    }

    /// A Dc save of the solution with the skeleton stages and the source of the model.
    pub fn save(
        solution: Solution,
        skeleton: Option<io::SkeletonSave>,
        source: Option<PathBuf>,
        path: PathBuf,
    ) -> Self {
        Self::new("saving", move || {
            if solution.mesh_ref.vert_ids().is_empty() {
                warn!("Nothing to save: the mesh is empty");
                return None;
            }
            if let Err(err) =
                io::formats::dc::write(&solution, skeleton.as_ref(), source.as_deref(), &path)
            {
                warn!("Failed to save {}: {err:#}", path.display());
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

// The figures as seen in `view`; returns the paths saved.
fn figures(
    solution: &Solution,
    dir: &Path,
    name: &str,
    view: &ViewSettings,
    date: &str,
) -> Vec<PathBuf> {
    let smoothed = if view.smooth_paths {
        render::smoothed(solution)
    } else {
        None
    };
    let solution = smoothed.as_ref().unwrap_or(solution);
    let (direction, up) = view.figure_camera();
    let params = FigureParams {
        view: direction,
        up,
        loop_width: view.loop_width,
        // As on the model (see `render::objects::input_mesh::path_bands`).
        path_width: 0.6 * view.loop_width,
        annotation: view.label.then(|| Annotation {
            input: name.to_owned(),
            score: solution.get_quality(),
            date: date.to_owned(),
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
    match io::figure::export_figures(
        solution,
        dir,
        name,
        view.format.extension(),
        &params,
        &style,
    ) {
        Ok(saved) => {
            for path in &saved {
                info!("Saved {}", path.display());
            }
            saved
        }
        Err(err) => {
            warn!("Failed to export the figures: {err:#}");
            vec![]
        }
    }
}

#[cfg(test)]
mod tests;
