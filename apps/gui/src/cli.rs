//! Startup commands for visual iteration. Computation stays in the background jobs.

use crate::jobs::{Job, SkeletonStage};
use crate::resources::Configuration;
use crate::ui::{ModelView, ModelViewKind};
use bevy::prelude::*;
use dualcube::skeleton::skeletonize::Options;
use std::ffi::OsString;
use std::path::PathBuf;

const HELP: &str = "DualCube GUI

Run from the repository root:
  cargo run -p gui -- --load <model> --auto-start skeleton
  cargo run -p gui -- --load <model> --stop-after surgery --view combined

  --load <path>              Import a mesh or saved solution
  --auto-start skeleton      Run skeleton initialization after loading
  --stop-after <stage>       Run through input, deflation, surgery, refinement, or labeling
  --view <view>              input, deflated, raw, skeleton, labeled, patches, combined
  --refinement               Enable embedding refinement (default off)
  --no-refinement            Skip embedding refinement
  --iterations <count>      Deflation iteration limit (default 50)
  --shape-weight <number>   Surgery shape weight (default 1)
  --sampling-weight <n>     Surgery sampling weight (default 0.1)
  --handle-bonus <number>   Surgery handle-cycle bonus (default 1000000)
  --quit-after <stage>       Run through a stage and exit (nonzero on failure)
  --help, -h                 Show this help

The GUI stays open unless --quit-after is given. --stop-after implies auto-start.
--quit-after accepts the same stages, or skeleton for the initialization pipeline.
Both --flag value and --flag=value are accepted.
";

#[derive(Resource, Default)]
pub struct Launch {
    path: Option<PathBuf>,
    pub stop: Option<SkeletonStage>,
    view: Option<ModelViewKind>,
    options: Options,
    quit: bool,
}

impl Launch {
    pub fn from_env_or_exit() -> Self {
        match Self::parse(std::env::args_os().skip(1)) {
            Ok(Some(launch)) => launch,
            Ok(None) => {
                println!("{HELP}");
                std::process::exit(0);
            }
            Err(error) => {
                eprintln!("{error}\n\nUse --help for the GUI startup commands.");
                std::process::exit(2);
            }
        }
    }

    fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Option<Self>, String> {
        let mut launch = Self::default();
        let mut args = args.into_iter();
        let mut auto_start = false;
        let mut refinement = None;
        let mut stop = None;
        let mut quit_after = None;
        let mut quit_skeleton = false;
        while let Some(arg) = args.next() {
            let arg = arg.to_str().ok_or("Expected a UTF-8 option name")?;
            let (flag, inline) = arg
                .split_once('=')
                .map_or((arg, None), |(k, v)| (k, Some(v)));
            if matches!(flag, "--help" | "-h") {
                return Ok(None);
            }
            if matches!(flag, "--refinement" | "--no-refinement") && inline.is_none() {
                refinement = Some(flag == "--refinement");
                continue;
            }
            if !matches!(
                flag,
                "--load"
                    | "--auto-start"
                    | "--stop-after"
                    | "--view"
                    | "--iterations"
                    | "--shape-weight"
                    | "--sampling-weight"
                    | "--handle-bonus"
                    | "--quit-after"
            ) {
                return Err(format!("Unknown option: {flag}"));
            }
            let value = inline
                .map(OsString::from)
                .or_else(|| args.next())
                .ok_or_else(|| format!("Missing value for {flag}"))?;
            if value.is_empty() || value.to_str().is_some_and(|v| v.starts_with("--")) {
                return Err(format!("Missing value for {flag}"));
            }
            if flag == "--load" {
                launch.path = Some(PathBuf::from(value));
                continue;
            }
            let value = value
                .to_str()
                .ok_or_else(|| format!("Invalid value for {flag}"))?;
            match flag {
                "--auto-start" if value == "skeleton" => auto_start = true,
                "--auto-start" => return Err("--auto-start currently supports skeleton".into()),
                "--stop-after" => stop = Some(parse_stage(value)?),
                "--quit-after" => {
                    launch.quit = true;
                    if value == "skeleton" {
                        auto_start = true;
                        quit_skeleton = true;
                    } else {
                        quit_after = Some(parse_stage(value)?);
                    }
                }
                "--view" => {
                    launch.view = Some(match value {
                        "input" => ModelViewKind::Input,
                        "deflated" => ModelViewKind::Deflated,
                        "raw" => ModelViewKind::Raw,
                        "skeleton" => ModelViewKind::Skeleton,
                        "labeled" => ModelViewKind::Labeled,
                        "patches" => ModelViewKind::Patches,
                        "combined" => ModelViewKind::Combined,
                        _ => return Err(format!("Unknown view: {value}")),
                    })
                }
                "--iterations" => {
                    launch.options.max_iterations = value
                        .parse()
                        .map_err(|_| "--iterations requires a nonnegative integer")?
                }
                "--shape-weight" => launch.options.surgery.shape_weight = weight(flag, value)?,
                "--sampling-weight" => {
                    launch.options.surgery.sampling_weight = weight(flag, value)?
                }
                "--handle-bonus" => launch.options.surgery.handle_bonus = weight(flag, value)?,
                _ => unreachable!(),
            }
        }
        let requests_refinement = stop == Some(Some(SkeletonStage::Refinement))
            || quit_after == Some(Some(SkeletonStage::Refinement));
        if requests_refinement && refinement == Some(false) {
            return Err("--no-refinement conflicts with stopping after refinement".into());
        }
        launch.options.refine_embedding = refinement.unwrap_or(requests_refinement);
        if quit_skeleton {
            quit_after = Some(Some(SkeletonStage::Complete));
        }
        if let (Some(stop), Some(quit)) = (stop, quit_after) {
            if stop != quit {
                return Err("--stop-after and --quit-after must name the same stage".into());
            }
        }
        launch.stop = stop
            .or(quit_after)
            .unwrap_or_else(|| auto_start.then_some(SkeletonStage::Complete));
        if launch.stop == Some(SkeletonStage::Surgery) {
            launch.options.refine_embedding = false;
        }
        if (auto_start
            || stop.is_some()
            || quit_after.is_some()
            || launch.quit
            || launch.view.is_some())
            && launch.path.is_none()
        {
            return Err("Startup actions require --load <path>".into());
        }
        if let Some(view) = launch.view {
            let available = match view {
                ModelViewKind::Input => true,
                ModelViewKind::Deflated => launch.stop.is_some(),
                ModelViewKind::Labeled => matches!(
                    launch.stop,
                    Some(SkeletonStage::Labeling | SkeletonStage::Complete)
                ),
                _ => matches!(
                    launch.stop,
                    Some(
                        SkeletonStage::Surgery
                            | SkeletonStage::Refinement
                            | SkeletonStage::Labeling
                            | SkeletonStage::Complete
                    )
                ),
            };
            if !available {
                return Err("The requested view is not available at the chosen stop stage".into());
            }
        }
        Ok(Some(launch))
    }

    pub fn configuration(&self) -> Configuration {
        Configuration {
            skeleton_options: self.options,
            ..default()
        }
    }

    pub fn finish(
        &self,
        success: bool,
        view: &mut ModelView,
        exit: &mut MessageWriter<'_, AppExit>,
    ) {
        if let Some(selected) = self.view {
            view.show_when_ready(selected);
        }
        if success {
            info!("GUI startup complete");
        } else {
            error!("GUI startup failed; see the preceding error");
        }
        if self.quit {
            exit.write(if success {
                AppExit::Success
            } else {
                AppExit::error()
            });
        }
    }
}

fn parse_stage(value: &str) -> Result<Option<SkeletonStage>, String> {
    match value {
        "input" => Ok(None),
        "deflation" => Ok(Some(SkeletonStage::Deflation)),
        "surgery" => Ok(Some(SkeletonStage::Surgery)),
        "refinement" => Ok(Some(SkeletonStage::Refinement)),
        "labeling" => Ok(Some(SkeletonStage::Labeling)),
        _ => Err(format!("Unknown stage: {value}")),
    }
}

fn weight(flag: &str, value: &str) -> Result<f64, String> {
    let weight: f64 = value
        .parse()
        .map_err(|_| format!("Invalid number for {flag}"))?;
    if !weight.is_finite() || weight < 0. {
        return Err(format!("{flag} requires a finite, nonnegative number"));
    }
    Ok(weight)
}

pub fn start(
    launch: Res<'_, Launch>,
    configuration: Res<'_, Configuration>,
    mut jobs: MessageWriter<'_, Job>,
) {
    if let Some(path) = &launch.path {
        jobs.write(Job::import(path.clone(), configuration.clone()).for_startup());
    }
}

#[cfg(test)]
mod tests;
