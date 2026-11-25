use super::{Job, JobResult};
use crate::resources::{ContractionStages, SkeletonResource, SkeletonState};
use crate::ui::ModelViewKind;
use bevy::prelude::*;
use dualcube::prelude::*;
use dualcube::skeleton::certify::Certificate;
use dualcube::skeleton::pipeline::{Failure, Progress, Stage};
use dualcube::skeleton::skeletonize::{self, Options};
use dualcube::skeleton::{Skeleton, SkeletonEdgeID, edit};
use std::sync::Arc;

pub(super) enum Update {
    /// A new contraction, with its stages: starts a new extraction.
    Deflated(SkeletonState, ContractionStages),
    State(SkeletonState),
    Progress(Progress),
}

impl Update {
    pub(super) fn apply(self, resource: &mut SkeletonResource) {
        match self {
            // A new extraction invalidates the previous initialization's downstream results.
            Self::Deflated(state, stages) => {
                *resource = SkeletonResource {
                    contraction_stages: Some(stages),
                    state: Some(state),
                    ..Default::default()
                };
            }
            Self::State(state) => {
                *resource = SkeletonResource {
                    contraction_stages: resource.contraction_stages.take(),
                    state: Some(state),
                    ..Default::default()
                };
            }
            Self::Progress(progress) => resource.progress = progress,
        }
    }
}

/// The view after stopping at `stop`, with `reached` the last stage that has its result.
fn view(stop: Stage, reached: Stage, refined: bool) -> ModelViewKind {
    match reached.min(stop) {
        Stage::Deflation => ModelViewKind::Deflated,
        Stage::Extraction | Stage::Certification if stop == Stage::Extraction && !refined => {
            ModelViewKind::Raw
        }
        Stage::Extraction | Stage::Certification => ModelViewKind::Skeleton,
        Stage::Labeling => ModelViewKind::Labeled,
        _ => ModelViewKind::Cubes,
    }
}

impl Job {
    pub fn extract_skeleton(mesh: Arc<mehsh::prelude::Mesh<INPUT>>, options: Options) -> Self {
        Self::extract_skeleton_until(mesh, options, Stage::LoopDrawing)
    }

    pub fn extract_skeleton_until(
        mesh: Arc<mehsh::prelude::Mesh<INPUT>>,
        options: Options,
        stop: Stage,
    ) -> Self {
        Self::with_updates("extracting skeleton", move |updates| {
            let publish = |update| {
                let _ = updates.send(update);
            };
            match extract_until_with_updates(&mesh, options, stop, &publish) {
                Ok(state) => {
                    let mut progress = Progress::default();
                    let mut failure = None;
                    if let SkeletonState::Extracted(extraction) = &state {
                        progress = Progress::from_skeleton(extraction.skeleton.clone());
                        failure = advance_until(&mesh, &mut progress, stop, &publish);
                    }
                    if let Some(error) = &failure {
                        warn!("Initialization stopped at {error}");
                    }
                    let reached = Stage::ALL
                        .into_iter()
                        .take_while(|&s| progress.done(s))
                        .last()
                        .unwrap_or(Stage::Deflation);
                    let view = view(stop, reached, options.refine_embedding);
                    Some(JobResult::SkeletonReady {
                        state,
                        view: Some(view),
                        progress,
                        failure,
                    })
                }
                Err(error) => {
                    warn!("Failed to extract skeleton: {error}");
                    None
                }
            }
        })
        .preempting()
    }

    /// Dissolve boundaries of the shown skeleton (see `SkeletonResource::current_skeleton`, `edit::dissolve_all`) and
    /// run the stages after extraction again through `stop`; the view stays. The certificate carries over: patches stay
    /// contractible in the volume under the merge (see `edit::dissolve`), as under the labeling's splits.
    pub fn dissolve_skeleton_loops(
        mesh: Arc<mehsh::prelude::Mesh<INPUT>>,
        extraction: skeletonize::Extraction,
        skeleton: Skeleton,
        certificate: Option<Certificate>,
        edges: Vec<SkeletonEdgeID>,
        stop: Stage,
    ) -> Self {
        Self::with_updates("dissolving skeleton loops", move |updates| {
            let publish = |update| {
                let _ = updates.send(update);
            };
            let mut skeleton = skeleton.clone();
            match edit::dissolve_all(&mesh, &mut skeleton, &edges) {
                Ok(0) => {
                    warn!("No skeleton loop dissolved");
                    return None;
                }
                Ok(dissolved) if dissolved < edges.len() => warn!(
                    "Dissolved {dissolved} of {} skeleton loops; the others were no longer dissolvable",
                    edges.len()
                ),
                Ok(dissolved) => info!("Dissolved {dissolved} skeleton loops"),
                Err(error) => {
                    warn!("Skeleton loops not dissolved: {error}");
                    return None;
                }
            }
            info!(
                "Skeleton: {} nodes, {} edges",
                skeleton.nodes.len(),
                skeleton.edges.len()
            );
            let state = SkeletonState::Extracted(skeletonize::Extraction {
                skeleton: skeleton.clone(),
                ..extraction.clone()
            });
            publish(Update::State(state.clone()));
            let mut progress = Progress::from_skeleton(skeleton);
            progress.certificate = certificate.clone();
            let failure = advance_until(&mesh, &mut progress, stop, &publish);
            if let Some(error) = &failure {
                warn!("Initialization stopped at {error}");
            }
            Some(JobResult::SkeletonReady {
                state,
                view: None,
                progress,
                failure,
            })
        })
    }
}

/// Run the stages after extraction through `stop`, publishing each; the failure that stopped them.
fn advance_until(
    mesh: &mehsh::prelude::Mesh<INPUT>,
    progress: &mut Progress,
    stop: Stage,
    publish: &impl Fn(Update),
) -> Option<Failure> {
    while progress.next().is_some_and(|s| s <= stop) {
        if let Err(failure) = progress.advance(mesh) {
            return Some(failure);
        }
        publish(Update::Progress(progress.clone()));
    }
    None
}

#[cfg(test)]
fn extract_until(
    mesh: &Arc<mehsh::prelude::Mesh<INPUT>>,
    options: Options,
    stop: Stage,
) -> dualcube::skeleton::SkeletonResult<SkeletonState> {
    extract_until_with_updates(mesh, options, stop, &|_| {})
}

fn extract_until_with_updates(
    mesh: &Arc<mehsh::prelude::Mesh<INPUT>>,
    mut options: Options,
    stop: Stage,
    publish: &impl Fn(Update),
) -> dualcube::skeleton::SkeletonResult<SkeletonState> {
    options.surgery.validate()?;
    let (contracted, stages) = skeletonize::deflate::run_with_stages(
        mesh,
        options.max_iterations,
        options.contraction_step,
    )?;
    publish(Update::Deflated(
        SkeletonState::Deflated(contracted.clone()),
        Arc::new(stages),
    ));
    if stop == Stage::Deflation {
        return Ok(SkeletonState::Deflated(contracted));
    }
    let refine_embedding = options.refine_embedding;
    options.refine_embedding = false;
    let mut extraction = skeletonize::extract_from_contracted(mesh, contracted, options)?;
    publish(Update::State(SkeletonState::Extracted(extraction.clone())));
    if refine_embedding {
        skeletonize::refinement::run(mesh, &mut extraction.skeleton)?;
        extraction.skeleton.validate(mesh)?;
        publish(Update::State(SkeletonState::Extracted(extraction.clone())));
    }
    info!(
        "Skeleton: {} nodes, {} edges",
        extraction.skeleton.nodes.len(),
        extraction.skeleton.edges.len()
    );
    Ok(SkeletonState::Extracted(extraction))
}

#[cfg(test)]
mod tests;
