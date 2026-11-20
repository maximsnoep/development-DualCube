mod cache;
pub use cache::ContractionCache;

use super::{Job, JobResult};
use crate::resources::{SkeletonResource, SkeletonState};
use crate::ui::ModelViewKind;
use bevy::prelude::*;
use dualcube::prelude::*;
use dualcube::skeleton::pipeline::{Progress, Stage};
use dualcube::skeleton::skeletonize::{self, Options};
use std::sync::Arc;

pub(super) enum Update {
    State(SkeletonState),
    Progress(Progress),
}

impl Update {
    pub(super) fn apply(self, resource: &mut SkeletonResource) {
        match self {
            Self::State(state) => {
                // A new extraction invalidates the previous initialization's downstream results.
                *resource = SkeletonResource {
                    contraction_cache: resource.contraction_cache.clone(),
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
    pub fn extract_skeleton(
        mesh: Arc<mehsh::prelude::Mesh<INPUT>>,
        options: Options,
        cache: ContractionCache,
    ) -> Self {
        Self::extract_skeleton_until(mesh, options, Stage::LoopDrawing, cache)
    }

    pub fn extract_skeleton_until(
        mesh: Arc<mehsh::prelude::Mesh<INPUT>>,
        options: Options,
        stop: Stage,
        cache: ContractionCache,
    ) -> Self {
        Self::with_updates("extracting skeleton", move |updates| {
            let publish = |update| {
                let _ = updates.send(update);
            };
            match extract_until_with_updates(&mesh, options, stop, &cache, &publish) {
                Ok(state) => {
                    let mut progress = Progress::default();
                    let mut failure = None;
                    if let SkeletonState::Extracted(extraction) = &state {
                        progress = Progress::from_skeleton(extraction.skeleton.clone());
                        failure = (|| {
                            while progress.next().is_some_and(|s| s <= stop) {
                                progress.advance(&mesh)?;
                                publish(Update::Progress(progress.clone()));
                            }
                            Ok::<_, dualcube::skeleton::pipeline::Failure>(())
                        })()
                        .err();
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
                        view,
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
}

#[cfg(test)]
fn extract_until(
    mesh: &Arc<mehsh::prelude::Mesh<INPUT>>,
    options: Options,
    stop: Stage,
    cache: &ContractionCache,
) -> dualcube::skeleton::SkeletonResult<SkeletonState> {
    extract_until_with_updates(mesh, options, stop, cache, &|_| {})
}

fn extract_until_with_updates(
    mesh: &Arc<mehsh::prelude::Mesh<INPUT>>,
    mut options: Options,
    stop: Stage,
    cache: &ContractionCache,
    publish: &impl Fn(Update),
) -> dualcube::skeleton::SkeletonResult<SkeletonState> {
    options.surgery.validate()?;
    let contracted = cache.get_or_compute(
        mesh.clone(),
        options.max_iterations,
        options.contraction_step,
    )?;
    publish(Update::State(SkeletonState::Deflated(
        (*contracted).clone(),
    )));
    if stop == Stage::Deflation {
        return Ok(SkeletonState::Deflated((*contracted).clone()));
    }
    let refine_embedding = options.refine_embedding;
    options.refine_embedding = false;
    let mut extraction =
        skeletonize::extract_from_contracted(mesh, (*contracted).clone(), options)?;
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
