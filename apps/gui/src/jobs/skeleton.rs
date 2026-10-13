mod cache;
pub use cache::ContractionCache;

use super::{Job, JobResult};
use crate::resources::{SkeletonResource, SkeletonState};
use crate::ui::ModelViewKind;
use bevy::prelude::*;
use dualcube::prelude::*;
use dualcube::skeleton::pipeline::Progress;
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
            Self::Progress(progress) => {
                resource.labeling = progress.labeling;
                resource.labeled = progress.labeled;
                resource.realized = progress.realized;
                resource.optimized = progress.optimized;
                resource.drawing = progress.drawing;
                resource.loops = progress.loops;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkeletonStage {
    Deflation,
    Surgery,
    Refinement,
    Labeling,
    Placement,
    Complete,
}

impl SkeletonStage {
    pub fn view(self) -> ModelViewKind {
        match self {
            Self::Deflation => ModelViewKind::Deflated,
            Self::Surgery => ModelViewKind::Raw,
            Self::Refinement => ModelViewKind::Skeleton,
            Self::Labeling => ModelViewKind::Labeled,
            Self::Placement | Self::Complete => ModelViewKind::Cubes,
        }
    }
}

impl Job {
    pub fn extract_skeleton(
        mesh: Arc<mehsh::prelude::Mesh<INPUT>>,
        options: Options,
        cache: ContractionCache,
    ) -> Self {
        let stop = SkeletonStage::Complete;
        Self::extract_skeleton_until(mesh, options, stop, cache)
    }

    pub fn extract_skeleton_until(
        mesh: Arc<mehsh::prelude::Mesh<INPUT>>,
        options: Options,
        stop: SkeletonStage,
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
                    if matches!(
                        stop,
                        SkeletonStage::Labeling
                            | SkeletonStage::Placement
                            | SkeletonStage::Complete
                    ) && let SkeletonState::Extracted(extraction) = &state
                    {
                        progress = Progress::from_skeleton(extraction.skeleton.clone());
                        failure = (|| {
                            progress.label(&mesh)?;
                            publish(Update::Progress(progress.clone()));
                            if matches!(stop, SkeletonStage::Placement | SkeletonStage::Complete) {
                                progress.place(&mesh)?;
                                publish(Update::Progress(progress.clone()));
                            }
                            if stop == SkeletonStage::Complete {
                                progress.optimize(&mesh)?;
                                publish(Update::Progress(progress.clone()));
                                progress.finish(&mesh)?;
                            }
                            Ok::<_, dualcube::skeleton::pipeline::Failure>(())
                        })()
                        .err();
                    }
                    if let Some(error) = &failure {
                        warn!("Initialization stopped at {error}");
                    }
                    let view = if matches!(
                        stop,
                        SkeletonStage::Labeling
                            | SkeletonStage::Placement
                            | SkeletonStage::Complete
                    ) && progress.labeling.is_none()
                    {
                        ModelViewKind::Skeleton
                    } else if matches!(stop, SkeletonStage::Placement | SkeletonStage::Complete)
                        && progress.realized.is_none()
                    {
                        ModelViewKind::Labeled
                    } else {
                        stop.view()
                    };
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
    stop: SkeletonStage,
    cache: &ContractionCache,
) -> dualcube::skeleton::SkeletonResult<SkeletonState> {
    extract_until_with_updates(mesh, options, stop, cache, &|_| {})
}

fn extract_until_with_updates(
    mesh: &Arc<mehsh::prelude::Mesh<INPUT>>,
    mut options: Options,
    stop: SkeletonStage,
    cache: &ContractionCache,
    publish: &impl Fn(Update),
) -> dualcube::skeleton::SkeletonResult<SkeletonState> {
    options.surgery.validate()?;
    let contracted = cache.get_or_compute(mesh.clone(), options.max_iterations)?;
    publish(Update::State(SkeletonState::Deflated(
        (*contracted).clone(),
    )));
    if stop == SkeletonStage::Deflation {
        return Ok(SkeletonState::Deflated((*contracted).clone()));
    }
    let refine_embedding = stop == SkeletonStage::Refinement
        || (matches!(
            stop,
            SkeletonStage::Labeling | SkeletonStage::Placement | SkeletonStage::Complete
        ) && options.refine_embedding);
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
