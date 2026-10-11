mod cache;
pub use cache::ContractionCache;

use super::{Job, JobResult};
use crate::resources::SkeletonState;
use crate::ui::ModelViewKind;
use bevy::prelude::*;
use dualcube::prelude::*;
use dualcube::skeleton::skeletonize::{self, Options};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkeletonStage {
    Deflation,
    Surgery,
    Refinement,
}

impl SkeletonStage {
    pub fn view(self) -> ModelViewKind {
        match self {
            Self::Deflation => ModelViewKind::Deflated,
            Self::Surgery => ModelViewKind::Raw,
            Self::Refinement => ModelViewKind::Skeleton,
        }
    }
}

impl Job {
    pub fn extract_skeleton(
        mesh: Arc<mehsh::prelude::Mesh<INPUT>>,
        options: Options,
        cache: ContractionCache,
    ) -> Self {
        let stop = if options.refine_embedding {
            SkeletonStage::Refinement
        } else {
            SkeletonStage::Surgery
        };
        Self::extract_skeleton_until(mesh, options, stop, cache)
    }

    pub fn extract_skeleton_until(
        mesh: Arc<mehsh::prelude::Mesh<INPUT>>,
        options: Options,
        stop: SkeletonStage,
        cache: ContractionCache,
    ) -> Self {
        Self::new("extracting skeleton", move || {
            match extract_until(&mesh, options, stop, &cache) {
                Ok(state) => Some(JobResult::SkeletonReady {
                    state,
                    view: stop.view(),
                }),
                Err(error) => {
                    warn!("Failed to extract skeleton: {error}");
                    None
                }
            }
        })
        .preempting()
    }
}

fn extract_until(
    mesh: &Arc<mehsh::prelude::Mesh<INPUT>>,
    mut options: Options,
    stop: SkeletonStage,
    cache: &ContractionCache,
) -> dualcube::skeleton::SkeletonResult<SkeletonState> {
    options.surgery.validate()?;
    let contracted = cache.get_or_compute(mesh.clone(), options.max_iterations)?;
    if stop == SkeletonStage::Deflation {
        return Ok(SkeletonState::Deflated((*contracted).clone()));
    }
    options.refine_embedding = stop == SkeletonStage::Refinement;
    let extraction = skeletonize::extract_from_contracted(mesh, (*contracted).clone(), options)?;
    info!(
        "Skeleton: {} nodes, {} edges",
        extraction.skeleton.nodes.len(),
        extraction.skeleton.edges.len()
    );
    Ok(SkeletonState::Extracted(extraction))
}

#[cfg(test)]
mod tests;
