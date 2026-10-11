use super::{Job, JobResult};
use bevy::prelude::*;
use dualcube::prelude::*;
use dualcube::skeleton::skeletonize::{self, Options};
use std::sync::Arc;

impl Job {
    pub fn extract_skeleton(mesh: Arc<mehsh::prelude::Mesh<INPUT>>, options: Options) -> Self {
        Self::new("extracting skeleton", move || {
            match skeletonize::extract(&mesh, options) {
                Ok(extraction) => {
                    info!(
                        "Skeleton: {} nodes, {} edges",
                        extraction.skeleton.nodes.len(),
                        extraction.skeleton.edges.len()
                    );
                    Some(JobResult::SkeletonExtracted(extraction))
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
