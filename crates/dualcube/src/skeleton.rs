//! Top-level view of skeleton-based initialization.
//! 
//! Edit stages in here to experiment with different methods.

pub use skeleton::*;

use crate::phases::InputPhase;

/// All data for one run of the skeleton-based initialization pipeline.
#[derive(Clone, Debug)]
pub struct SkeletonRun {
    pub input: InputPhase,
    pub generated: Skeleton,
    pub labeled: LabeledSkeleton,
    pub optimized: LabeledSkeleton,
    pub loops: DualLoops,
}

/// Main method for skeleton-based initialization. Runs all stages and returns the result.
pub fn run(input: InputPhase) -> StageResult<SkeletonRun> {
    let mesh = input.mesh_ref.as_ref();

    // Create a skeleton/surface partition that preserves the topology.
    let skeleton = skeletonize::run(mesh)?;

    let labeled = label::run(mesh, &skeleton)?;
    
    let optimized = optimize::run(mesh, &labeled)?;

    let loops = draw_loops::run(mesh, &optimized)?;

    Ok(SkeletonRun {
        input,
        generated: skeleton,
        labeled,
        optimized,
        loops,
    })
}
