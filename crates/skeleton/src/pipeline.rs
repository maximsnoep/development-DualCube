//! Specifies the full skeleton-based initialization pipeline.

use crate::{
    DualLoops, LabeledSkeleton, RealizedSkeleton, Skeleton, SkeletonResult, draw_loops, label,
    optimize, place_cubes, skeletonize,
};
use dualcube_types::prelude::{INPUT, Mesh};

/// Intermediate states and final loops from a completed initialization run.
#[derive(Clone, Debug)]
pub struct SkeletonRun {
    pub skeleton: Skeleton,
    pub labeled: LabeledSkeleton,
    pub realized: RealizedSkeleton,
    pub optimized: RealizedSkeleton,
    pub loops: DualLoops,
}

/// Run all steps, retaining each result for inspection.
/// Call steps individually to retain partial results if a later step fails.
pub fn run(mesh: &Mesh<INPUT>) -> SkeletonResult<SkeletonRun> {
    let skeleton = skeletonize::run(mesh)?;
    let labeled = label::run(mesh, &skeleton)?;
    let realized = place_cubes::run(mesh, &labeled)?;
    let optimized = optimize::run(mesh, &realized)?;
    let loops = draw_loops::run(mesh, &optimized)?;

    Ok(SkeletonRun {
        skeleton,
        labeled,
        realized,
        optimized,
        loops,
    })
}
