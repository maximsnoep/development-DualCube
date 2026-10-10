//! Surface dual loops from the optimized skeleton and its target cubes.

use crate::{DualLoops, RealizedSkeleton, SkeletonResult};
use dualcube_types::prelude::{INPUT, Mesh};

/// Draw loops on the original mesh to create a valid dual loop structure.
///
/// Patch boundaries are lifted to become dual loops, with their assigned direction from the labeling.
/// All other loops are drawn using the target cubes as a template.
pub fn run(_mesh: &Mesh<INPUT>, _skeleton: &RealizedSkeleton) -> SkeletonResult<DualLoops> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "Skeleton dual-loop drawing is not implemented yet",
    )
    .into())
}
