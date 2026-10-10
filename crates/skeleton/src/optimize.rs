//! Optimize/simplify the placed skeleton to reduce dissimilarity between patches and their target cubes.

use crate::{RealizedSkeleton, SkeletonResult};
use dualcube_types::prelude::{INPUT, Mesh};

/// Optimize using the cubes as a reference for the surface patches.
pub fn run(_mesh: &Mesh<INPUT>, _skeleton: &RealizedSkeleton) -> SkeletonResult<RealizedSkeleton> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "Skeleton optimization is not implemented yet",
    )
    .into())
}
