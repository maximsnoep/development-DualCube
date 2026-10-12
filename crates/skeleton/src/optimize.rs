//! Optimize/simplify the placed skeleton to reduce dissimilarity between patches and their target cubes.

use crate::{RealizedSkeleton, SkeletonResult};
use dualcube_types::prelude::{INPUT, Mesh};

/// Pass through placement unchanged until optimization is implemented.
pub fn run(_mesh: &Mesh<INPUT>, skeleton: &RealizedSkeleton) -> SkeletonResult<RealizedSkeleton> {
    // TODO: Optimize patches, labels, and placement against the target cubes.
    Ok(skeleton.clone())
}
