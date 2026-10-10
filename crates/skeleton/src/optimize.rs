//! Optimize/simplify the labeled skeleton to reduce dissimarity between patches and their target cubes.

use crate::{LabeledSkeleton, StageResult};
use dualcube_types::prelude::{INPUT, Mesh};

/// Optimize using the cubes as a reference for the surface patches.
pub fn run(_mesh: &Mesh<INPUT>, _skeleton: &LabeledSkeleton) -> StageResult<LabeledSkeleton> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "Skeleton optimization is not implemented yet",
    )
    .into())
}
