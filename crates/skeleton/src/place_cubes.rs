//! Place target cubes from the combinatorial labeling.

use crate::{LabeledSkeleton, RealizedSkeleton, SkeletonResult};
use dualcube_types::prelude::{INPUT, Mesh};

/// Choose cube counts, sizes, and positions consistent with the labels.
/// Preserve the labeled skeleton and assign cube ownership and face adjacencies.
/// Edge chains have one midpoint cube splitting the two endpoint patches.
pub fn run(_mesh: &Mesh<INPUT>, _skeleton: &LabeledSkeleton) -> SkeletonResult<RealizedSkeleton> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "Skeleton cube placement is not implemented yet",
    )
    .into())
}
