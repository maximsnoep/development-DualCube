//! Embedding refinement awaits boundary-region geometry.

use crate::{Skeleton, SkeletonResult};
use dualcube_types::prelude::*;

pub fn run(_mesh: &Mesh<INPUT>, _skeleton: &mut Skeleton) -> SkeletonResult<()> {
    // TODO(step 5): Integrate area-weighted centroids over boundary-defined regions.
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "Boundary-region centroid refinement is not implemented yet",
    )
    .into())
}
