//! Surface collars will be represented by parallel boundary loops.

use super::*;

pub(super) fn collar(
    _mesh: &Mesh<INPUT>,
    _skeleton: &Skeleton,
    _edge: SkeletonEdgeID,
    _side: usize,
) -> SkeletonResult<Option<(Skeleton, SkeletonEdgeID)>> {
    // TODO(step 3): Insert a parallel loop and its intervening annular region.
    // TODO: Compare alternative loop routes jointly with their path labeling costs.
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "Boundary-loop collar subdivision is not implemented yet",
    )
    .into())
}
