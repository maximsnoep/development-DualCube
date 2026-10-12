//! Junction subdivision is paused during the boundary-loop migration.

use super::*;

pub(super) fn reduce(_mesh: &Mesh<INPUT>, _result: &mut Attempt) -> SkeletonResult<()> {
    // TODO(step 4): Compare boundary separators and their local port costs, then split atomically.
    // TODO: Search alternative split sequences instead of committing greedily.
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "Boundary-loop junction subdivision is not implemented yet",
    )
    .into())
}
