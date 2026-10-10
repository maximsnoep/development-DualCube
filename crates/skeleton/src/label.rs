//! Combinatorial orthogonal drawing: assign each skeleton edge a signed axis.

use crate::{LabeledSkeleton, Skeleton, SkeletonResult};
use dualcube_types::prelude::{INPUT, Mesh};

/// Label the skeleton without assigning lengths or placing cubes.
///
/// Edges are labeled with +/- {X,Y,Z} such that the graph represents a valid orthogonal drawing.
///
/// The difficulty in this step primarily comes from dependencies between cycles. Tree-like parts are easy to label.
pub fn run(_mesh: &Mesh<INPUT>, _skeleton: &Skeleton) -> SkeletonResult<LabeledSkeleton> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "Skeleton labeling is not implemented yet",
    )
    .into())
}
