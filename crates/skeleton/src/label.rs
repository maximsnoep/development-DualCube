//! Make an orthogonal drawing of the skeleton, i.e., label each edge with a signed axis-direction. 
//! Produced labeling implies a polycube surface built from cubes.

use crate::{LabeledSkeleton, Skeleton, StageResult};
use dualcube_types::prelude::{INPUT, Mesh};

/// Label the skeleton and produce polycube surface.
/// 
/// Edges are labeled with +/- {X,Y,Z} such that the graph represents a valid orthogonal drawing.
/// This is a combinatorial structure, lengths are not yet assigned. TODO: perhaps we do do this to make cubes easier?
/// 
/// The difficulty in this step primarily comes from dependencies between cycles. Tree-like parts are easy to label.
pub fn run(_mesh: &Mesh<INPUT>, _skeleton: &Skeleton) -> StageResult<LabeledSkeleton> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "Skeleton labeling and cube construction are not implemented yet",
    )
    .into())
}
