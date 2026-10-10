//! Checks for the skeleton's surface partition and topology.

use crate::{Skeleton, SkeletonResult};
use dualcube_types::prelude::{INPUT, Mesh};

impl Skeleton {
    /// Validate against the original mesh after extraction or partition edits.
    pub fn validate(&self, _mesh: &Mesh<INPUT>) -> SkeletonResult<()> {
        // TODO: Check vertex ownership, connected genus-zero patches, disjoint
        // simple boundary loops without triple junctions, graph/boundary
        // correspondence.
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "Skeleton topology validation is not implemented yet",
        )
        .into())
    }
}
