//! Skeleton extraction / surface partitioning.

use crate::{Skeleton, StageResult};
use dualcube_types::prelude::{INPUT, Mesh};

/// Generate a skeleton, or (nearly) equivalently a surface partition that preserves the topology.
/// The genus of the manifold surface mesh becomes the cycle rank of the produced skeleton graph.
/// 
/// In the skeleton graph, each node correspond to a connected surface patch, and each edge corresponds to a shared boundary between two patches. 
/// The skeleton graph is a topological abstraction of the original mesh, capturing its essential structure while reducing complexity.
pub fn run(_mesh: &Mesh<INPUT>) -> StageResult<Skeleton> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "Skeleton generation is not implemented yet",
    )
    .into())
}
