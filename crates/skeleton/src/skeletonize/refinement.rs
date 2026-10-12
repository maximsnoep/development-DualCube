//! Area-weighted embedding refinement over boundary-defined surface regions.

use crate::{Skeleton, SkeletonResult};
use dualcube_types::prelude::*;

pub fn run(mesh: &Mesh<INPUT>, skeleton: &mut Skeleton) -> SkeletonResult<()> {
    let geometry = crate::geometry::SurfaceGeometry::build(mesh, skeleton)?;
    let centers = geometry.centroids()?;
    if skeleton.nodes.keys().any(|id| !centers.contains_key(id)) {
        return Err("Surface geometry omitted a region".into());
    }
    for (id, node) in &mut skeleton.nodes {
        node.position = centers[id];
    }
    Ok(())
}
