//! Move skeleton nodes to the area-weighted centroid of their surface patches.

use crate::{Skeleton, SkeletonResult};
use dualcube_types::prelude::*;

pub fn run(mesh: &Mesh<INPUT>, skeleton: &mut Skeleton) -> SkeletonResult<()> {
    skeleton.validate(mesh)?;
    let mut positions = Vec::with_capacity(skeleton.nodes.len());
    for (id, node) in &skeleton.nodes {
        let patch: HashSet<_> = node.patch_vertices.iter().copied().collect();
        let mut weighted = Vector3D::zeros();
        let mut area = 0.0;
        for &v in &node.patch_vertices {
            let weight: f64 = mesh
                .faces(v)
                .map(|f| {
                    let count = mesh.vertices(f).filter(|v| patch.contains(v)).count();
                    mesh.size(f) * count as f64 / 3.0
                })
                .sum();
            weighted += mesh.position(v) * weight;
            area += weight;
        }
        if !area.is_finite() || area <= 0.0 || !weighted.iter().all(|x| x.is_finite()) {
            return Err(format!("Patch {id:?} has no finite positive area").into());
        }
        positions.push((id, weighted / area));
    }
    for (id, position) in positions {
        skeleton.nodes[id].position = position;
    }
    Ok(())
}
