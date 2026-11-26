//! Manual skeleton edits.

use crate::{Skeleton, SkeletonEdgeID, SkeletonNodeID, SkeletonResult};
use dualcube_types::prelude::*;

/// Whether [`dissolve`] accepts the boundary: its patches are distinct and share no other
/// boundary, which would become a self-loop.
pub fn dissolvable(skeleton: &Skeleton, edge: SkeletonEdgeID) -> bool {
    let Some(&crate::SkeletonEdge { nodes: [a, b], .. }) = skeleton.edges.get(edge) else {
        return false;
    };
    a != b
        && !skeleton
            .edges
            .iter()
            .any(|(e, data)| e != edge && data.nodes.contains(&a) && data.nodes.contains(&b))
}

/// Delete a boundary between two distinct patches, merging them into the larger one, placed at
/// their area-weighted mean. Genus adds under the merge and the loops of the union come from its
/// parts, so patches stay genus zero and contractible in the volume: no certification is
/// needed. Parallel edges are subdivided as after surgery. Returns the merged patch.
pub fn dissolve(
    mesh: &Mesh<INPUT>,
    skeleton: &mut Skeleton,
    edge: SkeletonEdgeID,
) -> SkeletonResult<SkeletonNodeID> {
    if !dissolvable(skeleton, edge) {
        return Err("Only a boundary between two patches that share no other one dissolves".into());
    }
    let [a, b] = skeleton.edges[edge].nodes;
    let areas = crate::geometry::SurfaceGeometry::build(mesh, skeleton)?.areas();
    let (keep, other) = if areas[a] >= areas[b] { (a, b) } else { (b, a) };
    let total = areas[keep] + areas[other];
    let position = if total > 0. {
        (skeleton.nodes[keep].position * areas[keep]
            + skeleton.nodes[other].position * areas[other])
            / total
    } else {
        (skeleton.nodes[keep].position + skeleton.nodes[other].position) / 2.
    };
    merge(mesh, skeleton, edge, keep);
    skeleton.nodes[keep].position = position;
    crate::skeletonize::surgery::simplify(mesh, skeleton);
    skeleton.validate(mesh)?;
    Ok(keep)
}

/// [`dissolve`] each boundary in turn, skipping those that are gone or no longer dissolvable after
/// the earlier ones; the number dissolved.
pub fn dissolve_all(
    mesh: &Mesh<INPUT>,
    skeleton: &mut Skeleton,
    edges: &[SkeletonEdgeID],
) -> SkeletonResult<usize> {
    let mut dissolved = 0;
    for &edge in edges {
        if dissolvable(skeleton, edge) {
            dissolve(mesh, skeleton, edge)?;
            dissolved += 1;
        }
    }
    Ok(dissolved)
}

/// Delete a boundary and merge its other patch into `keep`, without checks; returns that patch.
pub(crate) fn merge(
    mesh: &Mesh<INPUT>,
    skeleton: &mut Skeleton,
    edge: SkeletonEdgeID,
    keep: SkeletonNodeID,
) -> SkeletonNodeID {
    let removed = skeleton.edges.remove(edge).unwrap();
    let other = removed.other(keep);
    for data in skeleton.edges.values_mut() {
        for n in &mut data.nodes {
            if *n == other {
                *n = keep;
            }
        }
    }
    for crossing in removed.boundary.crossings {
        skeleton.surface.remove(mesh, crossing);
    }
    skeleton.nodes.remove(other);
    other
}

#[cfg(test)]
mod tests;
