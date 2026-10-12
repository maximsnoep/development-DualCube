//! Insert a parallel boundary and the annular patch between the two loops.

use super::*;
use crate::{SkeletonEdge, SkeletonNode};

pub(super) fn collar(
    mesh: &Mesh<INPUT>,
    skeleton: &Skeleton,
    edge: SkeletonEdgeID,
    side: usize,
) -> SkeletonResult<Option<(Skeleton, SkeletonEdgeID)>> {
    if side > 1 {
        return Err("Invalid collar side".into());
    }
    let mut split = skeleton.clone();
    let old = &skeleton.edges[edge];
    let [a, b] = old.nodes;
    let middle = split.nodes.insert(SkeletonNode {
        // This position only supplies initial angle costs; it is not a surface centroid.
        position: (skeleton.nodes[a].position + skeleton.nodes[b].position) * 0.5,
    });
    let copy = crate::surface::edit::parallel(mesh, &mut split, &old.boundary, side);
    split.edges[edge].nodes = [a, middle];
    let boundary = if side == 0 {
        std::mem::replace(&mut split.edges[edge].boundary, copy)
    } else {
        copy
    };
    let added = split.edges.insert(SkeletonEdge {
        nodes: [middle, b],
        boundary,
    });
    // TODO: Compare alternative surface routes jointly with path labeling costs.
    split.validate(mesh)?;
    Ok(Some((split, added)))
}
