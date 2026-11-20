//! Simplify a placed skeleton by valid moves, then recompute target placement.
//!
//! The search over moves (ALNS against the primalized quality) lives on the branch
//! `skeleton-experiments`.

use crate::{
    LabeledSkeleton, RealizedSkeleton, SkeletonEdgeID, SkeletonNodeID, SkeletonResult, place_cubes,
};
use dualcube_types::prelude::*;

/// Dissolve straight degree-two patches only.
pub fn run(mesh: &Mesh<INPUT>, placed: &RealizedSkeleton) -> SkeletonResult<RealizedSkeleton> {
    placed.validate(mesh)?;
    place_cubes::run(mesh, &dissolve_straight(mesh, &placed.labeled)?)
}

/// Absorb straight annular patches into a neighbour, deleting their shared boundary.
pub fn dissolve_straight(
    mesh: &Mesh<INPUT>,
    labeled: &LabeledSkeleton,
) -> SkeletonResult<LabeledSkeleton> {
    let mut labeled = labeled.clone();
    loop {
        let degree = labeled.skeleton.degrees();
        let candidate = labeled.skeleton.nodes.keys().find_map(|node| {
            if degree[node] != 2 {
                return None;
            }
            let edges: Vec<_> = labeled.skeleton.incident(node).map(|(id, _)| id).collect();
            // Valid local ports on one axis must be opposite, irrespective of stored edge orientation.
            (labeled.labels[edges[0]].direction == labeled.labels[edges[1]].direction)
                .then(|| (edges[0], labeled.skeleton.edges[edges[0]].other(node)))
        });
        let Some((edge, keep)) = candidate else {
            break;
        };
        // TODO: Choose which boundary to retain using surface/label costs.
        dissolve(mesh, &mut labeled, edge, keep)?;
    }
    labeled.validate_labels()?;
    Ok(labeled)
}

/// Delete a boundary and merge its other patch into `keep` (merge lemma). Fails when the
/// graph would get parallel edges; labels may need repair afterwards.
fn dissolve(
    mesh: &Mesh<INPUT>,
    labeled: &mut LabeledSkeleton,
    edge: SkeletonEdgeID,
    keep: SkeletonNodeID,
) -> SkeletonResult<()> {
    let skeleton = &mut labeled.skeleton;
    let removed = skeleton.edges.remove(edge).ok_or("Missing edge")?;
    let other = removed
        .nodes
        .into_iter()
        .find(|&n| n != keep)
        .ok_or("Edge does not join `keep` to another node")?;
    let mut neighbors = HashSet::new();
    for data in skeleton.edges.values_mut() {
        for n in &mut data.nodes {
            if *n == other {
                *n = keep;
            }
        }
        if data.nodes[0] == data.nodes[1] {
            return Err("Merging would create a self-loop".into());
        }
        if data.nodes.contains(&keep) {
            let neighbor = data.other(keep);
            if !neighbors.insert(neighbor) {
                return Err("Merging would create parallel edges".into());
            }
        }
    }
    for crossing in removed.boundary.crossings {
        skeleton.surface.remove(mesh, crossing);
    }
    skeleton.nodes.remove(other);
    labeled.labels.remove(edge);
    Ok(())
}

#[cfg(test)]
mod tests;
