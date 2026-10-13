//! Simplify straight degree-two patches and recompute target placement.

use crate::{RealizedSkeleton, SkeletonResult, place_cubes};
use dualcube_types::prelude::*;

/// Absorb a straight annular patch into one neighbor, deleting its shared boundary.
pub fn run(mesh: &Mesh<INPUT>, placed: &RealizedSkeleton) -> SkeletonResult<RealizedSkeleton> {
    placed.validate(mesh)?;
    let mut labeled = placed.labeled.clone();
    let mut dissolved = 0;
    loop {
        let mut incident = HashMap::<_, Vec<_>>::new();
        for (edge, data) in &labeled.skeleton.edges {
            for node in data.nodes {
                incident.entry(node).or_default().push(edge);
            }
        }
        let candidate = labeled.skeleton.nodes.keys().find_map(|node| {
            let edges = incident.get(&node)?;
            if edges.len() != 2
                || labeled.labels[edges[0]].direction != labeled.labels[edges[1]].direction
            {
                return None;
            }
            // Valid local ports on one axis must be opposite, irrespective of stored edge orientation.
            Some((node, edges[0], edges[1]))
        });
        let Some((node, remove, keep)) = candidate else {
            break;
        };
        // TODO: Choose which boundary to retain using surface/label costs.
        let removed = labeled.skeleton.edges.remove(remove).unwrap();
        let neighbor = removed.nodes.into_iter().find(|&n| n != node).unwrap();
        for n in &mut labeled.skeleton.edges[keep].nodes {
            if *n == node {
                *n = neighbor;
            }
        }
        for crossing in removed.boundary.crossings {
            let data = labeled.skeleton.surface.crossings.remove(crossing).unwrap();
            let edge = crate::surface::canonical(mesh, data.exit);
            labeled
                .skeleton
                .surface
                .edge_order
                .get_mut(&edge)
                .unwrap()
                .retain(|&c| c != crossing);
        }
        labeled
            .skeleton
            .surface
            .edge_order
            .retain(|_, order| !order.is_empty());
        labeled.skeleton.nodes.remove(node);
        labeled.labels.remove(remove);
        dissolved += 1;
    }
    if dissolved == 0 {
        return Ok(placed.clone());
    }
    let result = place_cubes::run(mesh, &labeled)?;
    tracing::info!("Optimization: dissolved {dissolved} straight degree-two patches");
    // TODO: Optimize remaining patches, labels, and geometry against the target cubes.
    Ok(result)
}

#[cfg(test)]
mod tests;
