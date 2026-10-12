//! Signed-axis coordinate assignment and explicit cube chains for inspection.

use crate::{
    Cube, CubeLayout, CubeOwner, LabeledSkeleton, RealizedSkeleton, SkeletonNodeID, SkeletonResult,
};
use dualcube_types::prelude::*;
use slotmap::SecondaryMap;

mod validate;

/// Recompute target coordinates from labels alone. Input-space positions are unchanged.
/// Two units per level leave room for a midpoint cube on even the shortest edge.
pub fn coordinates(
    labeled: &LabeledSkeleton,
) -> SkeletonResult<SecondaryMap<SkeletonNodeID, Vector3D>> {
    if labeled.skeleton.nodes.is_empty() {
        return Err("Cannot place an empty skeleton".into());
    }
    labeled.validate_labels()?;
    let levels = crate::label::constraints::coordinate_levels(
        &labeled.skeleton,
        &labeled.labels,
        None,
        true,
    )
    .ok_or("Labels imply contradictory coordinate orders")?;
    // TODO: Optimize lengths for geometric fidelity beyond the two-pass level compaction.
    Ok(levels
        .iter()
        .map(|(id, p)| {
            (
                id,
                Vector3D::new(2. * p[0] as f64, 2. * p[1] as f64, 2. * p[2] as f64),
            )
        })
        .collect())
}

/// Place unit cubes along the assigned axes, preserving separate identities at overlaps.
/// Rebuilds all placement data, so this can be called again after relabeling.
pub fn run(mesh: &Mesh<INPUT>, labeled: &LabeledSkeleton) -> SkeletonResult<RealizedSkeleton> {
    labeled.skeleton.validate(mesh)?;
    realize(labeled)
}

fn realize(labeled: &LabeledSkeleton) -> SkeletonResult<RealizedSkeleton> {
    let node_positions = coordinates(labeled)?;
    let mut cubes = CubeLayout::default();
    let mut nodes = SecondaryMap::new();
    for id in labeled.skeleton.nodes.keys() {
        nodes.insert(
            id,
            cubes.cubes.insert(Cube {
                center: node_positions[id],
                size: 1.,
                owner: CubeOwner::Node(id),
            }),
        );
    }
    for (edge, data) in &labeled.skeleton.edges {
        let [a, b] = data.nodes;
        let delta = node_positions[b] - node_positions[a];
        let steps = delta.amax() as usize;
        let step = delta / steps as f64;
        let mut previous = nodes[a];
        for i in 1..steps {
            let owner = if i == steps / 2 {
                CubeOwner::Midpoint(edge)
            } else {
                CubeOwner::Edge {
                    edge,
                    patch: if i < steps / 2 { a } else { b },
                }
            };
            let current = cubes.cubes.insert(Cube {
                center: node_positions[a] + step * i as f64,
                size: 1.,
                owner,
            });
            cubes.connections.push([previous, current]);
            previous = current;
        }
        cubes.connections.push([previous, nodes[b]]);
    }
    let result = RealizedSkeleton {
        labeled: labeled.clone(),
        node_positions,
        cubes,
    };
    result.validate_geometry()?;
    tracing::info!(
        "Cube placement: {} nodes, {} cubes",
        result.node_positions.len(),
        result.cubes.cubes.len()
    );
    Ok(result)
}

#[cfg(test)]
mod tests;
