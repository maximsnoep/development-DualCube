//! Signed-axis coordinates, node cubes, and stretched edge cuboids.

use crate::{
    Cuboid, CuboidLayout, CuboidOwner, LabeledSkeleton, RealizedSkeleton, SkeletonNodeID,
    SkeletonResult,
};
use dualcube_types::prelude::*;
use slotmap::SecondaryMap;

mod validate;

/// Recompute target coordinates from labels alone. Input-space positions are unchanged.
/// Two units per level leave a positive gap between unit node cubes.
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

/// Connect unit node cubes with one cuboid per edge, preserving identities at overlaps.
/// Rebuilds all placement data, so this can be called again after relabeling.
pub fn run(mesh: &Mesh<INPUT>, labeled: &LabeledSkeleton) -> SkeletonResult<RealizedSkeleton> {
    labeled.skeleton.validate(mesh)?;
    realize(labeled)
}

fn realize(labeled: &LabeledSkeleton) -> SkeletonResult<RealizedSkeleton> {
    let node_positions = coordinates(labeled)?;
    let mut cuboids = CuboidLayout::default();
    let mut nodes = SecondaryMap::new();
    for id in labeled.skeleton.nodes.keys() {
        nodes.insert(
            id,
            cuboids.cuboids.insert(Cuboid {
                center: node_positions[id],
                size: Vector3D::repeat(1.),
                owner: CuboidOwner::Node(id),
            }),
        );
    }
    for (edge, data) in &labeled.skeleton.edges {
        let [a, b] = data.nodes;
        let delta = node_positions[b] - node_positions[a];
        let axis = crate::label::Port::from_label(labeled.labels[edge]) as usize / 2;
        let mut size = Vector3D::repeat(1.);
        size[axis] = delta[axis].abs() - 1.;
        let connection = cuboids.cuboids.insert(Cuboid {
            center: (node_positions[a] + node_positions[b]) / 2.,
            size,
            owner: CuboidOwner::Edge(edge),
        });
        cuboids
            .connections
            .extend([[nodes[a], connection], [connection, nodes[b]]]);
    }
    let result = RealizedSkeleton {
        labeled: labeled.clone(),
        node_positions,
        cuboids,
    };
    result.validate_geometry()?;
    tracing::info!(
        "Cube placement: {} node cubes, {} edge cuboids",
        result.node_positions.len(),
        result.labeled.skeleton.edges.len()
    );
    Ok(result)
}

#[cfg(test)]
mod tests;
