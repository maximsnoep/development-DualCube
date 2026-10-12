use super::*;
use crate::{PatchBoundary, Skeleton, SkeletonEdge, SkeletonNode, label::Port};

fn graph(count: usize, edges: &[(usize, usize, Port)]) -> LabeledSkeleton {
    let mut skeleton = Skeleton::default();
    let nodes: Vec<_> = (0..count)
        .map(|i| {
            skeleton.nodes.insert(SkeletonNode {
                position: Vector3D::repeat(i as f64),
            })
        })
        .collect();
    let mut labels = SecondaryMap::new();
    for &(a, b, port) in edges {
        let e = skeleton.edges.insert(SkeletonEdge {
            nodes: [nodes[a], nodes[b]],
            boundary: PatchBoundary::default(),
        });
        labels.insert(e, port.label());
    }
    LabeledSkeleton { skeleton, labels }
}

#[test]
fn signs_equalities_and_repeated_placement_preserve_input_embedding() {
    use Port::*;
    let mut labeled = graph(
        4,
        &[(0, 1, NegativeX), (1, 2, PositiveY), (2, 3, NegativeZ)],
    );
    let before: Vec<_> = labeled
        .skeleton
        .nodes
        .values()
        .map(|n| n.position)
        .collect();
    let first = realize(&labeled).unwrap();
    first.validate_geometry().unwrap();
    let nodes: Vec<_> = labeled.skeleton.nodes.keys().collect();
    assert_eq!(first.node_positions[nodes[0]], Vector3D::new(2., 0., 2.));
    assert_eq!(first.node_positions[nodes[3]], Vector3D::new(0., 2., 0.));
    assert_eq!(first.cubes.cubes.len(), 7);
    for (_, label) in &mut labeled.labels {
        label.sign = if label.sign == Sign::Positive {
            Sign::Negative
        } else {
            Sign::Positive
        };
    }
    let second = realize(&labeled).unwrap();
    assert_ne!(
        second.node_positions[nodes[0]],
        first.node_positions[nodes[0]]
    );
    assert_eq!(
        before,
        labeled
            .skeleton
            .nodes
            .values()
            .map(|n| n.position)
            .collect::<Vec<_>>()
    );
}

#[test]
fn slack_compaction_and_long_edges_keep_midpoint_ownership() {
    use Port::*;
    // Two horizontal rows force a three-level edge in the short row.
    let labeled = graph(
        6,
        &[
            (0, 1, PositiveX),
            (1, 2, PositiveX),
            (2, 3, PositiveX),
            (0, 4, PositiveY),
            (3, 5, PositiveY),
            (4, 5, PositiveX),
        ],
    );
    let result = realize(&labeled).unwrap();
    result.validate_geometry().unwrap();
    let edge = labeled.skeleton.edges.keys().last().unwrap();
    let cubes: Vec<_> = result
        .cubes
        .cubes
        .values()
        .filter(|c| match c.owner {
            CubeOwner::Midpoint(e) | CubeOwner::Edge { edge: e, .. } => e == edge,
            _ => false,
        })
        .collect();
    assert_eq!(cubes.len(), 5);
    assert_eq!(
        cubes
            .iter()
            .filter(|c| matches!(c.owner, CubeOwner::Midpoint(_)))
            .count(),
        1
    );
    let chain = graph(
        6,
        &[
            (0, 1, PositiveX),
            (1, 2, PositiveX),
            (2, 3, PositiveX),
            (3, 4, PositiveY),
            (5, 4, PositiveX),
        ],
    );
    let positions = coordinates(&chain).unwrap();
    let n: Vec<_> = chain.skeleton.nodes.keys().collect();
    assert_eq!(positions[n[5]].x, 4.); // reverse pass shortens the underconstrained edge
}

#[test]
fn intersections_are_not_welded_and_single_node_is_supported() {
    use Port::*;
    let labeled = graph(
        5,
        &[
            (0, 1, PositiveX),
            (1, 2, PositiveY),
            (2, 3, NegativeX),
            (3, 4, NegativeY),
        ],
    );
    let result = realize(&labeled).unwrap();
    let n: Vec<_> = labeled.skeleton.nodes.keys().collect();
    assert_eq!(result.node_positions[n[0]], result.node_positions[n[4]]);
    assert_eq!(result.cubes.cubes.len(), 9);
    assert_eq!(result.cubes.connections.len(), 8);
    result.validate_geometry().unwrap();
    let single = realize(&graph(1, &[])).unwrap();
    assert_eq!(single.cubes.cubes.len(), 1);
    assert!(single.cubes.connections.is_empty());
}

#[test]
fn contradictions_and_corrupt_geometry_are_rejected() {
    use Port::*;
    assert!(
        coordinates(&graph(
            3,
            &[(0, 1, PositiveX), (1, 2, PositiveX), (2, 0, PositiveX)]
        ))
        .is_err()
    );
    assert!(coordinates(&graph(0, &[])).is_err());
    let valid = realize(&graph(2, &[(0, 1, PositiveX)])).unwrap();
    let mut bad = valid.clone();
    bad.cubes.connections.pop();
    assert!(bad.validate_geometry().is_err());
    let mut bad = valid.clone();
    let id = bad
        .cubes
        .cubes
        .iter()
        .find(|(_, c)| matches!(c.owner, CubeOwner::Midpoint(_)))
        .unwrap()
        .0;
    bad.cubes.cubes[id].center.y += 1.;
    assert!(bad.validate_geometry().is_err());
    let mut bad = valid;
    let node = bad.labeled.skeleton.nodes.keys().next().unwrap();
    bad.node_positions[node].x = f64::NAN;
    assert!(bad.validate_geometry().is_err());
}

#[test]
fn placement_handles_actual_surface_splits_and_repaired_cycles() {
    let (mesh, skeleton) = crate::surface::tests::lenses(8);
    let labeled = crate::label::run(&mesh, &skeleton).unwrap();
    run(&mesh, &labeled).unwrap().validate(&mesh).unwrap();
    let (mesh, skeleton) =
        crate::surface::tests::bands(&[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]);
    let labeled = crate::label::run(&mesh, &skeleton).unwrap();
    run(&mesh, &labeled).unwrap().validate(&mesh).unwrap();
}
