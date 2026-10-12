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
    assert_eq!(first.cuboids.cuboids.len(), 7);
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
fn slack_compaction_and_long_edges_use_one_stretched_cuboid() {
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
        .cuboids
        .cuboids
        .values()
        .filter(|c| c.owner == CuboidOwner::Edge(edge))
        .collect();
    assert_eq!(cubes.len(), 1);
    assert_eq!(cubes[0].size, Vector3D::new(5., 1., 1.));
    assert_eq!(cubes[0].center, Vector3D::new(3., 2., 0.));
    assert_eq!(result.cuboids.cuboids.len(), 12);
    assert_eq!(result.cuboids.connections.len(), 12);
    for direction in [Direction::X, Direction::Y, Direction::Z] {
        for sign in [Sign::Positive, Sign::Negative] {
            let mut rotated = labeled.clone();
            for (_, label) in &mut rotated.labels {
                label.direction = match (label.direction, direction) {
                    (Direction::X, d) => d,
                    (Direction::Y, Direction::Y) => Direction::Z,
                    (Direction::Y, Direction::Z) => Direction::X,
                    (d, _) => d,
                };
                label.sign = sign;
            }
            let placed = realize(&rotated).unwrap();
            let cuboid = placed
                .cuboids
                .cuboids
                .values()
                .find(|c| c.owner == CuboidOwner::Edge(edge))
                .unwrap();
            let axis = Port::from_label(rotated.labels[edge]) as usize / 2;
            assert_eq!(cuboid.size[axis], 5.);
            assert!(cuboid.size.iter().all(|&x| x == 1. || x == 5.));
            placed.validate_geometry().unwrap();
        }
    }
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
    assert_eq!(result.cuboids.cuboids.len(), 9);
    assert_eq!(result.cuboids.connections.len(), 8);
    result.validate_geometry().unwrap();
    let single = realize(&graph(1, &[])).unwrap();
    assert_eq!(single.cuboids.cuboids.len(), 1);
    assert!(single.cuboids.connections.is_empty());
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
    bad.cuboids.connections.pop();
    assert!(bad.validate_geometry().is_err());
    let mut bad = valid.clone();
    let id = bad
        .cuboids
        .cuboids
        .iter()
        .find(|(_, c)| matches!(c.owner, CuboidOwner::Edge(_)))
        .unwrap()
        .0;
    bad.cuboids.cuboids[id].center.y += 1.;
    assert!(bad.validate_geometry().is_err());
    for size in [
        Vector3D::new(2., 1., 1.),
        Vector3D::new(1., 2., 1.),
        Vector3D::new(0., 1., 1.),
        Vector3D::repeat(f64::NAN),
    ] {
        let mut bad = valid.clone();
        bad.cuboids.cuboids[id].size = size;
        assert!(bad.validate_geometry().is_err());
    }
    let mut bad = valid.clone();
    bad.cuboids.cuboids.insert(bad.cuboids.cuboids[id].clone());
    assert!(bad.validate_geometry().is_err());
    let mut bad = valid.clone();
    bad.cuboids.cuboids.remove(id);
    assert!(bad.validate_geometry().is_err());
    let mut bad = valid.clone();
    bad.cuboids.connections.push(bad.cuboids.connections[0]);
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
