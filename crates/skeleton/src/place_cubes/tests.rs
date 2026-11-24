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
    assert_eq!(first.cuboids().cuboids.len(), 7);
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
    let layout = result.cuboids();
    let cubes: Vec<_> = layout
        .cuboids
        .values()
        .filter(|c| c.owner == CuboidOwner::Edge(edge))
        .collect();
    assert_eq!(cubes.len(), 1);
    assert_eq!(cubes[0].size, Vector3D::new(5., 1., 1.));
    assert_eq!(cubes[0].center, Vector3D::new(3., 2., 0.));
    assert_eq!(layout.cuboids.len(), 12);
    assert_eq!(layout.connections.len(), 12);
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
            let layout = placed.cuboids();
            let cuboid = layout
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
    assert_eq!(result.cuboids().cuboids.len(), 9);
    assert_eq!(result.cuboids().connections.len(), 8);
    result.validate_geometry().unwrap();
    let single = realize(&graph(1, &[])).unwrap();
    assert_eq!(single.cuboids().cuboids.len(), 1);
    assert!(single.cuboids().connections.is_empty());
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
    let nodes: Vec<_> = valid.labeled.skeleton.nodes.keys().collect();
    // Off the edge's axis, against its sign, and too close for a cuboid between the cubes.
    for (node, offset) in [
        (1, Vector3D::y()),
        (1, Vector3D::x() * -4.),
        (1, Vector3D::x() * -1.),
    ] {
        let mut bad = valid.clone();
        bad.node_positions[nodes[node]] += offset;
        assert!(bad.validate_geometry().is_err());
    }
    let mut bad = valid.clone();
    bad.node_positions.remove(nodes[0]);
    assert!(bad.validate_geometry().is_err());
    let mut bad = valid;
    let node = bad.labeled.skeleton.nodes.keys().next().unwrap();
    bad.node_positions[node].x = f64::NAN;
    assert!(bad.validate_geometry().is_err());
}

#[test]
fn placement_handles_actual_surface_splits_and_repaired_cycles() {
    let (mesh, skeleton) = crate::surface::tests::lenses(8);
    let labeled = crate::label::attempt(&mesh, &skeleton).unwrap().labeled();
    run(&mesh, &labeled).unwrap().validate(&mesh).unwrap();
    let (mesh, skeleton) =
        crate::surface::tests::bands(&[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]);
    let labeled = crate::label::attempt(&mesh, &skeleton).unwrap().labeled();
    run(&mesh, &labeled).unwrap().validate(&mesh).unwrap();
}

#[test]
fn template_solids_carry_closed_loops() {
    use Port::*;
    // A path of three nodes: the middle one a cube, the leaves' cubes in the edge cuboids.
    let labeled = graph(3, &[(0, 1, PositiveX), (1, 2, PositiveY)]);
    let realized = realize(&labeled).unwrap();
    let solids = realized.template_solids();
    assert_eq!(solids.len(), 3);
    let free = |s: &TemplateSolid| s.hidden.iter().flatten().filter(|h| !**h).count();
    assert_eq!(solids.iter().map(free).sum::<usize>(), 4 + 5 + 5);
    // Every edge solid reaches over its leaf's cube.
    let leaves =
        [0, 2].map(|i| realized.node_positions[labeled.skeleton.nodes.keys().nth(i).unwrap()]);
    for leaf in leaves {
        let covers = |s: &TemplateSolid| {
            (0..3).all(|k| s.lo[k] <= leaf[k] - 0.5 && leaf[k] + 0.5 <= s.hi[k])
        };
        assert_eq!(solids.iter().filter(|s| covers(s)).count(), 1);
    }
    // Two sections per free face; the patch boundaries go around the edge solids.
    let sections: Vec<_> = solids.iter().flat_map(TemplateSolid::sections).collect();
    assert_eq!(sections.len(), 2 * 14);
    assert_eq!(sections.iter().filter(|s| s.boundary).count(), 2 * 4);
    // Each end of a section meets exactly one other section of its loop axis: the loops close.
    let key = |p: Vector3D| p.map(|x| (2. * x).round() as i64);
    let mut ends: HashMap<_, usize> = HashMap::new();
    for section in &sections {
        for end in section.ends {
            *ends.entry((section.axis, key(end))).or_default() += 1;
        }
    }
    assert!(ends.values().all(|&count| count == 2), "{ends:?}");
}
