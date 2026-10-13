use super::*;
use crate::{LabeledSkeleton, label::Port};

fn placed(ports: [Port; 2]) -> (Mesh<INPUT>, RealizedSkeleton) {
    let (mesh, skeleton) = crate::surface::tests::lenses(2);
    let labels = skeleton.edges.keys().zip(ports.map(Port::label)).collect();
    let labeled = LabeledSkeleton { skeleton, labels };
    let placed = place_cubes::run(&mesh, &labeled).unwrap();
    (mesh, placed)
}

#[test]
fn dissolves_straight_patch_and_its_occurrences_without_mutating_source() {
    let (mesh, source) = placed([Port::PositiveX, Port::NegativeX]);
    let order = source.labeled.skeleton.surface.edge_order.clone();
    let result = run(&mesh, &source).unwrap();
    result.validate(&mesh).unwrap();
    assert_eq!(result.labeled.skeleton.nodes.len(), 2);
    assert_eq!(result.labeled.skeleton.edges.len(), 1);
    assert_eq!(result.labeled.skeleton.surface.crossings.len(), 2);
    assert_eq!(result.cuboids.cuboids.len(), 3);
    assert_eq!(source.labeled.skeleton.surface.edge_order, order);
    assert_eq!(source.labeled.skeleton.nodes.len(), 3);
    let again = run(&mesh, &result).unwrap();
    assert_eq!(again.node_positions, result.node_positions);
}

#[test]
fn stored_orientation_does_not_change_simplification() {
    let (mesh, mut source) = placed([Port::PositiveX, Port::NegativeX]);
    let edge = source.labeled.skeleton.edges.keys().next().unwrap();
    let mut boundary = source.labeled.skeleton.edges[edge].boundary.clone();
    crate::surface::edit::reverse(&mesh, &mut source.labeled.skeleton, &mut boundary);
    source.labeled.skeleton.edges[edge].boundary = boundary;
    source.labeled.skeleton.edges[edge].nodes.reverse();
    source.labeled.labels[edge] = Port::NegativeX.label();
    source = place_cubes::run(&mesh, &source.labeled).unwrap();
    assert_eq!(run(&mesh, &source).unwrap().labeled.skeleton.edges.len(), 1);
}

#[test]
fn preserves_bends() {
    let (mesh, source) = placed([Port::PositiveX, Port::PositiveY]);
    let result = run(&mesh, &source).unwrap();
    assert_eq!(result.node_positions, source.node_positions);
    assert_eq!(
        result.labeled.skeleton.surface.edge_order,
        source.labeled.skeleton.surface.edge_order
    );
}

#[test]
fn dissolves_an_entire_straight_chain() {
    let (mesh, mut skeleton) = crate::surface::tests::lenses(1);
    let edge = skeleton.edges.keys().next().unwrap();
    for _ in 0..5 {
        let old = skeleton.edges[edge].clone();
        let boundary = crate::surface::edit::parallel(&mesh, &mut skeleton, &old.boundary, 1);
        let node = skeleton.nodes.insert(crate::SkeletonNode {
            position: Vector3D::zeros(),
        });
        skeleton.edges[edge].nodes[1] = node;
        skeleton.edges.insert(crate::SkeletonEdge {
            nodes: [node, old.nodes[1]],
            boundary,
        });
    }
    let labels = skeleton
        .edges
        .keys()
        .map(|e| (e, Port::PositiveX.label()))
        .collect();
    let source = place_cubes::run(&mesh, &LabeledSkeleton { skeleton, labels }).unwrap();
    let result = run(&mesh, &source).unwrap();
    result.validate(&mesh).unwrap();
    assert_eq!(result.labeled.skeleton.edges.len(), 1);
    assert_eq!(result.labeled.skeleton.nodes.len(), 2);
    assert_eq!(result.labeled.skeleton.surface.crossings.len(), 2);
}
