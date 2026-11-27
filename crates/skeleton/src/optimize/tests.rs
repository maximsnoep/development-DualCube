use super::*;
use crate::label::Port;

fn placed(ports: [Port; 2]) -> (Mesh<INPUT>, RealizedSkeleton) {
    let (mesh, skeleton) = crate::surface::tests::lenses(2);
    let labels = skeleton.edges.keys().zip(ports.map(Port::label)).collect();
    let labeled = LabeledSkeleton { skeleton, labels };
    let placed = place_cubes::run(&mesh, &labeled).unwrap();
    (mesh, placed)
}

#[test]
fn reduces_without_mutating_source_and_is_idempotent() {
    let (mesh, source) = placed([Port::PositiveX, Port::NegativeX]);
    let order = source.labeled.skeleton.surface.edge_order.clone();
    let result = run(&mesh, &source).unwrap();
    result.validate(&mesh).unwrap();
    assert_eq!(result.labeled.skeleton.nodes.len(), 2);
    assert_eq!(result.labeled.skeleton.surface.crossings.len(), 2);
    assert_eq!(source.labeled.skeleton.surface.edge_order, order);
    assert_eq!(source.labeled.skeleton.nodes.len(), 3);
    // Labels tie on these flat loops, so only the structure is compared.
    let again = run(&mesh, &result).unwrap().labeled.skeleton;
    assert_eq!(again.nodes.len(), 2);
    assert_eq!(
        again.surface.edge_order,
        result.labeled.skeleton.surface.edge_order
    );
}

#[test]
fn stored_orientation_does_not_change_the_result() {
    let (mesh, source) = placed([Port::PositiveX, Port::NegativeX]);
    let mut reversed = source.clone();
    let edge = reversed.labeled.skeleton.edges.keys().next().unwrap();
    let mut boundary = reversed.labeled.skeleton.edges[edge].boundary.clone();
    crate::surface::edit::reverse(&mesh, &mut reversed.labeled.skeleton, &mut boundary);
    reversed.labeled.skeleton.edges[edge].boundary = boundary;
    reversed.labeled.skeleton.edges[edge].nodes.reverse();
    reversed.labeled.labels[edge] = Port::NegativeX.label();
    let reversed = place_cubes::run(&mesh, &reversed.labeled).unwrap();
    let [a, b] = [source, reversed].map(|s| run(&mesh, &s).unwrap().labeled.skeleton);
    assert_eq!(a.edges.len(), b.edges.len());
    assert_eq!(a.surface.crossings.len(), b.surface.crossings.len());
}

#[test]
fn reduces_an_entire_straight_chain() {
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
