use super::*;
use crate::surface::tests::{bands, lenses};

#[test]
fn dissolve_merges_two_patches() {
    let (mesh, mut skeleton) = lenses(2);
    let edge = skeleton.edges.keys().next().unwrap();
    let [a, b] = skeleton.edges[edge].nodes;
    let keep = dissolve(&mesh, &mut skeleton, edge).unwrap();
    assert!(keep == a || keep == b);
    assert_eq!((skeleton.nodes.len(), skeleton.edges.len()), (2, 1));
    assert_eq!(skeleton.surface.crossings.len(), 2);
    skeleton.validate(&mesh).unwrap();
}

#[test]
fn a_loop_whose_patches_share_another_is_refused() {
    let (mesh, mut skeleton) = bands(&[[3., 0., 0.], [-3., 0., 0.]]);
    let edge = skeleton.edges.keys().next().unwrap();
    assert!(!dissolvable(&skeleton, edge));
    let before = skeleton.surface.crossings.len();
    assert!(dissolve(&mesh, &mut skeleton, edge).is_err());
    assert_eq!(skeleton.surface.crossings.len(), before);
}

#[test]
fn parallel_edges_are_subdivided_on_a_cycle() {
    let (mesh, mut skeleton) = bands(&[[3., 0., 0.], [-1.5, 2.6, 0.], [-1.5, -2.6, 0.]]);
    let edge = skeleton.edges.keys().next().unwrap();
    assert!(dissolvable(&skeleton, edge));
    dissolve(&mesh, &mut skeleton, edge).unwrap();
    skeleton.validate(&mesh).unwrap();
    // Still one cycle (genus one), now without parallel edges.
    assert_eq!(skeleton.edges.len(), skeleton.nodes.len());
    let mut pairs = HashSet::new();
    for edge in skeleton.edges.values() {
        let [a, b] = edge.nodes;
        assert!(pairs.insert(if a < b { [a, b] } else { [b, a] }));
    }
}
