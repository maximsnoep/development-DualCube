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

#[test]
fn dissolve_all_skips_loops_an_earlier_dissolve_made_invalid() {
    let (mesh, mut skeleton) = lenses(3);
    let edges: Vec<_> = skeleton.edges.keys().collect();
    assert_eq!(dissolve_all(&mesh, &mut skeleton, &edges).unwrap(), 3);
    assert_eq!((skeleton.nodes.len(), skeleton.edges.len()), (1, 0));
    skeleton.validate(&mesh).unwrap();

    // On a 3-cycle the first dissolve leaves parallel edges, subdivided back into a 3-cycle; the
    // second selected loop then still dissolves or is skipped, never fails.
    let (mesh, mut skeleton) = bands(&[[3., 0., 0.], [-1.5, 2.6, 0.], [-1.5, -2.6, 0.]]);
    let edges: Vec<_> = skeleton.edges.keys().take(2).collect();
    let dissolved = dissolve_all(&mesh, &mut skeleton, &edges).unwrap();
    assert!((1..=2).contains(&dissolved));
    skeleton.validate(&mesh).unwrap();
    assert_eq!(skeleton.edges.len(), skeleton.nodes.len());
}
