use super::{bridges, spread};

// Distances between edges placed on a line.
fn on_line(at: &[f64]) -> Vec<Vec<f64>> {
    at.iter()
        .map(|a| at.iter().map(|b| (a - b).abs()).collect())
        .collect()
}

// Whether the edges other than `cut` connect all nodes.
fn connected(nodes: usize, ends: &[[usize; 2]], cut: &[usize]) -> bool {
    let mut component: Vec<_> = (0..nodes).collect();
    fn find(component: &mut [usize], x: usize) -> usize {
        if component[x] != x {
            component[x] = find(component, component[x]);
        }
        component[x]
    }
    for (e, &[a, b]) in ends.iter().enumerate() {
        if !cut.contains(&e) {
            let (a, b) = (find(&mut component, a), find(&mut component, b));
            component[a] = b;
        }
    }
    let root = find(&mut component, 0);
    (0..nodes).all(|x| find(&mut component, x) == root)
}

#[test]
fn a_tree_has_no_cuts() {
    let ends = [[0, 1], [1, 2]];
    assert!(spread(3, &ends, &on_line(&[0., 1.]), &[1., 1.]).is_empty());
}

#[test]
fn genus_one_takes_the_shortest_loop_off_the_bridges() {
    // A triangle with a pendant edge (a bridge, and the shortest).
    let ends = [[0, 1], [1, 2], [2, 0], [2, 3]];
    let cut = spread(4, &ends, &on_line(&[0., 1., 2., 3.]), &[3., 2., 5., 0.1]);
    assert_eq!(cut, vec![1]);
}

#[test]
fn spreads_over_the_handles_and_never_cuts_one_twice() {
    // Two triangles sharing node 0: a cut system takes one edge of each.
    let ends = [[0, 1], [1, 2], [2, 0], [0, 3], [3, 4], [4, 0]];
    // Edges 0 and 1 are farthest apart, but lie on the same triangle; 0 and 5 are the farthest valid pair.
    let distance = on_line(&[0., 10., 1., 4., 5., 7.]);
    let cut = spread(5, &ends, &distance, &[1.; 6]);
    assert_eq!(cut, vec![0, 5]);
    assert!(connected(5, &ends, &cut));
}

#[test]
fn cuts_leave_a_spanning_tree_of_a_multigraph() {
    // Parallel edges and a self-loop: genus 7 + 1 - 4 = 4.
    let ends = [[0, 1], [0, 1], [1, 2], [2, 3], [3, 0], [2, 2], [1, 3]];
    let at: Vec<f64> = (0..ends.len()).map(|e| e as f64).collect();
    let cut = spread(4, &ends, &on_line(&at), &[1.; 7]);
    assert_eq!(cut.len(), 4);
    assert!(connected(4, &ends, &cut));
}

#[test]
fn bridges_of_a_multigraph() {
    // A double edge (no bridge), a pendant edge, and a self-loop.
    let ends = [[0, 1], [0, 1], [1, 2], [2, 2]];
    assert_eq!(
        bridges(3, &ends, &[false; 4]),
        vec![false, false, true, false]
    );
    assert_eq!(
        bridges(3, &ends, &[true, false, false, false]),
        vec![false, true, true, false]
    );
}
