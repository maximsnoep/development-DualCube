//! Multiple loops may pass through the same triangles (and cross the same edges).

mod common;

use common::blub;
use dualcube::prelude::*;

fn initialized() -> Solution {
    // Initialization is randomized (and blub is a very coarse mesh), retry a few times.
    for _ in 0..20 {
        let mut solution = Solution::new(blub());
        solution.initialize();
        if solution.loops.len() == 3 {
            return solution;
        }
    }
    panic!("could not initialize a solution");
}

// The loops crossing an edge are ordered along it by their offsets (strictly increasing, inside the edge),
// in reverse order along its twin, and the offsets of a loop on both half-edges of an edge agree.
fn assert_consistent_offsets(solution: &Solution) {
    for (loop_id, lewp) in &solution.loops {
        assert_eq!(lewp.offsets.len(), lewp.edges.len());
        for (&edge, &offset) in lewp.edges.iter().zip(&lewp.offsets) {
            let on_edge = solution.loops_on_edge(edge);
            assert!(on_edge.contains(&loop_id));
            let offsets = on_edge
                .iter()
                .map(|&l| solution.loops[l].offset(edge).unwrap())
                .collect::<Vec<_>>();
            assert!(offsets.windows(2).all(|w| w[0] < w[1]));
            assert!(offsets[0] > 0. && *offsets.last().unwrap() < 1.);
            let twin = solution.mesh_ref.twin(edge);
            assert_eq!(
                solution.loops_on_edge(twin),
                on_edge.iter().rev().copied().collect::<Vec<_>>()
            );
            assert!((solution.loops[loop_id].offset(twin).unwrap() - (1. - offset)).abs() < 1e-12);
        }
    }
}

#[test]
fn parallel_copies_of_loops_share_triangles() {
    let mut solution = initialized();

    // Insert copies of every loop: the copies pass through exactly the same triangles as the originals.
    for _ in 0..2 {
        let loops = solution.loops.values().cloned().collect::<Vec<_>>();
        for lewp in loops {
            solution.add_loop(Loop::new(lewp.edges, lewp.direction));
        }
    }
    assert_eq!(solution.loops.len(), 12);
    assert_consistent_offsets(&solution);

    let max_on_edge = solution
        .mesh_ref
        .edge_ids()
        .into_iter()
        .map(|e| solution.loops_on_edge(e).len())
        .max()
        .unwrap();
    assert!(max_on_edge >= 4);

    // Copies run in parallel to their originals: loops of the same direction do not intersect.
    let dual = Dual::from(solution.mesh_ref.clone(), &solution.loops).expect("valid dual");
    for intersection in dual.loop_structure.vert_ids() {
        let [a, b] = dual.intersection(intersection).loops;
        assert_ne!(solution.loops[a].direction, solution.loops[b].direction);
    }
    // Every loop region between two copies contains no mesh vertices, but does contain points inside faces.
    let regions = dual.loop_structure.face_ids();
    assert!(regions.iter().any(|&r| dual.region_to_verts(r).is_empty()));
    assert!(
        regions
            .iter()
            .all(|&r| !dual.region_to_verts(r).is_empty() || !dual.region_to_points(r).is_empty())
    );

    let polycube = Polycube::from_dual(&dual);
    validate_polycube(&polycube).unwrap();

    // Removing loops keeps the remaining loops ordered.
    let removed = solution.loops.keys().take(4).collect::<Vec<_>>();
    for loop_id in removed {
        solution.del_loop(loop_id);
    }
    assert_consistent_offsets(&solution);
}

#[test]
fn sampled_loops_are_never_blocked() {
    let mut solution = initialized();

    let m = OrderedFloat;
    let s = |(_, s): (&[EdgeID], f64)| s;
    // Sample many loops of the same direction; all of them can be inserted without crossing each other.
    for _ in 0..6 {
        let edges = solution
            .sample_loops(1, Direction::X, m, s)
            .into_iter()
            .next()
            .expect("a loop can always be sampled");
        solution.add_loop(Loop::new(edges, Direction::X));
    }
    assert_consistent_offsets(&solution);
    assert_eq!(same_direction_crossings(&solution), 0);
}

// Number of pairs of loops of the same direction that cross inside a face (their positions on the boundary interleave).
fn same_direction_crossings(solution: &Solution) -> usize {
    let mut count = 0;
    for face in solution.mesh_ref.face_ids() {
        let mut ends: HashMap<LoopID, Vec<(usize, usize)>> = HashMap::new();
        for (side, edge) in solution.mesh_ref.edges(face).enumerate() {
            for (rank, loop_id) in solution.loops_on_edge(edge).into_iter().enumerate() {
                ends.entry(loop_id).or_default().push((side, rank));
            }
        }
        let chords = ends.into_iter().collect::<Vec<_>>();
        for (i, (a, ends_a)) in chords.iter().enumerate() {
            for (b, ends_b) in &chords[i + 1..] {
                if solution.loops[*a].direction != solution.loops[*b].direction {
                    continue;
                }
                let (p, q) = (ends_a[0].min(ends_a[1]), ends_a[0].max(ends_a[1]));
                let inside = |x: (usize, usize)| p < x && x < q;
                if inside(ends_b[0]) != inside(ends_b[1]) {
                    count += 1;
                }
            }
        }
    }
    count
}
