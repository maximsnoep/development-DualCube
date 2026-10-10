//! The input mesh refined along the loops (see `Dual::refined_mesh`).

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

// A loop crossing an edge very close to one of its vertices.
#[test]
fn loop_very_close_to_a_vertex() {
    let solution = initialized();
    let mesh = &solution.mesh_ref;
    let faces = Dual::from(mesh.clone(), &solution.loops)
        .expect("valid dual")
        .refined_mesh()
        .expect("refined mesh")
        .mesh
        .nr_faces();

    let mut tested = 0;
    for (loop_id, lewp) in &solution.loops {
        // The crossings with edges that no other loop crosses.
        let exits = (0..lewp.edges.len())
            .filter(|&i| mesh.twin(lewp.edges[i]) == lewp.edges[(i + 1) % lewp.edges.len()])
            .filter(|&i| solution.loops_on_edge(lewp.edges[i]).len() == 1)
            .take(10)
            .collect::<Vec<_>>();
        for i in exits {
            for t in [1e-8, 1. - 1e-8] {
                let mut loops = solution.loops.clone();
                let offsets = &mut loops[loop_id].offsets;
                let n = offsets.len();
                offsets[i] = t;
                offsets[(i + 1) % n] = 1. - t;
                let Ok(dual) = Dual::from(mesh.clone(), &loops) else {
                    continue;
                };
                let refined = dual
                    .refined_mesh()
                    .unwrap_or_else(|| panic!("refined mesh (crossing {i} of {loop_id:?} at {t})"));
                // The same cells (the loops cross the same faces in the same way).
                assert_eq!(refined.mesh.nr_faces(), faces);
                tested += 1;
            }
        }
    }
    assert!(tested > 0);
}
