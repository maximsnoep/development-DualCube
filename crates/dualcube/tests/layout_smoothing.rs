//! Layout embedding and smoothing (path straightening by intrinsic edge flips).

mod common;

use common::blub;
use dualcube::prelude::*;

fn reconstructed() -> Solution {
    // Initialization is randomized (and blub is a very coarse mesh), retry a few times.
    for _ in 0..20 {
        let mut solution = Solution::new(blub());
        solution.initialize();
        if solution.loops.len() == 3
            && solution
                .reconstruct_solution_with(false, LayoutParams::fast())
                .is_ok()
            && solution.layout.is_some()
        {
            return solution;
        }
    }
    panic!("could not initialize a solution");
}

#[test]
fn straightening_shortens_paths_and_keeps_patches() {
    let mut successes = 0;
    for _ in 0..3 {
        let solution = reconstructed();
        let mut layout = solution.layout.clone().unwrap();
        let length_before = layout.total_path_length();
        let Ok(stats) = layout.straighten_paths() else {
            continue;
        };
        successes += 1;

        // Never longer, and (on this coarse mesh) always noticeably shorter.
        assert!(stats.length_after <= stats.length_before + 1e-9);
        assert!(layout.total_path_length() < length_before);

        // Every polycube face still has a patch, and every path connects its corners.
        let polycube = &layout.polycube_ref.structure;
        assert_eq!(layout.face_to_patch.len(), polycube.face_ids().len());
        for edge in polycube.edge_ids() {
            let path = &layout.edge_to_path[&edge];
            let [u, v] = polycube.vertices(edge).collect_array::<2>().unwrap();
            assert_eq!(path[0], *layout.vert_to_corner.get_by_left(&u).unwrap());
            assert_eq!(
                *path.last().unwrap(),
                *layout.vert_to_corner.get_by_left(&v).unwrap()
            );
            for w in path.windows(2) {
                assert!(
                    layout
                        .granulated_mesh
                        .edge_between_verts(w[0], w[1])
                        .is_some()
                );
            }
        }
    }
    assert!(successes >= 2, "straightening failed too often");
}

#[test]
fn best_of_several_embeddings_is_valid() {
    let solution = reconstructed();
    let dual = solution.dual.as_ref().unwrap();
    let polycube = Polycube::from_dual(dual);
    let layout = Layout::embed_best(dual, &polycube, 6).unwrap();
    assert_eq!(
        layout.face_to_patch.len(),
        layout.polycube_ref.structure.face_ids().len()
    );
    assert!(layout.total_path_length() > 0.);
}
