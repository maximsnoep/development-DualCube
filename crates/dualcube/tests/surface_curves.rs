use dualcube::prelude::*;
use std::sync::Arc;

#[test]
fn explicit_skeleton_curves_reconstruct_without_zero_length_edges() {
    let mesh = Mesh::from(
        &[vec![0, 2, 1], vec![0, 1, 3], vec![0, 3, 2], vec![1, 2, 3]],
        &[
            Vector3D::zeros(),
            Vector3D::x(),
            Vector3D::y(),
            Vector3D::z(),
        ],
    )
    .unwrap()
    .0;
    let mut run =
        skeleton::pipeline::Progress::from_skeleton(skeleton::skeletonize::run(&mesh).unwrap());
    run.run_through(&mesh, skeleton::pipeline::Stage::LoopDrawing)
        .unwrap();
    let drawing = run.drawing.unwrap();
    let mut solution = Solution::new(Arc::new(mesh));
    solution.replace_loops(drawing.loops);
    // The drawing's dual structure stays current for the installed curves.
    solution.dual = Ok(drawing.dual.unwrap());
    assert!(solution.dual_is_current());
    solution.construct_dual_and_polycube().unwrap();
    solution.place_corners().unwrap();
    let layout = solution.layout.as_ref().unwrap();
    for &corner in layout.vert_to_corner.right_values() {
        assert!(
            layout
                .granulated_mesh
                .edges(corner)
                .all(|e| layout.granulated_mesh.size(e) > 0.)
        );
    }
    solution.place_paths().unwrap();
    assert!(solution.layout.as_ref().unwrap().is_complete());
    // Explicit curves evolve (by adding and removing loops).
    let params = EvolutionParams {
        population: 4,
        offspring: 4,
        max_generations: 2,
        ..EvolutionParams::default()
    };
    assert!(solution.evolve(&params).is_ok());
}
