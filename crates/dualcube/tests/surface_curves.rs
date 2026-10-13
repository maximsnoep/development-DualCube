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
    let run = skeleton::pipeline::run(&mesh).unwrap();
    let mut solution = Solution::new(Arc::new(mesh));
    solution.replace_loops(run.loops);
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
    assert!(matches!(
        solution.evolve(&EvolutionParams::default()),
        Err(SolutionError::SurfaceCurveEvolution)
    ));
}
