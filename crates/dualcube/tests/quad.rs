//! The quad mesh (and polycube map) of an initialized solution: well-formed, with a scale-invariant density.

use dualcube::prelude::*;
use std::sync::Arc;

mod common;
use common::blub;

fn initialized(mesh: Arc<Mesh<INPUT>>) -> Solution {
    let mut solution = Solution::new(mesh);
    solution.initialize();
    solution
}

fn quad(solution: &Solution, omega: usize) -> Quad {
    let mut solution = solution.clone();
    solution.construct_quad(QuadDensity::Fixed(omega)).unwrap();
    solution.quad.expect("quad construction failed")
}

#[test]
fn quad_mesh_is_well_formed() {
    let quad = quad(&initialized(blub()), 3);
    let mesh = &quad.quad_mesh;
    // A closed quad mesh of a genus-0 surface.
    assert!(mesh.nr_faces() > 0);
    assert_eq!(
        mesh.nr_verts() as i64 - mesh.nr_edges() as i64 / 2 + mesh.nr_faces() as i64,
        2
    );
    for face in mesh.face_ids() {
        assert_eq!(mesh.vertices(face).count(), 4);
    }
    for vert in mesh.vert_ids() {
        let position = mesh.position(vert);
        assert!(position.iter().all(|c| c.is_finite()), "{position:?}");
    }
    // The frozen vertices are those on the edges of the polycube: at least its 8 corners.
    assert!(quad.frozen.len() >= 8);
    for vert in quad.triangle_mesh_polycube.vert_ids() {
        assert!(
            quad.triangle_mesh_polycube
                .position(vert)
                .iter()
                .all(|c| c.is_finite())
        );
    }
}

#[test]
fn quad_density_does_not_depend_on_the_scale() {
    let solution = initialized(blub());
    let mut scaled_mesh = (*solution.mesh_ref).clone();
    for vert in scaled_mesh.vert_ids() {
        let position = scaled_mesh.position(vert);
        scaled_mesh.set_position(vert, position * 100.);
    }
    // The same loops on the scaled mesh (with the same element ids).
    let mut scaled = Solution::new(Arc::new(scaled_mesh));
    for lewp in solution.loops.values() {
        scaled.add_loop(Loop::new(lewp.edges.clone(), lewp.direction));
    }
    scaled.reconstruct_solution(true).unwrap();
    let small = quad(&solution, 4).quad_mesh.nr_faces();
    let large = quad(&scaled, 4).quad_mesh.nr_faces();
    assert_eq!(small, large);
}
