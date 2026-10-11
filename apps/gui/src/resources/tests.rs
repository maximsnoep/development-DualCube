use super::*;

#[test]
fn genus_handles_empty_sphere_and_torus() {
    assert_eq!(genus(&mehsh::prelude::Mesh::default()), None);
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/mehsh/assets/tetrahedron.obj");
    let sphere = io::import_solution(&path).unwrap();
    assert_eq!(genus(&sphere.mesh_ref), Some(0));

    let (n, m) = (8, 6);
    let mut positions = Vec::new();
    let mut faces = Vec::new();
    for i in 0..n {
        for j in 0..m {
            let u = std::f64::consts::TAU * i as f64 / n as f64;
            let v = std::f64::consts::TAU * j as f64 / m as f64;
            positions.push(Vector3D::new(
                (3. + v.cos()) * u.cos(),
                (3. + v.cos()) * u.sin(),
                v.sin(),
            ));
            let (a, b, c, d) = (
                i * m + j,
                ((i + 1) % n) * m + j,
                ((i + 1) % n) * m + (j + 1) % m,
                i * m + (j + 1) % m,
            );
            faces.extend([vec![a, b, c], vec![a, c, d]]);
        }
    }
    let torus = mehsh::prelude::Mesh::from(&faces, &positions).unwrap().0;
    assert_eq!(genus(&torus), Some(1));
}
