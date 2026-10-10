//! Remeshing (see `Mesh::remesh`): a valid closed mesh of the same topology, close to the input, without slivers.

use mehsh::prelude::*;

define_tag!(TEST);

fn remeshed(path: &str) -> (Mesh<TEST>, Mesh<TEST>, RemeshReport) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(path);
    let mesh = Mesh::<TEST>::from_obj(&path).unwrap().0;
    let timer = std::time::Instant::now();
    let (result, report) = mesh.remesh(&RemeshParams::default()).unwrap();
    println!("{path:?}: {report:?} in {:?}", timer.elapsed());
    (mesh, result, report)
}

#[test]
fn remeshing_keeps_the_surface_and_removes_slivers() {
    let (mesh, result, report) = remeshed("assets/blub001k.obj");
    let euler =
        |m: &Mesh<TEST>| m.nr_verts() as i64 - m.nr_edges() as i64 / 2 + m.nr_faces() as i64;
    assert_eq!(euler(&mesh), euler(&result));
    assert!(report.max_distance < 0.02, "{report:?}");
    assert!(
        report.min_angle_after > report.min_angle_before.min(15.),
        "{report:?}"
    );
}

/// Remesh the mesh at `REMESH_PATH` (an obj file) and print the report.
#[test]
#[ignore = "needs REMESH_PATH"]
fn remesh_file() {
    let path = std::env::var("REMESH_PATH").unwrap();
    let mesh = Mesh::<TEST>::from_obj(std::path::Path::new(&path))
        .unwrap()
        .0;
    let timer = std::time::Instant::now();
    let (_, report) = mesh.remesh(&RemeshParams::default()).unwrap();
    println!("{path}: {report:?} in {:?}", timer.elapsed());
}
