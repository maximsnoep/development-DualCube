use super::*;

#[test]
fn nonfinite_system_retains_previous_iterate() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../mehsh/assets/tetrahedron.obj");
    let mut mesh = Mesh::<CONTRACTION>::from_obj(&path).unwrap().0;
    let mut state = ContractionState::new(&mesh);
    contract_once(&mut mesh, &mut state).unwrap();
    assert!(state.qr.is_some());
    let previous: Vec<_> = state.vert_ids.iter().map(|&v| mesh.position(v)).collect();

    state.wl = f64::INFINITY;
    contract_once(&mut mesh, &mut state).unwrap();

    assert!(state.unstable);
    for (&v, position) in state.vert_ids.iter().zip(previous) {
        assert_eq!(mesh.position(v), position);
    }
}

#[test]
fn nonfinite_laplacian_stops_instead_of_changing_sparsity() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../mehsh/assets/tetrahedron.obj");
    let mut mesh = Mesh::<CONTRACTION>::from_obj(&path).unwrap().0;
    let state = ContractionState::new(&mesh);
    for &v in &state.vert_ids {
        mesh.set_position(v, mesh.position(v) * 1e200);
    }
    assert!(get_laplacian_triplets(&mesh, &state).is_none());
}

#[test]
#[ignore = "manual timing; run with --release -- --ignored --nocapture"]
fn contraction_benchmark() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../mehsh/assets/blub001k.obj");
    let mesh = Mesh::<CONTRACTION>::from_obj(&path).unwrap().0;
    run(&mesh, 50).unwrap();
    let mut elapsed = Vec::new();
    for _ in 0..7 {
        let start = std::time::Instant::now();
        run(&mesh, 50).unwrap();
        elapsed.push(start.elapsed());
    }
    elapsed.sort();
    println!("blub001k contraction median: {:?}", elapsed[3]);
}
