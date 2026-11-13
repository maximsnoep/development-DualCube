use super::*;

#[test]
fn nonfinite_system_retains_previous_iterate() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../mehsh/assets/tetrahedron.obj");
    let mesh = Mesh::<CONTRACTION>::from_obj(&path).unwrap().0;
    let mut state = ContractionState::new(&mesh);
    contract_once(&mut state).unwrap();
    assert!(state.solver.is_factorized());
    let previous = state.positions.clone();

    state.wl = f64::INFINITY;
    contract_once(&mut state).unwrap();

    assert!(state.unstable);
    assert_eq!(state.positions, previous);
}

#[test]
fn run_writes_contracted_positions_to_mesh() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../mehsh/assets/blub001k.obj");
    let mesh = Mesh::<CONTRACTION>::from_obj(&path).unwrap().0;
    let contracted = run(&mesh, 3).unwrap();
    let mut state = ContractionState::new(&mesh);
    for _ in 0..3 {
        contract_once(&mut state).unwrap();
    }
    for (&v, &position) in state.vert_ids.iter().zip(&state.positions) {
        assert_eq!(contracted.position(v), position);
    }
}

#[test]
fn stages_run_from_the_input_to_the_result() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../mehsh/assets/blub001k.obj");
    let mesh = Mesh::<CONTRACTION>::from_obj(&path).unwrap().0;
    let (contracted, stages) = run_with_stages(&mesh, 3).unwrap();
    assert_eq!(stages.len(), 4);
    let vertices = mesh.vert_ids();
    for (i, &v) in vertices.iter().enumerate() {
        assert_eq!(stages[0][i], mesh.position(v));
        assert_eq!(stages[3][i], contracted.position(v));
    }
    assert_ne!(stages[1], stages[0]);
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
