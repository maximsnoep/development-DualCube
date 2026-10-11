use super::*;

#[test]
fn stop_stages_retain_only_computed_states() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/mehsh/assets/tetrahedron.obj");
    let solution = io::import_solution(&path).unwrap();
    let mesh = &solution.mesh_ref;
    assert!(matches!(
        extract_until(
            mesh,
            Options::default(),
            SkeletonStage::Deflation,
            &ContractionCache::default()
        )
        .unwrap(),
        SkeletonState::Deflated(_)
    ));
    let SkeletonState::Extracted(raw) = extract_until(
        mesh,
        Options::default(),
        SkeletonStage::Surgery,
        &ContractionCache::default(),
    )
    .unwrap() else {
        panic!("Missing surgery result");
    };
    for (id, node) in &raw.raw.nodes {
        assert_eq!(node.position, raw.skeleton.nodes[id].position);
    }
    let SkeletonState::Extracted(refined) = extract_until(
        mesh,
        Options::default(),
        SkeletonStage::Refinement,
        &ContractionCache::default(),
    )
    .unwrap() else {
        panic!("Missing refined result");
    };
    refined.skeleton.validate(mesh).unwrap();
}

#[test]
fn cache_reuses_contraction_for_downstream_settings_and_invalidates_on_inputs() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/mehsh/assets/tetrahedron.obj");
    let mesh = io::import_solution(&path).unwrap().mesh_ref;
    let cache = ContractionCache::default();
    let first = cache.get_or_compute(mesh.clone(), 50).unwrap();
    let mut options = Options::default();
    options.surgery.shape_weight = 2.;
    options.surgery.sampling_weight = 0.2;
    options.surgery.handle_bonus = 0.;
    for stop in [SkeletonStage::Surgery, SkeletonStage::Refinement] {
        extract_until(&mesh, options, stop, &cache).unwrap();
        assert!(Arc::ptr_eq(
            &first,
            &cache.get_or_compute(mesh.clone(), 50).unwrap()
        ));
    }
    options.surgery.shape_weight = f64::NAN;
    assert!(extract_until(&mesh, options, SkeletonStage::Surgery, &cache).is_err());
    assert!(Arc::ptr_eq(
        &first,
        &cache.get_or_compute(mesh.clone(), 50).unwrap()
    ));

    let fewer = cache.get_or_compute(mesh.clone(), 1).unwrap();
    assert!(!Arc::ptr_eq(&first, &fewer));
    let mut other_input = mesh.clone();
    let v = other_input.vert_ids()[0];
    let position = other_input.position(v) + Vector3D::new(0.01, 0., 0.);
    Arc::make_mut(&mut other_input).set_position(v, position);
    let other = cache.get_or_compute(other_input, 1).unwrap();
    assert!(!Arc::ptr_eq(&fewer, &other));
    let fresh_cache = ContractionCache::default();
    assert!(!Arc::ptr_eq(
        &other,
        &fresh_cache.get_or_compute(mesh, 1).unwrap()
    ));
}
