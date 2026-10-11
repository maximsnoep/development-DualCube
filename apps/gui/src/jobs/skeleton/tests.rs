use super::*;

#[test]
fn stop_stages_retain_only_computed_states() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/mehsh/assets/tetrahedron.obj");
    let solution = io::import_solution(&path).unwrap();
    let mesh = &solution.mesh_ref;
    assert!(matches!(
        extract_until(mesh, Options::default(), SkeletonStage::Deflation).unwrap(),
        SkeletonState::Deflated(_)
    ));
    let SkeletonState::Extracted(raw) =
        extract_until(mesh, Options::default(), SkeletonStage::Surgery).unwrap()
    else {
        panic!("Missing surgery result");
    };
    for (id, node) in &raw.raw.nodes {
        assert_eq!(node.position, raw.skeleton.nodes[id].position);
    }
    let SkeletonState::Extracted(refined) =
        extract_until(mesh, Options::default(), SkeletonStage::Refinement).unwrap()
    else {
        panic!("Missing refined result");
    };
    refined.skeleton.validate(mesh).unwrap();
}
