use super::*;

#[test]
fn stop_stages_retain_only_computed_states() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/mehsh/assets/tetrahedron.obj");
    let solution = io::import_solution(&path).unwrap();
    let mesh = &solution.mesh_ref;
    assert!(matches!(
        extract_until(mesh, Options::default(), Stage::Deflation).unwrap(),
        SkeletonState::Deflated(_)
    ));
    let SkeletonState::Extracted(raw) =
        extract_until(mesh, Options::default(), Stage::Extraction).unwrap()
    else {
        panic!("Missing surgery result");
    };
    for (id, node) in &raw.raw.nodes {
        assert_eq!(node.position, raw.skeleton.nodes[id].position);
    }
    let SkeletonState::Extracted(refined) = extract_until(
        mesh,
        Options {
            refine_embedding: true,
            ..Options::default()
        },
        Stage::Extraction,
    )
    .unwrap() else {
        panic!("Missing refinement result");
    };
    let centers = dualcube::skeleton::geometry::SurfaceGeometry::build(mesh, &raw.raw)
        .unwrap()
        .centroids()
        .unwrap();
    for (id, node) in &refined.skeleton.nodes {
        assert_eq!(node.position, centers[id]);
        assert_eq!(refined.raw.nodes[id].position, raw.raw.nodes[id].position);
    }
}

#[test]
fn deflation_publishes_its_stages_and_starts_a_new_extraction() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/mehsh/assets/tetrahedron.obj");
    let mesh = io::import_solution(&path).unwrap().mesh_ref;
    let updates = std::sync::Mutex::new(vec![]);
    extract_until_with_updates(&mesh, Options::default(), Stage::Extraction, &|u| {
        updates.lock().unwrap().push(u);
    })
    .unwrap();
    let mut resource = SkeletonResource {
        contraction_stage: Some(1),
        ..Default::default()
    };
    for update in updates.into_inner().unwrap() {
        update.apply(&mut resource);
    }
    let stages = resource.contraction_stages.unwrap();
    assert!(stages.len() >= 2);
    assert_eq!(stages[0].len(), mesh.nr_verts());
    assert_eq!(resource.contraction_stage, None);
    assert!(matches!(resource.state, Some(SkeletonState::Extracted(_))));
}

mod dissolve;
mod live;
