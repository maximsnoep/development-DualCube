use super::*;
use std::sync::mpsc;

#[test]
fn dissolving_a_loop_merges_two_patches_and_reruns_through_the_stop() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/mehsh/assets/blub001k.obj");
    let mesh = io::import_solution(&path).unwrap().mesh_ref;
    let SkeletonState::Extracted(extraction) = extract_until(
        &mesh,
        Options::default(),
        Stage::Extraction,
        &ContractionCache::default(),
    )
    .unwrap() else {
        panic!("Missing surgery result");
    };
    let skeleton = extraction.skeleton.clone();
    let edge = skeleton
        .edges
        .keys()
        .next()
        .expect("blub has several patches");
    let job = Job::dissolve_skeleton_loops(
        mesh,
        extraction,
        skeleton.clone(),
        None,
        vec![edge],
        Stage::Labeling,
    );
    let (sender, receiver) = mpsc::channel();
    let Some(JobResult::SkeletonReady {
        state: SkeletonState::Extracted(edited),
        view: None,
        progress,
        failure: None,
    }) = (job.run)(&sender)
    else {
        panic!("Missing dissolve result");
    };
    assert_eq!(edited.skeleton.nodes.len() + 1, skeleton.nodes.len());
    assert!(!edited.skeleton.edges.contains_key(edge));
    assert!(progress.labeled.is_some() && progress.realized.is_none());
    assert!(matches!(
        receiver.try_iter().next(),
        Some(Update::State(SkeletonState::Extracted(_)))
    ));
}
