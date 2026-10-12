use dualcube_types::prelude::*;
use skeleton::pipeline::{self, Progress, Stage};

fn tetrahedron() -> Mesh<INPUT> {
    Mesh::from(
        &[vec![0, 2, 1], vec![0, 1, 3], vec![0, 3, 2], vec![1, 2, 3]],
        &[
            Vector3D::zeros(),
            Vector3D::x(),
            Vector3D::y(),
            Vector3D::z(),
        ],
    )
    .unwrap()
    .0
}

#[test]
fn full_run_reaches_the_real_unimplemented_stage_and_keeps_prior_results() {
    let mesh = tetrahedron();
    let error = pipeline::run(&mesh).unwrap_err();
    assert_eq!(error.failure.stage, Stage::CubePlacement);
    assert_eq!(
        error
            .failure
            .error
            .downcast_ref::<std::io::Error>()
            .unwrap()
            .kind(),
        std::io::ErrorKind::Unsupported
    );
    error
        .progress
        .skeleton
        .as_ref()
        .unwrap()
        .validate(&mesh)
        .unwrap();
    error
        .progress
        .labeled
        .as_ref()
        .unwrap()
        .validate(&mesh)
        .unwrap();
    assert!(error.progress.labeling.as_ref().unwrap().failure.is_none());
    assert!(error.progress.realized.is_none());
    assert!(error.progress.optimized.is_none());
    assert!(error.progress.loops.is_none());
}

#[test]
fn explicit_labeling_stop_succeeds_and_continuation_preserves_it() {
    let mesh = tetrahedron();
    let skeleton = skeleton::skeletonize::run(&mesh).unwrap();
    let mut progress = Progress::from_skeleton(skeleton);
    progress.label(&mesh).unwrap();
    let labels = progress.labeled.as_ref().unwrap().labels.clone();
    assert_eq!(
        progress.finish(&mesh).unwrap_err().stage,
        Stage::CubePlacement
    );
    for (edge, before) in labels {
        let after = progress.labeled.as_ref().unwrap().labels[edge];
        assert_eq!(before.direction, after.direction);
        assert_eq!(before.sign, after.sign);
    }
    assert!(progress.labeling.is_some());
}

#[test]
fn extraction_failure_has_no_completed_downstream_stages() {
    let error = pipeline::run(&Mesh::default()).unwrap_err();
    assert_eq!(error.failure.stage, Stage::Extraction);
    assert!(error.progress.skeleton.is_none());
    assert!(error.progress.labeling.is_none());
    assert!(error.progress.labeled.is_none());
}
