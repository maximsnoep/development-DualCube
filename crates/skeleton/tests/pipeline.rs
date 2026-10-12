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
fn full_run_retains_no_op_optimization_at_unimplemented_loop_drawing() {
    let mesh = tetrahedron();
    let error = pipeline::run(&mesh).unwrap_err();
    assert_eq!(error.failure.stage, Stage::LoopDrawing);
    assert_eq!(
        error
            .failure
            .error
            .downcast_ref::<std::io::Error>()
            .unwrap()
            .kind(),
        std::io::ErrorKind::Unsupported
    );
    let skeleton = error.progress.skeleton.as_ref().unwrap();
    assert!(!skeleton.nodes.is_empty());
    assert!(!skeleton.surface.crossings.is_empty());
    assert!(error.progress.labeling.is_some());
    error
        .progress
        .labeled
        .as_ref()
        .unwrap()
        .validate(&mesh)
        .unwrap();
    error
        .progress
        .realized
        .as_ref()
        .unwrap()
        .validate(&mesh)
        .unwrap();
    let realized = error.progress.realized.as_ref().unwrap();
    let optimized = error.progress.optimized.as_ref().unwrap();
    optimized.validate(&mesh).unwrap();
    assert_eq!(optimized.node_positions, realized.node_positions);
    assert_eq!(optimized.cuboids.connections, realized.cuboids.connections);
    assert_eq!(
        optimized.cuboids.cuboids.len(),
        realized.cuboids.cuboids.len()
    );
    for (id, cube) in &realized.cuboids.cuboids {
        let other = &optimized.cuboids.cuboids[id];
        assert_eq!(other.center, cube.center);
        assert_eq!(other.size, cube.size);
        assert_eq!(other.owner, cube.owner);
    }
    assert!(error.progress.loops.is_none());
}

#[test]
fn explicit_labeling_stop_succeeds_and_continuation_keeps_it() {
    let mesh = tetrahedron();
    let skeleton = skeleton::skeletonize::run(&mesh).unwrap();
    let before = skeleton.surface.edge_order.clone();
    let mut progress = Progress::from_skeleton(skeleton);
    progress.label(&mesh).unwrap();
    assert_eq!(
        progress.finish(&mesh).unwrap_err().stage,
        Stage::LoopDrawing
    );
    assert_eq!(
        progress.skeleton.as_ref().unwrap().surface.edge_order,
        before
    );
    assert!(progress.labeling.is_some());
    progress.labeled.as_ref().unwrap().validate(&mesh).unwrap();
}

#[test]
fn extraction_failure_has_no_completed_downstream_stages() {
    let error = pipeline::run(&Mesh::default()).unwrap_err();
    assert_eq!(error.failure.stage, Stage::Extraction);
    assert!(error.progress.skeleton.is_none());
    assert!(error.progress.labeling.is_none());
    assert!(error.progress.labeled.is_none());
}

#[test]
fn replacing_labels_and_placing_again_discards_stale_downstream_state() {
    let mesh = tetrahedron();
    let mut progress = Progress::from_skeleton(skeleton::skeletonize::run(&mesh).unwrap());
    progress.place(&mesh).unwrap();
    let before = progress.realized.clone().unwrap();
    progress.optimized = Some(before.clone());
    progress.loops = Some(Default::default());
    for (_, label) in &mut progress.labeled.as_mut().unwrap().labels {
        label.sign = if label.sign == Sign::Positive {
            Sign::Negative
        } else {
            Sign::Positive
        };
    }
    progress.place(&mesh).unwrap();
    let after = progress.realized.as_ref().unwrap();
    after.validate(&mesh).unwrap();
    assert!(
        after
            .node_positions
            .iter()
            .any(|(id, p)| *p != before.node_positions[id])
    );
    assert!(progress.optimized.is_none());
    assert!(progress.loops.is_none());
    progress.labeled.as_mut().unwrap().labels.clear();
    assert_eq!(
        progress.place(&mesh).unwrap_err().stage,
        Stage::CubePlacement
    );
    assert!(progress.realized.is_none());
}
