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
fn full_run_retains_no_op_optimization_and_valid_loops() {
    let mesh = tetrahedron();
    let run = pipeline::run(&mesh).unwrap();
    assert!(!run.skeleton.nodes.is_empty());
    run.labeled.validate(&mesh).unwrap();
    run.realized.validate(&mesh).unwrap();
    run.optimized.validate(&mesh).unwrap();
    assert_eq!(run.optimized.node_positions, run.realized.node_positions);
    assert_eq!(
        run.optimized.cuboids.connections,
        run.realized.cuboids.connections
    );
    assert_eq!(
        run.optimized.cuboids.cuboids.len(),
        run.realized.cuboids.cuboids.len()
    );
    for (id, cuboid) in &run.realized.cuboids.cuboids {
        let other = &run.optimized.cuboids.cuboids[id];
        assert_eq!(other.center, cuboid.center);
        assert_eq!(other.size, cuboid.size);
        assert_eq!(other.owner, cuboid.owner);
    }
    dualcube_dual::Dual::from(std::sync::Arc::new(mesh), &run.loops).unwrap();
}

#[test]
fn explicit_labeling_stop_succeeds_and_continuation_keeps_it() {
    let mesh = tetrahedron();
    let skeleton = skeleton::skeletonize::run(&mesh).unwrap();
    let before = skeleton.surface.edge_order.clone();
    let mut progress = Progress::from_skeleton(skeleton);
    progress.label(&mesh).unwrap();
    progress.finish(&mesh).unwrap();
    assert!(progress.loops.is_some());
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
    progress.finish(&mesh).unwrap();
    let before = progress.realized.clone().unwrap();
    assert!(progress.drawing.is_some());
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
    assert!(progress.drawing.is_none());
    progress.labeled.as_mut().unwrap().labels.clear();
    assert_eq!(
        progress.place(&mesh).unwrap_err().stage,
        Stage::CubePlacement
    );
    assert!(progress.realized.is_none());
}
