use dualcube_types::prelude::*;
use skeleton::pipeline::{Progress, Stage};

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
fn full_run_retains_placement_and_valid_optimized_loops() {
    let mesh = tetrahedron();
    let mut run = Progress::from_skeleton(skeleton::skeletonize::run(&mesh).unwrap());
    run.run_through(&mesh, Stage::LoopDrawing).unwrap();
    assert!(!run.skeleton.unwrap().nodes.is_empty());
    run.labeled.unwrap().validate(&mesh).unwrap();
    let (realized, optimized) = (run.realized.unwrap(), run.optimized.unwrap());
    realized.validate(&mesh).unwrap();
    optimized.validate(&mesh).unwrap();
    assert!(optimized.labeled.skeleton.nodes.len() <= realized.labeled.skeleton.nodes.len());
    dualcube_dual::Dual::from(std::sync::Arc::new(mesh), &run.loops.unwrap()).unwrap();
}

#[test]
fn explicit_labeling_stop_succeeds_and_continuation_keeps_it() {
    let mesh = tetrahedron();
    let skeleton = skeleton::skeletonize::run(&mesh).unwrap();
    let before = skeleton.surface.edge_order.clone();
    let mut progress = Progress::from_skeleton(skeleton);
    progress.rerun(&mesh, Stage::Labeling).unwrap();
    progress.run_through(&mesh, Stage::LoopDrawing).unwrap();
    assert!(progress.loops.is_some());
    assert_eq!(
        progress.skeleton.as_ref().unwrap().surface.edge_order,
        before
    );
    assert!(progress.labeling.is_some());
    progress.labeled.as_ref().unwrap().validate(&mesh).unwrap();
}

#[test]
fn replacing_labels_and_placing_again_discards_stale_downstream_state() {
    let mesh = tetrahedron();
    let mut progress = Progress::from_skeleton(skeleton::skeletonize::run(&mesh).unwrap());
    progress.run_through(&mesh, Stage::LoopDrawing).unwrap();
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
    progress.rerun(&mesh, Stage::CubePlacement).unwrap();
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
        progress
            .rerun(&mesh, Stage::CubePlacement)
            .unwrap_err()
            .stage,
        Stage::CubePlacement
    );
    assert!(progress.realized.is_none());
}
