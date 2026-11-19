use super::*;
use crate::resources::FigureFormat;
use bevy::prelude::Vec3;
use std::sync::mpsc;

#[test]
fn exported_figures_come_with_a_save_and_a_view_text_that_restore_them() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model = root.join("crates/mehsh/assets/tetrahedron.obj");
    let solution = io::import_solution(&model).unwrap();
    let output = solution.clone();
    let dir = root.join("target/ui-smoke/export_figures");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let view = ViewSettings {
        eye: Vec3::new(3., 2., 1.5),
        target: Vec3::new(0.25, 0., 0.),
        up: Vec3::Z,
        loop_width: 0.01,
        format: FigureFormat::Svg,
        label: true,
        smooth_paths: false,
    };
    let skeleton = io::SkeletonSave {
        failure: Some((
            dualcube::skeleton::pipeline::Stage::Labeling,
            "message".to_owned(),
        )),
        ..Default::default()
    };
    let job = Job::export_figures(
        solution,
        Configuration::default(),
        dir.clone(),
        "tet".to_owned(),
        view,
        Some(skeleton),
        Some(output),
        Some(model.clone()),
    );
    let (sender, _) = mpsc::channel();
    assert!((job.run)(&sender).is_none());

    // Everything goes in one new folder, `tet_<timestamp>`.
    let folders = std::fs::read_dir(&dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect::<Vec<_>>();
    let [dir] = &folders[..] else {
        panic!("expected one folder, found {folders:?}");
    };
    let folder = dir.file_name().unwrap().to_string_lossy();
    assert!(folder.starts_with("tet_20") && folder.len() == "tet_YYYY-MM-DD_HHMMSS".len());
    assert!(dir.join("tet_mesh.svg").exists());
    // The output has the same input mesh, so no figure of its own without loops.
    assert!(!dir.join("tet_output_mesh.svg").exists());
    let text = std::fs::read_to_string(dir.join("tet_view.txt")).unwrap();
    assert!(text.contains("figures: tet_mesh.svg"), "{text}");
    assert!(
        text.contains(&format!("source: {}", model.display())),
        "{text}"
    );
    let other = ViewSettings {
        eye: Vec3::ONE,
        target: Vec3::ZERO,
        up: Vec3::Y,
        loop_width: 0.006,
        format: FigureFormat::Pdf,
        label: false,
        smooth_paths: true,
    };
    assert_eq!(ViewSettings::parse(&text, other), Ok(view));

    let import = Job::import(dir.join("tet.dc"), Configuration::default());
    let Some(JobResult::Imported {
        skeleton, source, ..
    }) = (import.run)(&sender)
    else {
        panic!("the save does not import");
    };
    assert_eq!(source, model);
    let failure = skeleton.and_then(|s| s.failure).map(|(_, message)| message);
    assert_eq!(failure.as_deref(), Some("message"));
}
