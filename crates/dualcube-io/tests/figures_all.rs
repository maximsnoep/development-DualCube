use dualcube::prelude::*;
use dualcube::skeleton::pipeline::{Progress, Stage};
use dualcube::skeleton::skeletonize::{self, Options};
use io::figure::{Figure, FigureParams, FigureStyle, export_figures};
use io::{SkeletonExtraction, SkeletonSave};
use std::path::Path;
use std::sync::Arc;

#[test]
fn every_view_preset_has_a_figure() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../mehsh/assets/blub001k.obj");
    let mesh = io::import_mesh(&path).unwrap().0;
    let extraction = skeletonize::extract(&mesh, Options::default()).unwrap();
    let mut progress = Progress::from_skeleton(extraction.skeleton.clone());
    progress.run_through(&mesh, Stage::LoopDrawing).unwrap();

    let mut solution = Solution::new(Arc::new(mesh));
    solution.replace_loops(progress.loops.clone().unwrap());
    solution.construct_dual_and_polycube().unwrap();
    solution.place_corners().unwrap();
    solution.place_paths().unwrap();
    solution.resize_polycube(false).unwrap();
    solution.construct_quad(QuadDensity::Fixed(5)).unwrap();

    let skeleton = SkeletonSave {
        extraction: Some(SkeletonExtraction::Extracted(extraction)),
        labeling: progress.labeling,
        labeled: progress.labeled,
        realized: progress.realized,
        optimized: progress.optimized,
        drawing: progress.drawing,
        loops: progress.loops,
        failure: None,
        output: None,
    };
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("figures_all");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let params = FigureParams::default();
    let style = FigureStyle::default();
    for extension in ["svg", "png"] {
        let saved = export_figures(
            &solution,
            Some(&skeleton),
            &dir,
            "blub",
            extension,
            &params,
            &style,
        )
        .unwrap();
        let names = saved
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        let expected = Figure::ALL
            .map(|f| format!("blub_{}.{extension}", f.slug()))
            .to_vec();
        assert_eq!(names, expected);
    }

    // Lines stop flat where they disappear: those on the cubes are clipped to the faces they lie on, those on the mesh
    // are drawn as polylines with flat ends.
    let svg = |figure: Figure| {
        std::fs::read_to_string(dir.join(format!("blub_{}.svg", figure.slug()))).unwrap()
    };
    assert!(svg(Figure::CubesLoops).contains("<clipPath"));
    let loops = svg(Figure::SkeletonLoops);
    assert!(loops.contains(r#"stroke-linecap="butt""#));
    assert!(!loops.contains(r#"stroke-linecap="round""#));

    // Without the skeleton stages, only the figures of the solution remain.
    let saved = export_figures(&solution, None, &dir, "solution", "svg", &params, &style).unwrap();
    assert_eq!(saved.len(), 5);
}
