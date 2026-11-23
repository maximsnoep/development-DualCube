use super::*;

fn parse(args: &[&str]) -> Result<Launch, String> {
    Launch::parse(args.iter().map(OsString::from)).map(Option::unwrap)
}

#[test]
fn startup_defaults_and_options() {
    assert!(parse(&[]).unwrap().path.is_none());
    assert!(
        !parse(&[])
            .unwrap()
            .configuration()
            .skeleton_options
            .refine_embedding
    );
    let launch = parse(&[
        "--load",
        "models/a model.obj",
        "--auto-start=skeleton",
        "--view=combined",
    ])
    .unwrap();
    assert_eq!(launch.path.unwrap(), PathBuf::from("models/a model.obj"));
    assert_eq!(launch.stop, Some(Stage::LoopDrawing));
    assert!(!launch.options.refine_embedding);
    assert_eq!(launch.view, Some(ModelViewKind::Combined));
    assert!(!launch.quit);
    let launch = parse(&[
        "--load=a.obj",
        "--auto-start",
        "skeleton",
        "--no-refinement",
        "--shape-weight=2",
        "--sampling-weight=0",
        "--handle-bonus=12",
        "--iterations=7",
        "--contraction-step=2",
    ])
    .unwrap();
    assert_eq!(launch.stop, Some(Stage::LoopDrawing));
    assert_eq!(launch.options.max_iterations, 7);
    assert_eq!(launch.options.contraction_step, 2.);
    assert_eq!(launch.options.surgery.shape_weight, 2.);
    assert_eq!(launch.options.surgery.sampling_weight, 0.);
    assert_eq!(launch.options.surgery.handle_bonus, 12.);
}

#[test]
fn stage_controls_and_invalid_combinations() {
    for (name, stage) in [
        ("input", None),
        ("deflation", Some(Stage::Deflation)),
        ("surgery", Some(Stage::Extraction)),
        ("refinement", Some(Stage::Extraction)),
        ("labeling", Some(Stage::Labeling)),
        ("placement", Some(Stage::CubePlacement)),
        ("optimization", Some(Stage::Optimization)),
    ] {
        let launch = parse(&["--load=a.obj", "--stop-after", name]).unwrap();
        assert_eq!(launch.stop, stage);
        assert_eq!(launch.options.refine_embedding, name == "refinement");
        assert!(!launch.quit);
        let launch = parse(&["--load=a.obj", "--quit-after", name]).unwrap();
        assert_eq!(launch.stop, stage);
        assert!(launch.quit);
    }
    for args in [
        vec!["--auto-start=skeleton"],
        vec!["--load"],
        vec!["--load", "--auto-start=skeleton"],
        vec!["--load=a.obj", "--stop-after=unknown"],
        vec!["--load=a.obj", "--stop-after=surgery", "--view=labeled"],
        vec!["--load=a.obj", "--stop-after=deflation", "--view=combined"],
        vec![
            "--load=a.obj",
            "--stop-after=surgery",
            "--quit-after=refinement",
        ],
        vec!["--load=a.obj", "--stop-after=refinement", "--no-refinement"],
        vec!["--shape-weight=NaN"],
        vec!["--sampling-weight=-1"],
        vec!["--handle-bonus=inf"],
        vec!["--iterations=-1"],
        vec!["--unknown"],
    ] {
        assert!(parse(&args).is_err(), "{args:?}");
    }
    assert!(Launch::parse([OsString::from("--help")]).unwrap().is_none());
}

fn run_startup(launch: Launch) -> (App, AppExit) {
    use crate::render::store::{RenderObjectSettingStore, RenderObjectStore};
    use crate::resources::{InputResource, SkeletonResource, SolutionResource};
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .insert_resource(launch.configuration())
        .insert_resource(launch)
        .init_resource::<InputResource>()
        .init_resource::<SolutionResource>()
        .init_resource::<SkeletonResource>()
        .init_resource::<RenderObjectStore>()
        .init_resource::<RenderObjectSettingStore>()
        .init_resource::<ModelView>()
        .add_plugins(crate::jobs::JobPlugin)
        .add_systems(Startup, start)
        .add_systems(
            Update,
            (
                crate::render::objects::skeleton::sync,
                crate::render::store::update_render_settings,
                crate::ui::apply_pending_view,
            )
                .chain()
                .after(crate::jobs::poll_jobs),
        );
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        app.update();
        if let Some(exit) = app.should_exit() {
            return (app, exit);
        }
        assert!(
            std::time::Instant::now() < deadline,
            "Startup did not finish"
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

#[test]
fn startup_jobs_load_extract_select_view_and_report_failure() {
    use crate::render::Objects;
    use crate::render::store::RenderObjectSettingStore;
    use crate::resources::{SkeletonResource, SolutionResource};
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../crates/mehsh/assets/tetrahedron.obj");
    for (stage, refine) in [
        (Stage::Deflation, false),
        (Stage::Extraction, false),
        (Stage::Extraction, true),
        (Stage::Labeling, false),
        (Stage::CubePlacement, false),
    ] {
        let (app, exit) = run_startup(Launch {
            path: Some(path.clone()),
            stop: Some(stage),
            view: (stage != Stage::Deflation).then_some(ModelViewKind::Combined),
            options: Options {
                refine_embedding: refine,
                ..default()
            },
            quit: true,
            ..default()
        });
        assert_eq!(exit, AppExit::Success);
        let state = app.world().resource::<SkeletonResource>();
        assert!(state.contracted_mesh().is_some());
        assert_eq!(state.progress.labeled.is_some(), stage >= Stage::Labeling);
        assert!(state.initialization_failure.is_none());
        assert_eq!(state.extraction().is_some(), stage != Stage::Deflation);
        assert!(
            app.world()
                .resource::<SolutionResource>()
                .current_solution
                .flow_graphs
                .is_none()
        );
        let settings =
            &app.world().resource::<RenderObjectSettingStore>().objects[&Objects::InputMesh];
        let layer = if stage == Stage::Deflation {
            "deflated"
        } else if stage >= Stage::Labeling {
            "labeled skeleton"
        } else {
            "skeleton"
        };
        assert!(settings.settings[layer].visible);
        assert!(!settings.settings["lambert"].visible);
        if stage != Stage::Deflation {
            assert!(settings.settings["translucent patches"].visible);
            assert!(!settings.settings["patch boundaries"].visible);
        }
    }
    let (app, exit) = run_startup(Launch {
        path: Some(path.with_file_name("missing-startup-model.obj")),
        quit: true,
        ..default()
    });
    assert_eq!(exit, AppExit::error());
    assert!(app.world().resource::<SkeletonResource>().state.is_none());
}

#[test]
fn automatic_initialization_requests_the_full_pipeline_with_optional_refinement() {
    for action in ["--auto-start=skeleton", "--quit-after=skeleton"] {
        let launch = parse(&["--load=a.obj", action]).unwrap();
        assert_eq!(launch.stop, Some(Stage::LoopDrawing));
        let launch = parse(&["--load=a.obj", action, "--refinement"]).unwrap();
        assert_eq!(launch.stop, Some(Stage::LoopDrawing));
        assert!(launch.options.refine_embedding);
    }
}

#[test]
fn auto_start_reaches_loop_drawing_and_explicit_stops_still_win() {
    use crate::render::Objects;
    use crate::render::store::RenderObjectSettingStore;
    use crate::resources::SkeletonResource;
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../crates/mehsh/assets/tetrahedron.obj");
    for stop in [None, Some("surgery"), Some("labeling")] {
        let mut args = vec![
            OsString::from("--load"),
            path.as_os_str().to_owned(),
            OsString::from("--auto-start=skeleton"),
            OsString::from("--view=combined"),
        ];
        if let Some(stop) = stop {
            args.extend([OsString::from("--stop-after"), OsString::from(stop)]);
        }
        let mut launch = Launch::parse(args).unwrap().unwrap();
        launch.quit = true;
        let (app, exit) = run_startup(launch);
        assert_eq!(exit, AppExit::Success);
        let state = app.world().resource::<SkeletonResource>();
        if stop.is_none() {
            assert!(state.initialization_failure.is_none());
            assert!(state.progress.loops.is_some());
            let solution = &app
                .world()
                .resource::<crate::resources::SolutionResource>()
                .current_solution;
            assert_eq!(
                solution.loops.len(),
                state.progress.loops.as_ref().unwrap().len()
            );
            assert!(solution.dual.is_ok());
            // The reconstruction is kept as the skeleton output, up to its last stage.
            let output = state.output.as_ref().expect("the skeleton output is kept");
            assert_eq!(
                output.loops.keys().collect::<Vec<_>>(),
                solution.loops.keys().collect::<Vec<_>>()
            );
            assert_eq!(output.layout.is_some(), solution.layout.is_some());
            assert_eq!(output.quad.is_some(), solution.quad.is_some());
            assert!(output.fields.is_none());
            assert!(state.extraction().is_some());
            assert!(state.progress.labeling.is_some());
            state
                .progress
                .realized
                .as_ref()
                .unwrap()
                .validate(
                    &app.world()
                        .resource::<crate::resources::InputResource>()
                        .mesh,
                )
                .unwrap();
        } else {
            assert!(state.initialization_failure.is_none());
        }
        assert_eq!(state.progress.labeled.is_some(), stop != Some("surgery"));
        let settings = &app.world().resource::<RenderObjectSettingStore>().objects
            [&Objects::InputMesh]
            .settings;
        assert_eq!(settings["skeleton"].visible, stop == Some("surgery"));
        if stop != Some("surgery") {
            assert!(settings["labeled skeleton"].visible);
        }
    }
}

#[test]
fn placement_stop_and_cube_view_succeed_without_running_optimization() {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../crates/mehsh/assets/tetrahedron.obj");
    let launch = Launch::parse([
        OsString::from("--load"),
        path.into_os_string(),
        OsString::from("--quit-after=placement"),
        OsString::from("--view=cubes"),
    ])
    .unwrap()
    .unwrap();
    let (app, exit) = run_startup(launch);
    assert_eq!(exit, AppExit::Success);
    let state = app.world().resource::<crate::resources::SkeletonResource>();
    assert!(state.progress.realized.is_some());
    assert!(state.initialization_failure.is_none());
    let settings = &app
        .world()
        .resource::<crate::render::store::RenderObjectSettingStore>()
        .objects[&crate::render::Objects::InputMesh]
        .settings;
    assert!(settings["placed cubes"].visible);
    assert!(settings["cube edges"].visible);
    assert!(!settings["lambert"].visible);
    assert!(parse(&["--load=a.obj", "--stop-after=labeling", "--view=cubes"]).is_err());
}

#[test]
fn optimized_view_needs_complete_initialization() {
    let launch = parse(&["--load=a.obj", "--auto-start=skeleton", "--view=optimized"]).unwrap();
    assert_eq!(launch.view, Some(ModelViewKind::Optimized));
    assert!(parse(&["--load=a.obj", "--stop-after=placement", "--view=optimized"]).is_err());
}

#[test]
fn saves_may_open_in_any_skeleton_view() {
    let launch = parse(&["--load=a.dc", "--view=cubes"]).unwrap();
    assert_eq!(launch.view, Some(ModelViewKind::Cubes));
    assert!(parse(&["--load=a.obj", "--view=cubes"]).is_err());
}

#[test]
fn skeleton_loops_need_extraction() {
    let launch = parse(&[
        "--load=a.obj",
        "--stop-after=surgery",
        "--view=skeleton-loops",
    ])
    .unwrap();
    assert_eq!(launch.view, Some(ModelViewKind::SkeletonLoops));
    let launch = parse(&[
        "--load=a.obj",
        "--stop-after=surgery",
        "--view=skeleton-gray",
    ])
    .unwrap();
    assert_eq!(launch.view, Some(ModelViewKind::GraySkeleton));
    assert!(
        parse(&[
            "--load=a.obj",
            "--stop-after=deflation",
            "--view=skeleton-loops"
        ])
        .is_err()
    );
}

#[test]
fn placement_views_need_placement() {
    for (name, view) in [
        ("placed-skeleton", ModelViewKind::PlacedSkeleton),
        ("node-cubes", ModelViewKind::NodeCubes),
        ("cubes-primal", ModelViewKind::CubesPrimal),
        ("cubes-dual", ModelViewKind::CubesDual),
    ] {
        let flag = format!("--view={name}");
        let launch = parse(&["--load=a.obj", "--stop-after=placement", flag.as_str()]).unwrap();
        assert_eq!(launch.view, Some(view));
        assert!(parse(&["--load=a.obj", "--stop-after=labeling", flag.as_str()]).is_err());
    }
}
