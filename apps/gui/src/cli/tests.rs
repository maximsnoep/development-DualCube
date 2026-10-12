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
    assert_eq!(launch.stop, Some(SkeletonStage::Complete));
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
    ])
    .unwrap();
    assert_eq!(launch.stop, Some(SkeletonStage::Complete));
    assert_eq!(launch.options.max_iterations, 7);
    assert_eq!(launch.options.surgery.shape_weight, 2.);
    assert_eq!(launch.options.surgery.sampling_weight, 0.);
    assert_eq!(launch.options.surgery.handle_bonus, 12.);
}

#[test]
fn stage_controls_and_invalid_combinations() {
    for (name, stage) in [
        ("input", None),
        ("deflation", Some(SkeletonStage::Deflation)),
        ("surgery", Some(SkeletonStage::Surgery)),
        ("refinement", Some(SkeletonStage::Refinement)),
        ("labeling", Some(SkeletonStage::Labeling)),
    ] {
        let launch = parse(&["--load=a.obj", "--stop-after", name]).unwrap();
        assert_eq!(launch.stop, stage);
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
    for stage in [
        SkeletonStage::Deflation,
        SkeletonStage::Surgery,
        SkeletonStage::Refinement,
        SkeletonStage::Labeling,
    ] {
        let (app, exit) = run_startup(Launch {
            path: Some(path.clone()),
            stop: Some(stage),
            view: (stage != SkeletonStage::Deflation).then_some(ModelViewKind::Combined),
            quit: true,
            ..default()
        });
        assert_eq!(exit, AppExit::Success);
        let state = app.world().resource::<SkeletonResource>();
        assert!(state.contracted_mesh().is_some());
        assert_eq!(state.labeled.is_some(), stage == SkeletonStage::Labeling);
        assert!(state.initialization_failure.is_none());
        assert_eq!(
            state.extraction().is_some(),
            stage != SkeletonStage::Deflation
        );
        assert!(
            app.world()
                .resource::<SolutionResource>()
                .current_solution
                .flow_graphs
                .is_none()
        );
        let settings =
            &app.world().resource::<RenderObjectSettingStore>().objects[&Objects::InputMesh];
        let layer = if stage == SkeletonStage::Deflation {
            "deflated"
        } else if stage == SkeletonStage::Labeling {
            "labeled skeleton"
        } else {
            "skeleton"
        };
        assert!(settings.settings[layer].visible);
        assert!(!settings.settings["lambert"].visible);
        if stage != SkeletonStage::Deflation {
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
        assert_eq!(launch.stop, Some(SkeletonStage::Complete));
        let launch = parse(&["--load=a.obj", action, "--refinement"]).unwrap();
        assert_eq!(launch.stop, Some(SkeletonStage::Complete));
        assert!(launch.options.refine_embedding);
    }
}

#[test]
fn auto_start_reaches_cube_placement_and_explicit_stops_still_win() {
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
        assert_eq!(
            exit,
            if stop.is_none() {
                AppExit::error()
            } else {
                AppExit::Success
            }
        );
        let state = app.world().resource::<SkeletonResource>();
        if stop.is_none() {
            assert_eq!(
                state.initialization_failure.as_ref().unwrap().stage,
                dualcube::skeleton::pipeline::Stage::CubePlacement
            );
            assert!(state.extraction().is_some());
            assert!(state.labeling.is_some());
            assert!(state.realized.is_none());
        } else {
            assert!(state.initialization_failure.is_none());
        }
        assert_eq!(state.labeled.is_some(), stop != Some("surgery"));
        let settings = &app.world().resource::<RenderObjectSettingStore>().objects
            [&Objects::InputMesh]
            .settings;
        assert_eq!(settings["skeleton"].visible, stop == Some("surgery"));
        if stop != Some("surgery") {
            assert!(settings["labeled skeleton"].visible);
        }
    }
}
