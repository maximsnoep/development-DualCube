use super::*;
use crate::resources::SkeletonState;
use dualcube::skeleton::skeletonize;

fn app(busy: bool) -> App {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/mehsh/assets/blub001k.obj");
    let mesh = io::import_solution(&path).unwrap().mesh_ref;
    let extraction = skeletonize::extract(&mesh, Default::default()).unwrap();
    let key = boundaries_key(&extraction.skeleton);
    let selection = extraction.skeleton.edges.keys().take(2).collect();
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_message::<Job>()
        .insert_resource(InputResource::new(mesh))
        .insert_resource(Configuration {
            interactive_mode: InteractiveMode::SkeletonModification,
            ..Default::default()
        })
        .insert_resource(SkeletonResource {
            state: Some(SkeletonState::Extracted(extraction)),
            ..Default::default()
        })
        .insert_resource(CacheResource {
            skeleton_boundaries: Some((key, Arc::default())),
            skeleton_selection: selection,
            ..Default::default()
        })
        .init_resource::<JobState>()
        .add_systems(Update, apply_selection_on_leave);
    app.world_mut().resource_mut::<JobState>().request = busy.then_some("optimizing");
    app.update();
    app
}

fn leave(app: &mut App) -> usize {
    app.world_mut()
        .resource_mut::<Configuration>()
        .interactive_mode = InteractiveMode::None;
    app.update();
    app.world().resource::<Messages<Job>>().len()
}

#[test]
fn leaving_the_mode_dissolves_the_selection_once() {
    let mut app = app(false);
    assert_eq!(app.world().resource::<Messages<Job>>().len(), 0);
    assert_eq!(leave(&mut app), 1);
    assert!(
        app.world()
            .resource::<CacheResource>()
            .skeleton_selection
            .is_empty()
    );
    app.update();
    assert_eq!(app.world().resource::<Messages<Job>>().len(), 1);
}

#[test]
fn a_busy_worker_keeps_the_selection() {
    let mut app = app(true);
    assert_eq!(leave(&mut app), 0);
    assert_eq!(
        app.world()
            .resource::<CacheResource>()
            .skeleton_selection
            .len(),
        2
    );
}
