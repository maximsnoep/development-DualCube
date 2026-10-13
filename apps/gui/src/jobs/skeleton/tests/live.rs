use super::*;
use crate::jobs::{self, JobState};
use crate::render::{
    Objects,
    store::{RenderObject, RenderObjectSettingStore, RenderObjectStore},
};
use crate::resources::{Configuration, InputResource, SolutionResource};
use crate::ui::ModelView;
use bevy::prelude::Update as UiUpdate;
use dualcube::prelude::Mesh;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, mpsc};
use std::time::{Duration, Instant};

fn mesh() -> Arc<Mesh<INPUT>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/mehsh/assets/tetrahedron.obj");
    io::import_solution(&path).unwrap().mesh_ref
}

fn until(app: &mut App, ready: impl Fn(&App) -> bool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        app.update();
        if ready(app) {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "Timed out waiting for intermediate UI state"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn renders_snapshots_while_worker_is_running_and_discards_preempted_updates() {
    let mesh = mesh();
    let extraction = skeletonize::extract(&mesh, Options::default()).unwrap();
    let contracted = extraction.contracted_mesh.clone();
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .insert_resource(InputResource::new(mesh))
        .init_resource::<Configuration>()
        .init_resource::<SolutionResource>()
        .init_resource::<SkeletonResource>()
        .init_resource::<RenderObjectStore>()
        .init_resource::<RenderObjectSettingStore>()
        .init_resource::<ModelView>()
        .init_resource::<crate::cli::Launch>()
        .init_resource::<JobState>()
        .init_resource::<jobs::EvolutionLive>()
        .add_message::<Job>()
        .add_systems(
            UiUpdate,
            (
                jobs::submit_jobs,
                jobs::poll_jobs,
                crate::render::objects::skeleton::sync,
            )
                .chain(),
        );
    app.world_mut()
        .resource_mut::<RenderObjectStore>()
        .objects
        .insert(Objects::InputMesh, RenderObject::default());
    let (release, gate) = mpsc::channel();
    let gate = Mutex::new(gate);
    let done = Arc::new(AtomicBool::new(false));
    let finished = done.clone();
    let first = contracted.clone();
    let job = Job::with_updates("test extraction", move |updates| {
        updates
            .send(Update::State(SkeletonState::Deflated(first.clone())))
            .unwrap();
        gate.lock()
            .unwrap()
            .recv_timeout(Duration::from_secs(5))
            .unwrap();
        updates
            .send(Update::State(SkeletonState::Extracted(extraction.clone())))
            .unwrap();
        gate.lock()
            .unwrap()
            .recv_timeout(Duration::from_secs(5))
            .unwrap();
        // An obsolete worker may still finish after another job has replaced it.
        let _ = updates.send(Update::State(SkeletonState::Extracted(extraction.clone())));
        finished.store(true, Ordering::SeqCst);
        None
    });
    app.world_mut().write_message(job);
    until(&mut app, |app| {
        app.world()
            .resource::<SkeletonResource>()
            .contracted_mesh()
            .is_some()
    });
    assert_eq!(
        app.world().resource::<JobState>().request,
        Some("test extraction")
    );
    assert!(
        app.world()
            .resource::<SkeletonResource>()
            .extraction()
            .is_none()
    );
    assert!(
        app.world().resource::<RenderObjectStore>().objects[&Objects::InputMesh]
            .features
            .contains_key("deflated")
    );
    release.send(()).unwrap();
    until(&mut app, |app| {
        app.world()
            .resource::<SkeletonResource>()
            .extraction()
            .is_some()
    });
    assert_eq!(
        app.world().resource::<JobState>().request,
        Some("test extraction")
    );
    assert!(
        app.world().resource::<RenderObjectStore>().objects[&Objects::InputMesh]
            .features
            .contains_key("skeleton")
    );
    assert!(app.should_exit().is_none());

    app.world_mut().write_message(
        Job::with_updates("replacement", move |updates| {
            updates
                .send(Update::State(SkeletonState::Deflated(contracted.clone())))
                .unwrap();
            None
        })
        .preempting(),
    );
    until(&mut app, |app| {
        app.world().resource::<JobState>().request.is_none()
    });
    release.send(()).unwrap();
    until(&mut app, |_| done.load(Ordering::SeqCst));
    assert!(
        app.world()
            .resource::<SkeletonResource>()
            .extraction()
            .is_none()
    );
    assert!(
        !app.world().resource::<RenderObjectStore>().objects[&Objects::InputMesh]
            .features
            .contains_key("skeleton")
    );
}

#[test]
fn complete_initialization_publishes_each_completed_stage() {
    let job = Job::extract_skeleton(mesh(), Options::default(), ContractionCache::default());
    let (sender, receiver) = mpsc::channel();
    let Some(JobResult::SkeletonReady {
        failure, progress, ..
    }) = (job.run)(&sender)
    else {
        panic!("Missing final initialization result");
    };
    assert!(failure.is_none(), "{failure:?}");
    assert!(progress.loops.is_some());
    let mut updates = receiver.try_iter();
    assert!(matches!(
        updates.next(),
        Some(Update::State(SkeletonState::Deflated(_)))
    ));
    assert!(matches!(
        updates.next(),
        Some(Update::State(SkeletonState::Extracted(_)))
    ));
    for stage in 0..3 {
        let Some(Update::Progress(progress)) = updates.next() else {
            panic!("Missing stage snapshot");
        };
        assert!(progress.labeled.is_some());
        assert_eq!(progress.realized.is_some(), stage >= 1);
        assert_eq!(progress.optimized.is_some(), stage >= 2);
        assert!(progress.loops.is_none());
    }
    assert!(updates.next().is_none());
}
