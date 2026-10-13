//! Background jobs: definitions, submission, execution, and result handling.
//!
//! To add a new job, write a single constructor in the matching submodule
//! (or a new one):
//!
//! ```ignore
//! impl Job {
//!     pub fn my_job(solution: Solution) -> Self {
//!         Self::new("doing my thing", move || {
//!             // ... work with the captured data ...
//!             None // or Some(JobResult::...)
//!         })
//!     }
//! }
//! ```
//!
//! Only if it produces a new *kind* of result do you also add a [`JobResult`]
//! variant and handle it in [`poll_jobs`].

mod evolution;
mod io;
mod loops;
mod pipeline;
mod skeleton;

use crate::render::store::{RenderObjectSettingStore, RenderObjectStore};
use crate::resources::{Configuration, InputResource, SolutionResource};
use crate::ui::ModelView;
use bevy::prelude::*;
use bevy::tasks::futures_lite::future;
use bevy::tasks::{AsyncComputeTaskPool, Task};
use dualcube::prelude::*;
use pipeline::Stage;
use std::sync::{Arc, Mutex, mpsc};

pub use evolution::EvolutionLive;
pub use skeleton::{ContractionCache, SkeletonStage};

pub struct JobPlugin;

impl Plugin for JobPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<JobState>()
            .init_resource::<EvolutionLive>()
            .add_message::<Job>()
            .add_systems(
                Update,
                (
                    submit_jobs,
                    poll_jobs.run_if(bevy::time::common_conditions::on_timer(
                        std::time::Duration::from_millis(10),
                    )),
                    evolution::poll_evolution.run_if(bevy::time::common_conditions::on_timer(
                        std::time::Duration::from_millis(100),
                    )),
                ),
            );
    }
}

/// Singleton job state. At most one job runs at a time; `request` holds the
/// description of the running job (and doubles as the busy flag). A job that
/// arrives while another runs is dropped, unless it preempts (see
/// [`Job::preempting`]).
#[derive(Resource, Default)]
pub struct JobState {
    pub request: Option<&'static str>,
    current: Option<Task<Option<JobResult>>>,
    startup: bool,
    updates: Option<Mutex<mpsc::Receiver<skeleton::Update>>>,
}

/// A unit of background work: a description (shown in the UI while running)
/// and the closure that does the work on the worker thread.
///
/// Jobs are created through the constructors in the submodules, e.g.
/// [`Job::import`], [`Job::compute_dual`], [`Job::add_loop`].
#[derive(Clone, Message)]
pub struct Job {
    description: &'static str,
    preempt: bool,
    startup: bool,
    run: Arc<dyn Fn(&mpsc::Sender<skeleton::Update>) -> Option<JobResult> + Send + Sync>,
}

impl Job {
    fn new(
        description: &'static str,
        run: impl Fn() -> Option<JobResult> + Send + Sync + 'static,
    ) -> Self {
        Self::with_updates(description, move |_| run())
    }

    fn with_updates(
        description: &'static str,
        run: impl Fn(&mpsc::Sender<skeleton::Update>) -> Option<JobResult> + Send + Sync + 'static,
    ) -> Self {
        Self {
            description,
            preempt: false,
            startup: false,
            run: Arc::new(run),
        }
    }

    /// Part of the command-line startup sequence.
    pub fn for_startup(mut self) -> Self {
        self.startup = true;
        self
    }

    /// Replaces running work; its eventual result is discarded.
    pub fn preempting(mut self) -> Self {
        self.preempt = true;
        self
    }
}

/// What a finished job hands back to [`poll_jobs`].
enum JobResult {
    SkeletonReady {
        state: crate::resources::SkeletonState,
        view: crate::ui::ModelViewKind,
        progress: dualcube::skeleton::pipeline::Progress,
        failure: Option<dualcube::skeleton::pipeline::Failure>,
    },
    /// A solution was imported (with its renders); resets the input resources and the view.
    Imported {
        solution: Solution,
        store: RenderObjectStore,
        /// The name of the model.
        name: Option<String>,
    },
    /// A pipeline stage finished; the stage decides which job runs next.
    StageCompleted {
        stage: Stage,
        solution: Solution,
        configuration: Configuration,
    },
    /// New render objects are ready to be displayed.
    Refreshed(RenderObjectStore),
    /// A loop was added.
    AddedLoop {
        solution: Solution,
        configuration: Configuration,
    },
    /// A loop was removed.
    RemovedLoop {
        solution: Solution,
        configuration: Configuration,
    },
}

/// Submits jobs to the worker thread (only if idle, or if the job preempts).
fn submit_jobs(
    mut ev_reader: MessageReader<'_, '_, Job>,
    mut job_state: ResMut<'_, JobState>,
    evolution_live: Res<'_, EvolutionLive>,
) {
    for job in ev_reader.read() {
        if let Some(running) = job_state.request {
            if !job.preempt {
                continue;
            }
            info!("Stopping job: {running}");
            // An evolution ends at its next check; other jobs run to completion on their thread, but their results
            // are discarded (the task is dropped).
            evolution_live.stop();
            job_state.current = None;
        }
        info!("Starting job: {}", job.description);
        job_state.request = Some(job.description);
        job_state.startup = job.startup;
        let job = job.clone();
        // A fresh channel also discards updates from a preempted worker.
        let (sender, receiver) = mpsc::channel();
        job_state.updates = Some(Mutex::new(receiver));
        let task = AsyncComputeTaskPool::get().spawn(async move { (job.run)(&sender) });
        job_state.current = Some(task);
    }
}

/// Polls the current job for completion and applies its result.
pub(crate) fn poll_jobs(
    mut job_state: ResMut<'_, JobState>,
    mut jobs: MessageWriter<'_, Job>,
    mut input_resource: ResMut<'_, InputResource>,
    mut solution_resource: ResMut<'_, SolutionResource>,
    mut render_object_store: ResMut<'_, RenderObjectStore>,
    mut render_settings: ResMut<'_, RenderObjectSettingStore>,
    mut model_view: ResMut<'_, ModelView>,
    mut configuration: ResMut<'_, Configuration>,
    mut evolution_live: ResMut<'_, EvolutionLive>,
    mut skeleton: ResMut<'_, crate::resources::SkeletonResource>,
    launch: Res<'_, crate::cli::Launch>,
    mut exit: MessageWriter<'_, AppExit>,
) {
    let (Some(request), Some(mut task)) = (job_state.request.take(), job_state.current.take())
    else {
        return;
    };

    let result = future::block_on(future::poll_once(&mut task));
    if let Some(updates) = &job_state.updates {
        for update in updates.lock().unwrap().try_iter() {
            update.apply(&mut skeleton);
        }
    }
    let Some(result) = result else {
        // Not finished yet; put the job back.
        job_state.request = Some(request);
        job_state.current = Some(task);
        return;
    };

    job_state.updates = None;
    let Some(result) = result else {
        if job_state.startup {
            launch.finish(false, &mut model_view, &mut exit);
        }
        // Normal for jobs without a result (e.g. exports); failures have
        // already been logged by the job itself.
        return;
    };

    info!("Finished job: {request}");

    match result {
        JobResult::SkeletonReady {
            state,
            view,
            progress,
            failure,
        } => {
            // New initialization: previous loops and their downstream results are stale.
            let current = &solution_resource.current_solution;
            let mut initialized = Solution::new(current.mesh_ref.clone());
            initialized.fields = current.fields.clone();
            initialized.flow_graphs = current.flow_graphs.clone();
            initialized.quality = configuration.quality;
            if let Some(loops) = progress
                .loops
                .as_ref()
                .or_else(|| progress.drawing.as_ref().map(|a| &a.loops))
            {
                initialized.replace_loops(loops.clone());
            }
            solution_resource.current_solution = initialized;
            solution_resource.selected_corner = None;
            configuration.loop_anchors.clear();
            evolution_live.reset();
            let complete = failure.is_none() && progress.loops.is_some();
            let mut refresh = if complete {
                Job::compute_dual(
                    solution_resource.current_solution.clone(),
                    configuration.clone(),
                )
            } else {
                Job::refresh(
                    solution_resource.current_solution.clone(),
                    configuration.clone(),
                )
            };
            refresh.startup = job_state.startup;
            jobs.write(refresh);
            skeleton.state = Some(state);
            skeleton.labeled = progress.labeled;
            skeleton.labeling = progress.labeling;
            skeleton.realized = progress.realized;
            skeleton.optimized = progress.optimized;
            skeleton.drawing = progress.drawing;
            skeleton.loops = progress.loops;
            skeleton.initialization_failure = failure;
            model_view.show_when_ready(if complete {
                crate::ui::ModelViewKind::Dual
            } else {
                view
            });
        }

        JobResult::StageCompleted {
            stage,
            solution,
            configuration,
        } => {
            solution_resource.current_solution = solution.clone();
            let mut next = stage.next_job(solution, configuration);
            next.startup = job_state.startup;
            jobs.write(next);
        }

        JobResult::Imported {
            solution,
            store,
            name,
        } => {
            let name = name.unwrap_or_else(|| input_resource.name.clone());
            *input_resource = InputResource::new(solution.mesh_ref.clone());
            input_resource.name = name;
            solution_resource.current_solution = solution;
            // The selections and anchors refer to elements of the old model.
            solution_resource.selected_corner = None;
            configuration.loop_anchors.clear();
            *skeleton = Default::default();
            if crate::resources::genus(&input_resource.mesh) != Some(0) {
                configuration.initialization = crate::resources::Initialization::Skeleton;
            }
            // A new model: the histories of the evolutions belong to the old one.
            evolution_live.reset();
            // Show the new model right away, as the input.
            *render_object_store = store;
            model_view.show_input(&mut render_settings);
            if job_state.startup {
                if let Some(stop) = launch.stop {
                    jobs.write(
                        Job::extract_skeleton_until(
                            input_resource.mesh.clone(),
                            configuration.skeleton_options,
                            stop,
                            skeleton.contraction_cache.clone(),
                        )
                        .for_startup(),
                    );
                } else {
                    launch.finish(true, &mut model_view, &mut exit);
                }
            } else {
                // Interactive imports prepare the flow graphs; skeleton startup does not need them.
                jobs.write(Job::prepare_flow(
                    solution_resource.current_solution.clone(),
                    configuration.clone(),
                ));
            }
        }

        JobResult::AddedLoop {
            solution,
            configuration,
        } => {
            solution_resource.current_solution = solution;
            jobs.write(Job::compute_dual(
                solution_resource.current_solution.clone(),
                configuration,
            ));
        }

        JobResult::RemovedLoop {
            solution,
            configuration,
        } => {
            solution_resource.current_solution = solution;
            jobs.write(Job::compute_dual(
                solution_resource.current_solution.clone(),
                configuration,
            ));
        }

        JobResult::Refreshed(new_render_object_store) => {
            *render_object_store = new_render_object_store;
            if job_state.startup {
                launch.finish(
                    skeleton.initialization_failure.is_none(),
                    &mut model_view,
                    &mut exit,
                );
            }
        }
    }
}
