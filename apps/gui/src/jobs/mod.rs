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

use crate::render::store::{RenderObjectSettingStore, RenderObjectStore};
use crate::resources::{Configuration, InputResource, SolutionResource};
use crate::ui::ModelView;
use bevy::prelude::*;
use bevy::tasks::futures_lite::future;
use bevy::tasks::{AsyncComputeTaskPool, Task};
use dualcube::prelude::*;
use pipeline::Stage;
use std::sync::Arc;

pub use evolution::EvolutionLive;

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
    run: Arc<dyn Fn() -> Option<JobResult> + Send + Sync>,
}

impl Job {
    fn new(
        description: &'static str,
        run: impl Fn() -> Option<JobResult> + Send + Sync + 'static,
    ) -> Self {
        Self {
            description,
            preempt: false,
            run: Arc::new(run),
        }
    }

    /// This job replaces a running job (and stops a running evolution) instead of being dropped; the result of the
    /// replaced job is discarded.
    pub fn preempting(mut self) -> Self {
        self.preempt = true;
        self
    }
}

/// What a finished job hands back to [`poll_jobs`].
enum JobResult {
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
        let job = job.clone();
        let task = AsyncComputeTaskPool::get().spawn(async move { (job.run)() });
        job_state.current = Some(task);
    }
}

/// Polls the current job for completion and applies its result.
fn poll_jobs(
    mut job_state: ResMut<'_, JobState>,
    mut jobs: MessageWriter<'_, Job>,
    mut input_resource: ResMut<'_, InputResource>,
    mut solution_resource: ResMut<'_, SolutionResource>,
    mut render_object_store: ResMut<'_, RenderObjectStore>,
    mut render_settings: ResMut<'_, RenderObjectSettingStore>,
    mut model_view: ResMut<'_, ModelView>,
    mut configuration: ResMut<'_, Configuration>,
    mut evolution_live: ResMut<'_, EvolutionLive>,
) {
    let (Some(request), Some(mut task)) = (job_state.request.take(), job_state.current.take())
    else {
        return;
    };

    let Some(result) = future::block_on(future::poll_once(&mut task)) else {
        // Not finished yet; put the job back.
        job_state.request = Some(request);
        job_state.current = Some(task);
        return;
    };

    info!("Finished job: {request}");

    let Some(result) = result else {
        // Normal for jobs without a result (e.g. exports); failures have
        // already been logged by the job itself.
        debug!("Job '{request}' produced no result to apply");
        return;
    };

    match result {
        JobResult::StageCompleted {
            stage,
            solution,
            configuration,
        } => {
            solution_resource.current_solution = solution.clone();
            jobs.write(stage.next_job(solution, configuration));
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
            // A new model: the histories of the evolutions belong to the old one.
            evolution_live.reset();
            // Show the new model right away, as the input.
            *render_object_store = store;
            model_view.show_input(&mut render_settings);
            // The flow fields and graphs only depend on the mesh: compute them right away (then refresh).
            jobs.write(Job::prepare_flow(
                solution_resource.current_solution.clone(),
                configuration.clone(),
            ));
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
        }
    }
}
