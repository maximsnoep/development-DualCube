use crate::render::{self, RenderObjectStore};
use crate::{InputResource, SolutionResource};
use bevy::prelude::*;
use bevy::tasks::futures_lite::future;
use bevy::tasks::{AsyncComputeTaskPool, Task};
use dualcube::prelude::*;
use dualcube::solutions::{Loop, LoopID};
use io::Export;
use ordered_float::OrderedFloat;
use std::collections::HashMap;
use std::ops::Deref;
use std::path::PathBuf;

async fn run_job(job: Job) -> Option<JobResult> {
    match job {
        Job::Hex { solution } => None,
        Job::Import { path } => Some(JobResult::Imported(io::import_solution(path))),
        Job::Export { solution, path } => {
            if solution.mesh_ref.vert_ids().is_empty() {
                return None;
            }
            if let Err(err) = io::Dcube::export(&solution, &path) {
                warn!("Failed to export Dcube file: {err:?}");
            }
            if let Err(err) = io::Obj::export(&solution, &path) {
                warn!("Failed to export Obj file: {err:?}");
            }
            if let Err(err) = io::Flag::export(&solution, &path) {
                warn!("Failed to export Flag file: {err:?}");
            }
            None
        }
        Job::ExportNLR { solution, path } => {
            if solution.mesh_ref.vert_ids().is_empty() {
                return None;
            }
            if let Err(err) = io::Nlr::export(&solution, &path) {
                warn!("Failed to export NLR file: {err:?}");
            }
            None
        }
        Job::InitializeLoops { solution, flowgraphs } => {
            let mut solution_clone = solution.clone();
            solution_clone.initialize(&flowgraphs);
            Some(JobResult::Recomputed(solution_clone))
        }
        Job::AddLoop {
            solution,
            seed,
            direction,
            flowgraph,
        } => {
            let Some((candidate_loop, _)) = solution.construct_unbounded_loop(seed, direction, &flowgraph, |a: f64| OrderedFloat(a.powi(10))) else {
                return Some(JobResult::AddedLoop((seed, direction, None)));
            };
            let mut candidate_solution = solution.clone();
            candidate_solution.add_loop(Loop {
                edges: candidate_loop,
                direction,
            });
            if let Err(err) = candidate_solution.reconstruct_solution(false, 8) {
                warn!("Failed to reconstruct solution: {err:?}");
                return Some(JobResult::AddedLoop((seed, direction, None)));
            }
            Some(JobResult::AddedLoop((seed, direction, Some(candidate_solution))))
        }
        Job::RemoveLoop { solution, loop_id, force } => {
            let mut candidate_solution = solution.clone();
            candidate_solution.del_loop(loop_id);
            if candidate_solution.reconstruct_solution(true, 8).is_ok() || force {
                Some(JobResult::RemovedLoop(Some(candidate_solution)))
            } else {
                Some(JobResult::RemovedLoop(None))
            }
        }
        Job::Recompute { solution, unit, omega } => {
            let mut solution_clone = solution.clone();
            if let Err(err) = solution_clone.reconstruct_solution(unit, omega) {
                warn!("Failed to reconstruct solution: {err:?}");
            }
            Some(JobResult::Recomputed(solution_clone))
        }
        Job::Refresh { solution } => Some(JobResult::Refreshed(render::refresh(&solution))),
        Job::SmoothenQuad { solution } => {
            let mut solution_clone = solution.clone();
            if let Some(quad) = solution_clone.quad.as_mut() {
                quad.smoothing(10, &solution_clone.mesh_ref, false);
                Some(JobResult::Recomputed(solution_clone))
            } else {
                None
            }
        }
        Job::Evolve {
            solution,
            iterations,
            pool1,
            pool2,
            flowgraphs,
        } => {
            if let Some(new_sol) = solution.evolve(iterations, pool1, pool2, &flowgraphs) {
                Some(JobResult::Recomputed(new_sol))
            } else {
                warn!("Failed to evolve solution.");
                None
            }
        }
    }
}

/// Polls the current job for completion.
fn poll_jobs(
    mut job_state: ResMut<JobState>,
    mut jobs: EventWriter<JobRequest>,
    mut input_resource: ResMut<InputResource>,
    mut solution_resource: ResMut<SolutionResource>,
    mut render_object_store: ResMut<RenderObjectStore>,
) {
    if let (Some(request), Some(mut task)) = (job_state.request.take(), job_state.current.take()) {
        if let Some(result) = future::block_on(future::poll_once(&mut task)) {
            match result {
                Some(JobResult::Hexed(res)) => {
                    info!("Hexed completed: {:?}", res);
                    // TODO: insert into your resources
                }
                Some(JobResult::Recomputed(sol)) => {
                    solution_resource.current_solution = sol;
                    jobs.write(JobRequest::Run(Box::new(Job::Refresh {
                        solution: solution_resource.current_solution.clone(),
                    })));
                }
                Some(JobResult::Imported(sol)) => {
                    *input_resource = InputResource::new(sol.mesh_ref.clone());
                    solution_resource.current_solution = sol;
                    solution_resource.next[0] = HashMap::new();
                    solution_resource.next[1] = HashMap::new();
                    solution_resource.next[2] = HashMap::new();
                    jobs.write(JobRequest::Run(Box::new(Job::Recompute {
                        solution: solution_resource.current_solution.clone(),
                        unit: true,
                        omega: 1,
                    })));
                }
                Some(JobResult::AddedLoop((seed, direction, maybe))) => {
                    solution_resource.next[direction as usize].insert(seed, maybe);
                }
                Some(JobResult::RemovedLoop(maybe)) => {
                    if let Some(sol) = maybe {
                        solution_resource.current_solution = sol;
                        solution_resource.next[0].clear();
                        solution_resource.next[1].clear();
                        solution_resource.next[2].clear();
                        jobs.write(JobRequest::Run(Box::new(Job::Recompute {
                            solution: solution_resource.current_solution.clone(),
                            unit: true,
                            omega: 1,
                        })));
                    }
                }
                Some(JobResult::Refreshed(new_render_object_store)) => *render_object_store = new_render_object_store,
                None => {
                    info!("Job ended with no result (silently, cancelled or failed)");
                }
            }
        } else {
            job_state.request = Some(request);
            job_state.current = Some(task);
        }
    }
}

/// Job stuff
pub struct JobPlugin;

impl Plugin for JobPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<JobState>().add_event::<JobRequest>().add_systems(
            Update,
            (
                submit_jobs,
                poll_jobs.run_if(bevy::time::common_conditions::on_timer(std::time::Duration::from_millis(1000))),
            ),
        );
    }
}

/// Submits jobs to the worker thread (only if idle).
fn submit_jobs(mut ev_reader: EventReader<JobRequest>, mut job_state: ResMut<JobState>) {
    for ev in ev_reader.read() {
        match (ev, job_state.request.clone()) {
            (JobRequest::Run(job), None) => {
                info!("Received new job: {}", job.to_type());

                let job = job.clone();
                job_state.request = Some(job.to_type());

                let task = AsyncComputeTaskPool::get().spawn(async { run_job(*job).await });
                job_state.current = Some(task);
            }
            (JobRequest::Cancel, Some(request)) => {
                //
            }
            _ => {}
        }
    }
}

/// Jobs
#[derive(Clone)]
pub enum Job {
    Import {
        path: PathBuf,
    },
    Export {
        solution: Solution,
        path: PathBuf,
    },
    ExportNLR {
        solution: Solution,
        path: PathBuf,
    },
    InitializeLoops {
        solution: Solution,
        flowgraphs: [grapff::fixed::FixedGraph<EdgeID, f64>; 3],
    },
    AddLoop {
        solution: Solution,
        seed: [EdgeID; 2],
        direction: PrincipalDirection,
        flowgraph: grapff::fixed::FixedGraph<EdgeID, f64>,
    },
    RemoveLoop {
        solution: Solution,
        loop_id: LoopID,
        force: bool,
    },
    Hex {
        solution: Solution,
    },
    Evolve {
        solution: Solution,
        iterations: usize,
        pool1: usize,
        pool2: usize,
        flowgraphs: [grapff::fixed::FixedGraph<EdgeID, f64>; 3],
    },
    Recompute {
        solution: Solution,
        unit: bool,
        omega: usize,
    },
    Refresh {
        solution: Solution,
    },
    SmoothenQuad {
        solution: Solution,
    },
}

/// Job types
#[derive(PartialEq, Clone)]
pub enum JobType {
    Import,
    Export,
    ExportNLR,
    Hex,
    Evolve,
    InitializeLoops,
    AddLoop,
    RemoveLoop,
    Recompute,
    Refresh,
    SmoothenQuad,
}

impl std::fmt::Display for JobType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JobType::Import => write!(f, "importing"),
            JobType::Export => write!(f, "exporting"),
            JobType::ExportNLR => write!(f, "exporting (NLR)"),
            JobType::Hex => write!(f, "hexing"),
            JobType::Evolve => write!(f, "evolving"),
            JobType::Recompute => write!(f, "recomputing"),
            JobType::Refresh => write!(f, "refreshing"),
            JobType::AddLoop => write!(f, "adding loop"),
            JobType::RemoveLoop => write!(f, "removing loop"),
            JobType::SmoothenQuad => write!(f, "smoothening quad"),
            JobType::InitializeLoops => write!(f, "initializing loops"),
        }
    }
}

impl Job {
    fn to_type(&self) -> JobType {
        match self {
            Job::Import { .. } => JobType::Import,
            Job::Export { .. } => JobType::Export,
            Job::ExportNLR { .. } => JobType::ExportNLR,
            Job::Hex { .. } => JobType::Hex,
            Job::Evolve { .. } => JobType::Evolve,
            Job::Recompute { .. } => JobType::Recompute,
            Job::Refresh { .. } => JobType::Refresh,
            Job::AddLoop { .. } => JobType::AddLoop,
            Job::RemoveLoop { .. } => JobType::RemoveLoop,
            Job::SmoothenQuad { .. } => JobType::SmoothenQuad,
            Job::InitializeLoops { .. } => JobType::InitializeLoops,
        }
    }
}

/// Results of jobs
enum JobResult {
    Imported(Solution),
    Hexed(PathBuf),
    Recomputed(Solution),
    Refreshed(RenderObjectStore),
    AddedLoop(([EdgeID; 2], PrincipalDirection, Option<Solution>)),
    RemovedLoop(Option<Solution>),
}

/// Singleton job state
#[derive(Resource, Default)]
pub struct JobState {
    pub request: Option<JobType>,
    current: Option<Task<Option<JobResult>>>,
}

#[derive(Event)]
pub enum JobRequest {
    Run(Box<Job>),
    Cancel,
}
