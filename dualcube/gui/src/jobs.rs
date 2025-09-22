use crate::render::{self, RenderObjectStore};
use crate::{Configuration, InputResource, Phase, SolutionResource};
use bevy::prelude::*;
use bevy::tasks::futures_lite::future;
use bevy::tasks::{AsyncComputeTaskPool, Task};
use dualcube::polycube::POLYCUBE;
use dualcube::prelude::*;
use dualcube::solutions::{Loop, LoopID};
use io::Export;
use mehsh::prelude::VertKey;
use ordered_float::OrderedFloat;
use std::collections::HashMap;
use std::path::PathBuf;

async fn run_job(job: Job) -> Option<JobResult> {
    match job {
        Job::Hex { .. } => None,
        Job::Import { path } => Some(JobResult::Imported(io::import_solution(path))),

        Job::InitializeLoops {
            solution,
            flowgraphs,
            configuration,
        } => {
            let mut solution = Solution::new(solution.mesh_ref.clone());
            solution.initialize(&flowgraphs);
            Some(JobResult::LoopsChanged((solution, configuration)))
        }

        Job::Evolve {
            solution,

            flowgraphs,
            configuration,
        } => {
            if let Some(new_sol) = solution.evolve(configuration.iterations, configuration.pool1, configuration.pool2, &flowgraphs) {
                Some(JobResult::LoopsChanged((new_sol, configuration)))
            } else {
                warn!("Failed to evolve solution.");
                None
            }
        }

        Job::ComputeDual { solution, configuration } => {
            let mut solution_clone = solution.clone();
            if let Err(err) = solution_clone.construct_dual_and_polycube() {
                warn!("Failed to construct dual and polycube: {err:?}");
                return None;
            }
            Some(JobResult::DualChanged((solution_clone, configuration)))
        }

        Job::PlaceCorners { solution, configuration } => {
            let mut solution_clone = solution.clone();
            if let Err(err) = solution_clone.place_corners() {
                warn!("Failed to place corners and paths: {err:?}");
                return None;
            }
            Some(JobResult::CornersPlaced((solution_clone, configuration)))
        }

        Job::MoveCorner {
            configuration,
            solution,
            corner,
            new_vertex,
        } => {
            let mut solution_clone = solution.clone();
            if let Err(err) = solution_clone.move_corner(corner, new_vertex) {
                warn!("Failed to move corner: {err:?}");
                return None;
            }
            Some(JobResult::LayoutChanged((solution_clone, configuration)))
        }

        Job::PlacePaths { solution, configuration } => {
            let mut solution_clone = solution.clone();
            if let Err(err) = solution_clone.place_paths() {
                warn!("Failed to place paths: {err:?}");
                return None;
            }
            Some(JobResult::LayoutChanged((solution_clone, configuration)))
        }

        Job::SmoothenLayout { solution, configuration } => {
            let mut solution_clone = solution.clone();
            if let Err(err) = solution_clone.optimize_corners() {
                warn!("Failed to optimize corners: {err:?}");
                return None;
            }
            Some(JobResult::LayoutChanged((solution_clone, configuration)))
        }

        Job::ComputePolycube { configuration, solution } => {
            let mut solution_clone = solution.clone();
            solution_clone.resize_polycube(configuration.unit);

            Some(JobResult::PolycubeChanged((solution_clone, configuration)))
        }

        Job::ComputeQuad { solution, configuration } => {
            let mut solution_clone = solution.clone();
            if let Err(err) = solution_clone.construct_quad(configuration.omega) {
                warn!("Failed to construct quad: {err:?}");
                return None;
            }
            Some(JobResult::QuadChanged((solution_clone, configuration)))
        }

        Job::SmoothenQuad { solution, configuration } => {
            let mut solution_clone = solution.clone();
            solution_clone.optimize_quad();
            Some(JobResult::QuadChanged((solution_clone, configuration)))
        }

        Job::Refresh { solution } => Some(JobResult::Refreshed(render::refresh(&solution))),

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
            if let Err(err) = candidate_solution.construct_dual_and_polycube() {
                warn!("Failed to reconstruct solution: {err:?}");
                return Some(JobResult::AddedLoop((seed, direction, None)));
            }
            Some(JobResult::AddedLoop((seed, direction, Some(candidate_solution))))
        }
        Job::RemoveLoop {
            solution,
            loop_id,
            force,
            configuration,
        } => {
            let mut candidate_solution = solution.clone();
            candidate_solution.del_loop(loop_id);
            if candidate_solution.reconstruct_solution(true, 8).is_ok() || force {
                Some(JobResult::RemovedLoop(Some((candidate_solution, configuration))))
            } else {
                Some(JobResult::RemovedLoop(None))
            }
        }

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
                Some(JobResult::LoopsChanged((solution, configuration))) => {
                    solution_resource.current_solution = solution;
                    if configuration.stop == Phase::Loops {
                        jobs.write(JobRequest::Run(Box::new(Job::Refresh {
                            solution: solution_resource.current_solution.clone(),
                        })));
                    } else {
                        jobs.write(JobRequest::Run(Box::new(Job::ComputeDual {
                            solution: solution_resource.current_solution.clone(),
                            configuration,
                        })));
                    }
                }

                Some(JobResult::DualChanged((solution, configuration))) => {
                    solution_resource.current_solution = solution;
                    if configuration.stop == Phase::Dual {
                        jobs.write(JobRequest::Run(Box::new(Job::Refresh {
                            solution: solution_resource.current_solution.clone(),
                        })));
                    } else {
                        jobs.write(JobRequest::Run(Box::new(Job::PlaceCorners {
                            solution: solution_resource.current_solution.clone(),
                            configuration,
                        })));
                    }
                }

                Some(JobResult::CornersPlaced((solution, configuration))) => {
                    solution_resource.current_solution = solution;
                    jobs.write(JobRequest::Run(Box::new(Job::PlacePaths {
                        solution: solution_resource.current_solution.clone(),
                        configuration,
                    })));
                }

                Some(JobResult::LayoutChanged((solution, configuration))) => {
                    solution_resource.current_solution = solution;
                    if configuration.stop == Phase::Layout {
                        jobs.write(JobRequest::Run(Box::new(Job::Refresh {
                            solution: solution_resource.current_solution.clone(),
                        })));
                    } else {
                        jobs.write(JobRequest::Run(Box::new(Job::ComputePolycube {
                            solution: solution_resource.current_solution.clone(),
                            configuration,
                        })));
                    }
                }

                Some(JobResult::PolycubeChanged((solution, configuration))) => {
                    solution_resource.current_solution = solution;
                    if configuration.stop == Phase::Polycube {
                        jobs.write(JobRequest::Run(Box::new(Job::Refresh {
                            solution: solution_resource.current_solution.clone(),
                        })));
                    } else {
                        jobs.write(JobRequest::Run(Box::new(Job::ComputeQuad {
                            solution: solution_resource.current_solution.clone(),
                            configuration,
                        })));
                    }
                }

                Some(JobResult::QuadChanged((solution, configuration))) => {
                    solution_resource.current_solution = solution;
                    jobs.write(JobRequest::Run(Box::new(Job::Refresh {
                        solution: solution_resource.current_solution.clone(),
                    })));
                }

                Some(JobResult::Hexed(res)) => {
                    info!("Hexed completed: {:?}", res);
                    // TODO: insert into your resources
                }

                Some(JobResult::Imported(sol)) => {
                    *input_resource = InputResource::new(sol.mesh_ref.clone());
                    solution_resource.current_solution = sol;
                    solution_resource.next[0] = HashMap::new();
                    solution_resource.next[1] = HashMap::new();
                    solution_resource.next[2] = HashMap::new();
                    jobs.write(JobRequest::Run(Box::new(Job::Refresh {
                        solution: solution_resource.current_solution.clone(),
                    })));
                }

                Some(JobResult::AddedLoop((seed, direction, maybe))) => {
                    solution_resource.next[direction as usize].insert(seed, maybe);
                }
                Some(JobResult::RemovedLoop(maybe)) => {
                    if let Some((sol, configuration)) = maybe {
                        solution_resource.current_solution = sol;
                        solution_resource.next[0].clear();
                        solution_resource.next[1].clear();
                        solution_resource.next[2].clear();
                        jobs.write(JobRequest::Run(Box::new(Job::ComputeDual {
                            solution: solution_resource.current_solution.clone(),
                            configuration,
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
            (JobRequest::Cancel, Some(_)) => {
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
    InitializeLoops {
        solution: Solution,
        configuration: Configuration,
        flowgraphs: [grapff::fixed::FixedGraph<EdgeID, f64>; 3],
    },
    Evolve {
        solution: Solution,
        configuration: Configuration,
        flowgraphs: [grapff::fixed::FixedGraph<EdgeID, f64>; 3],
    },
    ComputeDual {
        configuration: Configuration,
        solution: Solution,
    },
    PlaceCorners {
        configuration: Configuration,
        solution: Solution,
    },
    MoveCorner {
        configuration: Configuration,
        solution: Solution,
        corner: VertKey<POLYCUBE>,
        new_vertex: VertID,
    },
    PlacePaths {
        configuration: Configuration,
        solution: Solution,
    },
    SmoothenLayout {
        configuration: Configuration,
        solution: Solution,
    },
    ComputePolycube {
        configuration: Configuration,
        solution: Solution,
    },
    ComputeQuad {
        configuration: Configuration,
        solution: Solution,
    },
    SmoothenQuad {
        configuration: Configuration,
        solution: Solution,
    },
    Refresh {
        solution: Solution,
    },
    // EXPORT
    Export {
        solution: Solution,
        path: PathBuf,
    },
    ExportNLR {
        solution: Solution,
        path: PathBuf,
    },
    // MANUAL
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
        configuration: Configuration,
    },
    // HEX MESHING
    Hex {
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
    ComputePolycube,
    InitializeLoops,
    AddLoop,
    RemoveLoop,
    Recompute,
    Refresh,
    SmoothenLayout,
    SmoothenQuad,
    ComputeDual,
    PlaceCorners,
    MoveCorner,
    PlacePaths,
    ComputeQuad,
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
            JobType::SmoothenLayout => write!(f, "smoothening layout"),
            JobType::SmoothenQuad => write!(f, "smoothening quad"),
            JobType::InitializeLoops => write!(f, "initializing loops"),
            JobType::ComputeDual => write!(f, "computing dual"),
            JobType::PlaceCorners => write!(f, "placing corners"),
            JobType::MoveCorner => write!(f, "moving corner"),
            JobType::PlacePaths => write!(f, "placing paths"),
            JobType::ComputeQuad => write!(f, "computing quad"),
            JobType::ComputePolycube => write!(f, "computing polycube"),
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
            Job::Refresh { .. } => JobType::Refresh,
            Job::AddLoop { .. } => JobType::AddLoop,
            Job::RemoveLoop { .. } => JobType::RemoveLoop,
            Job::SmoothenLayout { .. } => JobType::SmoothenLayout,
            Job::SmoothenQuad { .. } => JobType::SmoothenQuad,
            Job::InitializeLoops { .. } => JobType::InitializeLoops,
            Job::ComputeDual { .. } => JobType::ComputeDual,
            Job::PlaceCorners { .. } => JobType::PlaceCorners,
            Job::MoveCorner { .. } => JobType::MoveCorner,
            Job::PlacePaths { .. } => JobType::PlacePaths,
            Job::ComputeQuad { .. } => JobType::ComputeQuad,
            Job::ComputePolycube { .. } => JobType::ComputePolycube,
        }
    }
}

/// Results of jobs
enum JobResult {
    Imported(Solution),
    // For example after initializing or optimizing loop structure
    LoopsChanged((Solution, Configuration)),
    // For example after computing dual / polycube representation
    DualChanged((Solution, Configuration)),
    // For example after placing corners or paths
    CornersPlaced((Solution, Configuration)),
    LayoutChanged((Solution, Configuration)),
    // After polycube changed
    PolycubeChanged((Solution, Configuration)),
    // For example after optimizing quad structure
    QuadChanged((Solution, Configuration)),

    Hexed(PathBuf),
    Refreshed(RenderObjectStore),

    AddedLoop(([EdgeID; 2], PrincipalDirection, Option<Solution>)),
    RemovedLoop(Option<(Solution, Configuration)>),
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
