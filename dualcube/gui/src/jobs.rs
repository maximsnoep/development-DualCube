use crate::render::{self, RenderObjectStore};
use crate::{Configuration, InputResource, Phase, SolutionResource};
use bevy::prelude::*;
use bevy::tasks::futures_lite::future;
use bevy::tasks::{AsyncComputeTaskPool, Task};
use dualcube::polycube::POLYCUBE;
use dualcube::prelude::*;
use dualcube::solutions::{Loop, LoopID};
use io::Export;
use itertools::Itertools;
use mehsh::prelude::{HasPosition, SetPosition, VertKey};
use ordered_float::OrderedFloat;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;

async fn run_job(job: Job) -> Option<JobResult> {
    match job {
        Job::Hex { .. } => None,
        Job::Import {
            path,
            configuration,
        } => Some(JobResult::Imported((
            io::import_solution(path),
            configuration,
        ))),

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
            if let Some(new_sol) = solution.evolve(
                configuration.iterations,
                configuration.pool1,
                configuration.pool2,
                &flowgraphs,
            ) {
                Some(JobResult::LoopsChanged((new_sol, configuration)))
            } else {
                warn!("Failed to evolve solution.");
                None
            }
        }

        Job::ComputeDual {
            solution,
            configuration,
        } => {
            let mut solution_clone = solution.clone();
            if let Err(err) = solution_clone.construct_dual_and_polycube() {
                warn!("Failed to construct dual and polycube: {err:?}");
                return Some(JobResult::Refreshed(render::refresh(&solution)));
            }
            Some(JobResult::DualChanged((solution_clone, configuration)))
        }

        Job::PlaceCorners {
            solution,
            configuration,
        } => {
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
            if let Err(err) = solution_clone.move_corner_to(corner, new_vertex) {
                warn!("Failed to move corner: {err:?}");
                return None;
            }
            Some(JobResult::LayoutChanged((solution_clone, configuration)))
        }

        Job::PlacePaths {
            solution,
            configuration,
        } => {
            let mut solution_clone = solution.clone();
            if let Err(err) = solution_clone.place_paths() {
                warn!("Failed to place paths: {err:?}");
                return None;
            }
            Some(JobResult::LayoutChanged((solution_clone, configuration)))
        }

        Job::SmoothenLayout {
            solution,
            configuration,
        } => {
            let mut solution_clone = solution.clone();
            if let Err(err) = solution_clone.optimize_corners() {
                warn!("Failed to optimize corners: {err:?}");
                return None;
            }
            Some(JobResult::LayoutChanged((solution_clone, configuration)))
        }

        Job::ComputePolycube {
            configuration,
            solution,
        } => {
            let mut solution_clone = solution.clone();
            if let Err(err) = solution_clone.resize_polycube(configuration.unit) {
                warn!("Failed to resize polycube: {err:?}");
                return None;
            }
            Some(JobResult::PolycubeChanged((solution_clone, configuration)))
        }

        Job::ComputeQuad {
            solution,
            configuration,
        } => {
            let mut solution_clone = solution.clone();
            if let Err(err) = solution_clone.construct_quad(configuration.omega) {
                warn!("Failed to construct quad: {err:?}");
                return None;
            }
            Some(JobResult::QuadChanged((solution_clone, configuration)))
        }

        Job::Refresh { solution } => Some(JobResult::Refreshed(render::refresh(&solution))),

        Job::AddLoop {
            solution,
            anchors,
            direction,
            flowgraph,
        } => {
            let Some((candidate_loop, _)) =
                solution.construct_loop_with_anchors(&anchors, direction, &flowgraph, |a: f64| {
                    OrderedFloat(a.powi(3))
                })
            else {
                return Some(JobResult::AddedLoop((anchors, direction, None)));
            };
            let mut candidate_solution = solution.clone();
            candidate_solution.add_loop(Loop {
                edges: candidate_loop,
                direction,
            });
            if candidate_solution.loops.len() >= 14 {
                if let Err(err) = candidate_solution.construct_dual_and_polycube() {
                    warn!("Failed to reconstruct solution: {err:?}");
                    return Some(JobResult::AddedLoop((anchors, direction, None)));
                }
            }
            Some(JobResult::AddedLoop((
                anchors,
                direction,
                Some(candidate_solution),
            )))
        }
        Job::RemoveLoop {
            solution,
            loop_id,
            force,
            configuration,
        } => {
            let mut candidate_solution = solution.clone();
            candidate_solution.del_loop(loop_id);
            // if candidate_solution.reconstruct_solution(true, 8).is_ok() || force {
            Some(JobResult::RemovedLoop(Some((
                candidate_solution,
                configuration,
            ))))
            // } else {
            //     Some(JobResult::RemovedLoop(None))
            // }
        }

        Job::Export { solution, path } => {
            if solution.mesh_ref.vert_ids().is_empty() {
                return None;
            }
            if let Err(err) = io::Dcube::export(&solution, &path) {
                warn!("Failed to export Dcube file: {err:?}");
            }
            if let Err(err) = io::Dsol::export(&solution, &path) {
                warn!("Failed to export Dsol file: {err:?}");
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
        Job::ExportDotgraph { solution, path } => {
            if let Err(err) = io::Dotgraph::export(&solution, &path) {
                warn!("Failed to export Dotgraph file: {err:?}");
            }
            None
        }
        Job::PathStraightening {
            solution,
            configuration,
        } => {
            if let Some(layout) = &solution.layout {
                let mut solution_clone = solution.clone();
                for _ in 0..3 {
                    let input_path = PathBuf::from("C:\\Users\\20182085\\Documents\\flip-geodesics-demo\\build\\bin\\Release\\temp.obj");
                    let output_path = PathBuf::from("C:\\Users\\20182085\\Documents\\flip-geodesics-demo\\build\\bin\\Release\\temp.lines");

                    info!("Writing OBJ+ file to {input_path:?}");
                    let vertex_map = layout.granulated_mesh.to_obj(&input_path).unwrap();
                    // open the file and write the lines to it
                    let mut file = OpenOptions::new()
                        .append(true)
                        .open(input_path.clone())
                        .unwrap();
                    for path in layout.edge_to_path.values() {
                        let line = path
                            .iter()
                            .map(|vert_id| format!("{}", vertex_map.id(vert_id).unwrap()))
                            .join(" ");
                        writeln!(file, "l {}", line).unwrap();
                    }

                    // Run path straightening algorithm
                    // using command line: C:\Users\20182085\Documents\flip-geodesics-demo\build\bin\Release\flip_geodesics.exe .\{path.obj}
                    let status = std::process::Command::new("C:\\Users\\20182085\\Documents\\flip-geodesics-demo\\build\\bin\\Release\\flip_geodesics.exe")
                        .arg(&input_path)
                        .arg(&output_path)
                        .status()
                        .unwrap();
                    if !status.success() {
                        warn!("Path straightening failed");
                    }
                    info!("Path straightening succeeded");

                    // Read the output
                    let paths = std::fs::read_to_string(&output_path).unwrap();

                    // Split into lines
                    let mut lines = paths.lines();

                    let mut mesh = solution_clone
                        .layout
                        .as_ref()
                        .unwrap()
                        .granulated_mesh
                        .clone();

                    // Go through lines
                    while let Some(line) = lines.next() {
                        // Skip empty line
                        if line.trim().is_empty() {
                            continue;
                        }
                        // Next line should end with a integer
                        let n: usize = line
                            .split_whitespace()
                            .last()
                            .and_then(|s| s.parse().ok())
                            .unwrap();

                        // let random = random_range(0. ..360.);
                        for i in 0..n {
                            if let Some(line) = lines.next() {
                                // Lines are formatted either as:
                                // - v INDEX_A
                                // - e INDEX_A INDEX_B T_VALUE, where it should be positioned at T_VALUE from INDEX_A to INDEX_B
                                let parts: Vec<&str> = line.split_whitespace().collect();
                                match parts.as_slice() {
                                    ["v", index] => {
                                        let index: usize = index.parse::<usize>().unwrap() + 1;
                                        // get the position by using the vertex_map
                                        if let Some(&vert_id) = vertex_map.key(index) {}
                                    }
                                    ["e", start, end, t_value] => {
                                        let start: usize = start.parse::<usize>().unwrap() + 1;
                                        let end: usize = end.parse::<usize>().unwrap() + 1;
                                        let t_value: f64 = t_value.parse().unwrap();
                                        if t_value < 0.001 || t_value > 0.999 {
                                            continue;
                                        }
                                        if let (Some(&start_vert), Some(&end_vert)) =
                                            (vertex_map.key(start), vertex_map.key(end))
                                        {
                                            let start_pos = mesh.position(start_vert);
                                            let end_pos = mesh.position(end_vert);
                                            // get the position T_VALUE from start_pos towards end_pos
                                            let position = start_pos.lerp(&end_pos, t_value);

                                            // split edge at value t
                                            if let Some((edge_id, _)) =
                                                mesh.edge_between_verts(start_vert, end_vert)
                                            {
                                                let new_vert_id = mesh.split_edge(edge_id).0;
                                                mesh.set_position(new_vert_id, position);
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                        // Find in the layout, the path that starts and ends with the selected vertices
                    }

                    println!(
                        "nr of verts: {}",
                        solution_clone
                            .layout
                            .as_mut()
                            .unwrap()
                            .granulated_mesh
                            .nr_verts()
                    );

                    solution_clone.layout.as_mut().unwrap().granulated_mesh = mesh;

                    println!(
                        "nr of verts: {}",
                        solution_clone
                            .layout
                            .as_mut()
                            .unwrap()
                            .granulated_mesh
                            .nr_verts()
                    );

                    loop {
                        if solution_clone
                            .layout
                            .as_mut()
                            .unwrap()
                            .place_all_paths()
                            .is_err()
                        {
                            println!("Error placing paths");
                            continue;
                        };
                        if solution_clone
                            .layout
                            .as_mut()
                            .unwrap()
                            .assign_all_patches()
                            .is_err()
                        {
                            println!("Error assigning patches");
                            continue;
                        }
                        break;
                    }
                }
                return Some(JobResult::LayoutChanged((solution_clone, configuration)));
            }
            None
        }
    }
}

/// Polls the current job for completion.
fn poll_jobs(
    mut job_state: ResMut<JobState>,
    mut jobs: MessageWriter<JobRequest>,
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

                Some(JobResult::Imported((solution, configuration))) => {
                    *input_resource = InputResource::new(solution.mesh_ref.clone());
                    solution_resource.current_solution = solution;
                    solution_resource.next[0] = HashMap::new();
                    solution_resource.next[1] = HashMap::new();
                    solution_resource.next[2] = HashMap::new();

                    jobs.write(JobRequest::Run(Box::new(Job::Refresh {
                        solution: solution_resource.current_solution.clone(),
                    })));
                }

                Some(JobResult::AddedLoop((anchors, direction, maybe))) => {
                    for seed in anchors {
                        solution_resource.next[direction as usize].insert(seed, maybe.clone());
                    }
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

                Some(JobResult::Refreshed(new_render_object_store)) => {
                    *render_object_store = new_render_object_store
                }

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
        app.init_resource::<JobState>()
            .add_message::<JobRequest>()
            .add_systems(
                Update,
                (
                    submit_jobs,
                    poll_jobs.run_if(bevy::time::common_conditions::on_timer(
                        std::time::Duration::from_millis(1000),
                    )),
                ),
            );
    }
}

/// Submits jobs to the worker thread (only if idle).
fn submit_jobs(mut ev_reader: MessageReader<JobRequest>, mut job_state: ResMut<JobState>) {
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
        configuration: Configuration,
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
    ExportDotgraph {
        solution: Solution,
        path: PathBuf,
    },
    // MANUAL
    AddLoop {
        solution: Solution,
        anchors: Vec<[EdgeID; 2]>,
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
    // PATH STRAIGTHENING
    PathStraightening {
        solution: Solution,
        configuration: Configuration,
    },
}

/// Job types
#[derive(PartialEq, Clone)]
pub enum JobType {
    Import,
    Export,
    ExportNLR,
    ExportDotgraph,
    Hex,
    Evolve,
    ComputePolycube,
    InitializeLoops,
    AddLoop,
    RemoveLoop,
    Recompute,
    Refresh,
    SmoothenLayout,
    ComputeDual,
    PlaceCorners,
    MoveCorner,
    PlacePaths,
    ComputeQuad,
    PathStraightening,
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
            JobType::InitializeLoops => write!(f, "initializing loops"),
            JobType::ComputeDual => write!(f, "computing dual"),
            JobType::PlaceCorners => write!(f, "placing corners"),
            JobType::MoveCorner => write!(f, "moving corner"),
            JobType::PlacePaths => write!(f, "placing paths"),
            JobType::ComputeQuad => write!(f, "computing quad"),
            JobType::ComputePolycube => write!(f, "computing polycube"),
            JobType::ExportDotgraph => write!(f, "exporting (Dotgraph)"),
            JobType::PathStraightening => write!(f, "path straightening"),
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
            Job::InitializeLoops { .. } => JobType::InitializeLoops,
            Job::ComputeDual { .. } => JobType::ComputeDual,
            Job::PlaceCorners { .. } => JobType::PlaceCorners,
            Job::MoveCorner { .. } => JobType::MoveCorner,
            Job::PlacePaths { .. } => JobType::PlacePaths,
            Job::ComputeQuad { .. } => JobType::ComputeQuad,
            Job::ComputePolycube { .. } => JobType::ComputePolycube,
            Job::ExportDotgraph { .. } => JobType::ExportDotgraph,
            Job::PathStraightening { .. } => JobType::PathStraightening,
        }
    }
}

/// Results of jobs
enum JobResult {
    Imported((Solution, Configuration)),
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

    AddedLoop((Vec<[EdgeID; 2]>, PrincipalDirection, Option<Solution>)),
    RemovedLoop(Option<(Solution, Configuration)>),
}

/// Singleton job state
#[derive(Resource, Default)]
pub struct JobState {
    pub request: Option<JobType>,
    current: Option<Task<Option<JobResult>>>,
}

#[derive(Message)]
pub enum JobRequest {
    Run(Box<Job>),
    Cancel,
}
