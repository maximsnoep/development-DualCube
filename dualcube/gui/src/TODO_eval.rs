use bevy::ecs::error::BevyError;
use itertools::Itertools;
use log::info;
use std::path::PathBuf;
use std::process::Command;

const SEGMENTATION_EVALUATOR: &str = "python3 ~/polycube-to-hexmesh/seg_eval.py";
const HEXMESH_PIPELINE: &str = "~/polycube-to-hexmesh/pipeline.sh";
const HEXMESH_EVALUATOR: &str = "python3 ~/polycube-to-hexmesh/evaluator_old.py";

//             ActionEvent::ToHexmesh => {
//                 if mesh_resmut.mesh.vert_ids().is_empty() {
//                     return Ok(());
//                 }

//                 let t = SystemTime::now().duration_since(UNIX_EPOCH).expect("Time went backwards").as_secs();
//                 let n = mesh_resmut.properties.source.split("\\").last().unwrap().split('.').next().unwrap().to_owned();

//                 let path = PathBuf::from(format!("out/temp/{n}_{t}"));

//                 let current_dir = env::current_dir().unwrap();
//                 let path = current_dir.join(path);
//                 println!("Exporting to: {}", path.display());

//                 let obj_path = path.with_extension("obj");
//                 let flag_path = path.with_extension("flag");
//                 let hex_path = path.with_extension("hex");
//                 let save_path = path.with_extension("dcube");

//                 if let Err(err) = io::Dcube::export(&solution.current_solution, &save_path) {
//                     return Err(BevyError::from(format!("Failed to export Dcube file: {err:?}")));
//                 }
//                 if let Err(err) = io::Obj::export(&solution.current_solution, &obj_path) {
//                     return Err(BevyError::from(format!("Failed to export Obj file: {err:?}")));
//                 }
//                 if let Err(err) = io::Flag::export(&solution.current_solution, &flag_path) {
//                     return Err(BevyError::from(format!("Failed to export Flag file: {err:?}")));
//                 }

//                 configuration.hex_mesh_status = HexMeshStatus::Loading;

//                 let task_pool = AsyncComputeTaskPool::get();
//                 let task = task_pool.spawn(async move {
//                     eval::seg_eval(&obj_path, &flag_path);
//                     eval::hexer(&obj_path, &flag_path, &hex_path);
//                     eval::evaluator(&hex_path, &obj_path)
//                 });

//                 hextasks.generating_chunks.insert(999, task);
//             }

pub struct HexData {
    pub obj_path: PathBuf,
    pub flag_path: PathBuf,
    pub hex_path: PathBuf,
}

#[derive(Clone, Debug)]
pub struct HexEval {
    pub hausdorff: f32,
    pub avg_jacob: f32,
    pub min_jacob: f32,
    pub max_jacob: f32,
    pub irregular: f32,
}

fn run(script: &str, args: &[&str]) -> Result<std::process::Output, BevyError> {
    let mut wsl = Command::new("wsl");
    let wsl_args = args.into_iter().map(|arg| arg.replace("C:", "/mnt/c"));
    let command = wsl.args([script]).args(wsl_args);
    info!("Running command: {:?}", command);
    command.output().map_err(|e| BevyError::from(e))
}

pub fn seg_eval(obj_path_in: &PathBuf, flag_path_in: &PathBuf) {
    let Ok(out) = run(SEGMENTATION_EVALUATOR, &[obj_path_in.to_str().unwrap(), flag_path_in.to_str().unwrap()]) else {
        eprintln!("Failed to run segmentation evaluation script.");
        return;
    };

    println!("Seg output: {:?}", out);
}

pub fn hexer(obj_path_in: &PathBuf, flag_path_in: &PathBuf, hex_path_out: &PathBuf) -> Option<HexData> {
    let Ok(out) = run(
        HEXMESH_PIPELINE,
        &[
            obj_path_in.to_str().unwrap(),
            "-out",
            hex_path_out.to_str().unwrap(),
            "-algo",
            flag_path_in.to_str().unwrap(),
        ],
    ) else {
        eprintln!("Failed to run hexmesh pipeline script.");
        return None;
    };

    println!("Pipeline output: {:?}", out);

    Some(HexData {
        obj_path: obj_path_in.to_owned(),
        flag_path: flag_path_in.to_owned(),
        hex_path: hex_path_out.to_owned(),
    })
}

pub fn evaluator(hex_path_in: &PathBuf, obj_path_in: &PathBuf) -> Option<HexEval> {
    let Ok(out) = run(HEXMESH_EVALUATOR, &[hex_path_in.to_str().unwrap(), obj_path_in.to_str().unwrap()]) else {
        eprintln!("Failed to run hexmesh evaluation script.");
        return None;
    };

    println!("Evaluator output: {:?}", out);

    let out_str = String::from_utf8(out.stdout).unwrap();
    let out_vec = out_str.split('\n').filter(|s| !s.is_empty()).collect_vec();

    let eval = HexEval {
        hausdorff: out_vec[0].parse().unwrap(),
        avg_jacob: out_vec[1].parse().unwrap(),
        min_jacob: out_vec[2].parse().unwrap(),
        max_jacob: out_vec[3].parse().unwrap(),
        irregular: out_vec[4].parse().unwrap(),
    };

    Some(eval)
}
