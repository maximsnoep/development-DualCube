mod rw;
mod utils;

use mehsh::prelude::*;
use mehsh::utils::ids::IdMap;
use rw::*;
use std::{io, path::Path, process::Command};
use utils::*;

#[derive(Debug, Clone, Copy)]
struct TetNode {
    position: Vector3D,
    marker: i32,
}

#[derive(Debug, Clone)]
struct TetMeshData {
    nodes: Vec<TetNode>,
    tets: Vec<[usize; 4]>,
}

pub fn to_tet<M: Tag>(
    mesh: &Mesh<M>,
    path_windows: &Path,
    path_wsl: &Path,
    scale: f64,
) -> Result<IdMap<VERT, M>, io::Error> {
    let vtk_path = path_windows.with_extension("vtk");
    let smesh_path1 = path_windows.with_file_name("temp").with_extension("smesh");
    let smesh_path2 = wsl_path_arg(path_wsl.with_file_name("temp").with_extension("smesh"));

    let node_path = path_windows.with_file_name("temp.1.node");
    let ele_path = path_windows.with_file_name("temp.1.ele");

    remove_if_exists(&vtk_path)?;
    remove_if_exists(&node_path)?;
    remove_if_exists(&ele_path)?;

    let input_vertices = write_smesh(mesh, &smesh_path1)?;

    // TODO: Adapt this for other platforms / other ways of running TetGen
    let mut command = Command::new("wsl");
    command.arg("tetgen");
    command.arg(format!("-pYqa{}", tetgen_max_volume_from_mesh(mesh, scale)));
    command.arg(smesh_path2);

    info!("Running TetGen command: {:?}", command);
    let output = command.output()?;

    info!("TetGen finished (status={})", output.status);
    info!("stdout: {}", String::from_utf8_lossy(&output.stdout));
    warn!("stderr: {}", String::from_utf8_lossy(&output.stderr));
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "TetGen command failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        )));
    }

    info!(
        "Reading TetGen output ({}, {})",
        node_path.display(),
        ele_path.display()
    );
    let (nodes, raw_to_local) = read_node_file(&node_path)?;
    let tet_mesh = TetMeshData {
        tets: read_ele_file(&ele_path, &raw_to_local)?,
        nodes,
    };

    info!("Extracting boundary mesh from TetGen output");
    let mut tet_vertex_to_input_vertex = IdMap::<VERT, M>::new();
    for (i, node) in tet_mesh.nodes.iter().enumerate() {
        if let Some(&v) = input_vertices.get((-node.marker - 2) as usize) {
            tet_vertex_to_input_vertex.insert(i, v);
        }
    }

    info!("Writing the final VTK output");
    write_vtk(&vtk_path, &tet_mesh)?;

    info!("Done successfully");
    Ok(tet_vertex_to_input_vertex)
}

fn wsl_path_arg(path: impl AsRef<Path>) -> String {
    path.as_ref().to_string_lossy().replace('\\', "/")
}

fn tetgen_max_volume_from_mesh<M: Tag>(mesh: &Mesh<M>, scale: f64) -> f64 {
    let lengths = mesh
        .edge_ids_iter()
        .map(|edge| mesh.size(edge))
        .sorted_by(f64::total_cmp)
        .collect_vec();
    2.0_f64.sqrt() * (scale * lengths[lengths.len() / 2]).powi(3) / 12.0
}
