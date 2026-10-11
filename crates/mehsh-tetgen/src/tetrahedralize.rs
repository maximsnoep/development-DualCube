//! Tetrahedralization with an unchanged surface boundary.
//!
//! Uses `tetgen` on PATH, or `TETGEN_EXECUTABLE`. On Windows, commands run
//! through WSL unless the executable has an `.exe` extension. Explicit options
//! can override this choice. Temporary input/output files are removed on return.

use crate::{
    TetMeshData,
    rw::{read_ele_file, read_node_file, write_smesh},
};
use mehsh::prelude::*;
use std::{
    io,
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

#[derive(Clone, Debug)]
pub struct TetgenOptions {
    /// TetGen executable, interpreted inside WSL when `use_wsl` is set.
    pub executable: PathBuf,
    pub use_wsl: bool,
}

impl Default for TetgenOptions {
    fn default() -> Self {
        let executable = std::env::var_os("TETGEN_EXECUTABLE")
            .map(PathBuf::from)
            .unwrap_or_else(|| "tetgen".into());
        let use_wsl = cfg!(windows) && executable.extension().is_none_or(|ext| ext != "exe");
        Self {
            executable,
            use_wsl,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TetrahedralMesh<M: Tag> {
    /// Original vertices first, followed by any interior Steiner vertices.
    pub vertices: Vec<Vector3D>,
    pub tets: Vec<[usize; 4]>,
    /// `vertices[i]` corresponds to `input_vertices[i]` for every input vertex.
    pub input_vertices: Vec<VertKey<M>>,
}

pub fn tetrahedralize<M: Tag>(mesh: &Mesh<M>) -> io::Result<TetrahedralMesh<M>> {
    tetrahedralize_with_options(mesh, &TetgenOptions::default())
}

/// Run TetGen with boundary splitting disabled and verify the returned boundary.
pub fn tetrahedralize_with_options<M: Tag>(
    mesh: &Mesh<M>,
    options: &TetgenOptions,
) -> io::Result<TetrahedralMesh<M>> {
    if mesh.nr_verts() == 0 || mesh.face_ids().is_empty() {
        return Err(io::Error::other("Cannot tetrahedralize an empty mesh"));
    }
    if mesh
        .vert_ids()
        .iter()
        .any(|&v| !mesh.position(v).iter().all(|x| x.is_finite()))
    {
        return Err(io::Error::other(
            "TetGen input contains non-finite positions",
        ));
    }
    let work = TempDirectory::new()?;
    let input = work.0.join("surface.smesh");
    let input_vertices = write_smesh(mesh, &input)?;
    let mut command;
    let path;
    if options.use_wsl {
        let converted = Command::new("wsl.exe")
            .args(["--exec", "wslpath", "-u"])
            .arg(&input)
            .output()?;
        if !converted.status.success() {
            return Err(io::Error::other(format!(
                "WSL path conversion failed: {}",
                String::from_utf8_lossy(&converted.stderr)
            )));
        }
        path = PathBuf::from(String::from_utf8_lossy(&converted.stdout).trim());
        command = Command::new("wsl.exe");
        command.arg("--exec").arg(&options.executable);
    } else {
        path = input;
        command = Command::new(&options.executable);
    }
    let output = command.arg("-pMYQ").arg(path).output().map_err(|e| {
        io::Error::new(
            e.kind(),
            format!(
                "Could not launch TetGen ({}): {e}",
                options.executable.display()
            ),
        )
    })?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "TetGen failed ({}):\n{}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    let (nodes, raw_to_local) = read_node_file(&work.0.join("surface.1.node"))?;
    let tets = read_ele_file(&work.0.join("surface.1.ele"), &raw_to_local)?;
    map_output(mesh, input_vertices, TetMeshData { nodes, tets })
}

fn map_output<M: Tag>(
    mesh: &Mesh<M>,
    input_vertices: Vec<VertKey<M>>,
    data: TetMeshData,
) -> io::Result<TetrahedralMesh<M>> {
    let n = input_vertices.len();
    let mut original = vec![None; n];
    for (i, node) in data.nodes.iter().enumerate() {
        if !node.position.iter().all(|x| x.is_finite()) {
            return Err(io::Error::other("TetGen returned non-finite positions"));
        }
        let index = -(i64::from(node.marker)) - 2;
        if index >= 0 {
            let index = index as usize;
            if index >= n || original[index].replace(i).is_some() {
                return Err(io::Error::other(
                    "TetGen returned invalid or duplicate input markers",
                ));
            }
            let expected = mesh.position(input_vertices[index]);
            if (node.position - expected).norm() > 1e-12 * expected.norm().max(1.0) {
                return Err(io::Error::other("TetGen moved an input vertex"));
            }
        }
    }
    let mut order: Vec<usize> = original
        .into_iter()
        .collect::<Option<_>>()
        .ok_or_else(|| io::Error::other("TetGen omitted an input vertex"))?;
    let input_set: HashSet<_> = order.iter().copied().collect();
    order.extend((0..data.nodes.len()).filter(|i| !input_set.contains(i)));
    let mut remap = vec![0; order.len()];
    for (new, &old) in order.iter().enumerate() {
        remap[old] = new;
    }
    let vertices: Vec<_> = order.iter().map(|&i| data.nodes[i].position).collect();
    let mut tets = Vec::new();
    let mut tet_set = HashSet::new();
    let mut incidence = HashMap::<[usize; 3], usize>::new();
    for tet in data.tets {
        if tet.iter().any(|&i| i >= remap.len()) {
            return Err(io::Error::other("TetGen cell has an invalid vertex"));
        }
        let t = tet.map(|i| remap[i]);
        let mut key = t;
        key.sort();
        let [a, b, c, d] = t.map(|i| vertices[i]);
        let volume = (b - a).cross(&(c - a)).dot(&(d - a));
        if key.windows(2).any(|w| w[0] == w[1])
            || !tet_set.insert(key)
            || !volume.is_finite()
            || volume == 0.0
        {
            return Err(io::Error::other(
                "TetGen returned a duplicate or degenerate cell",
            ));
        }
        for [i, j, k] in [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]] {
            let mut face = [t[i], t[j], t[k]];
            face.sort();
            *incidence.entry(face).or_default() += 1;
        }
        tets.push(t);
    }
    if tets.is_empty() || incidence.values().any(|&count| count > 2) {
        return Err(io::Error::other(
            "TetGen returned an empty or non-manifold volume",
        ));
    }
    let indices: HashMap<_, _> = input_vertices
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();
    let expected: HashSet<_> = mesh
        .face_ids()
        .into_iter()
        .map(|f| {
            let mut tri: Vec<_> = mesh.vertices(f).map(|v| indices[&v]).collect();
            tri.sort();
            [tri[0], tri[1], tri[2]]
        })
        .collect();
    let boundary: HashSet<_> = incidence
        .into_iter()
        .filter_map(|(face, count)| (count == 1).then_some(face))
        .collect();
    if boundary != expected {
        return Err(io::Error::other(
            "TetGen changed the input surface triangulation",
        ));
    }
    Ok(TetrahedralMesh {
        vertices,
        tets,
        input_vertices,
    })
}

struct TempDirectory(PathBuf);
impl TempDirectory {
    fn new() -> io::Result<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let path = std::env::temp_dir().join(format!(
                "dualcube-tetgen-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match std::fs::create_dir(&path) {
                Ok(()) => return Ok(Self(path)),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
    }
}
impl Drop for TempDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[cfg(test)]
mod tests;
