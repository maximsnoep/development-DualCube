//! Tetrahedralization of a closed surface. TetGen may refine the surface with boundary Steiner
//! points; [`conform`] adopts that surface as the input and remembers its volume.
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

/// The volume of `mesh` from an earlier [`conform`], else TetGen's, which must keep the surface.
pub fn tetrahedralize_with_options<M: Tag>(
    mesh: &Mesh<M>,
    options: &TetgenOptions,
) -> io::Result<TetrahedralMesh<M>> {
    if let Some(volume) = cache::lookup(mesh) {
        return Ok(volume);
    }
    let (input_vertices, data) = run(mesh, options, FLAGS)?;
    map_output(mesh, input_vertices, data)
}

/// The input surface refined by the boundary Steiner points TetGen needed.
#[derive(Clone, Debug)]
pub struct Surface {
    /// Input vertices first, in `mesh.vert_ids()` order, then the Steiner points.
    pub positions: Vec<Vector3D>,
    /// Triangles oriented like the input.
    pub faces: Vec<[usize; 3]>,
}

/// Tetrahedralize `mesh` once for all later [`tetrahedralize`] calls in this process. Returns
/// `None` when the volume keeps the input surface, else the refined surface: TetGen may split
/// input edges and faces, and the refined surface replaces the input. The volume is remembered
/// for that surface, since TetGen need not reproduce it from the refined surface.
/// Boundary splitting is allowed (no `-Y`): TetGen's surface is taken as it is.
pub fn conform<M: Tag>(mesh: &Mesh<M>, options: &TetgenOptions) -> io::Result<Option<Surface>> {
    if cache::lookup(mesh).is_some() {
        return Ok(None);
    }
    let (input_vertices, data) = run(mesh, options, FLAGS)?;
    let (vertices, tets, incidence) = order(mesh, input_vertices.clone(), data)?;
    let surface = boundary_surface(mesh, &input_vertices, &vertices, &tets, &incidence)?;
    cache::store(vertices, tets);
    if let Some(surface) = &surface {
        info!(
            "TetGen refined the input surface with {} boundary Steiner points",
            surface.positions.len() - input_vertices.len()
        );
    }
    Ok(surface)
}

/// The volume's boundary as a surface, or `None` if it is the input's.
fn boundary_surface<M: Tag>(
    mesh: &Mesh<M>,
    input_vertices: &[VertKey<M>],
    vertices: &[Vector3D],
    tets: &[[usize; 4]],
    incidence: &HashMap<[usize; 3], usize>,
) -> io::Result<Option<Surface>> {
    let n = input_vertices.len();
    // Boundary triangles, oriented away from their tet.
    let mut faces = vec![];
    for t in tets {
        let positive = volume(vertices, *t) > 0.;
        for [i, j, k] in [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]] {
            let mut key = [t[i], t[j], t[k]];
            key.sort();
            if incidence[&key] == 1 {
                faces.push(if positive {
                    [t[i], t[j], t[k]]
                } else {
                    [t[i], t[k], t[j]]
                });
            }
        }
    }
    // Same orientation as the input faces TetGen kept.
    let index: HashMap<_, _> = input_vertices
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();
    let input: HashSet<[usize; 3]> = mesh
        .face_ids()
        .into_iter()
        .filter_map(|f| {
            let [a, b, c] = mesh.vertices(f).map(|v| index[&v]).collect_array::<3>()?;
            Some(rotated([a, b, c]))
        })
        .collect();
    let (mut same, mut flipped) = (0, 0);
    for &[a, b, c] in &faces {
        if input.contains(&rotated([a, b, c])) {
            same += 1;
        } else if input.contains(&rotated([a, c, b])) {
            flipped += 1;
        }
    }
    if flipped > same {
        for face in &mut faces {
            face.swap(1, 2);
        }
    }
    // Surface vertices: the input ones keep their index, Steiner points follow.
    let mut on = vec![false; vertices.len()];
    for &v in faces.iter().flatten() {
        on[v] = true;
    }
    if on[..n].iter().any(|&on| !on) {
        return Err(io::Error::other("TetGen moved an input vertex off the boundary"));
    }
    let mut surface = vec![usize::MAX; vertices.len()];
    let mut positions = vec![];
    for (v, _) in on.iter().enumerate().filter(|(_, on)| **on) {
        surface[v] = positions.len();
        positions.push(vertices[v]);
    }
    let faces: Vec<_> = faces.iter().map(|f| f.map(|v| surface[v])).collect();
    let unchanged = positions.len() == n
        && faces.len() == input.len()
        && faces.iter().all(|&f| input.contains(&rotated(f)));
    Ok((!unchanged).then_some(Surface { positions, faces }))
}

/// A triangle's rotation starting at its smallest vertex, so equal oriented triangles compare equal.
fn rotated(t: [usize; 3]) -> [usize; 3] {
    let i = (0..3).min_by_key(|&i| t[i]).unwrap();
    [t[i], t[(i + 1) % 3], t[(i + 2) % 3]]
}

fn volume(vertices: &[Vector3D], [a, b, c, d]: [usize; 4]) -> f64 {
    let [a, b, c, d] = [a, b, c, d].map(|i| vertices[i]);
    (b - a).cross(&(c - a)).dot(&(d - a))
}

/// Piecewise linear complex, no facet merging, quiet; boundary Steiner points allowed.
const FLAGS: &str = "-pMQ";

/// Run TetGen with `flags` on `mesh` and read the result, input vertices marked.
fn run<M: Tag>(
    mesh: &Mesh<M>,
    options: &TetgenOptions,
    flags: &str,
) -> io::Result<(Vec<VertKey<M>>, TetMeshData)> {
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
    let tetgen = |flags: &str| -> io::Result<std::process::Output> {
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
            path = input.clone();
            command = Command::new(&options.executable);
        }
        command.arg(flags).arg(path).output().map_err(|e| {
            io::Error::new(
                e.kind(),
                format!(
                    "Could not launch TetGen ({}): {e}",
                    options.executable.display()
                ),
            )
        })
    };
    let output = tetgen(flags)?;
    if !output.status.success() {
        // Exit code 3 is TetGen's self-intersection stop; it may also crash on intersecting
        // facets, which its detection mode reports before crashing too.
        let intersects = output.status.code() == Some(3)
            || tetgen("-pd").is_ok_and(|o| {
                String::from_utf8_lossy(&o.stdout).contains("intersect")
            });
        if intersects {
            return Err(io::Error::other(format!(
                "The input surface self-intersects (TetGen {flags}: {})",
                output.status
            )));
        }
        return Err(io::Error::other(format!(
            "TetGen failed ({}):\n{}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    let (nodes, raw_to_local) = read_node_file(&work.0.join("surface.1.node"))?;
    let tets = read_ele_file(&work.0.join("surface.1.ele"), &raw_to_local)?;
    Ok((input_vertices, TetMeshData { nodes, tets }))
}

fn map_output<M: Tag>(
    mesh: &Mesh<M>,
    input_vertices: Vec<VertKey<M>>,
    data: TetMeshData,
) -> io::Result<TetrahedralMesh<M>> {
    let (vertices, tets, incidence) = order(mesh, input_vertices.clone(), data)?;
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
    let boundary = |incidence: HashMap<[usize; 3], usize>| -> HashSet<[usize; 3]> {
        incidence
            .into_iter()
            .filter_map(|(face, count)| (count == 1).then_some(face))
            .collect()
    };
    if boundary(incidence) != expected {
        return Err(io::Error::other(
            "TetGen refined the input surface; use the surface from `conform` as the input",
        ));
    }
    Ok(TetrahedralMesh {
        vertices,
        tets,
        input_vertices,
    })
}

/// Input vertices first, validated cells, and how many cells contain each (sorted) triangle.
type Ordered = (Vec<Vector3D>, Vec<[usize; 4]>, HashMap<[usize; 3], usize>);

fn order<M: Tag>(
    mesh: &Mesh<M>,
    input_vertices: Vec<VertKey<M>>,
    data: TetMeshData,
) -> io::Result<Ordered> {
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
    Ok((vertices, tets, incidence))
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

mod cache;

#[cfg(test)]
mod tests;
