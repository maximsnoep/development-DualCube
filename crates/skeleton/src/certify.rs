//! Certify that every patch is contractible in the enclosed volume.
//!
//! The boundary loops off a spanning tree of the skeleton are g disjoint loops whose
//! complement on the surface is connected. If each bounds a disk in the volume, cutting along
//! these disks leaves a ball, and every patch lies on its boundary. So g disks certify the
//! whole skeleton. A loop without a disk joins the tree and others are tried instead; when the
//! loops without disks close a cycle, no tree avoids them and certification fails.
//!
//! Disks are found by DiskScissors (`diskscissors` on PATH, or `DISKSCISSORS_EXECUTABLE`) in a
//! TetGen tetrahedralization of the volume. A loop is given by the vertex walk beside it when
//! that walk is simple; otherwise the volume is refined along the loop ([`refine`]), where it is
//! itself a simple vertex cycle. Most disks are found within seconds, so loops are
//! first given a short time; loops that ran out of time are retried with longer ones only when
//! no spanning tree avoids them.

mod refine;

use crate::{Skeleton, SkeletonEdgeID, SkeletonResult};
use dualcube_types::prelude::*;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Clone, Debug)]
pub struct Options {
    /// DiskScissors executable, interpreted inside WSL when `use_wsl` is set.
    pub executable: PathBuf,
    /// Time per loop for each round; a loop without a disk may keep DiskScissors searching.
    pub timeouts: Vec<std::time::Duration>,
    /// Run through WSL, with paths converted. As for TetGen: the default on Windows unless
    /// the executable has an `.exe` extension.
    pub use_wsl: bool,
}

impl Default for Options {
    fn default() -> Self {
        let executable: PathBuf = std::env::var_os("DISKSCISSORS_EXECUTABLE")
            .map_or_else(|| "diskscissors".into(), PathBuf::from);
        let use_wsl = cfg!(windows) && executable.extension().is_none_or(|ext| ext != "exe");
        Self {
            executable,
            timeouts: [10, 60, 300].map(std::time::Duration::from_secs).to_vec(),
            use_wsl,
        }
    }
}

/// Why a loop has no disk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Miss {
    /// DiskScissors found no disk.
    NoDisk,
    /// DiskScissors did not finish within the timeout.
    Timeout,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Failure {
    /// The loops without disks close a cycle in the skeleton, so every spanning tree leaves
    /// one of them to certify.
    NoTree { misses: Vec<(SkeletonEdgeID, Miss)> },
    /// TetGen or DiskScissors could not run.
    Unavailable(String),
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoTree { misses } => write!(
                f,
                "Cannot certify the skeleton: {} boundary loops without disks close a cycle ({misses:?})",
                misses.len()
            ),
            Self::Unavailable(reason) => write!(f, "Cannot certify the skeleton: {reason}"),
        }
    }
}

impl std::error::Error for Failure {}

/// The loops whose disks certify the skeleton, and the ones that got none.
#[derive(Clone, Debug, Default)]
pub struct Certificate {
    pub tree: Vec<SkeletonEdgeID>,
    pub disks: Vec<SkeletonEdgeID>,
    pub misses: Vec<(SkeletonEdgeID, Miss)>,
}

pub fn run(mesh: &Mesh<INPUT>, skeleton: &Skeleton) -> Result<Certificate, Failure> {
    run_with(mesh, skeleton, &Options::default())
}

pub fn run_with(
    mesh: &Mesh<INPUT>,
    skeleton: &Skeleton,
    options: &Options,
) -> Result<Certificate, Failure> {
    let mut certificate = Certificate::default();
    let mut volume = None;
    let mut round = 0;
    loop {
        let forced: Vec<_> = certificate.misses.iter().map(|m| m.0).collect();
        let Some(tree) = spanning_tree(mesh, skeleton, &forced) else {
            if round + 1 < options.timeouts.len()
                && certificate.misses.iter().any(|m| m.1 == Miss::Timeout)
            {
                round += 1;
                certificate.misses.retain(|m| m.1 == Miss::NoDisk);
                tracing::info!(
                    "Certification: retrying loops that ran out of time with {:?}",
                    options.timeouts[round]
                );
                continue;
            }
            return Err(Failure::NoTree {
                misses: certificate.misses,
            });
        };
        let pending: Vec<_> = skeleton
            .edges
            .keys()
            .filter(|e| !tree.contains(e) && !certificate.disks.contains(e))
            .collect();
        let mut missed = false;
        for edge in pending {
            if volume.is_none() {
                tracing::info!(
                    "Certification: {} loops beside a spanning tree",
                    skeleton.edges.len() + 1 - skeleton.nodes.len()
                );
                volume = Some(Volume::new(mesh).map_err(|e| Failure::Unavailable(e.to_string()))?);
            }
            let volume = volume.as_ref().unwrap();
            let found = volume
                .certify(mesh, skeleton, edge, options, options.timeouts[round])
                .map_err(|e| Failure::Unavailable(e.to_string()))?;
            match found {
                Ok(()) => certificate.disks.push(edge),
                Err(miss) => {
                    tracing::info!("Certification: {edge:?} has no disk ({miss:?}); retrying");
                    certificate.misses.push((edge, miss));
                    missed = true;
                    break;
                }
            }
        }
        if !missed {
            certificate.tree = tree;
            tracing::info!(
                "Certification: {} disks, {} loops without disks",
                certificate.disks.len(),
                certificate.misses.len()
            );
            return Ok(certificate);
        }
    }
}

/// A spanning tree containing `forced`, otherwise preferring long loops, so the loops left to
/// certify are short. `None` when `forced` closes a cycle.
pub fn spanning_tree(
    mesh: &Mesh<INPUT>,
    skeleton: &Skeleton,
    forced: &[SkeletonEdgeID],
) -> Option<Vec<SkeletonEdgeID>> {
    let index: HashMap<_, _> = skeleton
        .nodes
        .keys()
        .enumerate()
        .map(|(i, n)| (n, i))
        .collect();
    let mut parent: Vec<_> = (0..index.len()).collect();
    fn root(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    let length = |e: SkeletonEdgeID| loop_length(mesh, skeleton, e);
    let mut rest: Vec<_> = skeleton
        .edges
        .keys()
        .filter(|e| !forced.contains(e))
        .collect();
    rest.sort_by(|&a, &b| length(b).total_cmp(&length(a)).then(a.cmp(&b)));
    let mut tree = Vec::new();
    for (i, &edge) in forced.iter().chain(&rest).enumerate() {
        let [a, b] = skeleton.edges[edge]
            .nodes
            .map(|n| root(&mut parent, index[&n]));
        if a == b {
            if i < forced.len() {
                return None;
            }
            continue;
        }
        parent[a] = b;
        tree.push(edge);
    }
    Some(tree)
}

fn loop_length(mesh: &Mesh<INPUT>, skeleton: &Skeleton, edge: SkeletonEdgeID) -> f64 {
    let points: Vec<_> = skeleton.edges[edge]
        .boundary
        .crossings
        .iter()
        .map(|&c| {
            let exit = skeleton.surface.crossings[c].exit;
            (mesh.position(mesh.root(exit)) + mesh.position(mesh.toor(exit))) / 2.
        })
        .collect();
    (0..points.len())
        .map(|i| (points[(i + 1) % points.len()] - points[i]).norm())
        .sum()
}

/// A simple vertex walk beside a boundary loop, if either side has one.
///
/// Consecutive crossings share a triangle, and the uncrossed vertices of that triangle on one
/// side are equal or joined by an uncrossed edge. So each side's walk stays in a strip beside
/// the loop and is homotopic to it; a simple one is then isotopic to it on the surface.
pub fn walk(mesh: &Mesh<INPUT>, skeleton: &Skeleton, edge: SkeletonEdgeID) -> Option<Vec<VertID>> {
    let exits: Vec<_> = skeleton.edges[edge]
        .boundary
        .crossings
        .iter()
        .map(|&c| skeleton.surface.crossings[c].exit)
        .collect();
    [
        exits.iter().map(|&e| mesh.toor(e)).collect::<Vec<_>>(),
        exits.iter().map(|&e| mesh.root(e)).collect(),
    ]
    .into_iter()
    .map(reduce)
    .find(|walk| walk.len() >= 3 && walk.iter().collect::<HashSet<_>>().len() == walk.len())
}

/// The closed walk without repeated vertices in a row and without backtracks (`a, b, a`),
/// which leaves its homotopy class unchanged.
fn reduce(walk: Vec<VertID>) -> Vec<VertID> {
    let mut result: Vec<VertID> = Vec::new();
    for v in walk {
        result.push(v);
        loop {
            let n = result.len();
            if n >= 2 && result[n - 1] == result[n - 2] {
                result.pop();
            } else if n >= 3 && result[n - 1] == result[n - 3] {
                result.truncate(n - 2);
            } else {
                break;
            }
        }
    }
    // Close the cycle: the same reductions across the seam.
    loop {
        let n = result.len();
        if n >= 2 && result[0] == result[n - 1] {
            result.pop();
        } else if n >= 3 && result[1] == result[n - 1] {
            result.pop();
            result.remove(0);
        } else if n >= 3 && result[0] == result[n - 2] {
            result.truncate(n - 2);
        } else {
            break;
        }
    }
    result
}

/// A tetrahedralization of the enclosed volume, written once for DiskScissors.
struct Volume {
    directory: PathBuf,
    mesh: PathBuf,
    /// Volume vertex of each surface vertex.
    index: HashMap<VertID, usize>,
    vertices: Vec<Vector3D>,
    tets: Vec<[usize; 4]>,
    /// `directory` as seen from WSL, converted on first use.
    wsl_directory: std::sync::OnceLock<String>,
}

impl Volume {
    fn new(mesh: &Mesh<INPUT>) -> SkeletonResult<Self> {
        let tets = mehsh_tetgen::tetrahedralize(mesh)?;
        let index = tets
            .input_vertices
            .iter()
            .enumerate()
            .map(|(i, &v)| (v, i))
            .collect();
        let directory = scratch_directory()?;
        let path = directory.join("volume.vtk");
        write_vtk(&path, &tets.vertices, &tets.tets)?;
        Ok(Self {
            directory,
            mesh: path,
            index,
            vertices: tets.vertices,
            tets: tets.tets,
            wsl_directory: std::sync::OnceLock::new(),
        })
    }

    /// Whether a boundary loop bounds a disk: through the walk beside it when simple, else in
    /// the volume refined along it.
    fn certify(
        &self,
        mesh: &Mesh<INPUT>,
        skeleton: &Skeleton,
        edge: SkeletonEdgeID,
        options: &Options,
        timeout: std::time::Duration,
    ) -> SkeletonResult<Result<(), Miss>> {
        if let Some(walk) = walk(mesh, skeleton, edge) {
            let walk: Vec<_> = walk.iter().map(|v| self.index[v]).collect();
            return self.disk(&self.mesh, options, timeout, &walk);
        }
        let boundary = &skeleton.edges[edge].boundary;
        let refined = refine::refine(
            mesh,
            &skeleton.surface,
            &[boundary],
            &self.index,
            &self.vertices,
            &self.tets,
        );
        let path = self.directory.join("refined.vtk");
        write_vtk(&path, &refined.vertices, &refined.tets)?;
        let walk: Vec<_> = boundary
            .crossings
            .iter()
            .map(|c| refined.crossings[c])
            .collect();
        self.disk(&path, options, timeout, &walk)
    }

    /// Whether DiskScissors finds a disk bounded by the walk. It reports success only by
    /// printing `Snip`; its exit status is zero either way.
    fn disk(
        &self,
        volume: &Path,
        options: &Options,
        timeout: std::time::Duration,
        walk: &[usize],
    ) -> SkeletonResult<Result<(), Miss>> {
        let path = self.directory.join("loop.txt");
        let log = self.directory.join("output.txt");
        let text: Vec<_> = walk.iter().map(usize::to_string).collect();
        std::fs::write(&path, text.join("\n") + "\n")?;
        let mut command;
        if options.use_wsl {
            let directory = match self.wsl_directory.get() {
                Some(directory) => directory,
                None => {
                    let converted = wsl_path(&self.directory)?;
                    self.wsl_directory.get_or_init(|| converted)
                }
            };
            // Joined by hand: `Path::join` would insert a Windows separator.
            command = Command::new("wsl.exe");
            command
                .arg("--exec")
                .arg(&options.executable)
                .arg(format!("{directory}/{}", file_name(volume)?))
                .arg(format!("{directory}/{}", file_name(&path)?));
        } else {
            command = Command::new(&options.executable);
            command.arg(volume).arg(&path);
        }
        let mut child = command
            .stdout(std::fs::File::create(&log)?)
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| format!("running {}: {e}", options.executable.display()))?;
        let start = std::time::Instant::now();
        let status = loop {
            if let Some(status) = child.try_wait()? {
                break status;
            }
            if start.elapsed() > timeout {
                child.kill()?;
                child.wait()?;
                return Ok(Err(Miss::Timeout));
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        };
        // A nonzero status means it did not run (e.g. WSL could not find it), not "no disk".
        if !status.success() {
            return Err(format!("running {}: {status}", options.executable.display()).into());
        }
        let output = std::fs::read_to_string(&log)?;
        Ok(if output.lines().any(|l| l.trim() == "Snip") {
            Ok(())
        } else {
            Err(Miss::NoDisk)
        })
    }
}

impl Drop for Volume {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

/// The final component of a path in the scratch directory, for joining onto its WSL form.
fn file_name(path: &Path) -> SkeletonResult<&str> {
    path.file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("no file name in {}", path.display()).into())
}

/// The path as seen from WSL.
fn wsl_path(path: &Path) -> SkeletonResult<String> {
    let converted = Command::new("wsl.exe")
        .args(["--exec", "wslpath", "-u"])
        .arg(path)
        .output()
        .map_err(|e| format!("running wslpath: {e}"))?;
    if !converted.status.success() {
        return Err(format!(
            "WSL path conversion failed: {}",
            String::from_utf8_lossy(&converted.stderr)
        )
        .into());
    }
    Ok(String::from_utf8_lossy(&converted.stdout).trim().to_owned())
}

fn scratch_directory() -> std::io::Result<PathBuf> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let directory = std::env::temp_dir().join(format!(
        "dualcube-certify-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&directory)?;
    Ok(directory)
}

/// Legacy ASCII VTK unstructured grid of tetrahedra.
fn write_vtk(path: &Path, vertices: &[Vector3D], tets: &[[usize; 4]]) -> std::io::Result<()> {
    use std::fmt::Write;
    let mut text =
        String::from("# vtk DataFile Version 2.0\ntets\nASCII\nDATASET UNSTRUCTURED_GRID\n");
    let _ = writeln!(text, "POINTS {} double", vertices.len());
    for p in vertices {
        let _ = writeln!(text, "{} {} {}", p.x, p.y, p.z);
    }
    let _ = writeln!(text, "CELLS {} {}", tets.len(), tets.len() * 5);
    for t in tets {
        let _ = writeln!(text, "4 {} {} {} {}", t[0], t[1], t[2], t[3]);
    }
    let _ = writeln!(text, "CELL_TYPES {}", tets.len());
    for _ in tets {
        text.push_str("10\n");
    }
    std::fs::write(path, text)
}

#[cfg(test)]
mod tests;
