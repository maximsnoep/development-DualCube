use dualcube::prelude::*;
use mehsh_tetgen;
use std::{
    fs,
    io::{self, ErrorKind},
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Clone, Debug, Default)]
pub struct Hex {}

/// One requested DiskScissors cut.
#[derive(Clone, Debug)]
pub struct DiskScissorsCutSpec {
    /// Used as filename prefix, e.g. `"cut_x_000"`.
    pub name: String,

    /// Loop vertex indices in the exported tetrahedral mesh.
    ///
    /// DiskScissors expects one vertex index per line.
    pub loop_vertices: Vec<usize>,
}

/// Output of one DiskScissors cut.
#[derive(Clone, Debug)]
pub struct DiskScissorsCutOutput {
    pub name: String,
    pub refined_mesh_path: PathBuf,
    pub triangle_indices_path: PathBuf,

    /// Triangles of the cutting disk.
    ///
    /// These indices refer to vertices of the refined tetrahedral mesh written by DiskScissors.
    pub cut_triangles: Vec<[usize; 3]>,

    pub stdout: String,
    pub stderr: String,
}

impl Hex {
    pub fn from_solution(solution: &Solution) -> io::Result<Self> {
        info!("building Hex from solution");
        let Some(layout) = &solution.layout else {
            return Err(io::Error::other("cannot build Hex: no layout available"));
        };

        let mesh = &layout.granulated_mesh;

        if mesh.vert_ids().is_empty() {
            return Err(io::Error::other("cannot build Hex: input mesh is empty"));
        }

        let job_dir1 = job_dir("DISK_SCISSORS_JOB_DIR_WINDOWS")?;
        let job_dir2 = job_dir("DISK_SCISSORS_JOB_DIR_WSL")?;

        info!("creating job directory at {}", job_dir1);
        fs::create_dir_all(&job_dir1)?;

        let tet_mesh_filename = "temp.vtk";
        let tet_mesh_path1 = PathBuf::from(job_dir1).join(tet_mesh_filename);
        let tet_mesh_path2 = PathBuf::from(job_dir2).join(tet_mesh_filename);

        info!("exporting TetGen/VTK mesh to {}", tet_mesh_path2.display());

        // This now:
        // 1. writes a TetGen .smesh
        // 2. runs command-line TetGen
        // 3. reads .node/.ele
        // 4. writes temp.vtk
        // 5. extracts the TetGen boundary mesh
        let tet_export = mehsh_tetgen::to_tet(mesh, &tet_mesh_path1, &tet_mesh_path2, 4.0)?;
        info!(
            "exported TetGen/VTK mesh: input_vertices={}",
            mesh.vert_ids().len()
        );

        let Some(polycube) = &solution.polycube else {
            return Err(io::Error::other("cannot build Hex: no polycube available"));
        };

        // Enumerate primal loops.
        let mut edge_to_loops = HashMap::new();

        for edge in polycube.structure.edge_ids_iter() {
            let mut loop_edges = Vec::new();
            loop_edges.push(edge);

            loop {
                let last_edge = *loop_edges.last().unwrap();

                let next_forward_edge = polycube
                    .structure
                    .next(polycube.structure.twin(polycube.structure.next(last_edge)));

                if next_forward_edge == loop_edges[0] {
                    break;
                }

                loop_edges.push(next_forward_edge);
            }

            edge_to_loops.insert(edge, loop_edges);
        }

        // Pick a primal loop that is not on the boundary.
        let mut non_boundary_loop = None;

        for (&loop_seed, loop_edges) in &edge_to_loops {
            for &edge in loop_edges {
                let vertices = polycube.structure.vertices(edge);

                let Some((v1, v2)) = vertices.collect_tuple() else {
                    continue;
                };

                // Your current heuristic:
                // both incident vertices must have degree 4.
                if polycube.structure.degree(v1) != 4 || polycube.structure.degree(v2) != 4 {
                    continue;
                }

                non_boundary_loop = Some(loop_seed);
                break;
            }

            if non_boundary_loop.is_some() {
                break;
            }
        }

        let Some(loop_seed) = non_boundary_loop else {
            return Err(io::Error::other(
                "cannot build Hex: could not find non-boundary primal loop",
            ));
        };

        // Turn the non-boundary primal loop into mesh vertices.
        let loop_edges = edge_to_loops
            .get(&loop_seed)
            .ok_or_else(|| io::Error::other("missing loop for selected loop seed"))?;

        let mut mesh_loop_vertices = Vec::new();

        for &edge in loop_edges {
            let Some(path) = layout.edge_to_path.get(&edge) else {
                return Err(io::Error::other(format!(
                    "cannot build Hex: missing mesh path for polycube edge {edge:?}",
                )));
            };

            if path.len() < 2 {
                return Err(io::Error::other(format!(
                    "cannot build Hex: mesh path for polycube edge {edge:?} is too short",
                )));
            }

            // Remove the last vertex, otherwise adjacent paths duplicate vertices.
            mesh_loop_vertices.extend(path.iter().copied().take(path.len() - 1));
        }

        if mesh_loop_vertices.is_empty() {
            return Err(io::Error::other(
                "cannot build Hex: selected loop contains no vertices",
            ));
        }

        // Convert vertices from `layout.granulated_mesh` to TetGen/VTK vertex ids.
        //
        // Since TetGen is run with -Y, the original surface vertices should be preserved.
        // Therefore this map should contain every vertex used by `layout.edge_to_path`.
        let disk_loop_vertices: Vec<usize> = mesh_loop_vertices
            .iter()
            .copied()
            .map(|v| {
                tet_export.id(&v).copied().ok_or_else(|| {
                    io::Error::other(format!(
                        "cannot build Hex: loop vertex {v:?} has no TetGen/VTK vertex id"
                    ))
                })
            })
            .collect::<io::Result<_>>()?;

        info!(
            "selected DiskScissors cut loop: seed={loop_seed:?} vertices={}",
            disk_loop_vertices.len()
        );

        let cut = DiskScissorsCutSpec {
            name: "cut".to_string(),
            loop_vertices: disk_loop_vertices,
        };

        let output = run_disk_scissors_cut(&tet_mesh_path2, &cut).map_err(|err| {
            io::Error::other(format!("DiskScissors failed while building Hex: {err}"))
        })?;

        let refined_positions = read_vtk_points(&output.refined_mesh_path)?;
        let stitched_cut = mesh
            .cut_and_cap_from_positions(
                &mesh_loop_vertices,
                &refined_positions,
                &output.cut_triangles,
            )
            .map_err(|err| {
                io::Error::other(format!(
                    "cannot build Hex: failed to stitch DiskScissors cut '{}': {err:?}",
                    output.name
                ))
            })?;

        info!(
            "stitched DiskScissors cut '{}': mesh_a_faces={} mesh_b_faces={}",
            output.name,
            stitched_cut.mesh_a.nr_faces(),
            stitched_cut.mesh_b.nr_faces()
        );

        let mesh1 = stitched_cut.mesh_a;
        let mesh2 = stitched_cut.mesh_b;

        let out_dir = PathBuf::from(job_dir("DISK_SCISSORS_JOB_DIR_WINDOWS")?);
        mesh1.to_obj(&out_dir.join(format!("{}_mesh_a.obj", output.name)))?;
        mesh2.to_obj(&out_dir.join(format!("{}_mesh_b.obj", output.name)))?;

        println!("stdout: {}", output.stdout);
        println!("stderr: {}", output.stderr);

        Ok(Self {})
    }
}

/// Reads a required job-directory environment variable, as an `io::Error` instead of a panic.
fn job_dir(var: &str) -> io::Result<String> {
    std::env::var(var).map_err(|e| io::Error::other(format!("environment variable {var}: {e}")))
}

/// Run one DiskScissors cut on an already exported VTK tet mesh.
fn run_disk_scissors_cut(
    tet_mesh_path: &Path,
    cut: &DiskScissorsCutSpec,
) -> io::Result<DiskScissorsCutOutput> {
    let loop_filename = format!("{}_loop.txt", cut.name);
    let tet_mesh_refined_filename = format!("{}_tet_mesh_refined.vtk", cut.name);
    let disk_triangles_filename = format!("{}_disk_triangles.txt", cut.name);

    let job_dir1 = job_dir("DISK_SCISSORS_JOB_DIR_WINDOWS")?;
    let job_dir2 = job_dir("DISK_SCISSORS_JOB_DIR_WSL")?;

    let loop_path1 = PathBuf::from(job_dir1.clone()).join(&loop_filename);
    let loop_path2 = PathBuf::from(job_dir2.clone()).join(&loop_filename);
    let tet_mesh_refined_path1 = PathBuf::from(job_dir1.clone()).join(&tet_mesh_refined_filename);
    let tet_mesh_refined_path2 = PathBuf::from(job_dir2.clone()).join(&tet_mesh_refined_filename);
    let disk_triangle_path1 = PathBuf::from(job_dir1.clone()).join(&disk_triangles_filename);
    let disk_triangle_path2 = PathBuf::from(job_dir2.clone()).join(&disk_triangles_filename);

    info!(
        "DiskScissors cut '{}': tet_mesh={} loop_file={} refined_output={} triangles_output={}",
        cut.name,
        tet_mesh_path.display(),
        loop_path1.display(),
        tet_mesh_refined_path2.display(),
        disk_triangle_path2.display(),
    );

    write_loop_file(&loop_path1, &cut.loop_vertices)?;
    info!(
        "wrote DiskScissors loop file: path={} vertices={}",
        loop_path1.display(),
        cut.loop_vertices.len()
    );

    let mut command = Command::new("wsl".to_string());
    command.arg("/home/snoep/DiskScissors/build/cut");
    command.arg(&tet_mesh_path);
    command.arg(&loop_path2);
    command.arg(&tet_mesh_refined_path2);
    command.arg(&disk_triangle_path2);

    info!(
        "Running DiskScissors command without path conversion: {:?}",
        command
    );

    let output = command.output()?;

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();

    info!(
        "DiskScissors finished: cut='{}' status={} stdout_bytes={} stderr_bytes={}",
        cut.name,
        output.status,
        output.stdout.len(),
        output.stderr.len()
    );

    if !output.status.success() {
        return Err(io::Error::other(format!(
            "DiskScissors failed for cut '{}'\nstdout:\n{}\nstderr:\n{}",
            cut.name, stdout, stderr
        )));
    }

    if !stdout.contains("Snip") {
        return Err(io::Error::other(format!(
            "DiskScissors did not report success for cut '{}'\nstdout:\n{}\nstderr:\n{}",
            cut.name, stdout, stderr
        )));
    }

    let cut_triangles = read_triangle_indices(&disk_triangle_path1)?;
    info!(
        "read DiskScissors triangle output: path={} triangles={}",
        disk_triangle_path1.display(),
        cut_triangles.len()
    );

    Ok(DiskScissorsCutOutput {
        name: cut.name.clone(),
        refined_mesh_path: tet_mesh_refined_path1,
        triangle_indices_path: disk_triangle_path1,
        cut_triangles,
        stdout,
        stderr,
    })
}

fn write_loop_file(path: &Path, loop_vertices: &[usize]) -> io::Result<()> {
    let text = loop_vertices.iter().map(|v| v.to_string()).join("\n");

    fs::write(path, format!("{text}\n"))
}

fn read_triangle_indices(path: &Path) -> io::Result<Vec<[usize; 3]>> {
    let text = fs::read_to_string(path)?;
    let mut triangles = Vec::new();

    for (line_index, line) in text.lines().enumerate() {
        let line = line.trim();

        if line.is_empty() {
            continue;
        }

        let indices = line
            .split_whitespace()
            .map(|token| {
                token.parse::<usize>().map_err(|err| {
                    io::Error::new(
                        ErrorKind::InvalidData,
                        format!(
                            "invalid triangle index '{}' on line {} in '{}': {}",
                            token,
                            line_index + 1,
                            path.display(),
                            err
                        ),
                    )
                })
            })
            .collect::<io::Result<Vec<_>>>()?;

        if indices.len() != 3 {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!(
                    "expected 3 indices on line {} in '{}', found {}",
                    line_index + 1,
                    path.display(),
                    indices.len()
                ),
            ));
        }

        triangles.push([indices[0], indices[1], indices[2]]);
    }

    Ok(triangles)
}

fn read_vtk_points(path: &Path) -> io::Result<Vec<Vector3D>> {
    let text = fs::read_to_string(path)?;
    let mut tokens = text.split_whitespace();

    while let Some(token) = tokens.next() {
        if !token.eq_ignore_ascii_case("POINTS") {
            continue;
        }

        let point_count_token = tokens.next().ok_or_else(|| {
            io::Error::new(
                ErrorKind::InvalidData,
                format!("missing POINTS count in '{}'", path.display()),
            )
        })?;
        let point_count = point_count_token.parse::<usize>().map_err(|err| {
            io::Error::new(
                ErrorKind::InvalidData,
                format!(
                    "invalid POINTS count '{}' in '{}': {}",
                    point_count_token,
                    path.display(),
                    err
                ),
            )
        })?;

        tokens.next().ok_or_else(|| {
            io::Error::new(
                ErrorKind::InvalidData,
                format!("missing POINTS data type in '{}'", path.display()),
            )
        })?;

        let mut points = Vec::with_capacity(point_count);
        for point_index in 0..point_count {
            let x = read_vtk_float(&mut tokens, path, point_index, "x")?;
            let y = read_vtk_float(&mut tokens, path, point_index, "y")?;
            let z = read_vtk_float(&mut tokens, path, point_index, "z")?;
            points.push(Vector3D::new(x, y, z));
        }

        return Ok(points);
    }

    Err(io::Error::new(
        ErrorKind::InvalidData,
        format!("VTK file '{}' has no POINTS section", path.display()),
    ))
}

fn read_vtk_float<'a>(
    tokens: &mut impl Iterator<Item = &'a str>,
    path: &Path,
    point_index: usize,
    coordinate: &str,
) -> io::Result<f64> {
    let token = tokens.next().ok_or_else(|| {
        io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "missing POINTS coordinate {coordinate} for point {point_index} in '{}'",
                path.display()
            ),
        )
    })?;

    token.parse::<f64>().map_err(|err| {
        io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "invalid POINTS coordinate {coordinate} '{}' for point {} in '{}': {}",
                token,
                point_index,
                path.display(),
                err
            ),
        )
    })
}

pub trait HexExt {
    fn construct_hex(&mut self) -> Result<(), PropertyViolationError>;
}

impl HexExt for Solution {
    fn construct_hex(&mut self) -> Result<(), PropertyViolationError> {
        let _hex_mesh = Hex::from_solution(self).map_err(|err| {
            warn!("Hex mesh construction failed: {err}");
            PropertyViolationError::UnknownError
        })?;
        Ok(())
    }
}
