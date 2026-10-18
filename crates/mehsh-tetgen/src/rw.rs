use crate::TetMeshData;
use crate::TetNode;
use crate::utils::*;
use mehsh::prelude::*;
use std::{collections::HashMap, fmt::Write as _, io, path::Path};

pub(crate) fn read_node_file(path: &Path) -> io::Result<(Vec<TetNode>, HashMap<usize, usize>)> {
    let lines = cleaned_lines(path)?;
    let header = parse_usizes(lines.first(), path, "node header")?;

    if header.len() < 4 || header[1] != 3 {
        return Err(io::Error::other(format!(
            "invalid .node header in '{}'",
            path.display()
        )));
    }

    let npoint = header[0];
    let nattr = header[2];
    let has_marker = header[3] != 0;

    if lines.len() < 1 + npoint {
        return Err(io::Error::other(format!(
            "invalid .node file '{}': expected {} points",
            path.display(),
            npoint
        )));
    }

    let mut nodes = Vec::with_capacity(npoint);
    let mut raw_to_local = HashMap::default();

    for local in 0..npoint {
        let tokens = lines[1 + local].split_whitespace().collect::<Vec<_>>();
        let min_len = 4 + nattr + usize::from(has_marker);

        if tokens.len() < min_len {
            return Err(io::Error::other(format!(
                "invalid .node line in '{}': '{}'",
                path.display(),
                lines[1 + local]
            )));
        }

        let raw_id = parse_token(tokens[0], path, "node id")?;
        let marker = if has_marker {
            parse_token(tokens[4 + nattr], path, "node marker")?
        } else {
            0
        };

        raw_to_local.insert(raw_id, local);
        nodes.push(TetNode {
            position: Vector3D::new(
                parse_token(tokens[1], path, "x")?,
                parse_token(tokens[2], path, "y")?,
                parse_token(tokens[3], path, "z")?,
            ),
            marker,
        });
    }

    Ok((nodes, raw_to_local))
}

pub(crate) fn read_ele_file(
    path: &Path,
    raw_to_local: &HashMap<usize, usize>,
) -> io::Result<Vec<[usize; 4]>> {
    let lines = cleaned_lines(path)?;
    let header = parse_usizes(lines.first(), path, "ele header")?;

    if header.len() < 3 || header[1] != 4 {
        return Err(io::Error::other(format!(
            "only tet4 .ele files are supported: '{}'",
            path.display()
        )));
    }

    let ncell = header[0];

    if lines.len() < 1 + ncell {
        return Err(io::Error::other(format!(
            "invalid .ele file '{}': expected {} cells",
            path.display(),
            ncell
        )));
    }

    (0..ncell)
        .map(|i| {
            let tokens = lines[1 + i].split_whitespace().collect::<Vec<_>>();

            if tokens.len() < 5 {
                return Err(io::Error::other(format!(
                    "invalid .ele line in '{}': '{}'",
                    path.display(),
                    lines[1 + i]
                )));
            }

            Ok([
                raw_to_local_id(parse_token(tokens[1], path, "tet v0")?, raw_to_local, path)?,
                raw_to_local_id(parse_token(tokens[2], path, "tet v1")?, raw_to_local, path)?,
                raw_to_local_id(parse_token(tokens[3], path, "tet v2")?, raw_to_local, path)?,
                raw_to_local_id(parse_token(tokens[4], path, "tet v3")?, raw_to_local, path)?,
            ])
        })
        .collect()
}

fn raw_to_local_id(
    raw_id: usize,
    raw_to_local: &HashMap<usize, usize>,
    path: &Path,
) -> io::Result<usize> {
    raw_to_local.get(&raw_id).copied().ok_or_else(|| {
        io::Error::other(format!(
            "cell references missing node id {raw_id} while reading '{}'",
            path.display()
        ))
    })
}

pub(crate) fn write_smesh<M: Tag>(mesh: &Mesh<M>, path: &Path) -> io::Result<Vec<VertKey<M>>> {
    let vert_ids = mesh.vert_ids();

    if vert_ids.len() > (i32::MAX as usize - 2) {
        return Err(io::Error::other(
            "too many vertices to assign unique i32 TetGen markers",
        ));
    }

    let input_vert_to_tet_id = vert_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect::<HashMap<_, _>>();
    let mut text = String::new();

    writeln!(text, "{} 3 0 1", vert_ids.len()).unwrap();
    for (i, vert_id) in vert_ids.iter().copied().enumerate() {
        let marker = -((i as i32) + 2);
        let p = mesh.position(vert_id);
        writeln!(text, "{} {:?} {:?} {:?} {}", i, p.x, p.y, p.z, marker).unwrap();
    }

    let face_ids = mesh.face_ids();
    writeln!(text, "{} 1", face_ids.len()).unwrap();

    for face_id in face_ids {
        let Some(f) = mesh
            .vertices(face_id)
            .map(|v| input_vert_to_tet_id[&v])
            .collect_array::<3>()
        else {
            return Err(io::Error::other("TetGen .smesh export expects triangles"));
        };
        writeln!(text, "3 {} {} {} 0", f[0], f[1], f[2]).unwrap();
    }

    writeln!(text, "0\n0").unwrap();
    write_text(path, text)?;

    Ok(vert_ids)
}

pub(crate) fn write_vtk(path: &Path, tet_mesh: &TetMeshData) -> io::Result<()> {
    let mut text = String::new();

    writeln!(text, "# vtk DataFile Version 3.0").unwrap();
    writeln!(text, "tet mesh").unwrap();
    writeln!(text, "ASCII").unwrap();
    writeln!(text, "DATASET UNSTRUCTURED_GRID").unwrap();

    writeln!(text, "POINTS {} double", tet_mesh.nodes.len()).unwrap();
    for node in &tet_mesh.nodes {
        let p = node.position;
        writeln!(text, "{:?} {:?} {:?}", p.x, p.y, p.z).unwrap();
    }

    writeln!(
        text,
        "CELLS {} {}",
        tet_mesh.tets.len(),
        tet_mesh.tets.len() * 5
    )
    .unwrap();
    for [a, b, c, d] in &tet_mesh.tets {
        writeln!(text, "4 {} {} {} {}", a, b, c, d).unwrap();
    }

    writeln!(text, "CELL_TYPES {}", tet_mesh.tets.len()).unwrap();
    for _ in &tet_mesh.tets {
        writeln!(text, "10").unwrap();
    }

    write_text(path, text)
}
