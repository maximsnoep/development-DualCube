use bitcode::{Decode, Encode};
use dualcube::{prelude::INPUT, solutions::Loop};
use mehsh::prelude::*;
use serde::{Deserialize, Serialize};
use std::{io::Write, sync::Arc};

pub struct Loops;

#[derive(Encode, Decode, Serialize, Deserialize)]
struct LoopsSerialization {
    loops: Vec<(u8, Vec<(usize, usize)>)>, // list of loops, each loop is a list of edges, each edge is two vertex ids
    faces: Vec<Vec<usize>>,                // list of faces, each face is a list of vertex ids
    verts: Vec<(f64, f64, f64)>,           // list of vertex positions
}

impl crate::Export for Loops {
    fn export(
        solution: &dualcube::prelude::Solution,
        path: &std::path::Path,
    ) -> anyhow::Result<()> {
        // Mesh to a list of faces and vertices (positions)
        let mut positions = vec![];
        let mut vert_ids = ids::IdMap::<VERT, INPUT>::new();
        for (i, vert_id) in solution.mesh_ref.vert_ids().into_iter().enumerate() {
            vert_ids.insert(i, vert_id);
            positions.push((
                solution.mesh_ref.position(vert_id).x,
                solution.mesh_ref.position(vert_id).y,
                solution.mesh_ref.position(vert_id).z,
            ));
        }

        let mut faces = vec![];
        for face_id in solution.mesh_ref.face_ids() {
            let face_verts = solution
                .mesh_ref
                .vertices(face_id)
                .map(|vert_id| {
                    vert_ids
                        .id(&vert_id)
                        .copied()
                        .ok_or_else(|| anyhow::anyhow!("missing vertex id mapping for {vert_id:?}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            faces.push(face_verts);
        }

        // Loops to list of edges (vertex pairs)
        let mut loops = vec![];
        for lewp in solution.loops.values() {
            let d = match lewp.direction {
                dualcube::prelude::PrincipalDirection::X => 0,
                dualcube::prelude::PrincipalDirection::Y => 1,
                dualcube::prelude::PrincipalDirection::Z => 2,
            };
            let mut loop_edges = Vec::new();
            for edge_id in lewp.edges.iter().copied() {
                let Some([v0, v1]) = solution.mesh_ref.vertices(edge_id).collect_array::<2>()
                else {
                    anyhow::bail!("expecting edge {edge_id:?} to have exactly two vertices");
                };

                let a = vert_ids
                    .id(&v0)
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("missing vertex id mapping for {v0:?}"))?;
                let b = vert_ids
                    .id(&v1)
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("missing vertex id mapping for {v1:?}"))?;

                loop_edges.push((a, b));
            }
            loops.push((d, loop_edges));
        }

        let path_save = path.with_extension("loops");
        let mut file = std::fs::File::create(&path_save)?;
        let serialized = bitcode::encode(&LoopsSerialization {
            loops,
            faces,
            verts: positions,
        });
        let compressed = zstd::stream::encode_all(std::io::Cursor::new(&serialized), 0)?;
        file.write_all(&compressed)?;
        Ok(())
    }
}

impl crate::Import for Loops {
    fn import(path: &std::path::Path) -> anyhow::Result<dualcube::prelude::Solution> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let decompressed = zstd::decode_all(reader)?;
        let serialized: LoopsSerialization = bitcode::decode(&decompressed)?;

        let positions = serialized
            .verts
            .into_iter()
            .map(|(x, y, z)| Vector3D::new(x, y, z))
            .collect::<Vec<_>>();

        // Create the mesh
        let (mesh, vert_map, _face_map) =
            match mehsh::prelude::Mesh::<INPUT>::from(&serialized.faces, &positions) {
                Ok(ok) => ok,
                Err(e) => anyhow::bail!("reading invalid mesh from {}: {:?}", path.display(), e),
            };

        let mut solution = dualcube::prelude::Solution::new(Arc::new(mesh.clone()));

        // Reconstruct the loops
        for (loop_dir, loop_edges) in serialized.loops {
            let edges = loop_edges
                .into_iter()
                .map(|(start, end)| {
                    let start_vert = vert_map.key(start).unwrap().to_owned();
                    let end_vert = vert_map.key(end).unwrap().to_owned();
                    mesh.edge_between_verts(start_vert, end_vert).unwrap().0
                })
                .collect::<Vec<_>>();
            let direction = match loop_dir {
                0 => dualcube::prelude::PrincipalDirection::X,
                1 => dualcube::prelude::PrincipalDirection::Y,
                2 => dualcube::prelude::PrincipalDirection::Z,
                _ => anyhow::bail!(
                    "reading invalid loop direction {loop_dir} from {}",
                    path.display()
                ),
            };
            solution.add_loop(Loop { edges, direction });
        }

        Ok(solution)
    }
}
