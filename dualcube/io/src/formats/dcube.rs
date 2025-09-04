use crate::{Export, Import};
use bitcode::{Decode, Encode};
use dualcube::{prelude::INPUT, solutions::Loop};
use log::info;
use mehsh::prelude::*;
use serde::{Deserialize, Serialize};
use std::{io::Write, sync::Arc};

pub struct Dcube;

#[derive(Encode, Decode, Serialize, Deserialize)]
pub struct DcubeSerialization {
    pub loops: Vec<(u8, Vec<(usize, usize)>)>, // list of loops, each loop is a list of edges, each edge is two vertex ids
    pub faces: Vec<Vec<usize>>,                // list of faces, each face is a list of vertex ids
    pub verts: Vec<(f64, f64, f64)>,           // list of vertex positions
}

impl Export for Dcube {
    fn export(solution: &dualcube::prelude::Solution, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
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
            let vertices = solution.mesh_ref.vertices(face_id);
            let face_verts = vertices.iter().map(|vert_id| vert_ids.id(vert_id).unwrap().to_owned()).collect::<Vec<_>>();
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
            let loop_edges = lewp
                .edges
                .clone()
                .into_iter()
                .map(|edge_id| {
                    let vs = solution.mesh_ref.vertices(edge_id);
                    (vs[0], vs[1])
                })
                .map(|(start, end)| (vert_ids.id(&start).unwrap().to_owned(), vert_ids.id(&end).unwrap().to_owned()))
                .collect::<Vec<_>>();
            loops.push((d, loop_edges));
        }

        let path_save = path.with_extension("dcube");
        info!("Writing DCUBE to {path_save:?}");
        let mut file = std::fs::File::create(path_save.clone())?;

        let serialized = bitcode::encode(&DcubeSerialization {
            loops,
            faces,
            verts: positions,
        });
        info!("Serialized size (bitcode): {} bytes", serialized.len());

        let compressed = zstd::stream::encode_all(std::io::Cursor::new(&serialized), 3)?;
        info!("Compressed size (zstd): {} bytes", compressed.len());

        file.write_all(&compressed)?;
        info!("Successfully written DCUBE to {path_save:?}");

        Ok(())
    }
}

impl Import for Dcube {
    fn import(path: &std::path::Path) -> Result<dualcube::prelude::Solution, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let decompressed = zstd::decode_all(reader)?;
        let serialized: DcubeSerialization = bitcode::decode(&decompressed)?;

        let positions = serialized.verts.into_iter().map(|(x, y, z)| Vector3D::new(x, y, z)).collect::<Vec<_>>();

        // Create the mesh
        let mesh_result = mehsh::prelude::Mesh::<INPUT>::from(&serialized.faces, &positions);
        let Ok((mesh, vert_map, _)) = mesh_result else {
            return Err(format!("Reading invalid mesh: {:?}", mesh_result).into());
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
                _ => {
                    return Err(format!("Reading invalid loop direction: {loop_dir}").into());
                }
            };
            solution.add_loop(Loop { edges, direction });
        }

        Ok(solution)
    }
}
