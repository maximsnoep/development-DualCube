use bitcode::{Decode, Encode};
use dualcube::prelude::*;
use serde::{Deserialize, Serialize};
use std::{io::Write, sync::Arc};

pub struct Loops;

#[derive(Encode, Decode, Serialize, Deserialize)]
struct LoopsSerialization {
    loops: Vec<(u8, Vec<(usize, usize)>)>, // list of loops, each loop is a list of edges, each edge is two vertex ids
    faces: Vec<Vec<usize>>,                // list of faces, each face is a list of vertex ids
    verts: Vec<(f64, f64, f64)>,           // list of vertex positions
}

// Version two preserves explicit face segments and legacy edge offsets.
#[derive(Encode, Decode)]
struct LoopsV2 {
    base: LoopsSerialization,
    surface: Vec<Vec<(usize, [f64; 2], [f64; 2])>>,
    offsets: Vec<Vec<f64>>,
}

impl crate::Export for Loops {
    fn export(solution: &Solution, path: &std::path::Path) -> anyhow::Result<()> {
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
        let mut face_ids = HashMap::new();
        for face_id in solution.mesh_ref.face_ids() {
            face_ids.insert(face_id, faces.len());
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
                Direction::X => 0,
                Direction::Y => 1,
                Direction::Z => 2,
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
        let base = LoopsSerialization {
            loops,
            faces,
            verts: positions,
        };
        let mut serialized = b"DCL2".to_vec();
        serialized.extend(bitcode::encode(&LoopsV2 {
            base,
            surface: solution
                .loops
                .values()
                .map(|l| {
                    l.surface
                        .iter()
                        .map(|a| (face_ids[&a.face], a.from, a.to))
                        .collect()
                })
                .collect(),
            offsets: solution.loops.values().map(|l| l.offsets.clone()).collect(),
        }));
        let compressed = zstd::stream::encode_all(std::io::Cursor::new(&serialized), 0)?;
        file.write_all(&compressed)?;
        Ok(())
    }
}

impl crate::Import for Loops {
    fn import(path: &std::path::Path) -> anyhow::Result<Solution> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let decompressed = zstd::decode_all(reader)?;
        let (serialized, surface, offsets) = if let Some(bytes) = decompressed.strip_prefix(b"DCL2")
        {
            let v: LoopsV2 = bitcode::decode(bytes)?;
            if v.surface.len() != v.base.loops.len() || v.offsets.len() != v.base.loops.len() {
                anyhow::bail!("inconsistent loop geometry counts");
            }
            (v.base, v.surface, v.offsets)
        } else {
            (
                bitcode::decode::<LoopsSerialization>(&decompressed)?,
                vec![],
                vec![],
            )
        };

        let positions = serialized
            .verts
            .into_iter()
            .map(|(x, y, z)| Vector3D::new(x, y, z))
            .collect::<Vec<_>>();

        // Create the mesh
        let (mesh, vert_map, face_map) = match Mesh::<INPUT>::from(&serialized.faces, &positions) {
            Ok(ok) => ok,
            Err(e) => anyhow::bail!("reading invalid mesh from {}: {:?}", path.display(), e),
        };

        let mut solution = Solution::new(Arc::new(mesh.clone()));

        // Reconstruct the loops
        let mut result = slotmap::SlotMap::with_key();
        for (index, (loop_dir, loop_edges)) in serialized.loops.into_iter().enumerate() {
            let edges = loop_edges
                .into_iter()
                .map(|(start, end)| {
                    let start_vert = vert_map.key(start).unwrap().to_owned();
                    let end_vert = vert_map.key(end).unwrap().to_owned();
                    mesh.edge_between_verts(start_vert, end_vert).unwrap().0
                })
                .collect::<Vec<_>>();
            let direction = match loop_dir {
                0 => Direction::X,
                1 => Direction::Y,
                2 => Direction::Z,
                _ => anyhow::bail!(
                    "reading invalid loop direction {loop_dir} from {}",
                    path.display()
                ),
            };
            let mut l = Loop::new(edges, direction);
            if let Some(arcs) = surface.get(index) {
                l.surface = arcs
                    .iter()
                    .map(|&(face, from, to)| {
                        let mapped = face_map
                            .key(face)
                            .copied()
                            .ok_or_else(|| anyhow::anyhow!("invalid curve face index"))?;
                        let corners = mesh
                            .vertices(mapped)
                            .map(|v| vert_map.id(&v).copied().unwrap())
                            .collect_vec();
                        let transform = |p: [f64; 2]| {
                            let weights = [1. - p[0] - p[1], p[0], p[1]];
                            let at = |i: usize| {
                                weights[serialized.faces[face]
                                    .iter()
                                    .position(|&v| v == corners[i])
                                    .unwrap()]
                            };
                            [at(1), at(2)]
                        };
                        Ok(SurfaceArc {
                            face: mapped,
                            from: transform(from),
                            to: transform(to),
                        })
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?;
                l.offsets = offsets[index].clone();
            }
            result.insert(l);
        }

        solution.replace_loops(result);
        Ok(solution)
    }
}
