use crate::prelude::*;
use crate::utils::ids::IdMap;
use itertools::Itertools;
use std::collections::HashMap;
use std::io::Write;
use std::{
    fs::OpenOptions,
    io::{BufRead, BufReader},
    path::Path,
};

impl<M: Tag> Mesh<M>
where
    M: Default + Eq + std::hash::Hash + Copy + Clone,
{
    pub fn from_obj(path: &Path) -> Result<(Self, IdMap<VERT, M>, IdMap<FACE, M>), MeshError<M>> {
        match OpenOptions::new().read(true).open(path) {
            Ok(file) => match path.extension().and_then(|e| e.to_str()) {
                Some(e) if e.eq_ignore_ascii_case("obj") => {
                    match Self::obj_to_elements(BufReader::new(file)) {
                        Ok((verts, faces)) => Self::from_input(&faces, &verts),
                        Err(e) => Err(MeshError::Unknown(format!(
                            "Something went wrong while reading the OBJ file: {path:?}\nErr: {e}"
                        ))),
                    }
                }
                _ => Err(MeshError::Unknown(format!(
                    "Unknown file extension: {path:?}",
                ))),
            },
            Err(e) => Err(MeshError::Unknown(format!(
                "Cannot read file: {path:?}\nErr: {e}"
            ))),
        }
    }

    pub fn to_obj(&self, path: &Path) -> Result<IdMap<VERT, M>, std::io::Error> {
        let mut file = std::fs::File::create(path)?;

        let mut vert_ids = IdMap::<VERT, M>::new();
        for (i, vert_id) in self.vert_ids().into_iter().enumerate() {
            vert_ids.insert(i + 1, vert_id);
        }

        writeln!(
            file,
            "{}",
            self.vert_ids()
                .into_iter()
                .map(|vert_id| format!(
                    "v {x:.6} {y:.6} {z:.6}",
                    x = self.position(vert_id).x,
                    y = self.position(vert_id).y,
                    z = self.position(vert_id).z
                ))
                .join("\n")
        )?;

        writeln!(
            file,
            "{}",
            self.face_ids()
                .into_iter()
                .map(|face_id| {
                    format!(
                        "vn {x:.6} {y:.6} {z:.6}",
                        x = self.normal(face_id).x,
                        y = self.normal(face_id).y,
                        z = self.normal(face_id).z
                    )
                })
                .join("\n")
        )?;

        writeln!(
            file,
            "{}",
            self.face_ids()
                .into_iter()
                .map(|face_id| {
                    format!(
                        "f {}",
                        self.vertices(face_id)
                            .map(|vert_id| format!("{}", vert_ids.id(&vert_id).unwrap()))
                            .join(" ")
                    )
                })
                .join("\n")
        )?;

        Ok(vert_ids)
    }

    // The vertices and faces of an OBJ file (with `tobj`, in double precision). Every object (`o`) or group (`g`) is
    // read by `tobj` as a model with its own vertices: vertices at the same position in different models are merged
    // (within a model, the vertices are kept as they are). The vertices are numbered in the order of their first use.
    fn obj_to_elements(
        reader: impl BufRead,
    ) -> Result<(Vec<Vector3D>, Vec<Vec<usize>>), tobj::LoadError> {
        let options = tobj::LoadOptions {
            single_index: false,
            triangulate: false,
            ignore_points: true,
            ignore_lines: true,
        };
        let mut reader = reader;
        let (models, _) = tobj::load_obj_buf(&mut reader, &options, |_| {
            Err(tobj::LoadError::GenericFailure)
        })?;
        let mut verts: Vec<Vector3D> = vec![];
        let mut faces: Vec<Vec<usize>> = vec![];
        // Per position: the first vertex there, and its model.
        let mut first: HashMap<[u64; 3], (usize, usize)> = HashMap::new();
        for (m, model) in models.iter().enumerate() {
            let mesh = &model.mesh;
            let index = mesh
                .positions
                .chunks(3)
                .map(|p| {
                    let key = [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()];
                    match first.get(&key) {
                        // A vertex of an earlier model at the same position.
                        Some(&(i, model)) if model < m => i,
                        _ => {
                            verts.push(Vector3D::new(p[0], p[1], p[2]));
                            first.entry(key).or_insert((verts.len() - 1, m));
                            verts.len() - 1
                        }
                    }
                })
                .collect_vec();
            // (Without triangulation, `face_arities` gives the number of corners of every face; empty if all are triangles.)
            let arities = if mesh.face_arities.is_empty() {
                vec![3; mesh.indices.len() / 3]
            } else {
                mesh.face_arities.clone()
            };
            let mut at = 0;
            for arity in arities {
                let arity = arity as usize;
                faces.push(
                    mesh.indices[at..at + arity]
                        .iter()
                        .map(|&i| index[i as usize])
                        .collect(),
                );
                at += arity;
            }
        }
        Ok((verts, faces))
    }
}
