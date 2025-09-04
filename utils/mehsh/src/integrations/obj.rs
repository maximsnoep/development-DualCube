use crate::prelude::*;
use itertools::Itertools;
use std::io::Write;
use std::{
    fs::OpenOptions,
    io::{BufRead, BufReader},
    path::PathBuf,
};

impl<M: Tag> Mesh<M>
where
    M: std::default::Default + std::cmp::Eq + std::hash::Hash + Copy + Clone,
{
    pub fn from_obj(path: &PathBuf) -> Result<(Self, ids::IdMap<VERT, M>, ids::IdMap<FACE, M>), MeshError<M>> {
        match OpenOptions::new().read(true).open(path) {
            Ok(file) => match path.extension().unwrap().to_str() {
                Some("obj") => match Self::obj_to_elements(BufReader::new(file)) {
                    Ok((verts, faces)) => Self::from(&faces, &verts),
                    Err(e) => Err(MeshError::Unknown(format!(
                        "Something went wrong while reading the OBJ file: {path:?}\nErr: {e}"
                    ))),
                },
                _ => Err(MeshError::Unknown(format!("Unknown file extension: {path:?}",))),
            },
            Err(e) => Err(MeshError::Unknown(format!("Cannot read file: {path:?}\nErr: {e}"))),
        }
    }

    pub fn to_obj(&self, path: &PathBuf) -> Result<(), std::io::Error> {
        let mut file = std::fs::File::create(path)?;

        let mut vert_ids = ids::IdMap::<VERT, M>::new();
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
                            .iter()
                            .map(|vert_id| format!("{}", vert_ids.id(vert_id).unwrap()))
                            .join(" ")
                    )
                })
                .join("\n")
        )?;

        Ok(())
    }

    fn obj_to_elements(reader: impl BufRead) -> Result<(Vec<Vector3D>, Vec<Vec<usize>>), obj::ObjError> {
        let obj = obj::ObjData::load_buf(reader)?;
        let verts = obj.position.iter().map(|v| Vector3D::new(v[0].into(), v[1].into(), v[2].into())).collect_vec();
        let faces = obj.objects[0].groups[0]
            .polys
            .iter()
            .map(|f| f.0.iter().map(|v| v.0).collect_vec())
            .collect_vec();
        Ok((verts, faces))
    }
}
