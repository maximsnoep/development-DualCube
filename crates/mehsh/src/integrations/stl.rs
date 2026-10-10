use crate::prelude::*;
use itertools::Itertools;
use std::{
    fs::OpenOptions,
    io::{BufRead, BufReader},
    path::Path,
};

impl<M: Tag> Mesh<M>
where
    M: Default + Eq + std::hash::Hash + Copy + Clone,
{
    pub fn from_stl(
        path: &Path,
    ) -> Result<(Self, ids::IdMap<VERT, M>, ids::IdMap<FACE, M>), MeshError<M>> {
        match OpenOptions::new().read(true).open(path) {
            Ok(file) => match path.extension().and_then(|e| e.to_str()) {
                Some(e) if e.eq_ignore_ascii_case("stl") => {
                    match Self::stl_to_elements(BufReader::new(file)) {
                        Ok((verts, faces)) => Self::from_input(&faces, &verts),
                        Err(e) => Err(MeshError::Unknown(format!(
                            "Something went wrong while reading the STL file: {path:?}\nErr: {e}"
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

    pub fn stl_to_elements(
        mut reader: impl BufRead + std::io::Seek,
    ) -> Result<(Vec<Vector3D>, Vec<Vec<usize>>), std::io::Error> {
        let stl = stl_io::read_stl(&mut reader)?;
        let verts = stl
            .vertices
            .iter()
            .map(|v| Vector3D::new(v[0].into(), v[1].into(), v[2].into()))
            .collect_vec();
        let faces = stl.faces.iter().map(|f| f.vertices.to_vec()).collect_vec();
        Ok((verts, faces))
    }
}
