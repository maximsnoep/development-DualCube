use crate::prelude::*;
use itertools::Itertools;
use std::{
    fs::OpenOptions,
    io::{BufRead, BufReader},
    path::PathBuf,
};

impl<M: Tag> Mesh<M>
where
    M: std::default::Default + std::cmp::Eq + std::hash::Hash + Copy + Clone,
{
    pub fn from_stl(path: &PathBuf) -> Result<(Self, ids::IdMap<VERT, M>, ids::IdMap<FACE, M>), MeshError<M>> {
        match OpenOptions::new().read(true).open(path) {
            Ok(file) => match path.extension().unwrap().to_str() {
                Some("stl") => match Self::stl_to_elements(BufReader::new(file)) {
                    Ok((verts, faces)) => Self::from(&faces, &verts),
                    Err(e) => Err(MeshError::Unknown(format!(
                        "Something went wrong while reading the STL file: {path:?}\nErr: {e}"
                    ))),
                },
                _ => Err(MeshError::Unknown(format!("Unknown file extension: {path:?}",))),
            },
            Err(e) => Err(MeshError::Unknown(format!("Cannot read file: {path:?}\nErr: {e}"))),
        }
    }

    pub fn to_stl(&self) -> Result<(), std::io::Error> {
        unimplemented!("writing to stl is not implemented yet");
    }

    pub fn stl_to_elements(mut reader: impl BufRead + std::io::Seek) -> Result<(Vec<Vector3D>, Vec<Vec<usize>>), std::io::Error> {
        let stl = stl_io::read_stl(&mut reader)?;
        let verts = stl.vertices.iter().map(|v| Vector3D::new(v[0].into(), v[1].into(), v[2].into())).collect_vec();
        let faces = stl.faces.iter().map(|f| f.vertices.to_vec()).collect_vec();
        Ok((verts, faces))
    }
}
