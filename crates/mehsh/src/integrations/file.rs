use crate::prelude::*;
use crate::utils::ids::IdMap;
use std::path::Path;

impl<M: Tag> Mesh<M>
where
    M: Default + Eq + std::hash::Hash + Copy + Clone,
{
    /// Read a mesh from a file, in the format given by its extension (OBJ or STL).
    pub fn from_file(path: &Path) -> Result<(Self, IdMap<VERT, M>, IdMap<FACE, M>), MeshError<M>> {
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase);
        match extension.as_deref() {
            #[cfg(feature = "obj")]
            Some("obj") => Self::from_obj(path),
            #[cfg(feature = "stl")]
            Some("stl") => Self::from_stl(path),
            _ => Err(MeshError::Unknown(format!(
                "Unknown file extension: {path:?}",
            ))),
        }
    }
}
