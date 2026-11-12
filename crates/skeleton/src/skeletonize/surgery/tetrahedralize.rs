//! Adapt the shared `TetGen` output to the homology calculation's indices.

use dualcube_types::prelude::*;

#[derive(Debug, Clone)]
pub struct TetMesh {
    pub n_vertices: usize,
    pub tets: Vec<[u32; 4]>,
}

impl TetMesh {
    pub fn n_tets(&self) -> usize {
        self.tets.len()
    }
}

/// Input vertices keep their indices in `mesh.vert_ids()` order.
pub fn tetrahedralize(mesh: &Mesh<INPUT>) -> crate::SkeletonResult<TetMesh> {
    let result = mehsh_tetgen::tetrahedralize(mesh)?;
    if result.vertices.len() > u32::MAX as usize {
        return Err("Too many TetGen vertices for homology indices".into());
    }
    if result.input_vertices != mesh.vert_ids() {
        return Err("TetGen input vertex ordering changed".into());
    }
    Ok(TetMesh {
        n_vertices: result.vertices.len(),
        tets: result
            .tets
            .into_iter()
            .map(|t| t.map(|i| i as u32))
            .collect(),
    })
}
