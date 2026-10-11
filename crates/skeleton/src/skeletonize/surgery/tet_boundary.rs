//! GF(2) boundary matrix ∂₂ for the 3-complex produced by TetGen.
//!
//! Given a [`TetMesh`] of the solid `V` bounded by the input surface `S`,
//! this module enumerates all unique triangles and edges across all
//! tetrahedra and builds the sparse boundary map
//!
//! ```text
//! ∂₂^T : C₂(T; ℤ/2) → C₁(T; ℤ/2)
//! ```
//!
//! over GF(2). The image of this map is exactly `B₁(T)`, the boundary
//! group of `T`. Piece 3 of the pipeline will use it to test whether a
//! given surface 1-cycle is a boundary in `V` — i.e. whether it's a handle
//! loop.
//!
//! Indexing conventions:
//! - Triangles and edges are interned by their sorted vertex tuples.
//!   Vertex indices are the same `u32`s that index into
//!   [`TetMesh::vertices`], so input-surface vertices keep their indices
//!   `0..n_input_vertices` here too.
//! - The matrix's rows are indexed by `triangles`, columns by `edges`.
//!   Surface-side code can look up the column for a surface edge via
//!   [`TetBoundaryComplex::edge_col`].

use std::collections::HashMap;

use crate::skeletonize::surgery::handle_subspace::sort_pair;

use super::f2_rref::F2Matrix;
use super::tetrahedralize::TetMesh;

/// The ∂₂ boundary complex of a [`TetMesh`].
///
/// `boundary_2` is left **unreduced** by [`build_tet_boundary`]; callers
/// that need its rank or its row-echelon form should call
/// [`F2Matrix::reduce`] explicitly (it's a one-time cost paid by whoever
/// owns the complex).
#[derive(Debug, Clone)]
pub struct TetBoundaryComplex {
    /// Sparse boundary matrix ∂₂^T over GF(2). One row per element of
    /// `triangles`, one column per element of `edges`. Each row has
    /// exactly three nonzero entries (the three edges bounding that
    /// triangle).
    pub boundary_2: F2Matrix,
    /// Row → triangle map: sorted vertex triple for each row.
    pub triangles: Vec<[u32; 3]>,
    /// Column → edge map: sorted vertex pair for each column.
    pub edges: Vec<[u32; 2]>,
    /// Sorted edge pair → column index in `boundary_2`.
    pub edge_to_col: HashMap<[u32; 2], u32>,
    /// Sorted triangle triple → row index in `boundary_2`.
    #[cfg(test)]
    pub triangle_to_row: HashMap<[u32; 3], usize>,
}

impl TetBoundaryComplex {
    /// Total number of unique edges across all tetrahedra of `T`.
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Total number of unique triangles across all tetrahedra of `T`.
    pub fn n_triangles(&self) -> usize {
        self.triangles.len()
    }

    /// Returns the column index of the edge `{a, b}` in `boundary_2`, or
    /// `None` if no such edge exists in the volume complex. (Won't happen
    /// for an edge of `S` after a valid tetrahedralization, but the lookup
    /// is offered as `Option` for safety.)
    pub fn edge_col(&self, a: u32, b: u32) -> Option<u32> {
        let pair = sort_pair(a, b);
        self.edge_to_col.get(&pair).copied()
    }
}

/// Walks every tetrahedron of `tm` and builds the unique-triangle and
/// unique-edge lists, then assembles ∂₂^T row by row.
///
/// A triangle is "unique" up to its sorted vertex triple — an interior
/// triangle is shared by exactly two tets but only appears once in
/// `triangles`. Similarly for edges (each is shared by many triangles).
pub fn build_tet_boundary(tm: &TetMesh) -> TetBoundaryComplex {
    let mut edge_to_col: HashMap<[u32; 2], u32> = HashMap::new();
    let mut edges: Vec<[u32; 2]> = Vec::new();
    let mut triangle_to_row: HashMap<[u32; 3], usize> = HashMap::new();
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    let mut rows: Vec<Vec<u32>> = Vec::new();

    for tet in &tm.tets {
        let [a, b, c, d] = *tet;
        debug_assert!(
            a != b && a != c && a != d && b != c && b != d && c != d,
            "degenerate tet with repeated vertices: {:?}",
            tet
        );

        // The four triangular faces of the tet — every 3-subset of {a, b, c, d}.
        let faces: [[u32; 3]; 4] = [
            sort_triple(a, b, c),
            sort_triple(a, b, d),
            sort_triple(a, c, d),
            sort_triple(b, c, d),
        ];

        for face in faces {
            if triangle_to_row.contains_key(&face) {
                // Already seen via the other tet that shares this interior face.
                continue;
            }

            // Intern the three edges of this face.
            let e0 = intern_edge([face[0], face[1]], &mut edges, &mut edge_to_col);
            let e1 = intern_edge([face[1], face[2]], &mut edges, &mut edge_to_col);
            let e2 = intern_edge([face[0], face[2]], &mut edges, &mut edge_to_col);

            let mut row = [e0, e1, e2];
            row.sort();
            let row_idx = rows.len();
            rows.push(row.to_vec());
            triangle_to_row.insert(face, row_idx);
            triangles.push(face);
        }
    }

    TetBoundaryComplex {
        boundary_2: F2Matrix::from_rows(rows),
        triangles,
        edges,
        edge_to_col,
        #[cfg(test)]
        triangle_to_row,
    }
}

#[inline]
fn sort_triple(a: u32, b: u32, c: u32) -> [u32; 3] {
    let mut x = [a, b, c];
    x.sort();
    x
}

/// Returns the column index for `pair` (sorted), inserting a new column if
/// this edge hasn't been seen yet.
fn intern_edge(
    pair: [u32; 2],
    edges: &mut Vec<[u32; 2]>,
    edge_to_col: &mut HashMap<[u32; 2], u32>,
) -> u32 {
    let pair = sort_pair(pair[0], pair[1]);
    *edge_to_col.entry(pair).or_insert_with(|| {
        let id = edges.len() as u32;
        edges.push(pair);
        id
    })
}

#[cfg(test)]
#[path = "test/test_tet_boundary.rs"]
mod test_tet_boundary;
