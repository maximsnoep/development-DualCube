//! Concrete numerical kernels, optimized independently of the calling crates.

use faer::{
    Conj, Mat, Par,
    dyn_stack::{MemBuffer, MemStack},
    sparse::{
        FaerError, SparseColMatRef, SymbolicSparseColMat,
        linalg::qr::{self, SymbolicQr},
    },
};

/// Reusable sparse QR storage for f64 systems with a fixed sparsity pattern.
pub struct QrWorkspace {
    pattern: SymbolicSparseColMat<usize>,
    symbolic: SymbolicQr<usize>,
    indices: Vec<usize>,
    numeric: Vec<f64>,
    scratch: MemBuffer,
    parallelism: Par,
    rhs_columns: usize,
}

impl QrWorkspace {
    pub fn new(
        matrix: SparseColMatRef<'_, usize, f64>,
        rhs_columns: usize,
    ) -> Result<Self, FaerError> {
        let symbolic = qr::factorize_symbolic_qr(matrix.symbolic(), Default::default())?;
        let parallelism = faer::get_global_parallelism();
        let scratch = MemBuffer::try_new(
            symbolic
                .factorize_numeric_qr_scratch::<f64>(parallelism, Default::default())
                .or(symbolic.solve_in_place_scratch::<f64>(rhs_columns, parallelism)),
        )?;
        let mut indices = Vec::new();
        let mut numeric = Vec::new();
        indices
            .try_reserve_exact(symbolic.len_idx())
            .map_err(|_| FaerError::OutOfMemory)?;
        numeric
            .try_reserve_exact(symbolic.len_val())
            .map_err(|_| FaerError::OutOfMemory)?;
        indices.resize(symbolic.len_idx(), 0);
        numeric.resize(symbolic.len_val(), 0.);
        Ok(Self {
            pattern: matrix.symbolic().to_owned()?,
            symbolic,
            indices,
            numeric,
            scratch,
            parallelism,
            rhs_columns,
        })
    }

    /// Refactor a matrix with the original sparsity pattern and solve in place.
    /// The first `matrix.ncols()` rows of rhs contain the least-squares solution.
    ///
    /// # Panics
    /// Panics if the matrix dimensions or stored row indices in any column change,
    /// or if `rhs` does not have `matrix.nrows()` rows and the configured columns.
    pub fn solve(&mut self, matrix: SparseColMatRef<'_, usize, f64>, rhs: &mut Mat<f64>) {
        assert_eq!(
            matrix.nrows(),
            self.pattern.nrows(),
            "QR matrix row count changed"
        );
        assert_eq!(
            matrix.ncols(),
            self.pattern.ncols(),
            "QR matrix column count changed"
        );
        assert!(
            (0..matrix.ncols()).all(|col| {
                matrix.symbolic().row_idx_of_col_raw(col) == self.pattern.row_idx_of_col_raw(col)
            }),
            "QR matrix sparsity pattern changed"
        );
        assert_eq!(
            rhs.nrows(),
            matrix.nrows(),
            "QR right-hand side row count mismatch"
        );
        assert_eq!(rhs.ncols(), self.rhs_columns);
        self.indices.fill(0);
        self.numeric.fill(0.);
        let qr = self.symbolic.factorize_numeric_qr(
            &mut self.indices,
            &mut self.numeric,
            matrix,
            self.parallelism,
            MemStack::new(&mut self.scratch),
            Default::default(),
        );
        qr.solve_in_place_with_conj(
            Conj::No,
            rhs.as_mut(),
            self.parallelism,
            MemStack::new(&mut self.scratch),
        );
    }
}

#[cfg(test)]
mod tests;
