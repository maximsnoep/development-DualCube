//! Sparse linear algebra over GF(2) (the field with two elements).
//!
//! Each row is stored as a sorted `Vec<u32>` of nonzero column indices;
//! row addition is the symmetric difference of two sorted index lists,
//! computed with a two-pointer merge. This is what's needed for tracking
//! the rank of a sparse Z/2 boundary matrix incrementally.
//!
//! The matrix supports soft-deleting rows (replace with the empty row) so
//! that external `face_id -> row_index` maps stay valid across deletes.
//! Columns are likewise never renumbered: `merge_columns(a, b)` collapses
//! column `b` into column `a` in place, after which `b` is empty for life.

use std::cmp::Ordering;

/// A sparse matrix over GF(2), stored as one sorted row per face.
#[derive(Clone, Debug)]
pub struct F2Matrix {
    /// `rows[i]` is the sorted list of column indices where row `i` has a 1.
    /// An empty row represents the zero row (used as soft-delete).
    rows: Vec<Vec<u32>>,
}

impl F2Matrix {
    /// Builds a matrix from raw rows. Each row must already be sorted with
    /// no duplicates — that's how the storage invariant is defined.
    pub fn from_rows(rows: Vec<Vec<u32>>) -> Self {
        debug_assert!(rows.iter().all(|r| r.windows(2).all(|w| w[0] < w[1])));
        Self { rows }
    }

    /// Number of nonzero rows. Only meaningful after `reduce()` — counts the
    /// rank of the matrix.
    pub fn rank(&self) -> usize {
        self.rows.iter().filter(|r| !r.is_empty()).count()
    }

    /// Symmetric difference of two sorted slices, written into `out`.
    fn xor_into(a: &[u32], b: &[u32], out: &mut Vec<u32>) {
        out.clear();
        out.reserve(a.len() + b.len());
        let (mut i, mut j) = (0, 0);
        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                Ordering::Less => {
                    out.push(a[i]);
                    i += 1;
                }
                Ordering::Greater => {
                    out.push(b[j]);
                    j += 1;
                }
                Ordering::Equal => {
                    i += 1;
                    j += 1;
                }
            }
        }
        out.extend_from_slice(&a[i..]);
        out.extend_from_slice(&b[j..]);
    }

    /// In-place row XOR: `rows[target] ^= rows[source]`. `source != target`.
    fn xor_row_with(&mut self, target: usize, source: usize) {
        debug_assert_ne!(target, source);
        // Swap rows out so we can borrow both mutably at the same time.
        let src = std::mem::take(&mut self.rows[source]);
        let tgt = std::mem::take(&mut self.rows[target]);
        let mut new_row = Vec::with_capacity(tgt.len() + src.len());
        Self::xor_into(&tgt, &src, &mut new_row);
        self.rows[target] = new_row;
        self.rows[source] = src;
    }

    /// Read-only access to the matrix rows. Useful when you want to do
    /// custom GF(2) arithmetic without going through the matrix's own ops.
    pub fn rows(&self) -> &[Vec<u32>] {
        &self.rows
    }

    /// Eliminate rows until each nonzero row has a distinct leading column.
    /// This is not full RREF. After this call, `rank()` returns the matrix's rank.
    ///
    /// Strategy: process rows in order; for each, repeatedly XOR-eliminate the
    /// leading nonzero column against the pivot row already assigned to that
    /// column. If no such pivot exists, the current row becomes the pivot for
    /// its leading column.
    pub fn reduce(&mut self) {
        // pivot_for_col[col] = Some(row_idx) if some row has been assigned as
        // the pivot for `col`. Stored sparsely as a Vec<(col, row)> sorted by
        // row order of assignment isn't needed; a HashMap is fine.
        use std::collections::HashMap;
        let mut pivot_for_col: HashMap<u32, usize> = HashMap::new();

        for i in 0..self.rows.len() {
            loop {
                if self.rows[i].is_empty() {
                    break;
                }
                let leading = self.rows[i][0];
                match pivot_for_col.get(&leading) {
                    Some(&piv) if piv != i => {
                        self.xor_row_with(i, piv);
                    }
                    Some(_) => break, // i is already its own pivot
                    None => {
                        pivot_for_col.insert(leading, i);
                        break;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "test/test_f2_rref.rs"]
mod test_f2_rref;
