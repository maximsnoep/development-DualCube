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

    /// Replaces row `i` with the zero row. Indices of other rows do not shift.
    pub fn clear_row(&mut self, i: usize) {
        self.rows[i].clear();
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

    /// Merges column `b` into column `a` (with `a != b`):
    /// for every row, the bit at column `min(a, b)` becomes the XOR of the
    /// old bits at columns `a` and `b`, and column `max(a, b)` becomes 0.
    ///
    /// This is the column-side analogue of XORing two rows together. It's
    /// exactly the right operation when two edges merge during an edge
    /// collapse: a face that was incident to either edge is now incident to
    /// the merged edge, while a (hypothetical) face incident to both becomes
    /// incident to the merged edge zero times.
    pub fn merge_columns(&mut self, keep: u32, drop: u32) {
        debug_assert!(keep != drop);
        // The caller picks which column survives — we don't reinterpret.
        // This matters when external bookkeeping (like edge_to_col in the
        // surgery's BoundaryMatrix) is keyed by which column is "kept";
        // reinterpreting via min/max would silently swap which side of
        // the merge ends up holding data, leaving external maps pointing
        // at the wrong column.
        for row in &mut self.rows {
            let drop_pos = row.binary_search(&drop);
            if let Ok(dp) = drop_pos {
                row.remove(dp);
                // After removing `drop`, decide what to do about `keep`.
                match row.binary_search(&keep) {
                    Ok(kp) => {
                        // Both bits were set → XOR is 0; drop `keep` too.
                        row.remove(kp);
                    }
                    Err(ip) => {
                        // Only `drop` was set → bit becomes `keep`.
                        row.insert(ip, keep);
                    }
                }
            }
            // else: `drop` not set; nothing to do (bit at keep stays as-is).
        }
    }

    /// Read-only access to the matrix rows. Useful when you want to do
    /// custom GF(2) arithmetic without going through the matrix's own ops.
    pub fn rows(&self) -> &[Vec<u32>] {
        &self.rows
    }

    /// After [`Self::reduce`] has been called, builds the map from each
    /// pivot column → the row index that pivots it. Useful for repeatedly
    /// reducing external 1-chains against this matrix's row span.
    pub fn pivot_map(&self) -> std::collections::HashMap<u32, usize> {
        let mut m = std::collections::HashMap::with_capacity(self.rows.len());
        for (i, row) in self.rows.iter().enumerate() {
            if let Some(&col) = row.first() {
                m.insert(col, i);
            }
        }
        m
    }

    /// Reduces `candidate` (a sorted GF(2) 1-chain) in place against this
    /// matrix's row span, using a precomputed pivot map from
    /// [`Self::pivot_map`]. Returns `true` iff the candidate reduces to the
    /// zero chain, i.e. lies in the row span of `self`.
    ///
    /// Caller must have already called [`Self::reduce`] (or otherwise put
    /// the matrix into a state where each non-empty row has a unique
    /// leading column) before calling [`Self::pivot_map`].
    pub fn reduce_against(
        &self,
        pivot_map: &std::collections::HashMap<u32, usize>,
        candidate: &mut Vec<u32>,
    ) -> bool {
        while let Some(&leading) = candidate.first() {
            match pivot_map.get(&leading) {
                Some(&piv) => {
                    let mut next = Vec::with_capacity(candidate.len() + self.rows[piv].len());
                    Self::xor_into(candidate, &self.rows[piv], &mut next);
                    *candidate = next;
                }
                None => return false,
            }
        }
        true
    }

    /// Computes the row-reduced echelon form in place using sparse Gaussian
    /// elimination. After this call, `rank()` returns the matrix's rank.
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
