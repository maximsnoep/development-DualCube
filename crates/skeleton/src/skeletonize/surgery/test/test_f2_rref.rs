#[cfg(test)]
mod tests {
    use crate::skeletonize::surgery::f2_rref::F2Matrix;

    /// Two identical rows reduce to one nonzero row → rank 1.
    #[test]
    fn rank_of_duplicate_rows() {
        let mut m = F2Matrix::from_rows(vec![vec![0, 2, 4], vec![0, 2, 4]]);
        m.reduce();
        assert_eq!(m.rank(), 1);
    }

    /// Three rows where row 3 = row 1 XOR row 2 → rank 2.
    #[test]
    fn rank_of_dependent_rows() {
        // r1 = {0,1}, r2 = {1,2}, r3 = r1 XOR r2 = {0,2}
        let mut m = F2Matrix::from_rows(vec![vec![0, 1], vec![1, 2], vec![0, 2]]);
        m.reduce();
        assert_eq!(m.rank(), 2);
    }

    /// A 3x3 identity-style matrix → full rank.
    #[test]
    fn rank_of_independent_rows() {
        let mut m = F2Matrix::from_rows(vec![vec![0], vec![1], vec![2]]);
        m.reduce();
        assert_eq!(m.rank(), 3);
    }

    /// `merge_columns` on disjoint rows: bit moves from the dropped col to the kept col.
    #[test]
    fn merge_columns_disjoint() {
        // row A has only col 5; row B has only col 2.
        // Drop col 5 into col 2 → both rows end up with col 2.
        let mut m = F2Matrix::from_rows(vec![vec![5], vec![2]]);
        m.merge_columns(2, 5);
        assert_eq!(m.rows[0], vec![2]);
        assert_eq!(m.rows[1], vec![2]);
    }

    /// `merge_columns` when a row has both: bits cancel out.
    #[test]
    fn merge_columns_both_set_cancels() {
        let mut m = F2Matrix::from_rows(vec![vec![1, 4]]);
        m.merge_columns(1, 4);
        assert_eq!(m.rows[0], Vec::<u32>::new());
    }

    /// `clear_row` + reduce: cleared rows don't count toward rank.
    #[test]
    fn clear_row_drops_from_rank() {
        let mut m = F2Matrix::from_rows(vec![vec![0], vec![1], vec![2]]);
        m.clear_row(1);
        m.reduce();
        assert_eq!(m.rank(), 2);
    }

    /// Symmetric difference correctness check.
    #[test]
    fn xor_into_basic() {
        let mut out = Vec::new();
        F2Matrix::xor_into(&[1, 3, 5], &[3, 4, 5, 6], &mut out);
        assert_eq!(out, vec![1, 4, 6]);
    }

    /// `xor_into` with one empty slice returns the other slice verbatim.
    #[test]
    fn xor_into_empty_input() {
        let mut out = Vec::new();
        F2Matrix::xor_into(&[], &[2, 4], &mut out);
        assert_eq!(out, vec![2, 4]);
        F2Matrix::xor_into(&[1, 3], &[], &mut out);
        assert_eq!(out, vec![1, 3]);
    }

    /// `xor_into` of identical slices yields the empty row.
    #[test]
    fn xor_into_identical_inputs() {
        let mut out = Vec::new();
        F2Matrix::xor_into(&[1, 2, 3], &[1, 2, 3], &mut out);
        assert_eq!(out, Vec::<u32>::new());
    }

    /// `xor_into` reuses the buffer (clears existing contents).
    #[test]
    fn xor_into_clears_buffer() {
        let mut out = vec![99, 100, 101];
        F2Matrix::xor_into(&[1], &[2], &mut out);
        assert_eq!(out, vec![1, 2]);
    }

    /// Empty matrix: rank 0 and reduce is a no-op.
    #[test]
    fn rank_of_empty_matrix() {
        let mut m = F2Matrix::from_rows(vec![]);
        m.reduce();
        assert_eq!(m.rank(), 0);
    }

    /// All-zero rows: rank 0 even before reduce.
    #[test]
    fn rank_of_all_zero_rows() {
        let m = F2Matrix::from_rows(vec![vec![], vec![], vec![]]);
        assert_eq!(m.rank(), 0);
    }

    /// Calling reduce on a matrix that's already in echelon form is a no-op.
    #[test]
    fn reduce_idempotent_on_reduced_input() {
        // Already in echelon: row 0 leads with col 0, row 1 with col 1, row 2 with col 2.
        let mut m = F2Matrix::from_rows(vec![vec![0, 3], vec![1, 4], vec![2, 5]]);
        let before = format!("{m:?}");
        m.reduce();
        let after_first = format!("{m:?}");
        assert_eq!(before, after_first);
        m.reduce();
        let after_second = format!("{m:?}");
        assert_eq!(after_first, after_second);
        assert_eq!(m.rank(), 3);
    }

    /// reduce twice yields the same rank as reduce once.
    #[test]
    fn reduce_idempotent_on_unreduced_input() {
        let mut m = F2Matrix::from_rows(vec![vec![0, 1, 2], vec![1, 2, 3], vec![0, 3]]);
        m.reduce();
        let r1 = m.rank();
        m.reduce();
        let r2 = m.rank();
        assert_eq!(r1, r2);
    }

    /// Cloning a matrix and mutating the clone must not affect the original.
    /// This is the invariant the snapshot-based legality check relies on.
    #[test]
    fn clone_is_independent_from_original() {
        let m = F2Matrix::from_rows(vec![vec![0, 1], vec![2, 3]]);
        let mut snap = m.clone();
        snap.clear_row(0);
        snap.merge_columns(2, 3);
        snap.reduce();

        // Original is untouched.
        assert_eq!(m.rank(), 2);
        let mut m_check = m.clone();
        m_check.reduce();
        assert_eq!(m_check.rank(), 2);
    }

    /// Chain of column merges: merging (0, 1) then (0, 2) collapses three
    /// disjoint single-column rows into a single column 0.
    #[test]
    fn chained_merge_columns() {
        let mut m = F2Matrix::from_rows(vec![vec![0], vec![1], vec![2]]);
        m.merge_columns(0, 1);
        m.merge_columns(0, 2);
        // Every row should now have exactly column 0.
        for i in 0..3 {
            assert_eq!(m.rows[i], vec![0], "row {i} after chained merges");
        }
        m.reduce();
        // All three rows are equal → rank 1.
        assert_eq!(m.rank(), 1);
    }

    /// Column merges preserve the caller's chosen column, regardless of index order.
    #[test]
    fn merge_columns_honours_caller_keep_drop() {
        // Drop column 5 into column 1: data shifts to column 1.
        let mut a = F2Matrix::from_rows(vec![vec![5], vec![1, 5]]);
        a.merge_columns(1, 5);
        assert_eq!(a.rows[0], vec![1], "drop column moves into keep column");
        assert_eq!(
            a.rows[1],
            Vec::<u32>::new(),
            "row with both columns cancels under XOR"
        );

        // Drop column 1 into column 5: data shifts to column 5 (opposite).
        let mut b = F2Matrix::from_rows(vec![vec![1], vec![1, 5]]);
        b.merge_columns(5, 1);
        assert_eq!(b.rows[0], vec![5], "with reversed args the kept col is 5");
        assert_eq!(b.rows[1], Vec::<u32>::new());
    }

    /// `merge_columns` where the dropped column is empty in every row is a no-op.
    #[test]
    fn merge_columns_dropped_column_unused() {
        let original = vec![vec![0, 2], vec![1]];
        let mut m = F2Matrix::from_rows(original.clone());
        // Column 4 is not present in any row; merging it into column 0 should
        // leave the matrix unchanged.
        m.merge_columns(0, 4);
        assert_eq!(m.rows, original);
    }

    /// `merge_columns` followed by reduce: column merges can create new
    /// dependencies; rank should reflect that.
    #[test]
    fn merge_columns_then_reduce_creates_dependency() {
        // Two rows: {0, 1} and {0, 2}. They are independent → rank 2.
        let mut m = F2Matrix::from_rows(vec![vec![0, 1], vec![0, 2]]);
        let mut baseline = m.clone();
        baseline.reduce();
        assert_eq!(baseline.rank(), 2);

        // Merge columns 1 and 2: rows become {0, 1} and {0, 1} → rank 1.
        m.merge_columns(1, 2);
        m.reduce();
        assert_eq!(m.rank(), 1);
    }

    /// Clearing all rows leaves rank 0.
    #[test]
    fn clear_all_rows_zeros_rank() {
        let mut m = F2Matrix::from_rows(vec![vec![0], vec![1], vec![2]]);
        for i in 0..3 {
            m.clear_row(i);
        }
        m.reduce();
        assert_eq!(m.rank(), 0);
    }

    /// Tetrahedron ∂₂ over GF(2): 4 triangular faces, 6 edges.
    /// Each face's row is its three edges. The four rows sum to zero (every
    /// edge sits on exactly two faces), so rank = 3 and β₁ = 6 − 4 + 1 − 3 = 0,
    /// matching the genus-0 sphere topology of a tetrahedron's surface.
    #[test]
    fn tetrahedron_boundary_rank() {
        // Vertices 0, 1, 2, 3. Edge column indices:
        //   col 0: (0,1),  col 1: (0,2),  col 2: (0,3),
        //   col 3: (1,2),  col 4: (1,3),  col 5: (2,3).
        // Faces:
        //   (0,1,2) -> edges (0,1),(0,2),(1,2) -> cols {0, 1, 3}
        //   (0,1,3) -> edges (0,1),(0,3),(1,3) -> cols {0, 2, 4}
        //   (0,2,3) -> edges (0,2),(0,3),(2,3) -> cols {1, 2, 5}
        //   (1,2,3) -> edges (1,2),(1,3),(2,3) -> cols {3, 4, 5}
        let mut m = F2Matrix::from_rows(vec![
            vec![0, 1, 3],
            vec![0, 2, 4],
            vec![1, 2, 5],
            vec![3, 4, 5],
        ]);
        m.reduce();
        assert_eq!(m.rank(), 3);

        // Cross-check β₁ = E − V + 1 − rank for V=4, E=6: should be 0 (sphere).
        let v = 4i64;
        let e = 6i64;
        let beta_1 = e - v + 1 - m.rank() as i64;
        assert_eq!(beta_1, 0);
    }

    /// reduce orders pivots so each row's leading column is unique.
    /// After reduce, no two nonzero rows share a leading column.
    #[test]
    fn reduce_pivots_are_distinct() {
        let mut m = F2Matrix::from_rows(vec![
            vec![0, 1, 2],
            vec![0, 2, 3],
            vec![1, 3],
            vec![0, 1, 4],
        ]);
        m.reduce();
        let leading_cols: Vec<u32> = m
            .rows
            .iter()
            .filter(|r| !r.is_empty())
            .map(|r| r[0])
            .collect();
        let mut sorted = leading_cols.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            leading_cols.len(),
            "duplicate leading columns after reduce: {leading_cols:?}"
        );
    }

    /// Stress: a matrix where every pair of rows differs in exactly one
    /// column. The first row is the pivot for col 0; row k differs from row 0
    /// only in column k, so it's independent. With n rows, rank should be n.
    #[test]
    fn reduce_handles_long_elimination_chain() {
        let n = 8u32;
        let mut rows: Vec<Vec<u32>> = vec![(0..n).collect()];
        for k in 1..n {
            // Row k = row 0 XOR {col k} effectively → strip col k from row 0.
            let mut row: Vec<u32> = (0..n).filter(|&c| c != k).collect();
            row.sort();
            rows.push(row);
        }
        let mut m = F2Matrix::from_rows(rows);
        m.reduce();
        assert_eq!(m.rank(), n as usize);
    }
}
