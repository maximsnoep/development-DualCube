use super::*;
use faer::sparse::{SparseColMat, Triplet};

#[test]
fn qr_reuses_storage_for_changed_values_and_multiple_rhs() {
    let matrix = |scale| {
        SparseColMat::try_new_from_triplets(
            3,
            2,
            &[
                Triplet::new(0, 0, scale),
                Triplet::new(1, 1, scale),
                Triplet::new(2, 0, scale),
                Triplet::new(2, 1, scale),
            ],
        )
        .unwrap()
    };
    let mut qr = QrWorkspace::new(matrix(1.).as_ref(), 2).unwrap();
    for scale in [1., 2., 0.5] {
        let mut rhs = faer::mat![[1., 2.], [2., 4.], [4., 8.]];
        qr.solve(matrix(scale).as_ref(), &mut rhs);
        for (i, expected) in [4. / 3., 7. / 3.].into_iter().enumerate() {
            assert!((rhs[(i, 0)] - expected / scale).abs() < 1e-12);
            assert!((rhs[(i, 1)] - 2. * expected / scale).abs() < 1e-12);
        }
    }
}

#[test]
#[should_panic(expected = "assertion `left == right` failed")]
fn qr_rejects_rhs_that_exceeds_its_workspace() {
    let matrix = SparseColMat::try_new_from_triplets(1, 1, &[Triplet::new(0, 0, 1.)]).unwrap();
    let mut qr = QrWorkspace::new(matrix.as_ref(), 1).unwrap();
    qr.solve(matrix.as_ref(), &mut Mat::zeros(1, 2));
}
