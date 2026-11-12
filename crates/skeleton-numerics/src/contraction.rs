//! Mesh contraction kernels (Au et al. 2008) over index-based triangle topology.
//! Arithmetic follows `mehsh` operation for operation, so results match it bit for bit.

use crate::QrWorkspace;
use faer::{
    Mat,
    sparse::{CreationError, SparseColMat, Triplet},
};
use nalgebra::Vector3;

pub type Position = Vector3<f64>;

/// Shorter cross products (near-degenerate corners) contribute no cotangent.
const MIN_CROSS_LENGTH: f64 = 1e-8;

/// Fixed triangle topology; indices follow the caller's vertex and face order.
pub struct Triangles {
    /// Corners per face; the first is the origin of the area sum.
    pub faces: Vec<[usize; 3]>,
    /// Incident faces per vertex, in summation order.
    pub vertex_faces: Vec<Vec<usize>>,
    /// Per vertex: each neighbor and the two corners opposite their shared edge.
    pub neighbors: Vec<Vec<(usize, [usize; 2])>>,
}

impl Triangles {
    /// Newell area relative to each face's first corner.
    pub fn face_areas(&self, positions: &[Position]) -> Vec<f64> {
        self.faces
            .iter()
            .map(|face| {
                let origin = positions[face[0]];
                (0..3)
                    .fold(Position::zeros(), |sum, k| {
                        let u = positions[face[k]] - origin;
                        let v = positions[face[(k + 1) % 3]] - origin;
                        sum + u.cross(&v)
                    })
                    .magnitude()
                    / 2.0
            })
            .collect()
    }

    pub fn one_ring_areas(&self, positions: &[Position]) -> Vec<f64> {
        let areas = self.face_areas(positions);
        self.vertex_faces
            .iter()
            .map(|faces| faces.iter().map(|&f| areas[f]).sum())
            .collect()
    }

    /// Unsigned enclosed volume, relative to the first vertex to limit cancellation.
    pub fn volume(&self, positions: &[Position]) -> f64 {
        let Some(&origin) = positions.first() else {
            return 0.0;
        };
        self.faces
            .iter()
            .map(|face| {
                let [a, b, c] = face.map(|v| positions[v] - origin);
                a.dot(&b.cross(&c)) / 6.0
            })
            .sum::<f64>()
            .abs()
    }

    /// Cotangent Laplacian as `(row, col, value)`: `L_ij` = cot weight, `L_ii` = −Σ weights.
    /// `None` if a weight is non-finite. See Eq. (1).
    pub fn cotangent_laplacian(&self, positions: &[Position]) -> Option<Vec<(usize, usize, f64)>> {
        let entries = self.neighbors.iter().map(Vec::len).sum::<usize>() + self.neighbors.len();
        let mut triplets = Vec::with_capacity(entries);
        for (i, neighbors) in self.neighbors.iter().enumerate() {
            let mut sum_weights = 0.0;
            for &(j, opposite) in neighbors {
                let weight = cotangent_weight(positions, i, j, opposite);
                if !weight.is_finite() {
                    return None;
                }
                triplets.push((i, j, weight));
                sum_weights += weight;
            }
            if !sum_weights.is_finite() {
                return None;
            }
            triplets.push((i, i, -sum_weights));
        }
        Some(triplets)
    }
}

fn cotangent_weight(positions: &[Position], i: usize, j: usize, opposite: [usize; 2]) -> f64 {
    let (p_i, p_j) = (positions[i], positions[j]);
    let mut sum_cot = 0.0;
    for k in opposite {
        let p_k = positions[k];
        let u = p_i - p_k;
        let v = p_j - p_k;
        let cross_len = u.cross(&v).magnitude();
        if cross_len > MIN_CROSS_LENGTH {
            sum_cot += u.dot(&v) / cross_len;
        }
    }
    sum_cot
}

/// Outcome of one contraction solve; positions change only when `Solved`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    Solved,
    NonFiniteLaplacian,
    NonFiniteWeights,
    NonFiniteRightHandSide,
    NonFiniteSolution,
}

/// Contraction solver; reuses the QR symbolic factorization, as topology is fixed.
#[derive(Default)]
pub struct ContractionSolver {
    qr: Option<QrWorkspace>,
}

impl ContractionSolver {
    pub fn is_factorized(&self) -> bool {
        self.qr.is_some()
    }

    /// Least-squares solve of `[wl·L; diag(wh)] x = [0; diag(wh) p]`, writing x into positions.
    pub fn step(
        &mut self,
        triangles: &Triangles,
        wl: f64,
        wh: &[f64],
        positions: &mut [Position],
    ) -> Result<Step, CreationError> {
        let n = positions.len();
        let Some(laplacian) = triangles.cotangent_laplacian(positions) else {
            return Ok(Step::NonFiniteLaplacian);
        };

        // Top block W_L·L, bottom block diagonal W_H.
        let mut triplets = Vec::with_capacity(laplacian.len() + n);
        triplets.extend(
            laplacian
                .into_iter()
                .map(|(r, c, value)| Triplet::new(r, c, wl * value)),
        );
        triplets.extend(
            wh.iter()
                .enumerate()
                .map(|(i, &w)| Triplet::new(n + i, i, w)),
        );
        if triplets.iter().any(|entry| !entry.val.is_finite()) {
            return Ok(Step::NonFiniteWeights);
        }
        let matrix = SparseColMat::try_new_from_triplets(2 * n, n, &triplets)?;

        let mut rhs = Mat::from_fn(2 * n, 3, |r, c| {
            if r < n {
                0.0
            } else {
                wh[r - n] * positions[r - n][c]
            }
        });
        if (0..rhs.nrows()).any(|i| (0..rhs.ncols()).any(|j| !rhs[(i, j)].is_finite())) {
            return Ok(Step::NonFiniteRightHandSide);
        }

        let qr = match &mut self.qr {
            Some(qr) => qr,
            None => self.qr.insert(QrWorkspace::new(matrix.as_ref(), 3)?),
        };
        qr.solve(matrix.as_ref(), &mut rhs);
        if (0..n).any(|i| (0..3).any(|j| !rhs[(i, j)].is_finite())) {
            return Ok(Step::NonFiniteSolution);
        }

        for (i, position) in positions.iter_mut().enumerate() {
            *position = Position::new(rhs[(i, 0)], rhs[(i, 1)], rhs[(i, 2)]);
        }
        Ok(Step::Solved)
    }
}

#[cfg(test)]
mod tests;
