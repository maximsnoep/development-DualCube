use super::*;
use std::collections::HashMap;

/// Corner tetrahedron with outward faces.
fn tetrahedron() -> (Triangles, Vec<Position>) {
    let faces = vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
    let positions = vec![
        Position::new(0., 0., 0.),
        Position::new(1., 0., 0.),
        Position::new(0., 1., 0.),
        Position::new(0., 0., 1.),
    ];
    let mut vertex_faces = vec![vec![]; positions.len()];
    let mut opposite: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (f, face) in faces.iter().enumerate() {
        for k in 0..3 {
            let (a, b, c) = (face[k], face[(k + 1) % 3], face[(k + 2) % 3]);
            vertex_faces[a].push(f);
            opposite.entry((a.min(b), a.max(b))).or_default().push(c);
        }
    }
    let neighbors = (0..positions.len())
        .map(|i| {
            (0..positions.len())
                .filter(|&j| j != i)
                .map(|j| {
                    let o = &opposite[&(i.min(j), i.max(j))];
                    (j, [o[0], o[1]])
                })
                .collect()
        })
        .collect();
    let triangles = Triangles {
        faces,
        vertex_faces,
        neighbors,
    };
    (triangles, positions)
}

#[test]
fn areas_and_volume_of_corner_tetrahedron() {
    let (triangles, positions) = tetrahedron();
    let areas = triangles.face_areas(&positions);
    for area in &areas[..3] {
        assert!((area - 0.5).abs() < 1e-15);
    }
    assert!((areas[3] - 3f64.sqrt() / 2.).abs() < 1e-15);
    assert!((triangles.one_ring_areas(&positions)[0] - 1.5).abs() < 1e-15);
    assert!((triangles.volume(&positions) - 1. / 6.).abs() < 1e-15);
}

#[test]
fn cotangent_laplacian_is_symmetric_with_zero_row_sums() {
    let (triangles, positions) = tetrahedron();
    let mut matrix = [[0.; 4]; 4];
    for (r, c, value) in triangles.cotangent_laplacian(&positions).unwrap() {
        matrix[r][c] += value;
    }
    for (r, row) in matrix.iter().enumerate() {
        assert!(row.iter().sum::<f64>().abs() < 1e-15);
        for (c, value) in row.iter().enumerate() {
            assert!((value - matrix[c][r]).abs() < 1e-15);
        }
    }
}

#[test]
fn nonfinite_laplacian_stops_instead_of_changing_sparsity() {
    let (triangles, positions) = tetrahedron();
    let huge: Vec<_> = positions.iter().map(|p| p * 1e200).collect();
    assert!(triangles.cotangent_laplacian(&huge).is_none());
}

#[test]
fn step_contracts_and_preserves_centroid_under_uniform_attraction() {
    let (triangles, mut positions) = tetrahedron();
    let centroid = |p: &[Position]| p.iter().sum::<Position>() / p.len() as f64;
    let before = centroid(&positions);
    let mut solver = ContractionSolver::default();
    let step = solver
        .step(&triangles, 1., &[1.; 4], &mut positions)
        .unwrap();
    assert_eq!(step, Step::Solved);
    assert!(solver.is_factorized());
    assert!(triangles.volume(&positions) < 1. / 6.);
    assert!((centroid(&positions) - before).norm() < 1e-12);
}

#[test]
fn nonfinite_weights_retain_positions() {
    let (triangles, mut positions) = tetrahedron();
    let original = positions.clone();
    let mut solver = ContractionSolver::default();
    let step = solver
        .step(&triangles, f64::INFINITY, &[1.; 4], &mut positions)
        .unwrap();
    assert_eq!(step, Step::NonFiniteWeights);
    assert_eq!(positions, original);
}
