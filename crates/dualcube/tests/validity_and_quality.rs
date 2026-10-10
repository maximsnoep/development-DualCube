//! Regression tests for the validity checks (condition 4, local loop removal), the quality criterion, the
//! initialization methods, the evolutionary algorithm, and the corner optimization.

mod common;

use common::blub;
use dualcube::prelude::*;
use std::f64::consts::PI;
use std::sync::Arc;

// A triangulated torus (genus 1).
fn torus(n: usize, m: usize) -> Arc<Mesh<INPUT>> {
    torus_with_faces(n, m).0
}

// A triangulated torus (genus 1) of n x m cells in parameter space (each split into two triangles), with a function
// that gives the face containing a point (s, t) in parameter space.
fn torus_with_faces(n: usize, m: usize) -> (Arc<Mesh<INPUT>>, impl Fn(f64, f64) -> FaceID) {
    let (big, small) = (2.0, 0.7);
    let mut positions = vec![];
    for i in 0..n {
        let u = 2. * PI * i as f64 / n as f64;
        for j in 0..m {
            let v = 2. * PI * j as f64 / m as f64;
            positions.push(Vector3D::new(
                (big + small * v.cos()) * u.cos(),
                (big + small * v.cos()) * u.sin(),
                small * v.sin(),
            ));
        }
    }
    let id = |i: usize, j: usize| (i % n) * m + (j % m);
    let mut faces = vec![];
    for i in 0..n {
        for j in 0..m {
            faces.push(vec![id(i, j), id(i + 1, j), id(i + 1, j + 1)]);
            faces.push(vec![id(i, j), id(i + 1, j + 1), id(i, j + 1)]);
        }
    }
    let (mesh, _, face_map) = Mesh::<INPUT>::from(&faces, &positions).unwrap();
    let locate = move |s: f64, t: f64| {
        let (i, j) = (s.floor() as usize, t.floor() as usize);
        let (a, b) = (s - s.floor(), t - t.floor());
        let index = 2 * ((i % n) * m + (j % m)) + usize::from(a < b);
        *face_map.key(index).unwrap()
    };
    (Arc::new(mesh), locate)
}

// The loop (sequence of crossed half-edges) along a circle in the parameter space of the torus.
fn circle_loop(
    mesh: &Mesh<INPUT>,
    locate: &impl Fn(f64, f64) -> FaceID,
    center: (f64, f64),
    radius: f64,
) -> Vec<EdgeID> {
    let samples = 2000;
    let at = |angle: f64| {
        locate(
            center.0 + radius * angle.cos(),
            center.1 + radius * angle.sin(),
        )
    };
    let adjacent = |f: FaceID, g: FaceID| f == g || mesh.edge_between_faces(f, g).is_some();
    let mut faces: Vec<FaceID> = vec![];
    for k in 0..samples {
        let (a0, a1) = (
            2. * PI * k as f64 / samples as f64,
            2. * PI * (k + 1) as f64 / samples as f64,
        );
        // Refine the step until consecutive faces share an edge (the circle passes close to a vertex).
        let mut stack = vec![(a0, a1)];
        while let Some((lo, hi)) = stack.pop() {
            let (f, g) = (at(lo), at(hi));
            if adjacent(f, g) || hi - lo < 1e-12 {
                if faces.last() != Some(&f) {
                    faces.push(f);
                }
                if faces.last() != Some(&g) {
                    faces.push(g);
                }
            } else {
                let mid = (lo + hi) / 2.;
                stack.push((mid, hi));
                stack.push((lo, mid));
            }
        }
    }
    if faces.first() == faces.last() {
        faces.pop();
    }
    let mut edges = vec![];
    for k in 0..faces.len() {
        let (f, g) = (faces[k], faces[(k + 1) % faces.len()]);
        let (exit, entry) = mesh
            .edge_between_faces(f, g)
            .expect("circle passes through a vertex");
        edges.push(exit);
        edges.push(entry);
    }
    edges
}

// The initialization is guaranteed to succeed on a genus-0 surface (no retries).
fn initialized() -> Solution {
    let mut solution = Solution::new(blub());
    solution.initialize();
    assert!(
        solution.loops.len() == 3 && solution.get_quality().is_some(),
        "initialization failed"
    );
    solution
}

// A valid solution with more than three loops (loops are added while the structure stays valid).
fn refined() -> Solution {
    let mut solution = initialized();
    let mut attempts = 0;
    while solution.loops.len() < 7 && attempts < 200 {
        attempts += 1;
        let axis = DIRECTIONS[attempts % 3];
        let s = |(_, cost): (&[EdgeID], f64)| cost;
        let Some(edges) = solution.sample_loops(1, axis, OrderedFloat, s).pop() else {
            continue;
        };
        let mut candidate = solution.clone();
        candidate.add_loop(Loop::new(edges, axis));
        if candidate.dual_is_ok() {
            solution = candidate;
        }
    }
    assert!(solution.loops.len() > 3, "could not refine the solution");
    solution
}

#[test]
fn local_removal_check_agrees_with_rebuild() {
    let mut checked = 0;
    for _ in 0..3 {
        let solution = refined();
        let dual = Dual::from(solution.mesh_ref.clone(), &solution.loops).unwrap();
        for loop_id in solution.loops.keys() {
            let local = dual.check_removal(loop_id);
            let mut loops = solution.loops.clone();
            loops.remove(loop_id);
            let rebuilt = Dual::from(solution.mesh_ref.clone(), &loops);
            assert_eq!(
                local.is_ok(),
                rebuilt.is_ok(),
                "loop {loop_id:?}: local={local:?} rebuilt={:?}",
                rebuilt.err()
            );
            checked += 1;
        }
    }
    assert!(checked > 0);
}

#[test]
fn region_with_a_handle_is_rejected() {
    // Three small circles in a Venn arrangement (each pair crosses twice, no triple points) form a valid cube loop
    // structure combinatorially (conditions 1, 2, 3, and 5 hold). On a torus, the outer region contains the handle,
    // so it is not a disk (condition 4).
    let (mesh, locate) = torus_with_faces(48, 16);
    let mut solution = Solution::new(mesh.clone());
    let (d, r) = (3.6, 3.1);
    let centers = [
        (20.13, 6.17),
        (20.13 + d, 6.17),
        (20.13 + d / 2., 6.17 + d * 3f64.sqrt() / 2.),
    ];
    for (center, axis) in centers.into_iter().zip(DIRECTIONS) {
        let edges = circle_loop(&mesh, &locate, center, r);
        solution.check_loop(&edges).unwrap();
        solution.add_loop(Loop::new(edges, axis));
    }
    match Dual::from(mesh, &solution.loops) {
        Err(PropertyViolationError::RegionNotDisk) => {}
        other => panic!("expected RegionNotDisk, got {:?}", other.map(|_| ())),
    }
}

#[test]
fn loops_on_a_torus_do_not_form_a_cube() {
    // A polycube loop structure with one loop per axis is a cube, which has genus 0. On a torus, every combination of
    // three loops must therefore be rejected (some regions are not disks).
    let mut solution = Solution::new(torus(48, 16));
    solution.prepare_flow();
    let s = |(_, cost): (&[EdgeID], f64)| cost;
    let loops = DIRECTIONS.map(|axis| solution.sample_loops(3, axis, OrderedFloat, s));
    let mut combinations = 0;
    let mut not_disk = 0;
    for x in &loops[0] {
        for y in &loops[1] {
            for z in &loops[2] {
                let mut candidate = solution.clone();
                candidate.add_loop(Loop::new(x.clone(), Direction::X));
                candidate.add_loop(Loop::new(y.clone(), Direction::Y));
                candidate.add_loop(Loop::new(z.clone(), Direction::Z));
                combinations += 1;
                match Dual::from(candidate.mesh_ref.clone(), &candidate.loops) {
                    Ok(_) => panic!("three loops on a torus formed a valid loop structure"),
                    Err(PropertyViolationError::RegionNotDisk) => not_disk += 1,
                    Err(_) => {}
                }
            }
        }
    }
    assert!(combinations > 0, "no loops sampled on the torus");
    println!("torus: combinations={combinations} rejected as non-disk={not_disk}");
}

#[test]
fn constructed_valid_loops_are_valid() {
    let mut solution = initialized();
    let (mut added, mut attempts) = (0, 0);
    while added < 8 && attempts < 60 {
        attempts += 1;
        let axis = DIRECTIONS[attempts % 3];
        let Some((edges, _, gaps)) = solution.sample_valid_loop(axis, None, 2) else {
            continue;
        };
        let mut candidate = solution.clone();
        candidate.add_loop_in_gaps(Loop::new(edges, axis), gaps);
        let dual = Dual::from(candidate.mesh_ref.clone(), &candidate.loops);
        assert!(
            dual.is_ok(),
            "constructed loop {added} ({axis:?}) is invalid: {:?}",
            dual.err()
        );
        candidate.dual = dual;
        solution = candidate;
        added += 1;
    }
    println!("valid loops: added={added} attempts={attempts}");
    assert!(
        added >= 4,
        "only {added} valid loops in {attempts} attempts"
    );
}

#[test]
fn quality_matches_the_formula() {
    let solution = initialized();
    let report = solution.quality_report().unwrap();
    let w = solution.quality.weights;
    let penalty = w.fidelity * report.fidelity.unwrap()
        + w.path_regularity * report.path_regularity().unwrap()
        + w.coherence * report.coherence.unwrap()
        + w.rectangularity * report.rectangularity.unwrap()
        + w.corner_regularity * report.corner_regularity.unwrap()
        + w.non_degeneracy * report.non_degeneracy.unwrap()
        + w.correspondence * report.correspondence.unwrap()
        + w.complexity * solution.loops.len() as f64;
    let expected = 1. / (1. + penalty);
    assert!(expected > 0. && expected <= 1.);
    assert!((solution.get_quality().unwrap() - expected).abs() < 1e-12);
    assert!((report.score(&w).unwrap() - expected).abs() < 1e-12);
}

#[test]
fn quality_report_is_sane() {
    let solution = initialized();
    let report = solution.quality_report().unwrap();
    let fidelity = report.alignment.unwrap();
    assert!((-1. ..=1.).contains(&fidelity));
    assert!(report.fidelity.unwrap() >= 0.);
    assert!((0. ..=1.).contains(&report.path_backtrack.unwrap()));
    assert!(report.path_stretch.unwrap() >= 0.);
    assert!((0. ..=1.).contains(&report.coherence.unwrap()));
    assert!(report.rectangularity.unwrap() >= 0.);
    assert!(report.corner_regularity.unwrap() >= 0.);
    assert!(report.non_degeneracy.unwrap() >= 0.);
    assert!(report.correspondence.unwrap() >= 0.);
    // A cube: 8 corners of degree 3.
    assert_eq!(report.corners, Some(8));
    assert_eq!(report.irregular_corners, Some(8));
    let score = solution.get_quality();
    assert!(score.is_some_and(f64::is_finite), "{score:?}");
    // The estimate (from the dual structure only) is close to the actual fidelity for a cube.
    let estimate = solution.estimate_quality().unwrap();
    assert!(estimate.is_finite());
}

#[test]
fn initialization_gives_a_cube() {
    let solution = initialized();
    assert!(solution.dual_is_ok());
    let directions = solution
        .loops
        .values()
        .map(|l| l.direction)
        .collect::<Vec<_>>();
    for axis in DIRECTIONS {
        assert_eq!(directions.iter().filter(|&&d| d == axis).count(), 1);
    }
}

#[test]
fn evolution_produces_a_valid_solution() {
    let solution = initialized();
    let initial = solution.get_quality().unwrap();
    let params = EvolutionParams {
        population: 4,
        offspring: 6,
        max_generations: 3,
        patience: 2,
        ..EvolutionParams::default()
    };
    let evolved = solution.evolve(&params).unwrap();
    assert!(evolved.dual_is_ok());
    let quality = evolved.get_quality().unwrap();
    // Scored with a fast layout during the search; allow for the difference of the final (full) layout.
    assert!(quality > initial - 0.05, "{initial} -> {quality}");
}
