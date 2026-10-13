use dualcube::prelude::*;
use io::{Dc, Export, Import, Loops};
use std::sync::Arc;

fn solution() -> Solution {
    let mesh = Mesh::from(
        &[vec![0, 2, 1], vec![0, 1, 3], vec![0, 3, 2], vec![1, 2, 3]],
        &[
            Vector3D::zeros(),
            Vector3D::x(),
            Vector3D::y(),
            Vector3D::z(),
        ],
    )
    .unwrap()
    .0;
    let face = mesh.face_ids()[0];
    let points = [[0.2, 0.2], [0.5, 0.2], [0.2, 0.5]];
    let mut solution = Solution::new(Arc::new(mesh));
    solution.add_loop(Loop::from_surface(
        Direction::X,
        (0..3)
            .map(|i| SurfaceArc {
                face,
                from: points[i],
                to: points[(i + 1) % 3],
            })
            .collect(),
    ));
    solution
}

fn roundtrip<T: Export + Import>(extension: &str) {
    let source = solution();
    let before: Vec<_> = source
        .loops
        .keys()
        .flat_map(|id| source.get_coordinates_of_loop(id))
        .collect();
    let path = std::env::temp_dir().join(format!(
        "dualcube-surface-{}.{extension}",
        std::process::id()
    ));
    T::export(&source, &path).unwrap();
    let loaded = T::import(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    assert!(loaded.has_surface_curves());
    assert_eq!(loaded.loops.len(), source.loops.len());
    let after: Vec<_> = loaded
        .loops
        .keys()
        .flat_map(|id| loaded.get_coordinates_of_loop(id))
        .collect();
    assert_eq!(before.len(), after.len());
    for (a, b) in before.iter().zip(after) {
        assert!((a - b).norm() < 1e-12);
    }
    let a = dualcube::dual::partial::PartialTopology::new(&source.mesh_ref, &source.loops).unwrap();
    let b = dualcube::dual::partial::PartialTopology::new(&loaded.mesh_ref, &loaded.loops).unwrap();
    assert!(a.equivalent(&b));
}

#[test]
fn dc_preserves_surface_curves() {
    roundtrip::<Dc>("dc");
}
#[test]
fn loops_preserves_surface_curves() {
    roundtrip::<Loops>("loops");
}

#[test]
fn explicit_curves_render_and_reject_legacy_evolution() {
    let solution = solution();
    let id = solution.loops.keys().next().unwrap();
    let frames = solution.loop_frames(id);
    assert_eq!(frames.len(), 3);
    assert!(frames.iter().all(|f| {
        f.position
            .iter()
            .chain(f.normal.iter())
            .chain(f.negative.iter())
            .all(|v| v.is_finite())
    }));
    assert!(matches!(
        solution.evolve(&EvolutionParams::default()),
        Err(SolutionError::SurfaceCurveEvolution)
    ));
}

#[test]
fn reads_headerless_legacy_loops() {
    #[derive(bitcode::Encode)]
    struct Legacy {
        loops: Vec<(u8, Vec<(usize, usize)>)>,
        faces: Vec<Vec<usize>>,
        verts: Vec<(f64, f64, f64)>,
    }
    let legacy = Legacy {
        loops: vec![(0, vec![(0, 2), (1, 0), (0, 1), (3, 0), (0, 3), (2, 0)])],
        faces: vec![vec![0, 2, 1], vec![0, 1, 3], vec![0, 3, 2], vec![1, 2, 3]],
        verts: vec![(0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)],
    };
    let path = std::env::temp_dir().join(format!("dualcube-legacy-{}.loops", std::process::id()));
    let encoded = bitcode::encode(&legacy);
    std::fs::write(
        &path,
        zstd::stream::encode_all(std::io::Cursor::new(encoded), 0).unwrap(),
    )
    .unwrap();
    let loaded = Loops::import(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    assert!(!loaded.has_surface_curves());
    assert_eq!(loaded.loops.len(), 1);
    let topology =
        dualcube::dual::partial::PartialTopology::new(&loaded.mesh_ref, &loaded.loops).unwrap();
    assert_eq!(topology.regions, 2);
}

#[test]
fn loops_preserves_legacy_offsets() {
    let mut source = solution();
    let mesh = &source.mesh_ref;
    let first = mesh.edge_ids()[0];
    let mut edge = first;
    let mut curve = Loop::new(vec![], Direction::X);
    loop {
        let exit = mesh.next(mesh.next(edge));
        curve.edges.extend([edge, exit]);
        curve.offsets.extend([0.2, 0.8]);
        edge = mesh.twin(exit);
        if edge == first {
            break;
        }
    }
    let expected: Vec<_> = curve
        .edges
        .iter()
        .zip(&curve.offsets)
        .map(|(&e, &t)| mesh.midpoint_offset(e, t))
        .collect();
    let mut loops = slotmap::SlotMap::with_key();
    loops.insert(curve);
    source.replace_loops(loops);
    let path = std::env::temp_dir().join(format!("dualcube-offsets-{}.loops", std::process::id()));
    Loops::export(&source, &path).unwrap();
    let loaded = Loops::import(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    let curve = loaded.loops.values().next().unwrap();
    for ((&edge, &offset), expected) in curve.edges.iter().zip(&curve.offsets).zip(expected) {
        assert!((loaded.mesh_ref.midpoint_offset(edge, offset) - expected).norm() < 1e-12);
    }
}
