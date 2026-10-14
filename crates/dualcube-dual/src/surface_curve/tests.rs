use super::*;

fn tetrahedron() -> Mesh<INPUT> {
    Mesh::from(
        &[vec![0, 2, 1], vec![0, 1, 3], vec![0, 3, 2], vec![1, 2, 3]],
        &[
            Vector3D::zeros(),
            Vector3D::x(),
            Vector3D::y(),
            Vector3D::z(),
        ],
    )
    .unwrap()
    .0
}

#[test]
fn converts_legacy_chords_without_moving_edge_crossings() {
    let mesh = tetrahedron();
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
    let surface = curve.surface_arcs(&mesh).unwrap();
    assert_eq!(surface.len(), 3);
    for (i, arc) in surface.iter().enumerate() {
        let actual = SurfaceArc::position(&mesh, arc.face, arc.from);
        let expected = mesh.midpoint_offset(curve.edges[2 * i], 0.2);
        assert!((actual - expected).norm() < 1e-14);
        let next = &surface[(i + 1) % surface.len()];
        assert!(
            (SurfaceArc::position(&mesh, arc.face, arc.to)
                - SurfaceArc::position(&mesh, next.face, next.from))
            .norm()
                < 1e-14
        );
    }
}

#[test]
fn retains_returning_arcs_and_repeated_triangle_visits() {
    let mesh = tetrahedron();
    let face = mesh.face_ids()[0];
    let surface = vec![
        SurfaceArc {
            face,
            from: [0.2, 0.],
            to: [0.2, 0.2],
        },
        SurfaceArc {
            face,
            from: [0.2, 0.2],
            to: [0.6, 0.],
        },
        SurfaceArc {
            face,
            from: [0.6, 0.],
            to: [0.4, 0.1],
        },
        SurfaceArc {
            face,
            from: [0.4, 0.1],
            to: [0.2, 0.],
        },
    ];
    let curve = Loop::from_surface(Direction::X, surface.clone());
    assert_eq!(curve.surface_arcs(&mesh).unwrap(), surface);
}

#[test]
fn reversing_an_interior_curve_reverses_every_segment() {
    let mesh = tetrahedron();
    let face = mesh.face_ids()[0];
    let points = [[0.2, 0.2], [0.5, 0.2], [0.2, 0.5]];
    let surface: Vec<_> = (0..3)
        .map(|i| SurfaceArc {
            face,
            from: points[i],
            to: points[(i + 1) % 3],
        })
        .collect();
    let mut curve = Loop::from_surface(Direction::Y, surface.clone());
    curve.reverse();
    assert_eq!(curve.surface[0].from, surface[2].to);
    assert_eq!(curve.surface[0].to, surface[2].from);
    curve.reverse();
    assert_eq!(curve.surface, surface);
}

#[test]
fn barycentric_positions_follow_the_face_vertex_order() {
    let mesh = tetrahedron();
    let face = mesh.face_ids()[0];
    let corners: Vec<_> = mesh.vertices(face).collect();
    let expected = mesh.position(corners[0]) * 0.5
        + mesh.position(corners[1]) * 0.2
        + mesh.position(corners[2]) * 0.3;
    assert!((SurfaceArc::position(&mesh, face, [0.2, 0.3]) - expected).norm() < 1e-14);
}

#[test]
fn interior_loop_cuts_a_disk_without_any_mesh_edge_crossing() {
    let mesh = tetrahedron();
    let face = mesh.face_ids()[0];
    let points = [[0.2, 0.2], [0.5, 0.2], [0.2, 0.5]];
    let mut loops = slotmap::SlotMap::with_key();
    loops.insert(Loop::from_surface(
        Direction::X,
        (0..3)
            .map(|i| SurfaceArc {
                face,
                from: points[i],
                to: points[(i + 1) % 3],
            })
            .collect(),
    ));
    let arrangement = crate::arrangement::Arrangement::new(&mesh, &loops).unwrap();
    let (components, count, _) = arrangement.components(&mesh).unwrap();
    assert_eq!(count, 2);
    let mut chi = arrangement.region_euler(&mesh, &components, count).unwrap();
    chi.sort();
    assert_eq!(chi, vec![1, 1]);
    let refined = arrangement.refine(&mesh).unwrap();
    let (mesh, _, _) = Mesh::<INPUT>::from(&refined.faces, &refined.positions).unwrap();
    assert_eq!(
        mesh.nr_verts() as i64 - mesh.nr_edges() as i64 / 2 + mesh.nr_faces() as i64,
        2
    );
}

#[test]
fn legacy_and_explicit_arrangements_have_identical_topology() {
    let mesh = tetrahedron();
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
    let mut loops = slotmap::SlotMap::with_key();
    let id = loops.insert(curve);
    let before = crate::partial::PartialTopology::new(&mesh, &loops).unwrap();
    loops[id] = Loop::from_surface(Direction::X, loops[id].surface_arcs(&mesh).unwrap());
    let after = crate::partial::PartialTopology::new(&mesh, &loops).unwrap();
    assert!(before.equivalent(&after));
}

#[test]
fn self_intersections_and_disconnected_curves_are_rejected() {
    let mesh = tetrahedron();
    let face = mesh.face_ids()[0];
    for points in [
        vec![[0.1, 0.1], [0.5, 0.4], [0.1, 0.4], [0.5, 0.1]],
        vec![[0.2, 0.2], [0.5, 0.2], [0.2, 0.5]],
    ] {
        let mut arcs: Vec<_> = (0..points.len())
            .map(|i| SurfaceArc {
                face,
                from: points[i],
                to: points[(i + 1) % points.len()],
            })
            .collect();
        if points.len() == 3 {
            arcs[0].to = [0.4, 0.2];
        }
        let mut loops = slotmap::SlotMap::with_key();
        loops.insert(Loop::from_surface(Direction::X, arcs));
        assert!(crate::partial::PartialTopology::new(&mesh, &loops).is_err());
    }
}

#[test]
fn routes_through_a_region_entirely_inside_one_triangle() {
    let mesh = tetrahedron();
    let face = mesh.face_ids()[0];
    let points = [[0.2, 0.2], [0.5, 0.2], [0.2, 0.5]];
    let mut loops = slotmap::SlotMap::with_key();
    let old = loops.insert(Loop::from_surface(
        Direction::X,
        (0..3)
            .map(|i| SurfaceArc {
                face,
                from: points[i],
                to: points[(i + 1) % 3],
            })
            .collect(),
    ));
    let (router, topology) = crate::surface_router::SurfaceRouter::with_topology(&mesh, &loops).unwrap();
    let separate = crate::surface_router::SurfaceRouter::new(&mesh, &loops).unwrap();
    assert_eq!(router.regions, separate.regions);
    assert!(topology.equivalent(&crate::partial::PartialTopology::new(&mesh, &loops).unwrap()));
    let mut found = false;
    for before in 0..2 {
        let after = 1 - before;
        for anchor in router.anchors(old, before, after) {
            let Ok(curve) = router.route(
                Direction::Y,
                &[old, old],
                &[after, before],
                anchor,
                |f, d| d.norm() * if f == face { 1. } else { 100. },
            ) else {
                continue;
            };
            let mut candidate = loops.clone();
            candidate.insert(curve.clone());
            let Ok(topology) = crate::partial::PartialTopology::new(&mesh, &candidate) else {
                continue;
            };
            if topology.crossings.len() == 2 && topology.regions == 4 {
                assert!(curve.surface.iter().all(|a| a.face == face));
                found = true;
                break;
            }
        }
        if found {
            break;
        }
    }
    assert!(found, "No route through the interior disk");
}
