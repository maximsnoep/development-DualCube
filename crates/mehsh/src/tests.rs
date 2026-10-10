use crate::prelude::*;
#[cfg(any(feature = "obj", feature = "stl"))]
use std::path::PathBuf;
define_tag!(TestMesh);

#[test]
fn from_manual() {
    let faces = vec![vec![0, 2, 1], vec![0, 1, 3], vec![1, 2, 3], vec![0, 3, 2]];
    let douconel = Mesh::<TestMesh>::from(&faces, &[Vector3D::new(0., 0., 0.); 4]);
    assert!(douconel.is_ok(), "{douconel:?}");
    if let Ok((douconel, _, _)) = douconel {
        assert!(douconel.nr_verts() == 4);
        assert!(douconel.nr_edges() == 6 * 2);
        assert!(douconel.nr_faces() == 4);

        for face_id in douconel.faces.ids() {
            assert!(douconel.vertices(face_id).count() == 3);
        }
    }
}

#[cfg(feature = "stl")]
#[test]
fn from_blub_stl() {
    let douconel = Mesh::<TestMesh>::from_stl(&PathBuf::from("assets/blub001k.stl"));
    assert!(douconel.is_ok(), "{douconel:?}");
    if let Ok((douconel, _, _)) = douconel {
        assert!(douconel.nr_verts() == 945);
        assert!(douconel.nr_edges() == 2829 * 2);
        assert!(douconel.nr_faces() == 1886);

        for face_id in douconel.faces.ids() {
            assert!(douconel.vertices(face_id).count() == 3);
        }
    }
}

#[cfg(feature = "obj")]
#[test]
fn from_blub_obj() {
    let douconel = Mesh::<TestMesh>::from_obj(&PathBuf::from("assets/blub001k.obj"));
    assert!(douconel.is_ok(), "{douconel:?}");
    if let Ok((douconel, _, _)) = douconel {
        assert!(douconel.nr_verts() == 945);
        assert!(douconel.nr_edges() == 2829 * 2);
        assert!(douconel.nr_faces() == 1886);

        for face_id in douconel.faces.ids() {
            assert!(douconel.vertices(face_id).count() == 3);
        }
    }
}

#[cfg(feature = "stl")]
#[test]
fn from_nefertiti_stl() {
    let douconel = Mesh::<TestMesh>::from_stl(&PathBuf::from("assets/nefertiti099k.stl"));
    assert!(douconel.is_ok(), "{douconel:?}");
    if let Ok((douconel, _, _)) = douconel {
        assert!(douconel.nr_verts() == 49971);
        assert!(douconel.nr_edges() == 149_907 * 2);
        assert!(douconel.nr_faces() == 99938);

        for face_id in douconel.faces.ids() {
            assert!(douconel.vertices(face_id).count() == 3);
        }
    }
}

#[cfg(feature = "obj")]
#[test]
fn from_hexahedron_obj() {
    let douconel = Mesh::<TestMesh>::from_obj(&PathBuf::from("assets/hexahedron.obj"));
    assert!(douconel.is_ok(), "{douconel:?}");
    if let Ok((douconel, _, _)) = douconel {
        assert!(douconel.nr_verts() == 8);
        assert!(douconel.nr_edges() == 4 * 6);
        assert!(douconel.nr_faces() == 6);

        for face_id in douconel.faces.ids() {
            assert!(douconel.vertices(face_id).count() == 4);
        }
    }
}

#[cfg(feature = "obj")]
#[test]
fn from_tetrahedron_obj() {
    let douconel = Mesh::<TestMesh>::from_obj(&PathBuf::from("assets/tetrahedron.obj"));
    assert!(douconel.is_ok(), "{douconel:?}");
    if let Ok((douconel, _, _)) = douconel {
        assert!(douconel.nr_verts() == 4);
        assert!(douconel.nr_edges() == 3 * 4);
        assert!(douconel.nr_faces() == 4);

        for face_id in douconel.faces.ids() {
            assert!(douconel.vertices(face_id).count() == 3);
        }
    }
}

#[test]
fn triangulate_hexahedron() {
    let (mesh, _, _) =
        Mesh::<TestMesh>::from(&hexahedron_faces(), &hexahedron_positions()).unwrap();
    assert!(mesh.is_triangular().is_err());

    let (triangulated, face_sources) = mesh.triangulate().unwrap();
    assert!(triangulated.is_triangular().is_ok());
    assert!(triangulated.nr_verts() == 8);
    assert!(triangulated.nr_edges() == 18 * 2);
    assert!(triangulated.nr_faces() == 12);
    assert!(face_sources.len() == triangulated.nr_faces());
}

#[test]
fn cut_and_cap_preserves_original_vertex_ids() {
    let (mesh, vertex_map, _) =
        Mesh::<TestMesh>::from(&hexahedron_faces(), &hexahedron_positions()).unwrap();

    let v0 = *vertex_map.key(0).unwrap();
    let v1 = *vertex_map.key(1).unwrap();
    let v2 = *vertex_map.key(2).unwrap();
    let v3 = *vertex_map.key(3).unwrap();
    let v4 = *vertex_map.key(4).unwrap();
    let v5 = *vertex_map.key(5).unwrap();
    let v6 = *vertex_map.key(6).unwrap();
    let v7 = *vertex_map.key(7).unwrap();

    let cut_loop = vec![v4, v5, v6, v7, v4];
    let cap_triangles = vec![[v4, v5, v6], [v4, v6, v7]];

    let (mesh_a, mesh_b) = mesh.cut_and_cap(&cut_loop, &cap_triangles).unwrap();
    let (top, bottom) = if mesh_a.nr_verts() == 4 {
        (mesh_a, mesh_b)
    } else {
        (mesh_b, mesh_a)
    };

    assert_eq!(top.nr_verts(), 4);
    assert_eq!(top.nr_faces(), 3);
    assert_eq!(bottom.nr_verts(), 8);
    assert_eq!(bottom.nr_faces(), 7);

    for vertex in [v4, v5, v6, v7] {
        assert!(top.verts.contains(vertex));
        assert!(bottom.verts.contains(vertex));
        assert_eq!(top.position(vertex), mesh.position(vertex));
        assert_eq!(bottom.position(vertex), mesh.position(vertex));
    }

    for vertex in [v0, v1, v2, v3] {
        assert!(!top.verts.contains(vertex));
        assert!(bottom.verts.contains(vertex));
        assert_eq!(bottom.position(vertex), mesh.position(vertex));
    }
}

#[test]
fn cut_and_cap_from_positions_stitches_boundary_and_adds_interior_vertices() {
    let (mesh, vertex_map, _) =
        Mesh::<TestMesh>::from(&hexahedron_faces(), &hexahedron_positions()).unwrap();

    let v4 = *vertex_map.key(4).unwrap();
    let v5 = *vertex_map.key(5).unwrap();
    let v6 = *vertex_map.key(6).unwrap();
    let v7 = *vertex_map.key(7).unwrap();
    let center = Vector3D::new(0.5, 1.0, 0.5);

    let mut cap_positions = vec![Vector3D::zeros(); 12];
    cap_positions[2] = mesh.position(v4);
    cap_positions[5] = mesh.position(v5);
    cap_positions[7] = mesh.position(v6);
    cap_positions[11] = mesh.position(v7);
    cap_positions[9] = center;

    let cut_loop = vec![v4, v5, v6, v7];
    let cap_triangles = vec![[2, 5, 9], [5, 7, 9], [7, 11, 9], [11, 2, 9]];

    let output = mesh
        .cut_and_cap_from_positions(&cut_loop, &cap_positions, &cap_triangles)
        .unwrap();

    assert_eq!(*output.cap_vertices_a.key(2).unwrap(), v4);
    assert_eq!(*output.cap_vertices_a.key(5).unwrap(), v5);
    assert_eq!(*output.cap_vertices_a.key(7).unwrap(), v6);
    assert_eq!(*output.cap_vertices_a.key(11).unwrap(), v7);
    assert_eq!(*output.cap_vertices_b.key(2).unwrap(), v4);
    assert_eq!(*output.cap_vertices_b.key(5).unwrap(), v5);
    assert_eq!(*output.cap_vertices_b.key(7).unwrap(), v6);
    assert_eq!(*output.cap_vertices_b.key(11).unwrap(), v7);

    let center_a = *output.cap_vertices_a.key(9).unwrap();
    let center_b = *output.cap_vertices_b.key(9).unwrap();
    assert!(!mesh.verts.contains(center_a));
    assert!(!mesh.verts.contains(center_b));
    assert_eq!(output.mesh_a.position(center_a), center);
    assert_eq!(output.mesh_b.position(center_b), center);

    assert!(output.mesh_a.verts.contains(v4));
    assert!(output.mesh_b.verts.contains(v4));
}

#[test]
fn triangulate_pentagonal_pyramid() {
    let positions = vec![
        Vector3D::new(1.0, 0.0, 0.0),
        Vector3D::new(0.309_016_994, 0.951_056_516, 0.0),
        Vector3D::new(-0.809_016_994, 0.587_785_252, 0.0),
        Vector3D::new(-0.809_016_994, -0.587_785_252, 0.0),
        Vector3D::new(0.309_016_994, -0.951_056_516, 0.0),
        Vector3D::new(0.0, 0.0, 1.0),
    ];
    let faces = vec![
        vec![0, 1, 2, 3, 4],
        vec![1, 0, 5],
        vec![2, 1, 5],
        vec![3, 2, 5],
        vec![4, 3, 5],
        vec![0, 4, 5],
    ];

    let (mesh, _, _) = Mesh::<TestMesh>::from(&faces, &positions).unwrap();
    let (triangulated, face_sources) = mesh.triangulate().unwrap();

    assert!(triangulated.is_triangular().is_ok());
    assert!(triangulated.nr_verts() == 6);
    assert!(triangulated.nr_edges() == 12 * 2);
    assert!(triangulated.nr_faces() == 8);
    assert!(face_sources.len() == triangulated.nr_faces());
}

#[test]
fn triangulate_concave_prism() {
    let (mesh, _, _) =
        Mesh::<TestMesh>::from(&concave_prism_faces(), &concave_prism_positions()).unwrap();
    let (triangulated, _) = mesh.triangulate().unwrap();

    assert!(triangulated.is_triangular().is_ok());
    assert!(triangulated.nr_verts() == 12);
    assert!(triangulated.nr_edges() == 30 * 2);
    assert!(triangulated.nr_faces() == 20);
}

fn hexahedron_faces() -> Vec<Vec<usize>> {
    vec![
        vec![0, 1, 2, 3],
        vec![7, 6, 5, 4],
        vec![0, 4, 5, 1],
        vec![1, 5, 6, 2],
        vec![2, 6, 7, 3],
        vec![3, 7, 4, 0],
    ]
}

fn hexahedron_positions() -> Vec<Vector3D> {
    vec![
        Vector3D::new(0.0, 0.0, 0.0),
        Vector3D::new(1.0, 0.0, 0.0),
        Vector3D::new(1.0, 0.0, 1.0),
        Vector3D::new(0.0, 0.0, 1.0),
        Vector3D::new(0.0, 1.0, 0.0),
        Vector3D::new(1.0, 1.0, 0.0),
        Vector3D::new(1.0, 1.0, 1.0),
        Vector3D::new(0.0, 1.0, 1.0),
    ]
}

fn concave_prism_faces() -> Vec<Vec<usize>> {
    vec![
        vec![5, 4, 3, 2, 1, 0],
        vec![6, 7, 8, 9, 10, 11],
        vec![0, 1, 7, 6],
        vec![1, 2, 8, 7],
        vec![2, 3, 9, 8],
        vec![3, 4, 10, 9],
        vec![4, 5, 11, 10],
        vec![5, 0, 6, 11],
    ]
}

fn concave_prism_positions() -> Vec<Vector3D> {
    vec![
        Vector3D::new(0.0, 0.0, 0.0),
        Vector3D::new(2.0, 0.0, 0.0),
        Vector3D::new(2.0, 1.0, 0.0),
        Vector3D::new(1.0, 1.0, 0.0),
        Vector3D::new(1.0, 2.0, 0.0),
        Vector3D::new(0.0, 2.0, 0.0),
        Vector3D::new(0.0, 0.0, 1.0),
        Vector3D::new(2.0, 0.0, 1.0),
        Vector3D::new(2.0, 1.0, 1.0),
        Vector3D::new(1.0, 1.0, 1.0),
        Vector3D::new(1.0, 2.0, 1.0),
        Vector3D::new(0.0, 2.0, 1.0),
    ]
}

/// Closed prism whose two caps are `n`-gons.
fn prism(n: usize) -> (Vec<Vec<usize>>, Vec<Vector3D>) {
    let mut positions = vec![];
    for z in [0., 1.] {
        for i in 0..n {
            let a = std::f64::consts::TAU * i as f64 / n as f64;
            positions.push(Vector3D::new(a.cos(), a.sin(), z));
        }
    }
    let mut faces = vec![(0..n).rev().collect::<Vec<_>>(), (n..2 * n).collect()];
    for i in 0..n {
        let j = (i + 1) % n;
        faces.push(vec![i, j, n + j, n + i]);
    }
    (faces, positions)
}

#[test]
fn from_accepts_large_polygons() {
    let (faces, positions) = prism(16);
    let mesh = Mesh::<TestMesh>::from(&faces, &positions);
    assert!(mesh.is_ok(), "{mesh:?}");
}

#[test]
fn from_rejects_out_of_range_indices() {
    let faces = vec![vec![0, 1, 7]];
    let positions = vec![Vector3D::new(0., 0., 0.); 3];
    assert!(Mesh::<TestMesh>::from(&faces, &positions).is_err());
}

#[test]
fn face_area_and_normal() {
    let (faces, positions) = prism(4);
    let (mesh, _, _) = Mesh::<TestMesh>::from(&faces, &positions).unwrap();
    // The cap of a unit-circle square has area 2; every side quad has area sqrt(2).
    let mut areas = mesh
        .face_ids()
        .iter()
        .map(|&f| mesh.size(f))
        .collect::<Vec<_>>();
    areas.sort_by(f64::total_cmp);
    for a in &areas[..4] {
        assert!((a - 2f64.sqrt()).abs() < 1e-9, "{areas:?}");
    }
    for a in &areas[4..] {
        assert!((a - 2.).abs() < 1e-9, "{areas:?}");
    }
    // Normals point outwards and agree with the vector area.
    for &f in &mesh.face_ids() {
        let n = mesh.normal(f);
        assert!((n.norm() - 1.).abs() < 1e-9);
        assert!(n.dot(&(mesh.position(f) - Vector3D::new(0., 0., 0.5))) > 0.);
        assert!(n.dot(&mesh.vector_area(f)) > 0.);
    }
}

#[test]
fn degenerate_face_has_finite_normal() {
    let faces = vec![vec![0, 2, 1], vec![0, 1, 3], vec![1, 2, 3], vec![0, 3, 2]];
    let (mesh, _, _) = Mesh::<TestMesh>::from(&faces, &[Vector3D::new(0., 0., 0.); 4]).unwrap();
    for &f in &mesh.face_ids() {
        assert!(mesh.normal(f).iter().all(|c| c.is_finite()));
    }
    for &v in &mesh.vert_ids() {
        assert!(mesh.normal(v).iter().all(|c| c.is_finite()));
    }
}

#[test]
fn point_on_triangle_matches_brute_force() {
    let t = (
        Vector3D::new(20.3, 19.1, 21.7),
        Vector3D::new(23.9, 18.2, 20.1),
        Vector3D::new(21.1, 22.6, 19.4),
    );
    // Deterministic pseudo-random query points around the triangle.
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut rnd = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    for _ in 0..500 {
        let p = Vector3D::new(17. + 10. * rnd(), 15. + 10. * rnd(), 16. + 10. * rnd());
        let q = geom::point_on_triangle(p, t);
        // Brute force over a fine barycentric grid.
        let steps = 200;
        let mut best = f64::MAX;
        for i in 0..=steps {
            for j in 0..=steps - i {
                let (u, v) = (i as f64 / steps as f64, j as f64 / steps as f64);
                let r = t.0 + (t.1 - t.0) * u + (t.2 - t.0) * v;
                best = best.min((p - r).norm());
            }
        }
        let d = (p - q).norm();
        assert!(d <= best + 1e-9, "closest point not optimal: {d} > {best}");
        assert!(d >= best - 0.05, "closest point too close?! {d} < {best}");
    }
}

#[test]
fn barycentric_coordinates_are_scale_invariant() {
    for scale in [1e3, 1., 1e-3, 1e-6] {
        let t = (
            Vector3D::new(0., 0., 0.) * scale,
            Vector3D::new(1., 0., 0.) * scale,
            Vector3D::new(0., 1., 0.) * scale,
        );
        let p = Vector3D::new(0.2, 0.3, 0.) * scale;
        let (u, v, w) = geom::calculate_barycentric_coordinates(p, t);
        assert!(
            (u - 0.5).abs() < 1e-9 && (v - 0.2).abs() < 1e-9 && (w - 0.3).abs() < 1e-9,
            "scale {scale}: {u} {v} {w}"
        );
        assert!(geom::is_point_inside_triangle(p, t), "scale {scale}");
        assert!(
            !geom::is_point_inside_triangle(Vector3D::new(0.8, 0.8, 0.) * scale, t),
            "scale {scale}"
        );
    }
}

#[test]
fn barycentric_coordinates_degenerate_uses_longest_edge() {
    // Collinear triangle whose longest edge is CA.
    let t = (
        Vector3D::new(0., 0., 0.),
        Vector3D::new(0.1, 0., 0.),
        Vector3D::new(1., 0., 0.),
    );
    let (u, v, w) = geom::calculate_barycentric_coordinates(Vector3D::new(0.5, 0., 0.), t);
    let q = t.0 * u + t.1 * v + t.2 * w;
    assert!(
        (q - Vector3D::new(0.5, 0., 0.)).norm() < 1e-12,
        "{u} {v} {w}"
    );
}
