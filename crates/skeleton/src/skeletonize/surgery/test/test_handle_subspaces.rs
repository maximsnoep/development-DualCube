#[cfg(test)]
mod tests {
    use crate::skeletonize::surgery::{
        handle_subspace::compute_handle_subspace, tet_boundary::build_tet_boundary,
        tetrahedralize::tetrahedralize,
    };
    use dualcube_types::prelude::INPUT;

    use mehsh::{mesh::connectivity::Mesh, prelude::Vector3D};

    fn make_mesh(positions: Vec<Vector3D>, faces: Vec<Vec<usize>>) -> Mesh<INPUT> {
        let (mesh, _, _) = Mesh::<INPUT>::from(&faces, &positions).expect("mesh build failed");
        mesh
    }

    #[test]
    fn excess_handle_generators_are_not_truncated_to_expected_genus() {
        use crate::skeletonize::surgery::tetrahedralize::TetMesh;
        use mehsh::prelude::{HasPosition, HasVertices};
        use std::collections::HashMap;

        let mesh = build_torus_surface(6, 4);
        let ids = mesh.vert_ids();
        let indices: HashMap<_, _> = ids
            .iter()
            .enumerate()
            .map(|(i, &v)| (v, i as u32))
            .collect();
        let apex = ids.len() as u32;
        let mut vertices: Vec<_> = ids.iter().map(|&v| mesh.position(v)).collect();
        vertices.push(Vector3D::zeros());
        // The cone on a torus kills both cycles, unlike a valid solid torus.
        let tets = mesh
            .face_ids()
            .into_iter()
            .map(|f| {
                let v: Vec<_> = mesh.vertices(f).map(|v| indices[&v]).collect();
                [v[0], v[1], v[2], apex]
            })
            .collect();
        let boundary = build_tet_boundary(&TetMesh {
            vertices,
            tets,
            n_input_vertices: ids.len(),
        });

        let handles = compute_handle_subspace(&mesh, &boundary, 1);
        assert_eq!(handles.genus(), 2);
    }

    fn build_tetrahedron_surface() -> Mesh<INPUT> {
        let positions = vec![
            Vector3D::new(0.0, 0.0, 0.0),
            Vector3D::new(1.0, 0.0, 0.0),
            Vector3D::new(0.0, 1.0, 0.0),
            Vector3D::new(0.0, 0.0, 1.0),
        ];
        let faces = vec![vec![0, 2, 1], vec![0, 1, 3], vec![0, 3, 2], vec![1, 2, 3]];
        make_mesh(positions, faces)
    }

    fn build_cube_surface() -> Mesh<INPUT> {
        let positions = vec![
            Vector3D::new(0.0, 0.0, 0.0),
            Vector3D::new(1.0, 0.0, 0.0),
            Vector3D::new(1.0, 1.0, 0.0),
            Vector3D::new(0.0, 1.0, 0.0),
            Vector3D::new(0.0, 0.0, 1.0),
            Vector3D::new(1.0, 0.0, 1.0),
            Vector3D::new(1.0, 1.0, 1.0),
            Vector3D::new(0.0, 1.0, 1.0),
        ];
        let faces = vec![
            vec![0, 2, 1],
            vec![0, 3, 2],
            vec![4, 5, 6],
            vec![4, 6, 7],
            vec![0, 1, 5],
            vec![0, 5, 4],
            vec![3, 7, 6],
            vec![3, 6, 2],
            vec![0, 4, 7],
            vec![0, 7, 3],
            vec![1, 2, 6],
            vec![1, 6, 5],
        ];
        make_mesh(positions, faces)
    }

    /// A small triangulated torus. Parameters: `n_major` divisions around
    /// the big circle, `n_minor` divisions around the tube. Outer/inner
    /// radii baked in. Output is a closed, oriented genus-1 surface.
    fn build_torus_surface(n_major: usize, n_minor: usize) -> Mesh<INPUT> {
        use std::f64::consts::PI;
        let r_major = 2.0;
        let r_minor = 0.6;
        let mut positions: Vec<Vector3D> = Vec::with_capacity(n_major * n_minor);
        let idx = |i: usize, j: usize| -> usize { (i % n_major) * n_minor + (j % n_minor) };
        for i in 0..n_major {
            let theta = 2.0 * PI * (i as f64) / (n_major as f64);
            let (ct, st) = (theta.cos(), theta.sin());
            for j in 0..n_minor {
                let phi = 2.0 * PI * (j as f64) / (n_minor as f64);
                let (cp, sp) = (phi.cos(), phi.sin());
                let x = (r_major + r_minor * cp) * ct;
                let y = (r_major + r_minor * cp) * st;
                let z = r_minor * sp;
                positions.push(Vector3D::new(x, y, z));
            }
        }
        let mut faces: Vec<Vec<usize>> = Vec::new();
        for i in 0..n_major {
            for j in 0..n_minor {
                let v00 = idx(i, j);
                let v10 = idx(i + 1, j);
                let v01 = idx(i, j + 1);
                let v11 = idx(i + 1, j + 1);
                // Two triangles per grid cell, consistent outward normal.
                faces.push(vec![v00, v10, v11]);
                faces.push(vec![v00, v11, v01]);
            }
        }
        make_mesh(positions, faces)
    }

    /// A tetrahedron (genus 0) has trivial handle subspace.
    #[test]
    #[ignore = "requires a TetGen executable; set TETGEN_EXECUTABLE and run with --include-ignored"]
    fn tetrahedron_has_empty_handle_subspace() {
        let mesh = build_tetrahedron_surface();
        let tm = tetrahedralize(&mesh).expect("tetgen failed");
        let tb = build_tet_boundary(&tm);
        let k = compute_handle_subspace(&mesh, &tb, 0);
        assert_eq!(k.genus(), 0, "tetrahedron is genus 0 → no handles");
        assert_eq!(k.cycles.len(), 0);
    }

    /// A cube surface (genus 0) has trivial handle subspace.
    #[test]
    #[ignore = "requires a TetGen executable; set TETGEN_EXECUTABLE and run with --include-ignored"]
    fn cube_has_empty_handle_subspace() {
        let mesh = build_cube_surface();
        let tm = tetrahedralize(&mesh).expect("tetgen failed");
        let tb = build_tet_boundary(&tm);
        let k = compute_handle_subspace(&mesh, &tb, 0);
        assert_eq!(k.genus(), 0, "cube is genus 0 → no handles");
        assert_eq!(k.cycles.len(), 0);
    }

    /// A torus (genus 1) has a handle subspace of dimension 1.
    #[test]
    #[ignore = "requires a TetGen executable; set TETGEN_EXECUTABLE and run with --include-ignored"]
    fn torus_has_one_dimensional_handle_subspace() {
        let mesh = build_torus_surface(20, 10);
        let tm = tetrahedralize(&mesh).expect("tetgen failed");
        let tb = build_tet_boundary(&tm);
        let k = compute_handle_subspace(&mesh, &tb, 1);
        assert_eq!(k.genus(), 1, "torus is genus 1 → exactly one handle");
        assert_eq!(k.cycles.len(), 1);
        assert!(!k.cycles[0].is_empty(), "handle cycle should have edges");
    }

    /// The returned cycles' edges should all reference valid surface-edge
    /// indices.
    #[test]
    #[ignore = "requires a TetGen executable; set TETGEN_EXECUTABLE and run with --include-ignored"]
    fn handle_cycles_reference_valid_edges() {
        let mesh = build_torus_surface(16, 8);
        let tm = tetrahedralize(&mesh).expect("tetgen failed");
        let tb = build_tet_boundary(&tm);
        let k = compute_handle_subspace(&mesh, &tb, 1);
        let n_edges = k.surface_edges.len() as u32;
        for cycle in &k.cycles {
            for &edge_idx in cycle {
                assert!(
                    edge_idx < n_edges,
                    "cycle references out-of-range edge {edge_idx}"
                );
            }
        }
    }
}
