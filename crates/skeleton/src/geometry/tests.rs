use super::*;
use crate::surface::{
    edit,
    tests::{bands, lenses},
};

fn check(mesh: &Mesh<INPUT>, skeleton: &Skeleton) -> SurfaceGeometry {
    let geometry = SurfaceGeometry::build(mesh, skeleton).unwrap();
    let mut areas = HashMap::<_, f64>::new();
    let mut total_moment = Vector3D::zeros();
    for triangle in &geometry.triangles {
        let [a, b, c] = triangle.points;
        let source: Vec<_> = mesh
            .vertices(triangle.face)
            .map(|v| mesh.position(v))
            .collect();
        let normal = (source[1] - source[0]).cross(&(source[2] - source[0]));
        let cross = (b - a).cross(&(c - a));
        assert!(
            cross.dot(&normal) > 0.,
            "Triangle winding/area: {triangle:?}"
        );
        let area = cross.norm() * 0.5;
        *areas.entry(triangle.face).or_default() += area;
        total_moment += (a + b + c) / 3. * area;
        for p in [a, b, c] {
            assert!(p.iter().all(|x| x.is_finite()));
            assert!((p - source[0]).dot(&normal).abs() < 1e-11);
            for i in 0..3 {
                let side = source[(i + 1) % 3] - source[i];
                assert!(side.cross(&(p - source[i])).dot(&normal) >= -1e-10);
            }
        }
    }
    let mut expected_moment = Vector3D::zeros();
    for face in mesh.face_ids() {
        let p: Vec<_> = mesh.vertices(face).map(|v| mesh.position(v)).collect();
        let area = (p[1] - p[0]).cross(&(p[2] - p[0])).norm() * 0.5;
        assert!(
            (areas[&face] - area).abs() < area * 1e-10,
            "Face coverage: {} vs {area}",
            areas[&face]
        );
        expected_moment += (p[0] + p[1] + p[2]) / 3. * area;
    }
    assert!((total_moment - expected_moment).norm() < 1e-9);
    assert_eq!(geometry.centroids().unwrap().len(), skeleton.nodes.len());
    assert_eq!(geometry.boundaries.len(), skeleton.edges.len());
    for boundary in &geometry.boundaries {
        assert_eq!(boundary.points.first(), boundary.points.last());
        assert!(boundary.points.len() > 3);
        // Every curve segment is also an edge of the patch tessellation on each side.
        for pair in boundary.points.windows(2) {
            if (pair[1] - pair[0]).norm() < 1e-12 {
                continue;
            }
            for node in skeleton.edges[boundary.edge].nodes {
                assert!(
                    geometry
                        .triangles
                        .iter()
                        .filter(|t| t.node == node)
                        .any(|t| {
                            (0..3).any(|i| {
                                let a = t.points[i];
                                let b = t.points[(i + 1) % 3];
                                ((a - pair[0]).norm() < 1e-10 && (b - pair[1]).norm() < 1e-10)
                                    || ((b - pair[0]).norm() < 1e-10
                                        && (a - pair[1]).norm() < 1e-10)
                            })
                        }),
                    "Boundary does not match region tessellation"
                );
            }
        }
    }
    geometry
}

#[test]
fn whole_surface_and_zero_vertex_regions_have_positive_area() {
    for count in [0, 1, 7, 25] {
        let (mesh, skeleton) = lenses(count);
        check(&mesh, &skeleton);
    }
}

#[test]
fn parallel_collars_and_junction_bands_have_consistent_geometry() {
    let (mesh, mut skeleton) = lenses(8);
    let owner = skeleton.nodes.keys().next().unwrap();
    let pair: Vec<_> = skeleton.edges.keys().take(2).collect();
    let regions = Regions::build(&mesh, &skeleton).unwrap();
    let pair = [pair[0], pair[1]];
    let route = edit::connector(&regions, &skeleton, owner, pair, None).unwrap();
    skeleton = edit::band(&mesh, &skeleton, &regions, owner, pair, &route)
        .unwrap()
        .0;
    check(&mesh, &skeleton);
    // A full labeling also exercises repeated splits and all accepted surface edits.
    let labeled = crate::label::run(&mesh, &skeleton).unwrap();
    check(&mesh, &labeled.skeleton);
}

#[test]
fn repaired_cycles_and_refinement_preserve_input_and_boundaries() {
    for positions in [
        vec![[0., 0., 0.], [1., 0., 0.]],
        vec![[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
    ] {
        let (mesh, skeleton) = bands(&positions);
        let before: Vec<_> = mesh.vert_ids().iter().map(|&v| mesh.position(v)).collect();
        let mut labeled = crate::label::run(&mesh, &skeleton).unwrap();
        let geometry = check(&mesh, &labeled.skeleton);
        let expected = geometry.centroids().unwrap();
        let boundaries: Vec<_> = labeled
            .skeleton
            .edges
            .values()
            .map(|e| e.boundary.crossings.clone())
            .collect();
        crate::skeletonize::refinement::run(&mesh, &mut labeled.skeleton).unwrap();
        for (id, node) in &labeled.skeleton.nodes {
            assert_eq!(node.position, expected[id]);
        }
        assert_eq!(
            boundaries,
            labeled
                .skeleton
                .edges
                .values()
                .map(|e| e.boundary.crossings.clone())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            before,
            mesh.vert_ids()
                .iter()
                .map(|&v| mesh.position(v))
                .collect::<Vec<_>>()
        );
        labeled.validate(&mesh).unwrap();
    }
}

#[test]
fn failed_refinement_does_not_change_embedding() {
    let (mesh, mut skeleton) = lenses(2);
    skeleton
        .edges
        .values_mut()
        .next()
        .unwrap()
        .boundary
        .crossings
        .clear();
    let before: Vec<_> = skeleton.nodes.values().map(|n| n.position).collect();
    assert!(crate::skeletonize::refinement::run(&mesh, &mut skeleton).is_err());
    assert_eq!(
        before,
        skeleton
            .nodes
            .values()
            .map(|n| n.position)
            .collect::<Vec<_>>()
    );
}

#[test]
fn ordinary_arcs_stay_straight_and_mixed_winding_bands_cover_the_surface() {
    let (mesh, skeleton) = bands(&[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]);
    let geometry = check(&mesh, &skeleton);
    for boundary in &geometry.boundaries {
        assert_eq!(
            boundary.points.len(),
            skeleton.edges[boundary.edge].boundary.crossings.len() + 1
        );
    }
    let (mesh, mut skeleton) = lenses(6);
    let carriers: Vec<_> = mesh
        .edge_ids()
        .into_iter()
        .filter(|&e| e == canonical(&mesh, e))
        .collect();
    let edges: Vec<_> = skeleton.edges.keys().collect();
    skeleton.surface.edge_order.clear();
    for (i, &edge) in edges.iter().enumerate() {
        let mut boundary = skeleton.edges[edge].boundary.clone();
        let ids = &boundary.crossings;
        skeleton.surface.crossings[ids[0]].exit = carriers[i];
        skeleton.surface.crossings[ids[1]].exit = mesh.twin(carriers[i]);
        skeleton.surface.edge_order.insert(carriers[i], ids.clone());
        if i % 2 == 0 {
            edit::reverse(&mesh, &mut skeleton, &mut boundary);
            skeleton.edges[edge].boundary = boundary;
            skeleton.edges[edge].nodes.swap(0, 1);
        }
    }
    let regions = Regions::build(&mesh, &skeleton).unwrap();
    let owner = skeleton.nodes.keys().next().unwrap();
    let mut crossed_triangle = false;
    for a in 0..edges.len() {
        for b in a + 1..edges.len() {
            let pair = [edges[a], edges[b]];
            let route = edit::connector(&regions, &skeleton, owner, pair, None).unwrap();
            crossed_triangle |= !route.steps.is_empty();
            let split = edit::band(&mesh, &skeleton, &regions, owner, pair, &route)
                .unwrap()
                .0;
            check(&mesh, &split);
        }
    }
    assert!(crossed_triangle);
}
