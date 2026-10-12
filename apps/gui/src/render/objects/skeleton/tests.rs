use super::*;
use bevy::mesh::{Mesh as BevyMesh, VertexAttributeValues};

#[test]
fn extraction_layers_survive_refresh_and_clear_with_new_input() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/mehsh/assets/tetrahedron.obj");
    let solution = io::import_solution(&path).unwrap();
    let extraction =
        dualcube::skeleton::skeletonize::extract(&solution.mesh_ref, Default::default()).unwrap();
    let mut store = RenderObjectStore::default();
    store.add_object(Objects::InputMesh, RenderObject::default());
    let mut app = App::new();
    app.insert_resource(InputResource::new(solution.mesh_ref))
        .insert_resource(SkeletonResource {
            state: Some(crate::resources::SkeletonState::Extracted(extraction)),
            labeled: None,
            ..default()
        })
        .insert_resource(store)
        .add_systems(Update, sync);
    app.update();
    let object = &app.world().resource::<RenderObjectStore>().objects[&Objects::InputMesh];
    assert!(object.features.contains_key("patches"));
    assert!(object.features.contains_key("translucent patches"));
    assert!(object.features.contains_key("patch boundaries"));
    assert!(object.features.contains_key("skeleton"));
    assert!(!object.features.contains_key("labeled skeleton"));
    for asset in object.features.values() {
        if let RenderAsset::Mesh(mesh) | RenderAsset::TranslucentMesh(mesh) = asset {
            let positions = mesh
                .attribute(BevyMesh::ATTRIBUTE_POSITION)
                .unwrap()
                .as_float3()
                .unwrap();
            assert!(positions.iter().flatten().all(|x| x.is_finite()));
            let normals = mesh
                .attribute(BevyMesh::ATTRIBUTE_NORMAL)
                .unwrap()
                .as_float3()
                .unwrap();
            assert!(normals.iter().flatten().all(|x| x.is_finite()));
        }
    }
    for (label, expected_alpha) in [("patches", 1.), ("translucent patches", 0.3)] {
        let (RenderAsset::Mesh(mesh) | RenderAsset::TranslucentMesh(mesh)) =
            &object.features[label]
        else {
            panic!("Missing patch mesh");
        };
        let Some(VertexAttributeValues::Float32x4(colors)) =
            mesh.attribute(BevyMesh::ATTRIBUTE_COLOR)
        else {
            panic!("Missing patch colors");
        };
        assert!(!colors.is_empty());
        assert!(colors.iter().all(|c| c[3] == expected_alpha));
    }
    app.world_mut()
        .resource_mut::<RenderObjectStore>()
        .add_object(Objects::InputMesh, RenderObject::default());
    app.update();
    assert!(
        app.world().resource::<RenderObjectStore>().objects[&Objects::InputMesh]
            .features
            .contains_key("skeleton")
    );
    *app.world_mut().resource_mut::<SkeletonResource>() = Default::default();
    *app.world_mut().resource_mut::<InputResource>() = Default::default();
    app.update();
    assert!(
        app.world().resource::<RenderObjectStore>().objects[&Objects::InputMesh]
            .features
            .is_empty()
    );
}

#[test]
fn labeled_graph_uses_labels_and_accepts_coincident_nodes() {
    use dualcube::skeleton::{EdgeLabel, PatchBoundary, SkeletonEdge, SkeletonNode};
    let mut skeleton = Skeleton::default();
    let a = skeleton.nodes.insert(SkeletonNode {
        position: Vector3D::zeros(),
    });
    let b = skeleton.nodes.insert(SkeletonNode {
        position: Vector3D::x(),
    });
    let edge = skeleton.edges.insert(SkeletonEdge {
        nodes: [a, b],
        boundary: PatchBoundary { crossings: vec![] },
    });
    let mut labeled = LabeledSkeleton {
        skeleton,
        labels: Default::default(),
    };
    labeled.labels.insert(
        edge,
        EdgeLabel {
            direction: Direction::X,
            sign: Sign::Positive,
        },
    );
    let gray = graph(&labeled.skeleton, None, 1., Vector3D::zeros());
    let colored = graph(&labeled.skeleton, Some(&labeled), 1., Vector3D::zeros());
    let Some(VertexAttributeValues::Float32x4(gray_colors)) =
        gray.attribute(BevyMesh::ATTRIBUTE_COLOR)
    else {
        panic!("Missing colors");
    };
    let Some(VertexAttributeValues::Float32x4(axis_colors)) =
        colored.attribute(BevyMesh::ATTRIBUTE_COLOR)
    else {
        panic!("Missing colors");
    };
    assert_ne!(gray_colors[0], axis_colors[0]);
    labeled.skeleton.nodes[b].position = Vector3D::zeros();
    let coincident = graph(&labeled.skeleton, Some(&labeled), 1., Vector3D::zeros());
    assert!(coincident.count_vertices() > 0);
}

#[test]
fn large_patch_sets_keep_saturated_colors_and_neighboring_indices_distinct() {
    for region in 0..10_000 {
        let color = region_color(region);
        assert!(
            color
                .iter()
                .all(|c| c.is_finite() && (0.0..=1.0).contains(c))
        );
        let hsv = Hsva::from(Srgba::rgb(color[0], color[1], color[2]));
        assert!(hsv.saturation >= 0.64);
        assert!((0.75..=0.99).contains(&hsv.value));
        let next = region_color(region + 1);
        let distance: f32 = color.iter().zip(next).map(|(a, b)| (a - b).powi(2)).sum();
        assert!(distance > 0.1);
    }
}

#[test]
fn cube_view_distinguishes_edge_cubes_and_keeps_finite_geometry() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/mehsh/assets/tetrahedron.obj");
    let solution = io::import_solution(&path).unwrap();
    let raw = dualcube::skeleton::skeletonize::run(&solution.mesh_ref).unwrap();
    let labeled = dualcube::skeleton::label::run(&solution.mesh_ref, &raw).unwrap();
    let realized = dualcube::skeleton::place_cubes::run(&solution.mesh_ref, &labeled).unwrap();
    let mut object = RenderObject::default();
    cubes::add(&mut object, &realized);
    let RenderAsset::Mesh(mesh) = &object.features[SkeletonLayer::Cubes.label()] else {
        panic!("Missing cube mesh");
    };
    assert!(mesh.count_vertices() > 0);
    let normals = mesh
        .attribute(BevyMesh::ATTRIBUTE_NORMAL)
        .unwrap()
        .as_float3()
        .unwrap();
    assert!(
        normals
            .iter()
            .any(|n| n.iter().filter(|&&x| x.abs() > 0.1).count() > 1)
    );
    assert!(
        normals
            .iter()
            .all(|n| n.iter().filter(|&&x| x.abs() > 0.1).count() <= 2)
    );
    let positions = mesh
        .attribute(BevyMesh::ATTRIBUTE_POSITION)
        .unwrap()
        .as_float3()
        .unwrap();
    for (points, normals) in positions.chunks_exact(3).zip(normals.chunks_exact(3)) {
        let [a, b, c] = [
            Vec3::from(points[0]),
            Vec3::from(points[1]),
            Vec3::from(points[2]),
        ];
        assert!((b - a).cross(c - a).dot(Vec3::from(normals[0])) > 0.);
    }
    for attribute in [BevyMesh::ATTRIBUTE_POSITION, BevyMesh::ATTRIBUTE_NORMAL] {
        assert!(
            mesh.attribute(attribute)
                .unwrap()
                .as_float3()
                .unwrap()
                .iter()
                .flatten()
                .all(|x| x.is_finite())
        );
    }
    assert!(!SkeletonLayer::Cubes.is_surface());
    assert!(
        object
            .features
            .contains_key(SkeletonLayer::CubeEdges.label())
    );
}
