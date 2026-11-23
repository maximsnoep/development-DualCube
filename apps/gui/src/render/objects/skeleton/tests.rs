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
    let gray = graph(
        &labeled.skeleton,
        None,
        &HashMap::new(),
        1.,
        Vector3D::zeros(),
    );
    let colored = graph(
        &labeled.skeleton,
        Some(&labeled),
        &HashMap::new(),
        1.,
        Vector3D::zeros(),
    );
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
    let coincident = graph(
        &labeled.skeleton,
        Some(&labeled),
        &HashMap::new(),
        1.,
        Vector3D::zeros(),
    );
    assert!(coincident.count_vertices() > 0);
}

#[test]
fn cube_view_distinguishes_stretched_edges_and_keeps_finite_geometry() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/mehsh/assets/tetrahedron.obj");
    let solution = io::import_solution(&path).unwrap();
    let raw = dualcube::skeleton::skeletonize::run(&solution.mesh_ref).unwrap();
    let labeled = dualcube::skeleton::label::attempt(&solution.mesh_ref, &raw)
        .unwrap()
        .labeled();
    let mut realized = dualcube::skeleton::place_cubes::run(&solution.mesh_ref, &labeled).unwrap();
    for (_, position) in &mut realized.node_positions {
        *position *= 3.;
    }
    realized.validate(&solution.mesh_ref).unwrap();
    let mut object = RenderObject::default();
    cubes::add(&mut object, &realized, &HashMap::new());
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
    for layer in [SkeletonLayer::CubesPrimal, SkeletonLayer::CubesDual] {
        let RenderAsset::Mesh(other) = &object.features[layer.label()] else {
            panic!("Missing {} mesh", layer.label());
        };
        assert_eq!(other.count_vertices(), mesh.count_vertices());
        assert!(!layer.is_surface());
    }
    // The node cubes alone: six faces of two triangles each.
    let RenderAsset::Mesh(nodes) = &object.features[SkeletonLayer::NodeCubes.label()] else {
        panic!("Missing node cube mesh");
    };
    assert_eq!(nodes.count_vertices(), 36 * realized.node_positions.len());
    let RenderAsset::Mesh(placed) = &object.features[SkeletonLayer::PlacedSkeleton.label()] else {
        panic!("Missing placed skeleton mesh");
    };
    assert!(placed.count_vertices() > 0);
    assert!(
        object
            .features
            .contains_key(SkeletonLayer::NodeCubeEdges.label())
    );
    assert!(
        object
            .features
            .contains_key(SkeletonLayer::CubeEdges.label())
    );
}

#[test]
fn optimized_patches_keep_original_colors() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/mehsh/assets/tetrahedron.obj");
    let solution = io::import_solution(&path).unwrap();
    let mesh = &solution.mesh_ref;
    let extraction = dualcube::skeleton::skeletonize::extract(mesh, Default::default()).unwrap();
    let mut progress =
        dualcube::skeleton::pipeline::Progress::from_skeleton(extraction.skeleton.clone());
    progress
        .run_through(mesh, dualcube::skeleton::pipeline::Stage::Optimization)
        .unwrap();
    let state = SkeletonResource {
        state: Some(crate::resources::SkeletonState::Extracted(extraction)),
        progress,
        ..default()
    };
    let object = build(mesh, &state, 0.006);
    let colors = |label: &str| {
        let RenderAsset::Mesh(mesh) = &object.features[label] else {
            panic!("Missing {label}");
        };
        let Some(VertexAttributeValues::Float32x4(colors)) =
            mesh.attribute(BevyMesh::ATTRIBUTE_COLOR)
        else {
            panic!("Missing colors of {label}");
        };
        colors
            .iter()
            .map(|c| c.map(f32::to_bits))
            .collect::<HashSet<_>>()
    };
    let original = colors(SkeletonLayer::Patches.label());
    let optimized = colors(SkeletonLayer::OptimizedPatches.label());
    assert!(!optimized.is_empty() && optimized.is_subset(&original));
    assert!(
        object
            .features
            .contains_key(SkeletonLayer::OptimizedBoundaries.label())
    );
    assert!(SkeletonLayer::OptimizedPatches.is_surface());
    // The optimized combined view.
    for layer in [
        SkeletonLayer::TranslucentOptimizedPatches,
        SkeletonLayer::OptimizedSkeleton,
    ] {
        assert!(object.features.contains_key(layer.label()), "{layer:?}");
    }
}
