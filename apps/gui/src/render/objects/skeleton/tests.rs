use super::*;
use bevy::mesh::Mesh as BevyMesh;

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
        }
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
fn patches_preserve_area_winding_and_midpoint_boundary() {
    let points = [
        Vector3D::zeros(),
        Vector3D::new(2., 0., 0.),
        Vector3D::new(0., 2., 0.),
    ];
    for regions in [[0, 0, 1], [1, 0, 0], [0, 1, 0]] {
        let triangles = split_triangle(points, regions);
        let mut areas = [0.; 2];
        for (p, region) in &triangles {
            let area = (p[1] - p[0]).cross(&(p[2] - p[0])).z * 0.5;
            assert!(area > 0.);
            areas[*region] += area;
        }
        assert!((areas[0] - 1.5).abs() < 1e-12);
        assert!((areas[1] - 0.5).abs() < 1e-12);
        let singleton = regions.iter().position(|&r| r == 1).unwrap();
        let boundary = [
            (points[singleton] + points[(singleton + 1) % 3]) * 0.5,
            (points[singleton] + points[(singleton + 2) % 3]) * 0.5,
        ];
        for midpoint in boundary {
            assert!(
                triangles
                    .iter()
                    .any(|(p, r)| *r == 0 && p.contains(&midpoint))
            );
            assert!(
                triangles
                    .iter()
                    .any(|(p, r)| *r == 1 && p.contains(&midpoint))
            );
        }
    }
}

#[test]
fn labeled_graph_uses_labels_and_accepts_coincident_nodes() {
    use dualcube::skeleton::{EdgeLabel, PatchBoundary, SkeletonEdge, SkeletonNode};
    let mut skeleton = Skeleton::default();
    let a = skeleton.nodes.insert(SkeletonNode {
        position: Vector3D::zeros(),
        patch_vertices: vec![],
    });
    let b = skeleton.nodes.insert(SkeletonNode {
        position: Vector3D::x(),
        patch_vertices: vec![],
    });
    let edge = skeleton.edges.insert(SkeletonEdge {
        nodes: [a, b],
        boundary: PatchBoundary {
            edge_midpoints: vec![],
        },
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
fn node_colors_match_patches_and_overlay_keeps_split_geometry() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/mehsh/assets/tetrahedron.obj");
    let solution = io::import_solution(&path).unwrap();
    let extraction =
        dualcube::skeleton::skeletonize::extract(&solution.mesh_ref, Default::default()).unwrap();
    let state = SkeletonResource {
        state: Some(crate::resources::SkeletonState::Extracted(extraction)),
        labeled: None,
        ..default()
    };
    let object = build(&solution.mesh_ref, &state);
    let RenderAsset::Mesh(graph) = &object.features[SkeletonLayer::Skeleton.label()] else {
        panic!()
    };
    let RenderAsset::Mesh(patches) = &object.features[SkeletonLayer::Patches.label()] else {
        panic!()
    };
    let RenderAsset::TranslucentMesh(overlay) =
        &object.features[SkeletonLayer::TranslucentPatches.label()]
    else {
        panic!()
    };
    let colors = |mesh: &BevyMesh| match mesh.attribute(BevyMesh::ATTRIBUTE_COLOR).unwrap() {
        VertexAttributeValues::Float32x4(colors) => colors.clone(),
        _ => panic!(),
    };
    let graph_colors = colors(graph);
    let patch_colors = colors(patches);
    assert!(patch_colors.iter().all(|c| graph_colors.contains(c)));
    assert_eq!(patches.count_vertices(), overlay.count_vertices());
    for (opaque, translucent) in patch_colors.iter().zip(colors(overlay)) {
        assert_eq!(opaque[..3], translucent[..3]);
        assert_eq!(translucent[3], 0.3);
    }
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
