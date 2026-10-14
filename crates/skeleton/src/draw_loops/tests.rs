use super::*;

#[test]
fn isolated_cube_template_has_three_loops_and_eight_regions() {
    let (mesh, skeleton) = crate::surface::tests::lenses(0);
    let labeled = crate::label::run(&mesh, &skeleton).unwrap();
    let realized = crate::place_cubes::run(&mesh, &labeled).unwrap();
    let template = template::Template::build(&realized).unwrap();
    assert_eq!(template.loops.len(), 3);
    let topology = PartialTopology::new(&template.mesh, &template.loops).unwrap();
    assert_eq!(topology.regions, 8);
    assert!(topology.equivalent(&topology));
}

#[test]
fn draws_returning_boundary_arcs_without_refining_the_input() {
    let (mesh, skeleton) = crate::surface::tests::lenses(1);
    let vertices = mesh.nr_verts();
    let faces = mesh.nr_faces();
    let labeled = crate::label::run(&mesh, &skeleton).unwrap();
    let realized = crate::place_cubes::run(&mesh, &labeled).unwrap();
    let result = attempt(&mesh, &realized).unwrap();
    assert!(result.failure.is_none(), "{:?}", result.failure);
    dualcube_dual::Dual::from(std::sync::Arc::new(mesh.clone()), &result.loops).unwrap();
    assert!(result.loops.values().all(|l| !l.surface.is_empty()));
    assert_eq!((mesh.nr_verts(), mesh.nr_faces()), (vertices, faces));
}

#[test]
fn draws_single_node_on_a_cube_surface() {
    let (input, skeleton) = crate::surface::tests::lenses(0);
    let labeled = crate::label::run(&input, &skeleton).unwrap();
    let realized = crate::place_cubes::run(&input, &labeled).unwrap();
    let template = template::Template::build(&realized).unwrap();
    let result = attempt(&template.mesh, &realized).unwrap();
    assert!(result.failure.is_none(), "{:?}", result.failure);
    assert_eq!(result.loops.len(), 3);
    dualcube_dual::Dual::from(std::sync::Arc::new(template.mesh), &result.loops).unwrap();
}

#[test]
fn draws_extracted_tetrahedron() {
    let mesh = crate::surface::tests::tetrahedron();
    let skeleton = crate::skeletonize::run(&mesh).unwrap();
    let labeled = crate::label::run(&mesh, &skeleton).unwrap();
    let realized = crate::place_cubes::run(&mesh, &labeled).unwrap();
    let result = attempt(&mesh, &realized).unwrap();
    assert!(result.failure.is_none(), "{:?}", result.failure);
    dualcube_dual::Dual::from(std::sync::Arc::new(mesh), &result.loops).unwrap();
    assert!(result.loops.len() >= realized.labeled.skeleton.edges.len());
}

fn star_realized(ports: &[crate::label::Port]) -> RealizedSkeleton {
    graph_realized(
        ports.len() + 1,
        &ports
            .iter()
            .enumerate()
            .map(|(i, &p)| (0, i + 1, p))
            .collect::<Vec<_>>(),
    )
}

fn graph_realized(
    count: usize,
    connections: &[(usize, usize, crate::label::Port)],
) -> RealizedSkeleton {
    let mut skeleton = crate::Skeleton::default();
    let nodes: Vec<_> = (0..count)
        .map(|_| {
            skeleton.nodes.insert(crate::SkeletonNode {
                position: Vector3D::zeros(),
            })
        })
        .collect();
    let mut labels = slotmap::SecondaryMap::new();
    for &(a, b, port) in connections {
        let edge = skeleton.edges.insert(crate::SkeletonEdge {
            nodes: [nodes[a], nodes[b]],
            boundary: Default::default(),
        });
        labels.insert(edge, port.label());
    }
    let labeled = crate::LabeledSkeleton { skeleton, labels };
    // Placement only needs valid labels; template tests need no input boundary realization.
    let positions = crate::place_cubes::coordinates(&labeled).unwrap();
    let mut layout = crate::CuboidLayout::default();
    let node_ids: HashMap<_, _> = nodes
        .iter()
        .map(|&id| {
            (
                id,
                layout.cuboids.insert(crate::Cuboid {
                    center: positions[id],
                    size: Vector3D::repeat(1.),
                    owner: crate::CuboidOwner::Node(id),
                }),
            )
        })
        .collect();
    for (edge, data) in &labeled.skeleton.edges {
        let [a, b] = data.nodes;
        let id = layout.cuboids.insert(crate::Cuboid {
            center: (positions[a] + positions[b]) / 2.,
            size: Vector3D::repeat(1.),
            owner: crate::CuboidOwner::Edge(edge),
        });
        layout
            .connections
            .extend([[node_ids[&a], id], [id, node_ids[&b]]]);
    }
    RealizedSkeleton {
        labeled,
        node_positions: positions,
        cuboids: layout,
    }
}

fn star_template(ports: &[crate::label::Port]) -> template::Template {
    template::Template::build(&star_realized(ports)).unwrap()
}

#[test]
fn rejects_wrong_hole_separation_with_the_same_crossing_sequence() {
    let template = star_template(&[
        crate::label::Port::NegativeX,
        crate::label::Port::PositiveX,
        crate::label::Port::PositiveY,
    ]);
    let boundaries: HashSet<_> = template.boundaries.values().copied().collect();
    let xs: Vec<_> = template
        .loops
        .iter()
        .filter(|(id, l)| boundaries.contains(id) && l.direction == Direction::X)
        .map(|(id, _)| id)
        .collect();
    assert_eq!(xs.len(), 2);
    let added = template
        .loops
        .iter()
        .find(|(id, l)| !boundaries.contains(id) && l.direction == Direction::X)
        .unwrap()
        .0;
    let mut wrong = template.loops.clone();
    wrong[xs[0]] = template.loops[xs[1]].clone();
    wrong[xs[1]] = template.loops[xs[0]].clone();
    for id in &xs {
        wrong[*id].edges.reverse();
        wrong[*id].offsets.reverse();
    }
    let before =
        PartialTopology::new(&template.mesh, &subset(&template.loops, &boundaries)).unwrap();
    let moved = PartialTopology::new(&template.mesh, &subset(&wrong, &boundaries)).unwrap();
    assert!(before.equivalent(&moved));
    let mut selected = boundaries;
    selected.insert(added);
    let correct =
        PartialTopology::new(&template.mesh, &subset(&template.loops, &selected)).unwrap();
    let wrong = PartialTopology::new(&template.mesh, &subset(&wrong, &selected)).unwrap();
    let partners = |t: &PartialTopology| {
        t.along[&added]
            .iter()
            .map(|&c| other(t.crossings[c], added))
            .collect::<Vec<_>>()
    };
    assert_eq!(partners(&correct), partners(&wrong));
    assert!(!correct.equivalent(&wrong));
}

#[test]
fn all_port_subsets_produce_valid_templates() {
    for mask in 0..64 {
        let ports: Vec<_> = crate::label::Port::ALL
            .into_iter()
            .enumerate()
            .filter(|(i, _)| mask & (1 << i) != 0)
            .map(|(_, p)| p)
            .collect();
        let template = star_template(&ports);
        let topology = PartialTopology::new(&template.mesh, &template.loops).unwrap();
        let mut rotated = template.loops.clone();
        for (_, l) in &mut rotated {
            l.edges.rotate_left(1);
            l.offsets.rotate_left(1);
        }
        let rotated = PartialTopology::new(&template.mesh, &rotated).unwrap();
        assert!(topology.equivalent(&rotated));
    }
}

#[test]
fn partial_topology_distinguishes_genus_before_any_crossings() {
    let sphere = crate::surface::tests::tetrahedron();
    let (torus, _) = crate::surface::tests::bands(&[[0.; 3]; 3]);
    let a = PartialTopology::new(&sphere, &DualLoops::with_key()).unwrap();
    let b = PartialTopology::new(&torus, &DualLoops::with_key()).unwrap();
    assert_eq!(a.regions, b.regions);
    assert!(!a.equivalent(&b));
}

#[test]
fn repaired_cycle_produces_a_valid_template() {
    let (mesh, skeleton) =
        crate::surface::tests::bands(&[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]);
    let labeled = crate::label::run(&mesh, &skeleton).unwrap();
    let realized = crate::place_cubes::run(&mesh, &labeled).unwrap();
    let template = template::Template::build(&realized).unwrap();
    let source = PartialTopology::new(&mesh, &DualLoops::with_key()).unwrap();
    let target = PartialTopology::new(&template.mesh, &DualLoops::with_key()).unwrap();
    assert!(source.equivalent(&target));
}

#[test]
fn coincident_endpoints_do_not_close_a_template_handle() {
    use crate::label::Port::*;
    let realized = graph_realized(
        5,
        &[
            (0, 1, PositiveX),
            (1, 2, PositiveY),
            (2, 3, NegativeX),
            (3, 4, NegativeY),
        ],
    );
    let nodes: Vec<_> = realized.labeled.skeleton.nodes.keys().collect();
    assert_eq!(
        realized.node_positions[nodes[0]],
        realized.node_positions[nodes[4]]
    );
    realized.validate_geometry().unwrap();
    let template = template::Template::build(&realized).unwrap();
    let target = PartialTopology::new(&template.mesh, &DualLoops::with_key()).unwrap();
    let sphere = PartialTopology::new(
        &crate::surface::tests::tetrahedron(),
        &DualLoops::with_key(),
    )
    .unwrap();
    assert!(target.equivalent(&sphere));
}

#[test]
fn draws_a_junction_preserving_boundary_topology() {
    let mut realized = star_realized(&[
        crate::label::Port::NegativeX,
        crate::label::Port::PositiveX,
        crate::label::Port::PositiveY,
    ]);
    let template = template::Template::build(&realized).unwrap();
    for (&edge, &id) in &template.boundaries {
        let mut l = template.loops[id].clone();
        if realized.labeled.labels[edge].sign == Sign::Negative {
            l.edges.reverse();
        }
        let skeleton = &mut realized.labeled.skeleton;
        for (i, &exit) in l.edges.iter().enumerate() {
            if template.mesh.twin(exit) != l.edges[(i + 1) % l.edges.len()] {
                continue;
            }
            let c = skeleton
                .surface
                .crossings
                .insert(crate::BoundaryCrossing { exit });
            skeleton.edges[edge].boundary.crossings.push(c);
            skeleton
                .surface
                .edge_order
                .entry(crate::surface::canonical(&template.mesh, exit))
                .or_default()
                .push(c);
        }
    }
    realized.validate(&template.mesh).unwrap();
    let before = realized.labeled.skeleton.surface.edge_order.clone();
    let result = attempt(&template.mesh, &realized).unwrap();
    assert!(result.failure.is_none(), "{:?}", result.failure);
    assert_eq!(realized.labeled.skeleton.surface.edge_order, before);
    let actual = PartialTopology::new(&template.mesh, &result.loops).unwrap();
    let expected = PartialTopology::new(&template.mesh, &template.loops).unwrap();
    assert!(actual.equivalent(&expected));
}

#[test]
fn reconstructs_each_missing_loop_on_a_six_way_template() {
    let template = star_template(&crate::label::Port::ALL);
    let flows = route::Flows::new(&template.mesh, true);
    let expected = PartialTopology::new(&template.mesh, &template.loops).unwrap();
    for id in template.loops.keys() {
        let drawn: HashSet<_> = template.loops.keys().filter(|&other| other != id).collect();
        let before = subset(&template.loops, &drawn);
        let itinerary =
            PartialTopology::insertion_itinerary(&template.mesh, &before, &template.loops, id)
                .unwrap();
        let crossings: Vec<_> = itinerary.iter().map(|&(l, _)| l).collect();
        let regions: Vec<_> = itinerary.iter().map(|&(_, r)| r).collect();
        let router =
            dualcube_dual::surface_router::SurfaceRouter::new(&template.mesh, &before).unwrap();
        let result = route::find(
            &template.mesh,
            &flows,
            &template.loops,
            &drawn,
            id,
            &crossings,
            &regions,
            &router,
            &expected,
            &[],
            &mut 0,
        );
        assert!(result.is_ok(), "{id:?}: {result:?}");
        let old = result.unwrap();
        let alternative = route::find(
            &template.mesh,
            &flows,
            &template.loops,
            &drawn,
            id,
            &crossings,
            &regions,
            &router,
            &expected,
            &[old],
            &mut 0,
        );
        assert!(
            alternative.is_ok(),
            "No alternative for {id:?}: {alternative:?}"
        );
    }
}

#[test]
fn route_duplicates_ignore_cyclic_start_and_orientation() {
    let (mesh, _) = crate::surface::tests::lenses(0);
    let face = mesh.face_ids()[0];
    let points = [[0.2, 0.2], [0.5, 0.2], [0.2, 0.5]];
    let original = Loop::from_surface(
        Direction::X,
        (0..3)
            .map(|i| dualcube_dual::SurfaceArc {
                face,
                from: points[i],
                to: points[(i + 1) % 3],
            })
            .collect(),
    );
    let mut shifted = original.clone();
    shifted.surface.rotate_left(1);
    assert!(route::same_geometry(&original, &shifted));
    shifted.reverse();
    assert!(route::same_geometry(&original, &shifted));
    shifted.surface[0].from[0] += 0.01;
    assert!(!route::same_geometry(&original, &shifted));
}

#[test]
#[ignore]
fn routing_comparison() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../meshes/dataset/a.bouba_zero/airplane1.stl");
    let mesh = Mesh::<INPUT>::from_stl(&path).unwrap().0;
    let skeleton = crate::skeletonize::run(&mesh).unwrap();
    let mut progress = crate::pipeline::Progress::from_skeleton(skeleton);
    progress.optimize(&mesh).unwrap();
    let realized = progress.optimized.unwrap();
    for repeat in 0..2 {
        let start = std::time::Instant::now();
        let old = crate::reference_drawing::attempt(&mesh, &realized).unwrap();
        let old_time = start.elapsed();
        assert!(old.failure.is_none(), "{:?}", old.failure);
        let start = std::time::Instant::now();
        let new = attempt(&mesh, &realized).unwrap();
        let new_time = start.elapsed();
        assert!(new.failure.is_none(), "{:?}", new.failure);
        assert!(PartialTopology::new(&mesh, &old.loops).unwrap()
            .equivalent(&PartialTopology::new(&mesh, &new.loops).unwrap()));
        eprintln!("COMPARISON repeat={repeat} before={:.3} after={:.3} searches={}/{} loops={}/{}",
            old_time.as_secs_f64(), new_time.as_secs_f64(),old.searches,new.searches,old.loops.len(),new.loops.len());
    }
}
