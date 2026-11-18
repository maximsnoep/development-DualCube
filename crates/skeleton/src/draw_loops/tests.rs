use super::*;

mod surface;

#[test]
fn isolated_cube_template_has_three_loops_and_eight_regions() {
    let (mesh, skeleton) = crate::surface::tests::lenses(0);
    let labeled = crate::label::run(&mesh, &skeleton).unwrap();
    let realized = crate::place_cubes::run(&mesh, &labeled).unwrap();
    let template = surface::Surface::build(&realized).unwrap();
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
fn chords_refuse_curves_that_revisit_triangles() {
    // Lens triangles are coarse: the drawn loops revisit them, which chords cannot express.
    let (mesh, skeleton) = crate::surface::tests::lenses(1);
    let labeled = crate::label::run(&mesh, &skeleton).unwrap();
    let realized = crate::place_cubes::run(&mesh, &labeled).unwrap();
    let result = attempt(&mesh, &realized).unwrap();
    assert!(result.failure.is_none(), "{:?}", result.failure);
    assert!(chords(&mesh, &result.loops).is_none());
}

#[test]
fn draws_single_node_on_a_cube_surface() {
    let (input, skeleton) = crate::surface::tests::lenses(0);
    let labeled = crate::label::run(&input, &skeleton).unwrap();
    let realized = crate::place_cubes::run(&input, &labeled).unwrap();
    let template = surface::Surface::build(&realized).unwrap();
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

fn star_template(ports: &[crate::label::Port]) -> surface::Surface {
    surface::Surface::build(&star_realized(ports)).unwrap()
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
            .map(|&c| {
                let [a, b] = t.crossings[c];
                if a == added { b } else { a }
            })
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
    let template = surface::Surface::build(&realized).unwrap();
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
    let template = surface::Surface::build(&realized).unwrap();
    let target = PartialTopology::new(&template.mesh, &DualLoops::with_key()).unwrap();
    let sphere = PartialTopology::new(
        &crate::surface::tests::tetrahedron(),
        &DualLoops::with_key(),
    )
    .unwrap();
    assert!(target.equivalent(&sphere));
}

/// A skeleton whose boundaries are the template's boundary loops, on the template's own mesh.
fn on_template(mut realized: RealizedSkeleton) -> (surface::Surface, RealizedSkeleton) {
    let template = surface::Surface::build(&realized).unwrap();
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
    (template, realized)
}

/// Draw every loop from the boundaries alone and compare with the template independently.
fn assert_draws(realized: RealizedSkeleton) {
    let (template, realized) = on_template(realized);
    let before = realized.labeled.skeleton.surface.edge_order.clone();
    let result = attempt(&template.mesh, &realized).unwrap();
    assert!(result.failure.is_none(), "{:?}", result.failure);
    assert_eq!(realized.labeled.skeleton.surface.edge_order, before);
    // The drawn curves, arranged geometrically, are the combinatorial template.
    let actual = PartialTopology::new(&template.mesh, &result.loops).unwrap();
    let expected = template::Template::build(&realized.labeled).unwrap();
    assert!(actual.equivalent(&PartialTopology::from_map(expected.target.map())));
}

#[test]
fn draws_every_loop_when_solids_coincide() {
    use crate::label::Port::*;
    // A cycle doubling back on itself: node 4 coincides with node 0 and node 7 with node 1.
    // The X-section through 0–4 and the Z-section through 0, 5, 6, 7, 4 each pass both cubes
    // of the pair, so the same two loops cross twice at one position.
    let realized = graph_realized(
        8,
        &[
            (0, 1, PositiveY),
            (1, 2, PositiveZ),
            (2, 3, NegativeY),
            (3, 4, NegativeZ),
            (0, 5, PositiveX),
            (5, 6, PositiveY),
            (6, 7, NegativeX),
            (7, 4, NegativeY),
        ],
    );
    let nodes: Vec<_> = realized.labeled.skeleton.nodes.keys().collect();
    let at = |i: usize| realized.node_positions[nodes[i]];
    assert_eq!((at(0), at(1)), (at(4), at(7)));
    assert_draws(realized);
}

#[test]
fn draws_a_junction_preserving_boundary_topology() {
    assert_draws(star_realized(&[
        crate::label::Port::NegativeX,
        crate::label::Port::PositiveX,
        crate::label::Port::PositiveY,
    ]));
}

#[test]
fn draws_a_ring_whose_loops_have_one_region_on_both_sides() {
    use crate::label::Port::*;
    assert_draws(graph_realized(
        4,
        &[
            (0, 1, PositiveX),
            (1, 2, PositiveY),
            (2, 3, NegativeX),
            (3, 0, NegativeY),
        ],
    ));
}

#[test]
fn draws_an_arc_leaving_a_partner_with_its_region_on_both_sides() {
    use crate::label::Port::*;
    // A ring with a branch of two cubes. When a loop is inserted, the partner interval it starts
    // from has one region on both sides, which the arc splits: the arc's side of the interval
    // decides where the other boundaries of the region belong.
    assert_draws(graph_realized(
        6,
        &[
            (0, 1, PositiveX),
            (1, 2, PositiveY),
            (2, 3, NegativeX),
            (3, 0, NegativeY),
            (3, 4, PositiveZ),
            (4, 5, PositiveY),
        ],
    ));
}

#[test]
fn draws_a_six_way_junction() {
    assert_draws(star_realized(&crate::label::Port::ALL));
}

#[test]
fn every_plan_crosses_the_drawn_loops_in_template_order() {
    let realized = star_realized(&crate::label::Port::ALL);
    let template = template::Template::build(&realized.labeled).unwrap();
    let target = &template.target;
    let topology = PartialTopology::from_map(target.map());
    for id in template.loops.keys() {
        let drawn: HashSet<_> = template.loops.keys().filter(|&other| other != id).collect();
        let plan = Plan::new(target, &drawn, id);
        assert_eq!(plan.len(), topology.along[&id].len());
        assert_eq!(plan.len(), target.partners(id).len());
        assert!(Plan::new(target, &HashSet::new(), id).is_empty());
    }
}

/// Loops per axis and crossing count, regions, and crossings.
fn summary(topology: &PartialTopology, loops: &DualLoops) -> (Vec<(usize, usize)>, usize, usize) {
    let mut per_loop: Vec<_> = loops
        .iter()
        .map(|(id, l)| (l.direction as usize, topology.along[&id].len()))
        .collect();
    per_loop.sort();
    (per_loop, topology.regions, topology.crossings.len())
}

#[test]
fn the_template_of_a_cube_has_three_loops_and_eight_regions() {
    let (mesh, skeleton) = crate::surface::tests::lenses(0);
    let labeled = crate::label::run(&mesh, &skeleton).unwrap();
    let template = template::Template::build(&labeled).unwrap();
    assert_eq!(template.loops.len(), 3);
    assert!(template.boundaries.is_empty());
    let topology = PartialTopology::from_map(template.target.map());
    assert_eq!((topology.regions, topology.crossings.len()), (8, 6));
}

#[test]
fn templates_agree_with_the_plane_sections_of_the_cuboid_surface() {
    use crate::label::Port::*;
    let mut cases: Vec<_> = (0..64)
        .map(|mask| {
            let ports: Vec<_> = crate::label::Port::ALL
                .into_iter()
                .filter(|p| mask & p.bit() != 0)
                .collect();
            star_realized(&ports)
        })
        .collect();
    // A ring, and a path whose ends coincide.
    cases.push(graph_realized(
        4,
        &[
            (0, 1, PositiveX),
            (1, 2, PositiveY),
            (2, 3, NegativeX),
            (3, 0, NegativeY),
        ],
    ));
    cases.push(graph_realized(
        5,
        &[
            (0, 1, PositiveX),
            (1, 2, PositiveY),
            (2, 3, NegativeX),
            (3, 4, NegativeY),
        ],
    ));
    for realized in cases {
        let template = template::Template::build(&realized.labeled).unwrap();
        let reference = surface::Surface::build(&realized).unwrap();
        assert_eq!(
            template.boundaries.len(),
            realized.labeled.skeleton.edges.len()
        );
        assert_eq!(
            summary(
                &PartialTopology::from_map(template.target.map()),
                &template.loops
            ),
            summary(
                &PartialTopology::new(&reference.mesh, &reference.loops).unwrap(),
                &reference.loops
            )
        );
        // Boundaries are the same loops: same axis and crossings.
        for (edge, &id) in &template.boundaries {
            assert_eq!(
                template.loops[id].direction,
                reference.loops[reference.boundaries[edge]].direction
            );
        }
    }
}
