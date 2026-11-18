use super::*;
use crate::surface::{Regions, Vertex};
use crate::{LabeledSkeleton, label::Port};

fn placed(ports: [Port; 2]) -> (Mesh<INPUT>, RealizedSkeleton) {
    let (mesh, skeleton) = crate::surface::tests::lenses(2);
    let labels = skeleton.edges.keys().zip(ports.map(Port::label)).collect();
    let labeled = LabeledSkeleton { skeleton, labels };
    let placed = place_cubes::run(&mesh, &labeled).unwrap();
    (mesh, placed)
}

fn dissolved(mesh: &Mesh<INPUT>, source: &RealizedSkeleton) -> RealizedSkeleton {
    place_cubes::run(mesh, &dissolve_straight(mesh, &source.labeled).unwrap()).unwrap()
}

/// Four annuli around a torus, labeled by the labeling stage.
fn torus() -> (Mesh<INPUT>, LabeledSkeleton) {
    let positions: Vec<_> = (0..4)
        .map(|i| {
            let angle = std::f64::consts::TAU * (2. * i as f64 + 0.5) / 8.;
            [3. * angle.cos(), 3. * angle.sin(), 0.]
        })
        .collect();
    let (mesh, skeleton) = crate::surface::tests::bands(&positions);
    let labeled = crate::label::run(&mesh, &skeleton).unwrap();
    (mesh, labeled)
}

#[test]
fn dissolves_straight_patch_and_its_occurrences_without_mutating_source() {
    let (mesh, source) = placed([Port::PositiveX, Port::NegativeX]);
    let order = source.labeled.skeleton.surface.edge_order.clone();
    let result = dissolved(&mesh, &source);
    result.validate(&mesh).unwrap();
    assert_eq!(result.labeled.skeleton.nodes.len(), 2);
    assert_eq!(result.labeled.skeleton.edges.len(), 1);
    assert_eq!(result.labeled.skeleton.surface.crossings.len(), 2);
    assert_eq!(result.cuboids.cuboids.len(), 3);
    assert_eq!(source.labeled.skeleton.surface.edge_order, order);
    assert_eq!(source.labeled.skeleton.nodes.len(), 3);
    let again = dissolved(&mesh, &result);
    assert_eq!(again.node_positions, result.node_positions);
}

#[test]
fn stored_orientation_does_not_change_simplification() {
    let (mesh, mut source) = placed([Port::PositiveX, Port::NegativeX]);
    let edge = source.labeled.skeleton.edges.keys().next().unwrap();
    let mut boundary = source.labeled.skeleton.edges[edge].boundary.clone();
    crate::surface::edit::reverse(&mesh, &mut source.labeled.skeleton, &mut boundary);
    source.labeled.skeleton.edges[edge].boundary = boundary;
    source.labeled.skeleton.edges[edge].nodes.reverse();
    source.labeled.labels[edge] = Port::NegativeX.label();
    source = place_cubes::run(&mesh, &source.labeled).unwrap();
    assert_eq!(dissolved(&mesh, &source).labeled.skeleton.edges.len(), 1);
}

#[test]
fn preserves_bends() {
    let (mesh, source) = placed([Port::PositiveX, Port::PositiveY]);
    let result = dissolved(&mesh, &source);
    assert_eq!(result.node_positions, source.node_positions);
    assert_eq!(
        result.labeled.skeleton.surface.edge_order,
        source.labeled.skeleton.surface.edge_order
    );
}

#[test]
fn dissolves_an_entire_straight_chain() {
    let (mesh, mut skeleton) = crate::surface::tests::lenses(1);
    let edge = skeleton.edges.keys().next().unwrap();
    for _ in 0..5 {
        let old = skeleton.edges[edge].clone();
        let boundary = crate::surface::edit::parallel(&mesh, &mut skeleton, &old.boundary, 1);
        let node = skeleton.nodes.insert(crate::SkeletonNode {
            position: Vector3D::zeros(),
        });
        skeleton.edges[edge].nodes[1] = node;
        skeleton.edges.insert(crate::SkeletonEdge {
            nodes: [node, old.nodes[1]],
            boundary,
        });
    }
    let labels = skeleton
        .edges
        .keys()
        .map(|e| (e, Port::PositiveX.label()))
        .collect();
    let source = place_cubes::run(&mesh, &LabeledSkeleton { skeleton, labels }).unwrap();
    let result = dissolved(&mesh, &source);
    result.validate(&mesh).unwrap();
    assert_eq!(result.labeled.skeleton.edges.len(), 1);
    assert_eq!(result.labeled.skeleton.nodes.len(), 2);
    assert_eq!(result.labeled.skeleton.surface.crossings.len(), 2);
}

#[test]
fn minimum_cut_separates_at_the_cheapest_arc() {
    let mut network = cut::Network::new(4);
    network.add(0, 1, 5., 0.);
    network.add(1, 2, 1., 1.);
    network.add(2, 3, 5., 0.);
    assert_eq!(network.source_side(0, 3), vec![true, true, false, false]);
}

#[test]
fn restricted_fidelity_is_at_most_the_naive_one() {
    let (mesh, labeled) = torus();
    let tables = Tables::build(&mesh, &labeled.skeleton).unwrap();
    let fidelity = tables.evaluate(&labeled.skeleton, &labeled.labels);
    assert!(tables.area > 0.);
    assert!(fidelity.alignment <= tables.naive.alignment + 1e-12);
    assert!(fidelity.distortion >= tables.naive.distortion - 1e-12);
    assert!((-1.0..=1.0).contains(&fidelity.alignment));
}

#[test]
fn chain_relabeling_is_valid_and_never_worse() {
    let (mesh, labeled) = torus();
    let tables = Tables::build(&mesh, &labeled.skeleton).unwrap();
    let before = tables.evaluate(&labeled.skeleton, &labeled.labels);
    let chains = moves::chains(&labeled.skeleton);
    assert!(!chains.is_empty());
    for chain in chains {
        for measure in [Measure::Alignment, Measure::Distortion] {
            let (labels, _) = moves::relabel(&tables, &labeled, &chain, measure).unwrap();
            let candidate = LabeledSkeleton {
                skeleton: labeled.skeleton.clone(),
                labels,
            };
            candidate.validate(&mesh).unwrap();
            let after = tables.evaluate(&candidate.skeleton, &candidate.labels);
            assert!(after.gain(measure) >= before.gain(measure) - 1e-12);
        }
    }
}

#[test]
fn tracing_the_current_ownership_reproduces_a_valid_loop() {
    let (mesh, labeled) = torus();
    let skeleton = &labeled.skeleton;
    let edge = skeleton.edges.keys().next().unwrap();
    let [a, b] = skeleton.edges[edge].nodes;
    let regions = Regions::build(&mesh, skeleton).unwrap();
    let mut owner = HashMap::new();
    for (cell, &node) in regions.cells.iter().zip(&regions.owners) {
        for v in &cell.vertices {
            if let Vertex::Input(v) = *v {
                owner.insert(v, node == a);
            }
        }
    }
    let mut merged = skeleton.clone();
    let old = merged.edges.remove(edge).unwrap();
    redraw::erase(&mesh, &mut merged, &old.boundary.crossings);
    let mut pinned = HashMap::new();
    for data in merged.edges.values_mut() {
        for &c in &data.boundary.crossings {
            pinned.insert(c, data.nodes.contains(&a));
        }
        for n in &mut data.nodes {
            if *n == b {
                *n = a;
            }
        }
    }
    merged.nodes.remove(b);
    let union = Regions::build(&mesh, &merged).unwrap();
    let side = |v: Vertex| match v {
        Vertex::Input(v) => owner[&v],
        Vertex::Crossing(c) => pinned[&c],
    };
    let path = redraw::trace(&mesh, &union, a, &side).unwrap();
    assert_eq!(path.len(), old.boundary.crossings.len());
}

#[test]
fn redraw_keeps_a_valid_partition() {
    let (mesh, labeled) = torus();
    let params = RedrawParams {
        smoothness: 0.,
        measure: Measure::Alignment,
        rounds: 2,
    };
    for edge in labeled.skeleton.edges.keys() {
        if let Some(result) = redraw::redraw(&mesh, &labeled, edge, &params) {
            result.validate(&mesh).unwrap();
            assert_eq!(
                result.skeleton.edges[edge].nodes,
                labeled.skeleton.edges[edge].nodes
            );
        }
    }
}

#[test]
fn search_returns_a_valid_labeling_that_is_not_worse() {
    let (mesh, labeled) = torus();
    let params = Params {
        iterations: 30,
        ..Params::default()
    };
    let (result, stats) = search::search(&mesh, &labeled, &params, None).unwrap();
    result.validate(&mesh).unwrap();
    let score = |f: Fidelity, nodes: usize| f.alignment - params.node_cost * nodes as f64;
    assert!(score(stats.result, stats.nodes[1]) >= score(stats.initial, stats.nodes[0]) - 1e-12);
}

#[test]
fn added_leaves_are_valid_disks_at_a_free_port() {
    let (mesh, labeled) = torus();
    let params = RedrawParams {
        smoothness: 0.,
        measure: Measure::Alignment,
        rounds: 1,
    };
    for node in labeled.skeleton.nodes.keys() {
        for port in Port::ALL {
            if let Some(result) = redraw::add_leaf(&mesh, &labeled, node, port, &params) {
                result.validate(&mesh).unwrap();
                assert_eq!(
                    result.skeleton.nodes.len(),
                    labeled.skeleton.nodes.len() + 1
                );
            }
        }
    }
}

/// Per-patch areas of a skeleton.
fn patch_areas(mesh: &Mesh<INPUT>, skeleton: &crate::Skeleton) -> Vec<f64> {
    let regions = Regions::build(mesh, skeleton).unwrap();
    let positions = fidelity::crossing_positions(mesh, skeleton);
    let mut areas: HashMap<_, f64> = HashMap::new();
    for cell in fidelity::cells(mesh, &regions, &positions) {
        *areas.entry(cell.owner).or_default() += cell.area;
    }
    areas.into_values().collect()
}

/// The torus with one boundary copied twice beside itself: two vertex-free slivers.
fn stacked() -> (Mesh<INPUT>, LabeledSkeleton) {
    let (mesh, mut labeled) = torus();
    let edge = labeled.skeleton.edges.keys().next().unwrap();
    for _ in 0..2 {
        let (split, added) =
            crate::label::subdivide::collar(&mesh, &labeled.skeleton, edge, 0).unwrap();
        labeled.labels.insert(added, labeled.labels[edge]);
        labeled.skeleton = split;
    }
    labeled.validate(&mesh).unwrap();
    (mesh, labeled)
}

#[test]
fn every_relaxation_move_is_valid_local_and_undoable() {
    for (mesh, labeled) in [torus(), stacked()] {
        let faces = mesh.face_ids();
        let (stars, free) = relax::stars(&mesh, &labeled, &labeled.skeleton).unwrap();
        let mut work = relax::Work::new(&mesh, &labeled, labeled.skeleton.clone()).unwrap();
        let measure = Measure::Alignment;
        let total = |work: &relax::Work<'_>| work.local(&faces, &stars, &free, measure).unwrap();
        let start = total(&work);
        let mut applied = 0;
        for mv in work.moves() {
            let Some(plan) = work.plan(mv) else {
                continue;
            };
            let before = work.local(&plan.faces, &stars, &free, measure).unwrap();
            let undo = work.apply(&plan);
            let after = work.local(&plan.faces, &stars, &free, measure).unwrap();
            work.export().validate(&mesh).unwrap();
            // Only the planned triangles change.
            let whole = total(&work);
            for (node, [area, gain]) in whole.sums() {
                let [s, a, b] = [&start, &after, &before].map(|l| l.sum(node));
                assert!((area - (s[0] + a[0] - b[0])).abs() < 1e-9);
                assert!((gain - (s[1] + a[1] - b[1])).abs() < 1e-9);
            }
            for (lp, length) in whole.lengths() {
                let expected = start.length(lp) + after.length(lp) - before.length(lp);
                assert!((length - expected).abs() < 1e-9);
            }
            work.undo(undo);
            let restored = work.export();
            restored.validate(&mesh).unwrap();
            assert_eq!(
                restored.surface.crossings.len(),
                labeled.skeleton.surface.crossings.len()
            );
            applied += 1;
        }
        assert!(applied > 0);
    }
}

#[test]
fn relaxing_keeps_the_patch_graph_and_labels() {
    for (mesh, labeled) in [torus(), stacked()] {
        let Some(result) = relax::relax(&mesh, &labeled, &RelaxParams::default()) else {
            continue;
        };
        result.validate(&mesh).unwrap();
        assert_eq!(result.skeleton.nodes.len(), labeled.skeleton.nodes.len());
        for (id, edge) in &labeled.skeleton.edges {
            assert_eq!(result.skeleton.edges[id].nodes, edge.nodes);
            assert!(moves::same_label(result.labels[id], labeled.labels[id]));
        }
    }
}

#[test]
fn relaxing_spreads_stacked_slivers() {
    let (mesh, labeled) = stacked();
    let smallest = |s: &crate::Skeleton| {
        patch_areas(&mesh, s)
            .into_iter()
            .fold(f64::INFINITY, f64::min)
    };
    let params = RelaxParams {
        smoothness: 0.,
        thinness: 1.,
        ..RelaxParams::default()
    };
    let result = relax::relax(&mesh, &labeled, &params).unwrap();
    result.validate(&mesh).unwrap();
    assert!(smallest(&result.skeleton) > 2. * smallest(&labeled.skeleton));
}

#[test]
fn reduction_keeps_a_valid_ring_and_drops_copies() {
    let (mesh, labeled) = stacked();
    let flow = crate::label::flow::Flow::new(&mesh);
    let params = ReduceParams::default();
    let relabelled = reduce::label(&mesh, &labeled, &flow, &params).unwrap();
    relabelled.validate(&mesh).unwrap();
    assert_eq!(relabelled.skeleton.edges.len(), 6);
    let result = reduce::run(&mesh, &labeled, &flow, &params).unwrap();
    result.validate(&mesh).unwrap();
    // A closed ring needs both signs of two axes.
    assert_eq!(result.skeleton.edges.len(), 4);
}

#[test]
fn flow_costs_prefer_the_boundary_planes_axis() {
    let (mesh, labeled) = torus();
    let flow = crate::label::flow::Flow::new(&mesh);
    for (id, _) in &labeled.skeleton.edges {
        let costs = flow.costs(&mesh, &labeled.skeleton, id);
        // Boundaries of the torus bands lie in vertical planes: Z is never their axis.
        let best = (0..6).min_by(|&a, &b| costs[a].total_cmp(&costs[b])).unwrap();
        assert_ne!(best / 2, 2, "{costs:?}");
        assert!(costs.iter().all(|&c| c >= 1. - 1e-9));
    }
}
