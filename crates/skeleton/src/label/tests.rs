use super::*;
use crate::{PatchBoundary, SkeletonEdge, SkeletonNode};
use dualcube_types::prelude::HasPosition;

fn costs(skeleton: &Skeleton) -> SecondaryMap<SkeletonEdgeID, [f64; 6]> {
    skeleton
        .edges
        .iter()
        .map(|(e, edge)| {
            (
                e,
                angle_costs(
                    skeleton.nodes[edge.nodes[1]].position - skeleton.nodes[edge.nodes[0]].position,
                ),
            )
        })
        .collect()
}

fn graph(positions: &[[f64; 3]], edges: &[[usize; 2]]) -> Skeleton {
    let mut graph = Skeleton::default();
    let nodes: Vec<_> = positions
        .iter()
        .map(|p| {
            graph.nodes.insert(SkeletonNode {
                position: Vector3D::from(*p),
            })
        })
        .collect();
    for &[a, b] in edges {
        graph.edges.insert(SkeletonEdge {
            nodes: [nodes[a], nodes[b]],
            boundary: PatchBoundary {
                crossings: Vec::new(),
            },
        });
    }
    graph
}

fn enumerate(n: usize, mut check: impl FnMut(&[Port])) {
    for mut number in 0..6usize.pow(n as u32) {
        let mut ports = Vec::new();
        for _ in 0..n {
            ports.push(Port::ALL[number % 6]);
            number /= 6;
        }
        check(&ports);
    }
}

#[test]
fn path_dp_matches_exhaustive_search_for_free_and_fixed_endpoints() {
    let costs: Vec<_> = (0..5)
        .map(|i| angle_costs(Vector3D::new(1. + i as f64, 2. - i as f64, 0.7)))
        .collect();
    for constraints in [
        path::Constraints::default(),
        path::Constraints {
            start_ports: Port::PositiveY.bit(),
            ..Default::default()
        },
        path::Constraints {
            start_ports: Port::PositiveY.bit(),
            end_ports: Port::NegativeZ.bit(),
            ..Default::default()
        }
        .with_orders([
            path::Order::Equal,
            path::Order::Increasing,
            path::Order::Decreasing,
        ]),
    ] {
        let mut best = f64::INFINITY;
        enumerate(costs.len(), |ports| {
            if constraints.start_ports & ports[0].bit() == 0
                || constraints.end_ports & ports.last().unwrap().opposite().bit() == 0
                || ports.windows(2).any(|p| p[0].opposite() == p[1])
            {
                return;
            }
            let mask = ports.iter().fold(0, |mask, p| mask | p.bit());
            if !constraints.masks[mask as usize] {
                return;
            }
            let cost = ports
                .iter()
                .zip(&costs)
                .map(|(&p, costs)| costs[p as usize])
                .sum::<f64>();
            best = best.min(cost);
        });
        let solved = path::optimize(&costs, &constraints);
        if best.is_finite() {
            let solution = solved.unwrap();
            assert!((solution.cost - best).abs() < 1e-12);
            assert_eq!(
                solution.mask,
                solution.ports.iter().fold(0, |mask, p| mask | p.bit())
            );
        } else {
            assert!(solved.is_none());
        }
    }
}

#[test]
fn path_ports_allow_straight_travel_but_not_reversals() {
    let x = Port::PositiveX.bit();
    let constraints = path::Constraints {
        start_ports: x,
        end_ports: x,
        ..Default::default()
    };
    assert!(path::optimize(&[[0.; 6]; 2], &constraints).is_none());
    let constraints = path::Constraints {
        end_ports: Port::NegativeX.bit(),
        ..constraints
    };
    assert_eq!(
        path::optimize(&[[0.; 6]; 2], &constraints).unwrap().ports,
        vec![Port::PositiveX; 2]
    );
    let fixed = constraints.with_orders([path::Order::Equal; 3]);
    assert!(path::optimize(&[[0.; 6]; 2], &fixed).is_none());
}

#[test]
fn masks_describe_strictly_positive_lengths() {
    for (order, expected) in [
        (path::Order::Equal, [true, false, false, true]),
        (path::Order::Increasing, [false, true, false, true]),
        (path::Order::Decreasing, [false, false, true, true]),
    ] {
        let limits =
            path::Constraints::default().with_orders([order, path::Order::Any, path::Order::Any]);
        assert_eq!(&limits.masks[..4], &expected);
    }
}

#[test]
fn tree_dp_matches_brute_force_and_handles_reversed_edges() {
    let skeleton = graph(
        &[
            [0., 0., 0.],
            [1., 0., 0.],
            [2., 0.1, 0.],
            [2., 0., 0.1],
            [3., 0., 0.],
        ],
        &[[1, 0], [1, 2], [3, 1], [2, 4]],
    );
    let costs = costs(&skeleton);
    let (tree, labels) = tree::solve(&skeleton, &costs).unwrap();
    assert_eq!(tree.len(), 4);
    let chosen: f64 = labels
        .iter()
        .map(|(e, l)| costs[e][Port::from_label(*l) as usize])
        .sum();
    let edges: Vec<_> = skeleton.edges.keys().collect();
    let mut best = f64::INFINITY;
    enumerate(edges.len(), |ports| {
        let labels = edges
            .iter()
            .zip(ports)
            .map(|(&e, &p)| (e, p.label()))
            .collect();
        if constraints::used_ports(&skeleton, &labels).is_ok() {
            best = best.min(
                edges
                    .iter()
                    .zip(ports)
                    .map(|(&e, &p)| costs[e][p as usize])
                    .sum(),
            );
        }
    });
    assert!((chosen - best).abs() < 1e-12);
    LabeledSkeleton { skeleton, labels }
        .validate_labels()
        .unwrap();
}

#[test]
fn rejects_high_degree_and_accepts_single_node() {
    let single = graph(&[[0.; 3]], &[]);
    assert!(tree::solve(&single, &costs(&single)).unwrap().1.is_empty());
    let star = graph(
        &[[0.; 3]; 8],
        &[[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7]],
    );
    assert!(
        tree::solve(&star, &costs(&star))
            .unwrap_err()
            .to_string()
            .contains("six ports")
    );
}

#[test]
fn validation_allows_global_overlap_but_rejects_coordinate_cycles() {
    let skeleton = graph(&[[0.; 3]; 5], &[[0, 1], [1, 2], [2, 3], [3, 4]]);
    let ports = [
        Port::PositiveX,
        Port::PositiveY,
        Port::NegativeX,
        Port::NegativeY,
    ];
    let labels = skeleton.edges.keys().zip(ports.map(Port::label)).collect();
    LabeledSkeleton { skeleton, labels }
        .validate_labels()
        .unwrap();
    let skeleton = graph(&[[0.; 3]; 3], &[[0, 1], [1, 2], [2, 0]]);
    let labels = skeleton
        .edges
        .keys()
        .map(|e| (e, Port::PositiveX.label()))
        .collect();
    assert!(constraints::used_ports(&skeleton, &labels).is_ok());
    assert!(!constraints::feasible(&skeleton, &labels, None));
}

#[test]
fn coincident_embedding_has_finite_unbiased_costs() {
    assert_eq!(
        angle_costs(Vector3D::zeros()),
        [std::f64::consts::FRAC_PI_2; 6]
    );
    let costs = angle_costs(Vector3D::new(1., 0., 0.));
    assert_eq!(costs[0], 0.);
    assert_eq!(costs[1], std::f64::consts::PI);
}

#[test]
fn six_way_junction_uses_every_port_once() {
    let skeleton = graph(
        &[[0.; 3]; 7],
        &[[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6]],
    );
    let (_, labels) = tree::solve(&skeleton, &costs(&skeleton)).unwrap();
    let used = constraints::used_ports(&skeleton, &labels).unwrap();
    assert_eq!(used[skeleton.nodes.keys().next().unwrap()], 63);
    LabeledSkeleton { skeleton, labels }
        .validate_labels()
        .unwrap();
}

#[test]
fn coordinate_checks_include_equalities_and_previously_repaired_cycles() {
    let skeleton = graph(&[[0.; 3]; 4], &[[0, 1], [1, 2], [2, 3], [0, 3]]);
    let nodes: Vec<_> = skeleton.nodes.keys().collect();
    let edges: Vec<_> = skeleton.edges.keys().collect();
    let mut labels: SecondaryMap<_, _> = edges[..3]
        .iter()
        .copied()
        .zip([Port::PositiveX, Port::PositiveY, Port::NegativeX].map(Port::label))
        .collect();
    let diagonal = Port::PositiveX.bit() | Port::PositiveY.bit();
    assert!(constraints::feasible(
        &skeleton,
        &labels,
        Some(([nodes[0], nodes[3]], diagonal))
    ));
    labels.insert(edges[3], Port::PositiveY.label());
    assert!(constraints::feasible(&skeleton, &labels, None));
    // The first closure fixed X equality; a second route cannot require strict +X.
    assert!(!constraints::feasible(
        &skeleton,
        &labels,
        Some(([nodes[0], nodes[3]], diagonal))
    ));
    let detour = diagonal | Port::NegativeX.bit();
    assert!(constraints::feasible(
        &skeleton,
        &labels,
        Some(([nodes[0], nodes[3]], detour))
    ));
}

#[test]
fn path_repair_can_require_all_four_subdivisions() {
    let costs = [[0.; 6]; 5];
    let constraints = path::Constraints {
        start_ports: Port::PositiveX.bit(),
        end_ports: Port::NegativeX.bit(),
        ..Default::default()
    }
    .with_orders([
        path::Order::Decreasing,
        path::Order::Equal,
        path::Order::Equal,
    ]);
    for length in 1..5 {
        assert!(path::optimize(&costs[..length], &constraints).is_none());
    }
    let solution = path::optimize(&costs, &constraints).unwrap();
    assert_eq!(solution.ports.len(), 5);
}

#[test]
fn collars_have_no_vertex_budget_on_either_side() {
    use crate::surface::{Regions, Vertex, tests::lenses};
    for side in 0..2 {
        let (mesh, original) = lenses(1);
        let edge = original.edges.keys().next().unwrap();
        let mut skeleton = original.clone();
        for count in 1..=12 {
            let (next, added) = subdivide::collar(&mesh, &skeleton, edge, side)
                .unwrap()
                .unwrap();
            let middle = next.edges[added].nodes[0];
            let regions = Regions::build(&mesh, &next).unwrap();
            assert!(
                regions
                    .cells
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| regions.owners[*i] == middle)
                    .all(|(_, c)| c.vertices.iter().all(|v| matches!(v, Vertex::Crossing(_))))
            );
            assert_eq!(next.nodes.len(), original.nodes.len() + count);
            skeleton = next;
        }
        assert_eq!(mesh.nr_verts(), 4);
        original.validate(&mesh).unwrap();
    }
}

#[test]
fn overfull_junctions_split_even_when_all_patches_share_one_mesh_edge() {
    for degree in [6, 7, 11] {
        let (mesh, skeleton) = crate::surface::tests::lenses(degree);
        let result = attempt(&mesh, &skeleton).unwrap();
        assert!(result.failure.is_none(), "{:?}", result.failure);
        assert_eq!(result.junction_splits.len(), degree.saturating_sub(6));
        result.completed().unwrap().validate(&mesh).unwrap();
        for node in result.skeleton.nodes.keys() {
            assert!(
                result
                    .skeleton
                    .edges
                    .values()
                    .filter(|e| e.nodes.contains(&node))
                    .count()
                    <= 6
            );
        }
        assert_eq!(mesh.nr_verts(), 4);
        skeleton.validate(&mesh).unwrap();
    }
}

#[test]
fn cycle_repair_uses_real_annuli_and_preserves_mst_membership() {
    for positions in [
        vec![[0., 0., 0.], [2., 0., 0.]],
        vec![[0., 0., 0.], [2., 0., 0.], [2., 2., 0.]],
        vec![[0., 0., 0.], [2., 0., 0.], [2., 2., 0.], [0., 2., 0.]],
    ] {
        let (mesh, skeleton) = crate::surface::tests::bands(&positions);
        let before: Vec<_> = mesh.vert_ids().iter().map(|&v| mesh.position(v)).collect();
        let result = attempt(&mesh, &skeleton).unwrap();
        assert!(result.failure.is_none(), "{:?}", result.failure);
        assert_eq!(result.repairs.len(), 1);
        assert_eq!(result.tree_edges.len(), skeleton.nodes.len() - 1);
        let path = &result.repairs[0].path;
        assert_eq!(
            result.skeleton.nodes.len(),
            skeleton.nodes.len() + path.len() - 1
        );
        if positions.len() < 4 {
            assert!(path.len() > 1);
        }
        for edge in path {
            assert!(!result.tree_edges.contains(edge));
        }
        for &edge in &result.tree_edges {
            assert_eq!(
                result.skeleton.edges[edge].boundary.crossings,
                skeleton.edges[edge].boundary.crossings
            );
            assert_eq!(
                result.skeleton.edges[edge].nodes,
                skeleton.edges[edge].nodes
            );
            assert_eq!(
                Port::from_label(result.labels[edge]),
                Port::from_label(result.tree_labels[edge])
            );
        }
        result.completed().unwrap().validate(&mesh).unwrap();
        skeleton.validate(&mesh).unwrap();
        assert_eq!(
            before,
            mesh.vert_ids()
                .iter()
                .map(|&v| mesh.position(v))
                .collect::<Vec<_>>()
        );
    }
}
