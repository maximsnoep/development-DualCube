//! Flow graphs: per-axis weighted edge graphs derived from the flow fields.

use crate::flowfield::Field;
use dualcube_types::prelude::*;

/// Parameters for the flow graphs. Moves cost their length, scaled by `1 + alignment_weight * misalignment +
/// confidence_weight * (1 - flow magnitude)`.
#[derive(Debug, Clone)]
pub struct GraphParams {
    pub alignment_weight: f64,
    pub confidence_weight: f64,
    /// Moves that deviate more than this angle (in radians) from the flow are pruned, where the flow magnitude is at
    /// least `min_confidence`. This is a local rule; pruned moves are restored where needed to keep the graph
    /// strongly connected (see `reconnect`).
    pub max_deviation: f64,
    pub min_confidence: f64,
    /// Weight of the exact per-face target direction, relative to the vertex field (interpolated over the face).
    pub face_weight: f64,
}

impl Default for GraphParams {
    fn default() -> Self {
        Self {
            alignment_weight: 10.0,
            confidence_weight: 10.0,
            max_deviation: 100f64.to_radians(),
            min_confidence: 0.25,
            face_weight: 0.5,
        }
    }
}

pub fn build_flow_graphs<T: Tag>(
    mesh: &Mesh<T>,
    fields: &crate::flowfield::Fields<T>,
    params: GraphParams,
) -> [grapff::fixed::FixedGraph<ids::Key<EDGE, T>, f64>; 3] {
    let params = &params;
    let nodes = mesh.edge_ids();
    let neighbors = mesh.neighbor_function_edgegraph();

    let mut flow_graphs = [
        grapff::fixed::FixedGraph::default(),
        grapff::fixed::FixedGraph::default(),
        grapff::fixed::FixedGraph::default(),
    ];

    for (field, axis) in [
        (&fields.field_x, Direction::X),
        (&fields.field_y, Direction::Y),
        (&fields.field_z, Direction::Z),
    ] {
        let (mut kept, mut pruned) = (vec![], vec![]);
        for &edge in &nodes {
            for next in neighbors(edge) {
                let (weight, allowed) = get_edge_weight(mesh, field, axis, edge, next, params);
                if allowed {
                    kept.push((edge, next, weight));
                } else {
                    pruned.push((edge, next, weight));
                }
            }
        }
        reconnect(&nodes, &mut kept, pruned);
        flow_graphs[axis as usize] = grapff::fixed::FixedGraph::from(nodes.clone(), kept);
    }

    flow_graphs
}

// Pruning moves locally can cut off small pockets of the graph (strongly connected components that a loop cannot
// leave or enter). Restore the pruned moves that leave or enter a node outside the largest strongly connected
// component, until the graph is strongly connected again (or nothing can be restored, e.g., for a mesh with several
// components).
fn reconnect<V: Copy + Eq + std::hash::Hash>(
    nodes: &[V],
    kept: &mut Vec<(V, V, f64)>,
    mut pruned: Vec<(V, V, f64)>,
) {
    let index: HashMap<V, usize> = nodes.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    loop {
        let edges = kept
            .iter()
            .map(|&(a, b, _)| (index[&a], index[&b]))
            .collect::<Vec<_>>();
        let component = strongly_connected_components(nodes.len(), &edges);
        let mut sizes = HashMap::<usize, usize>::new();
        for &c in &component {
            *sizes.entry(c).or_default() += 1;
        }
        if sizes.len() <= 1 {
            return;
        }
        let largest = sizes
            .iter()
            .max_by_key(|&(_, &size)| size)
            .map(|(&c, _)| c)
            .unwrap_or(0);
        let outside = |v: &V| component[index[v]] != largest;
        let (restore, rest): (Vec<_>, Vec<_>) = pruned
            .into_iter()
            .partition(|(a, b, _)| outside(a) || outside(b));
        if restore.is_empty() {
            return;
        }
        kept.extend(restore);
        pruned = rest;
    }
}

// The strongly connected component of every node (Kosaraju's algorithm, iterative), given the edges as index pairs.
fn strongly_connected_components(n: usize, edges: &[(usize, usize)]) -> Vec<usize> {
    let adjacency = |reverse: bool| {
        let mut adjacency = vec![vec![]; n];
        for &(a, b) in edges {
            if reverse {
                adjacency[b].push(a);
            } else {
                adjacency[a].push(b);
            }
        }
        adjacency
    };
    let (forward, backward) = (adjacency(false), adjacency(true));

    // Nodes in order of finishing a depth-first search on the forward graph.
    let mut order = Vec::with_capacity(n);
    let mut visited = vec![false; n];
    for root in 0..n {
        if visited[root] {
            continue;
        }
        visited[root] = true;
        let mut stack = vec![(root, 0)];
        while let Some((v, i)) = stack.last_mut() {
            if let Some(&w) = forward[*v].get(*i) {
                *i += 1;
                if !visited[w] {
                    visited[w] = true;
                    stack.push((w, 0));
                }
            } else {
                order.push(*v);
                stack.pop();
            }
        }
    }

    // Components: depth-first searches on the backward graph, in reverse finishing order.
    let mut component = vec![usize::MAX; n];
    let mut count = 0;
    for &root in order.iter().rev() {
        if component[root] != usize::MAX {
            continue;
        }
        component[root] = count;
        let mut stack = vec![root];
        while let Some(v) = stack.pop() {
            for &w in &backward[v] {
                if component[w] == usize::MAX {
                    component[w] = count;
                    stack.push(w);
                }
            }
        }
        count += 1;
    }
    component
}

// Cost of moving from edge `e0` to edge `e1` (through their face): the length of the move, scaled by how
// badly it is aligned with the flow (cf. the alignment term of Campen et al., Dual Strip Weaving, 2014), and by
// how uncertain the flow is. Moving to the twin edge (crossing the edge) is free. Moves that go (too much)
// against a confident flow are pruned (not allowed): otherwise, the cheapest loops are tiny loops that turn around
// (which would be prevented by a bending energy term in the objective of Campen et al.).
fn get_edge_weight<T: Tag>(
    mesh: &Mesh<T>,
    field: &Field<T>,
    axis: Direction,
    e0: ids::Key<EDGE, T>,
    e1: ids::Key<EDGE, T>,
    params: &GraphParams,
) -> (f64, bool) {
    if mesh.twin(e0) == e1 {
        return (0.0, true);
    }
    assert!(mesh.face(e0) == mesh.face(e1));
    // The field is defined at the vertices. Inside large flat faces (e.g., on CAD models, where almost all vertices
    // lie on sharp features with averaged normals), it is blended with the exact target direction of the face:
    // around the axis, with magnitude (confidence) the sine of the angle between the normal and the axis.
    let normal = mesh.normal(mesh.face(e0)).normalize();
    let axis_vector = Vector3D::from(axis);
    let face_target = normal.cross(&(axis_vector - normal * axis_vector.dot(&normal)));
    let flow_vector = (1.0 - params.face_weight) * get_flow_from_edge_to_edge(mesh, field, e0, e1)
        + params.face_weight * face_target;
    let flow_magnitude = flow_vector.norm().clamp(0.0, 1.0);
    let confidence = 1.0 - flow_magnitude;

    let edge_vector = mesh.position(e1) - mesh.position(e0);
    let length = edge_vector.norm();
    let cos = if flow_magnitude > 1e-12 && length > 1e-12 {
        flow_vector.dot(&edge_vector) / (flow_vector.norm() * length)
    } else {
        0.0
    };
    // tan^2(angle / 2): zero when aligned, one when orthogonal, unbounded when opposite to the flow.
    let misalignment = ((1.0 - cos) / (1.0 + cos).max(1e-3)).min(1e3);

    let allowed = flow_magnitude < params.min_confidence || cos >= params.max_deviation.cos();
    let weight = length
        * (1.0 + params.alignment_weight * misalignment + params.confidence_weight * confidence);
    (weight, allowed)
}

fn get_flow_from_edge_to_edge<T: Tag>(
    mesh: &Mesh<T>,
    field: &Field<T>,
    edge: ids::Key<EDGE, T>,
    next: ids::Key<EDGE, T>,
) -> Vector3D {
    let flow_edge = get_flow_at_edge(mesh, field, edge);
    let flow_next = get_flow_at_edge(mesh, field, next);
    0.5 * (flow_edge + flow_next)
}

fn get_flow_at_edge<T: Tag>(mesh: &Mesh<T>, field: &Field<T>, edge: ids::Key<EDGE, T>) -> Vector3D {
    let (Some(f0), Some(f1)) = (
        field.vector_at(mesh.root(edge)),
        field.vector_at(mesh.toor(edge)),
    ) else {
        return Vector3D::zeros();
    };
    0.5 * (f0 + f1)
}
