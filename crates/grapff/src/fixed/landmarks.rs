//! Landmark ("ALT") lower bounds on shortest-path distances (Goldberg & Harrelson, Computing the Shortest Path:
//! A* Search Meets Graph Theory, 2005).
//!
//! For a landmark `L`, the triangle inequality gives, for all nodes `a` and `b`:
//! `d(a, b) >= d(L, b) - d(L, a)` and `d(a, b) >= d(a, L) - d(b, L)`.
//! The maximum over a few well-spread landmarks is an admissible and consistent A* heuristic, which (unlike a
//! Euclidean bound) also captures detours, e.g., when the start and goal of a search are close in space but far
//! apart in the graph (such as when searching for a closed loop).

use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use petgraph::{Directed, Direction, Graph};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

#[derive(Debug, Clone, Default)]
pub struct Landmarks {
    count: usize,
    // `from[node * count + l]` = d(landmark l, node), `to[node * count + l]` = d(node, landmark l).
    from: Vec<f64>,
    to: Vec<f64>,
}

impl Landmarks {
    /// Select `count` landmarks by farthest-point sampling and compute their distance tables (two Dijkstra searches
    /// per landmark). Weights must be non-negative.
    pub(super) fn new<V, E: Copy + Into<f64>>(graph: &Graph<V, E, Directed>, count: usize) -> Self {
        let n = graph.node_count();
        let count = count.min(n);
        let mut from = vec![f64::INFINITY; n * count];
        let mut to = vec![f64::INFINITY; n * count];
        // Combined distance of every node to the landmarks chosen so far (for farthest-point sampling).
        let mut nearest = vec![f64::INFINITY; n];
        let mut next = 0;
        for l in 0..count {
            let forward = dijkstra(graph, NodeIndex::new(next), Direction::Outgoing);
            let backward = dijkstra(graph, NodeIndex::new(next), Direction::Incoming);
            for v in 0..n {
                from[v * count + l] = forward[v];
                to[v * count + l] = backward[v];
                let both = forward[v] + backward[v];
                if both.is_finite() {
                    nearest[v] = nearest[v].min(both);
                }
            }
            nearest[next] = 0.;
            // Next landmark: the (reachable) node farthest from all landmarks so far.
            next = (0..n)
                .filter(|&v| nearest[v].is_finite())
                .max_by(|&a, &b| nearest[a].total_cmp(&nearest[b]))
                .unwrap_or(0);
        }
        Self { count, from, to }
    }

    /// Lower bound on the distance from node `a` to node `b` (dense petgraph node indices). Never negative.
    #[must_use]
    pub fn lower_bound(&self, a: usize, b: usize) -> f64 {
        let (fa, fb) = (
            &self.from[a * self.count..(a + 1) * self.count],
            &self.from[b * self.count..(b + 1) * self.count],
        );
        let (ta, tb) = (
            &self.to[a * self.count..(a + 1) * self.count],
            &self.to[b * self.count..(b + 1) * self.count],
        );
        let mut bound: f64 = 0.;
        for l in 0..self.count {
            // Terms with unreachable landmarks carry no (safe) information.
            if fa[l].is_finite() && fb[l].is_finite() {
                bound = bound.max(fb[l] - fa[l]);
            }
            if ta[l].is_finite() && tb[l].is_finite() {
                bound = bound.max(ta[l] - tb[l]);
            }
        }
        bound
    }
}

fn dijkstra<V, E: Copy + Into<f64>>(
    graph: &Graph<V, E, Directed>,
    source: NodeIndex,
    direction: Direction,
) -> Vec<f64> {
    let mut distance = vec![f64::INFINITY; graph.node_count()];
    let mut heap = BinaryHeap::new();
    distance[source.index()] = 0.;
    heap.push((Reverse(ordered(0.)), source.index()));
    while let Some((Reverse(d), u)) = heap.pop() {
        let d = f64::from_bits(d);
        if d > distance[u] {
            continue;
        }
        for edge in graph.edges_directed(NodeIndex::new(u), direction) {
            let v = match direction {
                Direction::Outgoing => edge.target(),
                Direction::Incoming => edge.source(),
            }
            .index();
            let candidate = d + (*edge.weight()).into();
            if candidate < distance[v] {
                distance[v] = candidate;
                heap.push((Reverse(ordered(candidate)), v));
            }
        }
    }
    distance
}

// For non-negative floats, the bit pattern orders like the value.
fn ordered(x: f64) -> u64 {
    debug_assert!(x >= 0.);
    x.to_bits()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn landmark_bounds_are_admissible() {
        // A directed ring with chords: 0 -> 1 -> ... -> n-1 -> 0 (cheap), and expensive reverse edges.
        let n = 30;
        let mut graph = Graph::<(), f64, Directed>::new();
        let nodes = (0..n).map(|_| graph.add_node(())).collect::<Vec<_>>();
        for i in 0..n {
            graph.add_edge(nodes[i], nodes[(i + 1) % n], 1. + (i % 3) as f64);
            graph.add_edge(nodes[(i + 1) % n], nodes[i], 10.);
            if i % 7 == 0 {
                graph.add_edge(nodes[i], nodes[(i + 11) % n], 4.);
            }
        }
        let landmarks = Landmarks::new(&graph, 4);
        let (mut total_bound, mut total_exact) = (0., 0.);
        for a in 0..n {
            let exact = dijkstra(&graph, NodeIndex::new(a), Direction::Outgoing);
            for b in 0..n {
                let bound = landmarks.lower_bound(a, b);
                assert!(
                    bound <= exact[b] + 1e-9,
                    "d({a},{b}) = {} < bound {bound}",
                    exact[b]
                );
                total_bound += bound;
                total_exact += exact[b];
            }
        }
        // The bound is informative.
        assert!(
            total_bound > 0.3 * total_exact,
            "{total_bound} vs {total_exact}"
        );
    }
}
