use crate::{Float, Grapff, ZERO};
use core::hash::Hash;
use petgraph::algo::tarjan_scc;
use petgraph::{Directed, Graph, graph::NodeIndex};
use rustc_hash::FxHashMap as HashMap;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, OnceLock};

mod landmarks;
pub use landmarks::Landmarks;

// Number of landmarks for the A* lower bounds (see `FixedGraph::landmarks`).
const LANDMARKS: usize = 8;

// Graph struct, that builds an underlying Petgraph with helper functions for various graph algorithms, such as, shortest path, connected components, etc.
#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct FixedGraph<V: Eq + PartialEq + Hash, E> {
    petgraph: Graph<V, E, Directed>,
    node_to_index: HashMap<V, NodeIndex>,
    edge_to_weight: HashMap<(V, V), E>,
    nodes: Vec<V>,
    edges: Vec<(V, V, E)>,
    // Lazily computed landmark distance tables (see `landmarks`).
    #[serde(skip)]
    landmarks: OnceLock<Arc<Landmarks>>,
}

impl<V: Eq + PartialEq + Hash + Default + Copy, E: Copy + Into<f64>> FixedGraph<V, E> {
    /// Landmark lower bounds on the shortest-path distances in this graph (edge weights must be non-negative).
    /// Computed once (on first use) and shared by clones of the graph.
    pub fn landmarks(&self) -> Arc<Landmarks> {
        self.landmarks
            .get_or_init(|| Arc::new(Landmarks::new(&self.petgraph, LANDMARKS)))
            .clone()
    }
}

impl<V: Eq + PartialEq + Hash + Default + Copy, E: Copy> FixedGraph<V, E> {
    #[must_use]
    pub fn from(nodes: Vec<V>, edges: Vec<(V, V, E)>) -> Self {
        let mut petgraph = Graph::with_capacity(nodes.len(), edges.len());
        let node_to_index: HashMap<V, NodeIndex> = nodes
            .iter()
            .map(|&node| (node, petgraph.add_node(node)))
            .collect();
        let edges_indexed = edges
            .iter()
            .map(|(from, to, w)| (node_to_index[from], node_to_index[to], w));
        petgraph.extend_with_edges(edges_indexed);
        let edge_to_weight = edges
            .iter()
            .map(|&(from, to, weight)| ((from, to), weight))
            .collect();
        Self {
            petgraph,
            node_to_index,
            edge_to_weight,
            nodes,
            edges,
            landmarks: OnceLock::new(),
        }
    }

    /// Dense index of a node (as used by `Landmarks`).
    #[must_use]
    pub fn node_index(&self, node: &V) -> Option<usize> {
        self.node_to_index.get(node).map(|index| index.index())
    }

    #[must_use]
    pub fn nodes(&self) -> Vec<V> {
        self.nodes.clone()
    }

    #[must_use]
    pub fn edges(&self) -> Vec<(V, V, E)> {
        self.edges.clone()
    }

    /// Borrowed view of all edges (avoids the clone in [`Self::edges`]).
    #[must_use]
    pub fn edges_ref(&self) -> &[(V, V, E)] {
        &self.edges
    }

    /// Outgoing neighbors of `a` together with the edge weight, without allocating and without
    /// an extra hash lookup per neighbor (the weight is read from the petgraph edge).
    pub fn outgoing(&self, a: V) -> impl Iterator<Item = (V, E)> + '_ {
        use petgraph::visit::EdgeRef;
        self.petgraph
            .edges(self.node_to_index[&a])
            .map(|e| (self.petgraph[e.target()], *e.weight()))
    }

    #[must_use]
    pub fn node_to_index(&self, node: &V) -> Option<NodeIndex> {
        self.node_to_index.get(node).copied()
    }

    #[must_use]
    pub fn index_to_node(&self, index: NodeIndex) -> Option<&V> {
        self.petgraph.node_weight(index)
    }

    pub fn directed_edge_exists(&self, a: V, b: V) -> bool {
        self.edge_to_weight.contains_key(&(a, b))
    }

    pub fn node_exists(&self, a: V) -> bool {
        self.node_to_index.contains_key(&a)
    }

    pub fn neighbors(&self, a: V) -> Vec<V> {
        self.petgraph
            .neighbors(self.node_to_index[&a])
            .map(|index| self.index_to_node(index).unwrap().to_owned())
            .collect()
    }

    pub fn neighbors_undirected(&self, a: V) -> Vec<V> {
        self.petgraph
            .neighbors_undirected(self.node_to_index[&a])
            .map(|index| self.index_to_node(index).unwrap().to_owned())
            .collect()
    }

    #[must_use]
    pub fn get_directed_weight(&self, a: V, b: V) -> Option<E> {
        self.edge_to_weight.get(&(a, b)).copied()
    }

    pub fn topological_sort(&self) -> Option<Vec<V>> {
        petgraph::algo::toposort(&self.petgraph, None)
            .ok()
            .map(|sorted_indices| {
                sorted_indices
                    .into_iter()
                    .map(|index| self.index_to_node(index).unwrap().to_owned())
                    .collect()
            })
    }
}

impl<T: Eq + Hash + Clone + Copy + Default, E: Copy> Grapff<T, E> for FixedGraph<T, E> {
    fn neighbors(&self, v: T) -> Vec<T> {
        self.petgraph
            .neighbors(self.node_to_index[&v])
            .map(|index| self.index_to_node(index).unwrap().to_owned())
            .collect()
    }

    fn shortest_path(&self, a: T, b: T, w: impl Fn(E) -> Float) -> Option<(Vec<T>, Float)> {
        self.shortest_path_heuristic(a, b, w, |_| ZERO)
    }

    fn shortest_path_heuristic(
        &self,
        a: T,
        b: T,
        w: impl Fn(E) -> Float,
        h: impl Fn((T, T)) -> Float,
    ) -> Option<(Vec<T>, Float)> {
        if !self.node_exists(a) || !self.node_exists(b) {
            return None;
        }
        let shortest_path = petgraph::algo::astar(
            &self.petgraph,
            self.node_to_index(&a).unwrap(),
            |finish| finish == self.node_to_index(&b).unwrap(),
            |e| w(e.weight().to_owned()),
            |v| h((self.index_to_node(v).unwrap().to_owned(), b)),
        );

        let (cost, path) = shortest_path?;
        let path_nodes = path
            .into_iter()
            .map(|index| self.index_to_node(index).unwrap().to_owned())
            .collect();
        Some((path_nodes, cost))
    }

    // The nodes reachable from `v` along the edges (breadth-first).
    fn connected_component(&self, v: T) -> std::collections::HashSet<T> {
        let mut seen = std::collections::HashSet::from([v]);
        let mut queue = std::collections::VecDeque::from([v]);
        while let Some(node) = queue.pop_front() {
            for next in self.neighbors(node) {
                if seen.insert(next) {
                    queue.push_back(next);
                }
            }
        }
        seen
    }

    fn connected_components(&self, _: &[T]) -> Vec<std::collections::HashSet<T>> {
        tarjan_scc(&self.petgraph)
            .into_iter()
            .map(|cc| {
                cc.into_iter()
                    .map(|index| self.index_to_node(index).unwrap().to_owned())
                    .collect()
            })
            .collect()
    }
}
