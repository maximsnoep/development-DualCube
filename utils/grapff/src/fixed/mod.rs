use crate::{Float, Grapff, ZERO};
use core::hash::Hash;
use itertools::Itertools;
use petgraph::algo::tarjan_scc;
use petgraph::{Directed, Graph, graph::NodeIndex};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Graph struct, that builds an underlying Petgraph with helper functions for various graph algorithms, such as, shortest path, shortest cycle, connected components, etc.
// Also contains functionality to transform a graph into a modified graph. (e.g., filtering edges or vertices)
#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct FixedGraph<V: Eq + PartialEq + Hash, E> {
    petgraph: Graph<V, E, Directed>,
    node_to_index: HashMap<V, NodeIndex>,
    nodes: Vec<V>,
    edges: Vec<(V, V, E)>,
}

impl<V: Eq + PartialEq + Hash + Default + Copy, E: Copy> FixedGraph<V, E> {
    #[must_use]
    pub fn from(nodes: Vec<V>, edges: Vec<(V, V, E)>) -> Self {
        let mut petgraph = Graph::with_capacity(nodes.len(), edges.len());
        let node_to_index: HashMap<V, NodeIndex> = nodes.iter().map(|&node| (node, petgraph.add_node(node))).collect();
        let edges_indexed = edges.iter().map(|(from, to, w)| (node_to_index[from], node_to_index[to], w));
        petgraph.extend_with_edges(edges_indexed);
        Self {
            petgraph,
            node_to_index,
            nodes,
            edges,
        }
    }

    #[must_use]
    pub fn nodes(&self) -> Vec<V> {
        self.nodes.clone()
    }

    #[must_use]
    pub fn edges(&self) -> Vec<(V, V, E)> {
        self.edges.clone()
    }

    #[must_use]
    pub fn filter_edges(&self, predicate: impl Fn((&V, &V)) -> bool) -> Self {
        let nodes = self.nodes.clone();
        let edges = self.edges.iter().filter(|(from, to, _)| predicate((from, to))).copied().collect_vec();
        Self::from(nodes, edges)
    }

    #[must_use]
    pub fn filter_nodes(&self, predicate: impl Fn(&V) -> bool) -> Self {
        let nodes = self.nodes.iter().filter(|&&node| predicate(&node)).copied().collect_vec();
        let edges = self
            .edges
            .iter()
            .filter(|(from, to, _)| predicate(from) && predicate(to))
            .copied()
            .collect_vec();
        Self::from(nodes, edges)
    }

    pub fn extend(&mut self, nodes: &[V], edges: &[(V, V, E)]) {
        let extra_node_to_index: HashMap<V, NodeIndex> = nodes.iter().map(|&node| (node, self.petgraph.add_node(node))).collect();
        self.node_to_index.extend(extra_node_to_index);

        let extra_edges_indexed = edges.iter().map(|(from, to, w)| (self.node_to_index[from], self.node_to_index[to], w));
        self.petgraph.extend_with_edges(extra_edges_indexed);

        self.edges.extend_from_slice(edges);
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
        self.neighbors(a).iter().any(|n| n == &b)
    }

    pub fn node_exists(&self, a: V) -> bool {
        self.node_to_index.contains_key(&a)
    }

    pub fn edge_exists(&self, a: V, b: V) -> bool {
        self.directed_edge_exists(a, b) || self.directed_edge_exists(b, a)
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

    // pub fn shortest_cycle<W: Measure + Copy + FloatCore, F: Fn(E) -> W>(&self, a: NodeIndex, measure: &F) -> Option<Vec<NodeIndex>> {
    //     self.petgraph
    //         .neighbors(a)
    //         .map(|b| (a, b))
    //         .filter_map(|(a, b)| {
    //             let extra = measure(self.get_weight(a, b));
    //             let path = self.shortest_path(b, a, measure);
    //             path.map(|(cost, path)| (path, cost + extra))
    //         })
    //         .min_by_key(|(_, cost)| OrderedFloat(cost.to_owned()))
    //         .map(|(path, _)| path)
    // }

    pub fn shortest_cycle_edge<F: Fn(E) -> Float>(&self, (a, b): (V, V), measure: &F) -> Option<(Float, Vec<V>)> {
        let path = self.shortest_path(b, a, measure);
        path.map(|(cost, path)| (path, cost))
    }

    #[must_use]
    pub fn get_weight(&self, a: NodeIndex, b: NodeIndex) -> E {
        self.petgraph.edges_connecting(a, b).next().unwrap().weight().to_owned()
    }

    pub fn topological_sort(&self) -> Option<Vec<V>> {
        petgraph::algo::toposort(&self.petgraph, None)
            .ok()
            .map(|sorted_indices| sorted_indices.into_iter().map(|index| self.index_to_node(index).unwrap().to_owned()).collect())
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

    fn shortest_path_heuristic(&self, a: T, b: T, w: impl Fn(E) -> Float, h: impl Fn((T, T)) -> Float) -> Option<(Vec<T>, Float)> {
        if let Some((cost, path)) = petgraph::algo::astar(
            &self.petgraph,
            self.node_to_index(&a).unwrap(),
            |finish| finish == self.node_to_index(&b).unwrap(),
            |e| w(e.weight().to_owned()),
            |v| h((self.index_to_node(v).unwrap().to_owned(), b)),
        ) {
            let path_nodes = path.into_iter().map(|index| self.index_to_node(index).unwrap().to_owned()).collect();
            Some((path_nodes, cost))
        } else {
            None
        }
    }

    fn connected_component(&self, _v: T) -> std::collections::HashSet<T> {
        todo!()
    }

    fn connected_components(&self, _: &[T]) -> Vec<std::collections::HashSet<T>> {
        tarjan_scc(&self.petgraph)
            .into_iter()
            .map(|cc| cc.into_iter().map(|index| self.index_to_node(index).unwrap().to_owned()).collect())
            .collect()
    }
}
