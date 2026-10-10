use crate::{Float, Grapff, ZERO};
use itertools::Itertools;
use std::collections::HashSet;
use std::hash::Hash;

pub struct FluidGraph<'a, T: Eq + Hash + Clone + Copy> {
    neighborhood: Box<dyn Fn(T) -> Vec<T> + 'a>,
}

impl<'a, T: Eq + Hash + Clone + Copy> FluidGraph<'a, T> {
    pub fn new(neighborhood: impl Fn(T) -> Vec<T> + 'a) -> Self {
        Self {
            neighborhood: Box::new(neighborhood),
        }
    }

    pub fn topological_sort(&self, nodes: &[T]) -> Option<Vec<T>> {
        pathfinding::directed::topological_sort::topological_sort(nodes, |&x| {
            (self.neighborhood)(x)
        })
        .ok()
    }
}

impl<T: Eq + Hash + Clone + Copy> Grapff<T, (T, T)> for FluidGraph<'_, T> {
    fn neighbors(&self, v: T) -> Vec<T> {
        (self.neighborhood)(v)
    }

    fn shortest_path(
        &self,
        a: T,
        b: T,
        weight_function: impl Fn((T, T)) -> Float,
    ) -> Option<(Vec<T>, Float)> {
        self.shortest_path_heuristic(a, b, &weight_function, |_| ZERO)
    }

    fn shortest_path_heuristic(
        &self,
        a: T,
        b: T,
        w: impl Fn((T, T)) -> Float,
        h: impl Fn((T, T)) -> Float,
    ) -> Option<(Vec<T>, Float)> {
        pathfinding::prelude::astar(
            &a,
            |&elem| {
                self.neighbors(elem)
                    .into_iter()
                    .map(|neighbor| (neighbor, w((elem, neighbor))))
                    .collect_vec()
            },
            |&elem| h((elem, b)),
            |&elem| elem == b,
        )
    }

    fn connected_component(&self, v: T) -> HashSet<T> {
        pathfinding::prelude::bfs_reach(v, |&x| self.neighbors(x)).collect()
    }

    fn connected_components(&self, vs: &[T]) -> Vec<HashSet<T>> {
        let mut visited = HashSet::new();
        let mut ccs = vec![];
        for &node in vs {
            if visited.contains(&node) {
                continue;
            }
            let cc = self.connected_component(node);
            visited.extend(cc.clone());
            ccs.push(cc);
        }
        ccs.into_iter().collect()
    }
}
