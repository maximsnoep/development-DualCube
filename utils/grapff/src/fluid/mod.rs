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

    pub fn shortest_cycle(&self, a: T, weight_function: impl Fn((T, T)) -> Float) -> Option<(Vec<T>, Float)> {
        (self.neighborhood)(a)
            .iter()
            .filter_map(|&neighbor| self.shortest_path(neighbor, a, &weight_function))
            .sorted_by(|(_, cost1), (_, cost2)| cost1.cmp(cost2))
            .next()
            .map(|(path, score)| {
                let (last, rest) = path.split_last().unwrap();
                ([&[*last], rest].concat(), score + weight_function((a, *path.first().unwrap())))
            })
    }

    // Should do this for each connected component (degree of freedom!)
    pub fn two_color(&self, nodes: &[T]) -> Option<(HashSet<T>, HashSet<T>)> {
        let mut pool = nodes.to_vec();
        let mut color1 = HashSet::new();
        let mut color2 = HashSet::new();

        while let Some(s) = pool.pop() {
            let mut queue = vec![s];

            while let Some(node) = queue.pop() {
                pool.retain(|x| x != &node);
                if color1.contains(&node) || color2.contains(&node) {
                    continue;
                }

                let neighbors = (self.neighborhood)(node);

                if neighbors.iter().any(|x| color1.contains(x)) {
                    if neighbors.iter().any(|x| color2.contains(x)) {
                        return None;
                    }
                    color2.insert(node);
                } else if neighbors.iter().any(|x| color2.contains(x)) {
                    if neighbors.iter().any(|x| color1.contains(x)) {
                        return None;
                    }

                    color1.insert(node);
                } else {
                    // Degree of freedom.
                    color2.insert(node);
                }

                queue.extend(neighbors);
            }
        }
        Some((color1, color2))
    }

    pub fn topological_sort(&self, nodes: &[T]) -> Option<Vec<T>> {
        pathfinding::directed::topological_sort::topological_sort(nodes, |&x| (self.neighborhood)(x)).ok()
    }
}

impl<T: Eq + Hash + Clone + Copy> Grapff<T, (T, T)> for FluidGraph<'_, T> {
    fn neighbors(&self, v: T) -> Vec<T> {
        (self.neighborhood)(v)
    }

    fn shortest_path(&self, a: T, b: T, weight_function: impl Fn((T, T)) -> Float) -> Option<(Vec<T>, Float)> {
        self.shortest_path_heuristic(a, b, &weight_function, |_| ZERO)
    }

    fn shortest_path_heuristic(&self, a: T, b: T, w: impl Fn((T, T)) -> Float, h: impl Fn((T, T)) -> Float) -> Option<(Vec<T>, Float)> {
        pathfinding::prelude::astar(
            &a,
            |&elem| self.neighbors(elem).into_iter().map(|neighbor| (neighbor, w((elem, neighbor)))).collect_vec(),
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
