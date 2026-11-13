//! Minimum s-t cut by Dinic's algorithm on nonnegative real capacities.

use std::collections::VecDeque;

pub(crate) struct Network {
    /// Per arc: head, residual capacity. Arc `i ^ 1` is the reverse of arc `i`.
    arcs: Vec<(usize, f64)>,
    out: Vec<Vec<usize>>,
}

impl Network {
    pub fn new(nodes: usize) -> Self {
        Self {
            arcs: Vec::new(),
            out: vec![Vec::new(); nodes],
        }
    }

    /// Capacity `forward` from a to b and `backward` from b to a.
    pub fn add(&mut self, a: usize, b: usize, forward: f64, backward: f64) {
        self.out[a].push(self.arcs.len());
        self.arcs.push((b, forward.max(0.)));
        self.out[b].push(self.arcs.len());
        self.arcs.push((a, backward.max(0.)));
    }

    /// Nodes on the source side of a minimum cut.
    pub fn source_side(&mut self, source: usize, sink: usize) -> Vec<bool> {
        const EPS: f64 = 1e-12;
        let n = self.out.len();
        loop {
            let mut level = vec![usize::MAX; n];
            level[source] = 0;
            let mut queue = VecDeque::from([source]);
            while let Some(a) = queue.pop_front() {
                for &arc in &self.out[a] {
                    let (b, capacity) = self.arcs[arc];
                    if capacity > EPS && level[b] == usize::MAX {
                        level[b] = level[a] + 1;
                        queue.push_back(b);
                    }
                }
            }
            if level[sink] == usize::MAX {
                return level.iter().map(|&l| l != usize::MAX).collect();
            }
            let mut next = vec![0; n];
            while self.augment(source, sink, &level, &mut next) > EPS {}
        }
    }

    /// One blocking-flow path by iterative depth-first search.
    fn augment(&mut self, source: usize, sink: usize, level: &[usize], next: &mut [usize]) -> f64 {
        const EPS: f64 = 1e-12;
        let mut path: Vec<usize> = Vec::new();
        let mut node = source;
        loop {
            if node == sink {
                let flow = path
                    .iter()
                    .map(|&a| self.arcs[a].1)
                    .fold(f64::INFINITY, f64::min);
                for &a in &path {
                    self.arcs[a].1 -= flow;
                    self.arcs[a ^ 1].1 += flow;
                }
                return flow;
            }
            let mut advanced = false;
            while next[node] < self.out[node].len() {
                let arc = self.out[node][next[node]];
                let (b, capacity) = self.arcs[arc];
                if capacity > EPS && level[b] == level[node] + 1 {
                    path.push(arc);
                    node = b;
                    advanced = true;
                    break;
                }
                next[node] += 1;
            }
            if !advanced {
                if node == source {
                    return 0.;
                }
                let arc = path.pop().unwrap();
                node = self.arcs[arc ^ 1].0;
                next[node] += 1;
            }
        }
    }
}
