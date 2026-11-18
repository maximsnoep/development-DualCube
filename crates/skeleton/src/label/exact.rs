//! Exact labeling: the valid labels with the least summed penalty, by branch and bound.

use super::{Port, angle_costs, boundary_normal, constraints};
use crate::{EdgeLabel, Skeleton, SkeletonEdgeID, SkeletonNodeID};
use dualcube_types::prelude::*;
use slotmap::SecondaryMap;
use std::collections::VecDeque;

/// Search nodes of the labeling stage's exact labeling.
pub(crate) const BUDGET: usize = 200_000;

/// Angle of the boundary's plane normal to every label beyond the best label's, per 20°.
pub(crate) fn plane_excess(mesh: &Mesh<INPUT>, skeleton: &Skeleton, edge: SkeletonEdgeID) -> [f64; 6] {
    let angle = angle_costs(boundary_normal(mesh, skeleton, edge));
    let best = angle.iter().copied().fold(f64::INFINITY, f64::min);
    angle.map(|a| (a - best) / 20f64.to_radians())
}

/// Plane excess, growing twenty times as steeply beyond 20° (mislabelling).
pub(crate) fn plane_penalties(
    mesh: &Mesh<INPUT>,
    skeleton: &Skeleton,
) -> SecondaryMap<SkeletonEdgeID, [f64; 6]> {
    skeleton
        .edges
        .keys()
        .map(|e| (e, plane_excess(mesh, skeleton, e).map(|x| x + 20. * (x - 1.).max(0.))))
        .collect()
}

/// The valid labeling with the least summed penalty, by branch and bound from `current`
/// (valid); after `budget` search nodes, the best found.
pub(crate) fn search(
    skeleton: &Skeleton,
    current: &SecondaryMap<SkeletonEdgeID, EdgeLabel>,
    penalties: &SecondaryMap<SkeletonEdgeID, [f64; 6]>,
    budget: usize,
) -> SecondaryMap<SkeletonEdgeID, EdgeLabel> {
    // Edges in breadth-first order, so that cycles close early.
    let mut order: Vec<SkeletonEdgeID> = Vec::new();
    let mut seen = SecondaryMap::<SkeletonNodeID, ()>::new();
    for root in skeleton.nodes.keys() {
        if seen.insert(root, ()).is_some() {
            continue;
        }
        let mut queue = VecDeque::from([root]);
        while let Some(node) = queue.pop_front() {
            for (id, edge) in skeleton.incident(node) {
                if !order.contains(&id) {
                    order.push(id);
                }
                if seen.insert(edge.other(node), ()).is_none() {
                    queue.push_back(edge.other(node));
                }
            }
        }
    }
    let cost = |labels: &SecondaryMap<SkeletonEdgeID, EdgeLabel>| -> f64 {
        labels
            .iter()
            .map(|(e, &l)| penalties[e][Port::from_label(l) as usize])
            .sum()
    };
    struct Search<'a> {
        skeleton: &'a Skeleton,
        order: Vec<SkeletonEdgeID>,
        rows: Vec<[f64; 6]>,
        /// Least penalty of the edges from `i` on.
        rest: Vec<f64>,
        labels: SecondaryMap<SkeletonEdgeID, EdgeLabel>,
        used: SecondaryMap<SkeletonNodeID, u8>,
        best: (f64, SecondaryMap<SkeletonEdgeID, EdgeLabel>),
        nodes: usize,
        budget: usize,
    }
    impl Search<'_> {
        fn go(&mut self, i: usize, cost: f64) {
            self.nodes += 1;
            if self.nodes > self.budget || cost + self.rest[i] >= self.best.0 - 1e-9 {
                return;
            }
            if !constraints::feasible(self.skeleton, &self.labels, None) {
                return;
            }
            if i == self.order.len() {
                self.best = (cost, self.labels.clone());
                return;
            }
            let edge = self.order[i];
            let [a, b] = self.skeleton.edges[edge].nodes;
            let mut ports: Vec<usize> = (0..6).collect();
            ports.sort_by(|&p, &q| self.rows[i][p].total_cmp(&self.rows[i][q]));
            for p in ports {
                let port = Port::ALL[p];
                if self.used[a] & port.bit() != 0 || self.used[b] & port.opposite().bit() != 0 {
                    continue;
                }
                self.used[a] |= port.bit();
                self.used[b] |= port.opposite().bit();
                self.labels.insert(edge, port.label());
                self.go(i + 1, cost + self.rows[i][p]);
                self.labels.remove(edge);
                self.used[a] &= !port.bit();
                self.used[b] &= !port.opposite().bit();
            }
        }
    }
    let rows: Vec<[f64; 6]> = order.iter().map(|&e| penalties[e]).collect();
    let mut rest = vec![0.; order.len() + 1];
    for i in (0..order.len()).rev() {
        rest[i] = rest[i + 1] + rows[i].iter().copied().fold(f64::INFINITY, f64::min);
    }
    let mut search = Search {
        skeleton,
        order,
        rows,
        rest,
        labels: SecondaryMap::new(),
        used: skeleton.nodes.keys().map(|n| (n, 0)).collect(),
        best: (cost(current), current.clone()),
        nodes: 0,
        budget,
    };
    search.go(0, 0.);
    if search.nodes > budget {
        tracing::warn!("Exact labeling: budget of {budget} nodes exhausted");
    }
    search.best.1
}

