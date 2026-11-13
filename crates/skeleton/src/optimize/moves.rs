//! Topological moves: dissolving a boundary, chains, and exact chain relabeling.

use super::fidelity::{Measure, Tables};
use crate::label::{Port, constraints};
use crate::{EdgeLabel, LabeledSkeleton, SkeletonEdgeID, SkeletonNodeID, SkeletonResult};
use dualcube_types::prelude::*;
use slotmap::SecondaryMap;

pub(crate) fn degrees(skeleton: &crate::Skeleton) -> SecondaryMap<SkeletonNodeID, usize> {
    let mut degree: SecondaryMap<_, _> = skeleton.nodes.keys().map(|n| (n, 0)).collect();
    for edge in skeleton.edges.values() {
        for node in edge.nodes {
            degree[node] += 1;
        }
    }
    degree
}

/// Delete a boundary and merge its other patch into `keep` (merge lemma). Fails when the
/// graph would get parallel edges; labels may need repair afterwards.
pub(crate) fn dissolve(
    mesh: &Mesh<INPUT>,
    labeled: &mut LabeledSkeleton,
    edge: SkeletonEdgeID,
    keep: SkeletonNodeID,
) -> SkeletonResult<()> {
    let skeleton = &mut labeled.skeleton;
    let removed = skeleton.edges.remove(edge).ok_or("Missing edge")?;
    let other = removed
        .nodes
        .into_iter()
        .find(|&n| n != keep)
        .ok_or("Edge does not join `keep` to another node")?;
    let mut neighbors = HashSet::new();
    for data in skeleton.edges.values_mut() {
        for n in &mut data.nodes {
            if *n == other {
                *n = keep;
            }
        }
        if data.nodes[0] == data.nodes[1] {
            return Err("Merging would create a self-loop".into());
        }
        if data.nodes.contains(&keep) {
            let neighbor = data.nodes[usize::from(data.nodes[0] == keep)];
            if !neighbors.insert(neighbor) {
                return Err("Merging would create parallel edges".into());
            }
        }
    }
    for crossing in removed.boundary.crossings {
        let data = skeleton.surface.crossings.remove(crossing).unwrap();
        let key = crate::surface::canonical(mesh, data.exit);
        let order = skeleton.surface.edge_order.get_mut(&key).unwrap();
        order.retain(|&c| c != crossing);
        if order.is_empty() {
            skeleton.surface.edge_order.remove(&key);
        }
    }
    skeleton.nodes.remove(other);
    labeled.labels.remove(edge);
    Ok(())
}

pub(crate) fn same_label(a: EdgeLabel, b: EdgeLabel) -> bool {
    Port::from_label(a) == Port::from_label(b)
}

/// A maximal path whose interior nodes have degree two, from `nodes[0]` to its last node.
#[derive(Clone, Debug)]
pub(crate) struct Chain {
    pub nodes: Vec<SkeletonNodeID>,
    pub edges: Vec<SkeletonEdgeID>,
}

impl Chain {
    /// Whether edge `i` is stored in chain direction.
    pub fn forward(&self, skeleton: &crate::Skeleton, i: usize) -> bool {
        skeleton.edges[self.edges[i]].nodes[0] == self.nodes[i]
    }
}

/// All chains, each once.
pub(crate) fn chains(skeleton: &crate::Skeleton) -> Vec<Chain> {
    let degree = degrees(skeleton);
    let mut incident = SecondaryMap::<SkeletonNodeID, Vec<SkeletonEdgeID>>::new();
    for node in skeleton.nodes.keys() {
        incident.insert(node, Vec::new());
    }
    for (id, edge) in &skeleton.edges {
        for node in edge.nodes {
            incident[node].push(id);
        }
    }
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    for (start, &d) in &degree {
        if d == 2 {
            continue;
        }
        for &first in &incident[start] {
            if seen.contains(&first) {
                continue;
            }
            let mut chain = Chain {
                nodes: vec![start],
                edges: Vec::new(),
            };
            let mut edge = first;
            loop {
                seen.insert(edge);
                chain.edges.push(edge);
                let data = &skeleton.edges[edge];
                let current = *chain.nodes.last().unwrap();
                let next = data.nodes[usize::from(data.nodes[0] == current)];
                chain.nodes.push(next);
                if degree[next] != 2 {
                    break;
                }
                edge = *incident[next].iter().find(|&&e| e != edge).unwrap();
            }
            result.push(chain);
        }
    }
    // A component that is one cycle of degree-two nodes: a chain from a node back to itself.
    for (first, data) in &skeleton.edges {
        if seen.contains(&first) {
            continue;
        }
        let start = data.nodes[0];
        let mut chain = Chain {
            nodes: vec![start],
            edges: Vec::new(),
        };
        let mut edge = first;
        loop {
            seen.insert(edge);
            chain.edges.push(edge);
            let data = &skeleton.edges[edge];
            let current = *chain.nodes.last().unwrap();
            let next = data.nodes[usize::from(data.nodes[0] == current)];
            chain.nodes.push(next);
            if next == start {
                break;
            }
            edge = *incident[next].iter().find(|&&e| e != edge).unwrap();
        }
        result.push(chain);
    }
    result
}

/// Gain of a [alignment, distortion] pair; higher is better.
pub(crate) fn gain(measure: Measure, value: [f64; 2]) -> f64 {
    match measure {
        Measure::Alignment => value[0],
        Measure::Distortion => -value[1],
    }
}

/// Best labels for a chain under fixed geometry, subject to the cycle-closing test
/// (free end ports, no immediate reversal, an order-feasible displacement mask).
/// Returns the new labels and the chain's part of the (unnormalized) gain.
pub(crate) fn relabel(
    tables: &Tables,
    labeled: &LabeledSkeleton,
    chain: &Chain,
    measure: Measure,
) -> Option<(SecondaryMap<SkeletonEdgeID, EdgeLabel>, f64)> {
    let arm = |i: usize, port: usize| {
        let e = chain.edges[i];
        gain(measure, tables.arm(chain.nodes[i], e, port / 2))
            + gain(measure, tables.arm(chain.nodes[i + 1], e, port / 2))
    };
    let node = |n: SkeletonNodeID, used: u8| gain(measure, tables.node(n, 63 & !used));
    relabel_by(labeled, chain, arm, node)
}

/// Chain labels maximizing `arm(i, port)` per edge (port along the chain) plus `node(n, used)`
/// per chain node with its used ports, under the same validity rules as [`relabel`].
pub(crate) fn relabel_by(
    labeled: &LabeledSkeleton,
    chain: &Chain,
    arm: impl Fn(usize, usize) -> f64,
    node: impl Fn(SkeletonNodeID, u8) -> f64,
) -> Option<(SecondaryMap<SkeletonEdgeID, EdgeLabel>, f64)> {
    let skeleton = &labeled.skeleton;
    let mut others = labeled.labels.clone();
    for &e in &chain.edges {
        others.remove(e);
    }
    let used = constraints::used_ports(skeleton, &others).ok()?;
    let (u, v) = (chain.nodes[0], *chain.nodes.last().unwrap());
    let masks: [bool; 64] = std::array::from_fn(|mask| {
        constraints::feasible(skeleton, &others, Some(([u, v], mask as u8)))
    });
    let k = chain.edges.len();
    let arms = &arm;
    let mut best: Option<(f64, Vec<usize>)> = None;
    for first in 0..6 {
        if used[u] & (1 << first) != 0 {
            continue;
        }
        let mut value = vec![f64::NEG_INFINITY; 384];
        let mut back = vec![vec![usize::MAX; 384]; k];
        value[first * 64 + (1 << first)] = arms(0, first)
            + if u == v {
                0.
            } else {
                node(u, used[u] | 1 << first)
            };
        for i in 1..k {
            let mut next = vec![f64::NEG_INFINITY; 384];
            for (state, &current) in value.iter().enumerate() {
                if current == f64::NEG_INFINITY {
                    continue;
                }
                let (p, mask) = (state / 64, state % 64);
                for q in 0..6 {
                    if q == p ^ 1 {
                        continue;
                    }
                    let candidate =
                        current + node(chain.nodes[i], 1 << (p ^ 1) | 1 << q) + arms(i, q);
                    let target = q * 64 + (mask | 1 << q);
                    if candidate > next[target] {
                        next[target] = candidate;
                        back[i][target] = state;
                    }
                }
            }
            value = next;
        }
        for (state, &current) in value.iter().enumerate() {
            if current == f64::NEG_INFINITY || !masks[state % 64] {
                continue;
            }
            let end = (state / 64) ^ 1;
            let start_bit = if u == v { 1 << first } else { 0 };
            if used[v] & (1 << end) != 0 || start_bit & (1 << end) != 0 {
                continue;
            }
            let total = current + node(v, used[v] | start_bit | 1 << end);
            if best.as_ref().is_none_or(|(b, _)| total > *b) {
                let mut ports = vec![0; k];
                let mut s = state;
                for i in (0..k).rev() {
                    ports[i] = s / 64;
                    if i > 0 {
                        s = back[i][s];
                    }
                }
                best = Some((total, ports));
            }
        }
    }
    let (total, ports) = best?;
    let mut labels = labeled.labels.clone();
    for (i, &p) in ports.iter().enumerate() {
        let port = Port::ALL[p];
        let port = if chain.forward(skeleton, i) {
            port
        } else {
            port.opposite()
        };
        labels.insert(chain.edges[i], port.label());
    }
    Some((labels, total))
}
