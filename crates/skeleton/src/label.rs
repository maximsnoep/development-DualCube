//! Angle-based tree labeling, surface-preserving cycle repair, then exact relabeling.

pub(crate) mod constraints;
pub(crate) mod exact;
mod junction;
pub mod path;
pub(crate) mod subdivide;
mod tree;

use crate::{EdgeLabel, LabeledSkeleton, Skeleton, SkeletonEdgeID, SkeletonNodeID, SkeletonResult};
use dualcube_types::prelude::{Direction, HasPosition, INPUT, Mesh, Sign, Vector3D};
use serde::{Deserialize, Serialize};
use slotmap::SecondaryMap;
use std::collections::HashMap;

/// Stable indices into six-entry cost tables and six-bit direction masks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Port {
    PositiveX,
    NegativeX,
    PositiveY,
    NegativeY,
    PositiveZ,
    NegativeZ,
}

impl Port {
    pub const ALL: [Self; 6] = [
        Self::PositiveX,
        Self::NegativeX,
        Self::PositiveY,
        Self::NegativeY,
        Self::PositiveZ,
        Self::NegativeZ,
    ];
    pub fn opposite(self) -> Self {
        Self::ALL[(self as usize) ^ 1]
    }
    /// 0, 1, 2 for X, Y, Z.
    pub fn axis(self) -> usize {
        self as usize / 2
    }
    pub fn is_positive(self) -> bool {
        self as usize % 2 == 0
    }
    pub fn bit(self) -> u8 {
        1 << self as u8
    }
    pub fn label(self) -> EdgeLabel {
        EdgeLabel {
            direction: [Direction::X, Direction::Y, Direction::Z][self.axis()],
            sign: if self as usize % 2 == 0 {
                Sign::Positive
            } else {
                Sign::Negative
            },
        }
    }
    pub fn from_label(label: EdgeLabel) -> Self {
        let axis = match label.direction {
            Direction::X => 0,
            Direction::Y => 1,
            Direction::Z => 2,
        };
        Self::ALL[axis * 2 + usize::from(label.sign == Sign::Negative)]
    }
}

/// Unweighted angles in radians. Coincident embedded endpoints have no preference.
pub fn angle_costs(vector: Vector3D) -> [f64; 6] {
    // TODO: Evaluate surface patch deformation and label fit beyond edge-axis angle.
    let scale = vector.amax();
    if scale == 0. {
        return [std::f64::consts::FRAC_PI_2; 6];
    }
    let unit = (vector / scale).normalize();
    std::array::from_fn(|i| {
        let component = unit[i / 2] * if i % 2 == 0 { 1. } else { -1. };
        component.clamp(-1., 1.).acos()
    })
}

/// Distinct ports for `items` items at a node, none in `occupied`, with the least value:
/// `step(i, port, value)` adds item `i` on `port` to the value of the items before it. The
/// value and the ports in item order; no ports when no assignment is finite.
fn assign_ports(
    items: usize,
    occupied: u8,
    step: impl Fn(usize, Port, f64) -> f64,
) -> (f64, Vec<Port>) {
    let mut dp = [f64::INFINITY; 64];
    dp[occupied as usize] = 0.;
    let mut back = vec![[None; 64]; items];
    for (i, back) in back.iter_mut().enumerate() {
        let mut next = [f64::INFINITY; 64];
        for mask in 0..64 {
            if !dp[mask].is_finite() {
                continue;
            }
            for p in Port::ALL {
                if mask & p.bit() as usize != 0 {
                    continue;
                }
                let value = step(i, p, dp[mask]);
                let target = mask | p.bit() as usize;
                if value < next[target] {
                    next[target] = value;
                    back[target] = Some((mask, p));
                }
            }
        }
        dp = next;
    }
    let (mut mask, &best) = dp
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .unwrap();
    if !best.is_finite() {
        return (best, Vec::new());
    }
    let mut ports = Vec::with_capacity(items);
    for i in (0..items).rev() {
        let (previous, p) = back[i][mask].unwrap();
        ports.push(p);
        mask = previous;
    }
    ports.reverse();
    (best, ports)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JunctionSplit {
    pub original_node: SkeletonNodeID,
    pub added_node: SkeletonNodeID,
    pub boundary: SkeletonEdgeID,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Repair {
    pub original_edge: SkeletonEdgeID,
    pub path: Vec<SkeletonEdgeID>,
}

/// Retained junction splits, tree solution, and cycle repairs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Attempt {
    pub skeleton: Skeleton,
    pub labels: SecondaryMap<SkeletonEdgeID, EdgeLabel>,
    pub tree_edges: Vec<SkeletonEdgeID>,
    pub tree_labels: SecondaryMap<SkeletonEdgeID, EdgeLabel>,
    /// The skeleton `tree_labels` belong to: after junction splits, before cycle repairs.
    #[serde(default)]
    pub tree_skeleton: Option<Skeleton>,
    pub repairs: Vec<Repair>,
    pub junction_splits: Vec<JunctionSplit>,
}

impl Attempt {
    /// Whether an edge lies outside the spanning tree: a cycle edge or one of its repairs.
    pub fn outside_tree(&self, edge: SkeletonEdgeID) -> bool {
        !self.tree_edges.is_empty() && !self.tree_labels.contains_key(edge)
    }

    pub fn labeled(&self) -> LabeledSkeleton {
        LabeledSkeleton {
            skeleton: self.skeleton.clone(),
            labels: self.labels.clone(),
        }
    }
}

/// Label an extracted skeleton (validated by extraction); the result is validated.
pub fn attempt(mesh: &Mesh<INPUT>, skeleton: &Skeleton) -> SkeletonResult<Attempt> {
    skeleton.debug_validate(mesh);
    let start = std::time::Instant::now();
    let mut result = Attempt {
        skeleton: skeleton.clone(),
        labels: SecondaryMap::new(),
        tree_edges: Vec::new(),
        tree_labels: SecondaryMap::new(),
        tree_skeleton: None,
        repairs: Vec::new(),
        junction_splits: Vec::new(),
    };
    junction::reduce(mesh, &mut result)?;
    let mut cost_cache = HashMap::new();
    let table = result
        .skeleton
        .edges
        .keys()
        .map(|e| (e, cached_cost(mesh, &result.skeleton, e, &mut cost_cache)))
        .collect();
    let (tree_edges, labels) = tree::solve(&result.skeleton, &table)?;
    let remaining: Vec<_> = result
        .skeleton
        .edges
        .keys()
        .filter(|e| !labels.contains_key(*e))
        .collect();
    result.tree_edges = tree_edges;
    result.tree_labels = labels.clone();
    result.tree_skeleton = Some(result.skeleton.clone());
    result.labels = labels;
    for edge in remaining {
        let (skeleton, added, labels) = repair(
            mesh,
            &result.skeleton,
            &result.labels,
            edge,
            &mut cost_cache,
        )?;
        result.skeleton = skeleton;
        result.labels.extend(labels);
        let mut path = vec![edge];
        path.extend(added);
        result.repairs.push(Repair {
            original_edge: edge,
            path,
        });
    }
    // The tree and the repairs fix labels greedily; cycles they close often force mislabels
    // that a labeling of all edges at once avoids.
    let penalties = exact::plane_penalties(mesh, &result.skeleton);
    result.labels = exact::search(&result.skeleton, &result.labels, &penalties, exact::BUDGET);
    result.labeled().validate(mesh)?;
    tracing::info!(
        "Labeling: {} edges labeled, {} junction splits, {} cycle repairs, {:.2}s",
        result.labels.len(),
        result.junction_splits.len(),
        result.repairs.len(),
        start.elapsed().as_secs_f64()
    );
    Ok(result)
}

/// Direction of an edge: the normal of its boundary's plane, scaled by the enclosed area, from
/// its first patch to its second (the boundary runs counterclockwise around the first, seen
/// from outside). The loop is drawn with the label's axis, so its own plane decides; the
/// direction between patch centres can differ from it by any angle.
pub(crate) fn boundary_normal(
    mesh: &Mesh<INPUT>,
    skeleton: &Skeleton,
    edge: SkeletonEdgeID,
) -> Vector3D {
    let points: Vec<Vector3D> = skeleton.edges[edge]
        .boundary
        .crossings
        .iter()
        .map(|&c| {
            let e = skeleton.surface.crossings[c].exit;
            (mesh.position(mesh.root(e)) + mesh.position(mesh.toor(e))) / 2.
        })
        .collect();
    let centre: Vector3D = points.iter().sum::<Vector3D>() / points.len() as f64;
    let mut normal = Vector3D::zeros();
    for i in 0..points.len() {
        normal -= (points[i] - centre).cross(&(points[(i + 1) % points.len()] - centre));
    }
    normal
}

fn cached_cost(
    mesh: &Mesh<INPUT>,
    skeleton: &Skeleton,
    edge: SkeletonEdgeID,
    cache: &mut HashMap<[u64; 3], [f64; 6]>,
) -> [f64; 6] {
    let vector = boundary_normal(mesh, skeleton, edge);
    // TODO: Include surface candidate identity and labels when caching richer patch costs.
    let key = [vector.x.to_bits(), vector.y.to_bits(), vector.z.to_bits()];
    *cache.entry(key).or_insert_with(|| angle_costs(vector))
}

/// The chain through `edge`: the maximal path whose interior nodes have degree two, as steps
/// (edge, whether it is traversed from its first node) from the chain's first node to its last.
/// A component that is one ring is a chain from a node back to itself.
pub(crate) fn chain(
    skeleton: &Skeleton,
    edge: SkeletonEdgeID,
) -> (Vec<(SkeletonEdgeID, bool)>, [SkeletonNodeID; 2]) {
    let degree = skeleton.degrees();
    let [first, last] = skeleton.edges[edge].nodes;
    // Steps leaving `from` through degree-two nodes, not continuing past `stop`.
    let walk = |from: SkeletonNodeID, stop: SkeletonNodeID| {
        let (mut node, mut previous, mut steps) = (from, edge, Vec::new());
        while degree[node] == 2 && node != stop {
            let (next, data) = skeleton
                .edges
                .iter()
                .find(|(id, e)| *id != previous && e.nodes.contains(&node))
                .expect("a degree-two node has a second edge");
            steps.push((next, data.nodes[0] == node));
            node = data.other(node);
            previous = next;
        }
        (steps, node)
    };
    let (after, end) = walk(last, first);
    let (before, start) = if end == first {
        (Vec::new(), first)
    } else {
        walk(first, last)
    };
    // Steps before the edge were walked away from it.
    let mut steps: Vec<_> = before.into_iter().rev().map(|(e, f)| (e, !f)).collect();
    steps.push((edge, true));
    steps.extend(after);
    (steps, [start, end])
}

/// Costs along a chain: a step against its edge swaps the signs.
fn along(
    mesh: &Mesh<INPUT>,
    skeleton: &Skeleton,
    steps: &[(SkeletonEdgeID, bool)],
    cache: &mut HashMap<[u64; 3], [f64; 6]>,
) -> Vec<[f64; 6]> {
    steps
        .iter()
        .map(|&(e, forward)| {
            let cost = cached_cost(mesh, skeleton, e, cache);
            std::array::from_fn(|p| cost[if forward { p } else { p ^ 1 }])
        })
        .collect()
}

/// The cheapest valid ports along a chain for given costs, all other labels fixed.
fn chain_solver(
    skeleton: &Skeleton,
    labels: &SecondaryMap<SkeletonEdgeID, EdgeLabel>,
    steps: &[(SkeletonEdgeID, bool)],
    [start, end]: [SkeletonNodeID; 2],
) -> SkeletonResult<impl Fn(&[[f64; 6]]) -> Option<path::Solution>> {
    let mut others = labels.clone();
    for &(e, _) in steps {
        others.remove(e);
    }
    let used = constraints::used_ports(skeleton, &others)?;
    let mut masks = [false; 64];
    for (mask, feasible) in masks.iter_mut().enumerate() {
        *feasible = constraints::feasible(skeleton, &others, Some(([start, end], mask as u8)));
    }
    let (first, last) = (used[start], used[end]);
    Ok(move |costs: &[[f64; 6]]| {
        if start != end {
            let limits = path::Constraints {
                start_ports: 63 & !first,
                end_ports: 63 & !last,
                masks,
            };
            return path::optimize(costs, &limits);
        }
        // A ring leaves and enters one node: through two different ports.
        Port::ALL
            .into_iter()
            .filter(|p| first & p.bit() == 0)
            .filter_map(|p| {
                let limits = path::Constraints {
                    start_ports: p.bit(),
                    end_ports: 63 & !first & !p.bit(),
                    masks,
                };
                path::optimize(costs, &limits)
            })
            .min_by(|a, b| a.cost.total_cmp(&b.cost))
    })
}

/// Steps along labelled edges from `from` to `to`, fewest first.
fn route(
    skeleton: &Skeleton,
    labels: &SecondaryMap<SkeletonEdgeID, EdgeLabel>,
    from: SkeletonNodeID,
    to: SkeletonNodeID,
) -> Option<Vec<(SkeletonEdgeID, bool)>> {
    let mut reached: HashMap<SkeletonNodeID, Option<(SkeletonEdgeID, bool)>> =
        HashMap::from([(from, None)]);
    let mut queue = std::collections::VecDeque::from([from]);
    while let Some(node) = queue.pop_front() {
        if node == to {
            break;
        }
        for (id, edge) in &skeleton.edges {
            if labels.contains_key(id) && edge.nodes.contains(&node) {
                let next = edge.other(node);
                if !reached.contains_key(&next) {
                    reached.insert(next, Some((id, edge.nodes[0] == node)));
                    queue.push_back(next);
                }
            }
        }
    }
    let mut steps = Vec::new();
    let mut node = to;
    while let Some((id, forward)) = *reached.get(&node)? {
        steps.push((id, forward));
        node = skeleton.edges[id].other(node);
    }
    steps.reverse();
    Some(steps)
}

/// One way to close a cycle: steps, the slots that get a new boundary, and a port per
/// boundary of the resulting sequence.
struct Closing {
    steps: Vec<(SkeletonEdgeID, bool)>,
    slots: Vec<usize>,
    ports: Vec<Port>,
}

/// New boundaries per step: slot `2 i` lies before step `i` along the path, in the patch the
/// step leaves, and slot `2 i + 1` after it, in the patch it enters.
fn additions(steps: usize, slots: &[usize]) -> Vec<[usize; 2]> {
    let mut counts = vec![[0; 2]; steps];
    for &slot in slots {
        counts[slot / 2][slot % 2] += 1;
    }
    counts
}

/// Close the cycle of an unlabeled edge with the fewest new boundaries that admit a valid
/// labeling, and among those the cheapest. A new boundary runs through the middle of a patch
/// beside one of the cycle's boundaries (a parallel collar where the patch has no middle). The whole cycle is relabelled when that is valid:
/// the edge's chain and the labelled route between its ends, junctions included. Otherwise
/// only the chain is, which always works. Returns the skeleton, the added collars, and the
/// new labels.
fn repair(
    mesh: &Mesh<INPUT>,
    skeleton: &Skeleton,
    labels: &SecondaryMap<SkeletonEdgeID, EdgeLabel>,
    edge: SkeletonEdgeID,
    cache: &mut HashMap<[u64; 3], [f64; 6]>,
) -> SkeletonResult<(
    Skeleton,
    Vec<SkeletonEdgeID>,
    Vec<(SkeletonEdgeID, EdgeLabel)>,
)> {
    let (chain, [start, end]) = chain(skeleton, edge);
    let solve = chain_solver(skeleton, labels, &chain, [start, end])?;
    // The cycle: the chain, then back along labelled edges outside it.
    let mut others = labels.clone();
    for &(e, _) in &chain {
        others.remove(e);
    }
    let mut cycle = chain.clone();
    if start != end {
        cycle.extend(route(skeleton, &others, end, start).ok_or("Cycle edge closes no cycle")?);
    }
    for &(e, _) in &cycle {
        others.remove(e);
    }
    let used = constraints::used_ports(skeleton, &others)?;
    let costs = along(mesh, skeleton, &cycle, cache);
    // The node before every step, and the first again.
    let mut nodes = vec![start];
    for &(e, _) in &cycle {
        nodes.push(skeleton.edges[e].other(*nodes.last().unwrap()));
    }
    // What a new boundary in every slot would cost: its own plane decides, as for any edge.
    let ground = subdivide::Ground::build(mesh, skeleton)?;
    let mut slot_costs = Vec::new();
    for (i, &(e, _)) in cycle.iter().enumerate() {
        for (k, node) in [nodes[i], nodes[i + 1]].into_iter().enumerate() {
            slot_costs.push(match subdivide::middle(mesh, skeleton, &ground, e, node) {
                // Along the path the normal points into the patch after the step, and out
                // of the patch before it.
                Some(middle) => angle_costs(if k == 1 {
                    middle.normal
                } else {
                    -middle.normal
                }),
                None => costs[i],
            });
        }
    }
    // Costs and ports in use along the first `steps` steps with the slots filled; nothing is
    // in use between a boundary and the new ones beside it.
    let sequence = |steps: usize, slots: &[usize]| {
        let (mut sequence, mut taken) = (Vec::new(), Vec::new());
        for (i, [before, after]) in additions(steps, slots).into_iter().enumerate() {
            let from = sequence.len();
            sequence.extend(std::iter::repeat_n(slot_costs[2 * i], before));
            sequence.push(costs[i]);
            sequence.extend(std::iter::repeat_n(slot_costs[2 * i + 1], after));
            taken.resize(sequence.len(), 0);
            taken[from] = used[nodes[i]];
        }
        (sequence, taken)
    };
    // A closed path is order-feasible alone when every axis has both signs or neither.
    let mut closed = [false; 64];
    for (mask, ok) in closed.iter_mut().enumerate() {
        *ok = (0..3).all(|axis| matches!((mask >> (axis * 2)) & 3, 0 | 3));
    }
    let around = |slots: &[usize]| -> Option<(f64, Vec<Port>)> {
        let (sequence, taken) = sequence(cycle.len(), slots);
        // The cycle leaves and enters its first node through two different free ports.
        let best = Port::ALL
            .into_iter()
            .filter(|p| used[start] & p.bit() == 0)
            .filter_map(|p| {
                let limits = path::Constraints {
                    start_ports: p.bit(),
                    end_ports: 63 & !used[start] & !p.bit(),
                    masks: closed,
                };
                path::optimize_through(&sequence, &limits, &taken)
            })
            .min_by(|a, b| a.cost.total_cmp(&b.cost))?;
        // Other cycles through these nodes can still contradict it: check all orders.
        let mut runs = Vec::new();
        let mut at = 0;
        for (i, [before, after]) in additions(cycle.len(), slots).into_iter().enumerate() {
            let count = before + 1 + after;
            let mask = best.ports[at..at + count]
                .iter()
                .fold(0, |m, p| m | p.bit());
            runs.push(([nodes[i], nodes[i + 1]], mask));
            at += count;
        }
        constraints::feasible_with(skeleton, &others, &runs).then_some((best.cost, best.ports))
    };
    for added in 0..=4 {
        let mut best: Option<(f64, Closing)> = None;
        if cycle.len() > chain.len() {
            for slots in copies(2 * cycle.len(), added) {
                if let Some((cost, ports)) = around(&slots)
                    && best.as_ref().is_none_or(|(old, _)| cost < *old)
                {
                    let steps = cycle.clone();
                    best = Some((
                        cost,
                        Closing {
                            steps,
                            slots,
                            ports,
                        },
                    ));
                }
            }
        }
        if best.is_none() {
            for slots in copies(2 * chain.len(), added) {
                if let Some(solution) = solve(&sequence(chain.len(), &slots).0)
                    && best.as_ref().is_none_or(|(old, _)| solution.cost < *old)
                {
                    let (steps, ports) = (chain.clone(), solution.ports);
                    best = Some((
                        solution.cost,
                        Closing {
                            steps,
                            slots,
                            ports,
                        },
                    ));
                }
            }
        }
        let Some((_, closing)) = best else {
            continue;
        };
        let mut split = skeleton.clone();
        let mut new = Vec::new();
        let mut path = Vec::new();
        let counts = additions(closing.steps.len(), &closing.slots);
        for (i, &(e, forward)) in closing.steps.iter().enumerate() {
            // The run along the edge's own direction; `outer` are its boundaries next to the
            // patches at the edge's two ends.
            let mut run = vec![e];
            let mut outer = [e, e];
            let ends = split.edges[e].nodes;
            for (k, &count) in counts[i].iter().enumerate() {
                // Before the step is the edge's first end when the step runs forward.
                let side = usize::from((k == 1) == forward);
                for _ in 0..count {
                    if let Some((next, loop_)) =
                        subdivide::split(mesh, &split, outer[side], ends[side])
                    {
                        split = next;
                        if side == 1 {
                            run.push(loop_);
                        } else {
                            run.insert(0, loop_);
                        }
                        outer[side] = loop_;
                        new.push(loop_);
                    } else {
                        // The collar follows its original, which keeps its first end.
                        let (next, collar) = subdivide::collar(mesh, &split, outer[side], 1)?;
                        split = next;
                        if side == 1 {
                            run.push(collar);
                            outer[side] = collar;
                        } else {
                            run.insert(1, collar);
                            // The original's second end now belongs to the collar.
                            if outer[1] == outer[0] {
                                outer[1] = collar;
                            }
                        }
                        new.push(collar);
                    }
                }
            }
            if !forward {
                run.reverse();
            }
            path.extend(run.into_iter().map(|e| (e, forward)));
        }
        let added = new;
        let labels = path
            .iter()
            .zip(closing.ports)
            .map(|(&(e, forward), port)| (e, if forward { port } else { port.opposite() }.label()))
            .collect();
        return Ok((split, added, labels));
    }
    // Any unlabeled edge closes with at most four subdivisions (theory: labeling cycle closing).
    panic!("Impossible: cycle edge {edge:?} has no valid labeling with 0–4 subdivisions")
}

/// Which of `steps` chain steps to copy, `count` times in all, as sorted positions. Long chains
/// only copy their two ends.
fn copies(steps: usize, count: usize) -> Vec<Vec<usize>> {
    const LIMIT: usize = 2000;
    let grow = |positions: &[usize]| {
        let mut sets = vec![vec![]];
        for _ in 0..count {
            sets = sets
                .into_iter()
                .flat_map(|set: Vec<usize>| {
                    let least = set.last().copied().unwrap_or(0);
                    positions
                        .iter()
                        .filter(move |&&p| p >= least)
                        .map(move |&p| {
                            let mut next = set.clone();
                            next.push(p);
                            next
                        })
                })
                .collect();
            if sets.len() > LIMIT {
                return None;
            }
        }
        Some(sets)
    };
    let all: Vec<usize> = (0..steps).collect();
    grow(&all)
        .unwrap_or_else(|| grow(&[0, steps - 1]).expect("two positions stay within the limit"))
}

#[cfg(test)]
mod tests;
