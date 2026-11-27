//! Angle-based tree labeling, then surface-preserving cycle repair.

pub(crate) mod constraints;
mod junction;
pub mod path;
pub(crate) mod subdivide;
mod tree;

use crate::{EdgeLabel, LabeledSkeleton, Skeleton, SkeletonEdgeID, SkeletonNodeID, SkeletonResult};
use dualcube_types::prelude::{Direction, EdgeID, HasPosition, INPUT, Mesh, Sign, Vector3D};
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

/// The cost of every label (+X, -X, +Y, -Y, +Z, -Z) for a direction along the skeleton (no loop):
/// `1 - cos` of their angle, in units of a label 20° off, as [`alignment_costs`] of a loop whose
/// moves all point along `vector`. A zero vector has no preference.
pub fn label_costs(vector: Vector3D) -> [f64; 6] {
    let unit = vector.try_normalize(0.).unwrap_or_else(Vector3D::zeros);
    let scale = 1. - 20f64.to_radians().cos();
    std::array::from_fn(|i| {
        let component = unit[i / 2] * if i % 2 == 0 { 1. } else { -1. };
        (1. - component) / scale
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

/// DualCube's alignment of a loop that leaves a triangle through each of `exits` in turn: the
/// length-weighted mean of the unit `d × n` of its moves (`d` between the midpoints of the crossed
/// edges, as in DualCube's edge graph, in a triangle of normal `n`). Its component along an axis
/// is the mean cosine of the moves with that axis.
pub(crate) fn moves_alignment(mesh: &Mesh<INPUT>, exits: &[EdgeID]) -> Vector3D {
    let points: Vec<_> = exits
        .iter()
        .map(|&exit| {
            let middle = (mesh.position(mesh.root(exit)) + mesh.position(mesh.toor(exit))) / 2.;
            (middle, mesh.face(mesh.twin(exit)))
        })
        .collect();
    crate::quality::alignment(mesh, &points).direction
}

/// The cost of every label for a loop of [`moves_alignment`] `mean`: `1 - cos`, the mean cosine
/// lost against the label's axis, in units of a label 20° off.
pub(crate) fn alignment_costs(mean: Vector3D) -> [f64; 6] {
    let scale = 1. - 20f64.to_radians().cos();
    std::array::from_fn(|i| {
        let component = mean[i / 2] * if i % 2 == 0 { 1. } else { -1. };
        (1. - component) / scale
    })
}

/// The label costs of a boundary's loop, from its first patch to its second: its
/// [`alignment_costs`].
pub(crate) fn loop_costs(
    mesh: &Mesh<INPUT>,
    skeleton: &Skeleton,
    edge: SkeletonEdgeID,
) -> [f64; 6] {
    let surface = &skeleton.surface;
    let crossings = &skeleton.edges[edge].boundary.crossings;
    let exits: Vec<_> = crossings
        .iter()
        .map(|&c| surface.crossings[c].exit)
        .collect();
    alignment_costs(moves_alignment(mesh, &exits))
}

/// [`loop_costs`] of every boundary.
pub(crate) fn boundary_costs(
    mesh: &Mesh<INPUT>,
    skeleton: &Skeleton,
) -> SecondaryMap<SkeletonEdgeID, [f64; 6]> {
    skeleton
        .edges
        .keys()
        .map(|e| (e, loop_costs(mesh, skeleton, e)))
        .collect()
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
    // (The loop's normal identifies it: boundaries keep their crossings while labeling runs.)
    let vector = boundary_normal(mesh, skeleton, edge);
    let key = [vector.x.to_bits(), vector.y.to_bits(), vector.z.to_bits()];
    *cache
        .entry(key)
        .or_insert_with(|| loop_costs(mesh, skeleton, edge))
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

/// One way to close a cycle: the chain's steps, the slots that get a new boundary, and a port
/// per boundary of the resulting sequence.
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

/// Close the cycle of an unlabeled edge: relabel the edge's chain, all other labels fixed, with
/// the fewest new boundaries that admit a valid labeling, and among those the cheapest. A new
/// boundary runs through the middle of a patch beside one of the chain's boundaries (a parallel
/// collar where the patch has no middle); at most four always suffice. Returns the skeleton, the
/// added collars, and the new labels.
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
    // Every other label stays: the chain is relabeled with its new boundaries.
    let mut others = labels.clone();
    for &(e, _) in &chain {
        others.remove(e);
    }
    let used = constraints::used_ports(skeleton, &others)?;
    let costs = along(mesh, skeleton, &chain, cache);
    // The node before every step, and the first again.
    let mut nodes = vec![start];
    for &(e, _) in &chain {
        nodes.push(skeleton.edges[e].other(*nodes.last().unwrap()));
    }
    // What a new boundary in every slot would cost: its own plane decides, as for any edge.
    let ground = subdivide::Ground::build(mesh, skeleton)?;
    let mut slot_costs = Vec::new();
    for (i, &(e, _)) in chain.iter().enumerate() {
        for (k, node) in [nodes[i], nodes[i + 1]].into_iter().enumerate() {
            slot_costs.push(match subdivide::middle(mesh, skeleton, &ground, e, node) {
                // Along the path the normal points into the patch after the step, and out
                // of the patch before it.
                Some(middle) => alignment_costs(if k == 1 {
                    middle.alignment(mesh)
                } else {
                    -middle.alignment(mesh)
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
    for added in 0..=4 {
        let mut best: Option<(f64, Closing)> = None;
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
