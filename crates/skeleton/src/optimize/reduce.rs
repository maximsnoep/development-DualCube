//! Per-path reduction: label all boundaries exactly, then on every chain keep the fewest
//! boundaries whose labels still describe the dropped ones.

use super::moves::{self, Chain};
use crate::label::{Port, constraints, exact, flow::Flow, path};
use crate::{LabeledSkeleton, Skeleton, SkeletonEdgeID, SkeletonResult};
use dualcube_types::prelude::*;
use slotmap::SecondaryMap;

/// How a boundary's labels are judged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LabelMeasure {
    /// Flow cost of the boundary drawn as a loop, over its best label's, minus 1. Measured on
    /// skeleton boundaries, a plane tilted 10° from the label gives about 0.5, 20° about 1, and
    /// 30° about 2. Where a boundary lies on faces facing an axis, that axis's field is undefined
    /// and costs little, so this can prefer labels far off the boundary's plane.
    Flow,
    /// Angle of the boundary's plane normal to the label beyond the best label's, per 20°.
    Plane,
}

/// Costs of [`run`], in excess (see [`LabelMeasure`]).
#[derive(Clone, Copy, Debug)]
pub struct ReduceParams {
    pub measure: LabelMeasure,
    /// Cost of every kept boundary.
    pub boundary: f64,
    /// Excess above which the cost grows steeply (mislabelling), and how steeply.
    pub threshold: f64,
    pub steepness: f64,
    /// Search nodes of the exact labeling; beyond, the best labeling found is used.
    pub budget: usize,
}

impl Default for ReduceParams {
    fn default() -> Self {
        Self {
            measure: LabelMeasure::Plane,
            boundary: 0.5,
            threshold: 1.,
            steepness: 20.,
            budget: 200_000,
        }
    }
}

impl ReduceParams {
    fn penalty(&self, excess: f64) -> f64 {
        excess + self.steepness * (excess - self.threshold).max(0.)
    }
}

/// Penalty of every label, from `nodes[0]` to `nodes[1]`.
fn penalties(
    mesh: &Mesh<INPUT>,
    skeleton: &Skeleton,
    flow: &Flow,
    params: &ReduceParams,
) -> SecondaryMap<SkeletonEdgeID, [f64; 6]> {
    skeleton
        .edges
        .keys()
        .map(|e| {
            let excess = match params.measure {
                LabelMeasure::Flow => {
                    let cost = flow.costs(mesh, skeleton, e);
                    let best = cost.iter().copied().fold(f64::INFINITY, f64::min);
                    cost.map(|c| c / best - 1.)
                }
                LabelMeasure::Plane => exact::plane_excess(mesh, skeleton, e),
            };
            (e, excess.map(|x| params.penalty(x)))
        })
        .collect()
}

/// The valid labeling with the least summed penalty (see [`ReduceParams::budget`]).
pub fn label(
    mesh: &Mesh<INPUT>,
    labeled: &LabeledSkeleton,
    flow: &Flow,
    params: &ReduceParams,
) -> SkeletonResult<LabeledSkeleton> {
    let mut labeled = labeled.clone();
    let penalties = penalties(mesh, &labeled.skeleton, flow, params);
    labeled.labels = exact::search(&labeled.skeleton, &labeled.labels, &penalties, params.budget);
    labeled.validate_labels()?;
    Ok(labeled)
}

/// [`label`], then reduce chain by chain (all other labels fixed) until no chain improves.
pub fn run(
    mesh: &Mesh<INPUT>,
    labeled: &LabeledSkeleton,
    flow: &Flow,
    params: &ReduceParams,
) -> SkeletonResult<LabeledSkeleton> {
    let mut labeled = label(mesh, labeled, flow, params)?;
    let penalties = penalties(mesh, &labeled.skeleton, flow, params);
    'improve: loop {
        for chain in moves::chains(&labeled.skeleton) {
            if let Some(next) = reduce(mesh, &labeled, &chain, &penalties, params) {
                labeled = next;
                continue 'improve;
            }
        }
        return Ok(labeled);
    }
}

/// The chain relabelled and reduced if that lowers its cost and stays valid.
fn reduce(
    mesh: &Mesh<INPUT>,
    labeled: &LabeledSkeleton,
    chain: &Chain,
    penalties: &SecondaryMap<SkeletonEdgeID, [f64; 6]>,
    params: &ReduceParams,
) -> Option<LabeledSkeleton> {
    let skeleton = &labeled.skeleton;
    let n = chain.edges.len();
    let forward = |i: usize| chain.forward(skeleton, i);
    let along: Vec<[f64; 6]> = (0..n)
        .map(|i| {
            let row = penalties[chain.edges[i]];
            std::array::from_fn(|p| row[if forward(i) { p } else { p ^ 1 }])
        })
        .collect();
    let current: f64 = (0..n)
        .map(|i| {
            let port = Port::from_label(labeled.labels[chain.edges[i]]);
            along[i][if forward(i) { port } else { port.opposite() } as usize] + params.boundary
        })
        .sum();
    let mut others = labeled.labels.clone();
    for &e in &chain.edges {
        others.remove(e);
    }
    let used = constraints::used_ports(skeleton, &others).ok()?;
    let (start, end) = (chain.nodes[0], chain.nodes[n]);
    let masks: [bool; 64] = std::array::from_fn(|mask| {
        constraints::feasible(skeleton, &others, Some(([start, end], mask as u8)))
    });
    let selection = if start != end {
        let limits = path::Constraints {
            start_ports: 63 & !used[start],
            end_ports: 63 & !used[end],
            masks,
        };
        path::select(&along, &limits, params.boundary)
    } else {
        // A ring leaves and enters one node through two different ports; through a patch
        // of degree two, not straight on.
        let degree = skeleton.degrees();
        Port::ALL
            .into_iter()
            .filter(|p| used[start] & p.bit() == 0)
            .filter_map(|p| {
                let straight = if degree[start] == 2 { p.opposite().bit() } else { 0 };
                let limits = path::Constraints {
                    start_ports: p.bit(),
                    end_ports: 63 & !used[start] & !p.bit() & !straight,
                    masks,
                };
                path::select(&along, &limits, params.boundary)
            })
            .min_by(|a, b| a.cost.total_cmp(&b.cost))
    }?;
    if selection.cost >= current - 1e-9 {
        return None;
    }
    let mut next = labeled.clone();
    for (&i, &port) in selection.kept.iter().zip(&selection.ports) {
        next.labels[chain.edges[i]] = if forward(i) { port } else { port.opposite() }.label();
    }
    // Patches merge into the chain's ends.
    for (i, &edge) in chain.edges.iter().enumerate() {
        if !selection.kept.contains(&i) {
            let nodes = next.skeleton.edges[edge].nodes;
            let keep = nodes
                .into_iter()
                .find(|n| *n == start || *n == end)
                .unwrap_or(nodes[0]);
            moves::dissolve(mesh, &mut next, edge, keep).ok()?;
        }
    }
    next.validate(mesh).ok()?;
    Some(next)
}
