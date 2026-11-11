//! Angle-based tree labeling followed by surface-preserving cycle repair.

pub(crate) mod constraints;
mod junction;
pub mod path;
mod subdivide;
mod tree;

use crate::{EdgeLabel, LabeledSkeleton, Skeleton, SkeletonEdgeID, SkeletonNodeID, SkeletonResult};
use dualcube_types::prelude::{Direction, INPUT, Mesh, Sign, Vector3D};
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
    pub fn bit(self) -> u8 {
        1 << self as u8
    }
    pub fn label(self) -> EdgeLabel {
        EdgeLabel {
            direction: [Direction::X, Direction::Y, Direction::Z][self as usize / 2],
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

#[derive(Clone, Debug)]
pub struct JunctionSplit {
    pub original_node: SkeletonNodeID,
    pub added_node: SkeletonNodeID,
    pub boundary: SkeletonEdgeID,
}

#[derive(Clone, Debug)]
pub struct Repair {
    pub original_edge: SkeletonEdgeID,
    pub path: Vec<SkeletonEdgeID>,
}

/// Retained junction splits, tree solution, and cycle repairs.
#[derive(Clone, Debug)]
pub struct Attempt {
    pub skeleton: Skeleton,
    pub labels: SecondaryMap<SkeletonEdgeID, EdgeLabel>,
    pub tree_edges: Vec<SkeletonEdgeID>,
    pub tree_labels: SecondaryMap<SkeletonEdgeID, EdgeLabel>,
    pub repairs: Vec<Repair>,
    pub junction_splits: Vec<JunctionSplit>,
}

impl Attempt {
    pub fn labeled(&self) -> LabeledSkeleton {
        LabeledSkeleton {
            skeleton: self.skeleton.clone(),
            labels: self.labels.clone(),
        }
    }
}

/// Label without placing cubes. Use `attempt` to retain the intermediate results.
pub fn run(mesh: &Mesh<INPUT>, skeleton: &Skeleton) -> SkeletonResult<LabeledSkeleton> {
    Ok(attempt(mesh, skeleton)?.labeled())
}

pub fn attempt(mesh: &Mesh<INPUT>, skeleton: &Skeleton) -> SkeletonResult<Attempt> {
    skeleton.validate(mesh)?;
    let start = std::time::Instant::now();
    let mut result = Attempt {
        skeleton: skeleton.clone(),
        labels: SecondaryMap::new(),
        tree_edges: Vec::new(),
        tree_labels: SecondaryMap::new(),
        repairs: Vec::new(),
        junction_splits: Vec::new(),
    };
    junction::reduce(mesh, &mut result)?;
    let mut cost_cache = HashMap::new();
    let table = result
        .skeleton
        .edges
        .keys()
        .map(|e| (e, cached_cost(&result.skeleton, e, &mut cost_cache)))
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
    result.labels = labels;
    for edge in remaining {
        let (skeleton, path, solution) = repair(
            mesh,
            &result.skeleton,
            &result.labels,
            edge,
            &mut cost_cache,
        )?;
        result.skeleton = skeleton;
        for (&e, p) in path.iter().zip(solution.ports) {
            result.labels.insert(e, p.label());
        }
        result.repairs.push(Repair {
            original_edge: edge,
            path,
        });
    }
    // TODO: Refine this initial solution with joint path/junction optimization and simplification.
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

fn cached_cost(
    skeleton: &Skeleton,
    edge: SkeletonEdgeID,
    cache: &mut HashMap<[u64; 3], [f64; 6]>,
) -> [f64; 6] {
    let [a, b] = skeleton.edges[edge].nodes;
    let vector = skeleton.nodes[b].position - skeleton.nodes[a].position;
    // TODO: Include surface candidate identity and labels when caching richer patch costs.
    let key = [vector.x.to_bits(), vector.y.to_bits(), vector.z.to_bits()];
    *cache.entry(key).or_insert_with(|| angle_costs(vector))
}

fn repair(
    mesh: &Mesh<INPUT>,
    skeleton: &Skeleton,
    labels: &SecondaryMap<SkeletonEdgeID, EdgeLabel>,
    edge: SkeletonEdgeID,
    cache: &mut HashMap<[u64; 3], [f64; 6]>,
) -> SkeletonResult<(Skeleton, Vec<SkeletonEdgeID>, path::Solution)> {
    let endpoints = skeleton.edges[edge].nodes;
    let used = constraints::used_ports(skeleton, labels)?;
    let mut limits = path::Constraints {
        start_ports: 63 & !used[endpoints[0]],
        end_ports: 63 & !used[endpoints[1]],
        ..Default::default()
    };
    for mask in 0..64 {
        limits.masks[mask] = constraints::feasible(skeleton, labels, Some((endpoints, mask as u8)));
    }
    let mut candidates = vec![(skeleton.clone(), vec![edge])];
    // TODO: Allow more liberal subdivision tradeoffs instead of minimizing count first.
    for subdivisions in 0..=4 {
        let mut best = None;
        for (candidate, edges) in &candidates {
            let costs: Vec<_> = edges
                .iter()
                .map(|&e| cached_cost(candidate, e, cache))
                .collect();
            if let Some(solution) = path::optimize(&costs, &limits)
                && best.as_ref().is_none_or(
                    |(_, _, old): &(Skeleton, Vec<SkeletonEdgeID>, path::Solution)| {
                        solution.cost < old.cost
                    },
                )
            {
                best = Some((candidate.clone(), edges.clone(), solution));
            }
        }
        if let Some(best) = best {
            return Ok(best);
        }
        if subdivisions == 4 {
            break;
        }
        // TODO: Search loop sequences jointly with the path DP beyond endpoint-only parallel collars.
        let mut next = Vec::new();
        for (candidate, edges) in candidates {
            for side in 0..2 {
                let index = if side == 0 { 0 } else { edges.len() - 1 };
                let (split, added) = subdivide::collar(mesh, &candidate, edges[index], side)?;
                let mut path = edges.clone();
                path.insert(index + 1, added);
                next.push((split, path));
            }
        }
        candidates = next;
    }
    // Any unlabeled edge closes with at most four subdivisions (theory: labeling cycle closing).
    panic!("Impossible: cycle edge {edge:?} has no valid labeling with 0–4 subdivisions")
}

#[cfg(test)]
mod tests;
