//! Angle-based tree labeling followed by surface-preserving cycle repair.

pub(crate) mod constraints;
mod junction;
pub mod path;
pub(crate) mod subdivide;
mod tree;

use crate::{EdgeLabel, LabeledSkeleton, Skeleton, SkeletonEdgeID, SkeletonNodeID, SkeletonResult};
use dualcube_types::prelude::{Direction, HasPosition, INPUT, Mesh, Sign, Vector3D};
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

/// Direction of an edge: the normal of its boundary's plane, scaled by the enclosed area, from
/// its first patch to its second (the boundary runs counterclockwise around the first, seen
/// from outside). The loop is drawn with the label's axis, so its own plane decides; the
/// direction between patch centres can differ from it by any angle.
fn boundary_normal(mesh: &Mesh<INPUT>, skeleton: &Skeleton, edge: SkeletonEdgeID) -> Vector3D {
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
fn chain(
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

/// Close the cycle of an unlabeled edge: relabel its whole chain, tree edges included, with the
/// fewest parallel collars that admit a valid labeling, and among those the cheapest. Returns
/// the skeleton, the added collars, and the chain's labels.
fn repair(
    mesh: &Mesh<INPUT>,
    skeleton: &Skeleton,
    labels: &SecondaryMap<SkeletonEdgeID, EdgeLabel>,
    edge: SkeletonEdgeID,
    cache: &mut HashMap<[u64; 3], [f64; 6]>,
) -> SkeletonResult<(Skeleton, Vec<SkeletonEdgeID>, Vec<(SkeletonEdgeID, EdgeLabel)>)> {
    let (steps, [start, end]) = chain(skeleton, edge);
    let solve = chain_solver(skeleton, labels, &steps, [start, end])?;
    // Costs along the chain: a step against its edge swaps the signs.
    let costs = along(mesh, skeleton, &steps, cache);
    // A collar copies a boundary, so it costs what its original costs: choose which steps to
    // copy on the costs alone, then build only the chosen skeleton.
    for collars in 0..=4 {
        let mut best: Option<(f64, Vec<usize>)> = None;
        for copied in copies(steps.len(), collars) {
            let mut sequence = Vec::new();
            for (i, &cost) in costs.iter().enumerate() {
                let count = 1 + copied.iter().filter(|&&c| c == i).count();
                sequence.extend(std::iter::repeat_n(cost, count));
            }
            if let Some(solution) = solve(&sequence)
                && best.as_ref().is_none_or(|(old, _)| solution.cost < *old)
            {
                best = Some((solution.cost, copied));
            }
        }
        let Some((_, copied)) = best else {
            continue;
        };
        let mut split = skeleton.clone();
        let mut added = Vec::new();
        let mut path = Vec::new();
        for (i, &(e, forward)) in steps.iter().enumerate() {
            // Along the edge: the edge, then its collars from the last added to the first.
            let mut run = vec![e];
            for _ in copied.iter().filter(|&&c| c == i) {
                let (next, collar) = subdivide::collar(mesh, &split, e, 1)?;
                split = next;
                run.insert(1, collar);
                added.push(collar);
            }
            if !forward {
                run.reverse();
            }
            path.extend(run.into_iter().map(|e| (e, forward)));
        }
        let solution = solve(&along(mesh, &split, &path, cache))
            .expect("Impossible: a collar costs what its original costs");
        let labels = path
            .iter()
            .zip(solution.ports)
            .map(|(&(e, forward), port)| {
                (e, if forward { port } else { port.opposite() }.label())
            })
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
                    positions.iter().filter(move |&&p| p >= least).map(move |&p| {
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
    grow(&all).unwrap_or_else(|| grow(&[0, steps - 1]).expect("two positions stay within the limit"))
}

#[cfg(test)]
mod tests;
