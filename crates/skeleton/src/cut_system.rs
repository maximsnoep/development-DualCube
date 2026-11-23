//! A minimal cut system: g patch boundaries whose removal from the skeleton leaves a spanning tree. Every patch is
//! contractible in the volume (see `certify`), so each loop bounds a disk, and cutting along the g disks leaves a ball:
//! one meridian per handle. For figures only; the pipeline does not use it.
//!
//! Spread out: loops are added greedily, each farthest from those chosen so far, among the ones that keep the rest of
//! the skeleton connected (so the greedy choice always completes). Every loop is tried as the first; the choice whose
//! closest pair is farthest apart is kept, then the shortest.

use crate::{Skeleton, SkeletonEdgeID};
use dualcube_types::prelude::*;

// Points sampled per loop to measure the distance between loops.
const SAMPLES: usize = 16;

/// The loops (as skeleton edges) of a spread-out minimal cut system of a connected skeleton.
pub fn run(mesh: &Mesh<INPUT>, skeleton: &Skeleton) -> Vec<SkeletonEdgeID> {
    let edges: Vec<_> = skeleton.edges.keys().collect();
    let index: HashMap<_, _> = skeleton
        .nodes
        .keys()
        .enumerate()
        .map(|(i, n)| (n, i))
        .collect();
    let ends: Vec<_> = edges
        .iter()
        .map(|&e| skeleton.edges[e].nodes.map(|n| index[&n]))
        .collect();
    let positions = skeleton.surface.positions(mesh);
    let samples: Vec<Vec<Vector3D>> = edges
        .iter()
        .map(|&e| {
            let crossings = &skeleton.edges[e].boundary.crossings;
            let step = crossings.len().div_ceil(SAMPLES).max(1);
            crossings
                .iter()
                .step_by(step)
                .map(|c| positions[c])
                .collect()
        })
        .collect();
    let distance: Vec<Vec<f64>> = samples
        .iter()
        .map(|a| {
            samples
                .iter()
                .map(|b| {
                    a.iter()
                        .flat_map(|p| b.iter().map(move |q| (p - q).norm()))
                        .fold(f64::INFINITY, f64::min)
                })
                .collect()
        })
        .collect();
    let length: Vec<_> = edges
        .iter()
        .map(|&e| skeleton.edges[e].boundary.length(mesh, &skeleton.surface))
        .collect();
    spread(index.len(), &ends, &distance, &length)
        .into_iter()
        .map(|i| edges[i])
        .collect()
}

/// The cut system (as edge indices) of a connected multigraph with these edge ends, spread out by `distance` between
/// edges, then short by `length`.
pub(crate) fn spread(
    nodes: usize,
    ends: &[[usize; 2]],
    distance: &[Vec<f64>],
    length: &[f64],
) -> Vec<usize> {
    let genus = (ends.len() + 1).saturating_sub(nodes);
    if genus == 0 {
        return vec![];
    }
    // The closest pair (farther is better), then the total length (shorter is better).
    let score = |chosen: &[usize]| {
        let closest = chosen
            .iter()
            .flat_map(|&a| {
                chosen
                    .iter()
                    .filter(move |&&b| b != a)
                    .map(move |&b| distance[a][b])
            })
            .fold(f64::INFINITY, f64::min);
        (closest, -chosen.iter().map(|&e| length[e]).sum::<f64>())
    };
    let mut best: Option<(Vec<usize>, (f64, f64))> = None;
    let bridges_at_start = bridges(nodes, ends, &vec![false; ends.len()]);
    for start in (0..ends.len()).filter(|&e| !bridges_at_start[e]) {
        let mut removed = vec![false; ends.len()];
        removed[start] = true;
        let mut chosen = vec![start];
        while chosen.len() < genus {
            let bridge = bridges(nodes, ends, &removed);
            let next = (0..ends.len())
                .filter(|&e| !removed[e] && !bridge[e])
                .map(|e| {
                    let gap = chosen
                        .iter()
                        .map(|&c| distance[e][c])
                        .fold(f64::INFINITY, f64::min);
                    (e, gap)
                })
                .max_by(|(a, gap_a), (b, gap_b)| {
                    gap_a
                        .total_cmp(gap_b)
                        .then(length[*b].total_cmp(&length[*a]))
                        .then(b.cmp(a))
                });
            let Some((e, _)) = next else {
                break;
            };
            removed[e] = true;
            chosen.push(e);
        }
        if chosen.len() < genus {
            continue;
        }
        let candidate = score(&chosen);
        let better = best.as_ref().is_none_or(|(_, s)| {
            candidate
                .0
                .total_cmp(&s.0)
                .then(candidate.1.total_cmp(&s.1))
                == std::cmp::Ordering::Greater
        });
        if better {
            best = Some((chosen, candidate));
        }
    }
    let mut chosen = best.map(|(chosen, _)| chosen).unwrap_or_default();
    chosen.sort_unstable();
    chosen
}

/// Which edges are bridges of the multigraph without the `removed` ones (removed edges are not).
fn bridges(nodes: usize, ends: &[[usize; 2]], removed: &[bool]) -> Vec<bool> {
    let mut adjacent = vec![vec![]; nodes];
    for (e, &[a, b]) in ends.iter().enumerate() {
        if !removed[e] && a != b {
            adjacent[a].push((b, e));
            adjacent[b].push((a, e));
        }
    }
    let mut bridge = vec![false; ends.len()];
    let mut order = vec![usize::MAX; nodes];
    let mut low = vec![0; nodes];
    let mut time = 0;
    for root in 0..nodes {
        if order[root] != usize::MAX {
            continue;
        }
        order[root] = time;
        low[root] = time;
        time += 1;
        // Depth-first, iteratively: (node, the edge it was entered by, the next neighbor to visit).
        let mut stack = vec![(root, usize::MAX, 0)];
        while let Some(&mut (node, entered, ref mut next)) = stack.last_mut() {
            if let Some(&(other, edge)) = adjacent[node].get(*next) {
                *next += 1;
                if edge == entered {
                    continue;
                }
                if order[other] == usize::MAX {
                    order[other] = time;
                    low[other] = time;
                    time += 1;
                    stack.push((other, edge, 0));
                } else {
                    low[node] = low[node].min(order[other]);
                }
            } else {
                stack.pop();
                if let Some(&(parent, _, _)) = stack.last() {
                    low[parent] = low[parent].min(low[node]);
                    if low[node] > order[parent] {
                        bridge[entered] = true;
                    }
                }
            }
        }
    }
    bridge
}

#[cfg(test)]
mod tests;
