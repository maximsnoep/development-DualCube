//! Split high-degree patches along validated surface cuts before choosing the MST.

use super::*;
use crate::{PatchBoundary, SkeletonEdge, SkeletonNode};
use dualcube_types::prelude::{HasNeighbors, HasPosition, VertID};
use std::collections::{HashSet, VecDeque};

/// Split until every junction fits six ports. Failed searches retain earlier splits.
pub(super) fn reduce(mesh: &Mesh<INPUT>, result: &mut Attempt) -> SkeletonResult<()> {
    // TODO: Search alternative split sequences and labelings instead of committing greedily.
    loop {
        let mut degree = SecondaryMap::new();
        for n in result.skeleton.nodes.keys() {
            degree.insert(n, 0usize);
        }
        for edge in result.skeleton.edges.values() {
            for n in edge.nodes {
                degree[n] += 1;
            }
        }
        let Some((node, &count)) = degree.iter().find(|(_, count)| **count > 6) else {
            return Ok(());
        };
        let Some((skeleton, added, boundary)) = split(mesh, &result.skeleton, node)? else {
            result.failure = Some(Failure::JunctionSplitUnavailable {
                node,
                degree: count,
            });
            return Ok(());
        };
        result.skeleton = skeleton;
        result.junction_splits.push(JunctionSplit {
            original_node: node,
            added_node: added,
            boundary,
        });
    }
}

struct Candidate {
    side: Vec<bool>,
    centers: [Vector3D; 2],
    excess: usize,
    cost: f64,
}

fn distances(graph: &[Vec<usize>], seed: usize) -> Vec<usize> {
    let mut distance = vec![usize::MAX; graph.len()];
    distance[seed] = 0;
    let mut queue = VecDeque::from([seed]);
    while let Some(n) = queue.pop_front() {
        for &neighbor in &graph[n] {
            if distance[neighbor] == usize::MAX {
                distance[neighbor] = distance[n] + 1;
                queue.push_back(neighbor);
            }
        }
    }
    distance
}

/// Port assignment around one half, with the connecting port reserved.
/// Overfull halves use a relaxed angle bound until a later split reduces their degree.
fn star_cost(costs: &[[f64; 6]], reserved: usize) -> f64 {
    if costs.len() > 5 {
        return costs
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .filter(|(d, _)| *d != reserved)
                    .map(|(_, &cost)| cost)
                    .fold(f64::INFINITY, f64::min)
            })
            .sum();
    }
    let mut dp = [f64::INFINITY; 64];
    dp[1 << reserved] = 0.;
    for row in costs {
        let mut next = [f64::INFINITY; 64];
        for mask in 0..64 {
            if !dp[mask].is_finite() {
                continue;
            }
            for (d, &cost) in row.iter().enumerate() {
                if mask & (1 << d) == 0 {
                    let target = mask | (1 << d);
                    next[target] = next[target].min(dp[mask] + cost);
                }
            }
        }
        dp = next;
    }
    dp.into_iter().fold(f64::INFINITY, f64::min)
}

fn score(
    skeleton: &Skeleton,
    node: SkeletonNodeID,
    incident: &[SkeletonEdgeID],
    sides: &[bool],
    centers: [Vector3D; 2],
) -> f64 {
    let mut costs = [Vec::new(), Vec::new()];
    for (&e, &side) in incident.iter().zip(sides) {
        let edge = &skeleton.edges[e];
        let other = if edge.nodes[0] == node {
            edge.nodes[1]
        } else {
            edge.nodes[0]
        };
        costs[usize::from(side)].push(angle_costs(
            skeleton.nodes[other].position - centers[usize::from(side)],
        ));
    }
    let connector = angle_costs(centers[1] - centers[0]);
    (0..6)
        .map(|d| connector[d] + star_cost(&costs[0], d) + star_cost(&costs[1], d ^ 1))
        .fold(f64::INFINITY, f64::min)
}

fn split(
    mesh: &Mesh<INPUT>,
    skeleton: &Skeleton,
    node: SkeletonNodeID,
) -> SkeletonResult<Option<(Skeleton, SkeletonNodeID, SkeletonEdgeID)>> {
    let vertices = &skeleton.nodes[node].patch_vertices;
    let index: HashMap<VertID, usize> = vertices.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let incident: Vec<_> = skeleton
        .edges
        .iter()
        .filter(|(_, edge)| edge.nodes.contains(&node))
        .map(|(e, _)| e)
        .collect();
    let mut union = constraints::Union::new(vertices.len());
    let mut boundary_vertices = Vec::new();
    // A boundary and its adjacent triangles must remain on one side of the cut.
    for &edge in &incident {
        let adjacent: Vec<_> = skeleton.edges[edge]
            .boundary
            .edge_midpoints
            .iter()
            .flat_map(|&e| [mesh.root(e), mesh.toor(e)])
            .filter_map(|v| index.get(&v).copied())
            .collect();
        let first = adjacent[0];
        for &i in &adjacent {
            union.join(first, i);
        }
        boundary_vertices.push(first);
    }
    let mut roots = HashMap::new();
    let mut blocks = Vec::new();
    for i in 0..vertices.len() {
        let next = roots.len();
        blocks.push(*roots.entry(union.root(i)).or_insert(next));
    }
    let terminals: Vec<_> = boundary_vertices.iter().map(|&i| blocks[i]).collect();
    let mut graph = vec![Vec::new(); roots.len()];
    for (i, &v) in vertices.iter().enumerate() {
        for neighbor in mesh.neighbors(v) {
            if let Some(&j) = index.get(&neighbor)
                && blocks[i] != blocks[j]
            {
                graph[blocks[i]].push(blocks[j]);
            }
        }
    }
    for neighbors in &mut graph {
        neighbors.sort_unstable();
        neighbors.dedup();
    }
    let mut seeds = terminals.clone();
    seeds.sort_unstable();
    seeds.dedup();
    let distances: Vec<_> = seeds.iter().map(|&seed| distances(&graph, seed)).collect();
    let mut seen = HashSet::new();
    let mut candidates = Vec::new();
    // TODO: Guide surface-loop sampling with label costs beyond graph-distance bisectors;
    // retain competing loops so their best labelings can determine which cut to prefer.
    // Sweep bisectors between boundary pairs; boundary collars are indivisible blocks.
    for a in 0..seeds.len() {
        for b in a + 1..seeds.len() {
            let delta: Vec<_> = distances[a]
                .iter()
                .zip(&distances[b])
                .map(|(&x, &y)| x as isize - y as isize)
                .collect();
            let mut thresholds = delta.clone();
            thresholds.sort_unstable();
            thresholds.dedup();
            for threshold in thresholds {
                let mut side: Vec<_> = delta.iter().map(|&d| d <= threshold).collect();
                if side[terminals[0]] {
                    side.iter_mut().for_each(|v| *v = !*v);
                }
                let sides: Vec<_> = terminals.iter().map(|&t| side[t]).collect();
                let count = sides.iter().filter(|&&s| s).count();
                if count < 2 || incident.len() - count < 2 || !seen.insert(side.clone()) {
                    continue;
                }
                let mut centers = [Vector3D::zeros(); 2];
                let mut counts = [0usize; 2];
                for (i, &v) in vertices.iter().enumerate() {
                    let s = usize::from(side[blocks[i]]);
                    centers[s] += mesh.position(v);
                    counts[s] += 1;
                }
                for s in 0..2 {
                    centers[s] /= counts[s] as f64;
                }
                let excess =
                    (count + 1).saturating_sub(6) + (incident.len() - count + 1).saturating_sub(6);
                let cost = score(skeleton, node, &incident, &sides, centers);
                candidates.push(Candidate {
                    side,
                    centers,
                    excess,
                    cost,
                });
            }
        }
    }
    // First reduce excess degree, then minimize local signed-axis assignment cost.
    candidates.sort_by(|a, b| {
        a.excess
            .cmp(&b.excess)
            .then_with(|| a.cost.total_cmp(&b.cost))
    });
    for candidate in candidates {
        let mut split = skeleton.clone();
        let (moved, kept): (Vec<_>, Vec<_>) = vertices
            .iter()
            .enumerate()
            .partition(|(i, _)| candidate.side[blocks[*i]]);
        split.nodes[node].patch_vertices = kept.into_iter().map(|(_, &v)| v).collect();
        split.nodes[node].position = candidate.centers[0];
        let added = split.nodes.insert(SkeletonNode {
            position: candidate.centers[1],
            patch_vertices: moved.into_iter().map(|(_, &v)| v).collect(),
        });
        for (&edge, &terminal) in incident.iter().zip(&terminals) {
            if candidate.side[terminal] {
                for endpoint in &mut split.edges[edge].nodes {
                    if *endpoint == node {
                        *endpoint = added;
                    }
                }
            }
        }
        let boundary = split.edges.insert(SkeletonEdge {
            nodes: [node, added],
            boundary: PatchBoundary {
                edge_midpoints: Vec::new(),
            },
        });
        if crate::validate::assign_boundaries(&mut split, mesh).is_ok()
            && split.validate(mesh).is_ok()
        {
            return Ok(Some((split, added, boundary)));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests;
