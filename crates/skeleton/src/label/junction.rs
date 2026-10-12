//! Split overfull junctions using boundary pairs and surface connectors.

use super::*;
use crate::surface::{Regions, edit};

/// Compare every pair and a small set of connector alternatives using local port costs.
pub(super) fn reduce(mesh: &Mesh<INPUT>, result: &mut Attempt) -> SkeletonResult<()> {
    // TODO: Search alternative split sequences instead of committing greedily.
    loop {
        let mut degree = SecondaryMap::new();
        for node in result.skeleton.nodes.keys() {
            degree.insert(node, 0usize);
        }
        for edge in result.skeleton.edges.values() {
            for node in edge.nodes {
                degree[node] += 1;
            }
        }
        let Some((node, &count)) = degree.iter().find(|(_, d)| **d > 6) else {
            return Ok(());
        };
        let regions = Regions::build(mesh, &result.skeleton)?;
        let incident: Vec<_> = result
            .skeleton
            .edges
            .iter()
            .filter(|(_, e)| e.nodes.contains(&node))
            .map(|(id, _)| id)
            .collect();
        let mut best: Option<(f64, usize, Skeleton, SkeletonNodeID, SkeletonEdgeID)> = None;
        for a in 0..incident.len() {
            for b in a + 1..incident.len() {
                let pair = [incident[a], incident[b]];
                let Some(first) = edit::connector(&regions, &result.skeleton, node, pair, None)
                else {
                    continue;
                };
                let mut routes = vec![first.clone()];
                // TODO: Guide routing directly by label costs and search larger boundary subsets.
                for fraction in [1, 2] {
                    if !first.steps.is_empty() {
                        let blocked = first.steps[first.steps.len() * fraction / 3].1;
                        if let Some(route) =
                            edit::connector(&regions, &result.skeleton, node, pair, Some(blocked))
                        {
                            if !routes.iter().any(|old| {
                                old.start == route.start
                                    && old.end == route.end
                                    && old.steps == route.steps
                            }) {
                                routes.push(route);
                            }
                        }
                    }
                }
                for route in routes {
                    let (candidate, added, boundary) =
                        edit::band(mesh, &result.skeleton, &regions, node, pair, &route)?;
                    let cost = score(&candidate, boundary);
                    if best.as_ref().is_some_and(|(old, length, ..)| {
                        cost > *old || (cost == *old && route.steps.len() >= *length)
                    }) {
                        continue;
                    }
                    if candidate.validate(mesh).is_ok() {
                        best = Some((cost, route.steps.len(), candidate, added, boundary));
                    }
                }
            }
        }
        let Some((_, _, skeleton, added_node, boundary)) = best else {
            result.failure = Some(Failure::JunctionSplitUnavailable {
                node,
                degree: count,
            });
            return Ok(());
        };
        result.skeleton = skeleton;
        result.junction_splits.push(JunctionSplit {
            original_node: node,
            added_node,
            boundary,
        });
    }
}

/// Reserve the connecting port, then optimally assign the remaining local ports.
fn star(costs: &[[f64; 6]], reserved: usize) -> f64 {
    if costs.len() > 5 {
        // Still-overfull junctions use a lower bound until the next split.
        return costs
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .filter(|(d, _)| *d != reserved)
                    .map(|(_, &c)| c)
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
            for d in 0..6 {
                if mask & (1 << d) == 0 {
                    let target = mask | (1 << d);
                    next[target] = next[target].min(dp[mask] + row[d]);
                }
            }
        }
        dp = next;
    }
    dp.into_iter().fold(f64::INFINITY, f64::min)
}

fn score(skeleton: &Skeleton, connector: SkeletonEdgeID) -> f64 {
    let endpoints = skeleton.edges[connector].nodes;
    let mut costs = [Vec::new(), Vec::new()];
    for (side, node) in endpoints.into_iter().enumerate() {
        for (id, edge) in &skeleton.edges {
            if id == connector || !edge.nodes.contains(&node) {
                continue;
            }
            let other = edge.nodes[usize::from(edge.nodes[0] == node)];
            costs[side].push(angle_costs(
                skeleton.nodes[other].position - skeleton.nodes[node].position,
            ));
        }
    }
    let connector =
        angle_costs(skeleton.nodes[endpoints[1]].position - skeleton.nodes[endpoints[0]].position);
    (0..6)
        .map(|d| connector[d] + star(&costs[0], d) + star(&costs[1], d ^ 1))
        .fold(f64::INFINITY, f64::min)
}
