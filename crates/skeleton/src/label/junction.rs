//! Split overfull junctions using boundary pairs and surface connectors.

use super::*;
use crate::surface::{Regions, edit};

/// Compare every pair and a small set of connector alternatives by the alignment of the
/// boundaries at the split.
/// Every candidate is a valid split; only the chosen one is validated.
pub(super) fn reduce(mesh: &Mesh<INPUT>, result: &mut Attempt) -> SkeletonResult<()> {
    // TODO: Search alternative split sequences instead of committing greedily.
    loop {
        let degree = result.skeleton.degrees();
        let Some((node, &count)) = degree.iter().find(|(_, d)| **d > 6) else {
            return Ok(());
        };
        let regions = Regions::build(mesh, &result.skeleton)?;
        let incident: Vec<_> = result.skeleton.incident(node).map(|(id, _)| id).collect();
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
                    let cost = score(mesh, &candidate, boundary);
                    if best.as_ref().is_some_and(|(old, length, ..)| {
                        cost > *old || (cost == *old && route.steps.len() >= *length)
                    }) {
                        continue;
                    }
                    best = Some((cost, route.steps.len(), candidate, added, boundary));
                }
            }
        }
        // A genus-zero patch is connected, so two of its boundaries always have a connector, and
        // the band around them and the connector always splits it validly. Cost alone decides.
        let Some((_, _, skeleton, added_node, boundary)) = best else {
            panic!("Impossible: degree-{count} patch {node:?} has no connector between boundaries");
        };
        skeleton.debug_validate(mesh);
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
    assign_ports(costs.len(), 1 << reserved, |i, p, value| {
        value + costs[i][p as usize]
    })
    .0
}

/// The least alignment cost of the two patches the connector joins, as in labeling: ports
/// assigned per patch, the connector's in common.
fn score(mesh: &Mesh<INPUT>, skeleton: &Skeleton, connector: SkeletonEdgeID) -> f64 {
    let positions = skeleton.surface.positions(mesh);
    let alignment = |e| quality::boundary_alignment(mesh, skeleton, &positions, e);
    let endpoints = skeleton.edges[connector].nodes;
    let mut costs = [Vec::new(), Vec::new()];
    for (side, node) in endpoints.into_iter().enumerate() {
        for (id, edge) in skeleton.incident(node) {
            if id != connector {
                let outward = if edge.nodes[0] == node {
                    alignment(id)
                } else {
                    alignment(id).reversed()
                };
                costs[side].push(outward.costs());
            }
        }
    }
    let own = alignment(connector).costs();
    (0..6)
        .map(|d| own[d] + star(&costs[0], d) + star(&costs[1], d ^ 1))
        .fold(f64::INFINITY, f64::min)
}
