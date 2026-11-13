//! Improve a placed skeleton by valid moves against restricted-label fidelity, then recompute
//! target placement.

mod cut;
pub mod fidelity;
pub(crate) mod moves;
pub(crate) mod redraw;
pub(crate) mod relax;
pub mod search;

pub use fidelity::{Fidelity, Measure, Tables};
pub use redraw::RedrawParams;
pub use relax::RelaxParams;
pub use search::{InitialPass, Objective, Operator, Params, Stats};

use crate::{LabeledSkeleton, RealizedSkeleton, SkeletonResult, place_cubes};
use dualcube_types::prelude::*;

/// Dissolve straight degree-two patches only. Searching on restricted-label fidelity alone
/// lowers the reconstructed quality; the search needs an objective such as the primalized
/// quality (`run_with`).
pub fn run(mesh: &Mesh<INPUT>, placed: &RealizedSkeleton) -> SkeletonResult<RealizedSkeleton> {
    placed.validate(mesh)?;
    place_cubes::run(mesh, &dissolve_straight(mesh, &placed.labeled)?)
}

/// Dissolve straight degree-two patches, then search; the result is placed again.
pub fn run_with(
    mesh: &Mesh<INPUT>,
    placed: &RealizedSkeleton,
    params: &Params,
    objective: Option<Objective<'_>>,
) -> SkeletonResult<(RealizedSkeleton, Stats)> {
    placed.validate(mesh)?;
    let start = dissolve_straight(mesh, &placed.labeled)?;
    let (labeled, stats) = search::search(mesh, &start, params, objective)?;
    labeled.validate(mesh)?;
    let result = place_cubes::run(mesh, &labeled)?;
    tracing::info!(
        "Optimization: {} → {} patches, alignment {:.4} → {:.4} (naive {:.4}), {} steps, {:.1}s",
        stats.nodes[0],
        stats.nodes[1],
        stats.initial.alignment,
        stats.result.alignment,
        stats.naive.alignment,
        stats.steps,
        stats.seconds
    );
    Ok((result, stats))
}

/// Absorb straight annular patches into a neighbour, deleting their shared boundary.
pub fn dissolve_straight(
    mesh: &Mesh<INPUT>,
    labeled: &LabeledSkeleton,
) -> SkeletonResult<LabeledSkeleton> {
    let mut labeled = labeled.clone();
    loop {
        let degree = moves::degrees(&labeled.skeleton);
        let candidate = labeled.skeleton.nodes.keys().find_map(|node| {
            if degree[node] != 2 {
                return None;
            }
            let edges: Vec<_> = labeled
                .skeleton
                .edges
                .iter()
                .filter(|(_, e)| e.nodes.contains(&node))
                .map(|(id, _)| id)
                .collect();
            // Valid local ports on one axis must be opposite, irrespective of stored edge orientation.
            (labeled.labels[edges[0]].direction == labeled.labels[edges[1]].direction).then(|| {
                let removed = &labeled.skeleton.edges[edges[0]];
                let neighbor = removed.nodes[usize::from(removed.nodes[0] == node)];
                (edges[0], neighbor)
            })
        });
        let Some((edge, keep)) = candidate else {
            break;
        };
        // TODO: Choose which boundary to retain using surface/label costs.
        moves::dissolve(mesh, &mut labeled, edge, keep)?;
    }
    labeled.validate_labels()?;
    Ok(labeled)
}

#[cfg(test)]
mod tests;
