//! Incremental surface routing, accepting only the template's marked region topology.

use crate::{DualLoops, RealizedSkeleton, SkeletonResult};
use dualcube_dual::{
    Loop, LoopID, LoopSampler, LoopState, insertion::Plan, loop_map::LoopMap,
    partial::PartialTopology,
};
use dualcube_types::prelude::*;

mod lift;
mod route;
mod template;

#[derive(Clone, Debug, Default)]
pub struct SearchStats {
    pub searches: usize,
    pub no_route: usize,
    pub expansion_limit: usize,
    pub duplicates: usize,
    pub invalid_arrangement: usize,
    /// Drawings stopped because an arc could not be placed as routed; never expected.
    pub revisits: usize,
    pub topology_rejections: usize,
    /// Drawings stopped because the regions after an arc differed from the template.
    pub arc_mismatches: usize,
}

#[derive(Clone, Debug)]
pub enum Failure {
    BoundaryTopology,
    SearchExhausted { loop_id: LoopID, stats: SearchStats },
    InvalidDual(String),
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BoundaryTopology => {
                f.write_str("Lifted boundaries do not match the cuboid template topology")
            }
            Self::SearchExhausted { loop_id, stats } => write!(
                f,
                "Could not draw loop {loop_id:?} after {} searches ({} no route, {} search limits, {} duplicates, {} invalid arrangements, {} revisits, {} topology rejections, {} arc mismatches); retained the accepted partial drawing",
                stats.searches,
                stats.no_route,
                stats.expansion_limit,
                stats.duplicates,
                stats.invalid_arrangement,
                stats.revisits,
                stats.topology_rejections,
                stats.arc_mismatches
            ),
            Self::InvalidDual(error) => {
                write!(f, "Completed loop structure failed validation: {error}")
            }
        }
    }
}
impl std::error::Error for Failure {}

#[derive(Clone, Debug)]
pub struct Attempt {
    pub loops: DualLoops,
    pub planned: usize,
    pub searches: usize,
    pub failure: Option<Failure>,
}

pub fn run(mesh: &Mesh<INPUT>, skeleton: &RealizedSkeleton) -> SkeletonResult<DualLoops> {
    let attempt = attempt(mesh, skeleton)?;
    match attempt.failure {
        Some(error) => Err(error.into()),
        None => Ok(attempt.loops),
    }
}

pub fn attempt(mesh: &Mesh<INPUT>, realized: &RealizedSkeleton) -> SkeletonResult<Attempt> {
    let timer = std::time::Instant::now();
    realized.validate(mesh)?;
    let template = template::Template::build(realized)?;
    let mut drawn: HashSet<_> = template.boundaries.values().copied().collect();
    let mut result = Attempt {
        loops: DualLoops::with_key(),
        planned: template.loops.len(),
        searches: 0,
        failure: None,
    };
    // Topology lives in the loop map; curves are drawn from it only for output.
    let mut map = lift::boundaries(mesh, realized, &template)?;
    let output = |map: &LoopMap, drawn: &HashSet<LoopID>| -> SkeletonResult<DualLoops> {
        let mut loops = subset(&template.loops, drawn);
        for (id, arcs) in map.realize(mesh)? {
            loops[id] = Loop::from_surface(template.loops[id].direction, arcs);
        }
        Ok(loops)
    };
    let expected = PartialTopology::new(&template.mesh, &subset(&template.loops, &drawn))?;
    if !PartialTopology::from_map(&map).equivalent(&expected) {
        result.loops = output(&map, &drawn)?;
        result.failure = Some(Failure::BoundaryTopology);
        return Ok(result);
    }
    let flows = route::Flows::new(mesh, drawn.is_empty());
    let full = PartialTopology::new(&template.mesh, &template.loops)?;
    let mut history = Vec::<(LoopID, LoopMap)>::new();
    let mut alternatives = HashMap::<LoopID, Vec<HashSet<FaceID>>>::new();
    let mut backtracks = 0;
    let mut failures = HashMap::<LoopID, usize>::new();
    while drawn.len() < template.loops.len() {
        let id = template
            .loops
            .keys()
            .filter(|id| !drawn.contains(id))
            .max_by_key(|&id| {
                let neighbors: Vec<_> = full.along[&id]
                    .iter()
                    .map(|&c| other(full.crossings[c], id))
                    .filter(|id| drawn.contains(id))
                    .collect();
                (
                    neighbors.iter().collect::<HashSet<_>>().len(),
                    neighbors.len(),
                    std::cmp::Reverse(id),
                )
            })
            .unwrap();
        let mut selected = drawn.clone();
        selected.insert(id);
        let expected = PartialTopology::new(&template.mesh, &subset(&template.loops, &selected))?;
        let plan = Plan::new(
            &template.mesh,
            &subset(&template.loops, &drawn),
            &subset(&template.loops, &selected),
            id,
        )?;
        let actual = PartialTopology::from_map(&map);
        let correspondences = plan.before.correspondences(&actual, 16);
        if correspondences.is_empty() {
            return Err("Lost partial topology correspondence".into());
        }
        let crossings: Vec<_> = plan.itinerary.iter().map(|i| i.partner).collect();
        let (regions, _) = map.regions();
        let mut room = vec![0usize; actual.regions];
        for &r in &regions {
            room[r] += 1;
        }
        // Distinct correspondences can prescribe identical steps.
        let mut options: Vec<route::Target<'_>> = Vec::new();
        for correspondence in &correspondences {
            let steps = plan.steps(correspondence);
            if !options.iter().any(|o| o.steps == steps) {
                options.push(route::Target {
                    plan: &plan,
                    correspondence,
                    steps,
                });
            }
        }
        options.sort_by_key(|o| {
            std::cmp::Reverse(o.steps.iter().map(|s| room[s.region]).min().unwrap_or(0))
        });
        let mut routed = None;
        for target in &options {
            let candidate = route::find(
                mesh,
                &flows,
                &map,
                &actual,
                template.loops[id].direction,
                id,
                target,
                &expected,
                alternatives.get(&id).map_or(&[], Vec::as_slice),
                &mut result.searches,
            );
            let success = candidate.is_ok();
            routed = Some(candidate);
            if success {
                break;
            }
        }
        match routed.unwrap() {
            Ok(next) => {
                history.push((id, std::mem::replace(&mut map, next)));
                drawn.insert(id);
            }
            Err(failure) => {
                let neighbors: Vec<_> = history
                    .iter()
                    .enumerate()
                    .rev()
                    .filter(|(_, (previous, _))| crossings.contains(previous))
                    .map(|(i, _)| i)
                    .collect();
                let failed = failures.entry(id).or_default();
                let index = if neighbors.is_empty() {
                    None
                } else {
                    Some(neighbors[(*failed / 2) % neighbors.len()])
                };
                *failed += 1;
                if backtracks < 24
                    && let Some(index) = index
                {
                    let previous = history[index].0;
                    alternatives
                        .entry(previous)
                        .or_default()
                        .push(map.triangles(previous));
                    for (undone, before) in history.drain(index..).rev() {
                        drawn.remove(&undone);
                        if undone != previous {
                            alternatives.remove(&undone);
                        }
                        map = before;
                    }
                    backtracks += 1;
                    tracing::info!(
                        "Loop drawing: reconsidering {previous:?} after {id:?} failed ({backtracks}/24 retries)"
                    );
                    continue;
                }
                result.loops = output(&map, &drawn)?;
                result.failure = Some(failure);
                return Ok(result);
            }
        }
        tracing::info!(
            "Loop drawing: {}/{} loops",
            drawn.len(),
            template.loops.len()
        );
    }
    result.loops = output(&map, &drawn)?;
    // Drawing is for output only, but must keep the map's topology.
    let drawn_topology = PartialTopology::new(mesh, &result.loops);
    if !drawn_topology.is_ok_and(|t| t.equivalent(&full)) {
        result.failure = Some(Failure::InvalidDual(
            "Drawn curves lost the loop map's topology".into(),
        ));
    } else if let Err(error) =
        dualcube_dual::Dual::from(std::sync::Arc::new(mesh.clone()), &result.loops)
    {
        result.failure = Some(Failure::InvalidDual(error.to_string()));
    }
    tracing::info!(
        "Loop drawing complete: {} loops, {} searches, {backtracks} backtracks (elapsed {:.2}s)",
        result.loops.len(),
        result.searches,
        timer.elapsed().as_secs_f64()
    );
    Ok(result)
}

fn subset(loops: &DualLoops, selected: &HashSet<LoopID>) -> DualLoops {
    let mut result = loops.clone();
    result.retain(|id, _| selected.contains(&id));
    result
}
fn other(pair: [LoopID; 2], id: LoopID) -> LoopID {
    if pair[0] == id { pair[1] } else { pair[0] }
}

#[cfg(test)]
mod tests;
