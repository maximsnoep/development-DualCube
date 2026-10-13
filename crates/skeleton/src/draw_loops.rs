//! Incremental surface routing, accepting only the template's marked region topology.

use crate::{DualLoops, RealizedSkeleton, SkeletonResult};
use dualcube_dual::{Loop, LoopID, LoopSampler, LoopState, partial::PartialTopology};
use dualcube_types::prelude::*;

mod route;
mod template;

#[derive(Clone, Debug, Default)]
pub struct SearchStats {
    pub searches: usize,
    pub no_route: usize,
    pub expansion_limit: usize,
    pub duplicates: usize,
    pub invalid_arrangement: usize,
    pub topology_rejections: usize,
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
                "Could not draw loop {loop_id:?} after {} searches ({} no route, {} search limits, {} duplicates, {} invalid arrangements, {} topology rejections); retained the accepted partial drawing",
                stats.searches,
                stats.no_route,
                stats.expansion_limit,
                stats.duplicates,
                stats.invalid_arrangement,
                stats.topology_rejections
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
    let mut working = template.loops.clone();
    let mut drawn = HashSet::new();
    let mut result = Attempt {
        loops: DualLoops::with_key(),
        planned: working.len(),
        searches: 0,
        failure: None,
    };
    let geometry = crate::geometry::SurfaceGeometry::build(mesh, &realized.labeled.skeleton)?;
    for boundary in geometry.boundaries {
        let edge = boundary.edge;
        let id = template.boundaries[&edge];
        let mut l = Loop::from_surface(realized.labeled.labels[edge].direction, boundary.arcs);
        if realized.labeled.labels[edge].sign == Sign::Negative {
            l.reverse();
        }
        working[id] = l;
        drawn.insert(id);
    }
    result.loops = subset(&working, &drawn);
    let expected = PartialTopology::new(&template.mesh, &subset(&template.loops, &drawn))?;
    if !PartialTopology::new(mesh, &result.loops).is_ok_and(|actual| actual.equivalent(&expected)) {
        result.failure = Some(Failure::BoundaryTopology);
        return Ok(result);
    }
    let fields = dualcube_flow::Fields::new(mesh);
    let graphs = dualcube_flow::build_flow_graphs(
        mesh,
        &fields,
        dualcube_flow::GraphParams {
            max_deviation: std::f64::consts::PI,
            ..Default::default()
        },
    );
    let full = PartialTopology::new(&template.mesh, &template.loops)?;
    let mut history = Vec::<LoopID>::new();
    let mut alternatives = HashMap::<LoopID, Vec<Loop>>::new();
    let mut backtracks = 0;
    let mut failures = HashMap::<LoopID, usize>::new();
    while drawn.len() < working.len() {
        let id = working
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
        let before = subset(&template.loops, &drawn);
        let old = PartialTopology::new(&template.mesh, &before)?;
        let actual = PartialTopology::new(mesh, &result.loops)?;
        let mut mappings = old.region_correspondences(&actual, 16);
        if mappings.is_empty() {
            return Err("Lost partial topology correspondence".into());
        }
        let itinerary = PartialTopology::insertion_itinerary(
            &template.mesh,
            &before,
            &subset(&template.loops, &selected),
            id,
        )?;
        let crossings: Vec<_> = itinerary.iter().map(|&(partner, _)| partner).collect();
        let router = dualcube_dual::surface_router::SurfaceRouter::new(mesh, &result.loops)?;
        let mut gap_counts = vec![0usize; actual.regions];
        for &r in &router.regions {
            gap_counts[r] += 1;
        }
        mappings.sort_by_key(|mapping| {
            std::cmp::Reverse(
                itinerary
                    .iter()
                    .map(|&(_, r)| gap_counts[mapping[r]])
                    .min()
                    .unwrap_or(0),
            )
        });
        let mut routed = None;
        for mapping in mappings {
            let regions: Vec<_> = itinerary.iter().map(|&(_, r)| mapping[r]).collect();
            let candidate = route::find(
                mesh,
                &graphs,
                &working,
                &drawn,
                id,
                &crossings,
                &regions,
                &router,
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
            Ok(l) => {
                working[id] = l;
                drawn.insert(id);
                history.push(id);
                result.loops = subset(&working, &drawn);
            }
            Err(failure) => {
                let neighbors: Vec<_> = history
                    .iter()
                    .enumerate()
                    .rev()
                    .filter(|(_, previous)| crossings.contains(previous))
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
                    let previous = history[index];
                    alternatives
                        .entry(previous)
                        .or_default()
                        .push(working[previous].clone());
                    for undone in history.drain(index..) {
                        drawn.remove(&undone);
                        if undone != previous {
                            alternatives.remove(&undone);
                        }
                    }
                    result.loops = subset(&working, &drawn);
                    backtracks += 1;
                    tracing::info!(
                        "Loop drawing: reconsidering {previous:?} after {id:?} failed ({backtracks}/24 retries)"
                    );
                    continue;
                }
                result.failure = Some(failure);
                return Ok(result);
            }
        }
        tracing::info!("Loop drawing: {}/{} loops", drawn.len(), working.len());
    }
    if let Err(error) = dualcube_dual::Dual::from(std::sync::Arc::new(mesh.clone()), &result.loops)
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
