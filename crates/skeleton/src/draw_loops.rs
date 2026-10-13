//! Incremental surface routing, accepting only the template's marked region topology.

use crate::{DualLoops, RealizedSkeleton, SkeletonResult};
use dualcube_dual::{Loop, LoopID, LoopSampler, LoopState, partial::PartialTopology};
use dualcube_types::prelude::*;

mod route;
mod template;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reentry {
    Edge(EdgeID),
    Face(FaceID),
}

#[derive(Clone, Debug, Default)]
pub struct SearchStats {
    pub searches: usize,
    pub no_route: usize,
    pub expansion_limit: usize,
    pub repeated_edge: usize,
    pub repeated_face: usize,
    pub duplicates: usize,
    pub invalid_arrangement: usize,
    pub topology_rejections: usize,
}

#[derive(Clone, Debug)]
pub enum Failure {
    UnsupportedReentry {
        loop_id: LoopID,
        at: Reentry,
    },
    BoundaryTopology,
    SearchExhausted {
        loop_id: LoopID,
        stats: SearchStats,
        reentry: Option<Reentry>,
    },
    InvalidDual(String),
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedReentry { loop_id, at } => write!(
                f,
                "Dual loop {loop_id:?} re-enters {at:?}; repeated edge/triangle visits are not supported"
            ),
            Self::BoundaryTopology => {
                f.write_str("Lifted boundaries do not match the cuboid template topology")
            }
            Self::SearchExhausted {
                loop_id,
                stats,
                reentry,
            } => write!(
                f,
                "Could not draw loop {loop_id:?} after {} searches ({} no route, {} search limits, {} edge re-entries, {} face re-entries, {} duplicates, {} invalid arrangements, {} topology rejections; last re-entry: {reentry:?}); retained the accepted partial drawing",
                stats.searches,
                stats.no_route,
                stats.expansion_limit,
                stats.repeated_edge,
                stats.repeated_face,
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
    for (edge, data) in &realized.labeled.skeleton.edges {
        let id = template.boundaries[&edge];
        let skeleton = &realized.labeled.skeleton;
        let mut edges = Vec::new();
        let mut offsets = Vec::new();
        for &crossing in &data.boundary.crossings {
            let exit = skeleton.surface.crossings[crossing].exit;
            let canonical = crate::surface::canonical(mesh, exit);
            let order = &skeleton.surface.edge_order[&canonical];
            let rank = order.iter().position(|&c| c == crossing).unwrap();
            let t = (rank + 1) as f64 / (order.len() + 1) as f64;
            let t = if exit == canonical { t } else { 1. - t };
            edges.extend([exit, mesh.twin(exit)]);
            offsets.extend([t, 1. - t]);
        }
        if realized.labeled.labels[edge].sign == Sign::Negative {
            edges.reverse();
            offsets.reverse();
        }
        let l = Loop {
            edges,
            offsets,
            direction: realized.labeled.labels[edge].direction,
        };
        if let Some(at) = reentry(mesh, &l) {
            result.failure = Some(Failure::UnsupportedReentry { loop_id: id, at });
            result.loops = subset(&working, &drawn);
            return Ok(result);
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
        let gaps = PartialTopology::gap_regions(mesh, &result.loops)?;
        let mut gap_counts = vec![0usize; actual.regions];
        for &r in gaps.values() {
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
                &gaps,
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
fn reentry(mesh: &Mesh<INPUT>, l: &Loop) -> Option<Reentry> {
    let mut edges = HashSet::new();
    let mut faces = HashSet::new();
    for (i, &edge) in l.edges.iter().enumerate() {
        if !edges.insert(edge) {
            return Some(Reentry::Edge(edge));
        }
        let next = l.edges[(i + 1) % l.edges.len()];
        if mesh.twin(edge) != next && !faces.insert(mesh.face(edge)) {
            return Some(Reentry::Face(mesh.face(edge)));
        }
    }
    None
}

#[cfg(test)]
mod tests;
