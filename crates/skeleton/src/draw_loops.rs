//! Draw the template's loops on the input surface, one arc at a time.
//!
//! The patch boundaries are lifted first. Every further loop is inserted so that the drawing
//! stays equivalent to the template restricted to the loops drawn: crossings carry their
//! template vertices, and every arc cuts its region as the template's arc does. Such an arc
//! always exists, so nothing is searched or retried; geometry only chooses among valid routes.

use crate::{DualLoops, RealizedSkeleton, SkeletonResult};
use dualcube_dual::{
    Loop, LoopID, LoopSampler, LoopState,
    insertion::{DrawError, Plan},
    loop_map::LoopMap,
};
use dualcube_types::prelude::*;

mod lift;
mod route;
mod template;

/// Each of these reports a bug: valid input always has a drawing.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum Failure {
    BoundaryTopology,
    Arc { loop_id: LoopID, error: DrawError },
    InvalidDual(String),
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BoundaryTopology => {
                f.write_str("Lifted boundaries do not match the cuboid template topology")
            }
            Self::Arc { loop_id, error } => write!(
                f,
                "Could not draw loop {loop_id:?}: {error}; retained the partial drawing"
            ),
            Self::InvalidDual(error) => {
                write!(f, "Completed loop structure failed validation: {error}")
            }
        }
    }
}
impl std::error::Error for Failure {}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Attempt {
    pub loops: DualLoops,
    /// The lifted patch boundaries alone, before any template loop is routed.
    #[serde(default)]
    pub lifted: DualLoops,
    /// The dual structure of a complete drawing, read from its loop map.
    pub dual: Option<dualcube_dual::Dual>,
    pub planned: usize,
    pub failure: Option<Failure>,
}

pub fn attempt(mesh: &Mesh<INPUT>, realized: &RealizedSkeleton) -> SkeletonResult<Attempt> {
    let timer = std::time::Instant::now();
    realized.validate_geometry()?;
    realized.labeled.skeleton.debug_validate(mesh);
    let template = template::Template::build(&realized.labeled)?;
    let target = &template.target;
    let mut drawn: HashSet<_> = template.boundaries.values().copied().collect();
    let mut result = Attempt {
        loops: DualLoops::with_key(),
        lifted: DualLoops::with_key(),
        dual: None,
        planned: template.loops.len(),
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
    result.lifted = output(&map, &drawn)?;
    // The template vertex of every crossing vertex of the drawing.
    let mut known = HashMap::new();
    if !target.matches(&map, &drawn, &known) {
        result.loops = result.lifted.clone();
        result.failure = Some(Failure::BoundaryTopology);
        return Ok(result);
    }
    let flows = route::Flows::new(mesh, drawn.is_empty());
    let partners: HashMap<_, _> = template
        .loops
        .keys()
        .map(|id| (id, target.partners(id)))
        .collect();
    while drawn.len() < template.loops.len() {
        // Most drawn loops crossed first; the template is connected, so only the first loop
        // of an empty drawing crosses none.
        let id = template
            .loops
            .keys()
            .filter(|id| !drawn.contains(id))
            .max_by_key(|&id| {
                let neighbors: Vec<_> =
                    partners[&id].iter().filter(|p| drawn.contains(p)).collect();
                (
                    neighbors.iter().collect::<HashSet<_>>().len(),
                    neighbors.len(),
                    std::cmp::Reverse(id),
                )
            })
            .unwrap();
        let axis = template.loops[id].direction;
        let plan = Plan::new(target, &drawn, id);
        let next = if plan.is_empty() {
            route::seed(mesh, &flows, &map, id, axis)
                .map(|map| (map, vec![]))
                .ok_or(DrawError::NoStart)
        } else {
            plan.draw(&map, &known, flows.cost(axis))
        };
        match next {
            Ok((next, crossings)) => {
                map = next;
                known.extend(crossings);
                drawn.insert(id);
            }
            Err(error) => {
                result.loops = output(&map, &drawn)?;
                result.failure = Some(Failure::Arc { loop_id: id, error });
                return Ok(result);
            }
        }
        tracing::info!(
            "Loop drawing: {}/{} loops",
            drawn.len(),
            template.loops.len()
        );
    }
    debug_assert!(target.matches(&map, &drawn, &known));
    result.loops = output(&map, &drawn)?;
    // The dual structure comes from the map; drawn curves are for output only.
    match dualcube_dual::Dual::from_map(std::sync::Arc::new(mesh.clone()), &map, &result.loops) {
        Ok(dual) => result.dual = Some(dual),
        Err(error) => result.failure = Some(Failure::InvalidDual(error.to_string())),
    }
    tracing::info!(
        "Loop drawing complete: {} loops (elapsed {:.2}s)",
        result.loops.len(),
        timer.elapsed().as_secs_f64()
    );
    Ok(result)
}

fn subset(loops: &DualLoops, selected: &HashSet<LoopID>) -> DualLoops {
    let mut result = loops.clone();
    result.retain(|id, _| selected.contains(&id));
    result
}
#[cfg(test)]
mod tests;
