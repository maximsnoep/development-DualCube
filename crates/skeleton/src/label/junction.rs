//! Split overfull junctions: two of their boundaries off along a surface connector.

use super::*;
use crate::surface::{Regions, edit};

/// Split every junction of degree above six until none is left.
pub(super) fn reduce(mesh: &Mesh<INPUT>, result: &mut Attempt) -> SkeletonResult<()> {
    loop {
        let degree = result.skeleton.degrees();
        let Some((node, &count)) = degree.iter().find(|(_, d)| **d > 6) else {
            return Ok(());
        };
        // Any two boundaries give a valid split (into degrees 3 and `count - 1`): the first two,
        // along their first connector. A genus-zero patch is connected, so they always have one.
        let regions = Regions::build(mesh, &result.skeleton)?;
        let incident: Vec<_> = result.skeleton.incident(node).map(|(id, _)| id).collect();
        let pair = [incident[0], incident[1]];
        let Some(route) = edit::connector(&regions, &result.skeleton, node, pair, None) else {
            panic!("Impossible: degree-{count} patch {node:?} has no connector between boundaries");
        };
        let (skeleton, added_node, boundary) =
            edit::band(mesh, &result.skeleton, &regions, node, pair, &route)?;
        skeleton.debug_validate(mesh);
        result.skeleton = skeleton;
        result.junction_splits.push(JunctionSplit {
            original_node: node,
            added_node,
            boundary,
        });
    }
}
