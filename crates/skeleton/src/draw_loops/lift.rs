//! Patch boundaries as loops of an exact loop map, straight from the skeleton's crossing orders.

use super::*;
use dualcube_dual::loop_map::LoopMap;

/// The input mesh cut by every patch boundary, as the template loop of its edge cuboid.
/// Negative labels reverse the traversal.
pub(super) fn boundaries(
    mesh: &Mesh<INPUT>,
    realized: &RealizedSkeleton,
    template: &template::Template,
) -> SkeletonResult<LoopMap> {
    let skeleton = &realized.labeled.skeleton;
    let surface = &skeleton.surface;
    let index: HashMap<_, _> = surface
        .crossings
        .keys()
        .enumerate()
        .map(|(i, id)| (id, i))
        .collect();
    let mut exits: Vec<_> = surface.crossings.values().map(|c| c.exit).collect();
    let mut curves = vec![];
    for (edge, e) in &skeleton.edges {
        let mut occurrences: Vec<_> = e.boundary.crossings.iter().map(|c| index[c]).collect();
        if realized.labeled.labels[edge].sign == Sign::Negative {
            // Reversed, every occurrence enters where it used to exit.
            occurrences.reverse();
            for &o in &occurrences {
                exits[o] = mesh.twin(exits[o]);
            }
        }
        curves.push((template.boundaries[&edge], occurrences));
    }
    let order = surface
        .edge_order
        .iter()
        .map(|(&edge, list)| (edge, list.iter().map(|c| index[c]).collect()))
        .collect();
    let mut map = LoopMap::new(mesh);
    map.add_disjoint(mesh, &curves, &exits, &order)?;
    Ok(map)
}
