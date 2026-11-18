//! The drawn curves as legacy straight chords (the loops that loop evolution mutates).

use super::*;

/// Every loop as its mesh-edge crossings, at the drawn positions; a return through the edge
/// just crossed is dropped. `None` unless the chords have the curves' exact topology and a
/// valid dual: a loop may visit a triangle at most once, and straightening may reorder
/// crossings.
pub fn chords(mesh: &Mesh<INPUT>, loops: &DualLoops) -> Option<DualLoops> {
    let mut result = loops.clone();
    for (id, curve) in loops {
        let mut exits: Vec<(EdgeID, f64)> = Vec::new();
        for arc in curve.surface_arcs(mesh).ok()? {
            let weights = [1. - arc.to[0] - arc.to[1], arc.to[0], arc.to[1]];
            // Side `k` runs from corner `k` to corner `k + 1`.
            let Some(k) = (0..3).find(|&k| weights[(k + 2) % 3].abs() < 1e-9) else {
                continue;
            };
            let side = mesh.edges(arc.face).nth(k)?;
            exits.push((side, weights[(k + 1) % 3].clamp(1e-6, 1. - 1e-6)));
        }
        while let Some(i) = (0..exits.len())
            .find(|&i| exits[(i + 1) % exits.len()].0 == mesh.twin(exits[i].0))
        {
            let j = (i + 1) % exits.len();
            exits.remove(i.max(j));
            exits.remove(i.min(j));
        }
        if exits.len() < 3 {
            return None;
        }
        result[id] = Loop {
            edges: exits.iter().flat_map(|&(e, _)| [e, mesh.twin(e)]).collect(),
            offsets: exits.iter().flat_map(|&(_, t)| [t, 1. - t]).collect(),
            direction: curve.direction,
            surface: vec![],
        };
    }
    let exact = PartialTopology::new(mesh, loops).ok()?;
    let straight = PartialTopology::new(mesh, &result).ok()?;
    if !straight.equivalent(&exact) {
        return None;
    }
    dualcube_dual::Dual::from(std::sync::Arc::new(mesh.clone()), &result).ok()?;
    Some(result)
}
