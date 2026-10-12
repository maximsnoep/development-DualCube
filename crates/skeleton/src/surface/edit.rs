//! Boundary edits in crossing order, independent of mesh resolution.

use super::*;
use crate::{BoundaryCrossing, SkeletonEdge, SkeletonNode};
use std::collections::VecDeque;

/// Copy a loop immediately to its left (side 0) or right (side 1).
/// The caller must attach the returned loop to an edge before validation.
pub(crate) fn parallel(
    mesh: &Mesh<INPUT>,
    skeleton: &mut Skeleton,
    original: &PatchBoundary,
    side: usize,
) -> PatchBoundary {
    let mut crossings = Vec::new();
    for &old in &original.crossings {
        let crossing = skeleton.surface.crossings[old];
        let id = skeleton.surface.crossings.insert(crossing);
        let key = canonical(mesh, crossing.exit);
        let order = skeleton.surface.edge_order.get_mut(&key).unwrap();
        let rank = order.iter().position(|&c| c == old).unwrap();
        let after = (side == 0) == (key == crossing.exit);
        order.insert(rank + usize::from(after), id);
        crossings.push(id);
    }
    PatchBoundary { crossings }
}

pub(crate) fn reverse(mesh: &Mesh<INPUT>, skeleton: &mut Skeleton, boundary: &mut PatchBoundary) {
    boundary.crossings.reverse();
    for &id in &boundary.crossings {
        let crossing = &mut skeleton.surface.crossings[id];
        crossing.exit = mesh.twin(crossing.exit);
    }
}

#[derive(Clone)]
pub(crate) struct Connector {
    pub start: BoundaryCrossingID,
    pub end: BoundaryCrossingID,
    /// Source cell and crossed interval, in travel order.
    pub steps: Vec<(usize, Gap)>,
}

/// Route in the chosen region without crossing any existing boundary.
pub(crate) fn connector(
    regions: &Regions,
    skeleton: &Skeleton,
    owner: SkeletonNodeID,
    pair: [crate::SkeletonEdgeID; 2],
    blocked: Option<Gap>,
) -> Option<Connector> {
    let side = |e| usize::from(skeleton.edges[e].nodes[1] == owner);
    let mut targets = HashMap::new();
    for &arc in &skeleton.edges[pair[1]].boundary.crossings {
        targets
            .entry(regions.arc_sides[&arc][side(pair[1])])
            .or_insert(arc);
    }
    let mut source = vec![None; regions.cells.len()];
    let mut previous = vec![None; regions.cells.len()];
    let mut queue = VecDeque::new();
    for &arc in &skeleton.edges[pair[0]].boundary.crossings {
        let cell = regions.arc_sides[&arc][side(pair[0])];
        if source[cell].is_none() {
            source[cell] = Some(arc);
            queue.push_back(cell);
        }
    }
    while let Some(cell) = queue.pop_front() {
        if let Some(&end) = targets.get(&cell) {
            let mut steps = Vec::new();
            let mut current = cell;
            while let Some((parent, gap)) = previous[current] {
                steps.push((parent, gap));
                current = parent;
            }
            steps.reverse();
            return Some(Connector {
                start: source[cell].unwrap(),
                end,
                steps,
            });
        }
        for &(next, gap) in &regions.adjacent[cell] {
            if Some(gap) != blocked && regions.owners[next] == owner && source[next].is_none() {
                source[next] = source[cell];
                previous[next] = Some((cell, gap));
                queue.push_back(next);
            }
        }
    }
    None
}

/// Enclose two boundary components and an interior connector in a new patch.
pub(crate) fn band(
    mesh: &Mesh<INPUT>,
    skeleton: &Skeleton,
    regions: &Regions,
    owner: SkeletonNodeID,
    pair: [crate::SkeletonEdgeID; 2],
    route: &Connector,
) -> SkeletonResult<(Skeleton, SkeletonNodeID, crate::SkeletonEdgeID)> {
    let mut split = skeleton.clone();
    let mut copies = HashMap::new();
    let mut loops = Vec::new();
    for (edge, anchor) in pair.into_iter().zip([route.start, route.end]) {
        let original = &skeleton.edges[edge];
        let side = usize::from(original.nodes[1] == owner);
        let mut copy = parallel(mesh, &mut split, &original.boundary, side);
        let mapping: HashMap<_, _> = original
            .boundary
            .crossings
            .iter()
            .copied()
            .zip(copy.crossings.iter().copied())
            .collect();
        let start = if side == 0 {
            reverse(mesh, &mut split, &mut copy);
            let index = original
                .boundary
                .crossings
                .iter()
                .position(|&c| c == anchor)
                .unwrap();
            mapping[&original.boundary.crossings[(index + 1) % original.boundary.crossings.len()]]
        } else {
            mapping[&anchor]
        };
        // Traverse each copy from just after the removed attachment arc to just before it.
        let index = copy.crossings.iter().position(|&c| c == start).unwrap();
        let len = copy.crossings.len();
        copy.crossings.rotate_left((index + 1) % len);
        copies.extend(mapping);
        loops.push(copy.crossings);
    }
    let mut outward = Vec::new();
    let mut inward = Vec::new();
    for &(cell, gap) in &route.steps {
        let exit = if mesh.face(gap.edge) == regions.cells[cell].face {
            gap.edge
        } else {
            mesh.twin(gap.edge)
        };
        let forward = split.surface.crossings.insert(BoundaryCrossing { exit });
        let backward = split.surface.crossings.insert(BoundaryCrossing {
            exit: mesh.twin(exit),
        });
        let old = skeleton
            .surface
            .edge_order
            .get(&gap.edge)
            .map_or(&[][..], Vec::as_slice);
        let lower = gap.index.checked_sub(1).map(|i| old[i]);
        let upper = old.get(gap.index).copied();
        let order = split.surface.edge_order.entry(gap.edge).or_default();
        let mut lo = lower.map_or(0, |id| order.iter().position(|&c| c == id).unwrap() + 1);
        let mut hi = upper.map_or(order.len(), |id| {
            order.iter().position(|&c| c == id).unwrap()
        });
        // Stay outside the two new collars, inside the original region.
        if let Some(copy) = lower.and_then(|id| copies.get(&id)) {
            let i = order.iter().position(|c| c == copy).unwrap();
            if (lo..hi).contains(&i) {
                lo = i + 1;
            }
        }
        if let Some(copy) = upper.and_then(|id| copies.get(&id)) {
            let i = order.iter().position(|c| c == copy).unwrap();
            if (lo..hi).contains(&i) {
                hi = i;
            }
        }
        if lo != hi {
            return Err("Connector interval contains unexpected crossings".into());
        }
        let lanes = if exit == gap.edge {
            [forward, backward]
        } else {
            [backward, forward]
        };
        order.splice(lo..lo, lanes);
        outward.push(forward);
        inward.push(backward);
    }
    let mut crossings = loops.remove(0);
    crossings.extend(outward);
    crossings.extend(loops.remove(0));
    crossings.extend(inward.into_iter().rev());
    // TODO: Score region geometry beyond this boundary sample average.
    let center = crossings
        .iter()
        .map(|&id| position(mesh, &split, id))
        .sum::<Vector3D>()
        / crossings.len() as f64;
    let added = split.nodes.insert(SkeletonNode { position: center });
    for edge in pair {
        for node in &mut split.edges[edge].nodes {
            if *node == owner {
                *node = added;
            }
        }
    }
    let boundary = split.edges.insert(SkeletonEdge {
        nodes: [added, owner],
        boundary: PatchBoundary { crossings },
    });
    Ok((split, added, boundary))
}
