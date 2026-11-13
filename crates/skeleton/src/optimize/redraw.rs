//! New loops cut from vertex sides: redraw a boundary with the same partition (redraw lemma).
//!
//! Every input vertex of a region gets a side; crossings of the region's other loops are pinned.
//! Each interval whose ends get different sides receives one crossing, and inside every cell
//! the changes are paired so that runs of the `false` side are cut off. Validation of the
//! resulting skeleton is the only acceptance test.

use super::cut::Network;
use super::fidelity::{self, Element, Measure, Star};
use super::moves::gain;
use crate::label::Port;
use crate::surface::{Gap, Regions, Side, Vertex, canonical};
use crate::{
    BoundaryCrossing, BoundaryCrossingID, LabeledSkeleton, Skeleton, SkeletonEdgeID, SkeletonNodeID,
};
use dualcube_types::prelude::*;
use std::collections::{HashMap, HashSet};

/// The loop separating the sides within `owner`'s region, as (interval, exit) in traversal
/// order; `None` unless the side changes form exactly one loop.
pub(crate) fn trace(
    mesh: &Mesh<INPUT>,
    regions: &Regions,
    owner: SkeletonNodeID,
    side: &dyn Fn(Vertex) -> bool,
) -> Option<Vec<(Gap, EdgeID)>> {
    let mut partner = HashMap::<(usize, Gap), Gap>::new();
    let mut cells_of = HashMap::<Gap, Vec<usize>>::new();
    for (i, cell) in regions.cells.iter().enumerate() {
        if regions.owners[i] != owner {
            continue;
        }
        let n = cell.vertices.len();
        let sides: Vec<bool> = cell.vertices.iter().map(|&v| side(v)).collect();
        let mut changes = Vec::new();
        for j in 0..n {
            if sides[j] == sides[(j + 1) % n] {
                continue;
            }
            match cell.sides[j] {
                Side::Gap(gap) => changes.push((gap, sides[j])),
                // Both ends of an arc belong to one pinned loop.
                Side::Arc { .. } => return None,
            }
        }
        if changes.is_empty() {
            continue;
        }
        let start = changes.iter().position(|c| c.1)?;
        let m = changes.len();
        for pair in 0..m / 2 {
            let a = changes[(start + 2 * pair) % m].0;
            let b = changes[(start + 2 * pair + 1) % m].0;
            partner.insert((i, a), b);
            partner.insert((i, b), a);
        }
        for (gap, _) in changes {
            cells_of.entry(gap).or_default().push(i);
        }
    }
    let (&first, cells) = cells_of.iter().next()?;
    if cells_of.values().any(|c| c.len() != 2) {
        return None;
    }
    let (mut gap, mut from, mut into) = (first, cells[0], cells[1]);
    let mut path = Vec::new();
    loop {
        let exit = if mesh.face(gap.edge) == regions.cells[from].face {
            gap.edge
        } else {
            mesh.twin(gap.edge)
        };
        path.push((gap, exit));
        let next = *partner.get(&(into, gap))?;
        let cells = &cells_of[&next];
        from = into;
        into = if cells[0] == into { cells[1] } else { cells[0] };
        gap = next;
        if gap == first {
            break;
        }
        if path.len() > cells_of.len() {
            return None;
        }
    }
    (path.len() == cells_of.len()).then_some(path)
}

/// Insert crossings at intervals of the current edge orders (each interval at most once).
pub(crate) fn insert(skeleton: &mut Skeleton, path: &[(Gap, EdgeID)]) -> Vec<BoundaryCrossingID> {
    let mut per_edge = HashMap::<EdgeID, Vec<(usize, BoundaryCrossingID)>>::new();
    let ids: Vec<_> = path
        .iter()
        .map(|&(gap, exit)| {
            let id = skeleton.surface.crossings.insert(BoundaryCrossing { exit });
            per_edge.entry(gap.edge).or_default().push((gap.index, id));
            id
        })
        .collect();
    for (edge, mut added) in per_edge {
        added.sort_by_key(|a| a.0);
        let old = skeleton
            .surface
            .edge_order
            .remove(&edge)
            .unwrap_or_default();
        let mut order = Vec::with_capacity(old.len() + added.len());
        let mut next = added.into_iter().peekable();
        for (i, &c) in old.iter().enumerate() {
            while let Some((_, id)) = next.next_if(|a| a.0 == i) {
                order.push(id);
            }
            order.push(c);
        }
        order.extend(next.map(|a| a.1));
        skeleton.surface.edge_order.insert(edge, order);
    }
    ids
}

/// Remove a loop's crossings from the surface.
pub(crate) fn erase(mesh: &Mesh<INPUT>, skeleton: &mut Skeleton, crossings: &[BoundaryCrossingID]) {
    for &crossing in crossings {
        let data = skeleton.surface.crossings.remove(crossing).unwrap();
        let key = canonical(mesh, data.exit);
        let order = skeleton.surface.edge_order.get_mut(&key).unwrap();
        order.retain(|&c| c != crossing);
        if order.is_empty() {
            skeleton.surface.edge_order.remove(&key);
        }
    }
}

/// Replace a boundary's loop, keeping its orientation (`nodes[0]` on the left); `None` if
/// neither traversal direction validates.
fn replace(
    mesh: &Mesh<INPUT>,
    skeleton: &Skeleton,
    edge: SkeletonEdgeID,
    path: &[(Gap, EdgeID)],
) -> Option<Skeleton> {
    let mut result = skeleton.clone();
    let old = result.edges[edge].boundary.crossings.clone();
    erase(mesh, &mut result, &old);
    result.edges[edge].boundary.crossings = insert(&mut result, path);
    if result.validate(mesh).is_ok() {
        return Some(result);
    }
    let mut boundary = result.edges[edge].boundary.clone();
    crate::surface::edit::reverse(mesh, &mut result, &mut boundary);
    result.edges[edge].boundary = boundary;
    result.validate(mesh).ok().map(|()| result)
}

#[derive(Clone, Copy, Debug)]
pub struct RedrawParams {
    /// Cost of cutting an interval, per unit of interval length times edge length, relative
    /// to the area-weighted gain.
    pub smoothness: f64,
    pub measure: Measure,
    pub rounds: usize,
}

/// Gain of a cell placed in a star with the given labels and free ports.
fn cell_gain(
    star: &Star,
    free: u8,
    labeled: &LabeledSkeleton,
    cell: &fidelity::CellGeometry,
    measure: Measure,
) -> f64 {
    let c = match star.element(cell.centroid) {
        Element::Cube => fidelity::best_per_mask(cell.normal)[free as usize],
        Element::Arm(e) => fidelity::wall(
            cell.normal,
            Port::from_label(labeled.labels[e]) as usize / 2,
        ),
    };
    gain(measure, [c, fidelity::distortion(c)])
}

/// Redraw one boundary by a minimum cut on the union of its two patches; repeated with
/// updated stars for `rounds`. Returns `None` when nothing valid and different was found.
pub(crate) fn redraw(
    mesh: &Mesh<INPUT>,
    labeled: &LabeledSkeleton,
    edge: SkeletonEdgeID,
    params: &RedrawParams,
) -> Option<LabeledSkeleton> {
    let mut current = labeled.clone();
    let mut changed = false;
    for _ in 0..params.rounds.max(1) {
        let Some(next) = redraw_once(mesh, &current, edge, params) else {
            break;
        };
        let same = {
            let old: HashSet<_> = current.skeleton.edges[edge]
                .boundary
                .crossings
                .iter()
                .map(|&c| current.skeleton.surface.crossings[c].exit)
                .collect();
            let new: HashSet<_> = next.skeleton.edges[edge]
                .boundary
                .crossings
                .iter()
                .map(|&c| next.skeleton.surface.crossings[c].exit)
                .collect();
            old == new
        };
        current = next;
        if same {
            break;
        }
        changed = true;
    }
    changed.then_some(current)
}

fn redraw_once(
    mesh: &Mesh<INPUT>,
    labeled: &LabeledSkeleton,
    edge: SkeletonEdgeID,
    params: &RedrawParams,
) -> Option<LabeledSkeleton> {
    let skeleton = &labeled.skeleton;
    let [a, b] = skeleton.edges[edge].nodes;
    // Stars from the current geometry.
    let regions = Regions::build(mesh, skeleton).ok()?;
    let positions = fidelity::crossing_positions(mesh, skeleton);
    let cells = fidelity::cells(mesh, &regions, &positions);
    let stars = fidelity::stars(skeleton, &cells, &fidelity::loops(skeleton, &positions));
    let used = crate::label::constraints::used_ports(skeleton, &labeled.labels).ok()?;

    // The union of both patches as one region, with the boundary removed.
    let mut merged = skeleton.clone();
    let old = merged.edges.remove(edge).unwrap();
    erase(mesh, &mut merged, &old.boundary.crossings);
    let mut pinned = HashMap::<BoundaryCrossingID, bool>::new();
    for data in merged.edges.values_mut() {
        let side = data.nodes.contains(&a);
        if side || data.nodes.contains(&b) {
            for &c in &data.boundary.crossings {
                pinned.insert(c, side);
            }
        }
        for n in &mut data.nodes {
            if *n == b {
                *n = a;
            }
        }
    }
    merged.nodes.remove(b);
    let union = Regions::build(mesh, &merged).ok()?;
    let union_positions = fidelity::crossing_positions(mesh, &merged);
    let union_cells = fidelity::cells(mesh, &union, &union_positions);

    let gains = |i: usize| {
        [a, b].map(|node| {
            cell_gain(
                &stars[node],
                63 & !used[node],
                labeled,
                &union_cells[i],
                params.measure,
            )
        })
    };
    let (index, sides) = cut_sides(
        mesh,
        &union,
        &union_cells,
        &union_positions,
        a,
        &pinned,
        &gains,
        params.smoothness,
    )?;
    let side = |v: Vertex| match v {
        Vertex::Input(v) => sides[index[&v]],
        Vertex::Crossing(c) => pinned.get(&c).copied().unwrap_or(true),
    };
    let path = trace(mesh, &union, a, &side)?;
    let skeleton = replace(mesh, skeleton, edge, &path)?;
    Some(LabeledSkeleton {
        skeleton,
        labels: labeled.labels.clone(),
    })
}

/// Vertex sides within `owner`'s region by a minimum cut: `gains(cell)` is the cell's gain on
/// the `true` and `false` side, `pinned` fixes crossings, and cutting an interval costs
/// `smoothness` × its length × its edge length. Islands not touching their side's loops are
/// flipped (a side without loops keeps its largest component).
#[allow(clippy::too_many_arguments)]
fn cut_sides(
    mesh: &Mesh<INPUT>,
    union: &Regions,
    union_cells: &[fidelity::CellGeometry],
    union_positions: &HashMap<BoundaryCrossingID, Vector3D>,
    owner: SkeletonNodeID,
    pinned: &HashMap<BoundaryCrossingID, bool>,
    gains: &dyn Fn(usize) -> [f64; 2],
    smoothness: f64,
) -> Option<(HashMap<VertID, usize>, Vec<bool>)> {
    let mut index = HashMap::<VertID, usize>::new();
    for (i, cell) in union.cells.iter().enumerate() {
        if union.owners[i] != owner {
            continue;
        }
        for v in &cell.vertices {
            if let Vertex::Input(v) = *v {
                let n = index.len();
                index.entry(v).or_insert(n);
            }
        }
    }
    let n = index.len();
    if n == 0 {
        return None;
    }
    let (source, sink) = (n, n + 1);
    let mut cost = vec![[0f64; 2]; n];
    let mut network = Network::new(n + 2);
    let mut seen = HashSet::new();
    let mut neighbors = vec![Vec::new(); n];
    let mut anchored = [vec![false; n], vec![false; n]];
    for (i, cell) in union.cells.iter().enumerate() {
        if union.owners[i] != owner {
            continue;
        }
        let share = union_cells[i].area / cell.vertices.len() as f64;
        let gains = gains(i);
        for (j, v) in cell.vertices.iter().enumerate() {
            if let Vertex::Input(v) = *v {
                cost[index[&v]][0] -= share * gains[0];
                cost[index[&v]][1] -= share * gains[1];
            }
            let Side::Gap(gap) = cell.sides[j] else {
                continue;
            };
            if !seen.insert(gap) {
                continue;
            }
            let ends = [*v, cell.vertices[(j + 1) % cell.vertices.len()]];
            let length = |v: Vertex| match v {
                Vertex::Input(v) => mesh.position(v),
                Vertex::Crossing(c) => union_positions[&c],
            };
            let span = (length(ends[1]) - length(ends[0])).norm();
            let weight = smoothness * span * mesh.vector(gap.edge).norm();
            match ends {
                [Vertex::Input(x), Vertex::Input(y)] => {
                    let (x, y) = (index[&x], index[&y]);
                    network.add(x, y, weight, weight);
                    neighbors[x].push(y);
                    neighbors[y].push(x);
                }
                [Vertex::Input(x), Vertex::Crossing(c)]
                | [Vertex::Crossing(c), Vertex::Input(x)] => {
                    let x = index[&x];
                    let side = *pinned.get(&c)?;
                    // Assigning x to the other side cuts this interval.
                    cost[x][usize::from(side)] += weight;
                    anchored[usize::from(!side)][x] = true;
                }
                _ => {}
            }
        }
    }
    for (x, [on_a, on_b]) in cost.iter().enumerate() {
        let low = on_a.min(*on_b);
        // Source side is `true`: staying there cuts x→sink.
        network.add(source, x, on_b - low, 0.);
        network.add(x, sink, on_a - low, 0.);
    }
    let cut = network.source_side(source, sink);
    let mut sides: Vec<bool> = cut[..n].to_vec();
    // Drop islands: every component of a side must touch that side's loops (or be the
    // largest component of a side without other loops).
    for side in [true, false] {
        let anchors = &anchored[usize::from(!side)];
        let has_loops = pinned.values().any(|&s| s == side);
        let mut component = vec![usize::MAX; n];
        let mut groups: Vec<(Vec<usize>, bool)> = Vec::new();
        for start in 0..n {
            if sides[start] != side || component[start] != usize::MAX {
                continue;
            }
            let id = groups.len();
            let mut members = vec![start];
            component[start] = id;
            let mut k = 0;
            while k < members.len() {
                for &y in &neighbors[members[k]] {
                    if sides[y] == side && component[y] == usize::MAX {
                        component[y] = id;
                        members.push(y);
                    }
                }
                k += 1;
            }
            let touches = members.iter().any(|&x| anchors[x]);
            groups.push((members, touches));
        }
        let keep_largest = !has_loops;
        let largest = groups
            .iter()
            .enumerate()
            .max_by_key(|(_, g)| g.0.len())
            .map(|(i, _)| i);
        for (i, (members, touches)) in groups.iter().enumerate() {
            let keep = if keep_largest {
                Some(i) == largest
            } else {
                *touches
            };
            if !keep {
                for &x in members {
                    sides[x] = !side;
                }
            }
        }
    }
    Some((index, sides))
}

/// Split a disk off a node's patch as a new leaf at a free port (split lemma): vertices go to
/// the leaf where a cube with every port but the one back to the node fits better than the
/// node's star. `None` if the port is taken or no valid disk results.
pub(crate) fn add_leaf(
    mesh: &Mesh<INPUT>,
    labeled: &LabeledSkeleton,
    node: SkeletonNodeID,
    port: Port,
    params: &RedrawParams,
) -> Option<LabeledSkeleton> {
    let skeleton = &labeled.skeleton;
    let used = crate::label::constraints::used_ports(skeleton, &labeled.labels).ok()?;
    if used[node] & port.bit() != 0 {
        return None;
    }
    let regions = Regions::build(mesh, skeleton).ok()?;
    let positions = fidelity::crossing_positions(mesh, skeleton);
    let cells = fidelity::cells(mesh, &regions, &positions);
    let stars = fidelity::stars(skeleton, &cells, &fidelity::loops(skeleton, &positions));
    let pinned: HashMap<_, _> = skeleton
        .edges
        .values()
        .filter(|e| e.nodes.contains(&node))
        .flat_map(|e| e.boundary.crossings.iter().map(|&c| (c, false)))
        .collect();
    let free = 63 & !used[node];
    let leaf_free = 63 & !port.opposite().bit();
    let gains = |i: usize| {
        let cell = &cells[i];
        let leaf = fidelity::best_per_mask(cell.normal)[leaf_free as usize];
        [
            gain(params.measure, [leaf, fidelity::distortion(leaf)]),
            cell_gain(
                &stars[node],
                free & !port.bit(),
                labeled,
                cell,
                params.measure,
            ),
        ]
    };
    let (index, sides) = cut_sides(
        mesh,
        &regions,
        &cells,
        &positions,
        node,
        &pinned,
        &gains,
        // A disk pays its whole perimeter; the score decides whether it is worth it.
        params.smoothness / 8.,
    )?;
    if !sides.iter().any(|&s| s) {
        return None;
    }
    let side = |v: Vertex| match v {
        Vertex::Input(v) => sides[index[&v]],
        Vertex::Crossing(c) => pinned.get(&c).copied().unwrap_or(false),
    };
    let path = trace(mesh, &regions, node, &side)?;
    let centre = index
        .iter()
        .filter(|&(_, &i)| sides[i])
        .map(|(&v, _)| mesh.position(v))
        .sum::<Vector3D>()
        / sides.iter().filter(|&&s| s).count() as f64;
    let mut result = labeled.clone();
    let leaf = result
        .skeleton
        .nodes
        .insert(crate::SkeletonNode { position: centre });
    let crossings = insert(&mut result.skeleton, &path);
    let edge = result.skeleton.edges.insert(crate::SkeletonEdge {
        nodes: [node, leaf],
        boundary: crate::PatchBoundary { crossings },
    });
    result.labels.insert(edge, port.label());
    if result.skeleton.validate(mesh).is_err() {
        let mut boundary = result.skeleton.edges[edge].boundary.clone();
        crate::surface::edit::reverse(mesh, &mut result.skeleton, &mut boundary);
        result.skeleton.edges[edge].boundary = boundary;
        result.skeleton.validate(mesh).ok()?;
    }
    Some(result)
}
