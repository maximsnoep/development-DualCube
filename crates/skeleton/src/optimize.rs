//! Simplify a labeled skeleton chain by chain, then recompute target placement.
//!
//! On every chain (a maximal path through degree-two patches) a DP keeps the boundaries and
//! labels with the least cost, all other labels fixed; dropped boundaries merge their patches.
//! A chain from a junction to a leaf may drop all of them. This repeats until no chain improves.
//! The search over other moves (ALNS against the primalized quality) lives on the branch
//! `skeleton-experiments`.

use crate::label::{self, Port, constraints, path};
use crate::{
    LabeledSkeleton, RealizedSkeleton, Skeleton, SkeletonEdgeID, SkeletonNodeID, SkeletonResult,
    place_cubes,
};
use dualcube_types::prelude::*;
use slotmap::SecondaryMap;
use std::f64::consts::PI;

/// Cost of every kept boundary (a loop pair of the polycube).
const BOUNDARY: f64 = 0.5;
/// A merged patch (the junction patches at the chain ends included) pays up to this as its width
/// falls below `THIN_RATIO` times its loops' mean diameter.
const THIN: f64 = 1.;
const THIN_RATIO: f64 = 0.5;
/// A junction patch pays this per length of chain it absorbs, over the kept boundary's diameter:
/// loops sit at the root of an arm, not at its tip.
const ABSORB: f64 = 0.05;
/// Weight of a leaf cap's direction against its port, times how far the cap protrudes.
const CAP: f64 = 1.;
/// Toward a leaf, a dropped item counts less as what lies beyond its boundary protrudes less than
/// twice this times the boundary's diameter, and not at all below once.
const FADE: f64 = 0.2;
/// Weight of a patch's direction along the chain, per its length over its loops' diameter.
const DIRECTION: f64 = 0.5;

pub fn run(mesh: &Mesh<INPUT>, placed: &RealizedSkeleton) -> SkeletonResult<RealizedSkeleton> {
    placed.labeled.skeleton.debug_validate(mesh);
    let reduced = reduce(mesh, &placed.labeled)?;
    place_cubes::run(mesh, &reduced)
}

/// Reduce every chain until none improves; the result is validated. Labels should be the
/// labeling's.
pub fn reduce(mesh: &Mesh<INPUT>, labeled: &LabeledSkeleton) -> SkeletonResult<LabeledSkeleton> {
    let start = std::time::Instant::now();
    let mut labeled = labeled.clone();
    let skeleton = &labeled.skeleton;
    let mut tables = Tables {
        penalties: label::boundary_costs(mesh, skeleton),
        boundaries: boundary_measures(mesh, skeleton),
        patches: patch_measures(mesh, skeleton)?,
    };
    let (patches, mut branches) = (skeleton.nodes.len(), 0);
    // Every move lowers its chain's cost; the bound only guards against relabels cycling
    // between chains.
    'improve: for _ in 0..10 * (skeleton.edges.len() + 10) {
        for chain in chains(&labeled.skeleton) {
            if let Some((next, branch)) = reduce_chain(mesh, &labeled, &chain, &mut tables) {
                labeled = next;
                branches += usize::from(branch);
                continue 'improve;
            }
        }
        break;
    }
    labeled.validate(mesh)?;
    tracing::info!(
        "Optimization: {patches} → {} patches, {branches} branches dropped, {:.2}s",
        labeled.skeleton.nodes.len(),
        start.elapsed().as_secs_f64()
    );
    Ok(labeled)
}

/// A maximal path whose interior nodes have degree two, from `nodes[0]` to its last node; a
/// component that is one ring runs from a node back to itself.
struct Chain {
    nodes: Vec<SkeletonNodeID>,
    edges: Vec<SkeletonEdgeID>,
    /// Whether edge `i` is stored in chain direction.
    forward: Vec<bool>,
}

fn chains(skeleton: &Skeleton) -> Vec<Chain> {
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    for edge in skeleton.edges.keys() {
        if seen.contains(&edge) {
            continue;
        }
        let (steps, [start, _]) = label::chain(skeleton, edge);
        let mut nodes = vec![start];
        for &(e, forward) in &steps {
            seen.insert(e);
            nodes.push(skeleton.edges[e].nodes[usize::from(forward)]);
        }
        result.push(Chain {
            nodes,
            edges: steps.iter().map(|s| s.0).collect(),
            forward: steps.iter().map(|s| s.1).collect(),
        });
    }
    result
}

struct Boundary {
    length: f64,
    /// Norm of the vector area: the area of the disk the loop spans, robust to zigzags.
    span: f64,
    centre: Vector3D,
}

impl Boundary {
    /// Of the disk the loop spans; from the length if it spans none.
    fn diameter(&self) -> f64 {
        if self.span > 0. {
            2. * (self.span / PI).sqrt()
        } else {
            self.length / PI
        }
    }

    /// Height of a patch of area `area` beyond the loop (its excess over the spanned disk, per
    /// the disk's perimeter), over the loop's diameter; infinite for a degenerate loop.
    fn rise(&self, area: f64) -> f64 {
        let d = self.diameter();
        if d > 0. {
            (area - PI * d * d / 4.).max(0.) / (PI * d * d)
        } else {
            f64::INFINITY
        }
    }
}

/// Areas and area-weighted position sums of the patches; merged with them.
type Patches = (
    SecondaryMap<SkeletonNodeID, f64>,
    SecondaryMap<SkeletonNodeID, Vector3D>,
);

struct Tables {
    penalties: SecondaryMap<SkeletonEdgeID, [f64; 6]>,
    boundaries: SecondaryMap<SkeletonEdgeID, Boundary>,
    patches: Patches,
}

/// Measures of every boundary, with crossings spaced evenly along their mesh edges.
fn boundary_measures(
    mesh: &Mesh<INPUT>,
    skeleton: &Skeleton,
) -> SecondaryMap<SkeletonEdgeID, Boundary> {
    let mut positions = HashMap::new();
    for (&edge, order) in &skeleton.surface.edge_order {
        let (a, b) = (
            mesh.position(mesh.root(edge)),
            mesh.position(mesh.toor(edge)),
        );
        for (rank, &id) in order.iter().enumerate() {
            let t = (rank + 1) as f64 / (order.len() + 1) as f64;
            positions.insert(id, a * (1. - t) + b * t);
        }
    }
    skeleton
        .edges
        .iter()
        .map(|(id, edge)| {
            let c = &edge.boundary.crossings;
            let p = |i: usize| positions[&c[i % c.len()]];
            let vector: Vector3D = (0..c.len()).map(|i| p(i).cross(&p(i + 1))).sum();
            let boundary = Boundary {
                length: (0..c.len()).map(|i| (p(i + 1) - p(i)).norm()).sum(),
                span: vector.norm() / 2.,
                centre: (0..c.len()).map(p).sum::<Vector3D>() / c.len() as f64,
            };
            (id, boundary)
        })
        .collect()
}

pub(crate) fn patch_measures(mesh: &Mesh<INPUT>, skeleton: &Skeleton) -> SkeletonResult<Patches> {
    let geometry = crate::geometry::SurfaceGeometry::build(mesh, skeleton)?;
    let mut areas: SecondaryMap<_, f64> = skeleton.nodes.keys().map(|n| (n, 0.)).collect();
    let mut moments: SecondaryMap<_, Vector3D> = skeleton
        .nodes
        .keys()
        .map(|n| (n, Vector3D::zeros()))
        .collect();
    for t in &geometry.triangles {
        let [a, b, c] = t.points;
        let area = 0.5 * (b - a).cross(&(c - a)).norm();
        areas[t.node] += area;
        moments[t.node] += area * (a + b + c) / 3.;
    }
    Ok((areas, moments))
}

/// The chain relabelled and reduced if that lowers its cost and stays valid; whether the whole
/// branch was dropped.
fn reduce_chain(
    mesh: &Mesh<INPUT>,
    labeled: &LabeledSkeleton,
    chain: &Chain,
    tables: &mut Tables,
) -> Option<(LabeledSkeleton, bool)> {
    let skeleton = &labeled.skeleton;
    let degree = skeleton.degrees();
    let n = chain.edges.len();
    let (start, end) = (chain.nodes[0], chain.nodes[n]);
    let (areas, moments) = &tables.patches;
    let area = |nodes: &[SkeletonNodeID]| nodes.iter().map(|&v| areas[v]).sum::<f64>();
    let measure = |i: usize| &tables.boundaries[chain.edges[i]];
    // Costs and ports in chain direction: a boundary stored against it swaps the signs.
    let along = |i: usize, p: usize| {
        tables.penalties[chain.edges[i]][if chain.forward[i] { p } else { p ^ 1 }]
    };
    let travel = |i: usize| {
        let port = Port::from_label(labeled.labels[chain.edges[i]]);
        (if chain.forward[i] {
            port
        } else {
            port.opposite()
        }) as usize
    };
    let leaf_start = degree[start] == 1 && start != end;
    let leaf_end = degree[end] == 1 && start != end;

    // Items: patch `m` (between boundaries `m - 1` and `m`) at `2m`, boundary `m` at `2m + 1`.
    // A patch is judged by its direction along the chain.
    let mut items = vec![[0.; 6]; 2 * n + 1];
    for m in 0..n {
        items[2 * m + 1] = std::array::from_fn(|p| along(m, p));
    }
    for m in 1..n {
        let (a, b) = (measure(m - 1), measure(m));
        let d = b.centre - a.centre;
        let loops = (a.diameter() + b.diameter()) / 2.;
        if loops > 0. {
            let w = DIRECTION * d.norm() / loops;
            items[2 * m] = label::label_costs(d).map(|c| w * c);
        }
    }
    // Toward a leaf, how much a boundary counts by what lies beyond it; a patch counts like the
    // boundary on its far side from the leaf.
    let fade = |beyond: &[SkeletonNodeID], i: usize| {
        ((measure(i).rise(area(beyond)) - FADE) / FADE).clamp(0., 1.)
    };
    let fade_start: Vec<f64> = (0..n).map(|i| fade(&chain.nodes[..=i], i)).collect();
    let fade_end: Vec<f64> = (0..n).map(|i| fade(&chain.nodes[i + 1..], i)).collect();
    let item_fade_start = |m: usize| fade_start[m / 2];
    let item_fade_end = |m: usize| fade_end[(m - 1) / 2];

    // Width over the mean loop diameter, penalized below `THIN_RATIO`.
    let thinness = |area: f64, loops: f64, degree: usize| {
        if loops <= 0. {
            return 0.;
        }
        let ratio = (2. * area / loops) / (loops / (degree as f64 * PI));
        THIN * (1. - ratio / THIN_RATIO).max(0.)
    };
    let patch = |j: usize, i: usize| {
        thinness(
            area(&chain.nodes[j + 1..=i]),
            measure(j).length + measure(i).length,
            2,
        )
    };
    // End patch: `node` with its boundaries off the chain, merged with the chain's `nodes` up to
    // kept `i`; `near` is the chain boundary at `node`.
    let end_patch = |node: SkeletonNodeID, nodes: &[SkeletonNodeID], near: usize, i: usize| {
        if degree[node] < 2 || start == end {
            return 0.;
        }
        let off: f64 = skeleton
            .incident(node)
            .filter(|(e, _)| !chain.edges.contains(e))
            .map(|(e, _)| tables.boundaries[e].length)
            .sum();
        let thin = thinness(
            areas[node] + area(nodes),
            off + measure(i).length,
            degree[node],
        );
        // The absorbed piece of chain as a tube: its length is its area over its mean perimeter.
        let perimeter = (measure(near).length + measure(i).length) / 2.;
        let d = measure(i).diameter();
        let absorbed = if perimeter > 0. && d > 0. {
            ABSORB * area(nodes) / perimeter / d
        } else {
            0.
        };
        thin + absorbed
    };
    // A leaf cap beyond kept `i` should lie in direction `port` from the boundary, as far as it
    // protrudes.
    let cap = |nodes: &[SkeletonNodeID], i: usize, port: usize| {
        let a = area(nodes);
        if a <= 0. {
            return 0.;
        }
        let moment: Vector3D = nodes.iter().map(|&v| moments[v]).sum();
        let prominence = measure(i).rise(a).min(1.);
        CAP * prominence * label::label_costs(moment / a - measure(i).centre)[port]
    };
    let first = |i: usize, q: usize| {
        if leaf_start {
            let dropped: f64 = (1..=2 * i).map(|m| items[m][q] * item_fade_start(m)).sum();
            dropped + cap(&chain.nodes[..=i], i, q ^ 1)
        } else {
            let dropped: f64 = (1..=2 * i).map(|m| items[m][q]).sum();
            dropped + end_patch(start, &chain.nodes[1..=i], 0, i)
        }
    };
    let last = |i: usize, q: usize| {
        if leaf_end {
            let dropped: f64 = (2 * i + 2..2 * n)
                .map(|m| items[m][q] * item_fade_end(m))
                .sum();
            dropped + cap(&chain.nodes[i + 1..], i, q)
        } else {
            let dropped: f64 = (2 * i + 2..2 * n).map(|m| items[m][q]).sum();
            dropped + end_patch(end, &chain.nodes[i + 1..n], n - 1, i)
        }
    };
    let current: f64 = (0..n)
        .map(|i| {
            let before = if i > 0 {
                patch(i - 1, i) + items[2 * i][travel(i - 1)].min(items[2 * i][travel(i)])
            } else {
                first(0, travel(0))
            };
            let after = if i == n - 1 { last(i, travel(i)) } else { 0. };
            items[2 * i + 1][travel(i)] + BOUNDARY + before + after
        })
        .sum();

    // No boundary kept: the branch merges into the junction, each boundary counting as fully
    // misaligned (90°) as far as it does not fade.
    let misaligned = label::label_costs(Vector3D::x())[2];
    let none = match (leaf_start, leaf_end) {
        (true, false) if degree[end] >= 3 => Some((end, fade_start.iter().sum::<f64>())),
        (false, true) if degree[start] >= 3 => Some((start, fade_end.iter().sum::<f64>())),
        _ => None,
    }
    .map(|(into, fades)| (into, fades * misaligned));

    let mut others = labeled.labels.clone();
    for &e in &chain.edges {
        others.remove(e);
    }
    let used = constraints::used_ports(skeleton, &others).ok()?;
    let masks: [bool; 64] = std::array::from_fn(|mask| {
        constraints::feasible(skeleton, &others, Some(([start, end], mask as u8)))
    });
    // Ends joined outside the chain would get parallel edges from a single kept boundary.
    let joined = skeleton.edges.iter().any(|(e, data)| {
        !chain.edges.contains(&e) && data.nodes.contains(&start) && data.nodes.contains(&end)
    });
    let ends = path::Ends {
        first: &first,
        last: &last,
        patch: &patch,
    };
    let select = |min_kept: usize| {
        if start != end {
            let limits = path::Constraints {
                start_ports: 63 & !used[start],
                end_ports: 63 & !used[end],
                masks,
            };
            path::select(&items, &limits, BOUNDARY, &ends, min_kept)
        } else {
            // A ring leaves and enters one node through two different ports; through a patch of
            // degree two, not straight on. One or two kept boundaries close a self-loop or
            // parallel edges.
            Port::ALL
                .into_iter()
                .filter(|p| used[start] & p.bit() == 0)
                .filter_map(|p| {
                    let straight = if degree[start] == 2 {
                        p.opposite().bit()
                    } else {
                        0
                    };
                    let limits = path::Constraints {
                        start_ports: p.bit(),
                        end_ports: 63 & !used[start] & !p.bit() & !straight,
                        masks,
                    };
                    path::select(&items, &limits, BOUNDARY, &ends, min_kept.max(3))
                })
                .min_by(|a, b| a.cost.total_cmp(&b.cost))
        }
    };
    let apply = |kept: &[usize], ports: &[Port], into: Option<SkeletonNodeID>| {
        let mut next = labeled.clone();
        let mut patches = tables.patches.clone();
        for (&i, &port) in kept.iter().zip(ports) {
            let port = if chain.forward[i] {
                port
            } else {
                port.opposite()
            };
            next.labels[chain.edges[i]] = port.label();
        }
        // A whole branch merges outward from the junction, one neighbour at a time.
        let order: Vec<usize> = if into == Some(end) {
            (0..n).rev().collect()
        } else {
            (0..n).collect()
        };
        for i in order.into_iter().filter(|i| !kept.contains(i)) {
            let edge = chain.edges[i];
            let nodes = next.skeleton.edges[edge].nodes;
            let keep = into.unwrap_or_else(|| {
                nodes
                    .into_iter()
                    .find(|v| *v == start || *v == end)
                    .unwrap_or(nodes[0])
            });
            merge(mesh, &mut next, edge, keep, &mut patches).ok()?;
        }
        (next.skeleton.validate(mesh).is_ok() && next.validate_labels().is_ok())
            .then_some((next, patches))
    };

    let threshold = current - 1e-9;
    let selections: Vec<_> = [1 + usize::from(joined), 2, 3]
        .into_iter()
        .filter_map(select)
        .collect();
    if let Some((into, cost)) = none {
        let best = selections.first().map_or(f64::INFINITY, |s| s.cost);
        if cost < threshold && cost < best {
            if let Some((next, patches)) = apply(&[], &[], Some(into)) {
                tables.patches = patches;
                return Some((next, true));
            }
        }
    }
    // A selection that fails validation (parallel edges) is retried with more kept boundaries.
    for s in selections.into_iter().take_while(|s| s.cost < threshold) {
        if let Some((next, patches)) = apply(&s.kept, &s.ports, None) {
            tables.patches = patches;
            return Some((next, false));
        }
    }
    None
}

/// [`dissolve`], merging the patch measures.
fn merge(
    mesh: &Mesh<INPUT>,
    labeled: &mut LabeledSkeleton,
    edge: SkeletonEdgeID,
    keep: SkeletonNodeID,
    (areas, moments): &mut Patches,
) -> SkeletonResult<()> {
    let other = labeled.skeleton.edges[edge].other(keep);
    dissolve(mesh, labeled, edge, keep)?;
    let area = areas.remove(other).unwrap_or(0.);
    areas[keep] += area;
    let moment = moments.remove(other).unwrap_or_else(Vector3D::zeros);
    moments[keep] += moment;
    Ok(())
}

/// Delete a boundary and merge its other patch into `keep` (merge lemma). Fails when the
/// graph would get parallel edges; labels may need repair afterwards.
fn dissolve(
    mesh: &Mesh<INPUT>,
    labeled: &mut LabeledSkeleton,
    edge: SkeletonEdgeID,
    keep: SkeletonNodeID,
) -> SkeletonResult<()> {
    let skeleton = &labeled.skeleton;
    let other = skeleton.edges.get(edge).ok_or("Missing edge")?.other(keep);
    if other == keep {
        return Err("Edge does not join `keep` to another node".into());
    }
    let mut neighbors = HashSet::new();
    for (e, data) in &skeleton.edges {
        if e == edge || !data.nodes.iter().any(|&n| n == keep || n == other) {
            continue;
        }
        let nodes = data.nodes.map(|n| if n == other { keep } else { n });
        if nodes[0] == nodes[1] {
            return Err("Merging would create a self-loop".into());
        }
        if !neighbors.insert(nodes[usize::from(nodes[0] == keep)]) {
            return Err("Merging would create parallel edges".into());
        }
    }
    crate::edit::merge(mesh, &mut labeled.skeleton, edge, keep);
    labeled.labels.remove(edge);
    Ok(())
}

#[cfg(test)]
mod tests;
