//! Geometric boundary smoothing, for nicer pictures: every crossing slides along its mesh edge to
//! shorten its loop. Topology (which edges are crossed, and in which order) never changes; only
//! [`BoundarySurface::offsets`](crate::BoundarySurface::offsets) are set.

use crate::{BoundaryCrossingID, Skeleton};
use dualcube_types::prelude::*;

/// Sweeps of coordinate descent, stopping early once no crossing moves more than `TOLERANCE`.
const SWEEPS: usize = 500;
const TOLERANCE: f64 = 1e-6;
/// Smallest gap between neighbors on an edge (and to its ends), as a fraction of the uniform gap.
const GAP: f64 = 0.15;

struct Crossing {
    // The edge's root and direction (tip − root).
    root: Vector3D,
    direction: Vector3D,
    // Neighbors along the loop.
    loop_neighbors: [usize; 2],
    // Neighbors on the edge (root side, tip side), if any.
    edge_neighbors: [Option<usize>; 2],
    gap: f64,
}

/// Shorten every patch boundary by sliding its crossings along their edges, keeping their order.
pub fn run(mesh: &Mesh<INPUT>, skeleton: &mut Skeleton) {
    let surface = &skeleton.surface;
    let ids: Vec<BoundaryCrossingID> = surface.crossings.keys().collect();
    let index: HashMap<_, _> = ids.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    let mut t = vec![0.; ids.len()];
    let mut crossings: Vec<Option<Crossing>> = (0..ids.len()).map(|_| None).collect();
    for (&edge, order) in &surface.edge_order {
        let root = mesh.position(mesh.root(edge));
        let direction = mesh.position(mesh.toor(edge)) - root;
        let gap = GAP / (order.len() + 1) as f64;
        for (rank, (&id, offset)) in order.iter().zip(surface.parameters(order)).enumerate() {
            let i = index[&id];
            t[i] = offset;
            crossings[i] = Some(Crossing {
                root,
                direction,
                loop_neighbors: [i, i],
                edge_neighbors: [
                    rank.checked_sub(1).map(|r| index[&order[r]]),
                    order.get(rank + 1).map(|c| index[c]),
                ],
                gap,
            });
        }
    }
    for edge in skeleton.edges.values() {
        let c = &edge.boundary.crossings;
        for k in 0..c.len() {
            let previous = index[&c[(k + c.len() - 1) % c.len()]];
            let next = index[&c[(k + 1) % c.len()]];
            if let Some(crossing) = crossings[index[&c[k]]].as_mut() {
                crossing.loop_neighbors = [previous, next];
            }
        }
    }
    let point = |t: &[f64], i: usize| {
        let c = crossings[i].as_ref().unwrap();
        c.root + c.direction * t[i]
    };
    for _ in 0..SWEEPS {
        let mut moved: f64 = 0.;
        for i in 0..ids.len() {
            let Some(c) = &crossings[i] else { continue };
            let lo = c.edge_neighbors[0].map_or(0., |j| t[j]) + c.gap;
            let hi = c.edge_neighbors[1].map_or(1., |j| t[j]) - c.gap;
            if lo > hi {
                continue;
            }
            let [a, b] = c.loop_neighbors.map(|j| point(&t, j));
            let best = shortest(c.root, c.direction, a, b).clamp(lo, hi);
            moved = moved.max((best - t[i]).abs());
            t[i] = best;
        }
        if moved < TOLERANCE {
            break;
        }
    }
    let offsets = &mut skeleton.surface.offsets;
    offsets.clear();
    for (i, &id) in ids.iter().enumerate() {
        if crossings[i].is_some() {
            offsets.insert(id, t[i]);
        }
    }
}

/// The parameter of the point on the line `root + s · direction` with the least distance to `a`
/// plus to `b`: where the line meets the segment between them once `b` is turned about the line
/// to the side opposite `a`.
fn shortest(root: Vector3D, direction: Vector3D, a: Vector3D, b: Vector3D) -> f64 {
    let length = direction.norm_squared();
    if length < 1e-24 {
        return 0.5;
    }
    let split = |p: Vector3D| {
        let along = (p - root).dot(&direction) / length;
        (along, (p - root - direction * along).norm())
    };
    let ((sa, ra), (sb, rb)) = (split(a), split(b));
    if ra + rb < 1e-24 {
        return (sa + sb) / 2.;
    }
    sa + (sb - sa) * ra / (ra + rb)
}
