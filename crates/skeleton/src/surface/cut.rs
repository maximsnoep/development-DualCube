//! New loops from vertex sides: a minimum cut gives every input vertex of a region a side, and
//! the intervals where the side changes are traced into one loop.
//!
//! Crossings of the region's other loops are pinned. Each interval whose ends get different
//! sides receives one crossing, and inside every cell the changes are paired so that runs of the
//! `false` side are cut off.

use crate::surface::{Gap, Regions, Side, Vertex};
use crate::{BoundaryCrossing, BoundaryCrossingID, Skeleton, SkeletonEdgeID, SkeletonNodeID};
use dualcube_types::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};

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
        skeleton.surface.remove(mesh, crossing);
    }
}

/// Replace a boundary's loop, keeping its orientation (`nodes[0]` on the left); `None` if
/// neither traversal direction validates.
pub(crate) fn replace(
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
    super::edit::reverse(mesh, &mut result, &mut boundary);
    result.edges[edge].boundary = boundary;
    result.validate(mesh).ok().map(|()| result)
}

/// Vertex sides within `owner`'s region by a minimum cut: `pinned` fixes crossings, and cutting
/// an interval costs `cutting(its ends, its length × its edge length)`. Islands not touching
/// their side's loops are flipped (a side without loops keeps its largest component).
pub(crate) fn cut_sides(
    mesh: &Mesh<INPUT>,
    union: &Regions,
    union_positions: &HashMap<BoundaryCrossingID, Vector3D>,
    owner: SkeletonNodeID,
    pinned: &HashMap<BoundaryCrossingID, bool>,
    cutting: &dyn Fn([Vertex; 2], f64) -> f64,
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
        for (j, v) in cell.vertices.iter().enumerate() {
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
            let weight = cutting(ends, span * mesh.vector(gap.edge).norm());
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

/// Minimum s-t cut by Dinic's algorithm on nonnegative real capacities.
pub(crate) struct Network {
    /// Per arc: head, residual capacity. Arc `i ^ 1` is the reverse of arc `i`.
    arcs: Vec<(usize, f64)>,
    out: Vec<Vec<usize>>,
}

impl Network {
    pub fn new(nodes: usize) -> Self {
        Self {
            arcs: Vec::new(),
            out: vec![Vec::new(); nodes],
        }
    }

    /// Capacity `forward` from a to b and `backward` from b to a.
    pub fn add(&mut self, a: usize, b: usize, forward: f64, backward: f64) {
        self.out[a].push(self.arcs.len());
        self.arcs.push((b, forward.max(0.)));
        self.out[b].push(self.arcs.len());
        self.arcs.push((a, backward.max(0.)));
    }

    /// Nodes on the source side of a minimum cut.
    pub fn source_side(&mut self, source: usize, sink: usize) -> Vec<bool> {
        const EPS: f64 = 1e-12;
        let n = self.out.len();
        loop {
            let mut level = vec![usize::MAX; n];
            level[source] = 0;
            let mut queue = VecDeque::from([source]);
            while let Some(a) = queue.pop_front() {
                for &arc in &self.out[a] {
                    let (b, capacity) = self.arcs[arc];
                    if capacity > EPS && level[b] == usize::MAX {
                        level[b] = level[a] + 1;
                        queue.push_back(b);
                    }
                }
            }
            if level[sink] == usize::MAX {
                return level.iter().map(|&l| l != usize::MAX).collect();
            }
            let mut next = vec![0; n];
            while self.augment(source, sink, &level, &mut next) > EPS {}
        }
    }

    /// One blocking-flow path by iterative depth-first search.
    fn augment(&mut self, source: usize, sink: usize, level: &[usize], next: &mut [usize]) -> f64 {
        const EPS: f64 = 1e-12;
        let mut path: Vec<usize> = Vec::new();
        let mut node = source;
        loop {
            if node == sink {
                let flow = path
                    .iter()
                    .map(|&a| self.arcs[a].1)
                    .fold(f64::INFINITY, f64::min);
                for &a in &path {
                    self.arcs[a].1 -= flow;
                    self.arcs[a ^ 1].1 += flow;
                }
                return flow;
            }
            let mut advanced = false;
            while next[node] < self.out[node].len() {
                let arc = self.out[node][next[node]];
                let (b, capacity) = self.arcs[arc];
                if capacity > EPS && level[b] == level[node] + 1 {
                    path.push(arc);
                    node = b;
                    advanced = true;
                    break;
                }
                next[node] += 1;
            }
            if !advanced {
                if node == source {
                    return 0.;
                }
                let arc = path.pop().unwrap();
                node = self.arcs[arc ^ 1].0;
                next[node] += 1;
            }
        }
    }
}
