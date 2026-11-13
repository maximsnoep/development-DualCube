//! Move all boundary loops at once, keeping the patch graph and the labels.
//!
//! Loops change only by isotopies. A loop passes over a mesh vertex whose corner it cuts: it
//! holds the crossing nearest to the vertex on every edge it crosses there, and afterwards holds
//! the nearest crossing on the vertex's other edges. A returning arc whose two crossings are
//! adjacent on their edge encloses nothing and is cancelled. Both moves are checked on crossing
//! order alone and change only the triangles around them, so the energy change is local:
//! restricted-label gain of the cells, loop length, and the mean ln(perimeter² / area) over
//! patches, which spreads stacked slivers apart. Improving moves are applied greedily until none
//! is left; stars are refreshed between rounds. The result is validated as a whole.

use super::fidelity::{self, CellGeometry, Measure, Star};
use super::redraw::cell_gain;
use crate::surface::{Regions, Side, Vertex, canonical, split_face};
use crate::{
    BoundaryCrossing, BoundaryCrossingID, LabeledSkeleton, Skeleton, SkeletonEdgeID, SkeletonNodeID,
};
use dualcube_types::prelude::*;
use slotmap::SecondaryMap;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Clone, Copy, Debug)]
pub struct RelaxParams {
    /// Cost of loop length × mean mesh edge length, relative to the area-weighted gain.
    pub smoothness: f64,
    /// Weight of the mean ln(perimeter² / area) over patches.
    pub thinness: f64,
    pub measure: Measure,
    /// Star refreshes.
    pub rounds: usize,
    /// Passes over all candidate moves per round, until none improves.
    pub sweeps: usize,
}

impl Default for RelaxParams {
    fn default() -> Self {
        Self {
            smoothness: 0.5,
            thinness: 0.05,
            measure: Measure::Alignment,
            rounds: 3,
            sweeps: 30,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Move {
    /// Pass the loop of a crossing nearest to a vertex over that vertex.
    Pass(VertID, BoundaryCrossingID),
    /// Cancel the returning arc that starts at a crossing.
    Cancel(BoundaryCrossingID),
}

pub(crate) enum Candidate {
    Push {
        lp: SkeletonEdgeID,
        side: usize,
        vertices: Vec<VertID>,
    },
    Cancel(BoundaryCrossingID),
}

/// A checked move: `removed` (consecutive in the loop) is replaced by crossings with the
/// `added` exits, each nearest to `vertex` on its edge.
pub(crate) struct Plan {
    lp: SkeletonEdgeID,
    removed: Vec<BoundaryCrossingID>,
    added: Vec<EdgeID>,
    vertex: Option<VertID>,
    before: BoundaryCrossingID,
    after: BoundaryCrossingID,
    len: usize,
    pub faces: Vec<FaceID>,
}

type Link = (BoundaryCrossingID, BoundaryCrossingID, BoundaryCrossingID);

pub(crate) struct Undo {
    orders: Vec<(EdgeID, Option<Vec<BoundaryCrossingID>>)>,
    links: Vec<Link>,
    removed: Vec<BoundaryCrossingID>,
    added: Vec<BoundaryCrossingID>,
    head: (SkeletonEdgeID, BoundaryCrossingID, usize),
    owner: Option<(VertID, SkeletonNodeID)>,
}

/// Per-patch area and gain, and per-loop length, of some triangles.
#[derive(Default, Debug)]
pub(crate) struct Local {
    patches: Vec<(SkeletonNodeID, [f64; 2])>,
    loops: Vec<(SkeletonEdgeID, f64)>,
}

#[cfg(test)]
impl Local {
    pub fn sums(&self) -> impl Iterator<Item = (SkeletonNodeID, [f64; 2])> + '_ {
        self.patches.iter().copied()
    }
    pub fn lengths(&self) -> impl Iterator<Item = (SkeletonEdgeID, f64)> + '_ {
        self.loops.iter().copied()
    }
    pub fn sum(&self, node: SkeletonNodeID) -> [f64; 2] {
        self.sums().find(|p| p.0 == node).map_or([0.; 2], |p| p.1)
    }
    pub fn length(&self, lp: SkeletonEdgeID) -> f64 {
        self.lengths().find(|l| l.0 == lp).map_or(0., |l| l.1)
    }
}

fn entry<K: PartialEq + Copy, V: Default>(list: &mut Vec<(K, V)>, key: K) -> &mut V {
    let i = match list.iter().position(|(k, _)| *k == key) {
        Some(i) => i,
        None => {
            list.push((key, V::default()));
            list.len() - 1
        }
    };
    &mut list[i].1
}

/// Loops as linked crossings on a working copy of the skeleton. Removed crossings stay in the
/// slotmap (for undo) but leave `member` and the edge orders.
pub(crate) struct Work<'a> {
    mesh: &'a Mesh<INPUT>,
    labeled: &'a LabeledSkeleton,
    pub skeleton: Skeleton,
    next: SecondaryMap<BoundaryCrossingID, BoundaryCrossingID>,
    prev: SecondaryMap<BoundaryCrossingID, BoundaryCrossingID>,
    member: SecondaryMap<BoundaryCrossingID, SkeletonEdgeID>,
    heads: SecondaryMap<SkeletonEdgeID, (BoundaryCrossingID, usize)>,
    owner: HashMap<VertID, SkeletonNodeID>,
}

impl<'a> Work<'a> {
    pub fn new(
        mesh: &'a Mesh<INPUT>,
        labeled: &'a LabeledSkeleton,
        skeleton: Skeleton,
    ) -> Option<Self> {
        let regions = Regions::build(mesh, &skeleton).ok()?;
        let mut owner = HashMap::new();
        for (cell, &node) in regions.cells.iter().zip(&regions.owners) {
            for v in &cell.vertices {
                if let Vertex::Input(v) = *v {
                    owner.insert(v, node);
                }
            }
        }
        let mut work = Self {
            mesh,
            labeled,
            skeleton,
            next: SecondaryMap::new(),
            prev: SecondaryMap::new(),
            member: SecondaryMap::new(),
            heads: SecondaryMap::new(),
            owner,
        };
        for (id, edge) in &work.skeleton.edges {
            let path = &edge.boundary.crossings;
            for (i, &c) in path.iter().enumerate() {
                let n = path[(i + 1) % path.len()];
                work.next.insert(c, n);
                work.prev.insert(n, c);
                work.member.insert(c, id);
            }
            work.heads.insert(id, (path[0], path.len()));
        }
        Some(work)
    }

    fn edge_of(&self, c: BoundaryCrossingID) -> EdgeID {
        canonical(self.mesh, self.skeleton.surface.crossings[c].exit)
    }

    fn incident(&self, edge: EdgeID, v: VertID) -> bool {
        self.mesh.root(edge) == v || self.mesh.toor(edge) == v
    }

    fn nearest(&self, edge: EdgeID, v: VertID) -> Option<BoundaryCrossingID> {
        let order = self.skeleton.surface.edge_order.get(&edge)?;
        if self.mesh.root(edge) == v {
            order.first().copied()
        } else {
            order.last().copied()
        }
    }

    pub fn plan(&self, mv: Move) -> Option<Plan> {
        match mv {
            Move::Pass(v, c) => self.plan_pass(v, c),
            Move::Cancel(c) => self.plan_cancel(c),
        }
    }

    fn plan_pass(&self, v: VertID, c: BoundaryCrossingID) -> Option<Plan> {
        let mesh = self.mesh;
        let lp = *self.member.get(c)?;
        let edge = self.edge_of(c);
        if !self.incident(edge, v) || self.nearest(edge, v) != Some(c) {
            return None;
        }
        // The vertex lies on one side of the loop.
        if !self.skeleton.edges[lp].nodes.contains(self.owner.get(&v)?) {
            return None;
        }
        let len = self.heads[lp].1;
        // The run of corner cuts around `v` containing `c`.
        let mut run = VecDeque::from([c]);
        let mut edges = HashSet::from([edge]);
        for forward in [true, false] {
            loop {
                if run.len() >= len {
                    return None;
                }
                let d = if forward {
                    self.next[*run.back().unwrap()]
                } else {
                    self.prev[*run.front().unwrap()]
                };
                let e = self.edge_of(d);
                if !self.incident(e, v) {
                    break;
                }
                // A further crossing on a vertex edge, or a returning arc, is not a corner cut.
                if self.nearest(e, v) != Some(d) || !edges.insert(e) {
                    return None;
                }
                if forward {
                    run.push_back(d);
                } else {
                    run.push_front(d);
                }
            }
        }
        let (s, t) = (*run.front().unwrap(), *run.back().unwrap());
        let crossings = &self.skeleton.surface.crossings;
        let entry = mesh.face(crossings[s].exit);
        let exit = mesh.face(mesh.twin(crossings[t].exit));
        let last = self.edge_of(t);
        // Walk around `v` from the entry triangle, away from the run, to the exit triangle.
        let (mut face, mut from) = (entry, self.edge_of(s));
        let mut added = Vec::new();
        let mut done = false;
        for _ in 0..=mesh.edges(v).count() {
            let h = mesh
                .edges(face)
                .find(|&h| canonical(mesh, h) != from && self.incident(canonical(mesh, h), v))?;
            let g = canonical(mesh, h);
            if face == exit && g == last {
                done = true;
                break;
            }
            if edges.contains(&g) {
                return None;
            }
            added.push(h);
            face = mesh.face(mesh.twin(h));
            from = g;
        }
        let len = len - run.len() + added.len();
        if !done || len < 2 {
            return None;
        }
        Some(Plan {
            lp,
            removed: run.into(),
            added,
            vertex: Some(v),
            before: self.prev[s],
            after: self.next[t],
            len,
            faces: mesh.edges(v).map(|h| mesh.face(h)).collect(),
        })
    }

    fn plan_cancel(&self, c: BoundaryCrossingID) -> Option<Plan> {
        let lp = *self.member.get(c)?;
        let d = self.next[c];
        let edge = self.edge_of(c);
        let len = self.heads[lp].1;
        if self.edge_of(d) != edge || len < 4 {
            return None;
        }
        let order = &self.skeleton.surface.edge_order[&edge];
        let rank = |x| order.iter().position(|&y| y == x).unwrap();
        if rank(c).abs_diff(rank(d)) != 1 {
            return None;
        }
        let exit = self.skeleton.surface.crossings[c].exit;
        Some(Plan {
            lp,
            removed: vec![c, d],
            added: Vec::new(),
            vertex: None,
            before: self.prev[c],
            after: self.next[d],
            len: len - 2,
            faces: vec![self.mesh.face(exit), self.mesh.face(self.mesh.twin(exit))],
        })
    }

    pub fn apply(&mut self, plan: &Plan) -> Undo {
        let mesh = self.mesh;
        let mut touched: Vec<EdgeID> = plan.removed.iter().map(|&c| self.edge_of(c)).collect();
        touched.extend(plan.added.iter().map(|&h| canonical(mesh, h)));
        touched.sort_by(|a, b| a.partial_cmp(b).unwrap());
        touched.dedup();
        let orders = &mut self.skeleton.surface.edge_order;
        let saved = touched
            .iter()
            .map(|&e| (e, orders.get(&e).cloned()))
            .collect();
        let mut links = vec![(plan.before, self.next[plan.before], self.prev[plan.before])];
        if plan.after != plan.before {
            links.push((plan.after, self.next[plan.after], self.prev[plan.after]));
        }
        let head = (plan.lp, self.heads[plan.lp].0, self.heads[plan.lp].1);
        for &c in &plan.removed {
            let e = canonical(mesh, self.skeleton.surface.crossings[c].exit);
            let order = orders.get_mut(&e).unwrap();
            order.retain(|&x| x != c);
            if order.is_empty() {
                orders.remove(&e);
            }
            self.member.remove(c);
        }
        let mut added = Vec::new();
        let mut chain = vec![plan.before];
        for &exit in &plan.added {
            let id = self
                .skeleton
                .surface
                .crossings
                .insert(BoundaryCrossing { exit });
            let e = canonical(mesh, exit);
            let order = orders.entry(e).or_default();
            if Some(mesh.root(e)) == plan.vertex {
                order.insert(0, id);
            } else {
                order.push(id);
            }
            self.member.insert(id, plan.lp);
            added.push(id);
            chain.push(id);
        }
        chain.push(plan.after);
        for pair in chain.windows(2) {
            self.next.insert(pair[0], pair[1]);
            self.prev.insert(pair[1], pair[0]);
        }
        self.heads[plan.lp] = (plan.before, plan.len);
        let owner = plan.vertex.map(|v| {
            let old = self.owner[&v];
            let [a, b] = self.skeleton.edges[plan.lp].nodes;
            self.owner.insert(v, if old == a { b } else { a });
            (v, old)
        });
        Undo {
            orders: saved,
            links,
            removed: plan.removed.clone(),
            added,
            head,
            owner,
        }
    }

    pub fn undo(&mut self, undo: Undo) {
        let orders = &mut self.skeleton.surface.edge_order;
        for (e, order) in undo.orders {
            match order {
                Some(order) => orders.insert(e, order),
                None => orders.remove(&e),
            };
        }
        for (c, n, p) in undo.links {
            self.next.insert(c, n);
            self.prev.insert(c, p);
        }
        for c in undo.removed {
            self.member.insert(c, undo.head.0);
        }
        for c in undo.added {
            self.skeleton.surface.crossings.remove(c);
            self.member.remove(c);
            self.next.remove(c);
            self.prev.remove(c);
        }
        self.heads[undo.head.0] = (undo.head.1, undo.head.2);
        if let Some((v, node)) = undo.owner {
            self.owner.insert(v, node);
        }
    }

    /// Area, gain and loop length of some triangles under fixed stars.
    pub fn local(
        &self,
        faces: &[FaceID],
        stars: &SecondaryMap<SkeletonNodeID, Star>,
        free: &SecondaryMap<SkeletonNodeID, u8>,
        measure: Measure,
    ) -> Option<Local> {
        let mesh = self.mesh;
        let surface = &self.skeleton.surface;
        let mut local = Local::default();
        for &face in faces {
            let mut arcs = Vec::new();
            let mut position = HashMap::new();
            for h in mesh.edges(face) {
                let e = canonical(mesh, h);
                let Some(order) = surface.edge_order.get(&e) else {
                    continue;
                };
                let (a, b) = (mesh.position(mesh.root(e)), mesh.position(mesh.toor(e)));
                for (rank, &c) in order.iter().enumerate() {
                    let t = (rank + 1) as f64 / (order.len() + 1) as f64;
                    position.insert(c, a * (1. - t) + b * t);
                    if surface.crossings[c].exit == h {
                        arcs.push((self.prev[c], c));
                    }
                }
            }
            let normal = mesh.normal(face);
            for cell in split_face(mesh, &surface.edge_order, face, &arcs).ok()? {
                let owner = cell
                    .sides
                    .iter()
                    .find_map(|side| match *side {
                        Side::Arc { start, forward } => Some(
                            self.skeleton.edges[self.member[start]].nodes[usize::from(!forward)],
                        ),
                        Side::Gap(_) => None,
                    })
                    .or_else(|| {
                        cell.vertices.iter().find_map(|v| match v {
                            Vertex::Input(v) => self.owner.get(v).copied(),
                            Vertex::Crossing(_) => None,
                        })
                    })?;
                let points = cell
                    .vertices
                    .iter()
                    .map(|v| match *v {
                        Vertex::Input(v) => Some(mesh.position(v)),
                        Vertex::Crossing(c) => position.get(&c).copied(),
                    })
                    .collect::<Option<Vec<_>>>()?;
                let mut area = 0.;
                for i in 1..points.len().saturating_sub(1) {
                    area += (points[i] - points[0])
                        .cross(&(points[i + 1] - points[0]))
                        .dot(&normal)
                        / 2.;
                }
                let geometry = CellGeometry {
                    owner,
                    area: area.max(0.),
                    centroid: points.iter().sum::<Vector3D>() / points.len() as f64,
                    normal,
                };
                let gain = cell_gain(&stars[owner], free[owner], self.labeled, &geometry, measure);
                let sums = entry(&mut local.patches, owner);
                sums[0] += geometry.area;
                sums[1] += geometry.area * gain;
            }
            for (a, b) in arcs {
                let length = (*position.get(&a)? - *position.get(&b)?).norm();
                *entry(&mut local.loops, self.member[b]) += length;
            }
        }
        Some(local)
    }

    /// The working state as a skeleton (removed crossings dropped).
    pub fn export(&self) -> Skeleton {
        let mut skeleton = self.skeleton.clone();
        skeleton
            .surface
            .crossings
            .retain(|c, _| self.member.contains_key(c));
        for (id, edge) in &mut skeleton.edges {
            let (head, len) = self.heads[id];
            let mut path = Vec::with_capacity(len);
            let mut c = head;
            for _ in 0..len {
                path.push(c);
                c = self.next[c];
            }
            edge.boundary.crossings = path;
        }
        skeleton
    }

    /// Every atomic move that may apply.
    #[cfg(test)]
    pub(crate) fn moves(&self) -> Vec<Move> {
        let mesh = self.mesh;
        let mut edges: Vec<EdgeID> = self.skeleton.surface.edge_order.keys().copied().collect();
        edges.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut moves = Vec::new();
        for e in edges {
            let Some(order) = self.skeleton.surface.edge_order.get(&e) else {
                continue;
            };
            moves.push(Move::Pass(mesh.root(e), order[0]));
            moves.push(Move::Pass(mesh.toor(e), *order.last().unwrap()));
            for &c in order {
                if self.edge_of(self.next[c]) == e {
                    moves.push(Move::Cancel(c));
                }
            }
        }
        moves
    }

    /// Vertices on `side` of a loop whose corner it cuts, in loop order, each once.
    fn front(&self, lp: SkeletonEdgeID, side: usize) -> Vec<VertID> {
        let node = self.skeleton.edges[lp].nodes[side];
        let (head, len) = self.heads[lp];
        let mut seen = HashSet::new();
        let mut front = Vec::new();
        let mut c = head;
        for _ in 0..len {
            let e = self.edge_of(c);
            for v in [self.mesh.root(e), self.mesh.toor(e)] {
                if self.nearest(e, v) == Some(c)
                    && self.owner.get(&v) == Some(&node)
                    && seen.insert(v)
                {
                    front.push(v);
                }
            }
            c = self.next[c];
        }
        front
    }

    /// Pass a loop over the given vertices of `side` in order; `None` (with nothing changed)
    /// if one of them cannot be passed.
    fn push(&mut self, lp: SkeletonEdgeID, side: usize, vertices: &[VertID]) -> Option<Vec<Undo>> {
        let node = self.skeleton.edges[lp].nodes[side];
        let mut undos = Vec::new();
        for &v in vertices {
            let plan = (self.owner.get(&v) == Some(&node))
                .then(|| {
                    self.mesh
                        .edges(v)
                        .map(|h| canonical(self.mesh, h))
                        .filter_map(|e| self.nearest(e, v))
                        .find(|c| self.member.get(*c) == Some(&lp))
                })
                .flatten()
                .and_then(|c| self.plan(Move::Pass(v, c)));
            let Some(plan) = plan else {
                for undo in undos.into_iter().rev() {
                    self.undo(undo);
                }
                return None;
            };
            undos.push(self.apply(&plan));
        }
        Some(undos)
    }

    /// Composite moves: each loop pushed over stretches of its front on either side, longest
    /// first (a single vertex usually makes a detour; a whole front moves the loop a ring).
    fn candidates(&self) -> Vec<Candidate> {
        let mut all = Vec::new();
        for lp in self.skeleton.edges.keys() {
            for side in 0..2 {
                let front = self.front(lp, side);
                let n = front.len();
                let mut k = n;
                while k > 0 {
                    let step = (k / 2).max(1);
                    let starts = if k == n { 1 } else { n.div_ceil(step) };
                    for i in 0..starts {
                        let vertices = (0..k).map(|j| front[(i * step + j) % n]).collect();
                        all.push((k, Candidate::Push { lp, side, vertices }));
                    }
                    k /= 2;
                }
            }
            let (head, len) = self.heads[lp];
            let mut c = head;
            for _ in 0..len {
                all.push((usize::MAX, Candidate::Cancel(c)));
                c = self.next[c];
            }
        }
        // Stable: loops keep their order within a size.
        all.sort_by_key(|(k, _)| std::cmp::Reverse(*k));
        all.into_iter().map(|(_, c)| c).collect()
    }
}

/// Running totals of the energy terms.
struct Totals {
    area: SecondaryMap<SkeletonNodeID, f64>,
    gain: SecondaryMap<SkeletonNodeID, f64>,
    length: SecondaryMap<SkeletonEdgeID, f64>,
    perimeter: SecondaryMap<SkeletonNodeID, f64>,
    surface: f64,
    scale: f64,
}

impl Totals {
    fn new(skeleton: &Skeleton, local: &Local, scale: f64) -> Self {
        let mut totals = Self {
            area: skeleton.nodes.keys().map(|n| (n, 0.)).collect(),
            gain: skeleton.nodes.keys().map(|n| (n, 0.)).collect(),
            length: skeleton.edges.keys().map(|e| (e, 0.)).collect(),
            perimeter: skeleton.nodes.keys().map(|n| (n, 0.)).collect(),
            surface: 0.,
            scale,
        };
        totals.update(skeleton, &Local::default(), local);
        totals.surface = totals.area.values().sum();
        totals
    }

    fn thin(&self, perimeter: f64, area: f64) -> f64 {
        let tiny = 1e-12 * self.surface.max(1e-300);
        (perimeter.max(tiny).powi(2) / area.max(tiny)).ln()
    }

    /// Energy change (lower is better) from replacing `before` by `after`.
    fn delta(
        &self,
        skeleton: &Skeleton,
        before: &Local,
        after: &Local,
        params: &RelaxParams,
    ) -> f64 {
        let mut patches: Vec<(SkeletonNodeID, [f64; 2])> = Vec::new();
        let (mut gain, mut length) = (0., 0.);
        for (sign, local) in [(1., after), (-1., before)] {
            for &(n, [a, g]) in &local.patches {
                entry(&mut patches, n)[0] += sign * a;
                gain += sign * g;
            }
            for &(e, l) in &local.loops {
                length += sign * l;
                for n in skeleton.edges[e].nodes {
                    entry(&mut patches, n)[1] += sign * l;
                }
            }
        }
        let thin: f64 = patches
            .iter()
            .map(|&(n, [da, dp])| {
                let (a, p) = (self.area[n], self.perimeter[n]);
                self.thin(p + dp, a + da) - self.thin(p, a)
            })
            .sum();
        let surface = self.surface.max(1e-300);
        -gain / surface
            + params.smoothness * self.scale * length / surface
            + params.thinness * thin / skeleton.nodes.len() as f64
    }

    fn update(&mut self, skeleton: &Skeleton, before: &Local, after: &Local) {
        for (sign, local) in [(1., after), (-1., before)] {
            for &(n, [a, g]) in &local.patches {
                self.area[n] += sign * a;
                self.gain[n] += sign * g;
            }
            for &(e, l) in &local.loops {
                self.length[e] += sign * l;
                for n in skeleton.edges[e].nodes {
                    self.perimeter[n] += sign * l;
                }
            }
        }
    }
}

/// Stars and free ports of the current geometry.
pub(crate) fn stars(
    mesh: &Mesh<INPUT>,
    labeled: &LabeledSkeleton,
    skeleton: &Skeleton,
) -> Option<(
    SecondaryMap<SkeletonNodeID, Star>,
    SecondaryMap<SkeletonNodeID, u8>,
)> {
    let regions = Regions::build(mesh, skeleton).ok()?;
    let positions = fidelity::crossing_positions(mesh, skeleton);
    let cells = fidelity::cells(mesh, &regions, &positions);
    let stars = fidelity::stars(skeleton, &cells, &fidelity::loops(skeleton, &positions));
    let used = crate::label::constraints::used_ports(skeleton, &labeled.labels).ok()?;
    let free = used.iter().map(|(n, &u)| (n, 63 & !u)).collect();
    Some((stars, free))
}

/// Relax every loop; `None` if nothing moved or the result does not validate.
pub(crate) fn relax(
    mesh: &Mesh<INPUT>,
    labeled: &LabeledSkeleton,
    params: &RelaxParams,
) -> Option<LabeledSkeleton> {
    if labeled.skeleton.edges.is_empty() {
        return None;
    }
    let faces = mesh.face_ids();
    let scale = mesh
        .edge_ids()
        .iter()
        .map(|&e| mesh.vector(e).norm())
        .sum::<f64>()
        / mesh.edge_ids().len().max(1) as f64;
    let mut current = labeled.skeleton.clone();
    let mut moved = false;
    for _ in 0..params.rounds.max(1) {
        let (stars, free) = stars(mesh, labeled, &current)?;
        let mut work = Work::new(mesh, labeled, current.clone())?;
        let all = work.local(&faces, &stars, &free, params.measure)?;
        let mut totals = Totals::new(&work.skeleton, &all, scale);
        let mut accepted = 0;
        for _ in 0..params.sweeps.max(1) {
            let mut changed = false;
            for candidate in work.candidates() {
                let faces: Vec<FaceID> = match &candidate {
                    Candidate::Push { vertices, .. } => {
                        let mut faces: Vec<_> =
                            vertices.iter().flat_map(|&v| mesh.faces(v)).collect();
                        faces.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        faces.dedup();
                        faces
                    }
                    Candidate::Cancel(c) => match work.plan(Move::Cancel(*c)) {
                        Some(plan) => plan.faces,
                        None => continue,
                    },
                };
                let Some(before) = work.local(&faces, &stars, &free, params.measure) else {
                    continue;
                };
                let undos = match &candidate {
                    Candidate::Push { lp, side, vertices } => work.push(*lp, *side, vertices),
                    Candidate::Cancel(c) => work
                        .plan(Move::Cancel(*c))
                        .map(|plan| vec![work.apply(&plan)]),
                };
                let Some(undos) = undos else {
                    continue;
                };
                match work.local(&faces, &stars, &free, params.measure) {
                    Some(after)
                        if totals.delta(&work.skeleton, &before, &after, params) < -1e-12 =>
                    {
                        totals.update(&work.skeleton, &before, &after);
                        accepted += 1;
                        changed = true;
                    }
                    _ => {
                        for undo in undos.into_iter().rev() {
                            work.undo(undo);
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
        if accepted == 0 {
            break;
        }
        let next = work.export();
        // Only validated states continue.
        if next.validate(mesh).is_err() {
            tracing::warn!("Relaxation produced an invalid skeleton; keeping the last valid one");
            break;
        }
        current = next;
        moved = true;
    }
    moved.then(|| LabeledSkeleton {
        skeleton: current,
        labels: labeled.labels.clone(),
    })
}
