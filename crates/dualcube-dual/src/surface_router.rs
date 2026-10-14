//! Routing through arrangement cells; mesh edges impose no single-visit restriction.

use crate::arrangement::{
    Arrangement,
    general::{self, Cells},
};
use crate::{Loop, LoopID, PropertyViolationError, SurfaceArc};
use dualcube_types::prelude::*;
use slotmap::SlotMap;
use std::collections::{BinaryHeap, HashMap};

pub struct SurfaceRouter {
    cells: Cells,
    pub regions: Vec<usize>,
    neighbors: Vec<Vec<(usize, usize)>>,
    centers: Vec<Vector3D>,
    portals: Vec<Vector3D>,
}

#[derive(Clone, Copy, Debug)]
pub enum RouteError {
    NoRoute,
    ExpansionLimit,
    Degenerate,
}

impl SurfaceRouter {
    pub fn new(
        mesh: &Mesh<INPUT>,
        loops: &SlotMap<LoopID, Loop>,
    ) -> Result<Self, PropertyViolationError> {
        Self::from_arrangement(mesh, loops, Arrangement::new(mesh, loops)?)
    }

    /// Share arrangement construction with the topology used to constrain this router.
    pub fn with_topology(
        mesh: &Mesh<INPUT>,
        loops: &SlotMap<LoopID, Loop>,
    ) -> Result<(Self, crate::partial::PartialTopology), PropertyViolationError> {
        let arrangement = Arrangement::new(mesh, loops)?;
        let topology = crate::partial::PartialTopology::from_arrangement(mesh, &arrangement)?;
        Ok((Self::from_arrangement(mesh, loops, arrangement)?, topology))
    }

    fn from_arrangement(
        mesh: &Mesh<INPUT>,
        loops: &SlotMap<LoopID, Loop>,
        mut old: Arrangement,
    ) -> Result<Self, PropertyViolationError> {
        let (components, _, _) = old.components(mesh)?;
        let (cells, regions) = if let Some(cells) = old.general.take() {
            (cells, components)
        } else {
            let cells = general::build(mesh, loops)?.general.unwrap();
            let regions = cells
                .cells
                .iter()
                .map(|c| {
                    old.locate(c.face, center(&c.points))
                        .map(|i| components[i])
                        .ok_or(PropertyViolationError::UnknownError)
                })
                .collect::<Result<Vec<_>, _>>()?;
            (cells, regions)
        };
        let mut neighbors = vec![vec![]; cells.cells.len()];
        for (i, link) in cells.links.iter().enumerate() {
            neighbors[link.cells[0]].push((link.cells[1], i));
            neighbors[link.cells[1]].push((link.cells[0], i));
        }
        let centers = cells
            .cells
            .iter()
            .map(|c| SurfaceArc::position(mesh, c.face, center(&c.points)))
            .collect();
        let portals = cells
            .links
            .iter()
            .map(|link| {
                let cell = &cells.cells[link.cells[0]];
                let side = link.sides[0];
                SurfaceArc::position(
                    mesh,
                    cell.face,
                    lerp(
                        cell.points[side],
                        cell.points[(side + 1) % cell.points.len()],
                        0.5,
                    ),
                )
            })
            .collect();
        Ok(Self {
            cells,
            regions,
            neighbors,
            centers,
            portals,
        })
    }

    /// Directed crossing intervals with the prescribed region on either side.
    pub fn anchors(&self, partner: LoopID, before: usize, after: usize) -> Vec<(usize, usize)> {
        let mut anchors = vec![];
        for (i, link) in self.cells.links.iter().enumerate() {
            if link.wall != Some(partner) {
                continue;
            }
            for from in 0..2 {
                if self.regions[link.cells[from]] == before
                    && self.regions[link.cells[1 - from]] == after
                {
                    anchors.push((i, from));
                }
            }
        }
        anchors
    }

    /// One cell path realizes the ordered crossings. Geometry and final topology are validated by the caller.
    pub fn route(
        &self,
        axis: Direction,
        crossings: &[LoopID],
        regions: &[usize],
        anchor: (usize, usize),
        cost: impl Fn(FaceID, Vector3D) -> f64,
    ) -> Result<Loop, RouteError> {
        let (first, side) = anchor;
        let start = self.cells.links[first].cells[1 - side];
        let end = self.cells.links[first].cells[side];
        let k = crossings.len();
        if k == 0 || regions.len() != k {
            return Err(RouteError::NoRoute);
        }
        let mut heap = BinaryHeap::new();
        let mut best = HashMap::new();
        let mut previous = HashMap::new();
        let initial = (start, 1usize);
        best.insert(initial, 0.);
        heap.push((std::cmp::Reverse(OrderedFloat(0.)), initial));
        let centers = &self.centers;
        let mut expansions = 0;
        while let Some((std::cmp::Reverse(distance), (cell, progress))) = heap.pop() {
            if distance.0 > best[&(cell, progress)] {
                continue;
            }
            if (cell, progress) == (end, k) {
                let mut path = vec![];
                let mut state = (cell, progress);
                while state != initial {
                    let (prev, link, from) = previous[&state];
                    path.push((link, from));
                    state = prev;
                }
                path.reverse();
                path.insert(0, anchor);
                return self.curve(axis, &path);
            }
            expansions += 1;
            if expansions > 1_000_000 {
                return Err(RouteError::ExpansionLimit);
            }
            for &(next, link_id) in &self.neighbors[cell] {
                let link = &self.cells.links[link_id];
                let next_progress = if let Some(partner) = link.wall {
                    if crossings.get(progress) != Some(&partner) {
                        continue;
                    }
                    progress + 1
                } else {
                    progress
                };
                if self.regions[next] != regions[(next_progress + k - 1) % k] {
                    continue;
                }
                let from = usize::from(link.cells[1] == cell);
                let c = &self.cells.cells[cell];
                let portal = self.portals[link_id];
                let step = cost(c.face, portal - centers[cell])
                    + cost(self.cells.cells[next].face, centers[next] - portal);
                if !step.is_finite() || step <= 0. {
                    continue;
                }
                let d = distance.0 + step;
                let state = (next, next_progress);
                if best.get(&state).is_none_or(|&old| d < old) {
                    best.insert(state, d);
                    previous.insert(state, ((cell, progress), link_id, from));
                    heap.push((std::cmp::Reverse(OrderedFloat(d)), state));
                }
            }
        }
        Err(RouteError::NoRoute)
    }

    fn curve(&self, axis: Direction, path: &[(usize, usize)]) -> Result<Loop, RouteError> {
        // TODO: Choose lane order and crossing positions jointly with the required region topology.
        let mut counts = HashMap::new();
        for &(id, _) in path {
            *counts.entry(id).or_insert(0usize) += 1;
        }
        let mut used = HashMap::new();
        let mut portals = vec![];
        for &(id, from) in path {
            let link = &self.cells.links[id];
            let a = &self.cells.cells[link.cells[0]];
            let b = &self.cells.cells[link.cells[1]];
            let rank = used.entry(id).or_insert(0usize);
            *rank += 1;
            let t = *rank as f64 / (counts[&id] + 1) as f64;
            let p = lerp(
                a.points[link.sides[0]],
                a.points[(link.sides[0] + 1) % a.points.len()],
                t,
            );
            let q = lerp(
                b.points[link.sides[1]],
                b.points[(link.sides[1] + 1) % b.points.len()],
                1. - t,
            );
            let mut points = [p, q];
            if a.face == b.face {
                let ca = center(&a.points);
                let cb = center(&b.points);
                let d = [cb[0] - ca[0], cb[1] - ca[1]];
                let mut scale = 0.025;
                let mut found = false;
                for _ in 0..40 {
                    let u = [p[0] - scale * d[0], p[1] - scale * d[1]];
                    let v = [p[0] + scale * d[0], p[1] + scale * d[1]];
                    if general::contains(&a.points, u) && general::contains(&b.points, v) {
                        points = [u, v];
                        found = true;
                        break;
                    }
                    scale *= 0.5;
                }
                if !found {
                    return Err(RouteError::Degenerate);
                }
            }
            portals.push((
                link.cells[from],
                link.cells[1 - from],
                points[from],
                points[1 - from],
            ));
        }
        let mut arcs = vec![];
        for i in 0..portals.len() {
            let (before, after, from, to) = portals[i];
            if self.cells.cells[before].face == self.cells.cells[after].face {
                arcs.push(SurfaceArc {
                    face: self.cells.cells[before].face,
                    from,
                    to,
                });
            }
            let next = portals[(i + 1) % portals.len()];
            if after != next.0 {
                return Err(RouteError::Degenerate);
            }
            arcs.push(SurfaceArc {
                face: self.cells.cells[after].face,
                from: to,
                to: next.2,
            });
        }
        Ok(Loop::from_surface(axis, arcs))
    }
}

fn center(points: &[[f64; 2]]) -> [f64; 2] {
    let n = points.len() as f64;
    [
        points.iter().map(|p| p[0]).sum::<f64>() / n,
        points.iter().map(|p| p[1]).sum::<f64>() / n,
    ]
}
fn lerp(a: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
}
