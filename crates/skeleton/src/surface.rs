//! Combinatorial cells cut out by disjoint boundary arcs on an immutable mesh.

use crate::{
    BoundaryCrossing, BoundaryCrossingID, BoundarySurface, PatchBoundary, Skeleton, SkeletonNodeID,
    SkeletonResult,
};
use dualcube_types::prelude::*;
use std::collections::{HashMap, HashSet};

pub(crate) mod cut;
pub(crate) mod edit;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Vertex {
    Input(VertID),
    Crossing(BoundaryCrossingID),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Gap {
    pub edge: EdgeID,
    pub index: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Side {
    Gap(Gap),
    Arc {
        start: BoundaryCrossingID,
        forward: bool,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct Cell {
    pub face: FaceID,
    pub vertices: Vec<Vertex>,
    pub sides: Vec<Side>,
}

pub(crate) struct Regions {
    pub cells: Vec<Cell>,
    pub owners: Vec<SkeletonNodeID>,
    pub adjacent: Vec<Vec<(usize, Gap)>>,
    pub arc_sides: HashMap<BoundaryCrossingID, [usize; 2]>,
}

pub(crate) fn canonical(mesh: &Mesh<INPUT>, edge: EdgeID) -> EdgeID {
    if edge < mesh.twin(edge) {
        edge
    } else {
        mesh.twin(edge)
    }
}

impl BoundarySurface {
    /// The canonical edge an occurrence lies on and its index in that edge's order.
    pub(crate) fn rank(&self, mesh: &Mesh<INPUT>, id: BoundaryCrossingID) -> (EdgeID, usize) {
        let edge = canonical(mesh, self.crossings[id].exit);
        let rank = self.edge_order[&edge].iter().position(|&c| c == id);
        (edge, rank.unwrap())
    }

    /// Add an occurrence at `index` of its edge's order.
    pub(crate) fn insert(
        &mut self,
        mesh: &Mesh<INPUT>,
        exit: EdgeID,
        index: usize,
    ) -> BoundaryCrossingID {
        let id = self.crossings.insert(BoundaryCrossing { exit });
        self.edge_order
            .entry(canonical(mesh, exit))
            .or_default()
            .insert(index, id);
        id
    }

    pub(crate) fn remove(&mut self, mesh: &Mesh<INPUT>, id: BoundaryCrossingID) {
        let edge = canonical(mesh, self.crossings.remove(id).unwrap().exit);
        self.unorder(edge, id);
    }

    /// Take an occurrence out of its edge's order, keeping its data.
    pub(crate) fn unorder(&mut self, edge: EdgeID, id: BoundaryCrossingID) {
        let order = self.edge_order.get_mut(&edge).unwrap();
        order.retain(|&c| c != id);
        if order.is_empty() {
            self.edge_order.remove(&edge);
        }
    }

    /// Uniformly spaced point of an occurrence on its edge.
    pub(crate) fn position(&self, mesh: &Mesh<INPUT>, id: BoundaryCrossingID) -> Vector3D {
        let (edge, rank) = self.rank(mesh, id);
        let t = (rank + 1) as f64 / (self.edge_order[&edge].len() + 1) as f64;
        mesh.position(mesh.root(edge)) * (1. - t) + mesh.position(mesh.toor(edge)) * t
    }

    /// [`Self::position`] of every occurrence at once.
    pub(crate) fn positions(&self, mesh: &Mesh<INPUT>) -> HashMap<BoundaryCrossingID, Vector3D> {
        let mut positions = HashMap::new();
        for (&edge, order) in &self.edge_order {
            let (a, b) = (
                mesh.position(mesh.root(edge)),
                mesh.position(mesh.toor(edge)),
            );
            for (rank, &id) in order.iter().enumerate() {
                let t = (rank + 1) as f64 / (order.len() + 1) as f64;
                positions.insert(id, a * (1. - t) + b * t);
            }
        }
        positions
    }

    /// Midpoint of the edge an occurrence crosses.
    pub(crate) fn midpoint(&self, mesh: &Mesh<INPUT>, id: BoundaryCrossingID) -> Vector3D {
        let exit = self.crossings[id].exit;
        (mesh.position(mesh.root(exit)) + mesh.position(mesh.toor(exit))) / 2.
    }
}

impl PatchBoundary {
    /// Length of the closed polyline through the midpoints of the crossed edges.
    pub(crate) fn length(&self, mesh: &Mesh<INPUT>, surface: &BoundarySurface) -> f64 {
        let points: Vec<_> = self
            .crossings
            .iter()
            .map(|&c| surface.midpoint(mesh, c))
            .collect();
        (0..points.len())
            .map(|i| (points[(i + 1) % points.len()] - points[i]).norm())
            .sum()
    }

    /// Mean of the midpoints of the crossed edges.
    pub(crate) fn centroid(&self, mesh: &Mesh<INPUT>, surface: &BoundarySurface) -> Vector3D {
        let sum: Vector3D = self
            .crossings
            .iter()
            .map(|&c| surface.midpoint(mesh, c))
            .sum();
        sum / self.crossings.len() as f64
    }
}

impl Regions {
    pub fn build(mesh: &Mesh<INPUT>, skeleton: &Skeleton) -> SkeletonResult<Self> {
        crate::validate::input_genus(mesh)?;
        if skeleton.nodes.is_empty()
            || skeleton
                .nodes
                .values()
                .any(|n| !n.position.iter().all(|x| x.is_finite()))
        {
            return Err("Missing skeleton nodes or non-finite embedding".into());
        }
        let valid_edges: HashSet<_> = mesh.edge_ids().into_iter().collect();
        let mut ordered = HashSet::new();
        for (&edge, order) in &skeleton.surface.edge_order {
            if !valid_edges.contains(&edge) || canonical(mesh, edge) != edge || order.is_empty() {
                return Err("Invalid canonical crossing order".into());
            }
            for &id in order {
                let crossing = skeleton
                    .surface
                    .crossings
                    .get(id)
                    .ok_or("Crossing order references a missing occurrence")?;
                if !valid_edges.contains(&crossing.exit)
                    || canonical(mesh, crossing.exit) != edge
                    || !ordered.insert(id)
                {
                    return Err(
                        "Crossing occurrence is duplicated or ordered on the wrong edge".into(),
                    );
                }
            }
        }
        if ordered.len() != skeleton.surface.crossings.len() {
            return Err("Some crossing occurrences have no edge order".into());
        }
        let mut loop_owners = HashMap::new();
        let mut arcs = HashMap::<FaceID, Vec<(BoundaryCrossingID, BoundaryCrossingID)>>::new();
        for edge in skeleton.edges.values() {
            if edge.nodes[0] == edge.nodes[1]
                || edge.nodes.iter().any(|n| !skeleton.nodes.contains_key(*n))
            {
                return Err("Boundary has invalid incident regions".into());
            }
            let path = &edge.boundary.crossings;
            if path.len() < 2 {
                return Err("Boundary loop has fewer than two crossings".into());
            }
            for (i, &start) in path.iter().enumerate() {
                if !ordered.contains(&start) || loop_owners.insert(start, edge.nodes).is_some() {
                    return Err("Boundary occurrence is missing or shared by loops".into());
                }
                let end = path[(i + 1) % path.len()];
                let b = skeleton
                    .surface
                    .crossings
                    .get(end)
                    .ok_or("Boundary references a missing occurrence")?;
                // All exits were checked against the input before querying their topology.
                let face = mesh.face(mesh.twin(skeleton.surface.crossings[start].exit));
                if !ordered.contains(&end) || mesh.face(b.exit) != face {
                    return Err("Boundary arc does not stay in one triangle".into());
                }
                arcs.entry(face).or_default().push((start, end));
            }
        }
        if loop_owners.len() != ordered.len() {
            return Err("Crossing occurrence has no boundary loop".into());
        }

        let mut cells = Vec::new();
        for face in mesh.face_ids() {
            let arcs = arcs.get(&face).map_or(&[][..], Vec::as_slice);
            cells.extend(split_face(mesh, &skeleton.surface.edge_order, face, arcs)?);
        }
        let mut gaps = HashMap::<Gap, Vec<usize>>::new();
        let mut arc_sides = HashMap::new();
        for (i, cell) in cells.iter().enumerate() {
            for side in &cell.sides {
                match *side {
                    Side::Gap(gap) => gaps.entry(gap).or_default().push(i),
                    Side::Arc { start, forward } => {
                        let pair = arc_sides.entry(start).or_insert([usize::MAX; 2]);
                        let side = usize::from(!forward);
                        if pair[side] != usize::MAX {
                            return Err("Duplicated boundary arc side".into());
                        }
                        pair[side] = i;
                    }
                }
            }
        }
        let mut adjacent = vec![Vec::new(); cells.len()];
        for (gap, pair) in gaps {
            if pair.len() != 2 {
                return Err("Unmatched surface interval".into());
            }
            adjacent[pair[0]].push((pair[1], gap));
            adjacent[pair[1]].push((pair[0], gap));
        }
        // Stable traversal also makes connector search reproducible.
        for list in &mut adjacent {
            list.sort_by(|(a, x), (b, y)| {
                a.cmp(b)
                    .then_with(|| x.edge.partial_cmp(&y.edge).unwrap())
                    .then(x.index.cmp(&y.index))
            });
        }
        let mut assigned = vec![None; cells.len()];
        let mut used_nodes = HashSet::new();
        for first in 0..cells.len() {
            if assigned[first].is_some() {
                continue;
            }
            let mut component = Vec::new();
            let mut visited = HashSet::new();
            let mut stack = vec![first];
            let mut owner = None;
            while let Some(i) = stack.pop() {
                if !visited.insert(i) {
                    continue;
                }
                component.push(i);
                for side in &cells[i].sides {
                    if let Side::Arc { start, forward } = *side {
                        let node = loop_owners[&start][usize::from(!forward)];
                        if owner.is_some_and(|old| old != node) {
                            return Err("Boundary sides disagree on region ownership".into());
                        }
                        owner = Some(node);
                    }
                }
                stack.extend(adjacent[i].iter().map(|(j, _)| *j));
            }
            let owner = owner
                .or_else(|| {
                    (skeleton.nodes.len() == 1 && skeleton.edges.is_empty())
                        .then(|| skeleton.nodes.keys().next().unwrap())
                })
                .ok_or("Surface region has no skeleton node")?;
            if !used_nodes.insert(owner) {
                return Err("Skeleton node describes disconnected surface regions".into());
            }
            let mut vertices = HashSet::new();
            let mut intervals = HashSet::new();
            let mut boundary_arcs = HashSet::new();
            for &i in &component {
                assigned[i] = Some(owner);
                vertices.extend(cells[i].vertices.iter().copied());
                for side in &cells[i].sides {
                    match *side {
                        Side::Gap(gap) => {
                            intervals.insert(gap);
                        }
                        Side::Arc { start, .. } => {
                            boundary_arcs.insert(start);
                        }
                    }
                }
            }
            let boundaries = skeleton.incident(owner).count();
            let chi = vertices.len() as i64 - (intervals.len() + boundary_arcs.len()) as i64
                + component.len() as i64;
            if chi != 2 - boundaries as i64 {
                return Err(format!("Region {owner:?} is not genus zero (Euler characteristic {chi}, {boundaries} boundaries)").into());
            }
        }
        if used_nodes.len() != skeleton.nodes.len() {
            return Err("Skeleton node has no surface region".into());
        }
        Ok(Self {
            cells,
            owners: assigned.into_iter().map(Option::unwrap).collect(),
            adjacent,
            arc_sides,
        })
    }
}

/// A triangle's cells cut out by boundary arcs `(start, end)` lying in it.
pub(crate) fn split_face(
    mesh: &Mesh<INPUT>,
    edge_order: &HashMap<EdgeID, Vec<BoundaryCrossingID>>,
    face: FaceID,
    arcs: &[(BoundaryCrossingID, BoundaryCrossingID)],
) -> SkeletonResult<Vec<Cell>> {
    let mut perimeter = Cell {
        face,
        vertices: Vec::new(),
        sides: Vec::new(),
    };
    for edge in mesh.edges(face) {
        let key = canonical(mesh, edge);
        let order = edge_order.get(&key).map_or(&[][..], Vec::as_slice);
        perimeter.vertices.push(Vertex::Input(mesh.root(edge)));
        if edge == key {
            perimeter
                .vertices
                .extend(order.iter().copied().map(Vertex::Crossing));
        } else {
            perimeter
                .vertices
                .extend(order.iter().rev().copied().map(Vertex::Crossing));
        }
        perimeter.sides.extend((0..=order.len()).map(|i| {
            Side::Gap(Gap {
                edge: key,
                index: if edge == key { i } else { order.len() - i },
            })
        }));
    }
    let mut parts = vec![perimeter];
    for &(start, end) in arcs {
        let mut found = None;
        for (index, cell) in parts.iter().enumerate() {
            if let (Some(a), Some(b)) = (
                cell.vertices
                    .iter()
                    .position(|v| *v == Vertex::Crossing(start)),
                cell.vertices
                    .iter()
                    .position(|v| *v == Vertex::Crossing(end)),
            ) {
                found = Some((index, a, b));
                break;
            }
        }
        let (index, a, b) = found.ok_or("Boundary arcs cross inside a triangle")?;
        let cell = parts.remove(index);
        let cut = |from: usize, to: usize, forward: bool| {
            let mut vertices = vec![cell.vertices[from]];
            let mut sides = Vec::new();
            let mut i = from;
            while i != to {
                sides.push(cell.sides[i]);
                i = (i + 1) % cell.vertices.len();
                vertices.push(cell.vertices[i]);
            }
            sides.push(Side::Arc { start, forward });
            Cell {
                face,
                vertices,
                sides,
            }
        };
        // Digons are valid: a returning arc may enclose no input vertices.
        parts.push(cut(a, b, false));
        parts.push(cut(b, a, true));
    }
    Ok(parts)
}

/// Representative geometry for initial label costs, not a region centroid.
pub(crate) fn position(
    mesh: &Mesh<INPUT>,
    skeleton: &Skeleton,
    id: BoundaryCrossingID,
) -> Vector3D {
    // TODO: Optimize crossing placement using surface geometry and label costs.
    skeleton.surface.position(mesh, id)
}

#[cfg(test)]
pub(crate) mod tests;
