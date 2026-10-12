//! Temporary ownership used only to construct extraction's boundary loops.

use crate::{SkeletonEdgeID, SkeletonNodeID, SkeletonResult};
use dualcube_types::prelude::*;
use slotmap::SlotMap;

#[derive(Clone, Debug, Default)]
pub(super) struct Partition {
    pub nodes: SlotMap<SkeletonNodeID, Patch>,
    pub edges: SlotMap<SkeletonEdgeID, PartitionEdge>,
}

#[derive(Clone, Debug)]
pub(super) struct Patch {
    pub position: Vector3D,
    pub vertices: Vec<VertID>,
}

#[derive(Clone, Debug)]
pub(super) struct PartitionEdge {
    pub nodes: [SkeletonNodeID; 2],
    pub boundary: Boundary,
}

#[derive(Clone, Debug, Default)]
pub(super) struct Boundary {
    pub crossed_edges: Vec<EdgeID>,
}

impl Partition {
    pub fn diagnostic(&self) -> super::surgery::DiagnosticGraph {
        super::surgery::DiagnosticGraph {
            positions: self
                .nodes
                .iter()
                .map(|(id, node)| (id, node.position))
                .collect(),
            edges: self.edges.values().map(|edge| edge.nodes).collect(),
        }
    }

    /// Validate extraction's provenance, then discard ownership at the public handoff.
    pub fn to_skeleton(&self, mesh: &Mesh<INPUT>) -> SkeletonResult<crate::Skeleton> {
        self.validate(mesh)?;
        let owners: HashMap<_, _> = self
            .nodes
            .iter()
            .flat_map(|(id, node)| node.vertices.iter().map(move |&v| (v, id)))
            .collect();
        let mut skeleton = crate::Skeleton::default();
        let nodes: HashMap<_, _> = self
            .nodes
            .iter()
            .map(|(id, node)| {
                (
                    id,
                    skeleton.nodes.insert(crate::SkeletonNode {
                        position: node.position,
                    }),
                )
            })
            .collect();
        for edge in self.edges.values() {
            let mut cycle = edge.boundary.crossed_edges.clone();
            let mut exits = oriented_exits(mesh, &cycle)?;
            // At an exit through an oriented face edge, its tip lies left of the curve.
            if owners[&mesh.toor(exits[0])] != edge.nodes[0] {
                cycle.reverse();
                exits = oriented_exits(mesh, &cycle)?;
            }
            let mut crossings = Vec::with_capacity(exits.len());
            for exit in exits {
                if owners[&mesh.toor(exit)] != edge.nodes[0]
                    || owners[&mesh.root(exit)] != edge.nodes[1]
                {
                    return Err("Boundary orientation disagrees with its incident patches".into());
                }
                let id = skeleton
                    .surface
                    .crossings
                    .insert(crate::BoundaryCrossing { exit });
                let order = skeleton
                    .surface
                    .edge_order
                    .entry(canonical(mesh, exit))
                    .or_default();
                // Extraction places at most one crossing on each geometric mesh edge.
                if !order.is_empty() {
                    return Err("Extraction boundaries share a mesh-edge crossing".into());
                }
                order.push(id);
                crossings.push(id);
            }
            skeleton.edges.insert(crate::SkeletonEdge {
                nodes: edge.nodes.map(|id| nodes[&id]),
                boundary: crate::PatchBoundary { crossings },
            });
        }
        Ok(skeleton)
    }
}

fn oriented_exits(mesh: &Mesh<INPUT>, cycle: &[EdgeID]) -> SkeletonResult<Vec<EdgeID>> {
    if cycle.len() < 3 {
        return Err("Extraction boundary has fewer than three crossings".into());
    }
    let exits: Vec<_> = (0..cycle.len())
        .map(|i| {
            let previous = cycle[(i + cycle.len() - 1) % cycle.len()];
            [cycle[i], mesh.twin(cycle[i])]
                .into_iter()
                .find(|&exit| {
                    mesh.face(exit) == mesh.face(previous)
                        || mesh.face(exit) == mesh.face(mesh.twin(previous))
                })
                .ok_or_else(|| "Consecutive boundary crossings do not share a face".into())
        })
        .collect::<SkeletonResult<_>>()?;
    for i in 0..exits.len() {
        if mesh.face(mesh.twin(exits[i])) != mesh.face(exits[(i + 1) % exits.len()]) {
            return Err("Boundary traversal does not continue across twin half-edges".into());
        }
    }
    Ok(exits)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Hash)]
enum CutVertex {
    Original(VertID),
    Crossing(EdgeID),
}

#[derive(Default)]
struct PatchCells {
    vertices: HashSet<CutVertex>,
    edges: HashSet<[CutVertex; 2]>,
    faces: usize,
}

type Boundaries = HashMap<[SkeletonNodeID; 2], Vec<Vec<EdgeID>>>;

fn canonical(mesh: &Mesh<INPUT>, edge: EdgeID) -> EdgeID {
    if edge < mesh.twin(edge) {
        edge
    } else {
        mesh.twin(edge)
    }
}
fn pair(a: SkeletonNodeID, b: SkeletonNodeID) -> [SkeletonNodeID; 2] {
    if a < b { [a, b] } else { [b, a] }
}

fn analyze(skeleton: &Partition, mesh: &Mesh<INPUT>) -> SkeletonResult<Boundaries> {
    if skeleton.nodes.is_empty() {
        return Err("Skeleton is empty".into());
    }
    let vertices: HashSet<_> = mesh.vert_ids().into_iter().collect();
    let mut owners = HashMap::new();
    for (id, node) in &skeleton.nodes {
        if node.vertices.is_empty() || !node.position.iter().all(|x| x.is_finite()) {
            return Err(format!("Patch {id:?} is empty or has a non-finite embedding").into());
        }
        for &v in &node.vertices {
            if !vertices.contains(&v) || owners.insert(v, id).is_some() {
                return Err("Patch vertices must form a partition of the input vertices".into());
            }
        }
    }
    if owners.len() != vertices.len() {
        return Err("Some input vertices have no patch owner".into());
    }
    for (id, node) in &skeleton.nodes {
        let mut seen = HashSet::new();
        let mut stack = vec![node.vertices[0]];
        while let Some(v) = stack.pop() {
            if seen.insert(v) {
                stack.extend(mesh.neighbors(v).filter(|v| owners[v] == id));
            }
        }
        if seen.len() != node.vertices.len() {
            return Err(format!("Patch {id:?} is disconnected").into());
        }
    }
    let mut cells: HashMap<SkeletonNodeID, PatchCells> = HashMap::new();
    let mut walks: HashMap<[SkeletonNodeID; 2], HashMap<EdgeID, Vec<EdgeID>>> = HashMap::new();
    for face in mesh.face_ids() {
        let edges: Vec<_> = mesh.edges(face).collect();
        if edges.len() != 3 {
            return Err("Patch validation requires triangles".into());
        }
        let verts: Vec<_> = edges.iter().map(|&e| mesh.root(e)).collect();
        let labels: Vec<_> = verts.iter().map(|v| owners[v]).collect();
        let distinct: HashSet<_> = labels.iter().copied().collect();
        if distinct.len() > 2 {
            return Err(format!("Triple junction in face {face:?}").into());
        }
        for &label in &distinct {
            let mut polygon = Vec::new();
            for i in 0..3 {
                if labels[i] == label {
                    polygon.push(CutVertex::Original(verts[i]));
                }
                if labels[i] != labels[(i + 1) % 3]
                    && (labels[i] == label || labels[(i + 1) % 3] == label)
                {
                    polygon.push(CutVertex::Crossing(canonical(mesh, edges[i])));
                }
            }
            let c = cells.entry(label).or_default();
            c.faces += 1;
            c.vertices.extend(&polygon);
            for i in 0..polygon.len() {
                let mut edge = [polygon[i], polygon[(i + 1) % polygon.len()]];
                edge.sort_by(|a, b| a.partial_cmp(b).unwrap());
                c.edges.insert(edge);
            }
        }
        if distinct.len() == 2 {
            let crossing: Vec<_> = (0..3)
                .filter(|&i| labels[i] != labels[(i + 1) % 3])
                .map(|i| canonical(mesh, edges[i]))
                .collect();
            let mut labels = distinct.into_iter();
            let p = pair(labels.next().unwrap(), labels.next().unwrap());
            let adjacency = walks.entry(p).or_default();
            adjacency.entry(crossing[0]).or_default().push(crossing[1]);
            adjacency.entry(crossing[1]).or_default().push(crossing[0]);
        }
    }
    let mut boundaries = HashMap::new();
    let mut boundary_counts = HashMap::<SkeletonNodeID, usize>::new();
    for (p, adjacency) in walks {
        if adjacency.values().any(|v| v.len() != 2 || v[0] == v[1]) {
            return Err("Patch boundary is not a simple closed curve".into());
        }
        let mut pending: HashSet<_> = adjacency.keys().copied().collect();
        let mut loops = Vec::new();
        while let Some(&start) = pending.iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
            let mut cycle = vec![start];
            pending.remove(&start);
            let mut previous = start;
            let mut current = *adjacency[&start]
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            while current != start {
                if !pending.remove(&current) {
                    return Err("Patch boundary repeats an edge".into());
                }
                cycle.push(current);
                let neighbors = &adjacency[&current];
                let next = if neighbors[0] == previous {
                    neighbors[1]
                } else {
                    neighbors[0]
                };
                previous = current;
                current = next;
            }
            loops.push(cycle);
        }
        for node in p {
            *boundary_counts.entry(node).or_default() += loops.len();
        }
        boundaries.insert(p, loops);
    }
    for (node, c) in cells {
        let chi = c.vertices.len() as i64 - c.edges.len() as i64 + c.faces as i64;
        let holes = boundary_counts.get(&node).copied().unwrap_or(0) as i64;
        if chi != 2 - holes {
            return Err(format!(
                "Patch {node:?} is not genus zero (Euler characteristic {chi}, {holes} boundaries)"
            )
            .into());
        }
    }
    Ok(boundaries)
}

pub(super) fn assign_boundaries(
    skeleton: &mut Partition,
    mesh: &Mesh<INPUT>,
) -> SkeletonResult<()> {
    let mut expected = analyze(skeleton, mesh)?;
    for (_, edge) in &mut skeleton.edges {
        let candidates = expected
            .get_mut(&pair(edge.nodes[0], edge.nodes[1]))
            .ok_or("Skeleton edge has no corresponding patch boundary")?;
        let old: Vec<_> = edge
            .boundary
            .crossed_edges
            .iter()
            .map(|&e| canonical(mesh, e))
            .collect();
        let index = candidates.iter().position(|cycle| same_cycle(&old, cycle));
        let cycle = if let Some(index) = index {
            candidates.swap_remove(index)
        } else {
            candidates
                .pop()
                .ok_or("Skeleton edge has no corresponding patch boundary")?
        };
        edge.boundary = Boundary {
            crossed_edges: cycle,
        };
    }
    if expected.values().any(|v| !v.is_empty()) {
        return Err("Patch boundary has no corresponding skeleton edge".into());
    }
    Ok(())
}

impl Partition {
    /// Validate ownership, patch topology, and graph/boundary correspondence.
    pub fn validate(&self, mesh: &Mesh<INPUT>) -> SkeletonResult<()> {
        crate::validate::input_genus(mesh)?;
        let mut expected = analyze(self, mesh)?;
        let valid_edges: HashSet<_> = mesh.edge_ids().into_iter().collect();
        let mut graph = HashMap::<SkeletonNodeID, Vec<SkeletonNodeID>>::new();
        for (_, edge) in &self.edges {
            let [a, b] = edge.nodes;
            if a == b || !self.nodes.contains_key(a) || !self.nodes.contains_key(b) {
                return Err("Skeleton edge has invalid endpoints".into());
            }
            graph.entry(a).or_default().push(b);
            graph.entry(b).or_default().push(a);
            if edge
                .boundary
                .crossed_edges
                .iter()
                .any(|e| !valid_edges.contains(e))
            {
                return Err("Boundary references an invalid mesh edge".into());
            }
            let cycle: Vec<_> = edge
                .boundary
                .crossed_edges
                .iter()
                .map(|&e| canonical(mesh, e))
                .collect();
            let candidates = expected
                .get_mut(&pair(a, b))
                .ok_or("Skeleton edge does not separate its endpoint patches")?;
            let index = candidates
                .iter()
                .position(|candidate| same_cycle(&cycle, candidate))
                .ok_or("Boundary is incomplete, duplicated, or incorrectly ordered")?;
            candidates.swap_remove(index);
        }
        if expected.values().any(|v| !v.is_empty()) {
            return Err("Patch boundary has no skeleton edge".into());
        }
        let mut seen = HashSet::new();
        let mut stack = vec![self.nodes.keys().next().unwrap()];
        while let Some(v) = stack.pop() {
            if seen.insert(v)
                && let Some(neighbors) = graph.get(&v)
            {
                stack.extend(neighbors);
            }
        }
        if seen.len() != self.nodes.len() {
            return Err("Skeleton graph is disconnected".into());
        }
        Ok(())
    }
}

fn same_cycle(a: &[EdgeID], b: &[EdgeID]) -> bool {
    if a.len() != b.len() || a.is_empty() {
        return false;
    }
    let Some(start) = b.iter().position(|e| *e == a[0]) else {
        return false;
    };
    (0..a.len()).all(|i| a[i] == b[(start + i) % b.len()])
        || (0..a.len()).all(|i| a[i] == b[(start + b.len() - i) % b.len()])
}

#[cfg(test)]
mod tests;
