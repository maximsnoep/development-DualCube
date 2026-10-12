//! Checks on the surface cut along patch boundaries.

use crate::{PatchBoundary, Skeleton, SkeletonNodeID, SkeletonResult};
use dualcube_types::prelude::*;

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

fn analyze(skeleton: &Skeleton, mesh: &Mesh<INPUT>) -> SkeletonResult<Boundaries> {
    if skeleton.nodes.is_empty() {
        return Err("Skeleton is empty".into());
    }
    let vertices: HashSet<_> = mesh.vert_ids().into_iter().collect();
    let mut owners = HashMap::new();
    for (id, node) in &skeleton.nodes {
        if node.patch_vertices.is_empty() || !node.position.iter().all(|x| x.is_finite()) {
            return Err(format!("Patch {id:?} is empty or has a non-finite embedding").into());
        }
        for &v in &node.patch_vertices {
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
        let mut stack = vec![node.patch_vertices[0]];
        while let Some(v) = stack.pop() {
            if seen.insert(v) {
                stack.extend(mesh.neighbors(v).filter(|v| owners[v] == id));
            }
        }
        if seen.len() != node.patch_vertices.len() {
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

pub(crate) fn assign_boundaries(skeleton: &mut Skeleton, mesh: &Mesh<INPUT>) -> SkeletonResult<()> {
    let mut expected = analyze(skeleton, mesh)?;
    for (_, edge) in &mut skeleton.edges {
        let candidates = expected
            .get_mut(&pair(edge.nodes[0], edge.nodes[1]))
            .ok_or("Skeleton edge has no corresponding patch boundary")?;
        let old: Vec<_> = edge
            .boundary
            .edge_midpoints
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
        edge.boundary = PatchBoundary {
            edge_midpoints: cycle,
        };
    }
    if expected.values().any(|v| !v.is_empty()) {
        return Err("Patch boundary has no corresponding skeleton edge".into());
    }
    Ok(())
}

impl Skeleton {
    /// Validate ownership, patch topology, and graph/boundary correspondence.
    pub fn validate(&self, mesh: &Mesh<INPUT>) -> SkeletonResult<()> {
        input_genus(mesh)?;
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
                .edge_midpoints
                .iter()
                .any(|e| !valid_edges.contains(e))
            {
                return Err("Boundary references an invalid mesh edge".into());
            }
            let cycle: Vec<_> = edge
                .boundary
                .edge_midpoints
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

/// Check the input before algorithms assume a closed triangular surface.
pub(crate) fn input_genus<M: Tag>(mesh: &Mesh<M>) -> SkeletonResult<usize> {
    let vertices = mesh.vert_ids();
    if vertices.is_empty() || mesh.nr_faces() == 0 {
        return Err("Skeleton extraction requires a nonempty mesh".into());
    }
    if vertices
        .iter()
        .any(|&v| !mesh.position(v).iter().all(|x| x.is_finite()))
    {
        return Err("Input positions must be finite".into());
    }
    let mut incidence = HashMap::new();
    let mut directed = HashSet::new();
    let mut links = HashMap::<VertKey<M>, HashMap<VertKey<M>, Vec<VertKey<M>>>>::new();
    for f in mesh.face_ids() {
        let v: Vec<_> = mesh.vertices(f).collect();
        if v.len() != 3 {
            return Err("Skeleton extraction requires triangles".into());
        }
        let [a, b, c] = [
            mesh.position(v[0]),
            mesh.position(v[1]),
            mesh.position(v[2]),
        ];
        if (b - a).cross(&(c - a)).norm_squared() == 0.0 {
            return Err("Input contains a degenerate triangle".into());
        }
        for i in 0..3 {
            let link = links.entry(v[i]).or_default();
            let (a, b) = (v[(i + 1) % 3], v[(i + 2) % 3]);
            link.entry(a).or_default().push(b);
            link.entry(b).or_default().push(a);
            let (a, b) = (v[i], v[(i + 1) % 3]);
            if a == b || !directed.insert((a, b)) {
                return Err("Input has repeated vertices or inconsistent face winding".into());
            }
            let mut edge = [a, b];
            edge.sort_by(|a, b| a.partial_cmp(b).unwrap());
            *incidence.entry(edge).or_insert(0usize) += 1;
        }
    }
    if incidence.values().any(|&n| n != 2) {
        return Err("Input must be a closed two-manifold".into());
    }
    // Check vertex links too: edge incidence alone does not exclude pinched vertices.
    for &v in &vertices {
        let link = links.get(&v).ok_or("Input has an isolated vertex")?;
        if link.is_empty() || link.values().any(|n| n.len() != 2) {
            return Err("Input has a non-manifold vertex link".into());
        }
        let mut seen = HashSet::new();
        let mut stack = vec![*link.keys().next().unwrap()];
        while let Some(w) = stack.pop() {
            if seen.insert(w) {
                stack.extend(&link[&w]);
            }
        }
        if seen.len() != link.len() {
            return Err("Input has a disconnected vertex link".into());
        }
    }
    let mut seen = HashSet::new();
    let mut stack = vec![vertices[0]];
    while let Some(v) = stack.pop() {
        if seen.insert(v) {
            stack.extend(mesh.neighbors(v));
        }
    }
    if seen.len() != vertices.len() {
        return Err("Input must be connected".into());
    }
    let chi = vertices.len() as i64 - incidence.len() as i64 + mesh.nr_faces() as i64;
    if chi > 2 || (2 - chi) % 2 != 0 {
        return Err("Input has invalid surface Euler characteristic".into());
    }
    let volume = volume(mesh);
    if !volume.is_finite() || volume <= 0.0 {
        return Err("Input must enclose a finite nonzero volume".into());
    }
    Ok(((2 - chi) / 2) as usize)
}

/// Signed-volume magnitude, relative to a mesh vertex to limit cancellation.
pub(crate) fn volume<M: Tag>(mesh: &Mesh<M>) -> f64 {
    let Some(origin) = mesh.vert_ids().first().map(|&v| mesh.position(v)) else {
        return 0.0;
    };
    mesh.face_ids()
        .into_iter()
        .map(|f| {
            let v: Vec<_> = mesh
                .vertices(f)
                .map(|v| mesh.position(v) - origin)
                .collect();
            (1..v.len() - 1)
                .map(|i| v[0].dot(&v[i].cross(&v[i + 1])) / 6.0)
                .sum::<f64>()
        })
        .sum::<f64>()
        .abs()
}
