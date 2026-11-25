//! Reference longitudes that keep drawn loops from winding around handles.
//!
//! The lifted boundaries fix a drawing only up to Dehn twists around them: a twisted loop still
//! has the template's regions, so no combinatorial check sees it. Every handle gets a reference
//! curve on both surfaces, its shortest longitude: a closed path that crosses the boundary of
//! one cut edge (an edge off a spanning tree of the skeleton) once and no other cut boundary.
//! Insertion makes every loop cross the mesh's longitudes as the template loop crosses the
//! template's, with signs. Both are shortest, so the loops follow the mesh's natural framing,
//! whatever the template's placement in space.

use super::template::Template;
use super::*;
use crate::SkeletonEdgeID;
use dualcube_dual::insertion::{MAX_HANDLES, Turns, Winding};
use std::collections::{BinaryHeap, VecDeque};

/// Cost of ending an arc one crossing away from the template's count, in units of the square
/// root of the surface area.
const PENALTY: f64 = 10.;

pub(super) struct Longitudes {
    // Longitude edges on the mesh by incident triangle: handle and directed ends.
    segments: HashMap<FaceID, Vec<(usize, Vector3D, Vector3D)>>,
    // The template's crossings of every loop from each of its crossing vertices to the next.
    template: HashMap<(LoopID, usize), Turns>,
    penalty: f64,
}

impl Longitudes {
    /// `None` without handles, or when a longitude cannot be found (then loops are drawn freely).
    pub fn new(
        mesh: &Mesh<INPUT>,
        realized: &RealizedSkeleton,
        template: &Template,
    ) -> Option<Self> {
        let skeleton = &realized.labeled.skeleton;
        let cut = cut_edges(skeleton);
        if cut.is_empty() {
            return None;
        }
        if cut.len() > MAX_HANDLES {
            tracing::warn!(
                "Loop drawing: {} handles; loops may wind around them",
                cut.len()
            );
            return None;
        }
        let mesh_side = MeshSide::new(mesh, realized, &cut)?;
        let template_side = TemplateSide::new(realized, template, &cut)?;
        let mut segments: HashMap<FaceID, Vec<_>> = HashMap::new();
        for (h, edges) in mesh_side.longitudes.iter().enumerate() {
            for &(a, b) in edges {
                let (pa, pb) = (mesh_side.position(a), mesh_side.position(b));
                for &f in &mesh_side.faces[&(a.min(b), a.max(b))] {
                    segments.entry(f).or_default().push((h, pa, pb));
                }
            }
        }
        let mut counts: HashMap<(LoopID, usize), Turns> = HashMap::new();
        for (q, quad) in template.quads.iter().enumerate() {
            for (k, &(id, _)) in quad.loops.iter().enumerate() {
                let (a, b) = quad.side(quad.exit(k));
                let entry = counts
                    .entry((id, template.target.quad_crossing(q)))
                    .or_insert([0; MAX_HANDLES]);
                for (h, edges) in template_side.longitudes.iter().enumerate() {
                    entry[h] += edges.iter().map(|&e| crossing(e, (a, b))).sum::<i8>();
                }
            }
        }
        let area: f64 = mesh
            .face_ids()
            .into_iter()
            .map(|f| {
                let c: Vec<_> = mesh.vertices(f).map(|v| mesh.position(v)).collect();
                (c[1] - c[0]).cross(&(c[2] - c[0])).norm() / 2.
            })
            .sum();
        Some(Self {
            segments,
            template: counts,
            penalty: PENALTY * area.sqrt(),
        })
    }

    /// The template loop's crossings with the template's longitudes, in all.
    #[cfg(test)]
    pub fn template_total(&self, id: LoopID) -> Turns {
        let mut total = [0; MAX_HANDLES];
        for (_, turns) in self.template.iter().filter(|((l, _), _)| *l == id) {
            for h in 0..MAX_HANDLES {
                total[h] += turns[h];
            }
        }
        total
    }

    /// A drawn curve's crossings with the mesh's longitudes.
    #[cfg(test)]
    pub fn crossings(&self, mesh: &Mesh<INPUT>, curve: &Loop) -> Turns {
        let mut total = [0; MAX_HANDLES];
        let arcs = &curve.surface;
        for (i, arc) in arcs.iter().enumerate() {
            if arcs[(i + 1) % arcs.len()].face == arc.face {
                continue;
            }
            // The curve leaves the triangle across the edge holding its end.
            let end = dualcube_dual::SurfaceArc::position(mesh, arc.face, arc.to);
            let corners: Vec<_> = mesh.vertices(arc.face).map(|v| mesh.position(v)).collect();
            for j in 0..3 {
                let (x, y) = (corners[j], corners[(j + 1) % 3]);
                let t = (end - x).dot(&(y - x)) / (y - x).norm_squared();
                if (end - (x + (y - x) * t)).norm() <= 1e-9 * (y - x).norm() {
                    let turns = self.step(x, y, arc.face);
                    for h in 0..MAX_HANDLES {
                        total[h] += turns[h];
                    }
                }
            }
        }
        total
    }
}

impl Winding for Longitudes {
    fn step(&self, from: Vector3D, to: Vector3D, face: FaceID) -> Turns {
        let mut turns = [0; MAX_HANDLES];
        for &(h, a, b) in self.segments.get(&face).map_or(&[][..], Vec::as_slice) {
            let d = b - a;
            let on = |p: Vector3D| {
                let t = (p - a).dot(&d) / d.norm_squared();
                (-1e-9..=1. + 1e-9).contains(&t) && (p - (a + d * t)).norm() <= 1e-9 * d.norm()
            };
            if from != to && on(from) && on(to) {
                turns[h] += if (to - from).dot(&d) > 0. { 1 } else { -1 };
            }
        }
        turns
    }

    fn template(&self, id: LoopID, vertex: usize) -> Turns {
        self.template
            .get(&(id, vertex))
            .copied()
            .unwrap_or([0; MAX_HANDLES])
    }

    fn penalty(&self) -> f64 {
        self.penalty
    }
}

/// The edges off a breadth-first spanning tree of the skeleton: one per handle.
pub(super) fn cut_edges(skeleton: &crate::Skeleton) -> Vec<SkeletonEdgeID> {
    let mut reached = HashSet::new();
    let mut tree = HashSet::new();
    for root in skeleton.nodes.keys() {
        if !reached.insert(root) {
            continue;
        }
        let mut queue = VecDeque::from([root]);
        while let Some(node) = queue.pop_front() {
            for (edge, e) in &skeleton.edges {
                if e.nodes.contains(&node) {
                    let other = e.other(node);
                    if reached.insert(other) {
                        tree.insert(edge);
                        queue.push_back(other);
                    }
                }
            }
        }
    }
    skeleton
        .edges
        .keys()
        .filter(|e| !tree.contains(e))
        .collect()
}

/// The signed crossing of a directed edge `side`, crossed from its left to its right, with a
/// longitude edge.
fn crossing(longitude: (usize, usize), side: (usize, usize)) -> i8 {
    if longitude == side {
        1
    } else if longitude == (side.1, side.0) {
        -1
    } else {
        0
    }
}

/// The shortest closed path through one of `crossed` (directed edges a -> b that a boundary
/// crosses from left to right) avoiding `forbidden`, as directed edges oriented so that the
/// boundary crosses it from left to right.
fn longitude(
    adjacent: &[Vec<(usize, f64)>],
    crossed: &[(usize, usize)],
    forbidden: &HashSet<(usize, usize)>,
) -> Option<Vec<(usize, usize)>> {
    let mut best: Option<(f64, Vec<usize>)> = None;
    for &(a, b) in crossed {
        let closing = adjacent[b].iter().find(|&&(x, _)| x == a)?.1;
        if let Some((length, path)) = shortest(adjacent, a, b, forbidden)
            && best.as_ref().is_none_or(|(l, _)| length + closing < *l)
        {
            best = Some((length + closing, path));
        }
    }
    // The path a .. b closes along b -> a, which the boundary crosses from right to left.
    let (_, path) = best?;
    let mut edges: Vec<_> = path.windows(2).map(|w| (w[1], w[0])).collect();
    edges.push((path[0], path[path.len() - 1]));
    Some(edges)
}

/// Dijkstra from `from` to `to` without `forbidden` edges.
fn shortest(
    adjacent: &[Vec<(usize, f64)>],
    from: usize,
    to: usize,
    forbidden: &HashSet<(usize, usize)>,
) -> Option<(f64, Vec<usize>)> {
    let mut distance = vec![f64::INFINITY; adjacent.len()];
    let mut previous = vec![usize::MAX; adjacent.len()];
    let mut heap = BinaryHeap::new();
    distance[from] = 0.;
    heap.push((std::cmp::Reverse(OrderedFloat(0.)), from));
    while let Some((std::cmp::Reverse(d), x)) = heap.pop() {
        if x == to {
            break;
        }
        if d.0 > distance[x] {
            continue;
        }
        for &(y, w) in &adjacent[x] {
            if !forbidden.contains(&(x, y)) && d.0 + w < distance[y] {
                distance[y] = d.0 + w;
                previous[y] = x;
                heap.push((std::cmp::Reverse(OrderedFloat(d.0 + w)), y));
            }
        }
    }
    if !distance[to].is_finite() {
        return None;
    }
    let mut path = vec![to];
    while path[path.len() - 1] != from {
        path.push(previous[path[path.len() - 1]]);
    }
    path.reverse();
    Some((distance[to], path))
}

/// Longitudes along the edges of the input mesh.
struct MeshSide<'a> {
    mesh: &'a Mesh<INPUT>,
    vertices: Vec<VertID>,
    // Triangles of every edge, by its vertex indices (smaller first).
    faces: HashMap<(usize, usize), Vec<FaceID>>,
    longitudes: Vec<Vec<(usize, usize)>>,
}

impl<'a> MeshSide<'a> {
    fn new(
        mesh: &'a Mesh<INPUT>,
        realized: &RealizedSkeleton,
        cut: &[SkeletonEdgeID],
    ) -> Option<Self> {
        let skeleton = &realized.labeled.skeleton;
        let vertices: Vec<VertID> = mesh.vert_ids().into_iter().collect();
        let index: HashMap<VertID, usize> =
            vertices.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        let mut adjacent = vec![vec![]; vertices.len()];
        let mut faces: HashMap<(usize, usize), Vec<FaceID>> = HashMap::new();
        for f in mesh.face_ids() {
            let corners: Vec<usize> = mesh.vertices(f).map(|v| index[&v]).collect();
            for i in 0..3 {
                let (a, b) = (corners[i], corners[(i + 1) % 3]);
                let list = faces.entry((a.min(b), a.max(b))).or_default();
                if list.is_empty() {
                    let length = (mesh.position(vertices[a]) - mesh.position(vertices[b])).norm();
                    adjacent[a].push((b, length));
                    adjacent[b].push((a, length));
                }
                list.push(f);
            }
        }
        // Every cut boundary as the input edges it crosses from left to right, as lifted: a
        // negative label reverses the traversal.
        let crossed: Vec<Vec<(usize, usize)>> = cut
            .iter()
            .map(|&e| {
                let negative = realized.labeled.labels[e].sign == Sign::Negative;
                skeleton.edges[e]
                    .boundary
                    .crossings
                    .iter()
                    .map(|c| {
                        let exit = skeleton.surface.crossings[*c].exit;
                        let (r, t) = (index[&mesh.root(exit)], index[&mesh.toor(exit)]);
                        if negative { (t, r) } else { (r, t) }
                    })
                    .collect()
            })
            .collect();
        let forbidden: HashSet<_> = crossed
            .iter()
            .flatten()
            .flat_map(|&(a, b)| [(a, b), (b, a)])
            .collect();
        let longitudes = crossed
            .iter()
            .map(|edges| {
                // Edges the boundary crosses once.
                let mut count: HashMap<(usize, usize), usize> = HashMap::new();
                for &(a, b) in edges {
                    *count.entry((a.min(b), a.max(b))).or_default() += 1;
                }
                let once: Vec<_> = edges
                    .iter()
                    .copied()
                    .filter(|&(a, b)| count[&(a.min(b), a.max(b))] == 1)
                    .collect();
                longitude(&adjacent, &once, &forbidden)
            })
            .collect::<Option<Vec<_>>>();
        let Some(longitudes) = longitudes else {
            tracing::warn!("Loop drawing: no longitude on the mesh; loops may wind around handles");
            return None;
        };
        Some(Self {
            mesh,
            vertices,
            faces,
            longitudes,
        })
    }

    fn position(&self, i: usize) -> Vector3D {
        self.mesh.position(self.vertices[i])
    }
}

/// Longitudes along the sides of the template's quads.
struct TemplateSide {
    longitudes: Vec<Vec<(usize, usize)>>,
}

impl TemplateSide {
    fn new(
        realized: &RealizedSkeleton,
        template: &Template,
        cut: &[SkeletonEdgeID],
    ) -> Option<Self> {
        let solids = realized.template_solids();
        let corner = |solid: usize, code: usize| -> Vector3D {
            let (lo, hi) = (solids[solid].lo, solids[solid].hi);
            Vector3D::from_fn(|d, _| if code & (1 << d) != 0 { hi[d] } else { lo[d] })
        };
        let count = template
            .quads
            .iter()
            .flat_map(|q| q.corners)
            .max()
            .map_or(0, |m| m + 1);
        let mut adjacent: Vec<Vec<(usize, f64)>> = vec![vec![]; count];
        for quad in &template.quads {
            for i in 0..4 {
                let (a, b) = quad.side(i);
                if adjacent[a].iter().any(|&(x, _)| x == b) {
                    continue;
                }
                let length = (corner(quad.solid, quad.codes[i])
                    - corner(quad.solid, quad.codes[(i + 1) % 4]))
                .norm();
                adjacent[a].push((b, length));
                adjacent[b].push((a, length));
            }
        }
        // Every cut boundary as the quad sides its loop leaves through.
        let crossed: Vec<Vec<(usize, usize)>> = cut
            .iter()
            .map(|e| {
                let id = template.boundaries[e];
                template
                    .quads
                    .iter()
                    .flat_map(|quad| {
                        (0..2)
                            .filter(move |&k| quad.loops[k].0 == id)
                            .map(move |k| quad.side(quad.exit(k)))
                    })
                    .collect()
            })
            .collect();
        let forbidden: HashSet<_> = crossed
            .iter()
            .flatten()
            .flat_map(|&(a, b)| [(a, b), (b, a)])
            .collect();
        let longitudes = crossed
            .iter()
            .map(|edges| longitude(&adjacent, edges, &forbidden))
            .collect::<Option<Vec<_>>>();
        if longitudes.is_none() {
            tracing::warn!(
                "Loop drawing: no longitude on the template; loops may wind around handles"
            );
        }
        Some(Self {
            longitudes: longitudes?,
        })
    }
}
