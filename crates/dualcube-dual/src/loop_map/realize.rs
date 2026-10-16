//! Drawing a loop map's loops as explicit curves, for output only.
//!
//! Points on input edges are spaced uniformly by their order, so both triangles of an edge agree.
//! Inside a triangle, every face gets a center joined to its corners and every interior edge a
//! midpoint; no interior edge then joins two points of the triangle's sides, so the barycentric
//! (Tutte) embedding with the sides fixed is a planar drawing. At a crossing one loop passes
//! through the vertex and the other just beside it, so that curves cross inside segments.

use super::*;
use crate::SurfaceArc;

const CORNERS: [[f64; 2]; 3] = [[0., 0.], [1., 0.], [0., 1.]];

fn lerp(a: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
}

// Merge consecutive segments in one triangle whose shared point is all but on the line through
// the others: nearly coincident supporting lines would leave sliver cells in the arrangement,
// while moving that point so little separates nothing.
fn straighten(arcs: Vec<SurfaceArc>) -> Vec<SurfaceArc> {
    let mut result: Vec<SurfaceArc> = vec![];
    for arc in arcs {
        if let Some(last) = result.last_mut()
            && last.face == arc.face
        {
            let (a, m, b) = (last.from, last.to, arc.to);
            let d = [b[0] - a[0], b[1] - a[1]];
            let cross = d[0] * (m[1] - a[1]) - d[1] * (m[0] - a[0]);
            let forward = d[0] * (m[0] - a[0]) + d[1] * (m[1] - a[1]);
            let length = d[0] * d[0] + d[1] * d[1];
            if cross.abs() < 1e-9 * length && forward > 0. && forward < length {
                last.to = b;
                continue;
            }
        }
        result.push(arc);
    }
    result
}

/// Planar positions of a map's vertices, interior edge midpoints and face centers, in the frame
/// of their triangle.
#[derive(Clone, Debug, Default)]
pub(crate) struct Embedding {
    pub vertex: HashMap<(FaceID, usize), [f64; 2]>,
    /// By either half-edge; edges along input edges have none.
    pub middle: HashMap<usize, [f64; 2]>,
    pub center: HashMap<usize, [f64; 2]>,
}

impl LoopMap {
    /// The barycentric embedding of every triangle's faces, with the sides fixed.
    pub(crate) fn embedding(
        &self,
        mesh: &Mesh<INPUT>,
    ) -> Result<Embedding, PropertyViolationError> {
        // Frame positions of vertices on input edges and corners, per triangle.
        let starts = self.edge_starts();
        let mut fixed = HashMap::<(FaceID, usize), [f64; 2]>::new();
        for face in mesh.face_ids() {
            let sides = sides(mesh, face)?;
            for (s, &edge) in sides.iter().enumerate() {
                let along = self.along_edge(edge, &starts);
                let n = along.len() as f64 - 1.;
                for (i, &v) in along.iter().enumerate() {
                    let p = lerp(CORNERS[s], CORNERS[(s + 1) % 3], i as f64 / n);
                    fixed.insert((face, v), p);
                }
            }
        }
        let mut by_triangle = HashMap::<FaceID, Vec<usize>>::new();
        for (f, face) in self.faces.iter().enumerate() {
            by_triangle.entry(face.triangle).or_default().push(f);
        }
        let mut embedding = Embedding::default();
        for (&triangle, faces) in &by_triangle {
            self.embed(triangle, faces, &fixed, &mut embedding);
        }
        Ok(embedding)
    }

    /// Explicit curves of all closed loops, in their directions.
    pub fn realize(
        &self,
        mesh: &Mesh<INPUT>,
    ) -> Result<HashMap<LoopID, Vec<SurfaceArc>>, PropertyViolationError> {
        let Embedding {
            vertex: position,
            middle,
            ..
        } = self.embedding(mesh)?;
        let mut curves = HashMap::new();
        for (id, chain) in self.chains() {
            let mut arcs = vec![];
            for &h in &chain {
                let triangle = self.faces[self.halves[h].face].triangle;
                let point = |v: usize| -> Option<[f64; 2]> {
                    if self.points[v] == Point::Crossing && self.passes_beside(v, id) {
                        return self.beside(v, triangle, &position, &middle);
                    }
                    position.get(&(triangle, v)).copied()
                };
                let (a, b) = (self.halves[h].origin, self.target(h));
                let m = *middle
                    .get(&h)
                    .or_else(|| middle.get(&self.halves[h].twin))
                    .ok_or(PropertyViolationError::UnknownError)?;
                let (pa, pb) = (
                    point(a).ok_or(PropertyViolationError::UnknownError)?,
                    point(b).ok_or(PropertyViolationError::UnknownError)?,
                );
                arcs.push(SurfaceArc {
                    face: triangle,
                    from: pa,
                    to: m,
                });
                arcs.push(SurfaceArc {
                    face: triangle,
                    from: m,
                    to: pb,
                });
            }
            curves.insert(id, straighten(arcs));
        }
        Ok(curves)
    }

    // The half-edge leaving the root of every input half-edge, along it.
    pub(crate) fn edge_starts(&self) -> HashMap<EdgeID, usize> {
        self.halves
            .iter()
            .enumerate()
            .filter_map(|(i, h)| match h.piece {
                Piece::Mesh(e) if matches!(self.points[h.origin], Point::Input(_)) => Some((e, i)),
                _ => None,
            })
            .collect()
    }

    // The vertices along an input half-edge, from its root to its tip.
    fn along_edge(&self, edge: EdgeID, starts: &HashMap<EdgeID, usize>) -> Vec<usize> {
        let Some(&(mut h)) = starts.get(&edge) else {
            return vec![];
        };
        let mut result = vec![self.halves[h].origin];
        loop {
            let v = self.target(h);
            result.push(v);
            if matches!(self.points[v], Point::Input(_)) {
                break;
            }
            h = self
                .outgoing(v)
                .into_iter()
                .find(|&o| self.halves[o].piece == Piece::Mesh(edge))
                .unwrap();
        }
        result
    }

    // At a crossing, the loop with the larger ID passes beside the vertex.
    fn passes_beside(&self, v: usize, id: LoopID) -> bool {
        self.outgoing(v)
            .iter()
            .filter_map(|&h| self.halves[h].piece.wall())
            .any(|other| other < id)
    }

    // A point beside crossing `v`, between the outgoing pieces of both loops.
    fn beside(
        &self,
        v: usize,
        triangle: FaceID,
        position: &HashMap<(FaceID, usize), [f64; 2]>,
        middle: &HashMap<usize, [f64; 2]>,
    ) -> Option<[f64; 2]> {
        let x = position.get(&(triangle, v))?;
        let ahead: Vec<_> = self
            .outgoing(v)
            .into_iter()
            .filter(|&h| matches!(self.halves[h].piece, Piece::Loop(_, true)))
            .filter_map(|h| middle.get(&h).copied())
            .collect();
        if ahead.len() != 2 {
            return None;
        }
        let e = 0.05;
        Some([
            x[0] + e * (ahead[0][0] - x[0]) + e * (ahead[1][0] - x[0]),
            x[1] + e * (ahead[0][1] - x[1]) + e * (ahead[1][1] - x[1]),
        ])
    }

    // Barycentric embedding of one triangle's faces, added to `embedding`.
    fn embed(
        &self,
        triangle: FaceID,
        faces: &[usize],
        fixed: &HashMap<(FaceID, usize), [f64; 2]>,
        embedding: &mut Embedding,
    ) {
        // Nodes: map vertices, interior edge midpoints, and face centers.
        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        enum Node {
            Vertex(usize),
            Middle(usize),
            Center(usize),
        }
        let mut neighbors = HashMap::<Node, Vec<Node>>::new();
        let mut link = |a: Node, b: Node| {
            neighbors.entry(a).or_default().push(b);
            neighbors.entry(b).or_default().push(a);
        };
        let mut middles = HashSet::new();
        for &f in faces {
            let start = self.faces[f].half;
            let mut h = start;
            loop {
                let half = self.halves[h];
                let a = Node::Vertex(half.origin);
                link(Node::Center(f), a);
                if !matches!(half.piece, Piece::Mesh(_)) {
                    let key = h.min(half.twin);
                    let m = Node::Middle(key);
                    link(Node::Center(f), m);
                    if middles.insert(key) {
                        link(m, a);
                        link(m, Node::Vertex(self.target(h)));
                    }
                }
                h = half.next;
                if h == start {
                    break;
                }
            }
        }
        let mut placed: HashMap<Node, [f64; 2]> = HashMap::new();
        let mut free = vec![];
        for &node in neighbors.keys() {
            match node {
                Node::Vertex(v) if fixed.contains_key(&(triangle, v)) => {
                    placed.insert(node, fixed[&(triangle, v)]);
                }
                _ => {
                    placed.insert(node, [1. / 3., 1. / 3.]);
                    free.push(node);
                }
            }
        }
        // Gauss-Seidel on the barycentric equations; the solution is unique.
        for _ in 0..10_000 {
            let mut change: f64 = 0.;
            for &node in &free {
                let around = &neighbors[&node];
                let n = around.len() as f64;
                let mut p = [0., 0.];
                for q in around {
                    let x = placed[q];
                    p[0] += x[0] / n;
                    p[1] += x[1] / n;
                }
                let old = placed[&node];
                change = change.max((p[0] - old[0]).abs().max((p[1] - old[1]).abs()));
                placed.insert(node, p);
            }
            if change < 1e-15 {
                break;
            }
        }
        for (node, p) in placed {
            match node {
                Node::Vertex(v) => {
                    embedding.vertex.insert((triangle, v), p);
                }
                Node::Middle(h) => {
                    embedding.middle.insert(h, p);
                    embedding.middle.insert(self.halves[h].twin, p);
                }
                Node::Center(f) => {
                    embedding.center.insert(f, p);
                }
            }
        }
    }
}
