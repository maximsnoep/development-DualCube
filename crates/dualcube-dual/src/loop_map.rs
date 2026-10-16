//! The input surface cut by loops, as a combinatorial planar map.
//!
//! Every face of the map lies in one input triangle. Its edges are pieces of input edges, of
//! loops, or free separators that bound nothing. Loops may revisit triangles and edges; their
//! crossings with input edges and with each other are vertices. Topology is exact: it follows
//! from the map alone. Positions are approximate and only guide costs and drawing.
//!
//! Half-edges have their face on their left. Around a vertex, `twin(prev(h))` is the next
//! outgoing half-edge counterclockwise.

use crate::arrangement::{general, sides};
use crate::{Loop, LoopID, PropertyViolationError};
use dualcube_types::prelude::*;
use slotmap::SlotMap;

/// What a map edge lies on, seen from one of its half-edges.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Piece {
    /// Along this input half-edge.
    Mesh(EdgeID),
    /// Along a loop; `true` in the loop's direction.
    Loop(LoopID, bool),
    /// Inside a triangle, separating nothing.
    Free,
}

impl Piece {
    pub fn wall(self) -> Option<LoopID> {
        match self {
            Self::Loop(id, _) => Some(id),
            _ => None,
        }
    }
}

/// What a map vertex is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Point {
    Input(VertID),
    /// A loop crossing an input edge.
    OnEdge,
    /// Two loops crossing.
    Crossing,
    /// Any other vertex, such as a subdivision of a single piece.
    Other,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Half {
    pub twin: usize,
    pub next: usize,
    pub prev: usize,
    pub origin: usize,
    pub face: usize,
    pub piece: Piece,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Face {
    pub half: usize,
    pub triangle: FaceID,
}

#[derive(Clone, Debug, Default)]
pub struct LoopMap {
    pub(crate) halves: Vec<Half>,
    pub(crate) points: Vec<Point>,
    pub(crate) positions: Vec<Vector3D>,
    pub(crate) faces: Vec<Face>,
    // One half-edge leaving every vertex.
    leaving: Vec<usize>,
    // The half-edge starting at the root of every input half-edge, when built from the mesh.
    roots: HashMap<EdgeID, usize>,
}

fn root(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

fn join(parent: &mut [usize], a: usize, b: usize) {
    let (a, b) = (root(parent, a), root(parent, b));
    parent[a] = b;
}

impl LoopMap {
    /// The input mesh without loops.
    pub fn new(mesh: &Mesh<INPUT>) -> Self {
        let mut map = Self::default();
        let mut vertex = HashMap::new();
        for v in mesh.vert_ids() {
            vertex.insert(v, map.points.len());
            map.points.push(Point::Input(v));
            map.positions.push(mesh.position(v));
        }
        let edges = mesh.edge_ids();
        let index: HashMap<_, _> = edges.iter().enumerate().map(|(i, &e)| (e, i)).collect();
        let mut face = HashMap::new();
        for f in mesh.face_ids() {
            face.insert(f, map.faces.len());
            map.faces.push(Face {
                half: index[&mesh.edges(f).next().unwrap()],
                triangle: f,
            });
        }
        for &e in &edges {
            map.halves.push(Half {
                twin: index[&mesh.twin(e)],
                next: index[&mesh.next(e)],
                prev: 0,
                origin: vertex[&mesh.root(e)],
                face: face[&mesh.face(e)],
                piece: Piece::Mesh(e),
            });
        }
        for i in 0..map.halves.len() {
            let next = map.halves[i].next;
            map.halves[next].prev = i;
        }
        map.roots = index;
        map.index_vertices();
        map
    }

    fn index_vertices(&mut self) {
        self.leaving = vec![usize::MAX; self.points.len()];
        for (i, h) in self.halves.iter().enumerate() {
            self.leaving[h.origin] = i;
        }
    }

    /// The map of loops given by their geometric arrangement: legacy edge offsets or explicit
    /// curves. Topology equals that arrangement's.
    pub fn from_loops(
        mesh: &Mesh<INPUT>,
        loops: &SlotMap<LoopID, Loop>,
    ) -> Result<Self, PropertyViolationError> {
        let arrangement = general::build(mesh, loops)?;
        let cells = arrangement
            .general
            .as_ref()
            .ok_or(PropertyViolationError::UnknownError)?;
        let mut map = Self::default();
        let mut base = vec![];
        for (i, cell) in cells.cells.iter().enumerate() {
            base.push(map.halves.len());
            let n = cell.vertices.len();
            for j in 0..n {
                map.halves.push(Half {
                    twin: usize::MAX,
                    next: map.halves.len() - j + (j + 1) % n,
                    prev: map.halves.len() - j + (j + n - 1) % n,
                    origin: cell.vertices[j],
                    face: i,
                    piece: Piece::Free,
                });
            }
            map.faces.push(Face {
                half: base[i],
                triangle: cell.face,
            });
        }
        for link in &cells.links {
            let [a, b] = [0, 1].map(|s| base[link.cells[s]] + link.sides[s]);
            map.halves[a].twin = b;
            map.halves[b].twin = a;
            for s in 0..2 {
                let cell = &cells.cells[link.cells[s]];
                let h = base[link.cells[s]] + link.sides[s];
                map.halves[h].piece = if let Some(id) = link.wall {
                    Piece::Loop(id, link.left == s)
                } else if cells.cells[link.cells[0]].face != cells.cells[link.cells[1]].face {
                    let j = link.sides[s];
                    let (p, q) = (cell.points[j], cell.points[(j + 1) % cell.points.len()]);
                    let middle = [(p[0] + q[0]) / 2., (p[1] + q[1]) / 2.];
                    let (side, _) =
                        general::side_at(middle).ok_or(PropertyViolationError::UnknownError)?;
                    Piece::Mesh(sides(mesh, cell.face)?[side])
                } else {
                    Piece::Free
                };
            }
        }
        if map.halves.iter().any(|h| h.twin == usize::MAX) {
            return Err(PropertyViolationError::UnknownError);
        }
        map.positions = cells.positions.clone();
        map.points = cells.original.iter().map(|_| Point::Other).collect();
        let mut walls = vec![HashSet::new(); map.points.len()];
        let mut on_edge = vec![false; map.points.len()];
        for h in &map.halves {
            match h.piece {
                Piece::Loop(id, _) => {
                    walls[h.origin].insert(id);
                }
                Piece::Mesh(_) => on_edge[h.origin] = true,
                Piece::Free => {}
            }
        }
        for (v, point) in map.points.iter_mut().enumerate() {
            *point = match cells.original[v] {
                Some(input) => Point::Input(input),
                None if on_edge[v] && !walls[v].is_empty() => Point::OnEdge,
                None if walls[v].len() >= 2 => Point::Crossing,
                None => Point::Other,
            };
        }
        map.index_vertices();
        Ok(map)
    }

    /// Add disjoint curves given by their input-edge crossings. `exits[o]` is the half-edge that
    /// occurrence `o` leaves its triangle through; each curve lists its occurrences in order,
    /// and `order` lists the occurrences on every canonical half-edge from root to tip.
    pub fn add_disjoint(
        &mut self,
        mesh: &Mesh<INPUT>,
        curves: &[(LoopID, Vec<usize>)],
        exits: &[EdgeID],
        order: &HashMap<EdgeID, Vec<usize>>,
    ) -> Result<(), PropertyViolationError> {
        let mut vertex = vec![usize::MAX; exits.len()];
        for (&edge, occurrences) in order {
            let mut half = *self
                .roots
                .get(&edge)
                .ok_or(PropertyViolationError::UnknownError)?;
            let (a, b) = (
                mesh.position(mesh.root(edge)),
                mesh.position(mesh.toor(edge)),
            );
            let n = occurrences.len() as f64 + 1.;
            for (i, &o) in occurrences.iter().enumerate() {
                let t = (i + 1) as f64 / n;
                let x = self.split(half, Point::OnEdge, a + (b - a) * t);
                vertex[o] = x;
                half = self.halves[half].next;
            }
        }
        for (id, occurrences) in curves {
            let k = occurrences.len();
            for i in 0..k {
                let (from, to) = (occurrences[i], occurrences[(i + 1) % k]);
                let triangle = mesh.face(mesh.twin(exits[from]));
                if mesh.face(exits[to]) != triangle {
                    return Err(PropertyViolationError::UnknownError);
                }
                let [a, b] = [vertex[from], vertex[to]];
                let (ha, hb) = self
                    .common_face(a, b, triangle)
                    .ok_or(PropertyViolationError::UnknownError)?;
                self.connect(ha, hb, Piece::Loop(*id, true));
            }
        }
        Ok(())
    }

    /// Outgoing half-edges of `a` and `b` in one face of `triangle`.
    pub(crate) fn common_face(
        &self,
        a: usize,
        b: usize,
        triangle: FaceID,
    ) -> Option<(usize, usize)> {
        let from_a = self.outgoing(a);
        let from_b = self.outgoing(b);
        from_a.iter().find_map(|&ha| {
            let face = self.halves[ha].face;
            (self.faces[face].triangle == triangle)
                .then(|| from_b.iter().find(|&&hb| self.halves[hb].face == face))
                .flatten()
                .map(|&hb| (ha, hb))
        })
    }

    /// Half-edges leaving vertex `v`, counterclockwise.
    pub fn outgoing(&self, v: usize) -> Vec<usize> {
        let start = self.leaving[v];
        if start == usize::MAX {
            return vec![];
        }
        let mut result = vec![start];
        let mut h = self.halves[self.halves[start].prev].twin;
        while h != start {
            result.push(h);
            h = self.halves[self.halves[h].prev].twin;
        }
        result
    }

    /// Split the edge of half-edge `h` with a new vertex; `h` then ends there.
    pub(crate) fn split(&mut self, h: usize, point: Point, position: Vector3D) -> usize {
        let x = self.points.len();
        self.points.push(point);
        self.positions.push(position);
        let t = self.halves[h].twin;
        let (h2, t2) = (self.halves.len(), self.halves.len() + 1);
        self.leaving.push(h2);
        let (hh, th) = (self.halves[h], self.halves[t]);
        self.halves.push(Half {
            twin: t,
            next: hh.next,
            prev: h,
            origin: x,
            face: hh.face,
            piece: hh.piece,
        });
        self.halves.push(Half {
            twin: h,
            next: th.next,
            prev: t,
            origin: x,
            face: th.face,
            piece: th.piece,
        });
        self.halves[hh.next].prev = h2;
        self.halves[h].next = h2;
        self.halves[th.next].prev = t2;
        self.halves[t].next = t2;
        self.halves[h].twin = t2;
        self.halves[t].twin = h2;
        x
    }

    /// Connect the origins of `a` and `b`, which share a face, splitting it. Returns the new
    /// half-edge from `a`'s origin, which bounds the new face.
    pub(crate) fn connect(&mut self, a: usize, b: usize, piece: Piece) -> usize {
        let (n, m) = (self.halves.len(), self.halves.len() + 1);
        let (ha, hb) = (self.halves[a], self.halves[b]);
        let face = ha.face;
        let reverse = match piece {
            Piece::Loop(id, forward) => Piece::Loop(id, !forward),
            other => other,
        };
        self.halves.push(Half {
            twin: m,
            next: b,
            prev: ha.prev,
            origin: ha.origin,
            face,
            piece,
        });
        self.halves.push(Half {
            twin: n,
            next: a,
            prev: hb.prev,
            origin: hb.origin,
            face,
            piece: reverse,
        });
        self.halves[ha.prev].next = n;
        self.halves[hb.prev].next = m;
        self.halves[a].prev = m;
        self.halves[b].prev = n;
        let new = self.faces.len();
        self.faces.push(Face {
            half: n,
            triangle: self.faces[face].triangle,
        });
        self.faces[face].half = m;
        let mut h = n;
        loop {
            self.halves[h].face = new;
            h = self.halves[h].next;
            if h == n {
                break;
            }
        }
        n
    }

    /// Reverse the direction of loop `id`.
    pub fn reverse_loop(&mut self, id: LoopID) {
        for h in &mut self.halves {
            if let Piece::Loop(l, forward) = h.piece
                && l == id
            {
                h.piece = Piece::Loop(l, !forward);
            }
        }
    }

    /// Input triangles that loop `id` passes through.
    pub fn triangles(&self, id: LoopID) -> HashSet<FaceID> {
        self.halves
            .iter()
            .filter(|h| h.piece.wall() == Some(id))
            .map(|h| self.faces[h.face].triangle)
            .collect()
    }

    /// The end vertex of half-edge `h`.
    pub fn target(&self, h: usize) -> usize {
        self.halves[self.halves[h].twin].origin
    }

    /// The loop piece after `h` along its loop, if the loop continues there.
    pub fn next_along(&self, h: usize) -> Option<usize> {
        let piece = self.halves[h].piece;
        let back = self.halves[h].twin;
        self.outgoing(self.target(h))
            .into_iter()
            .find(|&o| o != back && self.halves[o].piece == piece)
    }

    /// The loop piece before `h` along its loop, if any.
    pub fn previous_along(&self, h: usize) -> Option<usize> {
        let Piece::Loop(id, forward) = self.halves[h].piece else {
            return None;
        };
        self.outgoing(self.halves[h].origin)
            .into_iter()
            .find(|&o| o != h && self.halves[o].piece == Piece::Loop(id, !forward))
            .map(|o| self.halves[o].twin)
    }

    /// The pieces of the loop through half-edge `h`, in its direction: from `h` if closed,
    /// from its first piece if open.
    pub fn chain(&self, h: usize) -> Vec<usize> {
        let h = match self.halves[h].piece {
            Piece::Loop(_, false) => self.halves[h].twin,
            _ => h,
        };
        let mut start = h;
        while let Some(p) = self.previous_along(start) {
            if p == h {
                break;
            }
            start = p;
        }
        let mut result = vec![start];
        while let Some(n) = self.next_along(*result.last().unwrap()) {
            if n == start {
                break;
            }
            result.push(n);
        }
        result
    }

    /// Every loop's pieces in its direction. Closed loops start anywhere; open ones at their
    /// first piece.
    pub fn chains(&self) -> HashMap<LoopID, Vec<usize>> {
        let mut result = HashMap::<LoopID, Vec<usize>>::new();
        let mut done = vec![false; self.halves.len()];
        for h in 0..self.halves.len() {
            let Piece::Loop(id, true) = self.halves[h].piece else {
                continue;
            };
            if done[h] {
                continue;
            }
            let mut start = h;
            while let Some(p) = self.previous_along(start) {
                if p == h {
                    break;
                }
                start = p;
            }
            let chain = result.entry(id).or_default();
            let mut current = Some(start);
            while let Some(c) = current {
                if done[c] {
                    break;
                }
                done[c] = true;
                chain.push(c);
                current = self.next_along(c);
            }
        }
        result
    }

    /// The region of every face (faces joined across non-loop edges), and their number.
    pub fn regions(&self) -> (Vec<usize>, usize) {
        let mut parent: Vec<usize> = (0..self.faces.len()).collect();
        for h in &self.halves {
            if h.piece.wall().is_none() {
                join(&mut parent, h.face, self.halves[h.twin].face);
            }
        }
        let mut ids = HashMap::new();
        let regions = (0..self.faces.len())
            .map(|f| {
                let r = root(&mut parent, f);
                let next = ids.len();
                *ids.entry(r).or_insert(next)
            })
            .collect();
        (regions, ids.len())
    }

    /// Euler characteristic of every region after cutting along all loops.
    pub fn euler(&self, regions: &[usize], count: usize) -> Vec<i64> {
        let mut chi = vec![0i64; count];
        for &r in regions {
            chi[r] += 1;
        }
        // Corners are identified by the half-edge leaving them in their face.
        let mut parent: Vec<usize> = (0..self.halves.len()).collect();
        for (i, h) in self.halves.iter().enumerate() {
            let r = regions[h.face];
            if h.piece.wall().is_some() {
                chi[r] -= 1;
            } else if i < h.twin {
                chi[r] -= 1;
                let t = &self.halves[h.twin];
                join(&mut parent, i, t.next);
                join(&mut parent, h.next, h.twin);
            }
        }
        for i in 0..self.halves.len() {
            if root(&mut parent, i) == i {
                chi[regions[self.halves[i].face]] += 1;
            }
        }
        chi
    }
}

mod realize;
mod refine;
pub(crate) use realize::Embedding;

#[cfg(test)]
mod tests;
