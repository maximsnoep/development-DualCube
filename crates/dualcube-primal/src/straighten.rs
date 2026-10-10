//! Straightening of layout paths by intrinsic edge flips.
//!
//! Implements the FlipOut algorithm of Sharp and Crane ("You Can Find Geodesic Paths in Triangle
//! Meshes by Just Flipping Edges", 2020) for the network of layout paths: every path is an edge path
//! in an intrinsic triangulation of the granulated mesh, and is shortened by flipping the edges in the
//! wedge of a joint with angle less than pi. Only joints that are not blocked by other paths are
//! shortened, so paths never cross, never touch, and keep their cyclic order around the corners. The
//! patches thus keep their topology, while their boundaries become (locally) geodesic.
//!
//! Afterwards, the straightened paths are traced over the granulated mesh (using signposts: the
//! direction of every intrinsic edge at its root), and the mesh is cut along them, such that the paths
//! are edge paths of the granulated mesh again.

use crate::layout::{Layout, LayoutError};
use dualcube_types::prelude::*;
use std::f64::consts::PI;
use std::time::Instant;

// Joints with angles larger than this are considered straight.
const STRAIGHT: f64 = PI - 1e-6;
// Paths along sharp features are kept (see `SHARP_ANGLE`).
const FEATURE_ANGLE: f64 = crate::layout::SHARP_ANGLE;

// Why a straightening failed: paths too close to each other (their indices), or anything else.
enum Failure {
    Conflict(HashSet<usize>),
    Other,
}

/// Statistics of a path straightening.
#[derive(Clone, Copy, Debug, Default)]
pub struct StraightenStats {
    pub flips: usize,
    pub length_before: f64,
    pub length_after: f64,
}

// A triangulation in halfedge representation with edge lengths and signposts. Faces are implicit (cycles of `next`).
#[derive(Clone, Debug)]
struct Intrinsic {
    next: Vec<usize>,
    twin: Vec<usize>,
    root: Vec<usize>,
    length: Vec<f64>,
    // Direction of every halfedge at its root: the angle (counter-clockwise) from the first outgoing halfedge of the root.
    signpost: Vec<f64>,
    // Some outgoing halfedge of every vertex.
    out: Vec<usize>,
    // Total angle around every vertex.
    theta: Vec<f64>,
}

// The state of the halfedges and vertices touched by a flip, to undo it.
struct FlipRecord {
    halfedges: [(usize, usize, usize, f64, f64); 6],
    verts: [(usize, usize); 2],
}

fn law_of_cosines(a: f64, b: f64, opposite: f64) -> f64 {
    ((a * a + b * b - opposite * opposite) / (2. * a * b))
        .clamp(-1., 1.)
        .acos()
}

impl Intrinsic {
    fn tip(&self, h: usize) -> usize {
        self.root[self.next[h]]
    }

    fn prev(&self, h: usize) -> usize {
        self.next[self.next[h]]
    }

    // Next outgoing halfedge (counter-clockwise) around the root of `h`.
    fn ccw(&self, h: usize) -> usize {
        self.twin[self.prev(h)]
    }

    fn edge(&self, h: usize) -> usize {
        h.min(self.twin[h])
    }

    // Angle at the root of `h`, in the face of `h`.
    fn corner(&self, h: usize) -> f64 {
        law_of_cosines(
            self.length[h],
            self.length[self.prev(h)],
            self.length[self.next[h]],
        )
    }

    fn degree(&self, v: usize) -> usize {
        let start = self.out[v];
        let mut h = start;
        let mut degree = 0;
        loop {
            degree += 1;
            h = self.ccw(h);
            if h == start || degree > 10_000 {
                return degree;
            }
        }
    }

    // Outgoing halfedges of the root of `start`, counter-clockwise from `start` up to (and including) `end`.
    fn sweep(&self, start: usize, end: usize) -> Option<Vec<usize>> {
        let mut sweep = vec![start];
        let mut h = start;
        while h != end {
            h = self.ccw(h);
            if h == start || sweep.len() > 10_000 {
                return None;
            }
            sweep.push(h);
        }
        Some(sweep)
    }

    fn sweep_angle(&self, sweep: &[usize]) -> f64 {
        sweep[..sweep.len() - 1]
            .iter()
            .map(|&h| self.corner(h))
            .sum()
    }

    fn from_mesh(
        mesh: &Mesh<INPUT>,
    ) -> (
        Self,
        HashMap<VertID, usize>,
        Vec<VertID>,
        HashMap<EdgeID, usize>,
    ) {
        let verts = mesh.vert_ids();
        let vert_index: HashMap<VertID, usize> =
            verts.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        let edges = mesh.edge_ids();
        let edge_index: HashMap<EdgeID, usize> =
            edges.iter().enumerate().map(|(i, &e)| (e, i)).collect();

        let mut triangulation = Self {
            next: edges.iter().map(|&e| edge_index[&mesh.next(e)]).collect(),
            twin: edges.iter().map(|&e| edge_index[&mesh.twin(e)]).collect(),
            root: edges.iter().map(|&e| vert_index[&mesh.root(e)]).collect(),
            length: edges
                .iter()
                .map(|&e| (mesh.position(mesh.toor(e)) - mesh.position(mesh.root(e))).norm())
                .collect(),
            signpost: vec![0.; edges.len()],
            out: vec![usize::MAX; verts.len()],
            theta: vec![0.; verts.len()],
        };
        for h in 0..edges.len() {
            triangulation.out[triangulation.root[h]] = h;
        }
        for v in 0..verts.len() {
            let start = triangulation.out[v];
            let mut h = start;
            let mut angle = 0.;
            loop {
                triangulation.signpost[h] = angle;
                angle += triangulation.corner(h);
                h = triangulation.ccw(h);
                if h == start {
                    break;
                }
            }
            triangulation.theta[v] = angle;
        }
        (triangulation, vert_index, verts, edge_index)
    }

    // Flip the edge of `h` (in its diamond). `h` and its twin are reused for the new edge.
    fn flip(&mut self, h0: usize) -> FlipRecord {
        let (h1, h2) = (self.next[h0], self.prev(h0));
        let g0 = self.twin[h0];
        let (g1, g2) = (self.next[g0], self.prev(g0));
        let (b, n) = (self.root[h0], self.root[g0]);
        let (p, q) = (self.root[h2], self.root[g2]);

        let record = FlipRecord {
            halfedges: [h0, h1, h2, g0, g1, g2].map(|h| {
                (
                    h,
                    self.next[h],
                    self.root[h],
                    self.length[h],
                    self.signpost[h],
                )
            }),
            verts: [(b, self.out[b]), (n, self.out[n])],
        };

        // Unfold the diamond: b at the origin, n on the x-axis, p above, q below.
        let l = self.length[h0];
        let place = |to_b: f64, to_n: f64, side: f64| {
            let x = (l * l + to_b * to_b - to_n * to_n) / (2. * l);
            let y = side * (to_b * to_b - x * x).max(0.).sqrt();
            (x, y)
        };
        let pp = place(self.length[h2], self.length[h1], 1.);
        let qq = place(self.length[g1], self.length[g2], -1.);
        let new_length = ((pp.0 - qq.0).powi(2) + (pp.1 - qq.1).powi(2)).sqrt();

        // New faces (q, n, p) and (p, b, q).
        self.root[h0] = p;
        self.root[g0] = q;
        self.next[g2] = h1;
        self.next[h1] = h0;
        self.next[h0] = g2;
        self.next[h2] = g1;
        self.next[g1] = g0;
        self.next[g0] = h2;
        self.length[h0] = new_length;
        self.length[g0] = new_length;
        if self.out[b] == h0 {
            self.out[b] = g1;
        }
        if self.out[n] == g0 {
            self.out[n] = h1;
        }
        self.signpost[h0] = (self.signpost[h2] + self.corner(h2)).rem_euclid(self.theta[p]);
        self.signpost[g0] = (self.signpost[g2] + self.corner(g2)).rem_euclid(self.theta[q]);

        record
    }

    fn undo(&mut self, record: &FlipRecord) {
        for &(h, next, root, length, signpost) in &record.halfedges {
            self.next[h] = next;
            self.root[h] = root;
            self.length[h] = length;
            self.signpost[h] = signpost;
        }
        for &(v, out) in &record.verts {
            self.out[v] = out;
        }
    }
}

// A network of paths in the intrinsic triangulation (every path is a sequence of halfedges).
struct Network {
    triangulation: Intrinsic,
    paths: Vec<Vec<usize>>,
    path_verts: HashMap<usize, usize>,
    path_edges: HashSet<usize>,
    // Path edges that may not be moved (e.g., because they follow sharp features).
    pinned: HashSet<usize>,
    flips: usize,
}

impl Network {
    fn new(triangulation: Intrinsic, paths: Vec<Vec<usize>>, pinned: HashSet<usize>) -> Self {
        let mut network = Self {
            triangulation,
            paths,
            path_verts: HashMap::new(),
            path_edges: HashSet::new(),
            pinned,
            flips: 0,
        };
        for path in network.paths.clone() {
            network.register(&path, 1);
        }
        network
    }

    // Add (`sign` = 1) or remove (`sign` = -1) the vertices and edges of a sequence of halfedges.
    fn register(&mut self, halfedges: &[usize], sign: isize) {
        for &h in halfedges {
            let edge = self.triangulation.edge(h);
            if sign > 0 {
                self.path_edges.insert(edge);
            } else {
                self.path_edges.remove(&edge);
            }
            for v in [self.triangulation.root[h], self.triangulation.tip(h)] {
                let count = self.path_verts.entry(v).or_insert(0);
                *count = count.saturating_add_signed(sign);
                if *count == 0 {
                    self.path_verts.remove(&v);
                }
            }
        }
    }

    fn length(&self) -> f64 {
        self.paths
            .iter()
            .flatten()
            .map(|&h| self.triangulation.length[h])
            .sum()
    }

    // Try to shorten the joint between halfedges `i - 1` and `i` of path `p`. Returns whether it was shortened.
    fn flip_out(&mut self, p: usize, i: usize) -> bool {
        let t = &self.triangulation;
        let to_a = t.twin[self.paths[p][i - 1]];
        let to_c = self.paths[p][i];
        if self.pinned.contains(&t.edge(to_a)) || self.pinned.contains(&t.edge(to_c)) {
            return false;
        }

        // The wedge with the smallest angle.
        let (Some(left), Some(right)) = (t.sweep(to_c, to_a), t.sweep(to_a, to_c)) else {
            return false;
        };
        let (left_angle, right_angle) = (t.sweep_angle(&left), t.sweep_angle(&right));
        let (wedge, angle, starts_at_a) = if left_angle <= right_angle {
            (left, left_angle, false)
        } else {
            (right, right_angle, true)
        };
        if angle >= STRAIGHT {
            return false;
        }

        // The joint must be flexible: no other paths inside its wedge.
        if wedge[1..wedge.len() - 1]
            .iter()
            .any(|&h| self.path_edges.contains(&t.edge(h)))
        {
            return false;
        }

        // Flip edges out of the wedge.
        let start = wedge[0];
        let end = wedge[wedge.len() - 1];
        let mut records = vec![];
        let arc = loop {
            let t = &self.triangulation;
            let Some(sweep) = t.sweep(start, end) else {
                break None;
            };
            let flippable = (1..sweep.len() - 1).find(|&j| {
                let n = t.tip(sweep[j]);
                let beta = t.corner(t.prev(sweep[j - 1])) + t.corner(t.next[sweep[j]]);
                beta < STRAIGHT && n != t.root[start] && t.degree(n) > 2
            });
            match flippable {
                Some(j) if records.len() < 10_000 => {
                    records.push(self.triangulation.flip(sweep[j]));
                }
                Some(_) => break None,
                None => {
                    // Any remaining angle below pi means the wedge could not be emptied.
                    let blocked = (1..sweep.len() - 1).any(|j| {
                        t.corner(t.prev(sweep[j - 1])) + t.corner(t.next[sweep[j]]) < STRAIGHT
                    });
                    break (!blocked).then(|| {
                        sweep[..sweep.len() - 1]
                            .iter()
                            .map(|&h| t.next[h])
                            .collect::<Vec<_>>()
                    });
                }
            }
        };

        // The new path (along the outer arc of the wedge) may not touch other paths.
        let t = &self.triangulation;
        let valid = arc.as_ref().is_some_and(|arc| {
            arc[..arc.len() - 1]
                .iter()
                .all(|&h| !self.path_verts.contains_key(&t.tip(h)))
                && (arc.len() > 1 || !self.path_edges.contains(&t.edge(arc[0])))
        });
        if !valid {
            for record in records.iter().rev() {
                self.triangulation.undo(record);
            }
            return false;
        }

        let arc = arc.unwrap();
        let segment = if starts_at_a {
            arc
        } else {
            arc.iter()
                .rev()
                .map(|&h| self.triangulation.twin[h])
                .collect()
        };
        let old = vec![self.paths[p][i - 1], self.paths[p][i]];
        self.register(&old, -1);
        self.register(&segment, 1);
        self.paths[p].splice(i - 1..=i, segment);
        self.flips += records.len();
        true
    }

    // Shorten all paths until every (flexible) joint is straight, or the budget of flips is used.
    fn straighten(&mut self, max_sweeps: usize, max_flips: usize) {
        for _ in 0..max_sweeps {
            let mut changed = false;
            for p in 0..self.paths.len() {
                let mut i = 1;
                while i < self.paths[p].len() {
                    if self.flips > max_flips {
                        return;
                    }
                    if self.flip_out(p, i) {
                        changed = true;
                        i = (i - 1).max(1);
                    } else {
                        i += 1;
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }
}

// Where a traced intrinsic edge crosses an edge of the input triangulation: the input halfedge (of the face
// that is exited) and the position along it.
type Crossing = (usize, f64);

fn cross2(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

fn sub2(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

// Intersection of the ray `p + s * d` with the segment `a + t * (b - a)`: (s, t).
fn ray_segment(p: [f64; 2], d: [f64; 2], a: [f64; 2], b: [f64; 2]) -> Option<(f64, f64)> {
    let e = sub2(b, a);
    let denom = cross2(d, e);
    if denom.abs() < 1e-300 {
        return None;
    }
    let ap = sub2(a, p);
    Some((cross2(ap, e) / denom, cross2(ap, d) / denom))
}

// Third vertex of a triangle, on the left of the segment from `a` to `b`, at distances `to_a` and `to_b`.
fn apex(a: [f64; 2], b: [f64; 2], to_a: f64, to_b: f64) -> [f64; 2] {
    let ab = sub2(b, a);
    let l = (ab[0] * ab[0] + ab[1] * ab[1]).sqrt();
    let x = (l * l + to_a * to_a - to_b * to_b) / (2. * l);
    let y = (to_a * to_a - x * x).max(0.).sqrt();
    let (ux, uy) = (ab[0] / l, ab[1] / l);
    [a[0] + x * ux - y * uy, a[1] + x * uy + y * ux]
}

// Trace the intrinsic edge from vertex `u` (direction `signpost`, length `length`) over the input triangulation.
// Returns the vertex where it ends, and the input edges it crosses.
fn trace(
    input: &Intrinsic,
    u: usize,
    signpost: f64,
    length: f64,
) -> Option<(usize, Vec<Crossing>)> {
    // The input corner at `u` that contains the direction.
    let start = input.out[u];
    let mut h = start;
    loop {
        let next = input.ccw(h);
        let upper = if next == start {
            input.theta[u]
        } else {
            input.signpost[next]
        };
        if signpost < upper || next == start {
            break;
        }
        h = next;
    }
    let corner = input.corner(h);
    if !corner.is_finite() {
        return None;
    }
    let theta = (signpost - input.signpost[h]).clamp(0., corner);
    let tolerance = 1e-9 * length.max(1e-12);

    // Along an input edge.
    if theta < 1e-9 && (length - input.length[h]).abs() < 1e-6 * length {
        return Some((input.tip(h), vec![]));
    }
    let prev = input.prev(h);
    if corner - theta < 1e-9 && (length - input.length[prev]).abs() < 1e-6 * length {
        return Some((input.root[prev], vec![]));
    }

    // Lay out the face of `h` with `u` at the origin.
    let x = [input.length[h], 0.];
    let y = [
        input.length[prev] * corner.cos(),
        input.length[prev] * corner.sin(),
    ];
    let d = [theta.cos(), theta.sin()];
    let mut p = [0., 0.];
    let mut traveled = 0.;
    // The halfedge (of the current face) to exit through, its endpoints, and the opposite vertex.
    let (mut e, mut a, mut b, mut opposite) = (input.next[h], x, y, (u, [0., 0.]));
    let mut crossings = vec![];

    for _ in 0..1_000_000 {
        let (s, t) = ray_segment(p, d, a, b)?;
        if traveled + s >= length - tolerance
            || !(0. ..=1.).contains(&t) && traveled + s >= length * 0.999
        {
            // The edge ends in the current face, at one of its vertices.
            let end = [
                p[0] + (length - traveled) * d[0],
                p[1] + (length - traveled) * d[1],
            ];
            let candidates = [(input.root[e], a), (input.tip(e), b), opposite];
            let (vertex, position) = candidates.into_iter().min_by_key(|(_, q)| {
                let diff = sub2(*q, end);
                OrderedFloat(diff[0] * diff[0] + diff[1] * diff[1])
            })?;
            let diff = sub2(position, end);
            if (diff[0] * diff[0] + diff[1] * diff[1]).sqrt() > 1e-5 * length.max(1e-12) {
                return None;
            }
            return Some((vertex, crossings));
        }
        if !(-1e-9..=1. + 1e-9).contains(&t) {
            return None;
        }
        let t = t.clamp(1e-12, 1. - 1e-12);
        crossings.push((e, t));
        traveled += s;
        p = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];

        // Continue in the neighboring face (b, a, c).
        let g = input.twin[e];
        let c = apex(
            b,
            a,
            input.length[input.prev(g)],
            input.length[input.next[g]],
        );
        let ac = input.next[g];
        let cb = input.prev(g);
        let hit = |from: [f64; 2], to: [f64; 2]| {
            ray_segment(p, d, from, to)
                .filter(|&(s, t)| s > 1e-12 && (-1e-9..=1. + 1e-9).contains(&t))
        };
        let (vertex_a, vertex_b, vertex_c) = (input.root[e], input.root[g], input.root[cb]);
        let next = match (hit(a, c), hit(c, b)) {
            (Some((s1, _)), Some((s2, _))) if s2 < s1 => (cb, c, b, (vertex_a, a)),
            (Some(_), _) => (ac, a, c, (vertex_b, b)),
            (None, Some(_)) => (cb, c, b, (vertex_a, a)),
            (None, None) => {
                // The edge ends at the apex of the neighboring face.
                let diff = sub2(c, p);
                let remaining = length - traveled;
                let distance = (diff[0] * diff[0] + diff[1] * diff[1]).sqrt();
                if (distance - remaining).abs() > 1e-5 * length.max(1e-12) {
                    return None;
                }
                return Some((vertex_c, crossings));
            }
        };
        (e, a, b, opposite) = next;
    }
    None
}

fn polygon_area(polygon: &[(usize, [f64; 2])]) -> f64 {
    (0..polygon.len())
        .map(|i| cross2(polygon[i].1, polygon[(i + 1) % polygon.len()].1))
        .sum::<f64>()
        / 2.
}

// Ear clipping of a (convex, possibly with collinear vertices) polygon. An ear is only clipped if the rest of the
// polygon does not degenerate (e.g., clipping the only vertex off a line of collinear vertices).
fn triangulate(polygon: &[(usize, [f64; 2])]) -> Option<Vec<[usize; 3]>> {
    let mut polygon = polygon.to_vec();
    let mut triangles = vec![];
    while polygon.len() > 3 {
        let n = polygon.len();
        let area = |i: usize| {
            let (a, b, c) = (
                polygon[(i + n - 1) % n].1,
                polygon[i].1,
                polygon[(i + 1) % n].1,
            );
            cross2(sub2(b, a), sub2(c, a))
        };
        let remainder = |i: usize| {
            let mut rest = polygon.clone();
            rest.remove(i);
            polygon_area(&rest)
        };
        let ear = (0..n)
            .filter(|&i| area(i) > 0. && remainder(i) > 0.)
            .max_by_key(|&i| OrderedFloat(area(i).min(remainder(i))))?;
        triangles.push([
            polygon[(ear + n - 1) % n].0,
            polygon[ear].0,
            polygon[(ear + 1) % n].0,
        ]);
        polygon.remove(ear);
    }
    let (a, b, c) = (polygon[0].1, polygon[1].1, polygon[2].1);
    if cross2(sub2(b, a), sub2(c, a)) <= 0. {
        return None;
    }
    triangles.push([polygon[0].0, polygon[1].0, polygon[2].0]);
    Some(triangles)
}

#[cfg(test)]
mod tests {
    #[test]
    fn triangulate_polygon_with_collinear_vertices() {
        // A quadrilateral with a vertex very close to a corner, on one of its sides.
        let polygon = vec![
            (0, [0.0, 0.0]),
            (1, [2.7e-8, 0.0]),
            (2, [1.0, 0.0]),
            (3, [0.084, 0.916]),
        ];
        assert_eq!(super::triangulate(&polygon).map(|t| t.len()), Some(2));
        // Many collinear vertices on one side.
        let polygon = vec![
            (0, [0.0, 0.0]),
            (1, [0.25, 0.0]),
            (2, [0.5, 0.0]),
            (3, [0.75, 0.0]),
            (4, [1.0, 0.0]),
            (5, [0.0, 1.0]),
        ];
        assert_eq!(super::triangulate(&polygon).map(|t| t.len()), Some(4));
    }
}

impl Layout {
    /// Straighten all paths of the layout (to locally shortest paths, keeping the patch topology),
    /// and cut the granulated mesh along them. If that fails, the paths are straightened one at a time, and those that
    /// cannot be straightened stay as they are. The layout is unchanged if this fails (e.g., on a mesh with zero-length
    /// edges, where the geometry of the triangles is undefined).
    pub fn straighten_paths(&mut self) -> Result<StraightenStats, LayoutError> {
        let timer = Instant::now();
        let mesh = &self.granulated_mesh;
        if mesh
            .edge_ids()
            .into_iter()
            .any(|edge| !(mesh.distance(mesh.root(edge), mesh.toor(edge)) > 0.))
        {
            info!(
                "Layout::straighten_paths: the mesh has zero-length edges (degenerate triangles)"
            );
            return Err(LayoutError::InvalidPath);
        }
        let backup = self.clone();
        // Paths that are straightened (numerically) too close to each other cannot be cut into the mesh: they are kept
        // as they are (frozen) and the others are straightened again.
        let mut frozen = HashSet::new();
        let mut result = Err(LayoutError::InvalidPath);
        for _ in 0..8 {
            match self.straighten_paths_inner(&frozen) {
                Ok(stats) => {
                    result = Ok(stats);
                    break;
                }
                Err(Failure::Conflict(paths)) if !paths.is_subset(&frozen) => {
                    *self = backup.clone();
                    frozen.extend(paths);
                }
                Err(_) => {
                    *self = backup.clone();
                    break;
                }
            }
        }
        if result.is_err() {
            // One path at a time: every path is straightened together with those straightened so far, if that works.
            let structure = &backup.polycube_ref.structure;
            let count = structure
                .edge_ids()
                .into_iter()
                .filter(|&e| e.raw() < structure.twin(e).raw())
                .count();
            let mut straight: HashSet<usize> = HashSet::new();
            let mut best: Option<(Self, StraightenStats)> = None;
            for path in 0..count {
                let frozen = (0..count)
                    .filter(|&q| q != path && !straight.contains(&q))
                    .collect();
                *self = backup.clone();
                if let Ok(stats) = self.straighten_paths_inner(&frozen) {
                    straight.insert(path);
                    best = Some((self.clone(), stats));
                }
            }
            info!(
                "Layout::straighten_paths: straightened {} of {count} paths one at a time",
                straight.len()
            );
            match best {
                Some((layout, stats)) => {
                    *self = layout;
                    result = Ok(stats);
                }
                None => *self = backup,
            }
        }
        info!(
            "Layout::straighten_paths: result={result:?} elapsed={:?}",
            timer.elapsed()
        );
        result
    }

    // See `straighten_paths`: the paths with the given indices (in the order of `polycube_edges`) are kept.
    fn straighten_paths_inner(
        &mut self,
        frozen: &HashSet<usize>,
    ) -> Result<StraightenStats, Failure> {
        let fail = |reason: &str| {
            debug!("Layout::straighten_paths: {reason}");
            Failure::Other
        };
        let mesh = &self.granulated_mesh;
        let (input, vert_index, verts, edge_index) = Intrinsic::from_mesh(mesh);

        // One path per polycube edge (in one direction).
        let polycube = &self.polycube_ref.structure;
        let polycube_edges = polycube
            .edge_ids()
            .into_iter()
            .filter(|&e| e.raw() < polycube.twin(e).raw())
            .collect_vec();
        let paths = polycube_edges
            .iter()
            .map(|e| {
                let path = self.edge_to_path.get(e).ok_or(LayoutError::InvalidPath)?;
                if path.len() < 2 {
                    return Err(LayoutError::InvalidPath);
                }
                path.windows(2)
                    .map(|w| {
                        mesh.edge_between_verts(w[0], w[1])
                            .map(|(edge, _)| edge_index[&edge])
                            .ok_or(LayoutError::InvalidPath)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| Failure::Other)?;

        // Path edges along sharp features, and the frozen paths, stay in place.
        let input_edges = mesh.edge_ids();
        let pinned = paths
            .iter()
            .enumerate()
            .flat_map(|(p, path)| path.iter().map(move |&h| (p, h)))
            .map(|(p, h)| (p, input.edge(h)))
            .filter(|&(p, edge)| {
                frozen.contains(&p) || mesh.dihedral(input_edges[edge]) > FEATURE_ANGLE
            })
            .map(|(_, edge)| edge)
            .collect();

        let mut network = Network::new(input.clone(), paths, pinned);
        let length_before = network.length();
        // Typically, the number of flips grows like m^1.5 for paths of m edges (Sharp and Crane), so this is generous.
        let path_edges = network.paths.iter().map(Vec::len).sum::<usize>();
        network.straighten(100, 1_000_000.max(200 * path_edges));
        let length_after = network.length();

        // Trace the paths over the granulated mesh: every path is a sequence of vertices and crossings,
        // and every piece between two consecutive points lies in a face (unless it is an input edge).
        #[derive(Clone, Copy, PartialEq)]
        enum Point {
            Vertex(usize),
            // Crossing of an (undirected) input edge, given by its canonical halfedge and the position along it.
            Crossing(usize, f64),
        }
        let face_of = |h: usize| mesh.face(input_edges[h]);
        let mut traced_paths = vec![];
        for path in &network.paths {
            let t = &network.triangulation;
            let mut points = vec![Point::Vertex(t.root[path[0]])];
            let mut pieces = vec![];
            for &h in path {
                let (end, crossings) = trace(&input, t.root[h], t.signpost[h], t.length[h])
                    .ok_or_else(|| fail("tracing an edge failed"))?;
                if end != t.tip(h) {
                    return Err(fail("a traced edge ends at the wrong vertex"));
                }
                // The piece up to a crossing lies in the face that is exited.
                for &(e, along) in &crossings {
                    let twin = input.twin[e];
                    points.push(if e < twin {
                        Point::Crossing(e, along)
                    } else {
                        Point::Crossing(twin, 1. - along)
                    });
                    pieces.push(Some(face_of(e)));
                }
                points.push(Point::Vertex(end));
                pieces.push(crossings.last().map(|&(e, _)| face_of(input.twin[e])));
            }
            traced_paths.push((points, pieces));
        }

        // New vertices: the input vertices, followed by all crossings.
        let mut positions = verts.iter().map(|&v| mesh.position(v)).collect_vec();
        let mut point_ids: Vec<Vec<usize>> = vec![];
        // For every input edge (canonical halfedge), the crossings on it: (position along it, vertex id).
        let mut edge_points: HashMap<usize, Vec<(f64, usize)>> = HashMap::new();
        // The pieces of the paths inside every input face (chords between two points on its boundary).
        // The pieces of the paths inside every input face (chords between two points on its boundary), with their path.
        let mut face_chords: HashMap<FaceID, Vec<([usize; 2], usize)>> = HashMap::new();
        for (path_index, (points, pieces)) in traced_paths.iter().enumerate() {
            let ids = points
                .iter()
                .map(|&point| match point {
                    Point::Vertex(v) => v,
                    Point::Crossing(e, along) => {
                        let edge = input_edges[e];
                        let (from, to) = (
                            mesh.position(mesh.root(edge)),
                            mesh.position(mesh.toor(edge)),
                        );
                        positions.push(from + (to - from) * along);
                        let id = positions.len() - 1;
                        edge_points.entry(e).or_default().push((along, id));
                        id
                    }
                })
                .collect_vec();
            for (k, piece) in pieces.iter().enumerate() {
                if let Some(face) = piece {
                    face_chords
                        .entry(*face)
                        .or_default()
                        .push(([ids[k], ids[k + 1]], path_index));
                }
            }
            point_ids.push(ids);
        }

        // Cut every face along its chords, and triangulate the resulting polygons.
        let mut faces = vec![];
        for face in mesh.face_ids() {
            let sides = mesh.edges(face).collect_vec();
            let corners = [[0., 0.], [1., 0.], [0., 1.]];
            let mut polygon: Vec<(usize, [f64; 2])> = vec![];
            for (side, &edge) in sides.iter().enumerate() {
                let (from, to) = (corners[side], corners[(side + 1) % 3]);
                polygon.push((vert_index[&mesh.root(edge)], from));
                let h = edge_index[&edge];
                let canonical = h.min(input.twin[h]);
                let mut points = edge_points
                    .get(&canonical)
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .map(|(along, id)| (if canonical == h { along } else { 1. - along }, id))
                    .collect_vec();
                points.sort_by_key(|&(along, _)| OrderedFloat(along));
                for (along, id) in points {
                    polygon.push((
                        id,
                        [
                            from[0] + along * (to[0] - from[0]),
                            from[1] + along * (to[1] - from[1]),
                        ],
                    ));
                }
            }

            let mut polygons = vec![polygon];
            let chords = face_chords.get(&face).map_or(&[][..], Vec::as_slice);
            for &([i, j], _) in chords {
                let Some(index) = polygons.iter().position(|polygon| {
                    polygon.iter().any(|&(id, _)| id == i) && polygon.iter().any(|&(id, _)| id == j)
                }) else {
                    // Paths that come (numerically) too close in this face: freeze them, and try again.
                    debug!("Layout::straighten_paths: conflicting paths in a face");
                    return Err(Failure::Conflict(chords.iter().map(|&(_, p)| p).collect()));
                };
                let polygon = polygons.swap_remove(index);
                let pi = polygon.iter().position(|&(id, _)| id == i).unwrap();
                let pj = polygon.iter().position(|&(id, _)| id == j).unwrap();
                let (lo, hi) = (pi.min(pj), pi.max(pj));
                if hi - lo == 1 || (lo == 0 && hi == polygon.len() - 1) {
                    // Chord along the boundary of the polygon: nothing to cut.
                    polygons.push(polygon);
                    continue;
                }
                let first = polygon[lo..=hi].to_vec();
                let second = polygon[hi..]
                    .iter()
                    .chain(polygon[..=lo].iter())
                    .copied()
                    .collect_vec();
                polygons.push(first);
                polygons.push(second);
            }

            for polygon in polygons {
                for triangle in
                    triangulate(&polygon).ok_or_else(|| fail("triangulating a polygon failed"))?
                {
                    faces.push(triangle.to_vec());
                }
            }
        }

        let Ok((new_mesh, vmap, _)) = Mesh::<INPUT>::from(&faces, &positions) else {
            return Err(fail("the cut mesh is not a valid mesh"));
        };
        let key = |id: usize| vmap.key(id).copied().ok_or(Failure::Other);

        // Update the layout.
        let mut vert_to_corner = bimap::BiHashMap::new();
        for (&corner, vert) in &self.vert_to_corner {
            vert_to_corner.insert(corner, key(vert_index[vert])?);
        }
        let mut edge_to_path = HashMap::new();
        for (&edge, ids) in polycube_edges.iter().zip(&point_ids) {
            let path = ids
                .iter()
                .map(|&id| key(id))
                .collect::<Result<Vec<_>, _>>()?;
            edge_to_path.insert(
                polycube.twin(edge),
                path.iter().rev().copied().collect_vec(),
            );
            edge_to_path.insert(edge, path);
        }
        self.granulated_mesh = new_mesh;
        // The faces of the new mesh are not tracked (by loop region); later path computations are unrestricted.
        self.regions = None;
        self.vert_to_corner = vert_to_corner;
        self.edge_to_path = edge_to_path;
        self.face_to_patch.clear();
        self.assign_all_patches().map_err(|err| {
            fail(&format!(
                "the patches of the cut mesh are invalid ({err:?})"
            ))
        })?;

        Ok(StraightenStats {
            flips: network.flips,
            length_before,
            length_after,
        })
    }
}
