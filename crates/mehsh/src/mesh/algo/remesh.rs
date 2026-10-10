//! Adaptive isotropic remeshing of closed triangle meshes, to make the input of further computations well-behaved:
//! no huge or tiny triangles, no slivers, and no thousands of small triangles on flat parts.
//!
//! The method of Botsch and Kobbelt ("A remeshing approach to multiresolution modeling", 2004), with a target edge
//! length that adapts to the curvature (as in Dunyach et al., "Adaptive remeshing for real-time mesh deformation",
//! 2013). Every iteration
//!
//! 1. splits the edges longer than 4/3 of their target length,
//! 2. collapses the edges shorter than 4/5 of their target length (if that keeps the mesh a manifold of the same
//!    topology, does not fold triangles, and does not create long edges),
//! 3. flips edges towards valence 6,
//! 4. moves every vertex towards the centroid of its neighbors, in its tangent plane, and projects it back onto the
//!    input surface.
//!
//! The target length is short where the surface curves (the normal may turn by `angle_per_edge` over one edge) and
//! long on flat parts, between `min_length` and `max_ratio` times that. Sharp features (edges with a dihedral angle
//! above `feature_angle`) are kept: they are never flipped, only collapsed along themselves, their vertices only move
//! along them, and their end points and junctions (corners) never move.
//!
//! The result is checked (a valid closed manifold with the same Euler characteristic); see `RemeshReport`.

use crate::prelude::*;
use crate::utils::geom;
use itertools::Itertools;
use rustc_hash::{FxHashMap, FxHashSet};

/// Parameters of the remeshing (lengths relative to the diagonal of the bounding box).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RemeshParams {
    /// The shortest target edge length.
    pub min_length: f64,
    /// The longest target edge length (on flat parts), relative to the shortest.
    pub max_ratio: f64,
    /// The angle (radians) by which the normal may turn over one target edge length.
    pub angle_per_edge: f64,
    /// Mesh edges whose faces' normals differ by more than this angle (radians) are sharp features.
    pub feature_angle: f64,
    pub iterations: usize,
}

impl Default for RemeshParams {
    fn default() -> Self {
        Self {
            min_length: 0.006,
            max_ratio: 4.,
            angle_per_edge: 15f64.to_radians(),
            feature_angle: 60f64.to_radians(),
            iterations: 6,
        }
    }
}

/// The result of a remeshing.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RemeshReport {
    pub faces_before: usize,
    pub faces_after: usize,
    /// The smallest angle of any triangle (degrees).
    pub min_angle_before: f64,
    pub min_angle_after: f64,
    /// The largest distance between the surfaces (from the vertices of each to the other), relative to the diagonal.
    pub max_distance: f64,
}

// Remeshing state: triangles with their directed edges, faces around every vertex, and per-vertex attributes.
struct Work {
    pos: Vec<Vector3D>,
    // Target edge length.
    size: Vec<f64>,
    // On a sharp feature line, and a corner of the features (never moved or removed).
    on_feature: Vec<bool>,
    fixed: Vec<bool>,
    vert_alive: Vec<bool>,
    tris: Vec<[u32; 3]>,
    alive: Vec<bool>,
    // Directed edge -> its face.
    edges: FxHashMap<(u32, u32), u32>,
    vfaces: Vec<Vec<u32>>,
    // Sharp feature edges (undirected, smallest vertex first).
    features: FxHashSet<(u32, u32)>,
}

fn key(a: u32, b: u32) -> (u32, u32) {
    if a < b { (a, b) } else { (b, a) }
}

impl Work {
    fn add_face(&mut self, t: [u32; 3]) -> u32 {
        let f = self.tris.len() as u32;
        self.tris.push(t);
        self.alive.push(true);
        for i in 0..3 {
            self.edges.insert((t[i], t[(i + 1) % 3]), f);
            self.vfaces[t[i] as usize].push(f);
        }
        f
    }

    fn remove_face(&mut self, f: u32) {
        let t = self.tris[f as usize];
        self.alive[f as usize] = false;
        for i in 0..3 {
            self.edges.remove(&(t[i], t[(i + 1) % 3]));
            self.vfaces[t[i] as usize].retain(|&g| g != f);
        }
    }

    // The vertex of face `f` other than `a` and `b`.
    fn third(&self, f: u32, a: u32, b: u32) -> u32 {
        *self.tris[f as usize]
            .iter()
            .find(|&&v| v != a && v != b)
            .unwrap()
    }

    fn neighbors(&self, v: u32) -> Vec<u32> {
        self.vfaces[v as usize]
            .iter()
            .flat_map(|&f| self.tris[f as usize])
            .filter(|&w| w != v)
            .unique()
            .collect()
    }

    fn is_feature(&self, a: u32, b: u32) -> bool {
        self.features.contains(&key(a, b))
    }

    fn target(&self, a: u32, b: u32) -> f64 {
        0.5 * (self.size[a as usize] + self.size[b as usize])
    }

    fn length(&self, a: u32, b: u32) -> f64 {
        (self.pos[a as usize] - self.pos[b as usize]).norm()
    }

    fn normal_of(&self, t: [u32; 3], positions: &dyn Fn(u32) -> Vector3D) -> Vector3D {
        let (a, b, c) = (positions(t[0]), positions(t[1]), positions(t[2]));
        (b - a).cross(&(c - a))
    }

    // The undirected edges (once each).
    fn edge_list(&self) -> Vec<(u32, u32)> {
        self.edges.keys().filter(|(a, b)| a < b).copied().collect()
    }

    fn split(&mut self, a: u32, b: u32) -> bool {
        let (Some(&f0), Some(&f1)) = (self.edges.get(&(a, b)), self.edges.get(&(b, a))) else {
            return false;
        };
        let (c, d) = (self.third(f0, a, b), self.third(f1, a, b));
        let m = self.pos.len() as u32;
        self.pos
            .push(0.5 * (self.pos[a as usize] + self.pos[b as usize]));
        self.size.push(self.target(a, b));
        let feature = self.is_feature(a, b);
        self.on_feature.push(feature);
        self.fixed.push(false);
        self.vert_alive.push(true);
        self.vfaces.push(vec![]);
        self.remove_face(f0);
        self.remove_face(f1);
        self.add_face([a, m, c]);
        self.add_face([m, b, c]);
        self.add_face([b, m, d]);
        self.add_face([m, a, d]);
        if feature {
            self.features.remove(&key(a, b));
            self.features.insert(key(a, m));
            self.features.insert(key(m, b));
        }
        true
    }

    // Collapse the edge, removing `b` and moving `a` (unless it is constrained). Fails if that changes the topology,
    // folds a triangle, creates a long edge, or does not respect the features.
    fn collapse(&mut self, a: u32, b: u32) -> bool {
        let (ai, bi) = (a as usize, b as usize);
        if self.fixed[bi] || !self.vert_alive[ai] || !self.vert_alive[bi] {
            return false;
        }
        let along_feature = self.is_feature(a, b);
        if self.on_feature[bi] && !along_feature {
            return false;
        }
        let (Some(&f0), Some(&f1)) = (self.edges.get(&(a, b)), self.edges.get(&(b, a))) else {
            return false;
        };
        let (c, d) = (self.third(f0, a, b), self.third(f1, a, b));
        // Link condition: the common neighbors are exactly the two opposite vertices.
        let (na, nb) = (self.neighbors(a), self.neighbors(b));
        let common = na.iter().filter(|v| nb.contains(v)).count();
        if common != 2 || self.neighbors(c).len() <= 3 || self.neighbors(d).len() <= 3 {
            return false;
        }
        if na.len() + nb.len() < 8 {
            // (A tetrahedron-like neighborhood would collapse into a degenerate one.)
            return false;
        }
        // The new position of `a`.
        let p = if self.fixed[ai] || (self.on_feature[ai] && !along_feature) {
            self.pos[ai]
        } else {
            0.5 * (self.pos[ai] + self.pos[bi])
        };
        // No folded triangles and no long edges around the moved vertex.
        let moved = |v: u32| {
            if v == a || v == b {
                p
            } else {
                self.pos[v as usize]
            }
        };
        let original = |v: u32| self.pos[v as usize];
        for &f in self.vfaces[ai].iter().chain(&self.vfaces[bi]) {
            if f == f0 || f == f1 {
                continue;
            }
            let t = self.tris[f as usize];
            let before = self.normal_of(t, &original);
            let after = self.normal_of(t, &moved);
            if after.norm() <= 1e-12 * before.norm().max(1e-300)
                || before.normalize().dot(&after.normalize()) < 0.5
            {
                return false;
            }
            for &v in &t {
                if v != a
                    && v != b
                    && (p - self.pos[v as usize]).norm() > 4. / 3. * self.target(a, v)
                {
                    return false;
                }
            }
        }
        // Apply: remove the two faces, and replace `b` by `a` in the others.
        self.remove_face(f0);
        self.remove_face(f1);
        for f in self.vfaces[bi].clone() {
            let t = self.tris[f as usize];
            self.remove_face(f);
            self.add_face(t.map(|v| if v == b { a } else { v }));
        }
        for n in nb {
            if n != a && self.features.remove(&key(b, n)) {
                self.features.insert(key(a, n));
            }
        }
        self.features.remove(&key(a, b));
        self.pos[ai] = p;
        self.size[ai] = self.size[ai].min(self.size[bi]);
        self.on_feature[ai] |= along_feature;
        self.vert_alive[bi] = false;
        true
    }

    // Flip the edge if that brings the valences closer to 6 (`delaunay`: if the two angles opposite to it sum to more
    // than pi, which improves the smallest angle), and keeps the surface's shape.
    fn flip(&mut self, a: u32, b: u32, max_dihedral: f64, delaunay: bool) -> bool {
        if self.is_feature(a, b) {
            return false;
        }
        let (Some(&f0), Some(&f1)) = (self.edges.get(&(a, b)), self.edges.get(&(b, a))) else {
            return false;
        };
        let (c, d) = (self.third(f0, a, b), self.third(f1, a, b));
        if c == d || self.edges.contains_key(&(c, d)) || self.edges.contains_key(&(d, c)) {
            return false;
        }
        let valence = |v: u32| self.neighbors(v).len() as i64;
        let (va, vb, vc, vd) = (valence(a), valence(b), valence(c), valence(d));
        if va <= 3 || vb <= 3 {
            return false;
        }
        if delaunay {
            let angle = |at: u32| {
                let p = self.pos[at as usize];
                (self.pos[a as usize] - p).angle(&(self.pos[b as usize] - p))
            };
            if angle(c) + angle(d) <= std::f64::consts::PI + 1e-9 {
                return false;
            }
        } else {
            let deviation = |x: i64| (x - 6) * (x - 6);
            let before = deviation(va) + deviation(vb) + deviation(vc) + deviation(vd);
            let after =
                deviation(va - 1) + deviation(vb - 1) + deviation(vc + 1) + deviation(vd + 1);
            if after >= before {
                return false;
            }
        }
        // Only over (nearly) flat edges, and without folding.
        let position = |v: u32| self.pos[v as usize];
        let (n0, n1) = (
            self.normal_of(self.tris[f0 as usize], &position),
            self.normal_of(self.tris[f1 as usize], &position),
        );
        if n0.norm() <= 0. || n1.norm() <= 0. || n0.angle(&n1) > max_dihedral {
            return false;
        }
        let (m0, m1) = (
            self.normal_of([a, d, c], &position),
            self.normal_of([d, b, c], &position),
        );
        let reference = (n0.normalize() + n1.normalize()).normalize();
        if m0.norm() <= 0.
            || m1.norm() <= 0.
            || m0.normalize().dot(&reference) < 0.5
            || m1.normalize().dot(&reference) < 0.5
        {
            return false;
        }
        self.remove_face(f0);
        self.remove_face(f1);
        self.add_face([a, d, c]);
        self.add_face([d, b, c]);
        true
    }
}

// The closest point of a triangle mesh to a point (with the mesh's search structure).
fn closest_point<M: Tag>(mesh: &Mesh<M>, bvh: &FaceLocation<M>, p: Vector3D) -> Vector3D {
    let face = bvh.nearest(&[p.x, p.y, p.z]);
    let Some([a, b, c]) = mesh
        .vertices(face)
        .map(|v| mesh.position(v))
        .collect_array::<3>()
    else {
        return p;
    };
    geom::point_on_triangle(p, (a, b, c))
}

// The smallest angle (degrees) of the triangles.
fn min_angle(positions: &[Vector3D], tris: impl Iterator<Item = [u32; 3]>) -> f64 {
    let mut min = f64::INFINITY;
    for t in tris {
        for i in 0..3 {
            let p = positions[t[i] as usize];
            let (u, v) = (
                positions[t[(i + 1) % 3] as usize] - p,
                positions[t[(i + 2) % 3] as usize] - p,
            );
            if u.norm() > 0. && v.norm() > 0. {
                min = min.min(u.angle(&v).to_degrees());
            }
        }
    }
    min
}

impl<M: Tag> Mesh<M> {
    /// Remesh this closed triangle mesh (see the module documentation). Fails if the mesh is not a closed triangle
    /// mesh, or the result is not a valid mesh of the same topology.
    pub fn remesh(&self, params: &RemeshParams) -> Result<(Self, RemeshReport), String> {
        let verts = self.vert_ids();
        let index: FxHashMap<VertKey<M>, u32> = verts
            .iter()
            .enumerate()
            .map(|(i, &v)| (v, i as u32))
            .collect();
        let mut work = Work {
            pos: verts.iter().map(|&v| self.position(v)).collect(),
            size: vec![0.; verts.len()],
            on_feature: vec![false; verts.len()],
            fixed: vec![false; verts.len()],
            vert_alive: vec![true; verts.len()],
            tris: vec![],
            alive: vec![],
            edges: FxHashMap::default(),
            vfaces: vec![vec![]; verts.len()],
            features: FxHashSet::default(),
        };
        for face in self.face_ids() {
            let Some(t) = self.vertices(face).map(|v| index[&v]).collect_array::<3>() else {
                return Err("the mesh is not a triangle mesh".to_owned());
            };
            work.add_face(t);
        }
        if work
            .edges
            .keys()
            .any(|&(a, b)| !work.edges.contains_key(&(b, a)))
        {
            return Err("the mesh is not closed".to_owned());
        }
        let faces_before = work.tris.len();
        let min_angle_before = min_angle(&work.pos, work.tris.iter().copied());

        // Bounding box diagonal, and the sharp features.
        let (lo, hi) = work.pos.iter().fold(
            (
                Vector3D::repeat(f64::INFINITY),
                Vector3D::repeat(f64::NEG_INFINITY),
            ),
            |(lo, hi), p| (lo.inf(p), hi.sup(p)),
        );
        let diagonal = (hi - lo).norm().max(1e-12);
        let min_length = params.min_length * diagonal;
        let max_length = params.max_ratio * min_length;
        let mut feature_degree = vec![0usize; verts.len()];
        // Per vertex: the largest normal change per length over its (non-feature) edges (curvature).
        let mut curvature = vec![0f64; verts.len()];
        for edge in self.edge_ids() {
            let twin = self.twin(edge);
            if edge > twin {
                continue;
            }
            let (a, b) = (index[&self.root(edge)], index[&self.root(twin)]);
            let dihedral = self
                .normal(self.face(edge))
                .angle(&self.normal(self.face(twin)));
            if dihedral > params.feature_angle {
                work.features.insert(key(a, b));
                feature_degree[a as usize] += 1;
                feature_degree[b as usize] += 1;
            } else {
                let length = work.length(a, b).max(1e-12);
                for v in [a, b] {
                    curvature[v as usize] = curvature[v as usize].max(dihedral / length);
                }
            }
        }
        for v in 0..verts.len() {
            work.on_feature[v] = feature_degree[v] > 0;
            work.fixed[v] = feature_degree[v] > 0 && feature_degree[v] != 2;
            work.size[v] = if curvature[v] > 0. {
                (params.angle_per_edge / curvature[v]).clamp(min_length, max_length)
            } else {
                max_length
            };
        }
        // Smooth the sizes (no sudden jumps between neighbors).
        for _ in 0..3 {
            let sizes = work.size.clone();
            for v in 0..verts.len() as u32 {
                let neighbors = work.neighbors(v);
                if !neighbors.is_empty() {
                    let mean = neighbors.iter().map(|&n| sizes[n as usize]).sum::<f64>()
                        / neighbors.len() as f64;
                    work.size[v as usize] = 0.5 * (sizes[v as usize] + mean);
                }
            }
        }

        let bvh = self.bvh();
        for _ in 0..params.iterations {
            // 1. Split long edges (longest first).
            let mut long = work
                .edge_list()
                .into_iter()
                .filter(|&(a, b)| work.length(a, b) > 4. / 3. * work.target(a, b))
                .collect_vec();
            long.sort_by(|&(a, b), &(c, d)| {
                (work.length(c, d) / work.target(c, d))
                    .total_cmp(&(work.length(a, b) / work.target(a, b)))
            });
            for (a, b) in long {
                if work.edges.contains_key(&(a, b))
                    && work.length(a, b) > 4. / 3. * work.target(a, b)
                {
                    work.split(a, b);
                }
            }
            // 2. Collapse short edges (shortest first).
            let mut short = work
                .edge_list()
                .into_iter()
                .filter(|&(a, b)| work.length(a, b) < 4. / 5. * work.target(a, b))
                .collect_vec();
            short.sort_by(|&(a, b), &(c, d)| {
                (work.length(a, b) / work.target(a, b))
                    .total_cmp(&(work.length(c, d) / work.target(c, d)))
            });
            for (a, b) in short {
                if work.edges.contains_key(&(a, b))
                    && work.length(a, b) < 4. / 5. * work.target(a, b)
                    && !work.collapse(a, b)
                {
                    work.collapse(b, a);
                }
            }
            // 3. Flip edges towards valence 6.
            for (a, b) in work.edge_list() {
                work.flip(a, b, params.feature_angle / 2., false);
            }
            // 4. Tangential smoothing, and projection onto the input surface.
            let positions = work.pos.clone();
            for v in 0..work.pos.len() as u32 {
                let vi = v as usize;
                if !work.vert_alive[vi] || work.fixed[vi] || work.vfaces[vi].is_empty() {
                    continue;
                }
                let neighbors = work.neighbors(v);
                let target = if work.on_feature[vi] {
                    // Along the feature line: the mean of its two neighbors on it.
                    let along = neighbors
                        .iter()
                        .filter(|&&n| work.is_feature(v, n))
                        .map(|&n| positions[n as usize])
                        .collect_vec();
                    if along.len() != 2 {
                        continue;
                    }
                    0.5 * (along[0] + along[1])
                } else {
                    let centroid = neighbors
                        .iter()
                        .map(|&n| positions[n as usize])
                        .sum::<Vector3D>()
                        / neighbors.len() as f64;
                    let normal = work.vfaces[vi]
                        .iter()
                        .map(|&f| work.normal_of(work.tris[f as usize], &|u| positions[u as usize]))
                        .sum::<Vector3D>();
                    let d = centroid - positions[vi];
                    if normal.norm() > 0. {
                        let n = normal.normalize();
                        positions[vi] + d - n * n.dot(&d)
                    } else {
                        positions[vi]
                    }
                };
                let moved = positions[vi] + 0.5 * (target - positions[vi]);
                work.pos[vi] = closest_point(self, &bvh, moved);
            }
        }

        // Finally, flip towards a Delaunay triangulation (on nearly flat edges), against the remaining small angles.
        for _ in 0..10 {
            let mut flipped = false;
            for (a, b) in work.edge_list() {
                flipped |= work.flip(a, b, 20f64.to_radians(), true);
            }
            if !flipped {
                break;
            }
        }

        // The result: the alive faces and their vertices.
        let mut new_index = FxHashMap::default();
        let mut positions = vec![];
        let mut faces = vec![];
        for (f, t) in work.tris.iter().enumerate() {
            if !work.alive[f] {
                continue;
            }
            let face = t
                .iter()
                .map(|&v| {
                    *new_index.entry(v).or_insert_with(|| {
                        positions.push(work.pos[v as usize]);
                        positions.len() - 1
                    })
                })
                .collect_vec();
            faces.push(face);
        }
        let (mesh, _, _) = Self::from(&faces, &positions)
            .map_err(|err| format!("the result is invalid: {err:?}"))?;
        let euler = |m: &Self| m.nr_verts() as i64 - m.nr_edges() as i64 / 2 + m.nr_faces() as i64;
        if euler(&mesh) != euler(self) {
            return Err("the topology changed".to_owned());
        }
        let min_angle_after = min_angle(
            &positions,
            faces
                .iter()
                .map(|f| [f[0] as u32, f[1] as u32, f[2] as u32]),
        );
        // The distance between the surfaces: from every vertex of each to the other.
        let new_bvh = mesh.bvh();
        let max_distance = mesh
            .vert_ids()
            .into_iter()
            .map(|v| (mesh.position(v) - closest_point(self, &bvh, mesh.position(v))).norm())
            .chain(self.vert_ids().into_iter().map(|v| {
                (self.position(v) - closest_point(&mesh, &new_bvh, self.position(v))).norm()
            }))
            .fold(0., f64::max)
            / diagonal;
        let report = RemeshReport {
            faces_before,
            faces_after: mesh.nr_faces(),
            min_angle_before,
            min_angle_after,
            max_distance,
        };
        Ok((mesh, report))
    }
}
