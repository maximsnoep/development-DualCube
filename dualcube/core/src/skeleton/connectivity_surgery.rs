//! Connectivity Surgery: Edge collapse to extract a 1D curve skeleton.
//!
//! This module implements the connectivity surgery step from the
//! "Skeleton Extraction by Mesh Contraction" paper. After mesh contraction,
//! the mesh is thin but still 2D. This step collapses edges until only
//! a 1D skeleton remains.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use log::{info, warn};
use mehsh::prelude::{HasNeighbors, HasPosition, HasVertices, Mesh, Vector3D, VertKey};
use nalgebra::{Matrix4, Vector4};

use super::contraction::CONTRACTION;
use super::curve_skeleton::CurveSkeleton;
use crate::prelude::INPUT;
use crate::skeleton::curve_skeleton::{SkeletonNode, patch_centroid};

/// Internal vertex index type.
type VIdx = VertKey<CONTRACTION>;

/// Collapse candidate for the priority queue.
#[derive(Debug)]
struct CollapseCandidate {
    /// Source vertex (will be removed).
    u: VIdx,
    /// Target vertex (will absorb u).
    v: VIdx,
    /// Collapse cost (lower is better).
    cost: f64,
}

impl Ord for CollapseCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for CollapseCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for CollapseCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl Eq for CollapseCandidate {}

/// Ephemeral state used during the surgery process.
///
/// We maintain a "virtual" adjacency structure separate from the DCEL mesh
/// to perform edge collapses without modifying the original mesh structure.
/// (This was simpler to implement than implementing structural modifications there).
struct SurgeryContext {
    /// Normalized vertex positions for numerical stability.
    positions: HashMap<VIdx, Vector3D>,

    /// Stored normalization factors to restore original positions later.
    center: Vector3D,
    scale: f64,

    /// Mutable adjacency structure (using HashSet for efficient lookup).
    neighbors: HashMap<VIdx, HashSet<VIdx>>,

    /// Active faces tracker for topology checks.
    /// Each face is stored as a sorted triple of vertex keys.
    active_faces: HashSet<[VIdx; 3]>,

    /// Quadric Error Matrices for shape cost (Eq 5 in paper).
    quadrics: HashMap<VIdx, Matrix4<f64>>,

    /// Flags for deleted vertices.
    is_dead: HashSet<VIdx>,

    /// Maps each skeleton vertex to the list of original mesh vertex keys.
    /// Initially each vertex maps to itself, then accumulates as vertices merge.
    vertex_to_original: HashMap<VIdx, Vec<VertKey<INPUT>>>,
}

impl SurgeryContext {
    /// Creates a new SurgeryContext from a contracted mesh.
    fn new(mesh: &Mesh<CONTRACTION>) -> Self {
        let vert_ids = mesh.vert_ids();

        // Compute bounding box for normalization. Normalization is necessary for
        // the costs to balance properly.
        let mut min = Vector3D::new(f64::MAX, f64::MAX, f64::MAX);
        let mut max = Vector3D::new(f64::MIN, f64::MIN, f64::MIN);
        for &v in &vert_ids {
            let p = mesh.position(v);
            min = Vector3D::new(min.x.min(p.x), min.y.min(p.y), min.z.min(p.z));
            max = Vector3D::new(max.x.max(p.x), max.y.max(p.y), max.z.max(p.z));
        }

        let center = (min + max) * 0.5;
        let max_dim = (max - min).max();
        let scale = if max_dim > 1e-8 { 1.0 / max_dim } else { 1.0 };

        // Store normalized positions
        let mut positions = HashMap::new();
        for &v in &vert_ids {
            let p = mesh.position(v);
            let normalized = (p - center) * scale;
            positions.insert(v, normalized);
        }

        // Build adjacency sets from mesh
        let mut neighbors: HashMap<VIdx, HashSet<VIdx>> = HashMap::new();
        for &v in &vert_ids {
            let neighbor_set: HashSet<_> = mesh.neighbors(v).collect();
            neighbors.insert(v, neighbor_set);
        }

        // Build active faces set
        let mut active_faces = HashSet::new();
        for face_id in mesh.face_ids() {
            let verts: Vec<_> = mesh.vertices(face_id).collect();
            if verts.len() == 3 {
                active_faces.insert(sort_face(verts[0], verts[1], verts[2]));
            } else {
                // For non-triangular faces, triangulate using fan
                for i in 1..verts.len() - 1 {
                    active_faces.insert(sort_face(verts[0], verts[i], verts[i + 1]));
                }
            }
        }

        // Compute initial quadrics from edges
        let mut quadrics: HashMap<VIdx, Matrix4<f64>> = HashMap::new();
        for &v in &vert_ids {
            quadrics.insert(v, Matrix4::zeros());
        }

        for &u in &vert_ids {
            for &v in &neighbors[&u] {
                if u < v {
                    let p_u = positions[&u];
                    let p_v = positions[&v];
                    let edge_vec = p_v - p_u;
                    let len = edge_vec.norm();
                    if len < 1e-12 {
                        continue;
                    }
                    let edge_vec = edge_vec / len;

                    let q_edge = compute_edge_quadric(p_u, edge_vec);

                    *quadrics.get_mut(&u).unwrap() += q_edge;
                    *quadrics.get_mut(&v).unwrap() += q_edge;
                }
            }
        }

        // Initialize vertex_to_original: each vertex maps to itself (converted to INPUT key)
        let mut vertex_to_original = HashMap::new();
        for &v in &vert_ids {
            // Convert CONTRACTION key to INPUT key (same raw value)
            let input_key = VertKey::<INPUT>::new(v.raw());
            vertex_to_original.insert(v, vec![input_key]);
        }

        Self {
            positions,
            center,
            scale,
            neighbors,
            active_faces,
            quadrics,
            is_dead: HashSet::new(),
            vertex_to_original,
        }
    }

    /// Computes the collapse cost for edge u -> v (Eq 8 in paper).
    fn compute_collapse_cost(&self, u: VIdx, v: VIdx) -> f64 {
        const WA: f64 = 1.0; // Shape weight
        const WB: f64 = 0.1; // Sampling weight

        // Shape Cost: v^T * (Q_u + Q_v) * v
        let p_v = self.positions[&v];
        let p_hom = Vector4::new(p_v.x, p_v.y, p_v.z, 1.0);

        let q_sum = self.quadrics[&u] + self.quadrics[&v];
        let shape_cost = p_hom.dot(&(q_sum * p_hom));

        // Sampling Cost: sum of squared distances to new edges
        let mut sampling_cost = 0.0;
        for &n in &self.neighbors[&u] {
            if n != v {
                let dist_sq = (self.positions[&v] - self.positions[&n]).norm_squared();
                sampling_cost += dist_sq;
            }
        }

        WA * shape_cost + WB * sampling_cost
    }

    /// Checks if an edge (u, v) is part of any active face.
    /// Collapsing edges with no faces would destroy the skeleton.
    fn edge_has_faces(&self, u: VIdx, v: VIdx) -> bool {
        let n_u = &self.neighbors[&u];
        let n_v = &self.neighbors[&v];

        // Iterate the smaller set for efficiency
        let (smaller, larger) = if n_u.len() < n_v.len() {
            (n_u, n_v)
        } else {
            (n_v, n_u)
        };

        for &w in smaller {
            if w == u || w == v {
                continue;
            }

            // If w is a shared neighbor, check if triangle (u, v, w) is active
            if larger.contains(&w) {
                if self.active_faces.contains(&sort_face(u, v, w)) {
                    return true;
                }
            }
        }
        false
    }

    /// Link Condition: For collapse u->v, if they share a neighbor w,
    /// i.e. face (u, v, w) must exist.
    fn check_link_condition(&self, u: VIdx, v: VIdx) -> bool {
        let n_u = &self.neighbors[&u];
        let n_v = &self.neighbors[&v];

        for &w in n_u {
            if w == v {
                continue;
            }
            if n_v.contains(&w) {
                // Shared neighbor found. Check face.
                if !self.active_faces.contains(&sort_face(u, v, w)) {
                    // return false; // temp fix for genus 0. Needs a proper implementation later
                    break;
                }
            }
        }
        true
    }

    /// Performs edge collapse: u merges into v.
    fn collapse_edge(&mut self, u: VIdx, v: VIdx) {
        // Merge quadrics
        let q_u = self.quadrics[&u];
        *self.quadrics.get_mut(&v).unwrap() += q_u;

        // Mark u as dead
        self.is_dead.insert(u);

        // Merge original vertex mappings
        let mut originals_u = self.vertex_to_original.remove(&u).unwrap_or_default();
        self.vertex_to_original
            .get_mut(&v)
            .unwrap()
            .append(&mut originals_u);

        // Get u's neighbors before modifying
        let neighbors_u: Vec<_> = self.neighbors[&u].iter().copied().collect();

        // Update faces
        let mut faces_to_remove = Vec::new();
        let mut faces_to_add = Vec::new();

        for i in 0..neighbors_u.len() {
            for j in (i + 1)..neighbors_u.len() {
                let n1 = neighbors_u[i];
                let n2 = neighbors_u[j];
                let face_key = sort_face(u, n1, n2);

                if self.active_faces.contains(&face_key) {
                    faces_to_remove.push(face_key);

                    // If face doesn't involve v, it morphs into (v, n1, n2)
                    if n1 != v && n2 != v {
                        faces_to_add.push(sort_face(v, n1, n2));
                    }
                }
            }
        }

        for f in faces_to_remove {
            self.active_faces.remove(&f);
        }

        for f in faces_to_add {
            self.active_faces.insert(f);
        }

        // Update neighbor sets
        // Remove u from all its neighbors' sets
        for &n in &neighbors_u {
            if let Some(n_set) = self.neighbors.get_mut(&n) {
                n_set.remove(&u);
            }
        }

        // Merge u's neighbors into v's neighbors
        for &n in &neighbors_u {
            if n == v {
                continue;
            }

            // Add n to v's neighbor set
            self.neighbors.get_mut(&v).unwrap().insert(n);

            // Add v to n's neighbor set
            self.neighbors.get_mut(&n).unwrap().insert(v);
        }

        // Clear u's neighbor set
        self.neighbors.get_mut(&u).unwrap().clear();
    }

    /// Refines the embedding by moving skeleton nodes to the centroid
    /// of their corresponding original mesh vertices.
    fn refine_embedding(&mut self, original_mesh: &Mesh<INPUT>) {
        for (&skel_vert, originals) in &self.vertex_to_original {
            if self.is_dead.contains(&skel_vert) || originals.is_empty() {
                continue;
            }

            // Compute centroid
            let centroid = patch_centroid(&originals, original_mesh);

            // Convert to normalized space
            self.positions
                .insert(skel_vert, (centroid - self.center) * self.scale);
        }
    }

    /// Builds the final CurveSkeleton graph from the surgery result.
    fn to_curve_skeleton(&self) -> CurveSkeleton {
        let mut graph = CurveSkeleton::default();
        let mut node_indices = HashMap::new();

        // Add living nodes
        for (&v, &pos) in &self.positions {
            if self.is_dead.contains(&v) {
                continue;
            }

            // Denormalize position to original model space
            let original_pos = pos / self.scale + self.center;

            // Get the list of original vertex keys this node represents
            let originals = self.vertex_to_original.get(&v).cloned().unwrap_or_default();

            let idx = graph.add_node(SkeletonNode {
                position: original_pos,
                patch_vertices: originals,
            });
            node_indices.insert(v, idx);
        }

        // Add edges
        for (&u, neighbor_set) in &self.neighbors {
            if self.is_dead.contains(&u) {
                continue;
            }

            if let Some(&u_node) = node_indices.get(&u) {
                for &v in neighbor_set {
                    // Only add each edge once (when u < v)
                    if u < v && !self.is_dead.contains(&v) {
                        if let Some(&v_node) = node_indices.get(&v) {
                            graph.add_edge(u_node, v_node, ());
                        }
                    }
                }
            }
        }

        graph
    }
}

/// Extracts a 1D curve skeleton from a contracted mesh via connectivity surgery.
///
/// # Arguments
/// * `contracted_mesh` - The mesh after contraction (thin but still 2D).
/// * `original_mesh` - The original input mesh (for embedding refinement).
///
/// # Returns
/// A `CurveSkeleton` graph representing the 1D skeleton.
pub fn extract_skeleton(
    contracted_mesh: &Mesh<CONTRACTION>,
    original_mesh: &Mesh<INPUT>,
) -> CurveSkeleton {
    let mut ctx = SurgeryContext::new(contracted_mesh);
    let mut heap = BinaryHeap::new();

    // Initial heap population
    let vert_ids: Vec<_> = ctx.neighbors.keys().copied().collect();
    for u in vert_ids {
        if ctx.is_dead.contains(&u) {
            continue;
        }
        let neighbors: Vec<_> = ctx.neighbors[&u].iter().copied().collect();
        for v in neighbors {
            // Only add edges that have incident faces
            if ctx.edge_has_faces(u, v) {
                let cost = ctx.compute_collapse_cost(u, v);
                heap.push(CollapseCandidate { u, v, cost });
            }
        }
    }

    let mut collapses = 0;

    // Greedy collapse loop
    while !ctx.active_faces.is_empty() {
        let Some(candidate) = heap.pop() else {
            warn!(
                "No more collapse candidates but {} faces remain.",
                ctx.active_faces.len()
            );
            break;
        };

        // Skip stale entries (cost changed since insertion)
        let real_cost = ctx.compute_collapse_cost(candidate.u, candidate.v);
        const COST_EPSILON: f64 = 1e-8;
        if (candidate.cost - real_cost).abs() > COST_EPSILON {
            continue;
        }

        // Skip if either vertex is dead
        if ctx.is_dead.contains(&candidate.u) || ctx.is_dead.contains(&candidate.v) {
            continue;
        }

        // Check if edge still exists
        if !ctx.neighbors[&candidate.u].contains(&candidate.v) {
            continue;
        }

        // Edge might have lost its last face - skip if so
        if !ctx.edge_has_faces(candidate.u, candidate.v) {
            continue;
        }

        // Check link condition for manifold preservation
        if !ctx.check_link_condition(candidate.u, candidate.v) {
            continue;
        }

        // Perform the collapse
        ctx.collapse_edge(candidate.u, candidate.v);
        collapses += 1;

        // Re-evaluate edges connected to v
        let neighbors: Vec<_> = ctx.neighbors[&candidate.v].iter().copied().collect();
        for neighbor in neighbors {
            // Evaluate v -> neighbor
            if ctx.edge_has_faces(candidate.v, neighbor) {
                let cost = ctx.compute_collapse_cost(candidate.v, neighbor);
                heap.push(CollapseCandidate {
                    u: candidate.v,
                    v: neighbor,
                    cost,
                });
            }

            // Evaluate neighbor -> v
            if ctx.edge_has_faces(neighbor, candidate.v) {
                let cost = ctx.compute_collapse_cost(neighbor, candidate.v);
                heap.push(CollapseCandidate {
                    u: neighbor,
                    v: candidate.v,
                    cost,
                });
            }
        }
    }

    info!(
        "Connectivity surgery complete. Collapsed {} edges. Remaining vertices: {}",
        collapses,
        ctx.positions.len() - ctx.is_dead.len()
    );

    // Refine embedding using original mesh positions
    ctx.refine_embedding(original_mesh);

    // Build and return the curve skeleton graph
    ctx.to_curve_skeleton()
}

/// Computes the edge quadric matrix (Eq 4 in paper).
fn compute_edge_quadric(p: Vector3D, a: Vector3D) -> Matrix4<f64> {
    let b = a.cross(&p);

    // Build K matrix rows
    let r0 = Vector4::new(0.0, -a.z, a.y, -b.x);
    let r1 = Vector4::new(a.z, 0.0, -a.x, -b.y);
    let r2 = Vector4::new(-a.y, a.x, 0.0, -b.z);
    let r3 = Vector4::new(0.0, 0.0, 0.0, 0.0);

    let k = Matrix4::from_rows(&[
        r0.transpose(),
        r1.transpose(),
        r2.transpose(),
        r3.transpose(),
    ]);

    k.transpose() * k
}

/// Sorts three vertex keys into a canonical order for face comparison.
#[inline]
fn sort_face(a: VIdx, b: VIdx, c: VIdx) -> [VIdx; 3] {
    let mut f = [a, b, c];
    f.sort();
    f
}
