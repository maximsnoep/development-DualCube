//! Connectivity Surgery: Edge collapse to extract a 1D curve skeleton.
//!
//! This module implements the connectivity surgery step from the
//! "Skeleton Extraction by Mesh Contraction" paper. After mesh contraction,
//! the mesh is thin but still 2D. This step collapses edges until only
//! a 1D skeleton remains.
//!
//! On top of the paper's strict Dey–Edelsbrunner link condition we run a
//! homology-aware variant that admits handle-killing collapses on
//! positive-genus inputs while still rejecting any collapse that would
//! destroy an essential tunnel cycle of the solid. The geometry and
//! correctness argument lives in `HOMOLOGY_PRESERVING_SURGERY.md`.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::time::Instant;

use mehsh::prelude::{HasNeighbors, HasPosition, HasVertices, Mesh, Vector3D, VertKey};
use nalgebra::{Matrix4, Vector4};
use tracing::{error, info, warn};

mod f2_rref;
mod handle_subspace;
mod tet_boundary;
mod tetrahedralize;

use super::deflate::CONTRACTION;
use crate::{PatchBoundary, Skeleton, SkeletonEdge, SkeletonNode};
use dualcube_types::prelude::INPUT;
use f2_rref::F2Matrix;
use handle_subspace::compute_handle_subspace;
use tet_boundary::build_tet_boundary;
use tetrahedralize::tetrahedralize;

/// Internal vertex index type for contraction mesh.
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
            .then_with(|| self.u.partial_cmp(&other.u).unwrap())
            .then_with(|| self.v.partial_cmp(&other.v).unwrap())
    }
}

impl PartialOrd for CollapseCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for CollapseCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost && self.u == other.u && self.v == other.v
    }
}

impl Eq for CollapseCandidate {}

/// Boundary matrix ∂₂ over GF(2) plus the maps needed to address rows by
/// face triple and columns by undirected edge, plus a basis of the handle
/// subspace `K ⊆ H₁(S; ℤ/2)` and a cached RREF of `B₁(X) + K`.
/// Maintained throughout surgery so [`SurgeryContext::check_link_condition`]
/// can decide in O(shared neighbours · cycle length) per candidate whether
/// any corner-cycle `γ_w` has tunnel content.
#[derive(Clone)]
struct BoundaryMatrix {
    matrix: F2Matrix,
    /// Sorted face triple -> row index in `matrix`. Row indices are stable
    /// across collapses; deleted faces are removed from this map and their
    /// rows in `matrix` are soft-cleared (left as the zero row).
    face_to_row: HashMap<[VIdx; 3], usize>,
    /// Sorted edge pair -> column index in `matrix`. Column indices are
    /// stable across collapses; merged columns are dropped from this map
    /// and become the zero column in `matrix`.
    edge_to_col: HashMap<[VIdx; 2], u32>,
    /// Handle-subspace generators: `g` 1-chains over the current edge
    /// basis (sorted column indices, exactly like a `matrix` row). Each
    /// generator is the chain-map image of one of the precomputed handle
    /// cycles `c_i ∈ K = ker(H₁(S) → H₁(V))`. Used by the cost bias to
    /// drive greedy toward handle-killing collapses, and as the extra
    /// rows that augment `∂₂` when computing `aug_pivots` below. Empty
    /// for genus 0.
    handle_cycles: Vec<Vec<u32>>,
    /// Cached RREF of `[∂₂; handle_cycles]`. Its row span is `B₁(X) + K`,
    /// so reducing a 1-chain against [`Self::aug_pivot_map`] returns the
    /// zero residue iff the chain is null-homologous *modulo handles* —
    /// i.e. iff it has no tunnel content. Used by the legality check
    /// [`SurgeryContext::check_link_condition`] on every candidate.
    /// Refresh is lazy: mutations set [`Self::pivots_dirty`], and the
    /// slow path of `check_link_condition` calls
    /// [`Self::ensure_pivots_fresh`] right before reading the cache.
    aug_pivots: F2Matrix,
    /// Pivot map of [`Self::aug_pivots`].
    aug_pivot_map: HashMap<u32, usize>,
    /// True when `matrix` or `handle_cycles` has been mutated since the
    /// last `rebuild_pivots`. The vast majority of collapses don't hit
    /// the slow path of `check_link_condition` (every shared face-vertex
    /// neighbour is triangular ⇒ fast path), so refreshing the F2 RREF
    /// after every mutation is wasted work. The slow path checks this
    /// flag via `ensure_pivots_fresh` and rebuilds only when actually
    /// about to consult the cache.
    pivots_dirty: bool,
}

impl BoundaryMatrix {
    /// Refreshes [`Self::aug_pivots`] and [`Self::aug_pivot_map`] from the
    /// current `matrix` and `handle_cycles`. Must be called after every
    /// mutation that could have invalidated the previous RREF — i.e. after
    /// every accepted collapse and every free-face collapse.
    ///
    /// Construction: take the live `∂₂` rows and append each handle cycle
    /// as an extra row, then reduce. The resulting pivot set spans
    /// `B₁(X) + K`. The live `matrix` is left untouched so its rows still
    /// correspond to face row indices via `face_to_row`.
    fn rebuild_pivots(&mut self) {
        // Skip empty rows — both the cleared face rows in `matrix` (which
        // accumulate as collapses proceed) and any handle cycles that
        // have died. They contribute nothing to the row span; filtering
        // them out keeps `reduce()` from looping over a steadily-growing
        // pile of no-ops.
        let mut combined: Vec<Vec<u32>> = self
            .matrix
            .rows()
            .iter()
            .filter(|r| !r.is_empty())
            .cloned()
            .collect();
        combined.extend(self.handle_cycles.iter().filter(|c| !c.is_empty()).cloned());
        let mut m = F2Matrix::from_rows(combined);
        m.reduce();
        self.aug_pivot_map = m.pivot_map();
        self.aug_pivots = m;
        self.pivots_dirty = false;
    }

    /// Refreshes the pivot cache only if it's stale. The slow path of
    /// `check_link_condition` calls this just before reading
    /// `aug_pivots` / `aug_pivot_map`; the fast path never calls it.
    /// For collapses whose every shared face-vertex neighbour is
    /// triangular (the common case during early/mid surgery), this is
    /// a single bool load.
    fn ensure_pivots_fresh(&mut self) {
        if self.pivots_dirty {
            self.rebuild_pivots();
        }
    }
}

/// The set of changes a candidate collapse u → v would make to the
/// boundary-matrix state. Computed once per collapse and applied either
/// to a snapshot (in [`SurgeryContext::simulate_post_collapse_stats`], a
/// diagnostic) or to the live matrix (in [`SurgeryContext::collapse_edge`]).
struct CollapseDelta {
    /// Faces incident to edge (u, v): rows to clear; map entries to drop.
    faces_removed: Vec<[VIdx; 3]>,
    /// Faces (u, x, y) where (v, x, y) already exists: row of the (u, x, y)
    /// face is cleared (after column merging the two are identical), and the
    /// (u, x, y) entry is dropped from face_to_row. The (v, x, y) row stays.
    faces_merged: Vec<([VIdx; 3], [VIdx; 3])>,
    /// Faces (u, x, y) with no corresponding (v, x, y): row is unchanged in
    /// content, only its key in face_to_row needs to be relabeled.
    faces_renamed: Vec<([VIdx; 3], [VIdx; 3])>,
    /// Column-merge ops to apply: `(keep, drop)`. `keep` survives in
    /// edge_to_col; `drop` becomes the zero column.
    edge_col_merges: Vec<(u32, u32)>,
    /// `(u, w)` edge keys whose column got merged into the corresponding
    /// `(v, w)` column. These keys need to be removed from `edge_to_col`
    /// after the column merge — parallel to `edge_col_merges`, in the
    /// same order. Stored explicitly so apply doesn't have to scan all
    /// of `edge_to_col` looking for keys whose value lives in `dropped`.
    dropped_edge_keys: Vec<[VIdx; 2]>,
    /// Edges (u, w) for non-shared neighbors w: their column survives but
    /// the key in edge_to_col is renamed from (u, w) to (v, w).
    edge_renames: Vec<([VIdx; 2], [VIdx; 2])>,
    /// The (u, v) edge itself: its column is implicitly emptied (every
    /// incident face is in `faces_removed`) and its key dropped from
    /// edge_to_col.
    removed_uv_edge: [VIdx; 2],
    delta_v: usize,
    delta_e: usize,
    delta_f: usize,
}

/// Sorts two vertex keys into canonical order for edge identity.
#[inline]
fn sort_edge(a: VIdx, b: VIdx) -> [VIdx; 2] {
    if a < b { [a, b] } else { [b, a] }
}

/// Merge two edge columns in a GF(2) chain.
fn merge_columns_in_chain(chain: &mut Vec<u32>, keep: u32, drop: u32) {
    debug_assert!(keep != drop);
    if let Ok(drop_pos) = chain.binary_search(&drop) {
        chain.remove(drop_pos);
        match chain.binary_search(&keep) {
            Ok(kp) => {
                chain.remove(kp);
            }
            Err(ip) => {
                chain.insert(ip, keep);
            }
        }
    }
}

/// Toggles `col` in a sorted GF(2) 1-chain: if present, remove; if
/// absent, insert. Used by the free-face collapse's chain map, which on
/// 1-chains sends the free edge to the sum of the face's other two
/// edges — i.e. each "other" edge's coefficient gets XOR'd in.
fn toggle_col_in_chain(chain: &mut Vec<u32>, col: u32) {
    match chain.binary_search(&col) {
        Ok(pos) => {
            chain.remove(pos);
        }
        Err(pos) => {
            chain.insert(pos, col);
        }
    }
}

/// Weights of collapse costs; topology checks are unchanged.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub shape_weight: f64,
    pub sampling_weight: f64,
    /// Subtracted once per handle cycle through the collapsed edge. Not from Au et al.
    pub handle_bonus: f64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            shape_weight: 1.0,
            sampling_weight: 0.1,
            handle_bonus: 1.0e6,
        }
    }
}

impl Options {
    pub fn validate(self) -> crate::SkeletonResult<()> {
        if [self.shape_weight, self.sampling_weight, self.handle_bonus]
            .iter()
            .any(|x| !x.is_finite() || *x < 0.0)
        {
            return Err("Surgery weights must be finite and nonnegative".into());
        }
        Ok(())
    }
}

/// Ephemeral state used during the surgery process.
///
/// We maintain a "virtual" adjacency structure separate from the DCEL mesh
/// to perform edge collapses without modifying the original mesh structure.
/// (This was simpler to implement than implementing structural modifications there).
struct SurgeryContext {
    options: Options,
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

    /// Genus of the closed orientable input surface. Used by the
    /// preprocessing sanity check and by the stuck-state diagnostics;
    /// the per-candidate legality check no longer reads it directly
    /// (it tests `γ_w`'s tunnel content against the cached pivots).
    target_g: usize,

    /// Boundary matrix ∂₂ + handle-subspace basis + cached RREF of
    /// `B₁(X) + K`. Always populated for a valid closed orientable input.
    boundary_matrix: BoundaryMatrix,
}

impl SurgeryContext {
    /// Initialize collapse costs and the original surface's handle subspace.
    fn new(mesh: &Mesh<CONTRACTION>, original_mesh: &Mesh<INPUT>) -> crate::SkeletonResult<Self> {
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

        // Topology preprocessing: verify the simplicial complex is a closed
        // connected 2-manifold, derive genus, and build the boundary matrix
        // ∂₂ over GF(2). When genus > 0, also compute a basis of the handle
        // subspace K = ker(H₁(S) → H₁(V)) via tetrahedralization of the
        // input surface, and attach those cycles to the boundary matrix.
        // At construction time no vertex has been collapsed yet.
        let is_dead: HashSet<VIdx> = HashSet::new();
        let (target_g, mut boundary_matrix) =
            preprocess_topology(&active_faces, &neighbors, &is_dead, vert_ids.len())
                .ok_or("Connectivity surgery topology preprocessing failed")?;

        if target_g > 0 {
            attach_handle_cycles(&mut boundary_matrix, original_mesh, target_g)?;
        }

        Ok(Self {
            options: Options::default(),
            positions,
            center,
            scale,
            neighbors,
            active_faces,
            quadrics,
            is_dead,
            vertex_to_original,
            target_g,
            boundary_matrix,
        })
    }

    /// Computes the collapse cost for edge u -> v (Eq 8 in paper).
    fn compute_collapse_cost(&self, u: VIdx, v: VIdx) -> f64 {
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

        // Handle bonus: subtract one full bonus per handle cycle that
        // currently includes the column for edge (u, v).
        let mut handle_bonus: f64 = 0.0;
        if let Some(&col) = self.boundary_matrix.edge_to_col.get(&sort_edge(u, v)) {
            for cycle in &self.boundary_matrix.handle_cycles {
                if cycle.binary_search(&col).is_ok() {
                    handle_bonus += self.options.handle_bonus;
                }
            }
        }

        self.options.shape_weight * shape_cost + self.options.sampling_weight * sampling_cost
            - handle_bonus
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

    /// Classical Dey–Edelsbrunner manifold-preservation check (every
    /// shared neighbour `w` of `u` and `v` must form a face `(u, v, w)`).
    /// **Not** in the legality gate any more — the new criterion permits
    /// non-manifold collapses that kill handle cycles — but kept around so
    /// the existing tests and diagnostics can ask "would this collapse have
    /// preserved manifoldness?".
    #[allow(dead_code)]
    fn check_manifold_preserved(&self, u: VIdx, v: VIdx) -> bool {
        let n_u = &self.neighbors[&u];
        let n_v = &self.neighbors[&v];
        for &w in n_u {
            if w == v {
                continue;
            }
            if n_v.contains(&w) && !self.active_faces.contains(&sort_face(u, v, w)) {
                return false;
            }
        }
        true
    }

    /// Homology-aware legality check, structurally mirroring the original
    /// Dey–Edelsbrunner edge-link condition.
    ///
    /// For each shared neighbour `w` of `u` and `v`, the three edges
    /// `(u,v)`, `(v,w)`, `(u,w)` form a 1-cycle `γ_w` in the current
    /// 1-skeleton. The edge-collapse chain map `q_#` sends `γ_w` to zero
    /// (the (u,v) edge is killed; the (u,w) and (v,w) columns merge),
    /// so the homology class `[γ_w] ∈ H₁(X)` is exactly the class the
    /// collapse would kill at this corner. In fact every class killed
    /// by the collapse is a sum of such `[γ_w]`, so this set of probes
    /// is complete (see `HOMOLOGY_PRESERVING_SURGERY.md`).
    ///
    /// Three cases per `w`:
    /// - **Face `(u,v,w)` exists.** Then `γ_w = ∂(u,v,w) ∈ B₁(X)`, so
    ///   `[γ_w] = 0` — nothing is being killed. This recovers the strict
    ///   link condition's accept case.
    /// - **`[γ_w]` is a pure handle class** (lies in `K`). The collapse
    ///   kills only a handle generator; that's *desired* — handles need
    ///   to die for surgery to reach a 1-skeleton on positive-genus
    ///   inputs. Accept.
    /// - **`[γ_w]` has tunnel content** (i.e. its image in `H₁(V)` is
    ///   non-zero). The collapse would destroy an essential cycle of the
    ///   underlying solid. Reject.
    ///
    /// Test: reduce `γ_w` against `aug_pivot_map`, the cached pivots of
    /// `B₁(X) + K`. Zero residue ⇒ no tunnel content (accept this `w`);
    /// non-zero residue ⇒ tunnel content (reject the whole collapse).
    fn check_link_condition(&mut self, u: VIdx, v: VIdx) -> bool {
        // Caller (`is_legal_collapse_candidate`) has already gated on
        // `edge_has_faces`, which means `(u, v)` is incident to at least
        // one active face → its column entry is live in `edge_to_col`.
        let uv_col = self.boundary_matrix.edge_to_col[&sort_edge(u, v)];

        // Walk shared neighbours and split them into fast-path corners
        // (face `(u, v, w)` exists → `γ_w ∈ B₁` trivially) and slow-path
        // corners (need an F2 reduction). We collect the slow ones
        // into a `Vec` first so the subsequent `ensure_pivots_fresh`
        // call (which mutably borrows `self.boundary_matrix`) doesn't
        // conflict with the iterator's immutable borrow of
        // `self.neighbors`.
        let mut slow_corners: Vec<VIdx> = Vec::new();
        {
            let n_u = &self.neighbors[&u];
            let n_v = &self.neighbors[&v];
            for &w in n_u {
                if w == v {
                    continue;
                }
                if !n_v.contains(&w) {
                    continue;
                }
                // Fast path — strict Dey–Edelsbrunner link condition at
                // this corner: if face `(u,v,w)` exists, then
                // `γ_w = ∂(u,v,w) ∈ B₁(X)` is already a boundary, so
                // `[γ_w] = 0` has no tunnel content.
                if self.active_faces.contains(&sort_face(u, v, w)) {
                    continue;
                }
                // For target_g == 0 the handlebody is a ball, so
                // K = ker(H₁(S) → H₁(V)) = 0 (H₁(S) of a sphere is
                // trivial) and every 1-cycle on the surface is already
                // a boundary (Z₁ = B₁). γ_w is therefore guaranteed to
                // reduce to zero against B₁ + K. Skip the F2 reduction
                // and the per-collapse pivot cache it relies on.
                if self.target_g == 0 {
                    continue;
                }
                slow_corners.push(w);
            }
        }

        if slow_corners.is_empty() {
            return true;
        }

        // Lazy pivot refresh: every accepted collapse marks the cache
        // dirty, but we only pay for the rebuild when a slow-path
        // probe is actually about to read it.
        self.boundary_matrix.ensure_pivots_fresh();
        let bm = &self.boundary_matrix;
        for w in slow_corners {
            let vw_col = bm.edge_to_col[&sort_edge(v, w)];
            let uw_col = bm.edge_to_col[&sort_edge(u, w)];
            let mut residue = vec![uv_col, vw_col, uw_col];
            residue.sort();
            if !bm
                .aug_pivots
                .reduce_against(&bm.aug_pivot_map, &mut residue)
            {
                return false;
            }
        }
        true
    }

    /// Diagnostic-only: returns post-collapse `(β₁, dim K, β₂)` so the
    /// stuck-state and per-edge dumps can report what each rejected
    /// collapse would have done. Not used by the legality criterion
    /// (which is the cheap local check `check_link_condition`).
    fn simulate_post_collapse_stats(&self, u: VIdx, v: VIdx) -> (i64, i64, i64) {
        let bm = &self.boundary_matrix;
        let delta = self.compute_collapse_delta(u, v);

        let mut snap = bm.clone();
        Self::apply_collapse_delta(&mut snap, &delta);
        snap.matrix.reduce();
        let new_rank = snap.matrix.rank();

        let v_now = (self.positions.len() - self.is_dead.len()) as i64;
        let e_now = bm.edge_to_col.len() as i64;
        let new_v = v_now - delta.delta_v as i64;
        let new_e = e_now - delta.delta_e as i64;
        let new_beta_1 = new_e - new_v + 1 - new_rank as i64;

        let pivot_map = snap.matrix.pivot_map();

        let mut new_dim_k: i64 = 0;
        for cycle in &snap.handle_cycles {
            let mut residue = cycle.clone();
            if !snap.matrix.reduce_against(&pivot_map, &mut residue) {
                new_dim_k += 1;
            }
        }

        let new_f = (self.active_faces.len() as i64) - delta.delta_f as i64;
        let new_beta_2 = new_f - new_rank as i64;

        (new_beta_1, new_dim_k, new_beta_2)
    }

    /// Computes what the (face_to_row, edge_to_col, matrix) state would
    /// change to under a hypothetical collapse u → v. Pure; mutates
    /// nothing. Caller must have already gated on adjacency
    /// (`is_legal_collapse_candidate` does this).
    fn compute_collapse_delta(&self, u: VIdx, v: VIdx) -> CollapseDelta {
        let bm = &self.boundary_matrix;
        let n_u = &self.neighbors[&u];
        let n_v = &self.neighbors[&v];

        // Shared neighbors of u and v (excluding u, v themselves).
        let mut shared: Vec<VIdx> = Vec::new();
        for &w in n_u {
            if w != v && w != u && n_v.contains(&w) {
                shared.push(w);
            }
        }

        // Faces incident to edge (u, v) — one per shared neighbor that
        // currently sits as an active face.
        let mut faces_removed = Vec::new();
        for &w in &shared {
            let f = sort_face(u, v, w);
            if self.active_faces.contains(&f) {
                faces_removed.push(f);
            }
        }

        // Faces (u, n1, n2) with n1 ≠ v and n2 ≠ v: either rename to
        // (v, n1, n2), or merge with an existing (v, n1, n2) face.
        let mut faces_merged = Vec::new();
        let mut faces_renamed = Vec::new();
        let neighbors_u: Vec<VIdx> = n_u.iter().copied().collect();
        for i in 0..neighbors_u.len() {
            for j in (i + 1)..neighbors_u.len() {
                let n1 = neighbors_u[i];
                let n2 = neighbors_u[j];
                if n1 == v || n2 == v {
                    continue;
                }
                let face_old = sort_face(u, n1, n2);
                if !self.active_faces.contains(&face_old) {
                    continue;
                }
                let face_new = sort_face(v, n1, n2);
                if self.active_faces.contains(&face_new) {
                    faces_merged.push((face_old, face_new));
                } else {
                    faces_renamed.push((face_old, face_new));
                }
            }
        }

        // Per-shared-neighbor column merge: keep the (v, w) column, drop
        // the (u, w) column. We also remember each (u, w) edge key so the
        // apply step can drop it from edge_to_col directly.
        let mut edge_col_merges = Vec::with_capacity(shared.len());
        let mut dropped_edge_keys = Vec::with_capacity(shared.len());
        for &w in &shared {
            let uw_key = sort_edge(u, w);
            let col_keep = bm.edge_to_col[&sort_edge(v, w)];
            let col_drop = bm.edge_to_col[&uw_key];
            edge_col_merges.push((col_keep, col_drop));
            dropped_edge_keys.push(uw_key);
        }

        // Edges (u, w) for non-shared w: same column, new key.
        let mut edge_renames = Vec::new();
        for &w in n_u {
            if w == v || w == u {
                continue;
            }
            if !n_v.contains(&w) {
                edge_renames.push((sort_edge(u, w), sort_edge(v, w)));
            }
        }

        let delta_v = 1usize;
        let delta_e = 1 + shared.len(); // (u, v) + one duplicate per shared
        let delta_f = faces_removed.len() + faces_merged.len();

        CollapseDelta {
            faces_removed,
            faces_merged,
            faces_renamed,
            edge_col_merges,
            dropped_edge_keys,
            edge_renames,
            removed_uv_edge: sort_edge(u, v),
            delta_v,
            delta_e,
            delta_f,
        }
    }

    /// Applies a [`CollapseDelta`] in place to the given boundary matrix,
    /// its face/edge maps, and the tracked handle cycles. Used both for
    /// the snapshot in [`Self::simulate_post_collapse_stats`] and for the
    /// live commit in [`Self::collapse_edge`]. The caller is responsible
    /// for refreshing [`BoundaryMatrix::aug_pivots`] afterwards.
    fn apply_collapse_delta(bm: &mut BoundaryMatrix, delta: &CollapseDelta) {
        // 0. Look up the (u, v) column BEFORE we mutate edge_to_col, so we
        //    can scrub it out of every tracked 1-chain below.
        let removed_uv_col = bm.edge_to_col.get(&delta.removed_uv_edge).copied();

        // 1a. Evolve each tracked handle cycle by the edge-collapse chain
        //     map. On 1-chains, the chain map for an edge collapse u → v
        //     is: (a) merge the columns of every shared-neighbour edge
        //     pair (same merge_columns op we do on matrix rows), and
        //     (b) drop the (u, v) column itself.
        for cycle in &mut bm.handle_cycles {
            for &(keep, drop) in &delta.edge_col_merges {
                merge_columns_in_chain(cycle, keep, drop);
            }
            if let Some(col) = removed_uv_col {
                if let Ok(pos) = cycle.binary_search(&col) {
                    cycle.remove(pos);
                }
            }
        }

        // 1b. Column merges on the matrix: shared-neighbour edge columns
        //     collapse into one.
        for &(keep, drop) in &delta.edge_col_merges {
            bm.matrix.merge_columns(keep, drop);
        }

        // 2. Clear rows of removed faces (incident to the (u, v) edge).
        for face in &delta.faces_removed {
            if let Some(&row) = bm.face_to_row.get(face) {
                bm.matrix.clear_row(row);
            }
            bm.face_to_row.remove(face);
        }

        // 3. Clear rows of merged faces; keep the partner row in place.
        for (old, _new_kept) in &delta.faces_merged {
            if let Some(&row) = bm.face_to_row.get(old) {
                bm.matrix.clear_row(row);
            }
            bm.face_to_row.remove(old);
        }

        // 4. Rename face entries: same row, new key.
        for (old, new) in &delta.faces_renamed {
            if let Some(row) = bm.face_to_row.remove(old) {
                bm.face_to_row.insert(*new, row);
            }
        }

        // 5. Rename edge entries: same column, new key. (Pre-collapse the
        //    new key cannot already be present — that would mean w is a
        //    shared neighbor, contradicting the rename branch.)
        for (old, new) in &delta.edge_renames {
            if let Some(col) = bm.edge_to_col.remove(old) {
                bm.edge_to_col.insert(*new, col);
            }
        }

        // 6. Drop edge_to_col entries for the (u, w) keys whose columns
        //    got merged away. The (v, w) keys already point to the kept
        //    column. Direct removals (O(shared)) rather than a `retain`
        //    scan over all of edge_to_col (O(E)).
        for key in &delta.dropped_edge_keys {
            bm.edge_to_col.remove(key);
        }

        // 7. Drop the (u, v) edge entry. Its column is now the zero column
        //    because every face incident to it was cleared in step 2.
        bm.edge_to_col.remove(&delta.removed_uv_edge);

        // 8. Mark the pivot cache stale. The actual rebuild is deferred
        //    until the next slow-path call in `check_link_condition`
        //    (via `ensure_pivots_fresh`), which is rare in practice.
        bm.pivots_dirty = true;
    }

    /// Returns whether directed collapse u -> v is currently legal.
    /// Takes `&mut self` because `check_link_condition` may lazily
    /// refresh the pivot cache on its slow path.
    fn is_legal_collapse_candidate(&mut self, u: VIdx, v: VIdx) -> bool {
        if self.is_dead.contains(&u) || self.is_dead.contains(&v) {
            return false;
        }

        if !self
            .neighbors
            .get(&u)
            .is_some_and(|neighbor_set| neighbor_set.contains(&v))
        {
            return false;
        }

        if !self.edge_has_faces(u, v) {
            return false;
        }

        // Homology-aware link condition: for each shared neighbour `w`
        // of `u` and `v`, the 3-cycle `γ_w` (the corner triangle in the
        // 1-skeleton) is what the collapse would kill in `H₁`. Reject
        // iff any such `γ_w` has tunnel content. Strictly manifold
        // collapses are *not* required: non-manifold collapses and
        // bubble creation are both fine as long as no essential cycle
        // of the underlying solid dies.
        if !self.check_link_condition(u, v) {
            return false;
        }

        true
    }

    /// Counts all currently legal directed collapse candidates.
    /// Takes `&mut self` because `is_legal_collapse_candidate` may
    /// lazily refresh the pivot cache.
    fn count_legal_collapse_candidates(&mut self) -> usize {
        // Collect directed pairs first so we don't hold a borrow on
        // `self.neighbors` while calling `is_legal_collapse_candidate`,
        // which needs `&mut self`.
        let pairs: Vec<(VIdx, VIdx)> = {
            let mut p = Vec::new();
            for (&u, neighbor_set) in &self.neighbors {
                if self.is_dead.contains(&u) {
                    continue;
                }
                for &v in neighbor_set {
                    p.push((u, v));
                }
            }
            p
        };
        pairs
            .into_iter()
            .filter(|&(u, v)| self.is_legal_collapse_candidate(u, v))
            .count()
    }

    /// Logs a full diagnostic of the stuck state: aggregate counts, the
    /// connected-component structure of the sub-complex still carrying
    /// faces, and a per-directed-edge probe (link-check verdict plus
    /// hypothetical β₁, dim K, β₂) for every remaining face.
    ///
    /// Intended to fire once when surgery exits with `active_faces`
    /// non-empty. The point is to make it obvious whether we're stuck on a
    /// single irreducible "torus core" (a topological obstruction) or on
    /// scattered pathologies (a likely bug).
    fn dump_stuck_state_diagnostic(&mut self) {
        let v_now = self.positions.len() - self.is_dead.len();
        let f_now = self.active_faces.len();
        let e_now = self.boundary_matrix.edge_to_col.len();
        let chi = v_now as i64 - e_now as i64 + f_now as i64;
        let n_handle_cycles = self.boundary_matrix.handle_cycles.len();

        // Compute current β₁ and dim K (number of handle cycles still
        // independent of B₁) so we can fold them into a single grep-able
        // summary line.
        let (beta_1, dim_k) = {
            let mut snap = self.boundary_matrix.matrix.clone();
            snap.reduce();
            let rank = snap.rank();
            let pivot_map = snap.pivot_map();
            let beta_1 = e_now as i64 - v_now as i64 + 1 - rank as i64;
            let mut dim_k = 0i64;
            for cycle in &self.boundary_matrix.handle_cycles {
                let mut residue = cycle.clone();
                if !snap.reduce_against(&pivot_map, &mut residue) {
                    dim_k += 1;
                }
            }
            (beta_1, dim_k)
        };

        // Component count over the live sub-complex (vertices touching at
        // least one active face). Cheap BFS — we redo it for the per-
        // component breakdown below, but doing it here lets the one-line
        // summary include the count.
        let n_components = {
            let mut face_verts: HashSet<VIdx> = HashSet::new();
            for &[a, b, c] in &self.active_faces {
                face_verts.insert(a);
                face_verts.insert(b);
                face_verts.insert(c);
            }
            let mut adj: HashMap<VIdx, HashSet<VIdx>> = HashMap::new();
            for &[a, b, c] in &self.active_faces {
                adj.entry(a).or_default().extend([b, c]);
                adj.entry(b).or_default().extend([a, c]);
                adj.entry(c).or_default().extend([a, b]);
            }
            let mut visited: HashSet<VIdx> = HashSet::new();
            let mut count: usize = 0;
            for &start in &face_verts {
                if !visited.insert(start) {
                    continue;
                }
                count += 1;
                let mut queue: VecDeque<VIdx> = VecDeque::new();
                queue.push_back(start);
                while let Some(v) = queue.pop_front() {
                    if let Some(ns) = adj.get(&v) {
                        for &n in ns {
                            if visited.insert(n) {
                                queue.push_back(n);
                            }
                        }
                    }
                }
            }
            count
        };

        warn!(
            "STUCK-SUMMARY F={} V={} E={} chi={} target_g={} beta_1={} dim_K={} live_handle_cycles={} components={}",
            f_now, v_now, e_now, chi, self.target_g, beta_1, dim_k, n_handle_cycles, n_components
        );
        warn!(
            "Stuck-state aggregate: V={}, E={}, F={}, χ={}, target_g={}, live_handle_cycles={}.",
            v_now, e_now, f_now, chi, self.target_g, n_handle_cycles
        );

        // Vertices that still belong to at least one active face form the
        // "live sub-complex". Compute its connected components by BFS over
        // the adjacency restricted to faces.
        let mut face_verts: HashSet<VIdx> = HashSet::new();
        for &[a, b, c] in &self.active_faces {
            face_verts.insert(a);
            face_verts.insert(b);
            face_verts.insert(c);
        }
        let mut adj: HashMap<VIdx, HashSet<VIdx>> = HashMap::new();
        for &[a, b, c] in &self.active_faces {
            adj.entry(a).or_default().extend([b, c]);
            adj.entry(b).or_default().extend([a, c]);
            adj.entry(c).or_default().extend([a, b]);
        }
        let mut visited: HashSet<VIdx> = HashSet::new();
        let mut components: Vec<(usize, usize)> = Vec::new(); // (verts, faces)
        for &start in &face_verts {
            if visited.contains(&start) {
                continue;
            }
            let mut comp_verts: HashSet<VIdx> = HashSet::new();
            let mut queue: VecDeque<VIdx> = VecDeque::new();
            queue.push_back(start);
            visited.insert(start);
            comp_verts.insert(start);
            while let Some(v) = queue.pop_front() {
                if let Some(ns) = adj.get(&v) {
                    for &n in ns {
                        if visited.insert(n) {
                            comp_verts.insert(n);
                            queue.push_back(n);
                        }
                    }
                }
            }
            let comp_face_count = self
                .active_faces
                .iter()
                .filter(|&&[a, b, c]| {
                    comp_verts.contains(&a) && comp_verts.contains(&b) && comp_verts.contains(&c)
                })
                .count();
            components.push((comp_verts.len(), comp_face_count));
        }
        warn!(
            "Stuck sub-complex has {} connected component(s): {:?} (V, F per component).",
            components.len(),
            components
        );

        // Handle-cycle dump: for each tracked handle generator, report
        // - total edges,
        // - how many of those edges currently appear in the boundary of
        //   some active face (= face-incident, i.e. the edge could become
        //   a collapse candidate),
        // - the actual vertex pairs of the cycle's edges, so the user can
        //   correlate with the geometric view.
        let bm = &self.boundary_matrix;
        let mut face_incident_cols: HashSet<u32> = HashSet::new();
        for face in &self.active_faces {
            if let Some(&row_idx) = bm.face_to_row.get(face) {
                for &col in &bm.matrix.rows()[row_idx] {
                    face_incident_cols.insert(col);
                }
            }
        }
        let col_to_edge: HashMap<u32, [VIdx; 2]> =
            bm.edge_to_col.iter().map(|(&k, &v)| (v, k)).collect();

        for (i, cycle) in bm.handle_cycles.iter().enumerate() {
            let total = cycle.len();
            let face_incident = cycle
                .iter()
                .filter(|col| face_incident_cols.contains(col))
                .count();
            let orphaned = total - face_incident;
            warn!(
                "Handle cycle {}: {} edges total, {} face-incident (collapse-candidatable), {} orphaned (no incident face).",
                i, total, face_incident, orphaned
            );
            // Per-edge dump: vertex pair + whether it's face-incident.
            for &col in cycle {
                let endpoints = col_to_edge.get(&col).copied();
                let endpoints_str = match endpoints {
                    Some([a, b]) => format!("{:?}-{:?}", a, b),
                    None => "<edge no longer in edge_to_col>".to_string(),
                };
                let where_ = if face_incident_cols.contains(&col) {
                    "face-incident"
                } else {
                    "orphaned"
                };
                warn!("    col {} = {} [{}]", col, endpoints_str, where_);
            }
        }

        // Per-face dump. With ~10–20 stuck faces this is ~60–120 lines —
        // verbose, but the whole point of this hook is to let us read every
        // rejection.
        warn!(
            "Per-face per-directed-edge diagnostics ({} faces × 6 edges):",
            f_now
        );
        // Snapshot the face set so the diagnostic call below can take
        // `&mut self` without conflicting with this iterator's borrow.
        let faces_snapshot: Vec<[VIdx; 3]> = self.active_faces.iter().copied().collect();
        for (i, face) in faces_snapshot.iter().enumerate() {
            let [a, b, c] = *face;
            warn!("  face {} {:?}:", i, face);
            for (u, v) in [(a, b), (b, a), (b, c), (c, b), (a, c), (c, a)] {
                let summary = self.edge_diagnostic_summary(u, v);
                warn!("    {}", summary);
            }
        }
    }

    /// Returns a one-line diagnostic summary for directed edge u -> v.
    /// Takes `&mut self` because `check_link_condition` may lazily
    /// refresh the pivot cache.
    fn edge_diagnostic_summary(&mut self, u: VIdx, v: VIdx) -> String {
        let u_dead = self.is_dead.contains(&u);
        let v_dead = self.is_dead.contains(&v);
        let adjacent = self
            .neighbors
            .get(&u)
            .is_some_and(|neighbor_set| neighbor_set.contains(&v));
        let has_faces = adjacent && !u_dead && !v_dead && self.edge_has_faces(u, v);
        let link_ok = has_faces && self.check_link_condition(u, v);

        let stats_str = if has_faces {
            let (b, k, b2) = self.simulate_post_collapse_stats(u, v);
            format!(
                "new_β₁={}, new_dim_K={}, new_β₂={} (target g={})",
                b, k, b2, self.target_g,
            )
        } else {
            "post-collapse stats unavailable".to_string()
        };
        format!(
            "{:?}->{:?}[dead u/v={}/{}, adj={}, faces={}, link_ok={}, {}]",
            u, v, u_dead, v_dead, adjacent, has_faces, link_ok, stats_str
        )
    }

    /// Builds the initial `edge → incident-face count` map used by the
    /// free-face cleanup loop. Maintained incrementally afterwards by
    /// [`Self::try_free_face_collapse`].
    fn build_edge_face_count(&self) -> HashMap<[VIdx; 2], usize> {
        let mut counts: HashMap<[VIdx; 2], usize> = HashMap::new();
        for face in &self.active_faces {
            let [a, b, c] = *face;
            *counts.entry(sort_edge(a, b)).or_insert(0) += 1;
            *counts.entry(sort_edge(b, c)).or_insert(0) += 1;
            *counts.entry(sort_edge(a, c)).or_insert(0) += 1;
        }
        counts
    }

    /// Performs a single **free-face elementary collapse** if one is
    /// available, using a caller-maintained `edge_face_count`. A face is
    /// "free" when at least one of its three edges is incident only to
    /// that face. The collapse removes the face and that free edge — a
    /// textbook CW elementary collapse: a deformation retraction, so β₁,
    /// dim K, and β₂ are all preserved exactly. The legality criterion
    /// can't reject this operation, so we don't check it.
    ///
    /// On success the `edge_face_count` is updated in place: the free
    /// edge entry is dropped, and the two other face edges have their
    /// counts decremented.
    ///
    /// Returns `true` if a collapse was performed. Caller is expected to
    /// loop until this returns `false`, after which no free faces remain.
    fn try_free_face_collapse(&mut self, edge_face_count: &mut HashMap<[VIdx; 2], usize>) -> bool {
        // Look for a face that owns at least one count-1 edge.
        let mut picked: Option<([VIdx; 3], [VIdx; 2])> = None;
        for &face in &self.active_faces {
            let [a, b, c] = face;
            for &(x, y) in &[(a, b), (b, c), (a, c)] {
                let edge = sort_edge(x, y);
                if edge_face_count.get(&edge).copied().unwrap_or(0) == 1 {
                    picked = Some((face, edge));
                    break;
                }
            }
            if picked.is_some() {
                break;
            }
        }
        let Some((face, free_edge)) = picked else {
            return false;
        };

        // Update the incidence counts before mutating the mesh state:
        // the free edge disappears entirely; the other two face edges
        // each lose one incident face.
        let [a, b, c] = face;
        for &(x, y) in &[(a, b), (b, c), (a, c)] {
            let e = sort_edge(x, y);
            if e == free_edge {
                edge_face_count.remove(&e);
            } else if let Some(cnt) = edge_face_count.get_mut(&e) {
                *cnt -= 1;
            }
        }

        self.do_free_face_collapse(face, free_edge);
        true
    }

    /// Performs a free-face elementary collapse: remove `face` and the
    /// `free_edge` (which must currently be incident only to `face`).
    /// Updates the boundary matrix, handle cycles, and the live
    /// face/edge/neighbor maps, then rebuilds the cached pivot RREF.
    fn do_free_face_collapse(&mut self, face: [VIdx; 3], free_edge: [VIdx; 2]) {
        let bm = &mut self.boundary_matrix;
        let face_row = bm.face_to_row.get(&face).copied();
        let free_col = bm.edge_to_col.get(&free_edge).copied();

        // The face's two NON-free edges, in canonical sorted form.
        let [a, b, c] = face;
        let mut other_cols: [Option<u32>; 2] = [None, None];
        let mut next = 0;
        for &(x, y) in &[(a, b), (b, c), (a, c)] {
            let e = sort_edge(x, y);
            if e == free_edge {
                continue;
            }
            other_cols[next] = bm.edge_to_col.get(&e).copied();
            next += 1;
        }

        // Evolve handle cycles via the free-face chain map: e_free ↦
        // e_2 + e_3 on 1-chains. For each cycle that contains the free
        // edge's column, remove that column and toggle the other two.
        if let (Some(fc), Some(c2), Some(c3)) = (free_col, other_cols[0], other_cols[1]) {
            for cycle in &mut bm.handle_cycles {
                if let Ok(pos) = cycle.binary_search(&fc) {
                    cycle.remove(pos);
                    toggle_col_in_chain(cycle, c2);
                    toggle_col_in_chain(cycle, c3);
                }
            }
        }

        // Clear the face's matrix row.
        if let Some(row) = face_row {
            bm.matrix.clear_row(row);
        }
        bm.face_to_row.remove(&face);
        bm.edge_to_col.remove(&free_edge);

        // Mirror the active-faces set and the 1-skeleton adjacency.
        self.active_faces.remove(&face);
        let [u, v] = free_edge;
        if let Some(set) = self.neighbors.get_mut(&u) {
            set.remove(&v);
        }
        if let Some(set) = self.neighbors.get_mut(&v) {
            set.remove(&u);
        }

        // The matrix and handle cycles were both mutated; the cached
        // pivot RREF is stale until refreshed. We just flag it dirty —
        // the slow path of `check_link_condition` will refresh on
        // demand. For target_g == 0 the cache is never read anyway, so
        // the flag has no effect.
        self.boundary_matrix.pivots_dirty = true;
    }

    fn collapse_edge(&mut self, u: VIdx, v: VIdx) {
        // Compute the boundary-matrix delta against the pre-collapse state.
        let delta = self.compute_collapse_delta(u, v);

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

        // Commit the boundary-matrix delta to the live state. The delta
        // application marks the pivot cache dirty; the rebuild itself
        // is deferred until the next slow-path read of `aug_pivots`,
        // which is rare during typical surgery.
        Self::apply_collapse_delta(&mut self.boundary_matrix, &delta);
    }

    /// Build a graph snapshot; boundaries are assigned after successful surgery.
    fn to_skeleton(&self) -> Skeleton {
        let mut graph = Skeleton::default();
        let mut node_indices = HashMap::new();

        // Add living nodes
        for v in {
            let mut keys: Vec<_> = self.positions.keys().copied().collect();
            keys.sort_by(|a, b| a.partial_cmp(b).unwrap());
            keys
        } {
            let pos = self.positions[&v];
            if self.is_dead.contains(&v) {
                continue;
            }

            // Denormalize position to original model space
            let original_pos = pos / self.scale + self.center;

            // Get the list of original vertex keys this node represents
            let originals = self.vertex_to_original.get(&v).cloned().unwrap_or_default();

            let idx = graph.nodes.insert(SkeletonNode {
                position: original_pos,
                patch_vertices: originals,
            });
            node_indices.insert(v, idx);
        }

        for (&u, neighbor_set) in &self.neighbors {
            if self.is_dead.contains(&u) {
                continue;
            }

            if let Some(&u_node) = node_indices.get(&u) {
                for &v in neighbor_set {
                    // Only add each edge once (when u < v)
                    if u < v && !self.is_dead.contains(&v) {
                        if let Some(&v_node) = node_indices.get(&v) {
                            graph.edges.insert(SkeletonEdge {
                                nodes: [u_node, v_node],
                                boundary: PatchBoundary {
                                    edge_midpoints: Vec::new(),
                                },
                            });
                        }
                    }
                }
            }
        }

        graph
    }

    /// Remaining triangles in input coordinates, for inspecting failed surgery.
    fn remaining_face_triangles(&self) -> Vec<[Vector3D; 3]> {
        self.active_faces
            .iter()
            .map(|&[a, b, c]| {
                [
                    self.positions[&a] / self.scale + self.center,
                    self.positions[&b] / self.scale + self.center,
                    self.positions[&c] / self.scale + self.center,
                ]
            })
            .collect()
    }
}

/// Collapse the deflated surface without changing its tunnel cycles.
fn extract_skeleton(
    contracted_mesh: &Mesh<CONTRACTION>,
    original_mesh: &Mesh<INPUT>,
    options: Options,
) -> crate::SkeletonResult<(Skeleton, usize)> {
    let mut ctx = SurgeryContext::new(contracted_mesh, original_mesh)?;
    ctx.options = options;
    let mut heap = BinaryHeap::new();

    // Initial heap population
    let mut vert_ids: Vec<_> = ctx.neighbors.keys().copied().collect();
    vert_ids.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for u in vert_ids {
        if ctx.is_dead.contains(&u) {
            continue;
        }
        let mut neighbors: Vec<_> = ctx.neighbors[&u].iter().copied().collect();
        neighbors.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for v in neighbors {
            // Only add edges that have incident faces
            if ctx.edge_has_faces(u, v) {
                let cost = ctx.compute_collapse_cost(u, v);
                heap.push(CollapseCandidate { u, v, cost });
            }
        }
    }

    let mut collapses = 0;

    // Progress indicator: log a line each time we cross a 10%-of-initial
    // bucket. `bucket_index(f_now)` rounds down to which 10% bucket
    // `initial_faces − f_now` falls into; logging fires only when that
    // integer strictly exceeds `buckets_logged`. This avoids the duplicate-
    // log trap of comparing against a face-count threshold that drifts
    // away from a clean 10% step (the first threshold `initial − bucket`
    // is not generally a multiple of `bucket`, so snapping to multiples
    // can produce two consecutive logs at the same percent).
    let start = Instant::now();
    let initial_faces = ctx.active_faces.len();
    let bucket_size = (initial_faces / 10).max(1);
    let mut buckets_logged: usize = 0;
    let mut last_bucket_at = start;
    let mut last_bucket_faces = initial_faces;
    info!(
        "Connectivity surgery starting: V={}, F={}, target_g={}.",
        ctx.positions.len() - ctx.is_dead.len(),
        initial_faces,
        ctx.target_g,
    );

    // Greedy collapse loop
    while !ctx.active_faces.is_empty() {
        let Some(candidate) = heap.pop() else {
            let legal_candidates = ctx.count_legal_collapse_candidates();
            let remaining_faces = ctx.active_faces.len();
            warn!(
                "No more collapse candidates but {} faces remain. Legal directed candidates now: {}.",
                remaining_faces, legal_candidates
            );

            if legal_candidates > 0 {
                warn!(
                    "Heap is empty despite legal collapse candidates; the surgery loop's re-add fan-out missed an edge that should have stayed reachable."
                );
            }

            ctx.dump_stuck_state_diagnostic();
            break;
        };

        if !ctx.is_legal_collapse_candidate(candidate.u, candidate.v) {
            continue;
        }

        // Refresh stale entries (cost changed since insertion) instead of dropping them.
        let real_cost = ctx.compute_collapse_cost(candidate.u, candidate.v);
        const COST_EPSILON: f64 = 1e-8;
        if (candidate.cost - real_cost).abs() > COST_EPSILON {
            heap.push(CollapseCandidate {
                u: candidate.u,
                v: candidate.v,
                cost: real_cost,
            });
            continue;
        }

        // Perform the collapse
        ctx.collapse_edge(candidate.u, candidate.v);
        collapses += 1;

        // Progress: log one line per new 10% bucket crossed. A single
        // collapse can take out several faces and jump multiple buckets —
        // we still emit exactly one line, reporting the highest bucket
        // reached.
        let f_now = ctx.active_faces.len();
        let faces_done = initial_faces - f_now;
        let b = (faces_done / bucket_size).min(10);
        if b > buckets_logged {
            let now = Instant::now();
            let bucket_elapsed = now.duration_since(last_bucket_at).as_secs_f64();
            let total_elapsed = now.duration_since(start).as_secs_f64();
            let removed_this_bucket = last_bucket_faces - f_now;
            let rate = if bucket_elapsed > 1e-6 {
                removed_this_bucket as f64 / bucket_elapsed
            } else {
                0.0
            };
            info!(
                "Surgery progress: {}% (F={}/{}, -{} in {:.2}s, {:.0}/s, elapsed {:.2}s)",
                b * 10,
                f_now,
                initial_faces,
                removed_this_bucket,
                bucket_elapsed,
                rate,
                total_elapsed
            );
            last_bucket_at = now;
            last_bucket_faces = f_now;
            buckets_logged = b;
        }

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

    // Free-face elementary-collapse cleanup pass. Whenever the main
    // surgery loop leaves face-incident leftovers, free-face collapses can
    // clean up disk-like remnants (preserve homotopy exactly: no β₁,
    // dim K, or β₂ change). Doesn't help with closed-sub-complex
    // remnants (e.g. residual torus cores) — those have no free edges.
    //
    // The edge-incidence count is built once and then maintained
    // incrementally by each `try_free_face_collapse` so the cleanup loop
    // is O(F) total work rather than O(F²).
    let mut free_face_collapses = 0;
    let mut edge_face_count = ctx.build_edge_face_count();
    while ctx.try_free_face_collapse(&mut edge_face_count) {
        collapses += 1;
        free_face_collapses += 1;
    }

    let total_elapsed = start.elapsed().as_secs_f64();
    let succeeded = ctx.active_faces.is_empty();
    if succeeded {
        info!(
            "Connectivity surgery complete in {:.2}s. Collapsed {} edges ({} free-face). Remaining vertices: {}",
            total_elapsed,
            collapses,
            free_face_collapses,
            ctx.positions.len() - ctx.is_dead.len()
        );
        let skeleton = ctx.to_skeleton();
        Ok((skeleton, ctx.target_g))
    } else {
        warn!(
            "Connectivity surgery did NOT complete after {:.2}s: {} faces remain after {} collapses. Returning the partial state in an error.",
            total_elapsed,
            ctx.active_faces.len(),
            collapses
        );
        Err(Box::new(SurgeryFailure {
            reason: "Connectivity surgery stopped with remaining faces".into(),
            skeleton: ctx.to_skeleton(),
            remaining_face_positions: ctx.remaining_face_triangles(),
        }))
    }
}

/// Verifies the input is a closed connected 2-manifold, derives the genus,
/// and builds the GF(2) boundary matrix ∂₂. Returns `(g, BoundaryMatrix)`
/// — the `handle_cycles` field of the matrix is left empty here; callers
/// fill it in afterwards via [`attach_handle_cycles`] when genus > 0.
///
/// On failure (non-manifold edge, disconnected, non-integer genus, sanity
/// check mismatch), logs an error and returns `None`.
fn preprocess_topology(
    active_faces: &HashSet<[VIdx; 3]>,
    neighbors: &HashMap<VIdx, HashSet<VIdx>>,
    is_dead: &HashSet<VIdx>,
    n_verts_total: usize,
) -> Option<(usize, BoundaryMatrix)> {
    // Closed-manifold check: every undirected edge bounds exactly two faces.
    // Building the edge -> face count map also gives us E.
    let mut edge_face_count: HashMap<[VIdx; 2], usize> = HashMap::new();
    for &face in active_faces {
        let [a, b, c] = face;
        *edge_face_count.entry(sort_edge(a, b)).or_insert(0) += 1;
        *edge_face_count.entry(sort_edge(b, c)).or_insert(0) += 1;
        *edge_face_count.entry(sort_edge(a, c)).or_insert(0) += 1;
    }
    for (&edge, &count) in &edge_face_count {
        if count != 2 {
            error!(
                "Connectivity surgery aborted: input mesh is not a closed 2-manifold (edge {:?} is incident to {} faces, expected 2).",
                edge, count
            );
            return None;
        }
    }

    // Connectedness check: BFS over `neighbors` from any alive vertex must
    // reach every alive vertex.
    let alive_count = n_verts_total - is_dead.len();
    if alive_count == 0 {
        error!("Connectivity surgery aborted: input mesh has no vertices.");
        return None;
    }
    let start = *neighbors
        .keys()
        .find(|v| !is_dead.contains(v))
        .expect("non-empty mesh");
    let mut visited: HashSet<VIdx> = HashSet::with_capacity(alive_count);
    let mut queue: VecDeque<VIdx> = VecDeque::new();
    visited.insert(start);
    queue.push_back(start);
    while let Some(v) = queue.pop_front() {
        if let Some(ns) = neighbors.get(&v) {
            for &n in ns {
                if !is_dead.contains(&n) && visited.insert(n) {
                    queue.push_back(n);
                }
            }
        }
    }
    if visited.len() != alive_count {
        error!(
            "Connectivity surgery aborted: input mesh is not connected ({} of {} vertices reachable from start).",
            visited.len(),
            alive_count
        );
        return None;
    }

    // Euler characteristic and genus.
    let v_count = alive_count;
    let e_count = edge_face_count.len();
    let f_count = active_faces.len();
    let chi: i64 = v_count as i64 - e_count as i64 + f_count as i64;
    let two_g: i64 = 2 - chi;
    if two_g < 0 || two_g % 2 != 0 {
        error!(
            "Connectivity surgery aborted: derived genus is not a non-negative integer (V={}, E={}, F={}, χ={}, 2g={}).",
            v_count, e_count, f_count, chi, two_g
        );
        return None;
    }
    let g = (two_g / 2) as usize;

    info!(
        "Connectivity surgery preprocessing: V={}, E={}, F={}, χ={}, genus={}.",
        v_count, e_count, f_count, chi, g
    );

    // Build ∂₂: assign columns to edges, rows to faces.
    let mut edge_to_col: HashMap<[VIdx; 2], u32> = HashMap::with_capacity(e_count);
    for &edge in edge_face_count.keys() {
        let next = edge_to_col.len() as u32;
        edge_to_col.insert(edge, next);
    }

    let mut rows: Vec<Vec<u32>> = Vec::with_capacity(f_count);
    let mut face_to_row: HashMap<[VIdx; 3], usize> = HashMap::with_capacity(f_count);
    for &face in active_faces {
        let [a, b, c] = face;
        let mut row = [
            edge_to_col[&sort_edge(a, b)],
            edge_to_col[&sort_edge(b, c)],
            edge_to_col[&sort_edge(a, c)],
        ];
        row.sort();
        let row_idx = rows.len();
        rows.push(row.to_vec());
        face_to_row.insert(face, row_idx);
    }

    let matrix = F2Matrix::from_rows(rows);
    // Compute the rank on a clone so the live `matrix` keeps each row as
    // its face's literal boundary. `apply_collapse_delta` later identifies
    // a face's row by `face_to_row[face]` and clears it; that's only
    // correct if the row content still matches that face. A row-echelon
    // reduce on the live matrix XORs rows together — at least one face's
    // row becomes empty (the closed surface's fundamental 2-cycle is the
    // sum of all face rows = 0) — and subsequent face removals would clear
    // the *wrong* content.
    let rank = {
        let mut clone = matrix.clone();
        clone.reduce();
        clone.rank()
    };

    // Sanity: β₁ = E − V + 1 − rank(∂₂) should equal 2g for a closed
    // connected orientable 2-manifold.
    let beta_1 = e_count as i64 - v_count as i64 + 1 - rank as i64;
    if beta_1 != two_g {
        error!(
            "Connectivity surgery aborted: β₁ sanity check failed (got {}, expected 2g = {}). Input topology is not what was assumed.",
            beta_1, two_g
        );
        return None;
    }

    let mut bm = BoundaryMatrix {
        matrix,
        face_to_row,
        edge_to_col,
        handle_cycles: Vec::new(),
        aug_pivots: F2Matrix::from_rows(Vec::new()),
        aug_pivot_map: HashMap::new(),
        // `rebuild_pivots` below clears this; for the genus-0 path we
        // leave it true since the cache is never consulted.
        pivots_dirty: true,
    };
    // For genus > 0, `attach_handle_cycles` will rebuild the pivot
    // cache after writing the handle generators. For genus-0 the cache
    // is never consulted (see the `target_g == 0` short-circuit in
    // `check_link_condition`), so we leave it empty.
    if g > 0 {
        bm.rebuild_pivots();
    }
    Some((g, bm))
}

/// Tetrahedralizes `original_mesh`, computes the handle subspace
/// `K = ker(H₁(S; ℤ/2) → H₁(V; ℤ/2))`, and stores a basis of `K` as
/// `handle_cycles` on `bm`, translated into surgery's contraction edge
/// basis (column indices in `bm.matrix`).
///
/// The translation works because the contracted mesh shares vertex raw
/// values with the input mesh (`SurgeryContext::new` initialises
/// `vertex_to_original` via `VertKey::<INPUT>::new(v.raw())`), and the
/// shared TetGen wrapper verifies every input surface triangle, so each surface
/// edge is also a contraction-mesh edge with a column entry in
/// `bm.edge_to_col`.
///
/// Returns `None` (after logging) if the input is malformed enough to
/// break this invariant. `target_g` is supplied for the post-condition
/// sanity check (we should produce exactly `g` handle cycles).
fn attach_handle_cycles(
    bm: &mut BoundaryMatrix,
    original_mesh: &Mesh<INPUT>,
    target_g: usize,
) -> crate::SkeletonResult<()> {
    // Orientation sanity check on the input. A closed manifold with χ=2-2g
    // is a handlebody surface only if it's orientable; a non-orientable
    // 2-manifold with the same χ would be a sum of crosscaps that doesn't
    // bound any 3-manifold in R³. We don't yet check intrinsic
    // orientability — that requires a BFS orientation propagation — but the
    // cheaper check ("is the face winding consistent across edges?") catches
    // both genuinely non-orientable inputs and inputs whose triangle list
    // mixes inward- and outward-facing windings. Either makes TetGen
    // misbehave, and that misbehaviour shows up downstream as a
    // handle-subspace dimension mismatch with no obvious cause.
    if let Err(diag) = check_face_winding_consistent(original_mesh) {
        error!(
            "Connectivity surgery aborted: input mesh has inconsistent face winding ({}). \
             This usually means the surface is non-orientable, or the triangle list mixes \
             outward- and inward-facing windings. Either prevents TetGen from producing a \
             valid interior tetrahedralization.",
            diag
        );
        return Err(diag.into());
    }

    let tm = match tetrahedralize(original_mesh) {
        Ok(t) => t,
        Err(e) => {
            error!(
                "Connectivity surgery aborted: tetrahedralization failed: {}",
                e
            );
            return Err(e);
        }
    };
    if tm.n_input_vertices != original_mesh.nr_verts() || tm.n_vertices() < tm.n_input_vertices {
        return Err("TetGen returned inconsistent vertex counts".into());
    }
    if tm.n_tets() == 0 {
        error!(
            "Connectivity surgery aborted: TetGen returned 0 tetrahedra. The input surface \
             likely violates a TetGen assumption (self-intersecting, near-degenerate \
             triangles, or a region marker that didn't land inside the solid)."
        );
        return Err("TetGen volume does not match the input surface topology".into());
    }
    let tb = build_tet_boundary(&tm);
    info!(
        "Tet boundary complex: {} triangles, {} edges across {} tetrahedra.",
        tb.n_triangles(),
        tb.n_edges(),
        tm.n_tets()
    );
    let hs = compute_handle_subspace(original_mesh, &tb, target_g);

    if hs.genus() != target_g {
        error!(
            "Connectivity surgery aborted: handle-subspace dimension mismatch (got {}, \
             expected genus {}). Pipeline state: surface V/E/F={}/{}/{}, tet T/E/cells={}/{}/{}. \
             Most common cause: the input surface and the TetGen-produced volume disagree on \
             which 1-cycles bound — e.g. TetGen produced an interior that's topologically \
             different from the handlebody we inferred from χ, because the surface is \
             non-orientable, self-intersecting, or has a non-watertight shell that nevertheless \
             happens to pass the 2-incidence manifold check.",
            hs.cycles.len(),
            target_g,
            // Surface stats are accessible from `hs.surface_edges` and the
            // input mesh; we report them inline so the operator doesn't
            // have to cross-reference earlier log lines.
            original_mesh.vert_ids().len(),
            hs.surface_edges.len(),
            original_mesh.face_ids().len(),
            tb.n_triangles(),
            tb.n_edges(),
            tm.n_tets(),
        );
        return Err("TetGen volume does not match the input surface topology".into());
    }

    // Build a translation table from surface-edge index → contraction-mesh
    // column index in bm.matrix. Surface edges are pairs of u32 positions
    // in original_mesh.vert_ids(); contraction VertKeys share raw values
    // with the input mesh.
    let input_vert_ids = original_mesh.vert_ids();
    let surface_to_col: Vec<u32> = hs
        .surface_edges
        .iter()
        .map(|&[a, b]| {
            let a_input = input_vert_ids[a as usize];
            let b_input = input_vert_ids[b as usize];
            let a_contr = VIdx::new(a_input.raw());
            let b_contr = VIdx::new(b_input.raw());
            *bm.edge_to_col.get(&sort_edge(a_contr, b_contr)).unwrap_or_else(|| {
                panic!(
                    "handle-cycle translation: input surface edge ({:?}, {:?}) missing from contraction edge_to_col",
                    a_input, b_input
                )
            })
        })
        .collect();

    let translate = |cycle: Vec<u32>| -> Vec<u32> {
        let mut translated: Vec<u32> = cycle
            .into_iter()
            .map(|surf_idx| surface_to_col[surf_idx as usize])
            .collect();
        translated.sort();
        translated
    };

    bm.handle_cycles = hs.cycles.into_iter().map(translate).collect();
    // Augmented pivot set is now `B₁(X) + K`, the full target span for
    // the legality check.
    bm.rebuild_pivots();

    info!(
        "Attached {} handle-cycle generators to the boundary matrix.",
        bm.handle_cycles.len()
    );

    Ok(())
}

/// Walks every face of `mesh` and checks that each undirected edge is
/// traversed in opposite directions by its two incident faces. If two
/// faces traverse the same edge in the same direction, the mesh is
/// either non-orientable or has at least one flipped triangle in its
/// storage — both make TetGen produce a malformed interior. Returns
/// `Err(diagnostic)` describing the first offending edge.
fn check_face_winding_consistent(mesh: &Mesh<INPUT>) -> Result<(), String> {
    // For each directed edge (x, y) we've seen, remember the face it came
    // from. A consistent orientation means every undirected edge is seen
    // exactly once in each direction: (x, y) by face A and (y, x) by
    // face B. Seeing the same directed edge twice = same winding on both
    // sides = inconsistent.
    let mut seen_directed: HashMap<(VertKey<INPUT>, VertKey<INPUT>), [VertKey<INPUT>; 3]> =
        HashMap::new();
    for fid in mesh.face_ids() {
        let verts: Vec<_> = mesh.vertices(fid).collect();
        if verts.len() != 3 {
            return Err(format!(
                "non-triangular face (vertex count = {})",
                verts.len()
            ));
        }
        let face = [verts[0], verts[1], verts[2]];
        for &(x, y) in &[(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] {
            if let Some(prev_face) = seen_directed.insert((x, y), face) {
                return Err(format!(
                    "directed edge ({:?}, {:?}) traversed twice — first by face {:?}, then by face {:?}",
                    x, y, prev_face, face
                ));
            }
        }
    }
    Ok(())
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
    f.sort_by(|a, b| a.partial_cmp(b).unwrap());
    f
}

#[cfg(test)]
#[path = "test/test_mod.rs"]
mod test_mod;
/// Failed surgery or partition construction, with the available partial state.
#[derive(Debug)]
pub struct SurgeryFailure {
    pub reason: String,
    /// Inspection only: incomplete surgery can leave empty boundaries.
    pub skeleton: Skeleton,
    pub remaining_face_positions: Vec<[Vector3D; 3]>,
}
impl std::fmt::Display for SurgeryFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.reason)
    }
}
impl std::error::Error for SurgeryFailure {}

/// Reduce the deflated mesh while keeping patches on the original surface.
pub fn run(
    contracted: &Mesh<CONTRACTION>,
    original: &Mesh<INPUT>,
) -> crate::SkeletonResult<Skeleton> {
    run_with_options(contracted, original, Options::default())
}

pub fn run_with_options(
    contracted: &Mesh<CONTRACTION>,
    original: &Mesh<INPUT>,
    options: Options,
) -> crate::SkeletonResult<Skeleton> {
    options.validate()?;
    crate::validate::input_genus(original)?;
    let converted = Mesh::<CONTRACTION>::convert(original);
    if converted.vert_ids() != contracted.vert_ids()
        || converted.edge_ids() != contracted.edge_ids()
        || converted.face_ids() != contracted.face_ids()
        || converted.edge_ids().into_iter().any(|e| {
            converted.root(e) != contracted.root(e)
                || converted.twin(e) != contracted.twin(e)
                || converted.next(e) != contracted.next(e)
                || converted.face(e) != contracted.face(e)
        })
    {
        return Err("Deflation must preserve mesh topology and vertex IDs".into());
    }
    if contracted
        .vert_ids()
        .iter()
        .any(|&v| !contracted.position(v).iter().all(|x| x.is_finite()))
    {
        return Err("Deflated positions must be finite".into());
    }
    let (mut skeleton, genus) = extract_skeleton(contracted, original, options)?;
    let result = crate::validate::assign_boundaries(&mut skeleton, original)
        .and_then(|()| skeleton.validate(original))
        .and_then(|()| {
            if skeleton.edges.len() + 1 != skeleton.nodes.len() + genus {
                Err("Extracted graph does not preserve input genus".into())
            } else {
                Ok(())
            }
        });
    if let Err(error) = result {
        return Err(Box::new(SurgeryFailure {
            reason: error.to_string(),
            skeleton,
            remaining_face_positions: Vec::new(),
        }));
    }
    Ok(skeleton)
}
