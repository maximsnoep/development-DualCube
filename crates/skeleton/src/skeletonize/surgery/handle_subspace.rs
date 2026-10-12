//! Compute the **handle subspace** `K = ker(H₁(S; ℤ/2) → H₁(V; ℤ/2))`
//! given the input surface mesh `S` and a tetrahedralization `T` of the
//! solid `V` bounded by it.
//!
//! The output is `g` linearly independent 1-cycles `c₁, …, c_g` on `S`,
//! each of which bounds a 2-chain in `T` (= bounds inside the solid =
//! handle loop, per the design note in `HOMOLOGY_PRESERVING_SURGERY.md`).
//!
//! Algorithm:
//! 1. Enumerate surface edges and faces with stable `u32` indices.
//! 2. Build `∂₂^S` over GF(2) and reduce it; its row span is `B₁(S)`, the
//!    surface-boundary subspace.
//! 3. Build a spanning tree of the surface 1-skeleton. Every non-tree edge
//!    `e` produces a *fundamental cycle*: `e` plus the unique tree path
//!    between its endpoints. The set of fundamental cycles is a basis for
//!    the cycle space `Z₁(S)`. There are `E_S − V_S + 1` of them.
//! 4. For each fundamental cycle, lift it to a 1-chain in `C₁(T)` by
//!    translating surface edge indices to T-edge column indices (input
//!    vertices keep their indices in T by the [`super::tetrahedralize::tetrahedralize`]
//!    invariant). Reduce the lifted chain against the RREF of `∂₂^T`;
//!    if it reduces to zero, the cycle bounds in `T` — it's a handle.
//! 5. Among the collected handle candidates, keep only those that are
//!    linearly independent in `H₁(S) = Z₁(S) / B₁(S)`. The result is a
//!    basis for `K`.

use std::collections::{HashMap, HashSet, VecDeque};

use mehsh::prelude::{HasVertices, Mesh, VertKey};
use tracing::{info, warn};

use super::f2_rref::F2Matrix;
use super::tet_boundary::TetBoundaryComplex;
use dualcube_types::prelude::INPUT;

/// A basis for the handle subspace `K`, plus the surface-edge indexing
/// the basis cycles are expressed over.
#[derive(Debug, Clone)]
pub struct HandleSubspace {
    /// `g` linearly independent handle cycles (= elements of `K`, the
    /// kernel of `H₁(S) → H₁(V)`). Each is a sorted list of surface-edge
    /// indices (indices into `surface_edges`).
    pub cycles: Vec<Vec<u32>>,
    /// Surface edges, each as a sorted pair of vertex indices. The vertex
    /// index is the position of the vertex in `mesh.vert_ids()` — i.e. the
    /// same indexing the [`super::tetrahedralize::TetMesh`] uses for the
    /// first `n_input_vertices` of its vertex list.
    pub surface_edges: Vec<[u32; 2]>,
}

impl HandleSubspace {
    /// `g` = number of handle cycles = genus of the solid (when input was
    /// a closed orientable surface).
    pub fn genus(&self) -> usize {
        self.cycles.len()
    }
}

/// Computes a basis for the handle subspace `K`.
///
/// The input mesh `mesh` should be the original closed orientable input
/// surface; `tet_boundary` must come from `build_tet_boundary` applied to
/// the result of `tetrahedralize(mesh)`. `target_g` is the expected
/// number of handle generators (= genus of the input surface, if it
/// bounds a handlebody). Return the full basis so callers can reject a
/// dimension mismatch, including excess generators.
pub fn compute_handle_subspace(
    mesh: &Mesh<INPUT>,
    tet_boundary: &TetBoundaryComplex,
    target_g: usize,
) -> HandleSubspace {
    // ---- Step 1: enumerate surface vertices, edges, and faces. ----
    let vert_ids: Vec<VertKey<INPUT>> = mesh.vert_ids();
    let vert_index: HashMap<VertKey<INPUT>, u32> = vert_ids
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i as u32))
        .collect();
    let n_verts = vert_ids.len();

    let mut surface_edges: Vec<[u32; 2]> = Vec::new();
    let mut edge_to_index: HashMap<[u32; 2], u32> = HashMap::new();
    let mut surface_faces: Vec<[u32; 3]> = Vec::new();
    for face_id in mesh.face_ids() {
        let verts: Vec<_> = mesh.vertices(face_id).collect();
        debug_assert_eq!(
            verts.len(),
            3,
            "compute_handle_subspace requires a triangulated mesh"
        );
        let [a, b, c] = [
            vert_index[&verts[0]],
            vert_index[&verts[1]],
            vert_index[&verts[2]],
        ];
        let face = sort_triple(a, b, c);
        surface_faces.push(face);
        for (x, y) in [(face[0], face[1]), (face[1], face[2]), (face[0], face[2])] {
            let pair = sort_pair(x, y);
            edge_to_index.entry(pair).or_insert_with(|| {
                let id = surface_edges.len() as u32;
                surface_edges.push(pair);
                id
            });
        }
    }

    // ---- Step 2: build and reduce ∂₂^S so its rows span B₁(S). ----
    let mut surface_d2 = F2Matrix::from_rows(
        surface_faces
            .iter()
            .map(|face| {
                let mut cols = [
                    edge_to_index[&sort_pair(face[0], face[1])],
                    edge_to_index[&sort_pair(face[1], face[2])],
                    edge_to_index[&sort_pair(face[0], face[2])],
                ];
                cols.sort();
                cols.to_vec()
            })
            .collect(),
    );
    surface_d2.reduce();

    // ---- Step 3: build a spanning tree of the surface 1-skeleton. ----
    let mut surface_adj: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
    for (edge_idx, &[a, b]) in surface_edges.iter().enumerate() {
        surface_adj.entry(a).or_default().push((b, edge_idx as u32));
        surface_adj.entry(b).or_default().push((a, edge_idx as u32));
    }
    let mut parent_vertex: Vec<Option<u32>> = vec![None; n_verts];
    let mut parent_edge: Vec<Option<u32>> = vec![None; n_verts];
    let mut visited: Vec<bool> = vec![false; n_verts];
    let mut tree_edges: HashSet<u32> = HashSet::new();
    visited[0] = true;
    let mut queue: VecDeque<u32> = VecDeque::new();
    queue.push_back(0);
    while let Some(v) = queue.pop_front() {
        if let Some(neighbors) = surface_adj.get(&v) {
            for &(n, edge_idx) in neighbors {
                if !visited[n as usize] {
                    visited[n as usize] = true;
                    parent_vertex[n as usize] = Some(v);
                    parent_edge[n as usize] = Some(edge_idx);
                    tree_edges.insert(edge_idx);
                    queue.push_back(n);
                }
            }
        }
    }

    // ---- Step 4: reduce ∂₂^T so its rows pivot uniquely (REF). The
    // tagged-Gauss pass below initialises its pivot table directly from
    // these rows — it doesn't need a separate `pivot_map`.
    let mut tet_d2 = tet_boundary.boundary_2.clone();
    tet_d2.reduce();
    let surface_b1_rank = surface_d2.rank();
    let surface_z1_dim = (surface_edges.len() as i64) - (n_verts as i64) + 1;
    let surface_h1_dim = surface_z1_dim - (surface_b1_rank as i64);
    info!(
        "Handle-subspace: surface V={}, E={}, F={}; tet T={}, E={}; rank(∂₂^S)={}, rank(∂₂^T)={}; β₁(S)={}.",
        n_verts,
        surface_edges.len(),
        surface_faces.len(),
        tet_boundary.n_triangles(),
        tet_boundary.n_edges(),
        surface_b1_rank,
        tet_d2.rank(),
        surface_h1_dim,
    );

    // Translation table: surface edge index → T-edge column index.
    // Input surface vertices keep their indices in T (see TetMesh docs),
    // and the shared wrapper uses TetGen's `M` + `Y` flags to preserve
    // every input triangle, so every surface edge must appear as a
    // T-edge.
    let surface_to_t_col: Vec<u32> = surface_edges
        .iter()
        .map(|&[a, b]| {
            tet_boundary
                .edge_col(a, b)
                .expect("every surface edge must appear in the tet complex")
        })
        .collect();

    // Sanity-check the lifting: GF(2) sum of "lift each surface edge"
    // must give a distinct column for each edge. If two surface edges land
    // in the same T column, the surface 1-skeleton wasn't preserved
    // injectively by TetGen and everything below this point is suspect.
    {
        let mut seen: HashSet<u32> = HashSet::with_capacity(surface_to_t_col.len());
        let mut collisions: usize = 0;
        for &c in &surface_to_t_col {
            if !seen.insert(c) {
                collisions += 1;
            }
        }
        if collisions > 0 {
            warn!(
                "Handle-subspace: surface→T lifting has {} duplicate column(s) ({} surface edges, {} unique T cols). The handle test relies on injectivity.",
                collisions,
                surface_to_t_col.len(),
                seen.len()
            );
        }
    }

    // ---- Step 5 + 6 (combined): tagged Gauss elimination over T-edges. ----
    //
    // Each non-tree edge gives one fundamental cycle of `S`. The
    // straightforward "test each cycle individually for K membership"
    // approach fails when `K`'s non-zero classes are only representable
    // as *sums* of multiple fundamental cycles (which happens whenever
    // the spanning tree gives an unfortunate basis for `H₁(S)`).
    //
    // To find such combinations we track, for every row in the running
    // RREF, a `tag` = the set of fundamental-cycle indices whose lifted
    // sum produced that row. Initial state:
    //   * `B₁(T)` rows: tag = ∅ (they're tet-face boundaries, not
    //     combinations of surface cycles).
    //   * Each lifted cycle `i`: tag = {i}.
    // During reduction we XOR both the T-edge chain and the tag in
    // lockstep, so the tag stays consistent with the chain's pedigree.
    //
    // When a row reduces to the empty chain in T-edge space, its tag is
    // exactly a subset of fundamental cycles whose sum lies in `B₁(T)`,
    // i.e. a *K element*. We then build that K element's surface chain
    // and reduce against `B₁(S) ∪ already-chosen K basis` to test
    // independence in `H₁(S)`. Keep every independent basis member.
    let edges_in_t: &[[u32; 2]] = &tet_boundary.edges;

    // Build every fundamental cycle up front so we can rebuild surface
    // chains from tags on demand.
    let mut all_cycles: HashMap<u32, Vec<u32>> = HashMap::new();
    for edge_idx in 0..(surface_edges.len() as u32) {
        if !tree_edges.contains(&edge_idx) {
            let cycle = fundamental_cycle_on_surface(
                edge_idx,
                &surface_edges,
                &parent_vertex,
                &parent_edge,
            );
            all_cycles.insert(edge_idx, cycle);
        }
    }

    // Pivots in T-edge space, each storing the chain and its tag pedigree.
    // Initialised with the (filtered) B₁(T) row span — empty tags.
    let mut tagged_pivots: HashMap<u32, (Vec<u32>, Vec<u32>)> = HashMap::new();
    for row in tet_d2.rows() {
        if let Some(&leading) = row.first() {
            tagged_pivots.insert(leading, (row.clone(), Vec::new()));
        }
    }

    // K-basis pivot system on surface edges: starts as B₁(S) pivots and
    // grows with each K basis member added.
    let mut k_basis_pivots: HashMap<u32, Vec<u32>> = HashMap::new();
    for row in surface_d2.rows() {
        if let Some(&leading) = row.first() {
            k_basis_pivots.insert(leading, row.clone());
        }
    }

    let mut chosen: Vec<Vec<u32>> = Vec::new();
    let mut tested: usize = 0;
    let mut bound_in_t: usize = 0;
    let mut bound_individually: usize = 0;
    let mut bound_multi: usize = 0;
    let mut sample_non_bounding: Vec<(usize, usize, usize)> = Vec::new();
    let mut non_cycle_chains: usize = 0;
    // For dim(image H₁(S) → H₁(T)) we just count how many lifted cycles
    // survived to become new tagged_pivots beyond the B₁(T) initial set.
    let mut dim_image: usize = 0;

    // Process all cycles so an unexpected kernel dimension reaches the caller.
    for edge_idx in 0..(surface_edges.len() as u32) {
        if tree_edges.contains(&edge_idx) {
            continue;
        }
        let cycle = &all_cycles[&edge_idx];

        let mut chain: Vec<u32> = cycle
            .iter()
            .map(|&i| surface_to_t_col[i as usize])
            .collect();
        chain.sort();
        let chain_len_before = chain.len();
        let mut tag: Vec<u32> = vec![edge_idx];

        tested += 1;

        // Reduce in T-edge space; XOR tags in lockstep.
        while let Some(&leading) = chain.first() {
            if let Some((p_chain, p_tag)) = tagged_pivots.get(&leading) {
                chain = xor_sorted(&chain, p_chain);
                tag = xor_sorted(&tag, p_tag);
                continue;
            }
            break;
        }

        if chain.is_empty() {
            bound_in_t += 1;
            if tag.len() == 1 {
                bound_individually += 1;
            } else {
                bound_multi += 1;
            }

            // `tag` indexes a sum of fundamental cycles whose lift is in
            // B₁(T) — a K element. Build its surface chain.
            let mut surface_chain: Vec<u32> = Vec::new();
            for &cyc_idx in &tag {
                surface_chain = xor_sorted(&surface_chain, &all_cycles[&cyc_idx]);
            }

            // Reduce mod (B₁(S) ∪ chosen K basis).
            let mut residue = surface_chain.clone();
            while let Some(&leading) = residue.first() {
                if let Some(pivot) = k_basis_pivots.get(&leading) {
                    residue = xor_sorted(&residue, pivot);
                    continue;
                }
                break;
            }
            if !residue.is_empty() {
                k_basis_pivots.insert(residue[0], residue);
                chosen.push(surface_chain);
            }
        } else {
            // Sanity: a surface 1-cycle lifted should be a 1-cycle in T.
            // If not, the lift isn't a chain map.
            if !chain_is_cycle_in_tet(&chain, edges_in_t) {
                non_cycle_chains += 1;
            }
            if sample_non_bounding.len() < 5 {
                sample_non_bounding.push((cycle.len(), chain_len_before, chain.len()));
            }
            // Independent in H₁(T): adds a new dim to the image.
            dim_image += 1;
            tagged_pivots.insert(chain[0], (chain, tag));
        }
    }

    let total_extracted = chosen.len();
    info!(
        "Handle-subspace: tested {} fundamental cycles, {} bound in T (= K elements): {} singleton-cycles, {} multi-cycle sums. dim(image H₁(S)→H₁(T)) = {}, total K basis members found = {}. Returning {} (target {}).",
        tested,
        bound_in_t,
        bound_individually,
        bound_multi,
        dim_image,
        total_extracted,
        chosen.len(),
        target_g,
    );
    if non_cycle_chains > 0 {
        warn!(
            "Handle-subspace: {} lifted residues were NOT cycles in T (∂_T ≠ 0). This means the lift `surface edge → T edge` is not a chain map — most likely cause: TetGen renumbered surface vertices, so `edge_col(a, b)` returns the column for a different T edge.",
            non_cycle_chains,
        );
    }
    // Algebraic cross-check: dim K = β₁(S) − dim(image H₁(S) → H₁(T)),
    // and dim K should match the total number of K basis members we
    // pulled out of the full tagged-Gauss pass. A mismatch means either
    // TetGen filled a region whose topology doesn't match the
    // handlebody we assumed, or there's a bug in this routine.
    let algebraic_k_dim = surface_h1_dim - (dim_image as i64);
    if (total_extracted as i64) != algebraic_k_dim {
        warn!(
            "Handle-subspace: total K basis members ({}) disagrees with algebraic K dim ({} = β₁(S){} − dim(image){}). TetGen likely filled a region whose topology differs from the expected handlebody.",
            total_extracted, algebraic_k_dim, surface_h1_dim, dim_image,
        );
    }
    if total_extracted != target_g {
        // Even if the algebraic check passes (rare), if we found more or
        // fewer K basis members than `target_g`, surgery's invariant
        // β₁ − dim K = g won't behave as expected.
        warn!(
            "Handle-subspace: found {} K basis members but `target_g` is {}. surgery may behave unexpectedly.",
            total_extracted, target_g
        );
    }

    HandleSubspace {
        cycles: chosen,
        surface_edges,
    }
}

/// Builds the fundamental cycle for a non-tree edge `e = (u, v)` as the
/// symmetric difference of `path_to_root(u)` and `path_to_root(v)` plus the
/// edge itself. Output is a sorted `Vec<u32>` of surface-edge indices.
fn fundamental_cycle_on_surface(
    non_tree_edge_idx: u32,
    surface_edges: &[[u32; 2]],
    parent_vertex: &[Option<u32>],
    parent_edge: &[Option<u32>],
) -> Vec<u32> {
    let [u, v] = surface_edges[non_tree_edge_idx as usize];

    let mut path_u: HashSet<u32> = HashSet::new();
    let mut cur = u;
    while let Some(pe) = parent_edge[cur as usize] {
        path_u.insert(pe);
        cur = parent_vertex[cur as usize].expect("parent_edge present implies parent_vertex");
    }
    let mut path_v: HashSet<u32> = HashSet::new();
    let mut cur = v;
    while let Some(pe) = parent_edge[cur as usize] {
        path_v.insert(pe);
        cur = parent_vertex[cur as usize].expect("parent_edge present implies parent_vertex");
    }

    let mut cycle: Vec<u32> = path_u
        .symmetric_difference(&path_v)
        .copied()
        .chain(std::iter::once(non_tree_edge_idx))
        .collect();
    cycle.sort();
    cycle
}

/// Tests whether a 1-chain (given as a sorted list of T-edge column
/// indices) is a cycle in the tet complex — i.e. every vertex appears in
/// an even number of incident edges. Used as a sanity check that the
/// lift `surface edge → T edge` is actually a chain map; if it isn't,
/// the lifted residue won't be a cycle.
fn chain_is_cycle_in_tet(chain: &[u32], edges: &[[u32; 2]]) -> bool {
    let mut parity: HashMap<u32, bool> = HashMap::new();
    for &col in chain {
        let [a, b] = edges[col as usize];
        let pa = parity.entry(a).or_insert(false);
        *pa = !*pa;
        let pb = parity.entry(b).or_insert(false);
        *pb = !*pb;
    }
    parity.values().all(|&p| !p)
}

/// Symmetric difference of two sorted-distinct `u32` slices.
fn xor_sorted(a: &[u32], b: &[u32]) -> Vec<u32> {
    use std::cmp::Ordering;
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

#[inline]
pub fn sort_pair(a: u32, b: u32) -> [u32; 2] {
    if a < b { [a, b] } else { [b, a] }
}

#[inline]
fn sort_triple(a: u32, b: u32, c: u32) -> [u32; 3] {
    let mut x = [a, b, c];
    x.sort();
    x
}

#[cfg(test)]
#[path = "test/test_handle_subspaces.rs"]
mod test_handle_subspaces;
