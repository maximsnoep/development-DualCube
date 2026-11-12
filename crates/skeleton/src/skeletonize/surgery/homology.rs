//! Z/2 homology classes of surface cycles and the handle subspace
//! `K = ker(H₁(S) → H₁(V))`, which surgery uses to test new loops.
//!
//! A tree-cotree decomposition gives 2g cocycles; the class of a cycle is its sum of
//! edge values, one bit per cocycle. K comes from `TetGen` once, as the surface is fixed.

use super::handle_subspace::compute_handle_subspace;
use super::tet_boundary::build_tet_boundary;
use super::tetrahedralize::tetrahedralize;
use crate::SkeletonResult;
use dualcube_types::prelude::*;
use std::collections::VecDeque;

/// Bit set over at most 128 cocycles or handle cycles.
pub(super) type Class = u128;

pub(super) struct Homology {
    /// Cocycle values of each edge, keyed by its sorted vertex indices.
    cocycles: HashMap<[usize; 2], Class>,
    /// Basis of K with distinct leading bits, highest first.
    kernel: Vec<Class>,
    /// Handle cycles through each edge (bit per basis cycle of K).
    handles: HashMap<[usize; 2], Class>,
}

impl Homology {
    pub fn new(
        mesh: &Mesh<INPUT>,
        index: &HashMap<VertID, usize>,
        genus: usize,
    ) -> SkeletonResult<Self> {
        let mut homology = Self {
            cocycles: HashMap::new(),
            kernel: Vec::new(),
            handles: HashMap::new(),
        };
        if genus == 0 {
            return Ok(homology);
        }
        if 2 * genus > Class::BITS as usize {
            return Err(format!("Genus {genus} exceeds the supported 64").into());
        }
        homology.cocycles = cocycles(mesh, index);
        let tets = tetrahedralize(mesh)?;
        if tets.n_tets() == 0 || tets.n_vertices < mesh.nr_verts() {
            return Err("TetGen volume does not match the input surface topology".into());
        }
        let subspace = compute_handle_subspace(mesh, &build_tet_boundary(&tets), genus);
        if subspace.genus() != genus {
            return Err("TetGen volume does not match the input surface topology".into());
        }
        for (i, cycle) in subspace.cycles.iter().enumerate() {
            let mut class = 0;
            for &e in cycle {
                let [a, b] = subspace.surface_edges[e as usize];
                let key = [a as usize, b as usize];
                class ^= homology.edge(key);
                *homology.handles.entry(key).or_default() |= 1 << i;
            }
            let residue = homology.reduce(class);
            if residue == 0 {
                return Err("Handle cycles are dependent in homology".into());
            }
            homology.kernel.push(residue);
            homology.kernel.sort_by(|a, b| b.cmp(a));
        }
        Ok(homology)
    }

    /// Cocycle values of the edge between two vertex indices.
    pub fn edge(&self, [a, b]: [usize; 2]) -> Class {
        self.cocycles.get(&key(a, b)).copied().unwrap_or(0)
    }

    pub fn handles(&self, [a, b]: [usize; 2]) -> Class {
        self.handles.get(&key(a, b)).copied().unwrap_or(0)
    }

    /// Whether a cycle class lies in K, i.e. the cycle bounds in V.
    pub fn bounds(&self, class: Class) -> bool {
        self.reduce(class) == 0
    }

    fn reduce(&self, mut class: Class) -> Class {
        for &b in &self.kernel {
            if class ^ b < class {
                class ^= b;
            }
        }
        class
    }
}

fn key(a: usize, b: usize) -> [usize; 2] {
    if a < b { [a, b] } else { [b, a] }
}

/// One cocycle per edge outside a primal spanning tree and a dual spanning tree of the
/// rest. Each is 1 on its edge, 0 on the tree and other leftovers, and closed on every
/// triangle, which fixes the dual tree edges from its leaves inward.
pub(super) fn cocycles(
    mesh: &Mesh<INPUT>,
    index: &HashMap<VertID, usize>,
) -> HashMap<[usize; 2], Class> {
    let edge = |e: EdgeID| key(index[&mesh.root(e)], index[&mesh.toor(e)]);
    let vertices = mesh.vert_ids();
    let mut tree = HashSet::new();
    let mut seen = HashSet::from([vertices[0]]);
    let mut queue = VecDeque::from([vertices[0]]);
    while let Some(v) = queue.pop_front() {
        for e in mesh.edges(v) {
            let w = mesh.toor(e);
            if seen.insert(w) {
                tree.insert(edge(e));
                queue.push_back(w);
            }
        }
    }
    let faces = mesh.face_ids();
    let mut parent = HashMap::new();
    let mut order = vec![faces[0]];
    let mut seen = HashSet::from([faces[0]]);
    let mut queue = VecDeque::from([faces[0]]);
    while let Some(f) = queue.pop_front() {
        for e in mesh.edges(f) {
            let g = mesh.face(mesh.twin(e));
            if !tree.contains(&edge(e)) && seen.insert(g) {
                parent.insert(g, mesh.twin(e));
                order.push(g);
                queue.push_back(g);
            }
        }
    }
    let cotree: HashSet<_> = parent.values().map(|&e| edge(e)).collect();
    let mut values = HashMap::new();
    let mut bit = 0;
    for e in mesh.edge_ids() {
        let k = edge(e);
        if !tree.contains(&k) && !cotree.contains(&k) && !values.contains_key(&k) {
            values.insert(k, 1 << bit);
            bit += 1;
        }
    }
    for &f in order.iter().skip(1).rev() {
        let up = edge(parent[&f]);
        let value = mesh
            .edges(f)
            .map(edge)
            .filter(|&k| k != up)
            .fold(0, |sum, k| sum ^ values.get(&k).copied().unwrap_or(0));
        values.insert(up, value);
    }
    values
}
