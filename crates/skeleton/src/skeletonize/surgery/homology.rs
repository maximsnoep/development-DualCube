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

/// Bit set over cocycles or handle cycles, without trailing zero words; ordered by its
/// highest bit.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct Class(Vec<u64>);

impl Class {
    pub fn bit(i: usize) -> Self {
        let mut words = vec![0; i / 64 + 1];
        words[i / 64] = 1 << (i % 64);
        Self(words)
    }

    pub fn count_ones(&self) -> u32 {
        self.0.iter().map(|w| w.count_ones()).sum()
    }

    pub fn is_zero(&self) -> bool {
        self.0.is_empty()
    }
}

impl std::ops::BitXorAssign<&Class> for Class {
    fn bitxor_assign(&mut self, other: &Class) {
        if self.0.len() < other.0.len() {
            self.0.resize(other.0.len(), 0);
        }
        for (a, b) in self.0.iter_mut().zip(&other.0) {
            *a ^= b;
        }
        while self.0.last() == Some(&0) {
            self.0.pop();
        }
    }
}

impl std::ops::BitXor<&Class> for &Class {
    type Output = Class;
    fn bitxor(self, other: &Class) -> Class {
        let mut result = self.clone();
        result ^= other;
        result
    }
}

impl std::ops::BitOrAssign<&Class> for Class {
    fn bitor_assign(&mut self, other: &Class) {
        if self.0.len() < other.0.len() {
            self.0.resize(other.0.len(), 0);
        }
        for (a, b) in self.0.iter_mut().zip(&other.0) {
            *a |= b;
        }
    }
}

impl Ord for Class {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .len()
            .cmp(&other.0.len())
            .then_with(|| self.0.iter().rev().cmp(other.0.iter().rev()))
    }
}

impl PartialOrd for Class {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

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
            let mut class = Class::default();
            for &e in cycle {
                let [a, b] = subspace.surface_edges[e as usize];
                let key = [a as usize, b as usize];
                class ^= homology.edge(key);
                *homology.handles.entry(key).or_default() |= &Class::bit(i);
            }
            let residue = homology.reduce(class);
            if residue.is_zero() {
                return Err("Handle cycles are dependent in homology".into());
            }
            homology.kernel.push(residue);
            homology.kernel.sort_by(|a, b| b.cmp(a));
        }
        Ok(homology)
    }

    /// Cocycle values of the edge between two vertex indices.
    pub fn edge(&self, [a, b]: [usize; 2]) -> &Class {
        static ZERO: Class = Class(Vec::new());
        self.cocycles.get(&key(a, b)).unwrap_or(&ZERO)
    }

    pub fn handles(&self, [a, b]: [usize; 2]) -> Class {
        self.handles.get(&key(a, b)).cloned().unwrap_or_default()
    }

    /// Whether a cycle class lies in K, i.e. the cycle bounds in V.
    pub fn bounds(&self, class: Class) -> bool {
        self.reduce(class).is_zero()
    }

    fn reduce(&self, mut class: Class) -> Class {
        for b in &self.kernel {
            let reduced = &class ^ b;
            if reduced < class {
                class = reduced;
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
            values.insert(k, Class::bit(bit));
            bit += 1;
        }
    }
    for &f in order.iter().skip(1).rev() {
        let up = edge(parent[&f]);
        let value =
            mesh.edges(f)
                .map(edge)
                .filter(|&k| k != up)
                .fold(Class::default(), |mut sum, k| {
                    if let Some(v) = values.get(&k) {
                        sum ^= v;
                    }
                    sum
                });
        values.insert(up, value);
    }
    values
}
