//! Integer homology of surface loops in the enclosed volume V.
//!
//! The tetrahedralization collapses onto a graph: tets go through free triangles, then
//! triangles through free edges. The graph's non-tree edges give g integer 1-cocycles of V,
//! extended back through the collapse. A surface loop is zero in H₁(V) exactly when all g
//! cocycles vanish on it (H₁(V) has no torsion for V in ℝ³). Mod 2 this would miss loops
//! that wind twice around a handle, which flips readily produce.
//!
//! The handle cycles of K = ker(H₁(S) → H₁(V)) bias the merge order only.

use super::handle_subspace::compute_handle_subspace;
use super::tet_boundary::build_tet_boundary;
use super::tetrahedralize::tetrahedralize;
use crate::SkeletonResult;
use dualcube_types::prelude::*;

/// Integer class in H₁(V), one coordinate per cocycle, without trailing zeros.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub(super) struct Class(Vec<i64>);

impl Class {
    pub fn unit(i: usize) -> Self {
        let mut values = vec![0; i + 1];
        values[i] = 1;
        Self(values)
    }

    pub fn is_zero(&self) -> bool {
        self.0.is_empty()
    }

    fn combine(&self, other: &Class, sign: i64) -> Class {
        let mut values = self.0.clone();
        values.resize(self.0.len().max(other.0.len()), 0);
        for (a, b) in values.iter_mut().zip(&other.0) {
            *a += sign * b;
        }
        while values.last() == Some(&0) {
            values.pop();
        }
        Class(values)
    }
}

impl std::ops::Add<&Class> for &Class {
    type Output = Class;
    fn add(self, other: &Class) -> Class {
        self.combine(other, 1)
    }
}

impl std::ops::Sub<&Class> for &Class {
    type Output = Class;
    fn sub(self, other: &Class) -> Class {
        self.combine(other, -1)
    }
}

impl std::ops::Neg for &Class {
    type Output = Class;
    fn neg(self) -> Class {
        Class(self.0.iter().map(|v| -v).collect())
    }
}

/// Bit set over handle cycles.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct Bits(Vec<u64>);

impl Bits {
    pub fn bit(i: usize) -> Self {
        let mut words = vec![0; i / 64 + 1];
        words[i / 64] = 1 << (i % 64);
        Self(words)
    }

    pub fn count_ones(&self) -> u32 {
        self.0.iter().map(|w| w.count_ones()).sum()
    }
}

impl std::ops::BitOrAssign<&Bits> for Bits {
    fn bitor_assign(&mut self, other: &Bits) {
        if self.0.len() < other.0.len() {
            self.0.resize(other.0.len(), 0);
        }
        for (a, b) in self.0.iter_mut().zip(&other.0) {
            *a |= b;
        }
    }
}

pub(super) struct Homology {
    /// Cocycle values of each surface edge from its smaller to its larger vertex index.
    cocycles: HashMap<[usize; 2], Class>,
    /// Handle cycles through each edge (bit per basis cycle of K).
    handles: HashMap<[usize; 2], Bits>,
}

impl Homology {
    pub fn new(
        mesh: &Mesh<INPUT>,
        index: &HashMap<VertID, usize>,
        genus: usize,
    ) -> SkeletonResult<Self> {
        let mut homology = Self {
            cocycles: HashMap::new(),
            handles: HashMap::new(),
        };
        if genus == 0 {
            return Ok(homology);
        }
        let tets = tetrahedralize(mesh)?;
        if tets.n_tets() == 0 || tets.n_vertices < mesh.nr_verts() {
            return Err("TetGen volume does not match the input surface topology".into());
        }
        let subspace = compute_handle_subspace(mesh, &build_tet_boundary(&tets), genus);
        if subspace.genus() != genus {
            return Err("TetGen volume does not match the input surface topology".into());
        }
        for (i, cycle) in subspace.cycles.iter().enumerate() {
            for &e in cycle {
                let [a, b] = subspace.surface_edges[e as usize];
                *homology
                    .handles
                    .entry(key(a as usize, b as usize))
                    .or_default() |= &Bits::bit(i);
            }
        }
        let (rank, values) = volume_cocycles(tets.n_vertices, &tets.tets)?;
        if rank != genus {
            return Err(
                format!("The volume has {rank} independent loops, expected {genus}").into(),
            );
        }
        for e in mesh.edge_ids() {
            let k = key(index[&mesh.root(e)], index[&mesh.toor(e)]);
            if let Some(value) = values.get(&[k[0] as u32, k[1] as u32]) {
                if !value.is_zero() {
                    homology.cocycles.insert(k, value.clone());
                }
            }
        }
        Ok(homology)
    }

    /// Cocycle values along the edge from vertex index `a` to `b`.
    pub fn edge(&self, a: usize, b: usize) -> Class {
        match self.cocycles.get(&key(a, b)) {
            None => Class::default(),
            Some(value) if a < b => value.clone(),
            Some(value) => -value,
        }
    }

    pub fn handles(&self, [a, b]: [usize; 2]) -> Bits {
        self.handles.get(&key(a, b)).cloned().unwrap_or_default()
    }
}

fn key(a: usize, b: usize) -> [usize; 2] {
    if a < b { [a, b] } else { [b, a] }
}

/// Integer 1-cocycles of a tetrahedralized volume that form a basis of H¹: the number of
/// them, and their values on every edge from its smaller to its larger vertex.
pub(super) fn volume_cocycles(
    n_vertices: usize,
    tets: &[[u32; 4]],
) -> SkeletonResult<(usize, HashMap<[u32; 2], Class>)> {
    // Triangles and edges, as sorted vertex triples and pairs.
    let mut triangles: Vec<[u32; 3]> = Vec::with_capacity(tets.len() * 4);
    for t in tets {
        let mut t = *t;
        t.sort_unstable();
        let [a, b, c, d] = t;
        triangles.extend([[b, c, d], [a, c, d], [a, b, d], [a, b, c]]);
    }
    triangles.sort_unstable();
    triangles.dedup();
    let triangle = |t: [u32; 3]| triangles.binary_search(&t).unwrap();
    let mut edges: Vec<[u32; 2]> = triangles
        .iter()
        .flat_map(|&[a, b, c]| [[a, b], [b, c], [a, c]])
        .collect();
    edges.sort_unstable();
    edges.dedup();
    let edge = |e: [u32; 2]| edges.binary_search(&e).unwrap();
    // Boundary of [a, b, c]: ab + bc - ac.
    let sides: Vec<[(usize, i64); 3]> = triangles
        .iter()
        .map(|&[a, b, c]| [(edge([a, b]), 1), (edge([b, c]), 1), (edge([a, c]), -1)])
        .collect();

    // Tets through free triangles.
    let mut cofaces: Vec<Vec<usize>> = vec![Vec::new(); triangles.len()];
    let faces: Vec<[usize; 4]> = tets
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let mut t = *t;
            t.sort_unstable();
            let [a, b, c, d] = t;
            let f = [[b, c, d], [a, c, d], [a, b, d], [a, b, c]].map(triangle);
            for &f in &f {
                cofaces[f].push(i);
            }
            f
        })
        .collect();
    let mut tet_alive = vec![true; tets.len()];
    let mut triangle_alive = vec![true; triangles.len()];
    let mut count: Vec<usize> = cofaces.iter().map(Vec::len).collect();
    let mut queue: Vec<usize> = (0..triangles.len()).filter(|&f| count[f] == 1).collect();
    let mut removed_tets = 0;
    while let Some(f) = queue.pop() {
        if !triangle_alive[f] || count[f] != 1 {
            continue;
        }
        let t = cofaces[f].iter().copied().find(|&t| tet_alive[t]).unwrap();
        tet_alive[t] = false;
        triangle_alive[f] = false;
        removed_tets += 1;
        for &g in &faces[t] {
            count[g] -= 1;
            if count[g] == 1 && triangle_alive[g] {
                queue.push(g);
            }
        }
    }
    if removed_tets != tets.len() {
        return Err("The tetrahedralized volume does not collapse".into());
    }

    // A spanning tree of the edges is zero: cocycles are only defined up to coboundaries.
    let mut parent: Vec<u32> = (0..n_vertices as u32).collect();
    fn root(parent: &mut [u32], mut i: u32) -> u32 {
        while parent[i as usize] != i {
            parent[i as usize] = parent[parent[i as usize] as usize];
            i = parent[i as usize];
        }
        i
    }
    let mut known = vec![false; edges.len()];
    for (e, &[a, b]) in edges.iter().enumerate() {
        let [a, b] = [a, b].map(|v| root(&mut parent, v));
        if a != b {
            parent[a as usize] = b;
            known[e] = true;
        }
    }

    // Values as combinations of free generators. A remaining triangle with one unknown edge
    // fixes it; when none has, an unknown edge becomes a generator. Triangles left with no
    // unknown edge that fixed nothing constrain the generators.
    let mut cofaces: Vec<Vec<usize>> = vec![Vec::new(); edges.len()];
    let alive: Vec<usize> = (0..triangles.len())
        .filter(|&f| triangle_alive[f])
        .collect();
    for &f in &alive {
        for (e, _) in sides[f] {
            cofaces[e].push(f);
        }
    }
    let mut unknown: Vec<u8> = vec![0; triangles.len()];
    for &f in &alive {
        unknown[f] = sides[f].iter().filter(|(e, _)| !known[*e]).count() as u8;
    }
    let mut values = vec![Class::default(); edges.len()];
    let mut queue: Vec<usize> = alive.iter().copied().filter(|&f| unknown[f] == 1).collect();
    let mut constraints = Vec::new();
    let mut generators = 0;
    let mut next_unknown = 0;
    loop {
        while let Some(f) = queue.pop() {
            if unknown[f] != 1 {
                continue;
            }
            let (e, sign) = sides[f].into_iter().find(|(e, _)| !known[*e]).unwrap();
            let mut sum = Class::default();
            for (g, s) in sides[f] {
                if g != e {
                    sum = if s > 0 {
                        &sum + &values[g]
                    } else {
                        &sum - &values[g]
                    };
                }
            }
            values[e] = if sign > 0 { -&sum } else { sum };
            known[e] = true;
            unknown[f] = u8::MAX; // used
            for &g in &cofaces[e] {
                if g == f || unknown[g] == u8::MAX {
                    continue;
                }
                unknown[g] -= 1;
                match unknown[g] {
                    1 => queue.push(g),
                    0 => constraints.push(g),
                    _ => {}
                }
            }
        }
        while next_unknown < edges.len() && known[next_unknown] {
            next_unknown += 1;
        }
        if next_unknown == edges.len() {
            break;
        }
        let e = next_unknown;
        values[e] = Class::unit(generators);
        generators += 1;
        known[e] = true;
        for &g in &cofaces[e] {
            if unknown[g] == u8::MAX {
                continue;
            }
            unknown[g] -= 1;
            match unknown[g] {
                1 => queue.push(g),
                0 => constraints.push(g),
                _ => {}
            }
        }
    }
    let rows: Vec<Vec<i64>> = constraints
        .iter()
        .map(|&f| {
            let mut row = Class::default();
            for (e, s) in sides[f] {
                row = if s > 0 {
                    &row + &values[e]
                } else {
                    &row - &values[e]
                };
            }
            let mut row = row.0;
            row.resize(generators, 0);
            row
        })
        .filter(|row| row.iter().any(|&v| v != 0))
        .collect();
    tracing::info!(
        "Volume cocycles: {generators} generators, {} constraints",
        rows.len()
    );
    let kernel = integer_kernel(rows, generators)?;
    let rank = kernel.len();
    let values = values
        .into_iter()
        .map(|value| {
            let mut class = Class(
                kernel
                    .iter()
                    .map(|k| value.0.iter().zip(k).map(|(a, b)| a * b).sum())
                    .collect(),
            );
            while class.0.last() == Some(&0) {
                class.0.pop();
            }
            class
        })
        .collect::<Vec<_>>();
    Ok((rank, edges.into_iter().zip(values).collect()))
}

/// A basis of the rational kernel of an integer matrix with `n` columns, scaled to integers.
fn integer_kernel(mut rows: Vec<Vec<i64>>, n: usize) -> SkeletonResult<Vec<Vec<i64>>> {
    fn gcd(a: i128, b: i128) -> i128 {
        if b == 0 { a.abs() } else { gcd(b, a % b) }
    }
    let mut rows: Vec<Vec<i128>> = rows
        .drain(..)
        .map(|r| r.into_iter().map(i128::from).collect())
        .collect();
    let normalize = |row: &mut Vec<i128>| {
        let g = row.iter().fold(0, |g, &v| gcd(g, v));
        if g > 1 {
            row.iter_mut().for_each(|v| *v /= g);
        }
    };
    let mut pivots: Vec<(usize, usize)> = Vec::new(); // (row, column)
    let mut r = 0;
    for c in 0..n {
        let Some(p) = (r..rows.len()).find(|&i| rows[i][c] != 0) else {
            continue;
        };
        rows.swap(r, p);
        let pivot = rows[r].clone();
        for i in 0..rows.len() {
            if i != r && rows[i][c] != 0 {
                let (a, b) = (pivot[c], rows[i][c]);
                for k in 0..n {
                    rows[i][k] = rows[i][k] * a - pivot[k] * b;
                }
                normalize(&mut rows[i]);
                if rows[i].iter().any(|v| v.unsigned_abs() > 1 << 100) {
                    return Err("Volume cocycle coefficients overflow".into());
                }
            }
        }
        pivots.push((r, c));
        r += 1;
    }
    let pivot_columns: HashSet<usize> = pivots.iter().map(|p| p.1).collect();
    let mut kernel = Vec::new();
    for free in (0..n).filter(|c| !pivot_columns.contains(c)) {
        let scale = pivots.iter().fold(1i128, |l, &(row, c)| {
            l / gcd(l, rows[row][c]) * rows[row][c].abs()
        });
        let mut x = vec![0i128; n];
        x[free] = scale;
        for &(row, c) in &pivots {
            x[c] = -scale / rows[row][c] * rows[row][free];
        }
        let g = x.iter().fold(0, |g, &v| gcd(g, v));
        kernel.push(
            x.into_iter()
                .map(|v| i64::try_from(v / g))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| "Volume cocycle coefficients overflow")?,
        );
    }
    Ok(kernel)
}
