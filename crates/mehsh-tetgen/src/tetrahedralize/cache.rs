//! Volumes from [`super::conform`], found again by the positions of their boundary vertices.
//! Every hit is checked like fresh TetGen output, so a stale or colliding entry is only a miss.

use super::{TetMeshData, TetrahedralMesh, map_output};
use crate::TetNode;
use mehsh::prelude::*;
use std::{
    hash::{BuildHasher, Hash, Hasher},
    sync::{Arc, Mutex},
};

/// Volumes kept, most recent last.
const CAPACITY: usize = 4;

struct Entry {
    key: u64,
    vertices: Vec<Vector3D>,
    tets: Vec<[usize; 4]>,
}

static CACHE: Mutex<Vec<Arc<Entry>>> = Mutex::new(Vec::new());

fn bits(p: Vector3D) -> [u64; 3] {
    [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()]
}

/// Independent of vertex order.
fn key(positions: impl Iterator<Item = Vector3D>) -> u64 {
    let state = std::hash::BuildHasherDefault::<std::hash::DefaultHasher>::default();
    let mut count = 0u64;
    let mut sum = 0u64;
    for p in positions {
        let mut hasher = state.build_hasher();
        bits(p).hash(&mut hasher);
        sum = sum.wrapping_add(hasher.finish());
        count += 1;
    }
    sum ^ count.rotate_left(32)
}

pub(super) fn store(vertices: Vec<Vector3D>, tets: Vec<[usize; 4]>) {
    let mut count = HashMap::<[usize; 3], usize>::new();
    for t in &tets {
        for [i, j, k] in [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]] {
            let mut face = [t[i], t[j], t[k]];
            face.sort();
            *count.entry(face).or_default() += 1;
        }
    }
    let boundary: HashSet<usize> = count
        .into_iter()
        .filter(|&(_, n)| n == 1)
        .flat_map(|(face, _)| face)
        .collect();
    let key = key(boundary.into_iter().map(|v| vertices[v]));
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    cache.retain(|e| e.key != key);
    if cache.len() == CAPACITY {
        cache.remove(0);
    }
    cache.push(Arc::new(Entry {
        key,
        vertices,
        tets,
    }));
}

/// A remembered volume whose boundary is exactly `mesh`.
pub(super) fn lookup<M: Tag>(mesh: &Mesh<M>) -> Option<TetrahedralMesh<M>> {
    let input = mesh.vert_ids();
    let key = key(input.iter().map(|&v| mesh.position(v)));
    let entry = {
        let cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
        cache.iter().rev().find(|e| e.key == key)?.clone()
    };
    let at: HashMap<[u64; 3], usize> = entry
        .vertices
        .iter()
        .enumerate()
        .map(|(i, &p)| (bits(p), i))
        .collect();
    let mut nodes: Vec<_> = entry
        .vertices
        .iter()
        .map(|&position| TetNode {
            position,
            marker: 0,
        })
        .collect();
    for (i, &v) in input.iter().enumerate() {
        let node = &mut nodes[*at.get(&bits(mesh.position(v)))?];
        if node.marker != 0 {
            return None;
        }
        node.marker = -i32::try_from(i).ok()? - 2;
    }
    let data = TetMeshData {
        nodes,
        tets: entry.tets.clone(),
    };
    let volume = map_output(mesh, input, data).ok()?;
    // `map_output` may move Steiner points inside; a hit must already have the boundary.
    (volume.vertices.len() == entry.vertices.len() && volume.tets.len() == entry.tets.len())
        .then_some(volume)
}
