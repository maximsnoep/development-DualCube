//! Meshes read from files: checked for what the algorithms need (a single closed, oriented, manifold surface with
//! finite coordinates), with errors that say what is wrong and where.

use crate::prelude::*;
use crate::utils::ids::IdMap;
use std::collections::{HashMap, HashSet};

// At most this many problems are listed in an error.
const LISTED: usize = 3;

impl<M: Tag> Mesh<M>
where
    M: Default + Eq + std::hash::Hash + Copy + Clone,
{
    /// A mesh from the vertices and faces of a file (see `from`), checked: every coordinate is a finite number, every
    /// edge borders exactly two faces that use it in opposite directions (a closed, consistently oriented, manifold
    /// surface), and the surface is connected.
    pub fn from_input(
        faces: &[Vec<usize>],
        verts: &[Vector3D],
    ) -> Result<(Self, IdMap<VERT, M>, IdMap<FACE, M>), MeshError<M>> {
        let at = |v: usize| {
            verts.get(v).map_or_else(
                || "?".to_owned(),
                |p| format!("({:.4}, {:.4}, {:.4})", p.x, p.y, p.z),
            )
        };
        let bad: Vec<usize> = (0..verts.len())
            .filter(|&v| verts[v].iter().any(|c| !c.is_finite()))
            .collect();
        if !bad.is_empty() {
            return Err(MeshError::Invalid(format!(
                "{} vertices have coordinates that are not finite numbers (NaN or infinity)",
                bad.len()
            )));
        }
        if let Some((f, &v)) = faces
            .iter()
            .enumerate()
            .find_map(|(f, face)| face.iter().find(|&&v| v >= verts.len()).map(|v| (f, v)))
        {
            return Err(MeshError::Invalid(format!(
                "face {} refers to vertex {}, but there are only {} vertices",
                f + 1,
                v + 1,
                verts.len()
            )));
        }

        // Every directed edge once, and every edge in both directions.
        let mut directed: HashMap<(usize, usize), usize> = HashMap::new();
        for face in faces {
            for i in 0..face.len() {
                let (a, b) = (face[i], face[(i + 1) % face.len()]);
                if a != b {
                    *directed.entry((a, b)).or_default() += 1;
                }
            }
        }
        let mut twice: Vec<(usize, usize)> = directed
            .iter()
            .filter(|&(_, &n)| n > 1)
            .map(|(&e, _)| e)
            .collect();
        let mut open: Vec<(usize, usize)> = directed
            .keys()
            .filter(|&&(a, b)| !directed.contains_key(&(b, a)))
            .copied()
            .collect();
        twice.sort_unstable();
        open.sort_unstable();
        let list = |edges: &[(usize, usize)]| {
            edges
                .iter()
                .take(LISTED)
                .map(|&(a, b)| format!("{} - {}", at(a), at(b)))
                .collect::<Vec<_>>()
                .join(", ")
        };
        if !twice.is_empty() {
            return Err(MeshError::Invalid(format!(
                "{} edges are used twice in the same direction: the faces are not consistently oriented (some are \
                 flipped), or more than two faces meet at an edge (the surface is not manifold). E.g., at {}",
                twice.len(),
                list(&twice)
            )));
        }
        if !open.is_empty() {
            return Err(MeshError::Invalid(format!(
                "the surface is not closed: {} edges border only one face (holes, or a boundary). E.g., at {}",
                open.len(),
                list(&open)
            )));
        }

        let (mesh, vert_map, face_map) = Self::from(faces, verts)?;
        let parts = mesh.count_parts();
        if parts > 1 {
            return Err(MeshError::NotConnected(parts));
        }
        // Zero-area faces (e.g., with coincident corners) are allowed, but their geometry is undefined: the paths of a
        // layout cannot be smoothed on them.
        let degenerate = mesh
            .face_ids()
            .into_iter()
            .filter(|&face| !(mesh.size(face) > 0.))
            .count();
        if degenerate > 0 {
            warn!(
                "{degenerate} faces have zero area (e.g., coincident corners): smoothing the paths will not work on this mesh"
            );
        }
        Ok((mesh, vert_map, face_map))
    }

    // The number of connected components (by faces sharing an edge).
    fn count_parts(&self) -> usize {
        let mut seen = HashSet::new();
        let mut parts = 0;
        for start in self.face_ids() {
            if !seen.insert(start) {
                continue;
            }
            parts += 1;
            let mut stack = vec![start];
            while let Some(face) = stack.pop() {
                for edge in self.edges(face) {
                    let next = self.face(self.twin(edge));
                    if seen.insert(next) {
                        stack.push(next);
                    }
                }
            }
        }
        parts
    }
}
