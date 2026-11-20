//! Input and combinatorial surface-region validation.

use crate::{Skeleton, SkeletonResult};
use dualcube_types::prelude::*;

impl Skeleton {
    /// Validate boundary incidence, disjoint arcs, connected regions, and genus-zero patches.
    /// The input must have passed [`input_genus`].
    pub fn validate(&self, mesh: &Mesh<INPUT>) -> SkeletonResult<()> {
        crate::surface::Regions::build(mesh, self).map(|_| ())
    }

    /// [`Self::validate`] in debug builds only, for skeletons validated by the previous stage
    /// or valid by construction.
    pub(crate) fn debug_validate(&self, mesh: &Mesh<INPUT>) {
        if cfg!(debug_assertions)
            && let Err(error) = self.validate(mesh)
        {
            panic!("Invalid skeleton: {error}");
        }
    }
}

/// Check the input before algorithms assume a closed triangular surface.
pub fn input_genus<M: Tag>(mesh: &Mesh<M>) -> SkeletonResult<usize> {
    let vertices = mesh.vert_ids();
    if vertices.is_empty() || mesh.nr_faces() == 0 {
        return Err("Skeleton extraction requires a nonempty mesh".into());
    }
    if vertices
        .iter()
        .any(|&v| !mesh.position(v).iter().all(|x| x.is_finite()))
    {
        return Err("Input positions must be finite".into());
    }
    let mut incidence = HashMap::new();
    let mut directed = HashSet::new();
    let mut links = HashMap::<VertKey<M>, HashMap<VertKey<M>, Vec<VertKey<M>>>>::new();
    for f in mesh.face_ids() {
        let v: Vec<_> = mesh.vertices(f).collect();
        if v.len() != 3 {
            return Err("Skeleton extraction requires triangles".into());
        }
        let [a, b, c] = [
            mesh.position(v[0]),
            mesh.position(v[1]),
            mesh.position(v[2]),
        ];
        if (b - a).cross(&(c - a)).norm_squared() == 0.0 {
            return Err("Input contains a degenerate triangle".into());
        }
        for i in 0..3 {
            let link = links.entry(v[i]).or_default();
            let (a, b) = (v[(i + 1) % 3], v[(i + 2) % 3]);
            link.entry(a).or_default().push(b);
            link.entry(b).or_default().push(a);
            let (a, b) = (v[i], v[(i + 1) % 3]);
            if a == b || !directed.insert((a, b)) {
                return Err("Input has repeated vertices or inconsistent face winding".into());
            }
            let mut edge = [a, b];
            edge.sort_by(|a, b| a.partial_cmp(b).unwrap());
            *incidence.entry(edge).or_insert(0usize) += 1;
        }
    }
    if incidence.values().any(|&n| n != 2) {
        return Err("Input must be a closed two-manifold".into());
    }
    // Check vertex links too: edge incidence alone does not exclude pinched vertices.
    for &v in &vertices {
        let link = links.get(&v).ok_or("Input has an isolated vertex")?;
        if link.is_empty() || link.values().any(|n| n.len() != 2) {
            return Err("Input has a non-manifold vertex link".into());
        }
        let mut seen = HashSet::new();
        let mut stack = vec![*link.keys().next().unwrap()];
        while let Some(w) = stack.pop() {
            if seen.insert(w) {
                stack.extend(&link[&w]);
            }
        }
        if seen.len() != link.len() {
            return Err("Input has a disconnected vertex link".into());
        }
    }
    let mut seen = HashSet::new();
    let mut stack = vec![vertices[0]];
    while let Some(v) = stack.pop() {
        if seen.insert(v) {
            stack.extend(mesh.neighbors(v));
        }
    }
    if seen.len() != vertices.len() {
        return Err("Input must be connected".into());
    }
    let chi = vertices.len() as i64 - incidence.len() as i64 + mesh.nr_faces() as i64;
    if chi > 2 || (2 - chi) % 2 != 0 {
        return Err("Input has invalid surface Euler characteristic".into());
    }
    let volume = volume(mesh);
    if !volume.is_finite() || volume <= 0.0 {
        return Err("Input must enclose a finite nonzero volume".into());
    }
    Ok(((2 - chi) / 2) as usize)
}

/// Signed-volume magnitude, relative to a mesh vertex to limit cancellation.
pub(crate) fn volume<M: Tag>(mesh: &Mesh<M>) -> f64 {
    let Some(origin) = mesh.vert_ids().first().map(|&v| mesh.position(v)) else {
        return 0.0;
    };
    mesh.face_ids()
        .into_iter()
        .map(|f| {
            let v: Vec<_> = mesh
                .vertices(f)
                .map(|v| mesh.position(v) - origin)
                .collect();
            (1..v.len() - 1)
                .map(|i| v[0].dot(&v[i].cross(&v[i + 1])) / 6.0)
                .sum::<f64>()
        })
        .sum::<f64>()
        .abs()
}

/// The input as the pipeline should use it: for positive genus, the surface of its TetGen volume,
/// with the boundary Steiner points TetGen added where it split input edges or faces.
/// Returns the refined mesh and the number of added vertices, or `None` if unchanged. The volume
/// is reused by every later tetrahedralization of the returned mesh in this process.
pub fn conform_input(mesh: &Mesh<INPUT>) -> SkeletonResult<Option<(Mesh<INPUT>, usize)>> {
    let genus = input_genus(mesh)?;
    if genus == 0 {
        return Ok(None);
    }
    let Some(surface) = mehsh_tetgen::conform(mesh, &mehsh_tetgen::TetgenOptions::default())?
    else {
        return Ok(None);
    };
    let faces: Vec<Vec<usize>> = surface.faces.iter().map(|f| f.to_vec()).collect();
    let refined = Mesh::from(&faces, &surface.positions)
        .map_err(|e| format!("TetGen surface is not a valid mesh: {e:?}"))?
        .0;
    if input_genus(&refined)? != genus {
        return Err("TetGen surface changed the genus".into());
    }
    Ok(Some((refined, surface.positions.len() - mesh.nr_verts())))
}
