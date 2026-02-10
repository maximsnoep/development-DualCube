use std::collections::HashSet;

use crate::prelude::{INPUT, VertID};
use mehsh::prelude::{HasFaces, HasPosition, HasSize, HasVertices, Mesh, Vector3D};

/// Computes the centroid position of a set of mesh vertices.
/// Weighs each vertex by the one-ring area inside the patch to approximate true surface centroid.
/// Note that this throws away area from shared faces... (TODO: figure out simple construction to do weigh these properly)
pub fn patch_centroid(vertices: &[VertID], mesh: &Mesh<INPUT>) -> Vector3D {
    if vertices.is_empty() {
        return Vector3D::zeros();
    }

    let vert_set: HashSet<VertID> = vertices.iter().copied().collect();
    let mut weighted_sum = Vector3D::zeros();
    let mut total_area = 0.0;

    for &v in vertices {
        // compute area of faces adjacent to v, weighted by how much of each face is in the patch
        let mut area_v = 0.0;
        for face in mesh.faces(v) {
            let face_verts: Vec<_> = mesh.vertices(face).collect();
            let count_in = face_verts
                .iter()
                .filter(|&&fv| vert_set.contains(&fv))
                .count();
            if count_in == 0 {
                continue;
            }
            let face_area = mesh.size(face);
            area_v += face_area * (count_in as f64) / (face_verts.len() as f64);
        }

        weighted_sum += mesh.position(v) * area_v;
        total_area += area_v;
    }

    weighted_sum / total_area
}
