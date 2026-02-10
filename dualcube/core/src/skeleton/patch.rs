use std::collections::HashSet;

use crate::{prelude::{CurveSkeleton, INPUT, VertID}, skeleton::curve_skeleton::CurveSkeletonSpatial};
use log::error;
use mehsh::prelude::{HasFaces, HasPosition, HasSize, HasVertices, Mesh, Vector3D};
use petgraph::graph::NodeIndex;


impl CurveSkeletonSpatial for CurveSkeleton {
    /// Places a virtual vertex in the centroid of the boundaries, to close up the holes. 
    /// Then uses tetrahedron volumes to compute the volume of the patch.
    fn patch_volume(&self, node_index: NodeIndex, mesh: &Mesh<INPUT>) -> f64 {
        let patch_verts = &self[node_index].patch_vertices;
        if patch_verts.len() < 3 {
            return 0.0;
        }

        let vert_set: HashSet<VertID> = patch_verts.iter().copied().collect();

        // Collect all faces that are fully inside the patch
        let mut faces: Vec<[Vector3D; 3]> = Vec::new();
        for face_id in mesh.face_ids() {
            let fv: Vec<VertID> = mesh.vertices(face_id).collect();
            if fv.iter().all(|v| vert_set.contains(v)) {
                let p0 = mesh.position(fv[0]);
                let p1 = mesh.position(fv[1]);
                let p2 = mesh.position(fv[2]);
                faces.push([p0, p1, p2]);
            }
        }

        // For each boundary loop adjacent to this node, create a centroid and cap the hole
        for edge_ref in self.edges(node_index) {
            let boundary_loop = edge_ref.weight();
            if boundary_loop.vertices.is_empty() {
                continue;
            }
            let loop_verts: Vec<VertID> = boundary_loop.vertices.iter().map(|&e| mesh.root(e)).collect();

            // Centroid of boundary vertices
            let mut centroid = Vector3D::zeros();
            for &v in &loop_verts {
                centroid += mesh.position(v);
            }
            centroid /= loop_verts.len() as f64;

            // Cap the loop by creating triangles (u, v, centroid) following the loop order
            for i in 0..loop_verts.len() {
                let u = mesh.position(loop_verts[i]);
                let v = mesh.position(loop_verts[(i + 1) % loop_verts.len()]);
                faces.push([u, v, centroid]);
            }
        }

        // Sum signed tetrahedron volumes from the origin for all triangles
        let mut vol= 0.0;
        for tri in faces {
            let p0 = tri[0];
            let p1 = tri[1];
            let p2 = tri[2];
            vol += p0.dot(&p1.cross(&p2));
        }

        (vol / 6.0).abs()
    }

    // Calculates a convex hull, then uses signed tetrahedron volumes to compute the volume of the hull.
    fn patch_hull_volume(&self, node_index: NodeIndex, mesh: &Mesh<INPUT>) -> f64 {
        // Collect the positions of the patch vertices
        let vertices = &self[node_index].patch_vertices;
        if vertices.len() < 4 {
            // All points are coplanar, so volume is always zero.
            return 0.0;
        }

        // Convert to Vec<Vec<f64>> convex hull calculation
        let points_f: Vec<Vec<f64>> = vertices
            .iter()
            .map(|&v| {
                let p = mesh.position(v);
                vec![p.x, p.y, p.z]
            })
            .collect();
        let convex_hull = match chull::ConvexHullWrapper::try_new(&points_f, None) {
            Ok(c) => c,
            Err(_) => return 0.0,
        };

        // Compute the volume of the convex hull // TODO: check whether this calculation lines up with other one we use (it should though).
        convex_hull.volume()
    }
    
    /// Calculates a convexity score by comparing the patch volume to the convex hull volume. 
    fn patch_convexity_score(&self, node_index: NodeIndex, mesh: &Mesh<INPUT>) -> f64 {
        let patch_volume = self.patch_volume(node_index, mesh);
        let hull_volume = self.patch_hull_volume(node_index, mesh);
        print!("Patch volume: {}, Hull volume: {}", patch_volume, hull_volume);
        if hull_volume == 0.0 {
            // Something likely went wrong
            error!("Convex hull volume is zero for node {:?}.", node_index);
            return 0.0;
        }
        return (patch_volume / hull_volume).min(1.0);
    }
}

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
