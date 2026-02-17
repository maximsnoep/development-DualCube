use std::collections::HashSet;

use crate::{
    prelude::{CurveSkeleton, VertID, INPUT},
    skeleton::{boundary_loop::BoundaryLoop, curve_skeleton::CurveSkeletonSpatial},
};
use log::error;
use mehsh::prelude::{HasFaces, HasPosition, HasSize, HasVertices, Mesh, Vector3D};
use petgraph::graph::NodeIndex;

impl CurveSkeletonSpatial for CurveSkeleton {
    /// Places a virtual vertex in the centroid of the boundaries, to close up the holes.
    /// Then uses tetrahedron volumes to compute the volume of the patch.
    fn patch_volume(&self, node_index: NodeIndex, mesh: &Mesh<INPUT>) -> f64 {
        let faces = patch_volume_triangles(&self, node_index, mesh);
        if faces.is_empty() {
            return 0.0;
        }

        // Sum signed tetrahedron volumes from the origin for all triangles
        let mut vol = 0.0;
        for tri in faces {
            let p0 = tri[0];
            let p1 = tri[1];
            let p2 = tri[2];
            vol += p0.dot(&p1.cross(&p2));
        }

        (vol / 6.0).abs()
    }

    // Calculates a convex hull, then uses signed tetrahedron volumes to compute the volume of the hull.
    // Uses the same construction as patch volume to approximate the holes with disks (centroids automatically in CH).
    fn patch_hull_volume(&self, node_index: NodeIndex, mesh: &Mesh<INPUT>) -> f64 {
        // Collect the positions of the patch vertices
        let vertices = &self[node_index].patch_vertices;
        if vertices.len() < 4 {
            // All points are coplanar, so volume is always zero.
            return 0.0;
        }

        // Convert to Vec<Vec<f64>> convex hull calculation
        let mut points_locations: Vec<Vec<f64>> = vertices
            .iter()
            .map(|&v| {
                let p = mesh.position(v);
                vec![p.x, p.y, p.z]
            })
            .collect();
        // Add midpoints of boundary edges to better approximate the hull of the patch
        for edge_ref in self.edges(node_index) {
            let boundary_loop = edge_ref.weight();
            if boundary_loop.edge_midpoints.is_empty() {
                continue;
            }

            // Add midpoints of each boundary edge so CH never underestimates
            let loop_midpoints = loop_midpoints(mesh, boundary_loop);
            points_locations.extend(loop_midpoints.iter().map(|m| vec![m.x, m.y, m.z]));
        }

        // Calculate the convex hull of the points
        let convex_hull = match chull::ConvexHullWrapper::try_new(&points_locations, None) {
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
        print!(
            "Patch volume: {}, Hull volume: {}",
            patch_volume, hull_volume
        );
        if hull_volume == 0.0 {
            // Something likely went wrong
            error!("Convex hull volume is zero for node {:?}.", node_index);
            return 0.0;
        }
        return patch_volume / hull_volume;
    }
}

// Calculates the midpoints of each edge along the boundary loop.
fn loop_midpoints(mesh: &Mesh<INPUT>, boundary_loop: &BoundaryLoop) -> Vec<Vector3D> {
    let loop_midpoints: Vec<Vector3D> = boundary_loop
        .edge_midpoints
        .iter()
        .map(|&e| {
            let a = mesh.position(mesh.root(e));
            let b = mesh.position(mesh.toor(e));
            (a + b) * 0.5
        })
        .collect();
    loop_midpoints
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

/// Returns the triangles used to compute the patch volume (surface triangles + boundary cap fans).
fn patch_volume_triangles(
    skeleton: &CurveSkeleton,
    node_index: NodeIndex,
    mesh: &Mesh<INPUT>,
) -> Vec<[Vector3D; 3]> {
    let patch_verts = &skeleton[node_index].patch_vertices;
    if patch_verts.len() < 3 {
        return Vec::new();
    }

    let vertices_in_patch: HashSet<VertID> = patch_verts.iter().copied().collect();

    // For each mesh face, add the portion that lies inside this patch.
    let mut faces: Vec<[Vector3D; 3]> = Vec::new();
    for face_id in mesh.face_ids() {
        let fv: Vec<VertID> = mesh.vertices(face_id).collect();
        if fv.len() != 3 {
            continue;
        }

        let in0 = vertices_in_patch.contains(&fv[0]);
        let in1 = vertices_in_patch.contains(&fv[1]);
        let in2 = vertices_in_patch.contains(&fv[2]);
        let count = in0 as usize + in1 as usize + in2 as usize;

        let p0 = mesh.position(fv[0]);
        let p1 = mesh.position(fv[1]);
        let p2 = mesh.position(fv[2]);

        match count {
            3 => {
                // Fully inside the patch
                faces.push([p0, p1, p2]);
            }
            2 => {
                // Two vertices inside, one outside.
                // Rotate into (a, b, c) with a,b = inside (majority), c = outside (minority),
                // using a cyclic rotation to preserve the original face winding order.
                let (pa, pb, pc) = if !in0 {
                    (p1, p2, p0)
                } else if !in1 {
                    (p2, p0, p1)
                } else {
                    (p0, p1, p2)
                };
                // Majority (inside) portion from split_triangle: (a, b, ac) and (b, bc, ac)
                let p_ac = (pa + pc) * 0.5;
                let p_bc = (pb + pc) * 0.5;
                faces.push([pa, pb, p_ac]);
                faces.push([pb, p_bc, p_ac]);
            }
            1 => {
                // One vertex inside, two outside.
                // Rotate into (a, b, c) with a,b = outside (majority), c = inside (minority),
                // using a cyclic rotation to preserve the original face winding order.
                let (pa, pb, pc) = if in0 {
                    (p1, p2, p0)
                } else if in1 {
                    (p2, p0, p1)
                } else {
                    (p0, p1, p2)
                };
                // Minority (inside) portion from split_triangle: (ac, bc, c)
                let p_ac = (pa + pc) * 0.5;
                let p_bc = (pb + pc) * 0.5;
                faces.push([p_ac, p_bc, pc]);
            }
            _ => {} // No vertices inside this patch
        }
    }

    // Cap each boundary loop with a fan of triangles through the centroid of its midpoints.
    for edge_ref in skeleton.edges(node_index) {
        let boundary_loop = edge_ref.weight();
        if boundary_loop.edge_midpoints.is_empty() {
            continue;
        }

        let loop_mids = loop_midpoints(mesh, boundary_loop);
        let n = loop_mids.len();

        let mut centroid = Vector3D::zeros();
        for m in &loop_mids {
            centroid += m;
        }
        centroid /= n as f64;

        // Check the first boundary half-edge's face to determine winding.
        let first_edge = boundary_loop.edge_midpoints[0];
        let face_verts: Vec<VertID> = mesh.vertices(mesh.face(first_edge)).collect();
        let in_count = face_verts
            .iter()
            .filter(|&&v| vertices_in_patch.contains(&v))
            .count();
        let flip = in_count >= 2;

        for i in 0..n {
            let u = loop_mids[i];
            let v = loop_mids[(i + 1) % n];
            if flip {
                faces.push([v, u, centroid]);
            } else {
                faces.push([u, v, centroid]);
            }
        }
    }

    faces
}
