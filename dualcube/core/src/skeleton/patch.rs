use std::collections::HashSet;

use crate::{
    prelude::{CurveSkeleton, VertID, INPUT},
    skeleton::{boundary_loop::BoundaryLoop, curve_skeleton::CurveSkeletonSpatial},
};
use log::error;
use mehsh::prelude::{HasFaces, HasPosition, HasSize, HasVertices, Mesh, Vector3D};
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;

impl CurveSkeletonSpatial for CurveSkeleton {
    /// Places a virtual vertex in the centroid of the boundaries, to close up the holes.
    /// Then uses tetrahedron volumes to compute the volume of the patch.
    fn patch_volume(&self, node_index: NodeIndex, mesh: &Mesh<INPUT>) -> f64 {
        self.patches_volume(&[node_index], mesh)
    }

    // Calculates a convex hull, then uses signed tetrahedron volumes to compute the volume of the hull.
    // Uses the same construction as patch volume to approximate the holes with disks (centroids automatically in CH).
    fn patch_hull_volume(&self, node_index: NodeIndex, mesh: &Mesh<INPUT>) -> f64 {
        self.patches_hull_volume(&[node_index], mesh)
    }

    /// Calculates a convexity score by comparing the patch volume to the convex hull volume.
    fn patch_convexity_score(&self, node_index: NodeIndex, mesh: &Mesh<INPUT>) -> f64 {
        self.patches_convexity_score(&[node_index], mesh)
    }

    /// Volume of a group of connected patches (union of their induced surface vertices).
    fn patches_volume(&self, node_indices: &[NodeIndex], mesh: &Mesh<INPUT>) -> f64 {
        let faces = patch_volume_triangles_for_nodes(self, node_indices, mesh);
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

    /// Convex-hull volume of the union of the given nodes' patches.
    fn patches_hull_volume(&self, node_indices: &[NodeIndex], mesh: &Mesh<INPUT>) -> f64 {
        // Collect the positions of the union of patch vertices
        let mut points_locations: Vec<Vec<f64>> = Vec::new();
        let mut seen_verts: HashSet<VertID> = HashSet::new();
        for &n in node_indices {
            for &v in &self[n].patch_vertices {
                if seen_verts.insert(v) {
                    let p = mesh.position(v);
                    points_locations.push(vec![p.x, p.y, p.z]);
                }
            }
        }

        // Add midpoints of external boundary edges to better approximate the hull
        let node_set: HashSet<NodeIndex> = node_indices.iter().copied().collect();
        for &n in node_indices {
            for edge_ref in self.edges(n) {
                let neighbor = if edge_ref.source() == n {
                    edge_ref.target()
                } else {
                    edge_ref.source()
                };
                // skip internal edges
                if node_set.contains(&neighbor) {
                    continue;
                }

                let boundary_loop = edge_ref.weight();
                if boundary_loop.edge_midpoints.is_empty() {
                    continue;
                }

                let loop_midpoints = loop_midpoints(mesh, boundary_loop);
                points_locations.extend(loop_midpoints.iter().map(|m| vec![m.x, m.y, m.z]));
            }
        }

        if points_locations.len() < 4 {
            // All points are coplanar or too few to form a hull with volume.
            return 0.0;
        }

        // Calculate the convex hull of the points
        let convex_hull = match chull::ConvexHullWrapper::try_new(&points_locations, None) {
            Ok(c) => c,
            Err(_) => return 0.0,
        };

        // Compute the volume of the convex hull
        convex_hull.volume()
    }

    /// Convexity score for a group of patches (union).
    fn patches_convexity_score(&self, node_indices: &[NodeIndex], mesh: &Mesh<INPUT>) -> f64 {
        let patch_volume = self.patches_volume(node_indices, mesh);
        let hull_volume = self.patches_hull_volume(node_indices, mesh);
        if hull_volume == 0.0 {
            error!("Convex hull volume is zero for nodes {:?}.", node_indices);
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

/// Build triangle faces for all mesh faces that intersect `vertices_in_patch`.
fn faces_from_vertex_set(
    vertices_in_patch: &HashSet<VertID>,
    mesh: &Mesh<INPUT>,
) -> Vec<[Vector3D; 3]> {
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
            3 => faces.push([p0, p1, p2]),
            2 => {
                // Two vertices inside, one outside.
                let (pa, pb, pc) = if !in0 {
                    (p1, p2, p0)
                } else if !in1 {
                    (p2, p0, p1)
                } else {
                    (p0, p1, p2)
                };
                let p_ac = (pa + pc) * 0.5;
                let p_bc = (pb + pc) * 0.5;
                faces.push([pa, pb, p_ac]);
                faces.push([pb, p_bc, p_ac]);
            }
            1 => {
                // One vertex inside, two outside.
                let (pa, pb, pc) = if in0 {
                    (p1, p2, p0)
                } else if in1 {
                    (p2, p0, p1)
                } else {
                    (p0, p1, p2)
                };
                let p_ac = (pa + pc) * 0.5;
                let p_bc = (pb + pc) * 0.5;
                faces.push([p_ac, p_bc, pc]);
            }
            _ => {}
        }
    }

    faces
}

/// Returns the triangles used to compute the patch volume for a group (union) of nodes.
fn patch_volume_triangles_for_nodes(
    skeleton: &CurveSkeleton,
    node_indices: &[NodeIndex],
    mesh: &Mesh<INPUT>,
) -> Vec<[Vector3D; 3]> {
    // Build union of patch vertices
    let mut vertices_in_group: HashSet<VertID> = HashSet::new();
    for &n in node_indices {
        vertices_in_group.extend(skeleton[n].patch_vertices.iter().copied());
    }
    if vertices_in_group.len() < 3 {
        return Vec::new();
    }

    // Surface portions
    let mut faces = faces_from_vertex_set(&vertices_in_group, mesh);

    // Cap only the external boundary loops
    let node_set: HashSet<NodeIndex> = node_indices.iter().copied().collect();
    for &node in node_indices {
        for edge_ref in skeleton.edges(node) {
            let neighbor = if edge_ref.source() == node {
                edge_ref.target()
            } else {
                edge_ref.source()
            };
            if node_set.contains(&neighbor) {
                // internal edge, skip capping
                continue;
            }

            let boundary_loop = edge_ref.weight();
            if boundary_loop.edge_midpoints.is_empty() {
                continue;
            }

            let loop_mids = loop_midpoints(mesh, boundary_loop);
            let mcount = loop_mids.len();

            let mut centroid = Vector3D::zeros();
            for m in &loop_mids {
                centroid += m;
            }
            centroid /= mcount as f64;

            for i in 0..mcount {
                let u = loop_mids[i];
                let v = loop_mids[(i + 1) % mcount];
                let e = boundary_loop.edge_midpoints[i];
                let root_in_patch = vertices_in_group.contains(&mesh.root(e));
                // Make sure winding order is consistent with the original topology
                if root_in_patch {
                    faces.push([v, u, centroid]); // flip
                } else {
                    faces.push([u, v, centroid]); // no flip
                }
            }
        }
    }

    faces
}
