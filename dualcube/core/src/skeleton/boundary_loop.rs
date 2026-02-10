use std::collections::HashSet;

use mehsh::prelude::{HasVertices, Mesh};
use serde::{Deserialize, Serialize};

use crate::prelude::{EdgeID, VertID, INPUT};

/// Edges of the graph geometrically represent boundary loops between patches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryLoop {
    /// The list of surface edges that form this boundary loop, in traversal order.
    /// This is always a simple cycle.
    pub vertices: Vec<EdgeID>,
}

impl BoundaryLoop {
    /// Computes the ordered boundary loop between two adjacent node patches using a face-walk.
    pub fn new(patch_a: &[VertID], patch_b: &[VertID], mesh: &Mesh<INPUT>) -> BoundaryLoop {
        let set_a: HashSet<_> = patch_a.iter().copied().collect();
        let set_b: HashSet<_> = patch_b.iter().copied().collect();

        // Collect faces whose vertices are all in union(A,B) and contain both labels.
        let mut boundary_faces = Vec::new();
        for face_id in mesh.face_ids() {
            let verts: Vec<VertID> = mesh.vertices(face_id).collect();
            if verts.len() != 3 {
                panic!("Non-triangular face found while computing boundary loop.");
            }

            let mut has_a = false;
            let mut has_b = false;
            let mut has_other = false;
            for &v in &verts {
                if set_a.contains(&v) {
                    has_a = true;
                } else if set_b.contains(&v) {
                    has_b = true;
                } else {
                    has_other = true;
                    break;
                }
            }
            if has_other {
                // Proven that any relevant face (on the boundary) has exactly A and B, but no others.
                continue;
            }
            if has_a && has_b {
                boundary_faces.push(face_id);
            }
        }

        if boundary_faces.is_empty() {
            unreachable!("No boundary faces found between patches.");
        }

        // Start from the first boundary face
        let start_face = boundary_faces[0];

        // find minority vertex in start_face
        let sv: Vec<_> = mesh.vertices(start_face).collect();
        let a_inds: Vec<_> = sv.iter().copied().filter(|&v| set_a.contains(&v)).collect();
        let b_inds: Vec<_> = sv
            .iter()
            .copied()
            .filter(|&v| !set_a.contains(&v))
            .collect();

        // Face with vertices from both sets must have exactly 1 vertex from one set (minority) and 2 from the other (majority)
        let (minority, majority0, _majority1) = if a_inds.len() == 1 {
            (a_inds[0], b_inds[0], b_inds[1])
        } else {
            (b_inds[0], a_inds[0], a_inds[1])
        };

        // Pick starting oriented edge on start_face
        let (edge_min_maj0, edge_maj0_min) = mesh
            .edge_between_verts(minority, majority0)
            .expect("Expected boundary edge to exist");
        let start_oriented = if mesh.face(edge_min_maj0) == start_face {
            edge_min_maj0
        } else if mesh.face(edge_maj0_min) == start_face {
            edge_maj0_min
        } else {
            unreachable!("Boundary edge not found on start face");
        };

        // Traverse face-to-face along boundary edges until we return to start edge
        let mut loop_edges: Vec<EdgeID> = Vec::new();
        let mut current_oriented = start_oriented;
        let mut current_face = start_face;
        let max_iters = mesh.face_ids().len() * 3 + 10;
        let mut iters = 0;

        loop {
            // Append the oriented edge
            loop_edges.push(current_oriented);

            // Get endpoints of the current oriented edge
            let u = mesh.root(current_oriented);
            let v = mesh.toor(current_oriented);

            // Get vertices of current face
            let fverts: Vec<_> = mesh.vertices(current_face).collect();
            assert!(fverts.len() == 3);

            // Identify minority and the other majority vertex different from the one we came from and
            // find the minority vertex in this face
            let a_vs: Vec<_> = fverts
                .iter()
                .copied()
                .filter(|&x| set_a.contains(&x))
                .collect();
            let b_vs: Vec<_> = fverts
                .iter()
                .copied()
                .filter(|&x| set_b.contains(&x))
                .collect();
            let (min, maj_a, maj_b) = if a_vs.len() == 1 {
                (a_vs[0], b_vs[0], b_vs[1])
            } else {
                (b_vs[0], a_vs[0], a_vs[1])
            };

            // Determine the other AB edge in this face (the one not equal to current edge)
            let other_major = if (u == min && v == maj_a) || (u == maj_a && v == min) {
                maj_b
            } else {
                maj_a
            };

            let (edge_min_maj, edge_maj_min) = mesh
                .edge_between_verts(min, other_major)
                .expect("Boundary edge missing while walking");

            // Identify which of edge_min_maj/edge_maj_min is on current_face; the other will be oriented for adjacent face.
            let (_edge_on_face, twin) = if mesh.face(edge_min_maj) == current_face {
                (edge_min_maj, edge_maj_min)
            } else if mesh.face(edge_maj_min) == current_face {
                (edge_maj_min, edge_min_maj)
            } else {
                panic!("Boundary edge not found on current face during walk");
            };

            // Move to adjacent face and oriented edge on that face
            let adjacent_face = mesh.face(twin);
            let next_oriented = twin;

            // If we've returned to the start oriented edge, we are done
            if next_oriented == start_oriented {
                break;
            }

            current_oriented = next_oriented;
            current_face = adjacent_face;

            iters += 1;
            if iters > max_iters {
                unreachable!("Boundary loop traversal exceeded iteration limit");
            }
        }

        // Verify we covered all boundary edges (there should be a single component)
        let mut expected_edges: HashSet<(VertID, VertID)> = HashSet::new();
        for face_id in boundary_faces.iter() {
            let fv: Vec<_> = mesh.vertices(*face_id).collect();
            for (i, j) in [(0, 1), (1, 2), (2, 0)] {
                let a = fv[i];
                let b = fv[j];
                // edge is a boundary edge if its endpoints belong to different patches
                if set_a.contains(&a) ^ set_a.contains(&b) {
                    let key = if a < b { (a, b) } else { (b, a) };
                    expected_edges.insert(key);
                }
            }
        }

        let traversed_edges: HashSet<(VertID, VertID)> = loop_edges
            .iter()
            .map(|&e| {
                let a = mesh.root(e);
                let b = mesh.toor(e);
                if a < b {
                    (a, b)
                } else {
                    (b, a)
                }
            })
            .collect();

        if expected_edges != traversed_edges {
            unreachable!("Boundary loop produced multiple components or mismatch");
        }

        BoundaryLoop {
            vertices: loop_edges,
        }
    }
}
