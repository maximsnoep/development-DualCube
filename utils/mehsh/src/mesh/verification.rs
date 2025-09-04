use crate::prelude::*;
use core::panic;

// This embedded mesh is:
//      a closed 2-manifold: Each edge corresponds to exactly two faces.
//      connected: There exists a path between any two vertices.
//      orientable: There exists a consistent normal for each face.
//      polygonal: Each face is a simple polygon (lies in a plane, no intersections).
// These requirements will be true per construction.

impl<M: Tag> Mesh<M> {
    // Asserts that all elements have their required properties set.
    // These assertions should all pass per construction.
    pub fn assert_properties(&self) {
        for edge_id in self.edge_ids() {
            assert!(self.edge_root.contains(edge_id), "{edge_id:?} has no root");
            assert!(self.edge_face.contains(edge_id), "{edge_id:?} has no face");
            assert!(self.edge_next.contains(edge_id), "{edge_id:?} has no next");
            assert!(self.edge_twin.contains(edge_id), "{edge_id:?} has no twin");
        }
        for vert_id in self.vert_ids() {
            assert!(self.vert_repr.contains(vert_id), "{vert_id:?} has no vrep");
        }
        for face_id in self.face_ids() {
            assert!(self.face_repr.contains(face_id), "{face_id:?} has no frep");
        }
    }

    // Asserts that all references between elements are valid.
    // These assertions should all pass per construction.
    pub fn assert_references(&self) {
        for edge_id in self.edge_ids() {
            let root_id = self.root(edge_id);
            assert!(self.verts.contains(root_id), "{edge_id:?} has non-existing root ({root_id:?})");

            let face_id = self.face(edge_id);
            assert!(self.faces.contains(face_id), "{edge_id:?} has non-existing face ({face_id:?})");

            let next_id = self.next(edge_id);
            assert!(self.edges.contains(next_id), "{edge_id:?} has non-existing next ({next_id:?})");

            let twin_id = self.twin(edge_id);
            assert!(self.edges.contains(twin_id), "{edge_id:?} has non-existing twin ({twin_id:?})");
        }
        for vert_id in self.vert_ids() {
            if let Some(repr_id) = self.vert_repr.get(vert_id) {
                assert!(self.edges.contains(repr_id), "{vert_id:?} has non-existing face reference ({repr_id:?})");
            } else {
                panic!("{vert_id:?} has no face reference defined");
            }
        }
        for face_id in self.face_ids() {
            if let Some(repr_id) = self.face_repr.get(face_id) {
                assert!(self.edges.contains(repr_id), "{face_id:?} has non-existing vertex reference ({repr_id:?})");
            } else {
                panic!("{face_id:?} has no vertex reference defined");
            }
        }
    }

    // Asserts the invariants of the DCEL structure.
    pub fn assert_invariants(&self) {
        // this->twin->twin == this
        for edge_id in self.edge_ids() {
            assert!(self.twin(self.twin(edge_id)) == edge_id, "{edge_id:?}: [this->twin->twin == this] violated");
        }
        // this->twin->next->root == this->root
        for edge_id in self.edge_ids() {
            assert!(
                self.root(self.next(self.twin(edge_id))) == self.root(edge_id),
                "{edge_id:?}: [this->twin->next->root == this->root] violated"
            );
        }
        // this->next->face == this->face
        for edge_id in self.edge_ids() {
            assert!(
                self.face(self.next(edge_id)) == self.face(edge_id),
                "{edge_id:?}: [this->next->face == this->face] violated"
            );
        }
        // this->next->...->next == this
        const MAX_FACE_SIZE: usize = 10;
        for edge_id in self.edge_ids() {
            let mut next_id = edge_id;
            for _ in 0..MAX_FACE_SIZE {
                next_id = self.next(next_id);
                if next_id == edge_id {
                    break;
                }
            }
            assert!(next_id == edge_id, "{edge_id:?}: [this->next->...->next == this] violated");
        }
    }

    // #[must_use]
    // pub fn is_connected(&self) -> bool {
    //     hutspot::graph::find_ccs(&self.vert_ids(), self.neighbor_function_primal()).len() == 1
    // }

    pub fn is_polygonal(&self) -> Result<(), MeshError<M>> {
        // Make sure the mesh is polygonal
        for face_id in self.face_ids() {
            let corners = self.vertices(face_id);

            // Check that the face is a polygon
            if corners.len() < 3 {
                return Err(MeshError::FaceNotPolygon(face_id));
            }

            // Check that the face is planar

            let a = corners[0];
            for o in corners.into_iter().skip(1) {
                if self.position(a) == self.position(o) && self.position(a) != Vector3D::zeros() {
                    println!("WARN: Face {face_id:?} has two identical corners: {a:?} and {o:?}");
                }
            }

            // Check that the face is simple
            for edge_a in self.edges(face_id) {
                for edge_b in self.edges(face_id) {
                    if edge_a == edge_b {
                        continue;
                    }
                    let a_u = self.position(self.root(edge_a));
                    let a_v = self.position(self.toor(edge_a));
                    let b_u = self.position(self.root(edge_b));
                    let b_v = self.position(self.toor(edge_b));
                    if geom::calculate_3d_lineseg_intersection(a_u, a_v, b_u, b_v).is_some() && a_u != b_u && a_u != b_v && a_v != b_u && a_v != b_v {
                        return Err(MeshError::FaceNotSimple(face_id));
                    }
                }
            }
        }
        Ok(())
    }
}
