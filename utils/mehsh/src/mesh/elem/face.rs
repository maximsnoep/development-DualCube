use crate::prelude::*;
use core::panic;
use std::collections::HashSet;

impl<M: Tag> Mesh<M> {
    #[must_use]
    pub fn frep(&self, id: FaceKey<M>) -> EdgeKey<M> {
        self.face_repr
            .get(id)
            .unwrap_or_else(|| panic!("{id:?} has no frep"))
    }

    // Returns the two edges of a given face that are connected to the given vertex.
    #[must_use]
    pub fn edges_in_face_with_vert(
        &self,
        face_id: FaceKey<M>,
        vert_id: VertKey<M>,
    ) -> Option<[EdgeKey<M>; 2]> {
        let edges = self.edges(face_id);
        edges
            .into_iter()
            .filter(|&edge_id| self.root(edge_id) == vert_id || self.toor(edge_id) == vert_id)
            .collect_tuple()
            .map(|(a, b)| if self.next(a) == b { [a, b] } else { [b, a] })
    }

    // Returns the edge between the two faces. Returns None if the faces do not share an edge.
    #[must_use]
    pub fn edge_between_faces(
        &self,
        id_a: FaceKey<M>,
        id_b: FaceKey<M>,
    ) -> Option<(EdgeKey<M>, EdgeKey<M>)> {
        for edge_a_id in self.edges(id_a) {
            for edge_b_id in self.edges(id_b) {
                if self.twin(edge_a_id) == edge_b_id {
                    return Some((edge_a_id, edge_b_id));
                }
            }
        }
        None
    }

    // Returns the face with given vertices.
    #[must_use]
    pub fn face_with_verts(&self, verts: &[VertKey<M>]) -> Option<FaceKey<M>> {
        self.faces(verts[0]).into_iter().find(|&face_id| {
            verts
                .iter()
                .all(|&vert_id| self.faces(vert_id).contains(&face_id))
        })
    }

    // Vector area of a given face.
    #[must_use]
    pub fn vector_area(&self, id: FaceKey<M>) -> Vector3D {
        self.edges(id).fold(Vector3D::zeros(), |sum, edge_id| {
            let u = self.vector(self.twin(edge_id));
            let v = self.vector(self.next(edge_id));
            sum + u.cross(&v)
        })
    }

    // Area of a given triangle.
    #[must_use]
    pub fn triangle_area(&self, id: FaceKey<M>) -> Float {
        self.vector_area(id).magnitude() / 2.0
    }
}

impl<M: Tag> HasPosition<FACE, M> for Mesh<M> {
    // Get centroid of a given polygonal face.
    // https://en.wikipedia.org/wiki/Centroid
    // Be careful with concave faces, the centroid might lay outside the face.
    fn position(&self, id: FaceKey<M>) -> Vector3D {
        math::calculate_average_f64(
            self.edges(id)
                .map(|edge_id| self.position(self.root(edge_id))),
        )
    }
}

impl<M: Tag> HasNormal<FACE, M> for Mesh<M> {
    fn compute_normal(&self, id: FaceKey<M>) -> Vector3D {
        // TODO: Make this better for non-planar faces.
        let mut vertices = self.vertices(id);
        let p_u = self.position(vertices.next().unwrap());
        let p_v = self.position(vertices.next().unwrap());
        let p_w = self.position(vertices.next().unwrap());
        let p = p_v - p_u;
        let q = p_w - p_u;
        p.cross(&q).normalize()
    }

    fn normal(&self, id: FaceKey<M>) -> Vector3D {
        self.face_normal_cache
            .get(&id)
            .copied()
            .unwrap_or_else(|| self.compute_normal(id))
    }
}

impl<M: Tag> HasSize<FACE, M> for Mesh<M> {
    // Area of a given face.
    fn size(&self, id: FaceKey<M>) -> Float {
        self.vector_area(id).magnitude() / 2.0
    }
}

impl<M: Tag> HasVertices<FACE, M> for Mesh<M> {
    fn vertices(&self, id: FaceKey<M>) -> impl Iterator<Item = VertKey<M>> {
        self.edges(id).map(|edge_id| self.root(edge_id))
    }
}

impl<M: Tag> HasEdges<FACE, M> for Mesh<M> {
    fn edges(&self, id: FaceKey<M>) -> impl Iterator<Item = EdgeKey<M>> {
        let rep = self.frep(id);
        std::iter::once(rep).chain(self.neighbors(rep))
    }
}

impl<M: Tag> HasNeighbors<FACE, M> for Mesh<M> {
    fn neighbors(&self, id: FaceKey<M>) -> impl Iterator<Item = FaceKey<M>> {
        self.edges(id).map(|edge_id| self.face(self.twin(edge_id)))
    }

    fn neighbors_k(
        &self,
        id: ids::Key<FACE, M>,
        k: usize,
    ) -> impl Iterator<Item = ids::Key<FACE, M>> {
        let mut neighbors = vec![id];
        for _ in 0..k {
            neighbors = neighbors
                .into_iter()
                .flat_map(|n| self.neighbors(n))
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();
        }
        neighbors.retain(|&n| n != id);
        neighbors.into_iter()
    }
}

impl<M: Tag> HasRing<FACE, M> for Mesh<M> {
    fn ring(&self, id: ids::Key<FACE, M>, k: usize) -> Vec<Vec<ids::Key<FACE, M>>> {
        // k = 0 [id]
        // k = 1 [neighbors of id, but not id]
        // etc
        let mut rings = vec![vec![id]];

        for _ in 1..k {
            let last_ring = rings.last().unwrap();
            let mut next_ring = vec![];
            for &face_id in last_ring {
                for neighbor in self.neighbors(face_id) {
                    if !next_ring.contains(&neighbor)
                        && neighbor != id
                        && !last_ring.contains(&neighbor)
                    {
                        next_ring.push(neighbor);
                    }
                }
            }
            rings.push(next_ring);
        }

        rings
    }
}
