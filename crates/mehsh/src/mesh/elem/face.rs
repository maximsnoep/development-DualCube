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
        // Newell / shoelace formula: sum of p_i x p_{i+1}. Its magnitude is twice the
        // (projected) area, and it points along the face normal (CCW winding).
        // Positions are taken relative to the first corner to avoid cancellation far from the origin.
        let origin = self.position(self.root(self.frep(id)));
        self.edges(id).fold(Vector3D::zeros(), |sum, edge_id| {
            let u = self.position(self.root(edge_id)) - origin;
            let v = self.position(self.toor(edge_id)) - origin;
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
        // Newell's method: robust for non-planar polygons and polygons whose first corners are
        // collinear. Degenerate (zero-area) faces get a zero normal instead of NaN.
        self.vector_area(id)
            .try_normalize(1e-300)
            .unwrap_or_else(Vector3D::zeros)
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
        // BFS: track visited across all depths so k-1 ring nodes are never dropped
        let mut visited: HashSet<ids::Key<FACE, M>> = HashSet::new();
        visited.insert(id);
        let mut frontier = vec![id];
        for _ in 0..k {
            let mut next_frontier = vec![];
            for n in frontier {
                for neighbor in self.neighbors(n) {
                    if visited.insert(neighbor) {
                        next_frontier.push(neighbor);
                    }
                }
            }
            frontier = next_frontier;
        }
        visited.remove(&id);
        visited.into_iter()
    }
}

impl<M: Tag> HasRing<FACE, M> for Mesh<M> {
    fn ring(&self, id: ids::Key<FACE, M>, k: usize) -> Vec<Vec<ids::Key<FACE, M>>> {
        // k = 0 [id]
        // k = 1 [neighbors of id, but not id]
        // etc
        let mut rings = vec![vec![id]];
        // Track ALL previously seen faces — not just last ring — so faces can't
        // bleed into later rings. Also gives O(1) membership checks.
        let mut visited: HashSet<ids::Key<FACE, M>> = HashSet::new();
        visited.insert(id);

        for _ in 0..k {
            // Clone so the borrow on `rings` ends before `rings.push` below.
            let last_ring = rings.last().unwrap().clone();
            let mut next_ring = vec![];
            for face_id in last_ring {
                for neighbor in self.neighbors(face_id) {
                    if visited.insert(neighbor) {
                        next_ring.push(neighbor);
                    }
                }
            }
            rings.push(next_ring);
        }

        rings
    }
}
