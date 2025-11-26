use crate::prelude::*;
use core::panic;
use std::collections::HashSet;

impl<M: Tag> Mesh<M> {
    #[must_use]
    pub fn root(&self, id: EdgeKey<M>) -> VertKey<M> {
        self.edge_root.get(id).unwrap_or_else(|| panic!("{id:?} has no root"))
    }

    #[must_use]
    pub fn toor(&self, id: EdgeKey<M>) -> VertKey<M> {
        self.root(self.next(id))
    }

    #[must_use]
    pub fn twin(&self, id: EdgeKey<M>) -> EdgeKey<M> {
        self.edge_twin.get(id).unwrap_or_else(|| panic!("{id:?} has no twin"))
    }

    #[must_use]
    pub fn next(&self, id: EdgeKey<M>) -> EdgeKey<M> {
        self.edge_next.get(id).unwrap_or_else(|| panic!("{id:?} has no next"))
    }

    // Returns the four edges around a given edge.
    #[must_use]
    pub fn quad(&self, id: EdgeKey<M>) -> [EdgeKey<M>; 4] {
        let edge0 = self.next(id);
        let edge1 = self.next(edge0);
        let twin = self.twin(id);
        let edge2 = self.next(twin);
        let edge3 = self.next(edge2);
        [edge0, edge1, edge2, edge3]
    }

    #[must_use]
    pub fn face(&self, id: EdgeKey<M>) -> FaceKey<M> {
        self.edge_face.get(id).unwrap_or_else(|| panic!("{id:?} has no face"))
    }

    #[must_use]
    pub fn common_endpoint(&self, edge_a: EdgeKey<M>, edge_b: EdgeKey<M>) -> Option<VertKey<M>> {
        let vertices_a = self.vertices(edge_a);
        let a0 = vertices_a[0];
        let a1 = vertices_a[1];
        let vertices_b = self.vertices(edge_b);
        let b0 = vertices_b[0];
        let b1 = vertices_b[1];
        if a0 == b0 || a0 == b1 {
            Some(a0)
        } else if a1 == b0 || a1 == b1 {
            Some(a1)
        } else {
            None
        }
    }

    // Get midpoint of a given edge with some offset
    #[must_use]
    pub fn midpoint_offset<T>(&self, edge_id: EdgeKey<M>, offset: T) -> Vector3D
    where
        T: Into<f64>,
    {
        self.position(self.root(edge_id)) + self.vector(edge_id) * offset.into()
    }

    // Get vector of a given edge.
    #[must_use]
    pub fn vector(&self, id: EdgeKey<M>) -> Vector3D {
        let vertices = self.vertices(id);
        let u = vertices[0];
        let v = vertices[1];
        self.position(v) - self.position(u)
    }

    // Get angle (in radians) between two edges `u` and `v`.
    #[must_use]
    pub fn angle(&self, u: EdgeKey<M>, v: EdgeKey<M>) -> Float {
        self.vector(u).angle(&self.vector(v))
    }

    // Get the dihedral angle (in radians) between the two faces of `u`.
    #[must_use]
    pub fn dihedral(&self, u: EdgeKey<M>) -> Float {
        let faces = self.faces(u);
        let n1 = self.normal(faces[0]);
        let n2 = self.normal(faces[1]);
        n1.angle(&n2)
    }
}

impl<M: Tag> HasPosition<EDGE, M> for Mesh<M> {
    fn position(&self, id: EdgeKey<M>) -> Vector3D {
        self.midpoint_offset(id, 0.5)
    }
}

impl<M: Tag> HasNormal<EDGE, M> for Mesh<M> {
    fn normal(&self, id: EdgeKey<M>) -> Vector3D {
        match self.faces(id)[..] {
            [f1, f2] => (self.normal(f1) + self.normal(f2)).normalize(),
            _ => panic!("Expected exactly two faces for edge {id:?}"),
        }
    }
}

impl<M: Tag> HasSize<EDGE, M> for Mesh<M> {
    fn size(&self, id: EdgeKey<M>) -> Float {
        self.vector(id).magnitude()
    }
}

impl<M: Tag> HasVertices<EDGE, M> for Mesh<M> {
    fn vertices(&self, id: EdgeKey<M>) -> Vec<VertKey<M>> {
        vec![self.root(id), self.root(self.next(id))]
    }
}

impl<M: Tag> HasFaces<EDGE, M> for Mesh<M> {
    fn faces(&self, id: EdgeKey<M>) -> Vec<FaceKey<M>> {
        vec![self.face(id), self.face(self.twin(id))]
    }
}

// This is incorrect ;)
impl<M: Tag> HasNeighbors<EDGE, M> for Mesh<M> {
    fn neighbors(&self, id: EdgeKey<M>) -> Vec<EdgeKey<M>> {
        let mut nexts = vec![];
        let mut cur = id;
        loop {
            cur = self.next(cur);
            if cur == id {
                return nexts;
            }
            nexts.push(cur);
        }
    }

    fn neighbors_k(&self, id: ids::Key<EDGE, M>, k: usize) -> Vec<ids::Key<EDGE, M>> {
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
        neighbors
    }
}

// More correct neighbors:
impl<M: Tag> Mesh<M> {
    pub fn neighbors2(&self, id: EdgeKey<M>) -> HashSet<EdgeKey<M>> {
        let mut neighbors = HashSet::new();

        let es1 = self.edges(self.vertices(id)[0]).into_iter().flat_map(|e| [e, self.twin(e)]);
        let es2 = self.edges(self.vertices(id)[1]).into_iter().flat_map(|e| [e, self.twin(e)]);
        neighbors.extend(es1);
        neighbors.extend(es2);
        neighbors.retain(|&e| e != id);
        neighbors
    }
}
