use crate::prelude::*;
use core::panic;
use std::collections::HashSet;

impl<M: Tag> Mesh<M> {
    #[must_use]
    pub fn root(&self, id: EdgeKey<M>) -> VertKey<M> {
        self.edge_root
            .get(id)
            .unwrap_or_else(|| panic!("{id:?} has no root"))
    }

    #[must_use]
    pub fn toor(&self, id: EdgeKey<M>) -> VertKey<M> {
        self.root(self.next(id))
    }

    #[must_use]
    pub fn twin(&self, id: EdgeKey<M>) -> EdgeKey<M> {
        self.edge_twin
            .get(id)
            .unwrap_or_else(|| panic!("{id:?} has no twin"))
    }

    #[must_use]
    pub fn next(&self, id: EdgeKey<M>) -> EdgeKey<M> {
        self.edge_next
            .get(id)
            .unwrap_or_else(|| panic!("{id:?} has no next"))
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
        self.edge_face
            .get(id)
            .unwrap_or_else(|| panic!("{id:?} has no face"))
    }

    #[must_use]
    pub fn common_endpoint(&self, edge_a: EdgeKey<M>, edge_b: EdgeKey<M>) -> Option<VertKey<M>> {
        if let (Some([a0, a1]), Some([b0, b1])) = (
            self.vertices(edge_a).collect_array::<2>(),
            self.vertices(edge_b).collect_array::<2>(),
        ) {
            if a0 == b0 || a0 == b1 {
                Some(a0)
            } else if a1 == b0 || a1 == b1 {
                Some(a1)
            } else {
                None
            }
        } else {
            panic!("Expected exactly two vertices for edges {edge_a:?} and {edge_b:?}");
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
        if let Some([u, v]) = self.vertices(id).collect_array::<2>() {
            self.position(v) - self.position(u)
        } else {
            panic!("Expected exactly two vertices for edge {id:?}");
        }
    }

    // Get angle (in radians) between two edges `u` and `v`.
    #[must_use]
    pub fn angle(&self, u: EdgeKey<M>, v: EdgeKey<M>) -> Float {
        self.vector(u).angle(&self.vector(v))
    }

    // Get the dihedral angle (in radians) between the two faces of `u`.
    #[must_use]
    pub fn dihedral(&self, u: EdgeKey<M>) -> Float {
        if let Some([f1, f2]) = self.faces(u).collect_array::<2>() {
            self.normal(f1).angle(&self.normal(f2))
        } else {
            panic!("Expected exactly two faces for edge {u:?}");
        }
    }
}

impl<M: Tag> HasPosition<EDGE, M> for Mesh<M> {
    fn position(&self, id: EdgeKey<M>) -> Vector3D {
        self.midpoint_offset(id, 0.5)
    }
}

impl<M: Tag> HasNormal<EDGE, M> for Mesh<M> {
    fn compute_normal(&self, id: EdgeKey<M>) -> Vector3D {
        if let Some([f1, f2]) = self.faces(id).collect_array::<2>() {
            (self.normal(f1) + self.normal(f2)).normalize()
        } else {
            panic!("Expected exactly two faces for edge {id:?}");
        }
    }

    fn normal(&self, id: EdgeKey<M>) -> Vector3D {
        self.compute_normal(id)
    }
}

impl<M: Tag> HasSize<EDGE, M> for Mesh<M> {
    fn size(&self, id: EdgeKey<M>) -> Float {
        self.vector(id).magnitude()
    }
}

impl<M: Tag> HasVertices<EDGE, M> for Mesh<M> {
    fn vertices(&self, id: EdgeKey<M>) -> impl Iterator<Item = VertKey<M>> {
        std::iter::once(self.root(id)).chain(std::iter::once(self.root(self.next(id))))
    }
}

impl<M: Tag> HasFaces<EDGE, M> for Mesh<M> {
    fn faces(&self, id: EdgeKey<M>) -> impl Iterator<Item = FaceKey<M>> {
        std::iter::once(self.face(id)).chain(std::iter::once(self.face(self.twin(id))))
    }
}

// This is incorrect ;)
impl<M: Tag> HasNeighbors<EDGE, M> for Mesh<M> {
    fn neighbors(&self, id: EdgeKey<M>) -> impl Iterator<Item = EdgeKey<M>> {
        let mut cur = id;
        std::iter::from_fn(move || {
            cur = self.next(cur);
            if cur == id { None } else { Some(cur) }
        })
    }

    fn neighbors_k(
        &self,
        id: ids::Key<EDGE, M>,
        k: usize,
    ) -> impl Iterator<Item = ids::Key<EDGE, M>> {
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

use std::collections::VecDeque;

// More correct neighbors:
impl<M: Tag> Mesh<M> {
    pub fn neighbors2(&self, id: EdgeKey<M>) -> HashSet<EdgeKey<M>> {
        let mut out = HashSet::new();
        for f in self.faces(id) {
            for e in self.edges(f) {
                if e != id {
                    out.insert(e);
                }
            }
        }
        out
    }

    pub fn neighbors2_k(&self, id: EdgeKey<M>, k: usize) -> HashSet<EdgeKey<M>> {
        let mut visited = HashSet::new();
        let mut q = VecDeque::new();

        visited.insert(id);
        q.push_back((id, 0usize));

        while let Some((e, dist)) = q.pop_front() {
            if dist == k {
                continue;
            }

            for n in self.neighbors2(e) {
                if visited.insert(n) {
                    q.push_back((n, dist + 1));
                }
            }
        }

        visited.remove(&id);
        visited
    }
}
