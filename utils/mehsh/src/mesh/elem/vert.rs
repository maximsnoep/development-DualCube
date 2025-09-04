use crate::prelude::*;
use core::panic;
use itertools::Itertools;

impl<M: Tag> Mesh<M> {
    #[must_use]
    pub fn vrep(&self, id: VertKey<M>) -> EdgeKey<M> {
        self.vert_repr.get(id).unwrap_or_else(|| panic!("{id:?} has no vrep"))
    }

    #[must_use]
    pub fn distance(&self, v_a: VertKey<M>, v_b: VertKey<M>) -> Float {
        self.position(v_a).metric_distance(&self.position(v_b))
    }

    #[must_use]
    pub fn vertex_angle(&self, a: VertKey<M>, b: VertKey<M>, c: VertKey<M>) -> Float {
        (self.position(b) - self.position(a)).angle(&(self.position(b) - self.position(c)))
    }

    // Angular defect of a vertex is 2*PI - C, where C is the sum of all the angles at the vertex.
    // https://en.wikipedia.org/wiki/Angular_defect
    #[must_use]
    pub fn defect(&self, id: VertKey<M>) -> Float {
        let sum_of_angles = self.edges(id).iter().fold(0., |sum, &outgoing_edge_id| {
            let incoming_edge_id = self.twin(outgoing_edge_id);
            let next_edge_id = self.next(incoming_edge_id);
            let angle = self.angle(outgoing_edge_id, next_edge_id);
            sum + angle
        });

        // 2PI - C
        Float::from(2.0).mul_add(PI, -sum_of_angles)
    }

    #[must_use]
    pub fn wedges(&self, a: VertKey<M>, b: VertKey<M>, c: VertKey<M>) -> (Vec<VertKey<M>>, Vec<VertKey<M>>) {
        // First wedge is a to c (around b)
        let wedge1 = std::iter::once(a)
            .chain(self.neighbors(b).into_iter().cycle().skip_while(|&v| v != a).skip(1).take_while(|&v| v != c))
            .chain([c])
            .collect_vec();

        // Second wedge is c to a (around b)
        let wedge2 = std::iter::once(c)
            .chain(self.neighbors(b).into_iter().cycle().skip_while(|&v| v != c).skip(1).take_while(|&v| v != a))
            .chain([a])
            .collect_vec();

        // Return the wedges
        (wedge1, wedge2)
    }

    #[must_use]
    pub fn wedge_alpha(&self, (b, wedge): (VertKey<M>, &[VertKey<M>])) -> f64 {
        wedge.windows(2).map(|vs| self.vertex_angle(vs[0], b, vs[1])).sum::<f64>()
    }

    #[must_use]
    pub fn shortest_wedge(&self, a: VertKey<M>, b: VertKey<M>, c: VertKey<M>) -> (Vec<VertKey<M>>, f64) {
        let (w1, w2) = self.wedges(a, b, c);
        let (a1, a2) = (self.wedge_alpha((b, &w1)), self.wedge_alpha((b, &w2)));
        if a1 < a2 { (w1, a1) } else { (w2.into_iter().rev().collect_vec(), a2) }
    }

    #[must_use]
    pub fn verts_to_edges(&self, verts: &[VertKey<M>]) -> Vec<EdgeKey<M>> {
        verts
            .iter()
            .flat_map(|&vert_id| {
                self.edges(vert_id)
                    .into_iter()
                    .filter(|&edge_id| verts.contains(&self.toor(edge_id)))
                    .collect_vec()
            })
            .collect_vec()
    }

    // Returns the edge between the two vertices. Returns None if the vertices are not connected.
    #[must_use]
    pub fn edge_between_verts(&self, id_a: VertKey<M>, id_b: VertKey<M>) -> Option<(EdgeKey<M>, EdgeKey<M>)> {
        for &edge_a_id in &self.edges(id_a) {
            let id_a2 = self.toor(edge_a_id);
            for &edge_b_id in &self.edges(id_b) {
                let id_b2 = self.toor(edge_b_id);
                if id_a2 == id_b && id_b2 == id_a {
                    return Some((edge_a_id, edge_b_id));
                }
            }
        }
        None
    }
}

impl<M: Tag> SetPosition<VERT, M> for Mesh<M> {
    fn set_position(&mut self, id: VertKey<M>, position: Vector3D) {
        if let Some(old_position) = self.verts.get_mut(id) {
            *old_position = position;
        } else {
            panic!("V:{id:?} not initialized");
        }
    }
}

impl<M: Tag> HasPosition<VERT, M> for Mesh<M> {
    fn position(&self, id: VertKey<M>) -> Vector3D {
        self.verts.get(id).copied().unwrap_or_else(|| panic!("V:{id:?} not initialized"))
    }
}

impl<M: Tag> HasNormal<VERT, M> for Mesh<M> {
    fn normal(&self, id: VertKey<M>) -> Vector3D {
        self.faces(id).iter().map(|&face_id| self.normal(face_id)).sum::<Vector3D>().normalize()
    }
}

impl<M: Tag> HasEdges<VERT, M> for Mesh<M> {
    fn edges(&self, id: VertKey<M>) -> Vec<EdgeKey<M>> {
        let mut edges = vec![self.vrep(id)];
        loop {
            let next_of_twin = self.next(self.twin(edges.last().copied().unwrap()));
            if edges.contains(&next_of_twin) {
                return edges;
            }
            edges.push(next_of_twin);
        }
    }
}

impl<M: Tag> HasFaces<VERT, M> for Mesh<M> {
    fn faces(&self, id: VertKey<M>) -> Vec<FaceKey<M>> {
        self.edges(id).iter().map(|&edge_id| self.face(edge_id)).collect()
    }
}

impl<M: Tag> HasNeighbors<VERT, M> for Mesh<M> {
    fn neighbors(&self, id: VertKey<M>) -> Vec<VertKey<M>> {
        self.edges(id).iter().map(|&edge_id| self.root(self.twin(edge_id))).collect()
    }
}
