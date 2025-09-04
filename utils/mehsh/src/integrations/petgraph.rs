use crate::prelude::*;
use itertools::Itertools;
use petgraph::{Directed, csr::Csr};
use std::collections::HashSet;

type Graph<T> = Csr<(), T, Directed, u32>;

impl<M: Tag> Mesh<M>
where
    M: std::default::Default + std::cmp::Eq + std::hash::Hash + Copy + Clone,
{
    // Construct a graph with edge filters, vertex filters, and a weight function.
    #[must_use]
    pub fn to_petgraph_modified<W, T>(
        &self,
        filter_verts: &HashSet<VertKey<M>>,
        filter_edges: &HashSet<EdgeKey<M>>,
        weight_function: W,
    ) -> (Graph<T>, ids::IdMap<VERT, M>)
    where
        W: Fn(EdgeKey<M>) -> T,
        T: Clone,
    {
        let mut key_to_int = ids::IdMap::new();
        for (i, id) in self.verts.ids().enumerate() {
            key_to_int.insert(i, id);
        }

        let sorted_edges = self
            .edges
            .ids()
            .filter(|&id| !filter_edges.contains(&id) && !filter_verts.contains(&self.root(id)) && !filter_verts.contains(&self.root(self.twin(id))))
            .map(|id| {
                let vertices = self.vertices(id);
                let u = vertices[0];
                let v = vertices[1];
                (
                    key_to_int.id(&u).unwrap().to_owned(),
                    key_to_int.id(&v).unwrap().to_owned(),
                    weight_function(id),
                )
            })
            .sorted_by_key(|&(u, v, _)| (u, v))
            .map(|(u, v, weight)| (u as u32, v as u32, weight))
            .collect::<Vec<_>>();

        (Csr::from_sorted_edges(&sorted_edges).unwrap(), key_to_int)
    }

    pub fn to_petgraph_with_weights<W, T>(&self, weight_function: W) -> (Graph<T>, ids::IdMap<VERT, M>)
    where
        W: Fn(EdgeKey<M>) -> T,
        T: Clone,
    {
        self.to_petgraph_modified(&HashSet::new(), &HashSet::new(), weight_function)
    }

    #[must_use]
    pub fn to_petgraph<W, T>(&self) -> (Graph<()>, ids::IdMap<VERT, M>)
    where
        W: Fn(EdgeKey<M>) -> T,
        T: Clone,
    {
        self.to_petgraph_modified(&HashSet::new(), &HashSet::new(), |_| ())
    }
}

// // Weight function
// pub fn weight_function_euclidean(&self) -> impl Fn(VertID, VertID) -> OrderedFloat<Float> + '_ {
//     |a, b| OrderedFloat(self.distance(a, b))
// }

// // Weight function
// pub fn weight_function_angle_edges(&self, slack: i32) -> impl Fn(EdgeID, EdgeID) -> OrderedFloat<Float> + '_ {
//     move |a, b| OrderedFloat(self.angle(a, b).powi(slack))
// }

// // Weight function
// pub fn weight_function_angle_edgepairs(&self, slack: i32) -> impl Fn((EdgeID, EdgeID), (EdgeID, EdgeID)) -> OrderedFloat<Float> + '_ {
//     move |a, b| {
//         let vector_a = self.midpoint(a.1) - self.midpoint(a.0);
//         let vector_b = self.midpoint(b.1) - self.midpoint(b.0);
//         OrderedFloat(self.vec_angle(vector_a, vector_b).powi(slack))
//     }
// }

// // Weight function
// pub fn weight_function_angle_edgepairs_aligned(
//     &self,
//     angular_slack: i32,
//     alignment_slack: i32,
//     axis: Vector3D,
// ) -> impl Fn([EdgeID; 2], [EdgeID; 2]) -> OrderedFloat<Float> + '_ {
//     move |a, b| {
//         let vector_a = self.midpoint(a[1]) - self.midpoint(a[0]);
//         let vector_b = self.midpoint(b[1]) - self.midpoint(b[0]);

//         let weight = self.vec_angle(vector_a, vector_b).powi(angular_slack)
//             + (self.vec_angle(vector_a.cross(&self.edge_normal(a[0])), axis)).powi(alignment_slack)
//             + (self.vec_angle(vector_b.cross(&self.edge_normal(b[0])), axis)).powi(alignment_slack);

//         OrderedFloat(weight)
//     }
// }

// // Weight function
// pub fn weight_function_angle_edgepairs_aligned_components(
//     &self,
//     axis: Vector3D,
// ) -> impl Fn([EdgeID; 2], [EdgeID; 2]) -> (OrderedFloat<Float>, OrderedFloat<Float>, OrderedFloat<Float>) + '_ {
//     move |a, b| {
//         let vector_a = self.midpoint(a[1]) - self.midpoint(a[0]);
//         let vector_b = self.midpoint(b[1]) - self.midpoint(b[0]);

//         (
//             OrderedFloat(self.vec_angle(vector_a, vector_b)),
//             OrderedFloat(self.vec_angle(vector_a.cross(&self.edge_normal(a[0])), axis)),
//             OrderedFloat(self.vec_angle(vector_b.cross(&self.edge_normal(b[0])), axis)),
//         )
//     }
// }
