use crate::prelude::*;
use bvh::{
    aabb::{Aabb, Bounded},
    bounding_hierarchy::BHShape,
    bvh::Bvh,
    point_query::PointDistance,
};
use itertools::Itertools;

#[derive(Debug, Clone)]
pub struct FaceLocation<M: Tag>((Bvh<f64, 3>, Vec<TriangleBvhShape<M>>));
impl<M: Tag> FaceLocation<M> {
    #[must_use]
    pub fn nearest(&self, point: &[f64; 3]) -> FaceKey<M> {
        let neighbor = self.0.0.nearest_to(nalgebra::Point3::from_slice(point), &self.0.1);
        let (t, _) = neighbor.unwrap();
        t.real_index
    }
    fn overwrite(&mut self, shapes: &mut [TriangleBvhShape<M>]) {
        self.0.0 = Bvh::build(shapes);
        self.0.1 = shapes.to_vec();
    }

    /// Return all faces whose bounding box intersects a axis-aligned box
    pub fn faces_intersecting_bounds(
        &self,
        min_corner: [f64; 3],
        max_corner: [f64; 3],
    ) -> Vec<FaceKey<M>> {
        let query = Aabb::with_bounds(
            nalgebra::Point3::from(min_corner),
            nalgebra::Point3::from(max_corner),
        );
        self.0
            .0
            .traverse(&query, &self.0.1)
            .into_iter()
            .map(|shape| shape.real_index)
            .collect()
    }
}
impl<M: Tag> Default for FaceLocation<M> {
    fn default() -> Self {
        Self((Bvh::build::<TriangleBvhShape<M>>(&mut []), Vec::new()))
    }
}

impl<M: Tag> Mesh<M> {
    #[must_use]
    pub fn bvh(&self) -> FaceLocation<M> {
        let mut bvh = FaceLocation::default();
        let mut triangles = self
            .face_ids()
            .iter()
            .enumerate()
            .map(|(i, &face_id)| {
                let Some([v0, v1, v2]) = self.vertices(face_id).collect_array::<3>() else {
                    panic!("Expected exactly three vertices for face {face_id:?}");
                };
                TriangleBvhShape {
                    corners: [self.position(v0), self.position(v1), self.position(v2)],
                    node_index: i,
                    real_index: face_id,
                }
            })
            .collect_vec();

        bvh.overwrite(&mut triangles);
        bvh
    }
}

// impl for triangles
#[derive(Debug, Clone)]
struct TriangleBvhShape<M: Tag> {
    corners: [Vector3D; 3],
    node_index: usize,
    real_index: FaceKey<M>,
}

impl<M: Tag> PointDistance<f64, 3> for TriangleBvhShape<M> {
    fn distance_squared(&self, query_point: nalgebra::Point<f64, 3>) -> f64 {
        geom::distance_to_triangle(
            Vector3D::new(query_point[0], query_point[1], query_point[2]),
            (self.corners[0], self.corners[1], self.corners[2]),
        )
    }
}

impl<M: Tag> Bounded<f64, 3> for TriangleBvhShape<M> {
    fn aabb(&self) -> Aabb<f64, 3> {
        let mut iter = self.corners.iter();
        let first = iter.next().unwrap();

        let (mut min_x, mut max_x) = (first.x, first.x);
        let (mut min_y, mut max_y) = (first.y, first.y);
        let (mut min_z, mut max_z) = (first.z, first.z);

        for v in iter {
            min_x = min_x.min(v.x);
            max_x = max_x.max(v.x);
            min_y = min_y.min(v.y);
            max_y = max_y.max(v.y);
            min_z = min_z.min(v.z);
            max_z = max_z.max(v.z);
        }

        let min = nalgebra::Point3::new(min_x, min_y, min_z);
        let max = nalgebra::Point3::new(max_x, max_y, max_z);
        Aabb::with_bounds(min, max)
    }
}

impl<M: Tag> BHShape<f64, 3> for TriangleBvhShape<M> {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}
