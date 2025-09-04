use crate::prelude::*;
use kdtree::{KdTree, distance::squared_euclidean};

#[derive(Debug, Clone)]
pub struct VertLocation<M: Tag>(KdTree<f64, VertKey<M>, [f64; 3]>);

impl<M: Tag> VertLocation<M> {
    #[must_use]
    pub fn nearest(&self, point: &[f64; 3]) -> (f64, VertKey<M>) {
        let neighbors = self.0.nearest(point, 1, &squared_euclidean).unwrap();
        let (d, i) = neighbors.first().unwrap();
        (*d, **i)
    }

    fn add(&mut self, point: [f64; 3], index: VertKey<M>) {
        self.0.add(point, index).unwrap();
    }
}
impl<M: Tag> Default for VertLocation<M> {
    fn default() -> Self {
        Self(KdTree::new(3))
    }
}

impl<M: Tag> Mesh<M> {
    #[must_use]
    pub fn kdtree(&self) -> VertLocation<M> {
        let mut tree = VertLocation::default();
        for id in self.vert_ids() {
            tree.add(self.position(id).into(), id);
        }
        tree
    }
}
