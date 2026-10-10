use crate::prelude::*;
use kiddo::{ImmutableKdTree, SquaredEuclidean};

/// The vertices of a mesh in a k-d tree (`kiddo`), for nearest-vertex queries.
pub struct VertLocation<M: Tag> {
    // `None` for a mesh without vertices.
    tree: Option<ImmutableKdTree<f64, 3>>,
    // The vertex of every item of the tree (by its index).
    verts: Vec<VertKey<M>>,
}

impl<M: Tag> VertLocation<M> {
    /// The vertex nearest to the given point, with its squared distance. Panics for a mesh without vertices.
    #[must_use]
    pub fn nearest(&self, point: &[f64; 3]) -> (f64, VertKey<M>) {
        let tree = self.tree.as_ref().expect("a mesh with vertices");
        let nearest = tree
            .query(point)
            .nearest_one::<SquaredEuclidean<f64>>()
            .execute();
        (nearest.distance, self.verts[nearest.item as usize])
    }
}

impl<M: Tag> std::fmt::Debug for VertLocation<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VertLocation({} vertices)", self.verts.len())
    }
}

impl<M: Tag> Mesh<M> {
    /// The vertices in a k-d tree (see `VertLocation`).
    #[must_use]
    pub fn kdtree(&self) -> VertLocation<M> {
        let verts = self.vert_ids();
        let points = verts
            .iter()
            .map(|&v| self.position(v).into())
            .collect::<Vec<[f64; 3]>>();
        VertLocation {
            tree: ImmutableKdTree::new_from_slice(&points).ok(),
            verts,
        }
    }
}
