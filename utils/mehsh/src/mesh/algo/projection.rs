use crate::prelude::*;

impl<M: Tag> Mesh<M> {
    // Project a face to 2d
    #[must_use]
    pub fn project(&self, face: FaceKey<M>) -> Vec<Vector2D> {
        // TODO: Find a better plane
        let normal = self.normal(face);
        let edge = self.vector(self.edges(face).next().unwrap());
        let plane = (edge.normalize(), edge.cross(&normal).normalize());
        let reference = self.position(self.vertices(face).next().unwrap());

        self.vertices(face)
            .map(|v| geom::project_point_onto_plane(self.position(v), plane, reference))
            .collect()
    }
}
