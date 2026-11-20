use super::*;

impl RealizedSkeleton {
    pub fn validate(&self, mesh: &Mesh<INPUT>) -> SkeletonResult<()> {
        self.labeled.skeleton.validate(mesh)?;
        self.validate_geometry()
    }

    /// Validate labels and one finite position per node, with every edge running along its
    /// signed axis far enough for a positive cuboid between the unit node cubes.
    pub fn validate_geometry(&self) -> SkeletonResult<()> {
        self.labeled.validate_labels()?;
        let skeleton = &self.labeled.skeleton;
        if skeleton.nodes.is_empty()
            || self.node_positions.len() != skeleton.nodes.len()
            || skeleton.nodes.keys().any(|id| {
                self.node_positions
                    .get(id)
                    .is_none_or(|p| p.iter().any(|x| !x.is_finite()))
            })
        {
            return Err("Invalid target node positions".into());
        }
        for (edge, data) in &skeleton.edges {
            let port = crate::label::Port::from_label(self.labeled.labels[edge]);
            let axis = port.axis();
            let sign = if port.is_positive() { 1. } else { -1. };
            let [start, end] = data.nodes.map(|n| self.node_positions[n]);
            if (end[axis] - start[axis]) * sign <= 1.
                || (0..3).any(|d| d != axis && end[d] != start[d])
            {
                return Err("Target coordinates violate a signed edge label".into());
            }
        }
        Ok(())
    }
}
