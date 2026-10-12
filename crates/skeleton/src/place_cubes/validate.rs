use super::*;
use std::collections::{HashMap, HashSet};

impl RealizedSkeleton {
    pub fn validate(&self, mesh: &Mesh<INPUT>) -> SkeletonResult<()> {
        self.labeled.skeleton.validate(mesh)?;
        self.validate_geometry()
    }

    /// Validate signed coordinates, one solid per element, and complete face connections.
    /// Incidental intersections and coincident solids do not imply adjacency.
    pub fn validate_geometry(&self) -> SkeletonResult<()> {
        self.labeled.validate_labels()?;
        let skeleton = &self.labeled.skeleton;
        if skeleton.nodes.is_empty()
            || self.node_positions.len() != skeleton.nodes.len()
            || skeleton
                .nodes
                .keys()
                .any(|id| !self.node_positions.contains_key(id))
            || self
                .node_positions
                .values()
                .any(|p| p.iter().any(|x| !x.is_finite()))
        {
            return Err("Invalid target node positions".into());
        }
        let mut nodes = HashMap::new();
        let mut edges = HashMap::new();
        for (id, cuboid) in &self.cuboids.cuboids {
            if cuboid.size.iter().any(|x| !x.is_finite() || *x <= 0.)
                || cuboid.center.iter().any(|x| !x.is_finite())
            {
                return Err("Cuboid geometry must be finite with positive dimensions".into());
            }
            match cuboid.owner {
                CuboidOwner::Node(node) => {
                    if !skeleton.nodes.contains_key(node)
                        || nodes.insert(node, id).is_some()
                        || cuboid.center != self.node_positions[node]
                        || cuboid.size.x != cuboid.size.y
                        || cuboid.size.x != cuboid.size.z
                    {
                        return Err("Invalid node cube ownership, center, or dimensions".into());
                    }
                }
                CuboidOwner::Edge(edge) => {
                    if !skeleton.edges.contains_key(edge) || edges.insert(edge, id).is_some() {
                        return Err("Invalid or duplicated edge cuboid owner".into());
                    }
                }
            }
        }
        if nodes.len() != skeleton.nodes.len() || edges.len() != skeleton.edges.len() {
            return Err("Each skeleton element requires exactly one cuboid".into());
        }
        let pair = |a, b| if a < b { [a, b] } else { [b, a] };
        let mut expected = HashSet::new();
        for (edge, data) in &skeleton.edges {
            let [a, b] = data.nodes;
            let port = crate::label::Port::from_label(self.labeled.labels[edge]) as usize;
            let axis = port / 2;
            let sign = if port % 2 == 0 { 1. } else { -1. };
            let start = self.node_positions[a];
            let end = self.node_positions[b];
            let distance = (end[axis] - start[axis]) * sign;
            if !distance.is_finite()
                || distance <= 0.
                || (0..3).any(|d| d != axis && end[d] != start[d])
            {
                return Err("Target coordinates violate a signed edge label".into());
            }
            let connection = edges[&edge];
            let cuboid = &self.cuboids.cuboids[connection];
            if cuboid.center != (start + end) / 2. {
                return Err("Edge cuboid must be centered between its endpoints".into());
            }
            for ids in [[nodes[&a], connection], [connection, nodes[&b]]] {
                let u = &self.cuboids.cuboids[ids[0]];
                let v = &self.cuboids.cuboids[ids[1]];
                if (v.center[axis] - u.center[axis]) * sign != (u.size[axis] + v.size[axis]) / 2.
                    || (0..3).any(|d| {
                        d != axis && (u.size[d] != v.size[d] || u.center[d] != v.center[d])
                    })
                {
                    return Err("Connected cuboids must share a complete face".into());
                }
                expected.insert(pair(ids[0], ids[1]));
            }
        }
        let mut actual = HashSet::new();
        for &[a, b] in &self.cuboids.connections {
            if a == b
                || !self.cuboids.cuboids.contains_key(a)
                || !self.cuboids.cuboids.contains_key(b)
                || !actual.insert(pair(a, b))
            {
                return Err("Invalid or duplicated cuboid connection".into());
            }
        }
        if actual != expected {
            return Err("Cuboid connections do not match skeleton endpoints".into());
        }
        Ok(())
    }
}
