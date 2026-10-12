use super::*;
use crate::{CubeID, SkeletonEdgeID};
use std::collections::{HashMap, HashSet};

impl RealizedSkeleton {
    pub fn validate(&self, mesh: &Mesh<INPUT>) -> SkeletonResult<()> {
        self.labeled.skeleton.validate(mesh)?;
        self.validate_geometry()
    }

    /// Validate labels, signed coordinates, ownership, and intended face adjacencies.
    /// Incidental intersections, including coincident cubes, do not imply adjacency.
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
        let mut chains = HashMap::<SkeletonEdgeID, Vec<CubeID>>::new();
        for (id, cube) in &self.cubes.cubes {
            if !cube.size.is_finite()
                || cube.size <= 0.
                || cube.center.iter().any(|x| !x.is_finite())
            {
                return Err("Cube geometry must be finite with positive size".into());
            }
            match cube.owner {
                CubeOwner::Node(node) => {
                    if !skeleton.nodes.contains_key(node)
                        || nodes.insert(node, id).is_some()
                        || cube.center != self.node_positions[node]
                    {
                        return Err("Invalid node cube ownership or center".into());
                    }
                }
                CubeOwner::Edge { edge, patch } => {
                    if !skeleton
                        .edges
                        .get(edge)
                        .is_some_and(|e| e.nodes.contains(&patch))
                    {
                        return Err("Invalid edge cube owner".into());
                    }
                    chains.entry(edge).or_default().push(id);
                }
                CubeOwner::Midpoint(edge) => {
                    if !skeleton.edges.contains_key(edge) {
                        return Err("Invalid midpoint cube owner".into());
                    }
                    chains.entry(edge).or_default().push(id);
                }
            }
        }
        if nodes.len() != skeleton.nodes.len() {
            return Err("Missing node cube".into());
        }
        let pair = |a, b| if a < b { [a, b] } else { [b, a] };
        let mut expected = HashSet::new();
        for (edge, data) in &skeleton.edges {
            let [a, b] = data.nodes;
            let label = self.labeled.labels[edge];
            let port = crate::label::Port::from_label(label) as usize;
            let axis = port / 2;
            let sign = if port % 2 == 0 { 1. } else { -1. };
            let start = self.node_positions[a];
            let end = self.node_positions[b];
            let distance = (end[axis] - start[axis]) * sign;
            if distance <= 0. || (0..3).any(|d| d != axis && end[d] != start[d]) {
                return Err("Target coordinates violate a signed edge label".into());
            }
            let mut chain = chains.remove(&edge).unwrap_or_default();
            chain.sort_by(|&a, &b| {
                (self.cubes.cubes[a].center[axis] * sign)
                    .total_cmp(&(self.cubes.cubes[b].center[axis] * sign))
            });
            let mut midpoints = 0;
            for &id in &chain {
                let cube = &self.cubes.cubes[id];
                let t = (cube.center[axis] - start[axis]) * sign;
                if t <= 0.
                    || t >= distance
                    || (0..3).any(|d| d != axis && cube.center[d] != start[d])
                {
                    return Err("Edge cube is not inside its axis segment".into());
                }
                match cube.owner {
                    CubeOwner::Midpoint(_) => {
                        midpoints += 1;
                        if t != distance / 2. {
                            return Err("Midpoint cube is not centered on its edge".into());
                        }
                    }
                    CubeOwner::Edge { patch, .. } => {
                        if t == distance / 2. || patch != if t < distance / 2. { a } else { b } {
                            return Err("Edge cube belongs to the wrong endpoint patch".into());
                        }
                    }
                    _ => unreachable!(),
                }
            }
            if midpoints != 1 {
                return Err("Each edge requires exactly one midpoint cube".into());
            }
            chain.insert(0, nodes[&a]);
            chain.push(nodes[&b]);
            for cubes in chain.windows(2) {
                let u = &self.cubes.cubes[cubes[0]];
                let v = &self.cubes.cubes[cubes[1]];
                if u.size != v.size || (v.center[axis] - u.center[axis]) * sign != u.size {
                    return Err("Connected cubes must share a complete face".into());
                }
                expected.insert(pair(cubes[0], cubes[1]));
            }
        }
        let mut actual = HashSet::new();
        for &[a, b] in &self.cubes.connections {
            if a == b
                || !self.cubes.cubes.contains_key(a)
                || !self.cubes.cubes.contains_key(b)
                || !actual.insert(pair(a, b))
            {
                return Err("Invalid or duplicated cube connection".into());
            }
        }
        if actual != expected {
            return Err("Cube connections do not match skeleton edge chains".into());
        }
        Ok(())
    }
}
