use super::*;
use petgraph::unionfind::UnionFind;
use std::collections::HashMap;
use std::collections::VecDeque;

/// A path's internal degree-two nodes can be eliminated exactly using its mask.
/// Neither sign means equality, one means strict order, both mean no constraint.
pub(crate) fn feasible(
    skeleton: &Skeleton,
    labels: &SecondaryMap<SkeletonEdgeID, EdgeLabel>,
    extra: Option<([SkeletonNodeID; 2], u8)>,
) -> bool {
    coordinate_levels(skeleton, labels, extra.as_slice(), false).is_some()
}

/// The same equalities/order DAG used for feasibility, with optional slack compaction.
pub(crate) fn coordinate_levels(
    skeleton: &Skeleton,
    labels: &SecondaryMap<SkeletonEdgeID, EdgeLabel>,
    extra: &[([SkeletonNodeID; 2], u8)],
    compact: bool,
) -> Option<SecondaryMap<SkeletonNodeID, [usize; 3]>> {
    let ids: HashMap<_, _> = skeleton
        .nodes
        .keys()
        .enumerate()
        .map(|(i, n)| (n, i))
        .collect();
    let mut coordinates = vec![[0; 3]; ids.len()];
    let mut relations = Vec::new();
    for (e, label) in labels {
        let Some(edge) = skeleton.edges.get(e) else {
            return None;
        };
        if edge.nodes.iter().any(|node| !ids.contains_key(node)) {
            return None;
        }
        relations.push((edge.nodes, Port::from_label(*label).bit()));
    }
    relations.extend(extra.iter().copied());
    if relations
        .iter()
        .any(|(nodes, _)| nodes.iter().any(|n| !ids.contains_key(n)))
    {
        return None;
    }
    for axis in 0..3 {
        let mut union = UnionFind::new(ids.len());
        for &(nodes, mask) in &relations {
            if (mask >> (axis * 2)) & 3 == 0 {
                union.union(ids[&nodes[0]], ids[&nodes[1]]);
            }
        }
        let mut outgoing = vec![Vec::new(); ids.len()];
        let mut degree = vec![0; ids.len()];
        for &(nodes, mask) in &relations {
            let signs = (mask >> (axis * 2)) & 3;
            if signs != 1 && signs != 2 {
                continue;
            }
            let mut a = union.find_mut(ids[&nodes[0]]);
            let mut b = union.find_mut(ids[&nodes[1]]);
            if signs == 2 {
                std::mem::swap(&mut a, &mut b);
            }
            if a == b {
                return None;
            }
            outgoing[a].push(b);
            degree[b] += 1;
        }
        let mut queue: VecDeque<_> = (0..ids.len()).filter(|&i| degree[i] == 0).collect();
        let mut order = Vec::new();
        let mut level = vec![0usize; ids.len()];
        while let Some(i) = queue.pop_front() {
            order.push(i);
            for &j in &outgoing[i] {
                level[j] = level[j].max(level[i] + 1);
                degree[j] -= 1;
                if degree[j] == 0 {
                    queue.push_back(j);
                }
            }
        }
        if order.len() != ids.len() {
            return None;
        }
        if compact {
            for &i in order.iter().rev() {
                if let Some(last) = outgoing[i].iter().map(|&j| level[j] - 1).min() {
                    level[i] = last;
                }
            }
        }
        for i in 0..ids.len() {
            coordinates[i][axis] = level[union.find_mut(i)];
        }
    }
    Some(
        ids.into_iter()
            .map(|(id, i)| (id, coordinates[i]))
            .collect(),
    )
}

pub(crate) fn used_ports(
    skeleton: &Skeleton,
    labels: &SecondaryMap<SkeletonEdgeID, EdgeLabel>,
) -> SkeletonResult<SecondaryMap<SkeletonNodeID, u8>> {
    let mut used = SecondaryMap::new();
    for n in skeleton.nodes.keys() {
        used.insert(n, 0);
    }
    for (e, label) in labels {
        let edge = skeleton
            .edges
            .get(e)
            .ok_or("Label refers to a missing edge")?;
        let port = Port::from_label(*label);
        for (n, p) in [(edge.nodes[0], port), (edge.nodes[1], port.opposite())] {
            let mask = used
                .get_mut(n)
                .ok_or("Labeled edge refers to a missing node")?;
            if *mask & p.bit() != 0 {
                return Err(format!("Repeated port {p:?} at {n:?}").into());
            }
            *mask |= p.bit();
        }
    }
    Ok(used)
}

impl LabeledSkeleton {
    /// Check surface correspondence, unique local ports, and positive-length
    /// coordinate feasibility. Global intersections and coincident nodes are allowed.
    pub fn validate(&self, mesh: &Mesh<INPUT>) -> SkeletonResult<()> {
        self.skeleton.validate(mesh)?;
        self.validate_labels()
    }

    pub fn validate_labels(&self) -> SkeletonResult<()> {
        if self.labels.len() != self.skeleton.edges.len()
            || self
                .skeleton
                .edges
                .keys()
                .any(|e| !self.labels.contains_key(e))
        {
            return Err("Every skeleton edge must have exactly one label".into());
        }
        used_ports(&self.skeleton, &self.labels)?;
        if !feasible(&self.skeleton, &self.labels, None) {
            return Err("Labels imply contradictory coordinate orders".into());
        }
        Ok(())
    }
}
