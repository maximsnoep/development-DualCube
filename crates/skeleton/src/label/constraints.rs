use super::*;
use std::collections::{HashMap, VecDeque};

pub(super) struct Union(pub Vec<usize>);
impl Union {
    pub fn new(n: usize) -> Self {
        Self((0..n).collect())
    }
    pub fn root(&mut self, i: usize) -> usize {
        let mut root = i;
        while self.0[root] != root {
            root = self.0[root];
        }
        let mut current = i;
        while self.0[current] != current {
            let next = self.0[current];
            self.0[current] = root;
            current = next;
        }
        root
    }
    pub fn join(&mut self, a: usize, b: usize) -> bool {
        let (a, b) = (self.root(a), self.root(b));
        if a == b {
            return false;
        }
        self.0[a] = b;
        true
    }
}

/// A path's internal degree-two nodes can be eliminated exactly using its mask.
/// Neither sign means equality, one means strict order, both mean no constraint.
pub(super) fn feasible(
    skeleton: &Skeleton,
    labels: &SecondaryMap<SkeletonEdgeID, EdgeLabel>,
    extra: Option<([SkeletonNodeID; 2], u8)>,
) -> bool {
    let ids: HashMap<_, _> = skeleton
        .nodes
        .keys()
        .enumerate()
        .map(|(i, n)| (n, i))
        .collect();
    let mut relations = Vec::new();
    for (e, label) in labels {
        let Some(edge) = skeleton.edges.get(e) else {
            return false;
        };
        relations.push((edge.nodes, Port::from_label(*label).bit()));
    }
    relations.extend(extra);
    for axis in 0..3 {
        let mut union = Union::new(ids.len());
        for &(nodes, mask) in &relations {
            if (mask >> (axis * 2)) & 3 == 0 {
                union.join(ids[&nodes[0]], ids[&nodes[1]]);
            }
        }
        let mut outgoing = vec![Vec::new(); ids.len()];
        let mut degree = vec![0; ids.len()];
        for &(nodes, mask) in &relations {
            let signs = (mask >> (axis * 2)) & 3;
            if signs != 1 && signs != 2 {
                continue;
            }
            let mut a = union.root(ids[&nodes[0]]);
            let mut b = union.root(ids[&nodes[1]]);
            if signs == 2 {
                std::mem::swap(&mut a, &mut b);
            }
            if a == b {
                return false;
            }
            outgoing[a].push(b);
            degree[b] += 1;
        }
        let mut queue: VecDeque<_> = (0..ids.len()).filter(|&i| degree[i] == 0).collect();
        let mut visited = 0;
        while let Some(i) = queue.pop_front() {
            visited += 1;
            for &j in &outgoing[i] {
                degree[j] -= 1;
                if degree[j] == 0 {
                    queue.push_back(j);
                }
            }
        }
        if visited != ids.len() {
            return false;
        }
    }
    true
}

pub(super) fn used_ports(
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
