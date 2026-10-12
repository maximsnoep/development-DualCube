//! A deliberately conservative surface split: peel a connected collar off an endpoint.

use super::*;
use crate::{PatchBoundary, SkeletonEdge, SkeletonNode};
use dualcube_types::prelude::HasNeighbors;
use std::collections::{HashMap, HashSet, VecDeque};

/// Return the first valid graph-distance collar on this side of the boundary.
/// Failed trials only mutate clones; original ownership and IDs remain untouched.
pub(super) fn collar(
    mesh: &Mesh<INPUT>,
    skeleton: &Skeleton,
    edge: SkeletonEdgeID,
    side: usize,
) -> SkeletonResult<Option<(Skeleton, SkeletonEdgeID)>> {
    let [a, b] = skeleton.edges[edge].nodes;
    let owner = [a, b][side];
    let vertices = &skeleton.nodes[owner].patch_vertices;
    let owned: HashSet<_> = vertices.iter().copied().collect();
    let mut distance = HashMap::new();
    let mut queue = VecDeque::new();
    for &e in &skeleton.edges[edge].boundary.edge_midpoints {
        for v in [mesh.root(e), mesh.toor(e)] {
            if owned.contains(&v) && distance.insert(v, 0).is_none() {
                queue.push_back(v);
            }
        }
    }
    while let Some(v) = queue.pop_front() {
        let next = distance[&v] + 1;
        for w in mesh.neighbors(v) {
            if owned.contains(&w) && !distance.contains_key(&w) {
                distance.insert(w, next);
                queue.push_back(w);
            }
        }
    }
    let max = distance.values().copied().max().unwrap_or(0);
    for depth in 0..max {
        let mut candidate = skeleton.clone();
        let (collar, rest): (Vec<_>, Vec<_>) =
            vertices.iter().copied().partition(|v| distance[v] <= depth);
        if collar.is_empty() || rest.is_empty() {
            continue;
        }
        candidate.nodes[owner].patch_vertices = rest;
        let fraction = (depth + 1) as f64 / (max + 1) as f64;
        let other = [a, b][1 - side];
        let position = skeleton.nodes[owner].position * (0.5 + 0.5 * fraction)
            + skeleton.nodes[other].position * (0.5 - 0.5 * fraction);
        let middle = candidate.nodes.insert(SkeletonNode {
            position,
            patch_vertices: collar,
        });
        candidate.edges[edge].nodes = [a, middle];
        let added = candidate.edges.insert(SkeletonEdge {
            nodes: [middle, b],
            boundary: PatchBoundary {
                edge_midpoints: Vec::new(),
            },
        });
        // Retain the old boundary on the edge that still crosses it.
        if side == 0 {
            candidate.edges[added].boundary = skeleton.edges[edge].boundary.clone();
            candidate.edges[edge].boundary.edge_midpoints.clear();
        }
        if crate::validate::assign_boundaries(&mut candidate, mesh).is_ok()
            && candidate.validate(mesh).is_ok()
        {
            return Ok(Some((candidate, added)));
        }
    }
    Ok(None)
}
