use super::*;
use std::collections::HashMap;

pub(super) fn solve(
    skeleton: &Skeleton,
    costs: &SecondaryMap<SkeletonEdgeID, [f64; 6]>,
) -> SkeletonResult<(Vec<SkeletonEdgeID>, SecondaryMap<SkeletonEdgeID, EdgeLabel>)> {
    let nodes: Vec<_> = skeleton.nodes.keys().collect();
    if nodes.is_empty() {
        return Err("Skeleton is empty".into());
    }
    let index: HashMap<_, _> = nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();
    let mut degree = vec![0; nodes.len()];
    let mut edges: Vec<_> = skeleton.edges.keys().collect();
    for edge in skeleton.edges.values() {
        for n in edge.nodes {
            degree[index[&n]] += 1;
        }
    }
    if degree.iter().any(|&d| d > 6) {
        return Err(
            "Tree labeling requires at most six ports per junction; split oversized patches first"
                .into(),
        );
    }
    // TODO: Experiment with other MST weights than the best axis alignment.
    edges.sort_by(|a, b| {
        let min = |e| costs[e].iter().copied().fold(f64::INFINITY, f64::min);
        min(*a).total_cmp(&min(*b)).then_with(|| a.cmp(b))
    });
    let mut union = petgraph::unionfind::UnionFind::new(nodes.len());
    let mut tree = Vec::new();
    let mut adjacent = vec![Vec::new(); nodes.len()];
    for e in edges {
        let [a, b] = skeleton.edges[e].nodes.map(|n| index[&n]);
        if union.union(a, b) {
            tree.push(e);
            adjacent[a].push((b, e));
            adjacent[b].push((a, e));
        }
    }
    if tree.len() + 1 != nodes.len() {
        return Err("Skeleton is disconnected".into());
    }
    let mut order = vec![0];
    let mut parent = vec![usize::MAX; nodes.len()];
    parent[0] = 0;
    let mut children = vec![Vec::new(); nodes.len()];
    let mut cursor = 0;
    while cursor < order.len() {
        let n = order[cursor];
        cursor += 1;
        for &(child, edge) in &adjacent[n] {
            if parent[child] != usize::MAX {
                continue;
            }
            parent[child] = n;
            children[n].push((child, edge));
            order.push(child);
        }
    }
    // State 0..6 is the occupied parent port; 6 is the root without a parent.
    let mut values = vec![[f64::INFINITY; 7]; nodes.len()];
    let mut choices = vec![vec![Vec::new(); 7]; nodes.len()];
    for &n in order.iter().rev() {
        for incoming in 0..7 {
            let occupied = if incoming < 6 { 1 << incoming } else { 0 };
            let (best, selected) = assign_ports(children[n].len(), occupied, |i, p, value| {
                let (child, edge) = children[n][i];
                let label = if skeleton.edges[edge].nodes[0] == nodes[n] {
                    p
                } else {
                    p.opposite()
                };
                value + costs[edge][label as usize] + values[child][p.opposite() as usize]
            });
            values[n][incoming] = best;
            choices[n][incoming] = selected;
        }
    }
    if !values[0][6].is_finite() {
        return Err("Tree has no port-feasible labeling".into());
    }
    let mut labels = SecondaryMap::new();
    let mut stack = vec![(0, 6)];
    while let Some((n, incoming)) = stack.pop() {
        for (&(child, edge), &p) in children[n].iter().zip(&choices[n][incoming]) {
            labels.insert(
                edge,
                if skeleton.edges[edge].nodes[0] == nodes[n] {
                    p
                } else {
                    p.opposite()
                }
                .label(),
            );
            stack.push((child, (p as usize) ^ 1));
        }
    }
    Ok((tree, labels))
}
