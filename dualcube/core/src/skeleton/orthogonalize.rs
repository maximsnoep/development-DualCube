use log::{error, warn};
use petgraph::graph::{EdgeIndex, NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};

use std::collections::{HashMap, HashSet, VecDeque};

use crate::{
    prelude::{CurveSkeleton, PrincipalDirection},
    skeleton::curve_skeleton::SkeletonNode,
};

/// A 3-dimensional integer vector.
pub type IVector3D = nalgebra::Vector3<i32>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrthogonalSkeletonNode {
    pub skeleton_node: SkeletonNode,
    pub grid_position: IVector3D,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrthogonalSkeletonEdge {
    pub direction: PrincipalDirection,
    pub length: u32,
}
/// Curve skeleton with axis-aligned edges and integer grid coordinates assigned to each node.
pub type LabeledCurveSkeleton = UnGraph<OrthogonalSkeletonNode, OrthogonalSkeletonEdge>;

/// Curve skeleton where every edge has an axis label, but node coordinates are not yet assigned.
type EdgeLabeledCurveSkeleton = UnGraph<SkeletonNode, OrthogonalSkeletonEdge>;

/// Curve skeleton where edges are either labeled with an axis (`Some`) or not yet assigned (`None`).
/// Used while building up a labeling incrementally.
type PartialEdgeLabeledCurveSkeleton = UnGraph<SkeletonNode, Option<PrincipalDirection>>;

/// Gives an ordering to axes based on the displacement between the edge endpoints, from most aligned to least.
fn preferred_axes_from_displacement(disp: nalgebra::Vector3<f64>) -> [PrincipalDirection; 3] {
    let mut entries = [
        (PrincipalDirection::X, disp.x.abs()),
        (PrincipalDirection::Y, disp.y.abs()),
        (PrincipalDirection::Z, disp.z.abs()),
    ];
    entries.sort_by(|(axis_a, value_a), (axis_b, value_b)| {
        value_b
            .partial_cmp(value_a)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then((*axis_a as usize).cmp(&(*axis_b as usize)))
    });
    [entries[0].0, entries[1].0, entries[2].0]
}

/// Returns the component of a 3‑D vector along the given principal axis.
fn axis_value(v: nalgebra::Vector3<f64>, axis: PrincipalDirection) -> f64 {
    match axis {
        PrincipalDirection::X => v.x,
        PrincipalDirection::Y => v.y,
        PrincipalDirection::Z => v.z,
    }
}

// TODO: REMOVE IN FAVOR OF PROPER 1 DIR + SIDE PER NODE PER EDGE
fn axis_count_around_node(
    partial: &PartialEdgeLabeledCurveSkeleton,
    node: NodeIndex,
    axis: PrincipalDirection,
) -> usize {
    partial
        .edges(node)
        .filter_map(|edge| *edge.weight())
        .filter(|&dir| dir == axis)
        .count()
}

/// Build a working copy of the skeleton with all edges unlabeled.
///
/// Returns the partial graph plus two maps: one mapping original nodes to the
/// new ones, and a reverse map.
fn build_unlabeled_partial(
    curve_skeleton: &CurveSkeleton,
) -> (
    PartialEdgeLabeledCurveSkeleton,
    HashMap<NodeIndex, NodeIndex>,
    HashMap<NodeIndex, NodeIndex>,
) {
    let mut partial = PartialEdgeLabeledCurveSkeleton::new_undirected();
    let mut node_map: HashMap<NodeIndex, NodeIndex> = HashMap::new();
    let mut reverse_node_map: HashMap<NodeIndex, NodeIndex> = HashMap::new();

    for node in curve_skeleton.node_indices() {
        let w = curve_skeleton.node_weight(node).unwrap().clone();
        let out_n = partial.add_node(w);
        node_map.insert(node, out_n);
        reverse_node_map.insert(out_n, node);
    }

    for edge in curve_skeleton.edge_indices() {
        let (a, b) = curve_skeleton.edge_endpoints(edge).unwrap();
        partial.add_edge(node_map[&a], node_map[&b], None);
    }

    (partial, node_map, reverse_node_map)
}

/// Collects all node indices and builds a map from NodeIndex to dense vector index.
fn build_node_list_and_index_map<N, E>(
    graph: &UnGraph<N, E>,
) -> (Vec<NodeIndex>, HashMap<NodeIndex, usize>) {
    let nodes: Vec<_> = graph.node_indices().collect();
    let mut idx_map: HashMap<_, _> = HashMap::new();
    for (i, n) in nodes.iter().enumerate() {
        idx_map.insert(*n, i);
    }
    (nodes, idx_map)
}

/// Generic component labelling helper.
///
/// Traverses the graph using BFS, only following allowed edges.
fn components_excluding_with<N, E, F>(
    g: &UnGraph<N, E>,
    nodes: &[NodeIndex],
    idx_map: &HashMap<NodeIndex, usize>,
    mut edge_allowed: F,
) -> Vec<usize>
where
    F: FnMut(&E) -> bool,
{
    let mut comp = vec![usize::MAX; nodes.len()];
    let mut cur = 0usize;

    for &start in nodes {
        let si = idx_map[&start];
        if comp[si] != usize::MAX {
            continue;
        }

        let mut q = VecDeque::new();
        q.push_back(start);
        comp[si] = cur;

        while let Some(u) = q.pop_front() {
            for edge in g.edges(u) {
                if !edge_allowed(edge.weight()) {
                    continue;
                }

                let v = if edge.source() == u {
                    edge.target()
                } else {
                    edge.source()
                };
                let vi = idx_map[&v];
                if comp[vi] == usize::MAX {
                    comp[vi] = cur;
                    q.push_back(v);
                }
            }
        }

        cur += 1;
    }

    comp
}

/// Connected components in a partial labeling while excluding a given axis.
///
/// Allowed traversal edges are labeled edges whose axis is not `forbidden`.
/// Unlabeled edges are ignored.
fn components_excluding_partial(
    g: &PartialEdgeLabeledCurveSkeleton,
    forbidden: PrincipalDirection,
    nodes: &[NodeIndex],
    idx_map: &HashMap<NodeIndex, usize>,
) -> Vec<usize> {
    components_excluding_with(
        g,
        nodes,
        idx_map,
        |w| matches!(*w, Some(dir) if dir != forbidden),
    )
}

/// Connected components in a fully labeled skeleton while excluding a given axis.
fn components_excluding_labeled(
    g: &EdgeLabeledCurveSkeleton,
    forbidden: PrincipalDirection,
    nodes: &[NodeIndex],
    idx_map: &HashMap<NodeIndex, usize>,
) -> Vec<usize> {
    components_excluding_with(g, nodes, idx_map, |w| w.direction != forbidden)
}

/// Initially all D-components have their own coordinate. We use a DAG to compress these coordinates as much as possible.
fn rank_and_compress_components(
    skeleton: &EdgeLabeledCurveSkeleton,
    axis: PrincipalDirection,
    comps: &[usize],
    nodes: &[NodeIndex],
    idx_map: &HashMap<NodeIndex, usize>,
) -> Vec<i32> {
    let nr_components = comps.iter().copied().max().map_or(0, |m| m + 1);

    // Get mean coordinate per components
    let mut sums: Vec<f64> = vec![0.0; nr_components];
    let mut counts: Vec<usize> = vec![0; nr_components];
    for (i, &node) in nodes.iter().enumerate() {
        let c = comps[i];
        sums[c] += axis_value(skeleton[node].position, axis);
        counts[c] += 1;
    }
    let means: Vec<f64> = (0..nr_components)
        .map(|c| {
            if counts[c] == 0 {
                0.0
            } else {
                sums[c] / counts[c] as f64
            }
        })
        .collect();

    // Sort components by mean
    let mut component_ids: Vec<usize> = (0..nr_components).collect();
    component_ids.sort_by(|&a, &b| {
        means[a]
            .partial_cmp(&means[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });

    let mut sorted_index = vec![0; nr_components];
    for (i, &cid) in component_ids.iter().enumerate() {
        sorted_index[cid] = i;
    }

    // Build adjacency in sorted space
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); nr_components];
    for e in skeleton.edge_indices() {
        if let Some(w) = skeleton.edge_weight(e) {
            if w.direction == axis {
                let (u, v) = skeleton.edge_endpoints(e).unwrap();
                let cu = comps[idx_map[&u]];
                let cv = comps[idx_map[&v]];
                if cu != cv {
                    let iu = sorted_index[cu];
                    let iv = sorted_index[cv];
                    if iu < iv {
                        adjacency[iu].push(iv);
                    } else if iv < iu {
                        adjacency[iv].push(iu);
                    }
                }
            }
        }
    }

    // Propagate coordinates
    let mut coord_per_sorted = vec![0i32; nr_components];
    for i in 0..nr_components {
        for &j in &adjacency[i] {
            if coord_per_sorted[j] < coord_per_sorted[i] + 1 {
                coord_per_sorted[j] = coord_per_sorted[i] + 1;
            }
        }
    }

    comps
        .iter()
        .map(|&cid| coord_per_sorted[sorted_index[cid]])
        .collect()
}

/// Converts a fully labeled skeleton to partial form, so we method can be reused.
fn partial_from_edge_labeled(s: &EdgeLabeledCurveSkeleton) -> PartialEdgeLabeledCurveSkeleton {
    let mut out = PartialEdgeLabeledCurveSkeleton::new_undirected();
    let mut map: HashMap<NodeIndex, NodeIndex> = HashMap::new();

    for n in s.node_indices() {
        let w = s.node_weight(n).unwrap().clone();
        let on = out.add_node(w);
        map.insert(n, on);
    }

    for e in s.edge_references() {
        let a = e.source();
        let b = e.target();
        let dir = e.weight().direction;
        out.add_edge(map[&a], map[&b], Some(dir));
    }

    out
}

/// Realizes a partial labeling as a fully labeled skeleton.
fn finalize_partial_to_realized(
    partial: &PartialEdgeLabeledCurveSkeleton,
) -> Option<LabeledCurveSkeleton> {
    for edge_ref in partial.edge_references() {
        if edge_ref.weight().is_none() {
            error!(
                "edge from {:?} to {:?} is still unlabeled after assignment, this should not happen",
                edge_ref.source(),
                edge_ref.target()
            );
            return None;
        }
    }

    let mut full = EdgeLabeledCurveSkeleton::new_undirected();
    let mut out_map: HashMap<NodeIndex, NodeIndex> = HashMap::new();
    for n in partial.node_indices() {
        let w = partial.node_weight(n).unwrap().clone();
        let out_n = full.add_node(w);
        out_map.insert(n, out_n);
    }
    for e in partial.edge_references() {
        let a = e.source();
        let b = e.target();
        let dir = match *e.weight() {
            Some(d) => d,
            None => unreachable!(),
        };
        full.add_edge(
            out_map[&a],
            out_map[&b],
            OrthogonalSkeletonEdge {
                direction: dir,
                length: 1,
            },
        );
    }

    realize(&full)
}

/// Global search for labelings that maximize preferred-axis matches.
///
/// Edge priority is weighted by patch size and directional confidence.
fn backtracking_orthogonalization(curve_skeleton: &CurveSkeleton) -> Option<LabeledCurveSkeleton> {
    let (mut partial, _node_map, reverse_node_map) = build_unlabeled_partial(curve_skeleton);

    warn!("Greedy orthogonalization failed, falling back to backtracking search.");

    #[derive(Clone)]
    struct EdgePlan {
        edge: EdgeIndex,
        primary: PrincipalDirection,
        preference_weight: i64,
        confidence: f64,
    }

    let mut plans: Vec<EdgePlan> = partial
        .edge_indices()
        .map(|edge| {
            let (u, v) = partial.edge_endpoints(edge).unwrap();
            let orig_u = reverse_node_map[&u];
            let orig_v = reverse_node_map[&v];
            let disp = curve_skeleton[orig_v].position - curve_skeleton[orig_u].position;
            let pref = preferred_axes_from_displacement(disp);
            let preference_weight = std::cmp::max(
                curve_skeleton[orig_u].patch_vertices.len(),
                curve_skeleton[orig_v].patch_vertices.len(),
            ) as i64;
            let confidence = {
                let mut mags = [disp.x.abs(), disp.y.abs(), disp.z.abs()];
                mags.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                mags[0] - mags[1]
            };
            EdgePlan {
                edge,
                primary: pref[0],
                preference_weight,
                confidence,
            }
        })
        .collect();

    plans.sort_by(|a, b| {
        b.preference_weight.cmp(&a.preference_weight).then(
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal),
        )
    });

    let mut suffix_max_possible = vec![0i64; plans.len() + 1];
    for i in (0..plans.len()).rev() {
        suffix_max_possible[i] = suffix_max_possible[i + 1] + plans[i].preference_weight;
    }

    let mut best_score = i64::MIN;
    let mut best_labels: Option<HashMap<EdgeIndex, PrincipalDirection>> = None;

    fn dfs(
        depth: usize,
        plans: &[EdgePlan],
        partial: &mut PartialEdgeLabeledCurveSkeleton,
        curve_skeleton: &CurveSkeleton,
        reverse_node_map: &HashMap<NodeIndex, NodeIndex>,
        current_score: i64,
        best_score: &mut i64,
        best_labels: &mut Option<HashMap<EdgeIndex, PrincipalDirection>>,
        suffix_max_possible: &[i64],
    ) {
        if current_score + suffix_max_possible[depth] < *best_score {
            return;
        }

        if depth == plans.len() {
            if !is_partially_realizable(partial) {
                return;
            }

            if current_score > *best_score {
                let labels = partial
                    .edge_indices()
                    .map(|e| (e, partial.edge_weight(e).copied().flatten().unwrap()))
                    .collect();
                *best_score = current_score;
                *best_labels = Some(labels);
            }
            return;
        }

        let plan = &plans[depth];
        let (u, v) = partial.edge_endpoints(plan.edge).unwrap();
        let orig_u = reverse_node_map[&u];
        let orig_v = reverse_node_map[&v];
        let disp = curve_skeleton[orig_v].position - curve_skeleton[orig_u].position;
        let candidate_axes = preferred_axes_from_displacement(disp);

        for cand in candidate_axes {
            if axis_count_around_node(partial, u, cand) >= 2
                || axis_count_around_node(partial, v, cand) >= 2
            {
                continue;
            }

            *partial.edge_weight_mut(plan.edge).unwrap() = Some(cand);
            if is_partially_realizable(partial) {
                let next_score = current_score
                    + if cand == plan.primary {
                        plan.preference_weight
                    } else {
                        0
                    };
                dfs(
                    depth + 1,
                    plans,
                    partial,
                    curve_skeleton,
                    reverse_node_map,
                    next_score,
                    best_score,
                    best_labels,
                    suffix_max_possible,
                );
            }
            *partial.edge_weight_mut(plan.edge).unwrap() = None;
        }
    }

    dfs(
        0,
        &plans,
        &mut partial,
        curve_skeleton,
        &reverse_node_map,
        0,
        &mut best_score,
        &mut best_labels,
        &suffix_max_possible,
    );

    if let Some(labels) = best_labels {
        for (edge, dir) in labels {
            *partial.edge_weight_mut(edge).unwrap() = Some(dir);
        }
        finalize_partial_to_realized(&partial)
    } else {
        None
    }
}

/// Returns true if the partial edge-labeling is still consistent with some valid orthogonal
/// integer-grid embedding. Unlabeled edges are ignored.
///
/// Two conditions are checked. First, for each axis D, no D-labeled edge may connect two
/// nodes in the same D-component (connected component with D-edges removed) — that would
/// force the endpoints to share their D-coordinate while the edge demands they differ.
/// Second, no two nodes may share the same (cx, cy, cz) component-ID triple, since that
/// would place them on the same grid point.
pub fn is_partially_realizable(p: &PartialEdgeLabeledCurveSkeleton) -> bool {
    let (nodes, idx_map) = build_node_list_and_index_map(p);

    // Check condition 1: no D-edge connects two nodes already in the same D-component.
    for &d in &[
        PrincipalDirection::X,
        PrincipalDirection::Y,
        PrincipalDirection::Z,
    ] {
        let comps = components_excluding_partial(p, d, &nodes, &idx_map);
        for edge in p.edge_references() {
            if let Some(dir) = *edge.weight() {
                if dir != d {
                    continue;
                }
                let a = edge.source();
                let b = edge.target();
                let ia = idx_map[&a];
                let ib = idx_map[&b];
                if comps[ia] == comps[ib] {
                    return false;
                }
            }
        }
    }

    // Check condition 2: all (cx, cy, cz) triples must be distinct.
    let cx = components_excluding_partial(p, PrincipalDirection::X, &nodes, &idx_map);
    let cy = components_excluding_partial(p, PrincipalDirection::Y, &nodes, &idx_map);
    let cz = components_excluding_partial(p, PrincipalDirection::Z, &nodes, &idx_map);

    let mut seen = HashSet::new();
    for i in 0..nodes.len() {
        let triple = (cx[i], cy[i], cz[i]);
        if !seen.insert(triple) {
            return false;
        }
    }

    true
}

/// Assigns integer grid coordinates and recomputes axis-aligned edge lengths.
pub fn realize(s: &EdgeLabeledCurveSkeleton) -> Option<LabeledCurveSkeleton> {
    // Convert to partial form once so realizability and coordinate assignment
    // use the same edge semantics.
    let partial = partial_from_edge_labeled(s);
    if !is_partially_realizable(&partial) {
        return None;
    }

    let (nodes, idx_map) = build_node_list_and_index_map(s);

    // Determine the component ID of each node when all edges of the current
    // axis are removed.  the subsequent call computes compressed coordinates
    // that still respect the adjacency constraints.
    let comp_x = components_excluding_labeled(s, PrincipalDirection::X, &nodes, &idx_map);
    let comp_y = components_excluding_labeled(s, PrincipalDirection::Y, &nodes, &idx_map);
    let comp_z = components_excluding_labeled(s, PrincipalDirection::Z, &nodes, &idx_map);

    let cx = rank_and_compress_components(s, PrincipalDirection::X, &comp_x, &nodes, &idx_map);
    let cy = rank_and_compress_components(s, PrincipalDirection::Y, &comp_y, &nodes, &idx_map);
    let cz = rank_and_compress_components(s, PrincipalDirection::Z, &comp_z, &nodes, &idx_map);

    // Assemble coordinate vector for each node.
    let mut coords: Vec<IVector3D> = Vec::with_capacity(nodes.len());
    for i in 0..nodes.len() {
        coords.push(IVector3D::new(cx[i], cy[i], cz[i]));
    }

    // Build the output graph: copy nodes (with computed coordinates) and edges.
    let mut out = LabeledCurveSkeleton::new_undirected();
    let mut out_index_map: HashMap<NodeIndex, NodeIndex> = HashMap::new();

    for &orig in &nodes {
        let node = s.node_weight(orig).unwrap().clone();
        let weight = OrthogonalSkeletonNode {
            skeleton_node: node,
            grid_position: IVector3D::new(0, 0, 0),
        };
        let out_n = out.add_node(weight);
        out_index_map.insert(orig, out_n);
    }

    for e in s.edge_indices() {
        if let Some(w) = s.edge_weight(e) {
            let (a, b) = s.edge_endpoints(e).unwrap();
            out.add_edge(
                out_index_map[&a],
                out_index_map[&b],
                OrthogonalSkeletonEdge {
                    direction: w.direction,
                    length: w.length,
                },
            );
        }
    }

    // Write integer coordinates into each node.
    for &orig in &nodes {
        let i = idx_map[&orig];
        let out_n = out_index_map[&orig];
        if let Some(node_w) = out.node_weight_mut(out_n) {
            node_w.grid_position = coords[i];
        }
    }

    // Recompute edge lengths from the assigned coordinates.
    let mut rev_out_map: HashMap<NodeIndex, NodeIndex> = HashMap::new();
    for (orig, outn) in out_index_map.iter() {
        rev_out_map.insert(*outn, *orig);
    }

    for e in out.edge_indices() {
        let (a, b) = out.edge_endpoints(e).unwrap();
        let orig_a = rev_out_map[&a];
        let orig_b = rev_out_map[&b];
        let ia = idx_map[&orig_a];
        let ib = idx_map[&orig_b];

        if let Some(edge_w) = out.edge_weight_mut(e) {
            let dir = edge_w.direction;
            let diff = match dir {
                PrincipalDirection::X => (coords[ia].x - coords[ib].x).abs() as u32,
                PrincipalDirection::Y => (coords[ia].y - coords[ib].y).abs() as u32,
                PrincipalDirection::Z => (coords[ia].z - coords[ib].z).abs() as u32,
            };
            assert!(
                diff > 0,
                "edge of direction {:?} has zero length after realization",
                dir
            );
            edge_w.length = diff;
        }
    }

    Some(out)
}

/// Assigns axis labels to edges and realizes the orthogonal skeleton.
///
/// BFS starts from the highest-degree node (secondarily largest patch).
/// For each unlabeled edge, the axis most aligned with the geometric displacement
/// is tried first; the other two axes are fallbacks. The first choice that keeps the partial
/// labeling consistent is accepted.
///
/// If greedy gets stuck, a global backtracking search is used.
pub fn greedy_orthogonalization(curve_skeleton: &CurveSkeleton) -> Option<LabeledCurveSkeleton> {
    let (mut partial, node_map, _reverse_node_map) = build_unlabeled_partial(curve_skeleton);

    // Helper to pick the best (highest-degree, then largest patch) unvisited node.
    let pick_best_unvisited = |visited: &HashSet<NodeIndex>| {
        let mut best: Option<NodeIndex> = None;
        for n in curve_skeleton.node_indices() {
            if visited.contains(&n) {
                continue;
            }
            if let Some(bn) = best {
                let deg_n = curve_skeleton.neighbors(n).count();
                let deg_b = curve_skeleton.neighbors(bn).count();
                if deg_n > deg_b {
                    best = Some(n);
                } else if deg_n == deg_b {
                    if curve_skeleton[n].patch_vertices.len()
                        > curve_skeleton[bn].patch_vertices.len()
                    {
                        best = Some(n);
                    }
                }
            } else {
                best = Some(n);
            }
        }
        best
    };

    let mut visited: HashSet<NodeIndex> = HashSet::new();
    let mut queue: VecDeque<NodeIndex> = VecDeque::new();

    // Process all
    while visited.len() < curve_skeleton.node_count() {
        if queue.is_empty() {
            // Should only happen once, as we only have one component.
            let start = match pick_best_unvisited(&visited) {
                Some(s) => s,
                None => break,
            };
            visited.insert(start);
            queue.push_back(start);
        }

        while let Some(u) = queue.pop_front() {
            // Sort neighbors so higher-degree / larger-patch neighbors are preferred.
            let mut neighbors: Vec<_> = curve_skeleton.neighbors(u).collect();
            neighbors.sort_by_key(|&v| {
                let degree = curve_skeleton.neighbors(v).count();
                let patch_u = curve_skeleton[u].patch_vertices.len();
                let patch_v = curve_skeleton[v].patch_vertices.len();
                let size = std::cmp::max(patch_u, patch_v);
                let displacement = curve_skeleton[v].position - curve_skeleton[u].position;
                let mut magnitudes = [
                    displacement.x.abs(),
                    displacement.y.abs(),
                    displacement.z.abs(),
                ];
                magnitudes.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                let confidence = magnitudes[0] - magnitudes[1]; // How good the best is relative to second best

                (
                    std::cmp::Reverse((size as f64 * confidence) as usize),
                    std::cmp::Reverse(size),
                    std::cmp::Reverse(confidence.to_bits()),
                    std::cmp::Reverse(degree),
                    std::cmp::Reverse(patch_v),
                )
            });

            for v in neighbors {
                // Edge in the partial graph.
                let pu = node_map[&u];
                let pv = node_map[&v];
                let eidx = partial
                    .find_edge(pu, pv)
                    .expect("Edge must exist in partial copy");

                // Edge already labeled (encountered from the other endpoint earlier in BFS).
                // Nothing to re-label, just ensure v is queued if not yet visited.
                if partial.edge_weight(eidx).unwrap().is_some() {
                    if visited.insert(v) {
                        queue.push_back(v);
                    }
                    continue;
                }

                // Try the geometrically closest axis first, then fall back to the other two.
                let disp = curve_skeleton[v].position - curve_skeleton[u].position;
                let candidates = preferred_axes_from_displacement(disp);

                // Accept the first candidate that keeps the partial labeling realizable.
                let mut assigned = false;
                for cand in candidates {
                    if axis_count_around_node(&partial, pu, cand) >= 2
                        || axis_count_around_node(&partial, pv, cand) >= 2
                    {
                        // At most 2 edges of the same direction per node
                        continue;
                    }

                    *partial.edge_weight_mut(eidx).unwrap() = Some(cand);
                    if is_partially_realizable(&partial) {
                        assigned = true;
                        break;
                    }
                    *partial.edge_weight_mut(eidx).unwrap() = None;
                }

                if !assigned {
                    // Fall back to global search from scratch, maximizing preferred directions.
                    return backtracking_orthogonalization(curve_skeleton);
                }

                if visited.insert(v) {
                    queue.push_back(v);
                }
            }
        }
    }

    finalize_partial_to_realized(&partial)
}
