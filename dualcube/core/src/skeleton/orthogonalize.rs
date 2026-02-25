use log::error;
use petgraph::graph::{NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};

use std::collections::{HashMap, HashSet, VecDeque};

use crate::prelude::to_principal_direction;
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

// Collects all node indices and builds a map from NodeIndex to a dense position in that slice.
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

// BFS connected-component labeling over the partial skeleton, ignoring all edges labeled
// `forbidden` (or unlabeled). Each node gets the integer ID of its component.
//
// The resulting component IDs serve directly as integer grid coordinates: two nodes share
// a D-coordinate iff they are in the same D-component (connected without crossing a D-edge).
fn components_excluding_partial(
    g: &PartialEdgeLabeledCurveSkeleton,
    forbidden: PrincipalDirection,
    nodes: &[NodeIndex],
    idx_map: &HashMap<NodeIndex, usize>,
) -> Vec<usize> {
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
                match *edge.weight() {
                    Some(dir) if dir != forbidden => {
                        // Pick the endpoint that is not u.
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
                    // Unlabeled or forbidden edges are excluded from this component.
                    _ => {}
                }
            }
        }

        cur += 1;
    }

    comp
}

// Wrap edges such that we can reuse labeling and realizability-checking code.
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

/// Assigns integer grid coordinates to every node and computes correct edge lengths.
///
/// Each node's coordinate along axis D is the ID of its D-component (connected component
/// with all D-edges removed). Edge lengths are recomputed as the coordinate difference
/// along the labeled axis.
pub fn realize(s: &EdgeLabeledCurveSkeleton) -> Option<LabeledCurveSkeleton> {
    // Convert to partial form once so we can use the same methods for both the
    // realizability check and the coordinate computation. // TODO: maybe better way with traits?
    let partial = partial_from_edge_labeled(s);
    if !is_partially_realizable(&partial) {
        return None;
    }

    let (nodes, idx_map) = build_node_list_and_index_map(s);

    // Collect per-axis component IDs, each is the assigned integer coordinate along that axis.
    let cx = components_excluding_partial(&partial, PrincipalDirection::X, &nodes, &idx_map);
    let cy = components_excluding_partial(&partial, PrincipalDirection::Y, &nodes, &idx_map);
    let cz = components_excluding_partial(&partial, PrincipalDirection::Z, &nodes, &idx_map);

    // Assemble coordinate vector for each node.
    let mut coords: Vec<IVector3D> = Vec::with_capacity(nodes.len());
    for i in 0..nodes.len() {
        coords.push(IVector3D::new(cx[i] as i32, cy[i] as i32, cz[i] as i32));
    }

    // Build the output graph: copy nodes (with computed coordinates) and edges.
    let mut out = LabeledCurveSkeleton::new_undirected();
    let mut out_index_map: HashMap<NodeIndex, NodeIndex> = HashMap::new();

    for &orig in &nodes {
        let node = s.node_weight(orig).unwrap().clone();
        let weight = OrthogonalSkeletonNode {
            skeleton_node: node,
            grid_position: IVector3D::new(0,0,0) // overwritten later
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

/// Greedily assigns an axis label to every edge so that the skeleton can be realized as
/// an orthogonal integer-grid embedding.
///
/// BFS starts from the highest-degree node (secondarily largest patch).
/// For each unlabeled edge, the axis most aligned with the geometric displacement
/// is tried first; the other two axes are fallbacks. The first choice that keeps the partial
/// labeling consistent is accepted.
///
/// Can fail when cycles impose conflicting global constraints that the greedy has no
/// lookahead to resolve.
///
/// TODO: fallback ILP or backtracking search for cases greedy cannot solve.
pub fn greedy_orthogonalization(curve_skeleton: &CurveSkeleton) -> Option<LabeledCurveSkeleton> {
    // Build an unlabeled copy of the skeleton. node_map translates NodeIndex values
    // so we can update edge weights in `partial` while searching over curve_skeleton.
    let mut partial = PartialEdgeLabeledCurveSkeleton::new_undirected();
    let mut node_map: HashMap<NodeIndex, NodeIndex> = HashMap::new();

    for n in curve_skeleton.node_indices() {
        let w = curve_skeleton.node_weight(n).unwrap().clone();
        let out_n = partial.add_node(w);
        node_map.insert(n, out_n);
    }

    for e in curve_skeleton.edge_indices() {
        let (a, b) = curve_skeleton.edge_endpoints(e).unwrap();
        partial.add_edge(node_map[&a], node_map[&b], None);
    }

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

    // Process all connected components.
    while visited.len() < curve_skeleton.node_count() {
        if queue.is_empty() {
            let start = match pick_best_unvisited(&visited) {
                Some(s) => s,
                None => break,
            };
            visited.insert(start);
            queue.push_back(start);
        }

        while let Some(u) = queue.pop_front() {
            // Sort neighbors so higher-degree / larger-patch neighbors are preferred.
            let mut neighs: Vec<_> = curve_skeleton.neighbors(u).collect();
            neighs.sort_by_key(|&v| {
                let deg = curve_skeleton.neighbors(v).count();
                let patch = curve_skeleton[v].patch_vertices.len();
                (std::cmp::Reverse(deg), std::cmp::Reverse(patch))
            });

            for v in neighs {
                // Edge in the partial graph.
                let pu = node_map[&u];
                let pv = node_map[&v];
                let eidx = partial
                    .find_edge(pu, pv)
                    .expect("edge must exist in partial copy");

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
                let (primary_axis, _orient) = to_principal_direction(disp);
                let mut candidates = vec![primary_axis];
                for &d in &[
                    PrincipalDirection::X,
                    PrincipalDirection::Y,
                    PrincipalDirection::Z,
                ] {
                    if d != primary_axis {
                        candidates.push(d);
                    }
                }

                // Accept the first candidate that keeps the partial labeling realizable.
                let mut assigned = false;
                for cand in candidates {
                    *partial.edge_weight_mut(eidx).unwrap() = Some(cand);
                    if is_partially_realizable(&partial) {
                        assigned = true;
                        break;
                    }
                    *partial.edge_weight_mut(eidx).unwrap() = None;
                }

                if !assigned {
                    // TODO: some kind of backtracking later?
                    return None;
                }

                if visited.insert(v) {
                    queue.push_back(v);
                }
            }
        }
    }

    // Convert to EdgeLabeledCurveSkeleton, edge lengths are placeholders that realize will recompute.
    for edge_ref in partial.edge_references() {
        if edge_ref.weight().is_none() {
            error!("edge from {:?} to {:?} is still unlabeled after greedy assignment, this should not happen",
                edge_ref.source(), edge_ref.target());
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
        let weight = OrthogonalSkeletonEdge {
            direction: dir,
            length: 1, // placeholder, will be recomputed in realize
        };
        full.add_edge(out_map[&a], out_map[&b], weight);
    }

    // Compute integer coordinates and correct edge lengths.
    realize(&full)
}
