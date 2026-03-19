use log::{error, warn};
use mehsh::prelude::{EPS, HasPosition, Mesh};
use petgraph::graph::{EdgeIndex, NodeIndex, UnGraph};
use petgraph::prelude::StableUnGraph;
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};

use bimap::BiHashMap;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

use crate::{
    prelude::{CurveSkeleton, INPUT, PrincipalDirection, VertID},
    skeleton::{boundary_loop::BoundaryLoop, curve_skeleton::SkeletonNode},
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
    #[serde(default)]
    pub sign: AxisSign,
    pub length: u32,
    pub boundary_loop: BoundaryLoop,
}
/// Curve skeleton with axis-aligned edges and integer grid coordinates assigned to each node.
pub type LabeledCurveSkeleton = StableUnGraph<OrthogonalSkeletonNode, OrthogonalSkeletonEdge>;

/// Curve skeleton where every edge has an axis label, but node coordinates are not yet assigned.
type EdgeLabeledCurveSkeleton = UnGraph<SkeletonNode, OrthogonalSkeletonEdge>;

/// Sign along a principal axis (+X vs -X).
#[derive(Copy, Clone, Default, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
pub enum AxisSign {
    #[default]
    Positive,
    Negative,
}
impl AxisSign {
    #[must_use]
    pub fn flipped(self) -> Self {
        match self {
            Self::Positive => Self::Negative,
            Self::Negative => Self::Positive,
        }
    }
}

/// Curve skeleton where edges are either labeled with an axis (`Some`) or not yet assigned (`None`).
/// Used while building up a labeling incrementally.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct SignedAxis {
    axis: PrincipalDirection,
    /// Sign, relative to edge_endpoints order.
    sign: AxisSign,
}

type PartialEdgeLabeledCurveSkeleton = UnGraph<SkeletonNode, Option<SignedAxis>>;

/// Convenience for querying edge sign from a given endpoint node.
pub trait LabeledSkeletonSignExt {
    /// Returns the sign of `edge` as seen from `node`.
    ///
    /// If `node` is the stored `source()` of the edge, this is the edge's stored sign.
    /// If `node` is the stored `target()`, this is the flipped sign.
    /// Returns `None` if `node` is not an endpoint of `edge`.
    fn edge_sign_from(&self, edge: EdgeIndex, node: NodeIndex) -> Option<AxisSign>;
}

impl LabeledSkeletonSignExt for LabeledCurveSkeleton {
    fn edge_sign_from(&self, edge: EdgeIndex, node: NodeIndex) -> Option<AxisSign> {
        let (a, b) = self.edge_endpoints(edge)?;
        let sign = self.edge_weight(edge)?.sign;
        if node == a {
            Some(sign)
        } else if node == b {
            Some(sign.flipped())
        } else {
            None
        }
    }
}

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

fn axis_sign_from_value(v: f64) -> AxisSign {
    if v >= 0.0 {
        AxisSign::Positive
    } else {
        AxisSign::Negative
    }
}

/// Estimates a boundary-loop normal by fitting a least-squares plane
/// to boundary vertices and taking the smallest-variance eigenvector.
fn best_fit_boundary_plane_normal(
    boundary_loop: &BoundaryLoop,
    mesh: &Mesh<INPUT>,
) -> Option<nalgebra::Vector3<f64>> {
    let mut unique_vertices: HashSet<VertID> = HashSet::new();
    for &edge in &boundary_loop.edge_midpoints {
        unique_vertices.insert(mesh.root(edge));
        unique_vertices.insert(mesh.toor(edge));
    }

    if unique_vertices.len() < 3 {
        return None;
    }

    let count = unique_vertices.len() as f64;
    let centroid = unique_vertices
        .iter()
        .fold(nalgebra::Vector3::zeros(), |acc, &v| acc + mesh.position(v))
        / count;

    let mut covariance = nalgebra::Matrix3::<f64>::zeros();
    for &v in &unique_vertices {
        let d = mesh.position(v) - centroid;
        covariance += d * d.transpose();
    }

    let eig = nalgebra::SymmetricEigen::new(covariance);
    let min_idx = if eig.eigenvalues[0] <= eig.eigenvalues[1] {
        if eig.eigenvalues[0] <= eig.eigenvalues[2] {
            0
        } else {
            2
        }
    } else if eig.eigenvalues[1] <= eig.eigenvalues[2] {
        1
    } else {
        2
    };

    let normal = eig.eigenvectors.column(min_idx).into_owned();
    let nrm = normal.norm();
    if nrm <= EPS {
        None
    } else {
        Some(normal / nrm)
    }
}

fn directional_confidence(v: nalgebra::Vector3<f64>) -> f64 {
    let mut mags = [v.x.abs(), v.y.abs(), v.z.abs()];
    mags.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    mags[0] - mags[1]
}

fn sign_from_node_along_edge(node: NodeIndex, edge_source: NodeIndex, sign: AxisSign) -> AxisSign {
    if node == edge_source {
        sign
    } else {
        sign.flipped()
    }
}

/// Returns true if assigning (axis, sign) to the candidate edge would violate the
/// per-node capacity of one edge per (axis, sign) around the node.
fn axis_sign_conflict_around_node(
    partial: &PartialEdgeLabeledCurveSkeleton,
    node: NodeIndex,
    cand_edge_source: NodeIndex,
    cand_axis: PrincipalDirection,
    cand_sign: AxisSign,
) -> bool {
    let cand_sign_at_node = sign_from_node_along_edge(node, cand_edge_source, cand_sign);

    // Check all already-labeled incident edges at `node` for same (axis, sign-at-node).
    // IMPORTANT: We stored existing.sign relative to (a,b) = edge_endpoints(edge_id), not
    // relative to edge.source()/edge.target(). For UnGraph, source/target order is arbitrary,
    // so we must use edge_endpoints to interpret the sign consistently.
    for edge in partial.edges(node) {
        let &Some(existing) = edge.weight() else {
            continue;
        };
        if existing.axis != cand_axis {
            continue;
        }

        let (ea, _) = partial.edge_endpoints(edge.id()).expect("edge must exist");
        let existing_sign_at_node = if node == ea {
            existing.sign
        } else {
            // Order according to edge_endpoints
            existing.sign.flipped()
        };
        if existing_sign_at_node == cand_sign_at_node {
            return true;
        }
    }

    false
}

/// Build a working copy of the skeleton with all edges unlabeled.
///
/// Returns the partial graph, a bidirectional node-index map, and a map from
/// each partial `EdgeIndex` to the boundary loop already stored on that edge
/// in the source skeleton.
fn build_unlabeled_partial(
    curve_skeleton: &CurveSkeleton,
) -> (
    PartialEdgeLabeledCurveSkeleton,
    BiHashMap<NodeIndex, NodeIndex>,
    HashMap<EdgeIndex, BoundaryLoop>,
) {
    let mut partial = PartialEdgeLabeledCurveSkeleton::new_undirected();
    let mut map: BiHashMap<NodeIndex, NodeIndex> = BiHashMap::new();
    let mut boundary_loops: HashMap<EdgeIndex, BoundaryLoop> = HashMap::new();

    for node in curve_skeleton.node_indices() {
        let w = curve_skeleton.node_weight(node).unwrap().clone();
        let out_n = partial.add_node(w);
        map.insert(node, out_n);
    }

    for edge in curve_skeleton.edge_indices() {
        let (a, b) = curve_skeleton.edge_endpoints(edge).unwrap();
        let pa = *map.get_by_left(&a).expect("Node missing in map");
        let pb = *map.get_by_left(&b).expect("Node missing in map");
        let partial_edge = partial.add_edge(pa, pb, None);
        let bl = curve_skeleton.edge_weight(edge)
            .expect("CurveSkeleton edge missing weight")
            .clone();
        boundary_loops.insert(partial_edge, bl);
    }

    (partial, map, boundary_loops)
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
        |w| matches!(*w, Some(s) if s.axis != forbidden),
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

/// Compresses components by solving the oriented unit-length constraints induced by axis-edges:
/// each edge requires its endpoints to differ by at least 1 along `axis`, in the edge's stored sign.
fn compress_components_from_oriented_edges(
    skeleton: &EdgeLabeledCurveSkeleton,
    axis: PrincipalDirection,
    comps: &[usize],
    idx_map: &HashMap<NodeIndex, usize>,
) -> Option<Vec<i32>> {
    let nr_components = comps.iter().copied().max().map_or(0, |m| m + 1);
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); nr_components];
    let mut indegree: Vec<usize> = vec![0; nr_components];

    for edge in skeleton.edge_references() {
        let w = edge.weight();
        if w.direction != axis {
            continue;
        }
        let u = edge.source();
        let v = edge.target();
        let cu = comps[idx_map[&u]];
        let cv = comps[idx_map[&v]];
        if cu == cv {
            return None;
        }

        let (from, to) = match w.sign {
            AxisSign::Positive => (cu, cv),
            AxisSign::Negative => (cv, cu),
        };
        if from != to {
            adjacency[from].push(to);
            indegree[to] += 1;
        }
    }

    // Topological order
    let mut q: BinaryHeap<std::cmp::Reverse<usize>> = BinaryHeap::new();
    for c in 0..nr_components {
        if indegree[c] == 0 {
            q.push(std::cmp::Reverse(c));
        }
    }

    let mut coord_per_comp = vec![0i32; nr_components];
    let mut processed = 0usize;
    while let Some(std::cmp::Reverse(c)) = q.pop() {
        processed += 1;
        let base = coord_per_comp[c];
        for &n in &adjacency[c] {
            coord_per_comp[n] = coord_per_comp[n].max(base + 1);
            indegree[n] -= 1;
            if indegree[n] == 0 {
                q.push(std::cmp::Reverse(n));
            }
        }
    }
    if processed != nr_components {
        return None;
    }

    Some(comps.iter().map(|&cid| coord_per_comp[cid]).collect())
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
        let w = e.weight();
        out.add_edge(
            map[&a],
            map[&b],
            Some(SignedAxis {
                axis: w.direction,
                sign: w.sign,
            }),
        );
    }

    out
}

/// Realizes a partial labeling as a fully labeled skeleton.
fn finalize_partial_to_realized(
    partial: &PartialEdgeLabeledCurveSkeleton,
    boundary_loops: &HashMap<EdgeIndex, BoundaryLoop>,
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
        let signed = match *e.weight() {
            Some(d) => d,
            None => unreachable!(),
        };
        let bl = boundary_loops
            .get(&e.id())
            .cloned()
            .expect("every partial edge must have a boundary loop");
        full.add_edge(
            out_map[&a],
            out_map[&b],
            OrthogonalSkeletonEdge {
                direction: signed.axis,
                sign: signed.sign,
                length: 1,
                boundary_loop: bl,
            },
        );
    }

    realize(&full)
}

/// Global search for labelings that maximize preferred-axis matches.
///
/// Edge priority is weighted by patch size and directional confidence.
fn backtracking_orthogonalization(
    curve_skeleton: &CurveSkeleton,
    mesh: &Mesh<INPUT>,
) -> Option<LabeledCurveSkeleton> {
    let (mut partial, node_map, boundary_loops) = build_unlabeled_partial(curve_skeleton);

    warn!("Greedy orthogonalization failed, falling back to backtracking search.");

    #[derive(Clone)]
    struct EdgePlan {
        edge: EdgeIndex,
        primary: PrincipalDirection,
        preference_weight: i64,
        confidence: f64,
    }

    let boundary_normals: HashMap<EdgeIndex, Option<nalgebra::Vector3<f64>>> = partial
        .edge_indices()
        .map(|edge| {
            let normal = boundary_loops
                .get(&edge)
                .and_then(|bl| best_fit_boundary_plane_normal(bl, mesh));
            (edge, normal)
        })
        .collect();

    let mut plans: Vec<EdgePlan> = partial
        .edge_indices()
        .map(|edge| {
            let (u, v) = partial.edge_endpoints(edge).unwrap();
            let orig_u = *node_map
                .get_by_right(&u)
                .expect("Partial node not found in map");
            let orig_v = *node_map
                .get_by_right(&v)
                .expect("Partial node not found in map");
            let disp = curve_skeleton[orig_v].position - curve_skeleton[orig_u].position;
            let pref_vector = boundary_normals
                .get(&edge)
                .copied()
                .flatten()
                .unwrap_or(disp);
            let pref = preferred_axes_from_displacement(pref_vector);
            let preference_weight = std::cmp::max(
                curve_skeleton[orig_u].patch_vertices.len(),
                curve_skeleton[orig_v].patch_vertices.len(),
            ) as i64;
            let confidence = directional_confidence(pref_vector);
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
    let mut best_labels: Option<HashMap<EdgeIndex, SignedAxis>> = None;

    fn dfs(
        depth: usize,
        plans: &[EdgePlan],
        partial: &mut PartialEdgeLabeledCurveSkeleton,
        curve_skeleton: &CurveSkeleton,
        node_map: &BiHashMap<NodeIndex, NodeIndex>,
        boundary_normals: &HashMap<EdgeIndex, Option<nalgebra::Vector3<f64>>>,
        current_score: i64,
        best_score: &mut i64,
        best_labels: &mut Option<HashMap<EdgeIndex, SignedAxis>>,
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
        let orig_u = *node_map
            .get_by_right(&u)
            .expect("Partial node not found in map");
        let orig_v = *node_map
            .get_by_right(&v)
            .expect("Partial node not found in map");
        let disp = curve_skeleton[orig_v].position - curve_skeleton[orig_u].position;
        let pref_vector = boundary_normals
            .get(&plan.edge)
            .copied()
            .flatten()
            .unwrap_or(disp);
        let candidate_axes = preferred_axes_from_displacement(pref_vector);

        for cand in candidate_axes {
            // Determine edge endpoint order in the partial graph; sign is stored from a->b.
            let (a, b) = partial.edge_endpoints(plan.edge).unwrap();
            let orig_a = *node_map
                .get_by_right(&a)
                .expect("Partial node not found in map");
            let orig_b = *node_map
                .get_by_right(&b)
                .expect("Partial node not found in map");
            let disp_ab = curve_skeleton[orig_b].position - curve_skeleton[orig_a].position;
            let sign_vector = boundary_normals
                .get(&plan.edge)
                .copied()
                .flatten()
                .unwrap_or(disp_ab);
            let preferred_sign = axis_sign_from_value(axis_value(sign_vector, cand));
            let sign_candidates = [preferred_sign, preferred_sign.flipped()];

            for sign in sign_candidates {
                if axis_sign_conflict_around_node(partial, a, a, cand, sign)
                    || axis_sign_conflict_around_node(partial, b, a, cand, sign)
                {
                    continue;
                }

                let signed = SignedAxis { axis: cand, sign };
                *partial.edge_weight_mut(plan.edge).unwrap() = Some(signed);
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
                        node_map,
                        boundary_normals,
                        next_score,
                        best_score,
                        best_labels,
                        suffix_max_possible,
                    );
                }
                *partial.edge_weight_mut(plan.edge).unwrap() = None;
            }
        }
    }

    dfs(
        0,
        &plans,
        &mut partial,
        curve_skeleton,
        &node_map,
        &boundary_normals,
        0,
        &mut best_score,
        &mut best_labels,
        &suffix_max_possible,
    );

    if let Some(labels) = best_labels {
        for (edge, dir) in labels {
            *partial.edge_weight_mut(edge).unwrap() = Some(dir);
        }
        finalize_partial_to_realized(&partial, &boundary_loops)
    } else {
        None
    }
}

/// Returns true if the partial edge-labeling is still consistent with some valid orthogonal
/// integer-grid embedding. Unlabeled edges are ignored.
///
/// Three conditions are checked. First, for each axis D, no D-labeled edge may connect two
/// nodes in the same D-component (connected component with D-edges removed) — that would
/// force the endpoints to share their D-coordinate while the edge demands they differ.
/// Second, for each axis D, the directed graph on D-components induced by the edge signs
/// (from lower to higher coordinate) must be acyclic, or no integer coordinate assignment
/// exists. Third, no two nodes may share the same (cx, cy, cz) component-ID triple, since
/// that would place them on the same grid point.
fn is_partially_realizable(p: &PartialEdgeLabeledCurveSkeleton) -> bool {
    let (nodes, idx_map) = build_node_list_and_index_map(p);

    // Condition 1: no D-edge connects two nodes already in the same D-component.
    for &d in &[
        PrincipalDirection::X,
        PrincipalDirection::Y,
        PrincipalDirection::Z,
    ] {
        let comps = components_excluding_partial(p, d, &nodes, &idx_map);
        for edge in p.edge_references() {
            if let Some(signed) = *edge.weight() {
                if signed.axis != d {
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

    // Condition 2: for each axis, the sign-induced directed component graph must be acyclic.
    for &d in &[
        PrincipalDirection::X,
        PrincipalDirection::Y,
        PrincipalDirection::Z,
    ] {
        let comps = components_excluding_partial(p, d, &nodes, &idx_map);
        let nr_components = comps.iter().copied().max().map_or(0, |m| m + 1);
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); nr_components];
        let mut indegree: Vec<usize> = vec![0; nr_components];

        for edge in p.edge_references() {
            let Some(signed) = *edge.weight() else {
                continue;
            };
            if signed.axis != d {
                continue;
            }
            let (a, b) = p.edge_endpoints(edge.id()).expect("edge must exist");
            let ca = comps[idx_map[&a]];
            let cb = comps[idx_map[&b]];
            if ca == cb {
                return false;
            }
            let (from, to) = match signed.sign {
                AxisSign::Positive => (ca, cb),
                AxisSign::Negative => (cb, ca),
            };
            adjacency[from].push(to);
            indegree[to] += 1;
        }

        let mut q: VecDeque<usize> = VecDeque::new();
        for c in 0..nr_components {
            if indegree[c] == 0 {
                q.push_back(c);
            }
        }

        let mut processed = 0usize;
        while let Some(c) = q.pop_front() {
            processed += 1;
            for &n in &adjacency[c] {
                indegree[n] -= 1;
                if indegree[n] == 0 {
                    q.push_back(n);
                }
            }
        }
        if processed != nr_components {
            return false;
        }
    }

    // Condition 3: all (cx, cy, cz) triples must be distinct.
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

    let cx = compress_components_from_oriented_edges(s, PrincipalDirection::X, &comp_x, &idx_map)?;
    let cy = compress_components_from_oriented_edges(s, PrincipalDirection::Y, &comp_y, &idx_map)?;
    let cz = compress_components_from_oriented_edges(s, PrincipalDirection::Z, &comp_z, &idx_map)?;

    // Assemble coordinate vector for each node.
    let mut coords: Vec<IVector3D> = Vec::with_capacity(nodes.len());
    for i in 0..nodes.len() {
        coords.push(IVector3D::new(cx[i], cy[i], cz[i]));
    }

    // Canonicalize global orientation: sign choices are otherwise ambiguous up to
    // per-axis reflection, which can produce mirrored polycubes.
    let mut mean_orig = nalgebra::Vector3::<f64>::zeros();
    let mut mean_realized = nalgebra::Vector3::<f64>::zeros();
    for (i, &orig) in nodes.iter().enumerate() {
        mean_orig += s[orig].position;
        mean_realized += nalgebra::Vector3::new(
            coords[i].x as f64,
            coords[i].y as f64,
            coords[i].z as f64,
        );
    }
    let n = nodes.len() as f64;
    mean_orig /= n;
    mean_realized /= n;

    let mut corr_x = 0.0;
    let mut corr_y = 0.0;
    let mut corr_z = 0.0;
    for (i, &orig) in nodes.iter().enumerate() {
        let p = s[orig].position - mean_orig;
        let q = nalgebra::Vector3::new(
            coords[i].x as f64,
            coords[i].y as f64,
            coords[i].z as f64,
        ) - mean_realized;
        corr_x += p.x * q.x;
        corr_y += p.y * q.y;
        corr_z += p.z * q.z;
    }

    // If an axis correlation is negative, flip that realized axis.
    let flip_x = corr_x < -EPS;
    let flip_y = corr_y < -EPS;
    let flip_z = corr_z < -EPS;

    if flip_x || flip_y || flip_z {
        for c in &mut coords {
            if flip_x {
                c.x = -c.x;
            }
            if flip_y {
                c.y = -c.y;
            }
            if flip_z {
                c.z = -c.z;
            }
        }
    }

    // Build the output graph: copy nodes (with computed coordinates) and edges.
    let mut out: LabeledCurveSkeleton = Default::default();
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
            let sign = match w.direction {
                PrincipalDirection::X if flip_x => w.sign.flipped(),
                PrincipalDirection::Y if flip_y => w.sign.flipped(),
                PrincipalDirection::Z if flip_z => w.sign.flipped(),
                _ => w.sign,
            };
            out.add_edge(
                out_index_map[&a],
                out_index_map[&b],
                OrthogonalSkeletonEdge {
                    direction: w.direction,
                    sign,
                    length: w.length,
                    boundary_loop: w.boundary_loop.clone(),
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

    for e in out.edge_indices().collect::<Vec<_>>() {
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
/// For each unlabeled edge, we estimate a boundary-loop plane and use its normal as the
/// preferred direction; the axis most aligned with that normal is tried first and the other
/// two axes are fallbacks. The first choice that keeps the partial labeling consistent is
/// accepted.
///
/// If greedy gets stuck, a global backtracking search is used.
pub fn greedy_orthogonalization(
    curve_skeleton: &CurveSkeleton,
    mesh: &Mesh<INPUT>,
) -> Option<LabeledCurveSkeleton> {
    let (mut partial, node_map, boundary_loops) = build_unlabeled_partial(curve_skeleton);

    let boundary_normals: HashMap<EdgeIndex, Option<nalgebra::Vector3<f64>>> = partial
        .edge_indices()
        .map(|edge| {
            let normal = boundary_loops
                .get(&edge)
                .and_then(|bl| best_fit_boundary_plane_normal(bl, mesh));
            (edge, normal)
        })
        .collect();

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
                let pref_vector = partial
                    .find_edge(
                        *node_map.get_by_left(&u).expect("Original node missing in map"),
                        *node_map.get_by_left(&v).expect("Original node missing in map"),
                    )
                    .and_then(|e| boundary_normals.get(&e).copied().flatten())
                    .unwrap_or(displacement);
                let confidence = directional_confidence(pref_vector); // How good the best is relative to second best

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
                let pu = *node_map
                    .get_by_left(&u)
                    .expect("Original node missing in map");
                let pv = *node_map
                    .get_by_left(&v)
                    .expect("Original node missing in map");
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
                let pref_vector = boundary_normals
                    .get(&eidx)
                    .copied()
                    .flatten()
                    .unwrap_or(disp);
                let candidates = preferred_axes_from_displacement(pref_vector);

                // Accept the first candidate that keeps the partial labeling realizable.
                let mut assigned = false;
                for cand in candidates {
                    // Determine edge endpoint order in the partial graph; sign is stored from a->b.
                    let (a, b) = partial.edge_endpoints(eidx).unwrap();
                    let orig_a = *node_map
                        .get_by_right(&a)
                        .expect("Partial node not found in map");
                    let orig_b = *node_map
                        .get_by_right(&b)
                        .expect("Partial node not found in map");
                    let disp_ab = curve_skeleton[orig_b].position - curve_skeleton[orig_a].position;
                    let sign_vector = boundary_normals
                        .get(&eidx)
                        .copied()
                        .flatten()
                        .unwrap_or(disp_ab);
                    let preferred_sign = axis_sign_from_value(axis_value(sign_vector, cand));
                    let sign_candidates = [preferred_sign, preferred_sign.flipped()];

                    for sign in sign_candidates {
                        if axis_sign_conflict_around_node(&partial, a, a, cand, sign)
                            || axis_sign_conflict_around_node(&partial, b, a, cand, sign)
                        {
                            // At most one edge per (axis, sign) around a node.
                            continue;
                        }

                        *partial.edge_weight_mut(eidx).unwrap() =
                            Some(SignedAxis { axis: cand, sign });
                        if is_partially_realizable(&partial) {
                            assigned = true;
                            break;
                        }
                        *partial.edge_weight_mut(eidx).unwrap() = None;
                    }
                    if assigned {
                        break;
                    }
                }

                if !assigned {
                    // Fall back to global search from scratch, maximizing preferred directions.
                    return backtracking_orthogonalization(curve_skeleton, mesh);
                }

                if visited.insert(v) {
                    queue.push_back(v);
                }
            }
        }
    }

    finalize_partial_to_realized(&partial, &boundary_loops)
}
