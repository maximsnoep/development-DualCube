use crate::prelude::*;
use log::info;
use mehsh::prelude::*;
use petgraph::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;

// We are given some mesh that is a triangulation (Vm,Em,Fm).
// Then we construct an extended neighborhood graph g (Vg,Eg).
//      For each em in Em, we have a vg in Vg.
//      The vertices in Vg are connected with k-ring neighbors. I.e., any two vertices in Vg
//      (edges in the original mesh) are connected in g if their distance in the original mesh is at most k.
// Then we construct the derivative of g (sometimes called line-graph), which we call g'.
//      For each eg in Eg, we have a vgp in Vg'.
//      The vertices in Vg' are connected if the corresponding edges in g form a directed path of length 2,
//      satisfy the angle threshold, and pass the axis filter.

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
struct AxisGraph<T: Tag> {
    derivative_graph: DiGraph<(), f64>,
    eg_to_vgp: HashMap<EdgeIndex, NodeIndex>,
    vgp_to_eg: HashMap<NodeIndex, EdgeIndex>,

    #[serde(skip)]
    start_vgps_by_em: HashMap<ids::Key<EDGE, T>, Vec<usize>>,

    #[serde(skip)]
    mwc_from_vgp_cache: HashMap<(usize, usize), Option<(f64, Vec<usize>)>>,

    #[serde(skip)]
    mwc_cache: HashMap<ids::Key<EDGE, T>, Option<(f64, Vec<ids::Key<EDGE, T>>)>>,

    #[serde(skip)]
    out_adj: Vec<Vec<(usize, f64)>>,

    #[serde(skip)]
    source_em_of_vgp: Vec<ids::Key<EDGE, T>>,

    #[serde(skip)]
    source_pos_of_vgp: Vec<Vector3D>,

    #[serde(skip)]
    source_normal_of_vgp: Vec<Vector3D>,

    #[serde(skip)]
    source_kappa_min_of_vgp: Vec<f64>,

    #[serde(skip)]
    source_kappa_max_of_vgp: Vec<f64>,

    #[serde(skip)]
    source_dmin_of_vgp: Vec<Vector3D>,

    #[serde(skip)]
    source_dmax_of_vgp: Vec<Vector3D>,
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct ElasticaGraph<T: Tag> {
    extended_neighborhood_graph: DiGraph<(), ()>,
    em_to_vg: HashMap<ids::Key<EDGE, T>, NodeIndex>,
    vg_to_em: HashMap<NodeIndex, ids::Key<EDGE, T>>,

    #[serde(skip)]
    axis_graphs: [AxisGraph<T>; 3],
}

impl<T: Tag> ElasticaGraph<T> {
    pub fn new(
        mesh: Arc<Mesh<T>>,
        k: usize,
        threshold_degrees: usize,
        max_out_degree: usize,
        beta_curvature: f64,
        beta_axis: f64,
    ) -> Self {
        let threshold_radians = (threshold_degrees as f64).to_radians();

        // --------------------------------------------------------------------
        // Construct g
        // --------------------------------------------------------------------
        let mut extended_neighborhood_graph = DiGraph::<(), ()>::new();
        let g = &mut extended_neighborhood_graph;
        let mut em_to_vg = HashMap::new();
        let mut vg_to_em = HashMap::new();

        let edge_ids = mesh.edge_ids();
        let mut vg_pos = Vec::with_capacity(edge_ids.len());
        let mut vg_normal = Vec::with_capacity(edge_ids.len());
        let mut vg_kappa_min = Vec::with_capacity(edge_ids.len());
        let mut vg_kappa_max = Vec::with_capacity(edge_ids.len());
        let mut vg_dmin = Vec::with_capacity(edge_ids.len());
        let mut vg_dmax = Vec::with_capacity(edge_ids.len());

        for em_id in &edge_ids {
            let vg_id = g.add_node(());
            em_to_vg.insert(*em_id, vg_id);
            vg_to_em.insert(vg_id, *em_id);
            vg_pos.push(mesh.position(*em_id));

            let faces = mesh.faces(*em_id).collect::<Vec<_>>();
            let average_normal = if faces.is_empty() {
                Vector3D::new(0.0, 0.0, 1.0)
            } else {
                let summed = faces.iter().fold(Vector3D::zeros(), |acc, &face_id| {
                    acc + mesh.normal(face_id)
                });
                if summed.norm_squared() <= 1e-12 {
                    Vector3D::new(0.0, 0.0, 1.0)
                } else {
                    summed.normalize()
                }
            };
            vg_normal.push(average_normal);

            let (kappa_min, kappa_max, dmin, dmax) = estimate_principal_frame(&mesh, *em_id);
            vg_kappa_min.push(kappa_min);
            vg_kappa_max.push(kappa_max);
            vg_dmin.push(dmin);
            vg_dmax.push(dmax);
        }

        let mut g_edge_set = HashSet::new();
        let max_g_out_degree = max_out_degree.saturating_mul(2).max(1);

        for em_id in &edge_ids {
            let Some(&src) = em_to_vg.get(em_id) else {
                continue;
            };

            let src_pos = mesh.position(*em_id);
            let mut neighbors = mesh
                .neighbors2_k(*em_id, k)
                .into_iter()
                .filter_map(|n_id| {
                    let dst = em_to_vg.get(&n_id).copied()?;
                    if dst == src {
                        return None;
                    }
                    let dist2 = (mesh.position(n_id) - src_pos).norm_squared();
                    Some((dst, dist2))
                })
                .collect::<Vec<_>>();

            neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            neighbors.truncate(max_g_out_degree);

            for (dst, _) in neighbors {
                if g_edge_set.insert((src.index(), dst.index())) {
                    g.add_edge(src, dst, ());
                }
            }
        }

        info!(
            "Extended neighborhood graph: {} vertices, {} edges, avg out-degree {:.3}",
            g.node_count(),
            g.edge_count(),
            g.edge_count() as f64 / g.node_count().max(1) as f64
        );

        let g_edge_endpoints = g
            .edge_indices()
            .map(|eg_id| {
                let (u, v) = g.edge_endpoints(eg_id).expect("valid g edge");
                (u.index(), v.index())
            })
            .collect::<Vec<_>>();

        let axis_graphs = [
            Self::build_axis_graph(
                &g,
                &vg_to_em,
                &vg_pos,
                &vg_normal,
                &g_edge_endpoints,
                threshold_radians,
                max_out_degree,
                beta_curvature,
                beta_axis,
                PrincipalDirection::X,
                &vg_kappa_min,
                &vg_kappa_max,
                &vg_dmin,
                &vg_dmax,
            ),
            Self::build_axis_graph(
                &g,
                &vg_to_em,
                &vg_pos,
                &vg_normal,
                &g_edge_endpoints,
                threshold_radians,
                max_out_degree,
                beta_curvature,
                beta_axis,
                PrincipalDirection::Y,
                &vg_kappa_min,
                &vg_kappa_max,
                &vg_dmin,
                &vg_dmax,
            ),
            Self::build_axis_graph(
                &g,
                &vg_to_em,
                &vg_pos,
                &vg_normal,
                &g_edge_endpoints,
                threshold_radians,
                max_out_degree,
                beta_curvature,
                beta_axis,
                PrincipalDirection::Z,
                &vg_kappa_min,
                &vg_kappa_max,
                &vg_dmin,
                &vg_dmax,
            ),
        ];

        Self {
            extended_neighborhood_graph,
            em_to_vg,
            vg_to_em,
            axis_graphs,
        }
    }

    fn build_axis_graph(
        g: &DiGraph<(), ()>,
        vg_to_em: &HashMap<NodeIndex, ids::Key<EDGE, T>>,
        vg_pos: &[Vector3D],
        vg_normal: &[Vector3D],
        g_edge_endpoints: &[(usize, usize)],
        threshold_radians: f64,
        max_out_degree: usize,
        beta_curvature: f64,
        beta_axis: f64,
        axis: PrincipalDirection,
        vg_kappa_min: &[f64],
        vg_kappa_max: &[f64],
        vg_dmin: &[Vector3D],
        vg_dmax: &[Vector3D],
    ) -> AxisGraph<T> {
        let mut derivative_graph = DiGraph::<(), f64>::new();
        let gprime = &mut derivative_graph;
        let mut eg_to_vgp = HashMap::new();
        let mut vgp_to_eg = HashMap::new();

        for eg_id in g.edge_indices() {
            let vgp_id = gprime.add_node(());
            eg_to_vgp.insert(eg_id, vgp_id);
            vgp_to_eg.insert(vgp_id, eg_id);
        }

        for eg_id in g.edge_indices() {
            let (u, v) = g_edge_endpoints[eg_id.index()];

            let mut best_candidates = BinaryHeap::new();
            for next_ref in g.edges_directed(NodeIndex::new(v), Direction::Outgoing) {
                let next_eg_id = next_ref.id();
                if next_eg_id == eg_id {
                    continue;
                }

                let (v2, w) = g_edge_endpoints[next_eg_id.index()];
                debug_assert_eq!(v, v2);

                if w == u {
                    continue;
                }

                let Some(weight) = transition_weight(
                    vg_pos,
                    vg_normal,
                    g_edge_endpoints,
                    eg_id.index(),
                    next_eg_id.index(),
                    threshold_radians,
                    beta_curvature,
                    beta_axis,
                    axis,
                    vg_kappa_min,
                    vg_kappa_max,
                    vg_dmin,
                    vg_dmax,
                ) else {
                    continue;
                };

                best_candidates.push((MinScored(weight, next_eg_id.index()), next_eg_id));
                if best_candidates.len() > max_out_degree {
                    best_candidates.pop();
                }
            }

            let src_vgp = eg_to_vgp[&eg_id];
            for (_, next_eg_id) in best_candidates.into_sorted_vec() {
                let weight = transition_weight(
                    vg_pos,
                    vg_normal,
                    g_edge_endpoints,
                    eg_id.index(),
                    next_eg_id.index(),
                    threshold_radians,
                    beta_curvature,
                    beta_axis,
                    axis,
                    vg_kappa_min,
                    vg_kappa_max,
                    vg_dmin,
                    vg_dmax,
                )
                .expect("selected derivative edge should still have a valid weight");

                let dst_vgp = eg_to_vgp[&next_eg_id];
                gprime.add_edge(src_vgp, dst_vgp, weight);
            }
        }

        info!(
            "Derivative graph ({axis:?}): {} vertices, {} edges, avg out-degree {:.3}",
            gprime.node_count(),
            gprime.edge_count(),
            gprime.edge_count() as f64 / gprime.node_count().max(1) as f64
        );

        let n = derivative_graph.node_count();
        let mut out_adj = vec![Vec::<(usize, f64)>::new(); n];
        let mut in_adj = vec![Vec::<(usize, f64)>::new(); n];
        let mut source_em_of_vgp: Vec<Option<ids::Key<EDGE, T>>> = vec![None; n];

        for e in derivative_graph.edge_references() {
            let a = e.source().index();
            let b = e.target().index();
            let w = *e.weight();
            out_adj[a].push((b, w));
            in_adj[b].push((a, w));
        }

        for (&vgp, &eg) in &vgp_to_eg {
            let (src_vg, _dst_vg) = g.edge_endpoints(eg).expect("valid g edge");
            source_em_of_vgp[vgp.index()] = Some(vg_to_em[&src_vg]);
        }

        let source_em_of_vgp = source_em_of_vgp
            .into_iter()
            .map(|x| x.expect("every derivative node should map to a mesh edge"))
            .collect::<Vec<_>>();

        let mut start_vgps_by_em: HashMap<ids::Key<EDGE, T>, Vec<usize>> = HashMap::new();
        for (vg_node, &em_id) in vg_to_em {
            let starts = g
                .edges_directed(*vg_node, Direction::Outgoing)
                .filter_map(|e| eg_to_vgp.get(&e.id()).copied())
                .map(|vgp| vgp.index())
                .collect::<Vec<_>>();
            start_vgps_by_em.insert(em_id, starts);
        }

        let source_pos_of_vgp = (0..n)
            .map(|vgp_idx| {
                let vgp = NodeIndex::new(vgp_idx);
                let eg = vgp_to_eg[&vgp];
                let (src_vg, _) = g.edge_endpoints(eg).expect("valid g edge");
                vg_pos[src_vg.index()]
            })
            .collect::<Vec<_>>();

        let source_normal_of_vgp = (0..n)
            .map(|vgp_idx| {
                let vgp = NodeIndex::new(vgp_idx);
                let eg = vgp_to_eg[&vgp];
                let (src_vg, _) = g.edge_endpoints(eg).expect("valid g edge");
                vg_normal[src_vg.index()]
            })
            .collect::<Vec<_>>();

        let source_kappa_min_of_vgp = (0..n)
            .map(|vgp_idx| {
                let vgp = NodeIndex::new(vgp_idx);
                let eg = vgp_to_eg[&vgp];
                let (src_vg, _) = g.edge_endpoints(eg).expect("valid g edge");
                vg_kappa_min[src_vg.index()]
            })
            .collect::<Vec<_>>();

        let source_kappa_max_of_vgp = (0..n)
            .map(|vgp_idx| {
                let vgp = NodeIndex::new(vgp_idx);
                let eg = vgp_to_eg[&vgp];
                let (src_vg, _) = g.edge_endpoints(eg).expect("valid g edge");
                vg_kappa_max[src_vg.index()]
            })
            .collect::<Vec<_>>();

        let source_dmin_of_vgp = (0..n)
            .map(|vgp_idx| {
                let vgp = NodeIndex::new(vgp_idx);
                let eg = vgp_to_eg[&vgp];
                let (src_vg, _) = g.edge_endpoints(eg).expect("valid g edge");
                vg_dmin[src_vg.index()]
            })
            .collect::<Vec<_>>();

        let source_dmax_of_vgp = (0..n)
            .map(|vgp_idx| {
                let vgp = NodeIndex::new(vgp_idx);
                let eg = vgp_to_eg[&vgp];
                let (src_vg, _) = g.edge_endpoints(eg).expect("valid g edge");
                vg_dmax[src_vg.index()]
            })
            .collect::<Vec<_>>();

        AxisGraph {
            derivative_graph,
            eg_to_vgp,
            vgp_to_eg,
            start_vgps_by_em,
            mwc_from_vgp_cache: HashMap::new(),
            mwc_cache: HashMap::new(),
            out_adj,
            source_em_of_vgp,
            source_pos_of_vgp,
            source_normal_of_vgp,
            source_kappa_min_of_vgp,
            source_kappa_max_of_vgp,
            source_dmin_of_vgp,
            source_dmax_of_vgp,
        }
    }

    fn axis_graph(&self, axis: PrincipalDirection) -> &AxisGraph<T> {
        &self.axis_graphs[axis as usize]
    }

    fn axis_graph_mut(&mut self, axis: PrincipalDirection) -> &mut AxisGraph<T> {
        &mut self.axis_graphs[axis as usize]
    }

    pub fn extended_edges(&self) -> Vec<(ids::Key<EDGE, T>, Vec<ids::Key<EDGE, T>>)> {
        self.extended_neighborhood_graph
            .node_indices()
            .map(|src| {
                let src_em = self.vg_to_em[&src];
                let dsts = self
                    .extended_neighborhood_graph
                    .neighbors_directed(src, Direction::Outgoing)
                    .map(|dst| self.vg_to_em[&dst])
                    .collect::<Vec<_>>();
                (src_em, dsts)
            })
            .collect()
    }

    pub fn extended_edge_segments(&self) -> Vec<(ids::Key<EDGE, T>, ids::Key<EDGE, T>)> {
        self.extended_neighborhood_graph
            .edge_references()
            .map(|e| {
                let src = self.vg_to_em[&e.source()];
                let dst = self.vg_to_em[&e.target()];
                (src, dst)
            })
            .collect()
    }

    pub fn derivative_edge_segments(
        &self,
        axis: PrincipalDirection,
    ) -> Vec<(ids::Key<EDGE, T>, ids::Key<EDGE, T>)> {
        let axis_graph = self.axis_graph(axis);
        axis_graph
            .derivative_graph
            .edge_references()
            .map(|e| {
                let src_vgp = e.source();
                let dst_vgp = e.target();
                (
                    axis_graph.source_em_of_vgp[src_vgp.index()],
                    axis_graph.source_em_of_vgp[dst_vgp.index()],
                )
            })
            .collect()
    }

    pub fn derivative_edge_polylines(
        &self,
        axis: PrincipalDirection,
    ) -> Vec<(ids::Key<EDGE, T>, ids::Key<EDGE, T>, ids::Key<EDGE, T>, f64)> {
        let axis_graph = self.axis_graph(axis);
        axis_graph
            .derivative_graph
            .edge_references()
            .map(|e| {
                let src_vgp = e.source();
                let dst_vgp = e.target();

                let src_eg = axis_graph.vgp_to_eg[&src_vgp];
                let dst_eg = axis_graph.vgp_to_eg[&dst_vgp];

                let (m0_vg, m1_vg) = self
                    .extended_neighborhood_graph
                    .edge_endpoints(src_eg)
                    .expect("valid extended-neighborhood edge");
                let (m1_check_vg, m2_vg) = self
                    .extended_neighborhood_graph
                    .edge_endpoints(dst_eg)
                    .expect("valid extended-neighborhood edge");

                debug_assert_eq!(m1_vg, m1_check_vg);

                (
                    self.vg_to_em[&m0_vg],
                    self.vg_to_em[&m1_vg],
                    self.vg_to_em[&m2_vg],
                    *e.weight(),
                )
            })
            .collect()
    }

    pub fn compute_mwc_for_edge(
        &mut self,
        em_id: ids::Key<EDGE, T>,
        axis: PrincipalDirection,
    ) -> Option<(f64, Vec<ids::Key<EDGE, T>>)> {
        let axis_graph = self.axis_graph_mut(axis);

        if let Some(cached) = axis_graph.mwc_cache.get(&em_id).cloned() {
            return cached;
        }

        let starts = axis_graph.start_vgps_by_em.get(&em_id)?;
        let (start, first_next, first_w) =
            select_best_start_pair(starts, &axis_graph.out_adj, &axis_graph.source_pos_of_vgp)?;

        let cache_key = (start, first_next);
        let cycle_for_start =
            if let Some(cached) = axis_graph.mwc_from_vgp_cache.get(&cache_key).cloned() {
                cached
            } else {
                let computed = minimum_weight_cycle_astar(
                    &axis_graph.out_adj,
                    start,
                    first_next,
                    first_w,
                    &axis_graph.source_pos_of_vgp,
                );
                axis_graph
                    .mwc_from_vgp_cache
                    .insert(cache_key, computed.clone());
                computed
            };

        let best = cycle_for_start.map(|(cost, derivative_cycle)| {
            let em_cycle = derivative_cycle
                .into_iter()
                .map(|vgp| axis_graph.source_em_of_vgp[vgp])
                .collect::<Vec<_>>();
            (cost, em_cycle)
        });

        axis_graph.mwc_cache.insert(em_id, best.clone());
        best
    }
}

// --------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------

#[derive(Copy, Clone, Debug)]
pub struct MinScored<K, T>(pub K, pub T);

impl<K: PartialOrd, T> PartialEq for MinScored<K, T> {
    fn eq(&self, other: &MinScored<K, T>) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl<K: PartialOrd, T> Eq for MinScored<K, T> {}
impl<K: PartialOrd, T> PartialOrd for MinScored<K, T> {
    fn partial_cmp(&self, other: &MinScored<K, T>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<K: PartialOrd, T> Ord for MinScored<K, T> {
    fn cmp(&self, other: &MinScored<K, T>) -> Ordering {
        let a = &self.0;
        let b = &other.0;
        if a == b {
            Ordering::Equal
        } else if a < b {
            Ordering::Greater
        } else if a > b {
            Ordering::Less
        } else if a.ne(a) && b.ne(b) {
            Ordering::Equal
        } else if a.ne(a) {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    }
}

fn minimum_weight_cycle_astar(
    out_adj: &[Vec<(usize, f64)>],
    start: usize,
    first_next: usize,
    first_w: f64,
    positions: &[Vector3D],
) -> Option<(f64, Vec<usize>)> {
    let mut g_score = vec![f64::INFINITY; out_adj.len()];
    let mut parent = vec![usize::MAX; out_adj.len()];
    let mut settled = vec![false; out_adj.len()];
    let mut heap = BinaryHeap::new();
    let goal = positions[start];

    g_score[first_next] = 0.0;
    heap.push(MinScored(
        heuristic(positions[first_next], goal),
        first_next,
    ));

    while let Some(MinScored(_, node)) = heap.pop() {
        if settled[node] {
            continue;
        }
        settled[node] = true;

        if node == start {
            break;
        }

        for &(next, w) in &out_adj[node] {
            let tentative = g_score[node] + w;
            if tentative < g_score[next] {
                g_score[next] = tentative;
                parent[next] = node;
                let f_score = tentative + heuristic(positions[next], goal);
                heap.push(MinScored(f_score, next));
            }
        }
    }

    if !g_score[start].is_finite() {
        return None;
    }

    let total_cost = first_w + g_score[start];

    let mut suffix = Vec::new();
    let mut cur = start;
    suffix.push(cur);

    while cur != first_next {
        cur = parent[cur];
        if cur == usize::MAX {
            return None;
        }
        suffix.push(cur);
        if suffix.len() > out_adj.len() + 1 {
            return None;
        }
    }

    suffix.reverse();

    let mut cycle = Vec::with_capacity(suffix.len() + 1);
    cycle.push(start);
    cycle.extend(suffix);

    if cycle.len() < 4 {
        return None;
    }

    Some((total_cost, cycle))
}

fn select_best_start_pair(
    starts: &[usize],
    out_adj: &[Vec<(usize, f64)>],
    positions: &[Vector3D],
) -> Option<(usize, usize, f64)> {
    let mut best: Option<(f64, usize, usize, f64)> = None;

    for &start in starts {
        let goal = positions[start];
        for &(next, first_w) in &out_adj[start] {
            if next == start {
                continue;
            }

            let score = first_w + heuristic(positions[next], goal);
            if best
                .as_ref()
                .is_none_or(|(best_score, _, _, _)| score < *best_score)
            {
                best = Some((score, start, next, first_w));
            }
        }
    }

    best.map(|(_, start, next, first_w)| (start, next, first_w))
}

fn heuristic(a: Vector3D, b: Vector3D) -> f64 {
    (a - b).norm()
}

// An edge (a, b) in g' corresponds to the sequence of two edges (e_a, e_b) in g.
// A single edge e_a = (a0, a1) in g corresponds to a pair of edges in the original mesh, represented
// here by their edge-midpoints m_a0 and m_a1. Likewise e_b corresponds to (m_b0, m_b1).
//
// Because (a, b) is a valid transition in g', we have a1 = b0 and therefore m_a1 = m_b0.
// The transition geometry is therefore the polyline (m0, m1, m2) = (m_a0, m_a1, m_b1), with
// segment directions d01 = normalize(m1 - m0) and d12 = normalize(m2 - m1).
//
// We score a transition with:
//   w(a, b) = (beta_l + beta_c * curvature_deviation + beta_a * axis_deviation + beta_g * geodesic_deviation) * length
// where:
//   length = |m1 - m0| + |m2 - m1|
//
// Curvature deviation
// -------------------
// We want low cost when segment directions follow principal curvature directions.
//
// In the full intended model, each midpoint has interpolated principal directions (d_min, d_max) and
// principal curvatures (kappa_min, kappa_max). For a tangent unit vector t at that midpoint, define:
//
//   dev(p, t) = (1 - max(|<t, d_min>|, |<t, d_max>|)) / (1 - 1 / sqrt(2))
//
// This lies in [0, 1], with:
//   0 -> perfectly aligned with either principal direction
//   1 -> worst in-plane case, 45 degrees from both principal directions
//
// The tangent projection is:
//   t_tan = dot(t, d_min) * d_min + dot(t, d_max) * d_max
// followed by normalization.
//
// The transition curvature deviation is:
//   (dev(m0, d01) + dev(m1, d01) + dev(m1, d12) + dev(m2, d12)) / 4
//
// Current implementation gap:
// We currently store only scalar principal curvatures per midpoint, not the full principal direction frame.
// So for now we use a conservative proxy based on curvature magnitude mismatch between consecutive midpoints.
// This keeps the implementation explicit about what is known and leaves the principal-direction term open.
//
// Axis deviation
// --------------
// For each segment, let n be the normalized average normal of its endpoints.
// The axis-alignment direction is a = normalize(d x n), and the deviation is:
//
//   axis_dev = (1 - <a, axis>) / 2
//
// This lies in [0, 1], with:
//   0 -> aligned with the requested positive axis
//   1 -> aligned with the negative axis
//
// We average this over both segments.
//
// Geodesic deviation
// ------------------
// We penalize turning using:
//
//   geodesic_dev = (1 - <d01, d12>) / 2
//
// This lies in [0, 1], with:
//   0 -> perfectly straight
//   1 -> U-turn
//
// A hard turn-angle threshold is still applied before evaluating the weighted score.
fn transition_weight(
    vg_pos: &[Vector3D],
    vg_normal: &[Vector3D],
    g_edge_endpoints: &[(usize, usize)],
    eg_id: usize,
    next_eg_id: usize,
    threshold_radians: f64,
    beta_curvature: f64,
    beta_axis: f64,
    axis: PrincipalDirection,
    vg_kappa_min: &[f64],
    vg_kappa_max: &[f64],
    vg_dmin: &[Vector3D],
    vg_dmax: &[Vector3D],
) -> Option<f64> {
    const BETA_LENGTH: f64 = 1.0;
    const BETA_GEODESIC: f64 = 1.0;
    const EPS: f64 = 1e-12;

    let (u, v) = *g_edge_endpoints.get(eg_id)?;
    let (v2, w) = *g_edge_endpoints.get(next_eg_id)?;
    if v != v2 {
        return None;
    }

    let m0 = *vg_pos.get(u)?;
    let m1 = *vg_pos.get(v)?;
    let m2 = *vg_pos.get(w)?;

    let n0 = normalize_or(*vg_normal.get(u)?, Vector3D::new(0.0, 0.0, 1.0));
    let n1 = normalize_or(*vg_normal.get(v)?, Vector3D::new(0.0, 0.0, 1.0));
    let n2 = normalize_or(*vg_normal.get(w)?, Vector3D::new(0.0, 0.0, 1.0));

    let kappa_min_0 = *vg_kappa_min.get(u)?;
    let kappa_min_1 = *vg_kappa_min.get(v)?;
    let kappa_min_2 = *vg_kappa_min.get(w)?;
    let kappa_max_0 = *vg_kappa_max.get(u)?;
    let kappa_max_1 = *vg_kappa_max.get(v)?;
    let kappa_max_2 = *vg_kappa_max.get(w)?;

    let dmin0 = normalize_or(*vg_dmin.get(u)?, orthogonal_unit_vector(n0));
    let dmin1 = normalize_or(*vg_dmin.get(v)?, orthogonal_unit_vector(n1));
    let dmin2 = normalize_or(*vg_dmin.get(w)?, orthogonal_unit_vector(n2));

    let dmax0 = normalize_or(*vg_dmax.get(u)?, n0.cross(&dmin0));
    let dmax1 = normalize_or(*vg_dmax.get(v)?, n1.cross(&dmin1));
    let dmax2 = normalize_or(*vg_dmax.get(w)?, n2.cross(&dmin2));

    let seg01 = m1 - m0;
    let seg12 = m2 - m1;

    let l01 = seg01.norm();
    let l12 = seg12.norm();
    if l01 <= EPS || l12 <= EPS {
        return None;
    }

    let d01 = seg01 / l01;
    let d12 = seg12 / l12;

    let gamma = d01.angle(&d12);
    if gamma > threshold_radians {
        return None;
    }

    let length = l01 + l12;

    let axis_dev_01 = axis_deviation(d01, normalize_or(n0 + n1, n1), Vector3D::from(axis));
    let axis_dev_12 = axis_deviation(d12, normalize_or(n1 + n2, n1), Vector3D::from(axis));
    let axis_deviation = 0.5 * (axis_dev_01 + axis_dev_12);

    let geodesic_deviation = ((1.0 - d01.dot(&d12)) * 0.5).clamp(0.0, 1.0);

    let curvature_deviation = 0.25
        * ((kappa_min_0.abs() - kappa_max_0.abs()).powf(2.)
            * principal_direction_deviation(d01, dmin0, dmax0)
            + (kappa_min_1.abs() - kappa_max_1.abs()).powf(2.)
                * principal_direction_deviation(d01, dmin1, dmax1)
            + (kappa_min_1.abs() - kappa_max_1.abs()).powf(2.)
                * principal_direction_deviation(d12, dmin1, dmax1)
            + (kappa_min_2.abs() - kappa_max_2.abs()).powf(2.)
                * principal_direction_deviation(d12, dmin2, dmax2));

    Some(geodesic_deviation)
}

fn estimate_principal_frame<T: Tag>(
    mesh: &Mesh<T>,
    edge_id: ids::Key<EDGE, T>,
) -> (f64, f64, Vector3D, Vector3D) {
    let v0 = mesh.root(edge_id);
    let v1 = mesh.toor(edge_id);

    let (kappa_min_0, kappa_max_0, dmin_0, dmax_0) = estimate_vertex_principal_frame(mesh, v0);
    let (kappa_min_1, kappa_max_1, dmin_1, dmax_1) = estimate_vertex_principal_frame(mesh, v1);

    let n_mid = normalize_or(
        mesh.position(v1) - mesh.position(v0),
        Vector3D::new(1.0, 0.0, 0.0),
    );
    let dmin_1 = if dmin_0.dot(&dmin_1) < 0.0 {
        -dmin_1
    } else {
        dmin_1
    };
    let dmax_1 = if dmax_0.dot(&dmax_1) < 0.0 {
        -dmax_1
    } else {
        dmax_1
    };

    let dmin = normalize_or(dmin_0 + dmin_1, orthogonal_unit_vector(n_mid));
    let dmax = normalize_or(dmax_0 + dmax_1, orthogonal_unit_vector(dmin));

    (
        0.5 * (kappa_min_0 + kappa_min_1),
        0.5 * (kappa_max_0 + kappa_max_1),
        dmin,
        dmax,
    )
}

pub fn estimate_vertex_principal_frame<T: Tag>(
    mesh: &Mesh<T>,
    vert_id: ids::Key<VERT, T>,
) -> (f64, f64, Vector3D, Vector3D) {
    let v = mesh.position(vert_id);

    let (t1_raw, _t2_raw, n_raw) = mesh.tangent_frame(vert_id);
    let n = normalize_or(n_raw, Vector3D::new(0.0, 0.0, 1.0));
    let t1_proj = t1_raw - n * t1_raw.dot(&n);
    let t1 = normalize_or(t1_proj, orthogonal_unit_vector(n));
    let t2 = normalize_or(n.cross(&t1), orthogonal_unit_vector(n));

    let neighbors = mesh.neighbors(vert_id).collect::<Vec<_>>();
    if neighbors.is_empty() {
        return (0.0, 0.0, t1, t2);
    }

    let mut ata = [[0.0_f64; 4]; 4];
    let mut atb = [0.0_f64; 4];

    for neighbor_id in neighbors {
        let p = mesh.position(neighbor_id);

        let e = p - v;
        let e_t = e - n * e.dot(&n);
        let len2 = e_t.dot(&e_t);
        if len2 < 1e-12 {
            continue;
        }

        let nj = normalize_or(mesh.normal(neighbor_id), n);
        let dn = nj - n;
        let dn_t = dn - n * dn.dot(&n);

        let ux = t1.dot(&e_t);
        let uy = t2.dot(&e_t);
        let dn2x = t1.dot(&dn_t);
        let dn2y = t2.dot(&dn_t);

        let alpha = (1.0 / len2).min(1e6);

        let rows = [[ux, uy, 0.0, 0.0], [0.0, 0.0, ux, uy]];
        let rhs = [-dn2x, -dn2y];

        for row_id in 0..2 {
            for i in 0..4 {
                atb[i] += alpha * rows[row_id][i] * rhs[row_id];
                for j in 0..4 {
                    ata[i][j] += alpha * rows[row_id][i] * rows[row_id][j];
                }
            }
        }
    }

    let ata = nalgebra::Matrix4::new(
        ata[0][0], ata[0][1], ata[0][2], ata[0][3], ata[1][0], ata[1][1], ata[1][2], ata[1][3],
        ata[2][0], ata[2][1], ata[2][2], ata[2][3], ata[3][0], ata[3][1], ata[3][2], ata[3][3],
    );
    let Some(inv) = ata.try_inverse() else {
        return (0.0, 0.0, t1, t2);
    };
    let a = [
        inv[(0, 0)] * atb[0] + inv[(0, 1)] * atb[1] + inv[(0, 2)] * atb[2] + inv[(0, 3)] * atb[3],
        inv[(1, 0)] * atb[0] + inv[(1, 1)] * atb[1] + inv[(1, 2)] * atb[2] + inv[(1, 3)] * atb[3],
        inv[(2, 0)] * atb[0] + inv[(2, 1)] * atb[1] + inv[(2, 2)] * atb[2] + inv[(2, 3)] * atb[3],
        inv[(3, 0)] * atb[0] + inv[(3, 1)] * atb[1] + inv[(3, 2)] * atb[2] + inv[(3, 3)] * atb[3],
    ];

    let shape = nalgebra::Matrix2::new(a[0], 0.5 * (a[1] + a[2]), 0.5 * (a[1] + a[2]), a[3]);
    let eig = nalgebra::SymmetricEigen::new(shape);

    let kappa_min = eig.eigenvalues[0];
    let kappa_max = eig.eigenvalues[1];

    let dmin_x = eig.eigenvectors[(0, 0)];
    let dmin_y = eig.eigenvectors[(1, 0)];
    let dmax_x = eig.eigenvectors[(0, 1)];
    let dmax_y = eig.eigenvectors[(1, 1)];

    let dmin = normalize_or(t1 * dmin_x + t2 * dmin_y, t1);
    let dmax = normalize_or(t1 * dmax_x + t2 * dmax_y, t2);
    let dmax = normalize_or(dmax - n * dmax.dot(&n), n.cross(&dmin));
    let dmin = normalize_or(n.cross(&dmax), dmin);

    (kappa_min, kappa_max, dmin, dmax)
}

fn normalize_or(v: Vector3D, fallback: Vector3D) -> Vector3D {
    if v.norm_squared() <= 1e-12 {
        fallback
    } else {
        v.normalize()
    }
}

fn axis_deviation(direction: Vector3D, normal: Vector3D, axis: Vector3D) -> f64 {
    let alignment = direction.cross(&normal);
    let res = 0.5 - alignment.dot(&axis) * 0.5;
    assert!(res >= 0.0 && res <= 1.0);
    res
}

fn principal_direction_deviation(direction: Vector3D, dmin: Vector3D, dmax: Vector3D) -> f64 {
    let tangent = direction.dot(&dmin) * dmin + direction.dot(&dmax) * dmax;
    if tangent.norm_squared() <= 1e-12 {
        return 1.0;
    }

    let tangent = tangent.normalize();
    let alignment = tangent.dot(&dmin).abs().max(tangent.dot(&dmax).abs());
    ((1.0 - alignment) / (1.0 - std::f64::consts::FRAC_1_SQRT_2)).clamp(0.0, 1.0)
}

fn orthogonal_unit_vector(n: Vector3D) -> Vector3D {
    let seed = if n.x.abs() < 0.9 {
        Vector3D::new(1.0, 0.0, 0.0)
    } else {
        Vector3D::new(0.0, 1.0, 0.0)
    };
    normalize_or(n.cross(&seed), Vector3D::new(0.0, 0.0, 1.0))
}
