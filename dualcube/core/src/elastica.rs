use mehsh::prelude::*;
use petgraph::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::{collections::HashMap, sync::Arc};

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
pub struct ElasticaGraph<T: Tag> {
    extended_neighborhood_graph: DiGraph<(), ()>,
    em_to_vg: HashMap<ids::Key<EDGE, T>, NodeIndex>,
    vg_to_em: HashMap<NodeIndex, ids::Key<EDGE, T>>,

    derivative_graph: DiGraph<(), f64>,
    eg_to_vgp: HashMap<EdgeIndex, NodeIndex>,
    vgp_to_eg: HashMap<NodeIndex, EdgeIndex>,

    #[serde(skip)]
    start_vgps_by_em: HashMap<ids::Key<EDGE, T>, Vec<usize>>,

    #[serde(skip)]
    mwc_from_vgp_cache: HashMap<usize, Option<(f64, Vec<usize>)>>,

    #[serde(skip)]
    mwc_cache: HashMap<ids::Key<EDGE, T>, Option<(f64, Vec<ids::Key<EDGE, T>>)>>,

    #[serde(skip)]
    out_adj: Vec<Vec<(usize, f64)>>,

    #[serde(skip)]
    in_adj: Vec<Vec<(usize, f64)>>,

    #[serde(skip)]
    source_em_of_vgp: Vec<ids::Key<EDGE, T>>,
}

impl<T: Tag> ElasticaGraph<T> {
    pub fn new(mesh: Arc<Mesh<T>>, k: usize, threshold_degrees: usize, beta: f64) -> Self {
        let threshold_radians = (threshold_degrees as f64).to_radians();

        // --------------------------------------------------------------------
        // Construct g
        // --------------------------------------------------------------------
        let mut extended_neighborhood_graph = DiGraph::<(), ()>::new();
        let g = &mut extended_neighborhood_graph;
        let mut em_to_vg = HashMap::new();
        let mut vg_to_em = HashMap::new();

        // Add Vg
        for em_id in mesh.edge_ids() {
            let vg_id = g.add_node(());
            em_to_vg.insert(em_id, vg_id);
            vg_to_em.insert(vg_id, em_id);
        }

        // Add Eg without duplicates.
        // If neighbors2_k is symmetric, this still gives both directions,
        // but only one copy of each ordered pair.
        for em_id in mesh.edge_ids() {
            let Some(&src) = em_to_vg.get(&em_id) else {
                continue;
            };

            for n_id in mesh.neighbors2_k(em_id, k) {
                let Some(&dst) = em_to_vg.get(&n_id) else {
                    continue;
                };
                add_directed_edge_if_missing(g, src, dst);
            }
        }

        println!(
            "Extended neighborhood graph has {} vertices and {} edges",
            g.node_count(),
            g.edge_count()
        );
        println!(
            "Average out-degree: {}",
            g.edge_count() as f64 / g.node_count() as f64
        );
        print_graph_duplicate_stats(g, "g");

        // --------------------------------------------------------------------
        // Construct g'
        // --------------------------------------------------------------------
        let mut derivative_graph = DiGraph::<(), f64>::new();
        let gprime = &mut derivative_graph;
        let mut eg_to_vgp = HashMap::new();
        let mut vgp_to_eg = HashMap::new();

        // One derivative node per directed edge of g
        for eg_id in g.edge_indices() {
            let vgp_id = gprime.add_node(());
            eg_to_vgp.insert(eg_id, vgp_id);
            vgp_to_eg.insert(vgp_id, eg_id);
        }

        // Add derivative edges only for directed continuations:
        //    (u -> v) followed by (v -> w)
        // Optional: reject w == u to forbid immediate backtracking.
        for eg_id in g.edge_indices() {
            let (u, v) = g.edge_endpoints(eg_id).expect("valid g edge");

            for next_ref in g.edges_directed(v, Direction::Outgoing) {
                let next_eg_id = next_ref.id();
                if next_eg_id == eg_id {
                    continue;
                }

                let (v2, w) = g.edge_endpoints(next_eg_id).expect("valid g edge");
                debug_assert_eq!(v, v2);

                // Reject immediate reverse step: (u -> v) then (v -> u)
                if w == u {
                    continue;
                }

                let Some(weight) = transition_weight(
                    &mesh,
                    &vg_to_em,
                    g,
                    eg_id,
                    next_eg_id,
                    threshold_radians,
                    beta,
                ) else {
                    continue;
                };

                // avoid duplicate derivative edges too
                let src_vgp = eg_to_vgp[&eg_id];
                let dst_vgp = eg_to_vgp[&next_eg_id];
                if gprime.find_edge(src_vgp, dst_vgp).is_none() {
                    gprime.add_edge(src_vgp, dst_vgp, weight);
                }
            }
        }

        println!(
            "Derivative graph has {} vertices and {} edges",
            gprime.node_count(),
            gprime.edge_count()
        );
        println!(
            "Average out-degree: {}",
            gprime.edge_count() as f64 / gprime.node_count() as f64
        );
        print_weighted_graph_duplicate_stats(gprime, "g'");

        // ------------------------------------------------------------
        // Fast query data for interactive cycle computation
        // ------------------------------------------------------------
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
            let (src_vg, _dst_vg) = extended_neighborhood_graph
                .edge_endpoints(eg)
                .expect("valid g edge");

            source_em_of_vgp[vgp.index()] = Some(vg_to_em[&src_vg]);
        }

        let source_em_of_vgp = source_em_of_vgp
            .into_iter()
            .map(|x| x.expect("every derivative node should map to a mesh edge"))
            .collect::<Vec<_>>();

        let mut start_vgps_by_em: HashMap<ids::Key<EDGE, T>, Vec<usize>> = HashMap::new();
        for (&em_id, &vg_id) in &em_to_vg {
            let starts = extended_neighborhood_graph
                .edges_directed(vg_id, Direction::Outgoing)
                .filter_map(|e| eg_to_vgp.get(&e.id()).copied())
                .map(|vgp| vgp.index())
                .collect::<Vec<_>>();
            start_vgps_by_em.insert(em_id, starts);
        }

        Self {
            extended_neighborhood_graph,
            em_to_vg,
            vg_to_em,
            derivative_graph,
            eg_to_vgp,
            vgp_to_eg,
            start_vgps_by_em,
            mwc_from_vgp_cache: HashMap::new(),
            mwc_cache: HashMap::new(),
            out_adj,
            in_adj,
            source_em_of_vgp,
        }
    }

    pub fn compute_mwc_for_edge(
        &mut self,
        em_id: ids::Key<EDGE, T>,
    ) -> Option<(f64, Vec<ids::Key<EDGE, T>>)> {
        if let Some(cached) = self.mwc_cache.get(&em_id).cloned() {
            return cached;
        }

        let starts = self.start_vgps_by_em.get(&em_id)?;
        let mut best: Option<(f64, Vec<ids::Key<EDGE, T>>)> = None;

        for &start in starts {
            let cycle_for_start = if let Some(cached) = self.mwc_from_vgp_cache.get(&start).cloned()
            {
                cached
            } else {
                let computed = minimum_weight_cycle_fast(&self.out_adj, &self.in_adj, start);
                self.mwc_from_vgp_cache.insert(start, computed.clone());
                computed
            };

            let Some((cost, derivative_cycle)) = cycle_for_start else {
                continue;
            };

            let em_cycle = derivative_cycle
                .into_iter()
                .map(|vgp| self.source_em_of_vgp[vgp])
                .collect::<Vec<_>>();

            if best.as_ref().is_none_or(|(best_cost, _)| cost < *best_cost) {
                best = Some((cost, em_cycle));
            }
        }

        self.mwc_cache.insert(em_id, best.clone());
        best
    }
}

// --------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------
use core::cmp::Ordering;
use std::collections::BinaryHeap;

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

fn minimum_weight_cycle_fast(
    out_adj: &[Vec<(usize, f64)>],
    in_adj: &[Vec<(usize, f64)>],
    start: usize,
) -> Option<(f64, Vec<usize>)> {
    let relevant_nexts = out_adj[start]
        .iter()
        .copied()
        .filter(|(nxt, _)| *nxt != start)
        .collect::<Vec<_>>();

    if relevant_nexts.is_empty() {
        return None;
    }

    let n = out_adj.len();
    let mut dist = vec![f64::INFINITY; n];
    let mut next_toward_start = vec![usize::MAX; n];
    let mut settled = vec![false; n];
    let mut target_set = vec![false; n];
    let mut remaining_targets = 0usize;

    for &(nxt, _) in &relevant_nexts {
        if !target_set[nxt] {
            target_set[nxt] = true;
            remaining_targets += 1;
        }
    }

    let mut heap = BinaryHeap::new();
    dist[start] = 0.0;
    heap.push(MinScored(0.0, start));

    // Reverse Dijkstra from start on in_adj:
    // dist[x] = shortest cost x -> start in original graph
    while let Some(MinScored(d, node)) = heap.pop() {
        if settled[node] {
            continue;
        }
        if d > dist[node] {
            continue;
        }
        settled[node] = true;

        if target_set[node] {
            remaining_targets -= 1;
            if remaining_targets == 0 {
                break;
            }
        }

        for &(prev, w) in &in_adj[node] {
            let nd = d + w;
            if nd < dist[prev] {
                dist[prev] = nd;
                next_toward_start[prev] = node;
                heap.push(MinScored(nd, prev));
            }
        }
    }

    let mut best: Option<(f64, usize)> = None;
    for &(nxt, first_w) in &relevant_nexts {
        let back = dist[nxt];
        if !back.is_finite() {
            continue;
        }
        let total = first_w + back;
        if best
            .as_ref()
            .is_none_or(|(best_cost, _)| total < *best_cost)
        {
            best = Some((total, nxt));
        }
    }

    let (total_cost, first_next) = best?;

    let mut cycle = Vec::new();
    cycle.push(start);
    cycle.push(first_next);

    let mut cur = first_next;
    while cur != start {
        cur = next_toward_start[cur];
        if cur == usize::MAX {
            return None;
        }
        cycle.push(cur);
        if cycle.len() > n + 1 {
            return None;
        }
    }

    if cycle.len() < 4 {
        return None;
    }

    Some((total_cost, cycle))
}

fn add_directed_edge_if_missing(g: &mut DiGraph<(), ()>, src: NodeIndex, dst: NodeIndex) {
    if src == dst {
        return;
    }
    if g.find_edge(src, dst).is_none() {
        g.add_edge(src, dst, ());
    }
}

fn print_graph_duplicate_stats(g: &DiGraph<(), ()>, name: &str) {
    let total = g.edge_count();
    let unique = g
        .edge_references()
        .map(|e| (e.source(), e.target()))
        .collect::<HashSet<_>>()
        .len();

    println!(
        "{name}: total directed edges = {total}, unique (src,dst) pairs = {unique}, duplicates = {}",
        total.saturating_sub(unique)
    );
}

fn print_weighted_graph_duplicate_stats(g: &DiGraph<(), f64>, name: &str) {
    let total = g.edge_count();
    let unique = g
        .edge_references()
        .map(|e| (e.source(), e.target()))
        .collect::<HashSet<_>>()
        .len();

    println!(
        "{name}: total directed edges = {total}, unique (src,dst) pairs = {unique}, duplicates = {}",
        total.saturating_sub(unique)
    );
}

fn transition_weight<T: Tag>(
    mesh: &Mesh<T>,
    vg_to_em: &HashMap<NodeIndex, ids::Key<EDGE, T>>,
    g: &DiGraph<(), ()>,
    eg_id: EdgeIndex,
    next_eg_id: EdgeIndex,
    threshold_radians: f64,
    beta: f64,
) -> Option<f64> {
    let (u, v) = g.edge_endpoints(eg_id)?;
    let (v2, w) = g.edge_endpoints(next_eg_id)?;
    if v != v2 {
        return None;
    }

    let em_u = vg_to_em[&u];
    let em_v = vg_to_em[&v];
    let em_w = vg_to_em[&w];

    // g-nodes are mesh edges, interpreted at their midpoints
    let p_u = mesh.position(em_u);
    let p_v = mesh.position(em_v);
    let p_w = mesh.position(em_w);

    let a = p_v - p_u;
    let b = p_w - p_v;

    let l0 = a.norm();
    let l1 = b.norm();
    if l0 <= 1e-12 || l1 <= 1e-12 {
        return None;
    }

    let d0 = a / l0;
    let d1 = b / l1;

    // Turn-angle threshold
    let gamma = d0.angle(&d1);
    if gamma > threshold_radians {
        return None;
    }

    let lmin = l0.min(l1);
    if lmin <= 1e-12 {
        return None;
    }

    // Curvature term from the paper (approximation)
    let kappa2 = (gamma * gamma) / lmin;

    // Length term from the paper
    let half = gamma * 0.5;
    let length = if half.abs() < 1e-8 {
        lmin + 0.5 * (l0 - l1).abs()
    } else {
        gamma * lmin / (2.0 * half.tan()) + 0.5 * (l0 - l1).abs()
    };

    Some(length + beta * kappa2)
}
