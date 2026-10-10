//! Loop sampling and loop-query algorithms.
//!
//! Any number of loops may cross the same mesh edge (and thus pass through the same triangles). The
//! loops crossing an edge are ordered along it, and a new loop can pass through any *gap* between
//! them. Loops are therefore constructed in a gap graph: its nodes are pairs of a flow-graph node
//! (half-edge) and a gap along that half-edge. A move inside a face crosses exactly the existing
//! loops that separate its two gaps. Crossing a loop of the same direction is not allowed, crossing
//! loops of other directions is allowed (but avoided when an equally short alternative exists).
//! Existing loops thus never block the construction of a new loop; they only determine on which
//! side of them a new loop of the same direction has to run.

use crate::PropertyViolationError;
use crate::dual::{Dual, LoopRegionID, LoopSegmentID};
use crate::loops::{Loop, LoopID};
use dualcube_types::prelude::*;
use orx_parallel::*;
use rand::{
    rng,
    seq::{IteratorRandom, SliceRandom},
};
use rustc_hash::FxHashMap;
use slotmap::SlotMap;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::rc::Rc;
use std::time::Instant;

pub type FlowGraph = grapff::fixed::FixedGraph<EdgeID, f64>;

/// Scales of the A* lower bounds of a flow graph and measure (see `LoopSampler::flow_graph_heuristic`).
#[derive(Clone, Copy, Debug)]
pub struct HeuristicScale {
    euclidean: f64,
    landmark: f64,
}

impl HeuristicScale {
    const NONE: Self = Self {
        euclidean: 0.,
        landmark: 0.,
    };
}

/// A node of the gap graph: a half-edge and a gap along it. Gap `i` lies between the `i-1`th and
/// the `i`th loop crossing the half-edge (counted from the root of the half-edge).
pub type GapNode = (EdgeID, usize);

// Cost of crossing a loop of the same direction when inserting a loop into the existing loop orders.
const SAME_DIRECTION_CROSSING_COST: f64 = 1e6;
// Fraction of the gap between two neighboring loops that a new loop keeps free on both sides.
const LOOP_MARGIN: f64 = 0.1;
// Fraction of an edge that a loop keeps free from its endpoints (mesh vertices).
const VERTEX_MARGIN: f64 = 0.01;
// Moving across the target direction of a loop costs this much more than moving along it (squared), see `pull_taut`.
const ANISOTROPY: f64 = 4.;
// Sampled loops must separate at least this fraction of the surface area from the rest.
const MIN_SEPARATED_AREA: f64 = 0.02;

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

// A position on the boundary of a face. Sides are the half-edges of the face (in order), and positions
// along a side are ordered from root to tip, such that the coordinates follow the boundary cyclically.
fn boundary_coord(side: usize, sub: usize) -> u64 {
    ((side as u64) << 32) | sub as u64
}

// Whether `x` lies strictly between `p` and `q`, when walking along the (cyclic) boundary from `p` to `q`.
fn strictly_between(x: u64, p: u64, q: u64) -> bool {
    if p < q {
        p < x && x < q
    } else {
        x > p || x < q
    }
}

// An existing loop passing through a face, given by its two boundary coordinates.
#[derive(Clone, Copy, Debug)]
struct FaceChord {
    loop_id: LoopID,
    direction: Direction,
    ends: [u64; 2],
}

impl FaceChord {
    // Whether this chord separates the boundary coordinates `p` and `q` (i.e., the chord from `p` to `q` crosses it).
    fn separates(&self, p: u64, q: u64) -> bool {
        strictly_between(self.ends[0], p, q) != strictly_between(self.ends[1], p, q)
    }
}

// Number of crossings of the chord from `p` to `q` with chords of the same direction and with chords of other directions.
fn count_crossings(
    chords: &[FaceChord],
    p: u64,
    q: u64,
    direction: Option<Direction>,
) -> (usize, usize) {
    chords
        .iter()
        .filter(|chord| chord.separates(p, q))
        .fold((0, 0), |(same, other), chord| {
            if Some(chord.direction) == direction {
                (same + 1, other)
            } else {
                (same, other + 1)
            }
        })
}

#[derive(Clone, Copy, Debug)]
struct SearchEntry {
    priority: OrderedFloat<f64>,
    crossings: usize,
    insertion_order: usize,
    node: GapNode,
}

impl PartialEq for SearchEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for SearchEntry {}

impl Ord for SearchEntry {
    // Reversed, such that the binary (max-)heap pops the entry with the lowest priority first.
    fn cmp(&self, other: &Self) -> Ordering {
        (other.priority, other.crossings, other.insertion_order).cmp(&(
            self.priority,
            self.crossings,
            self.insertion_order,
        ))
    }
}

impl PartialOrd for SearchEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// The gap graph of a flow graph, given the loops that currently exist. Its edges are computed on the fly.
struct GapGraph<'s, 'a, F: Fn(f64) -> OrderedFloat<f64>> {
    sampler: &'s LoopSampler<'a>,
    graph: &'s FlowGraph,
    // Direction of the loop that is being constructed. Crossing loops of this direction is not allowed.
    direction: Option<Direction>,
    measure: &'s F,
    chords: RefCell<FxHashMap<FaceID, Rc<Vec<FaceChord>>>>,
}

impl<'s, 'a, F: Fn(f64) -> OrderedFloat<f64>> GapGraph<'s, 'a, F> {
    fn new(
        sampler: &'s LoopSampler<'a>,
        graph: &'s FlowGraph,
        direction: Option<Direction>,
        measure: &'s F,
    ) -> Self {
        Self {
            sampler,
            graph,
            direction,
            measure,
            chords: RefCell::new(FxHashMap::default()),
        }
    }

    fn gaps(&self, edge: EdgeID) -> usize {
        self.sampler.crossing_count(edge) + 1
    }

    fn chords(&self, face: FaceID) -> Rc<Vec<FaceChord>> {
        self.chords
            .borrow_mut()
            .entry(face)
            .or_insert_with(|| Rc::new(self.sampler.face_chords(face)))
            .clone()
    }

    // All moves from a gap node: (target, measured cost, number of crossed loops).
    fn successors(&self, node: GapNode) -> Vec<(GapNode, f64, usize)> {
        let mut successors = vec![];
        self.successors_into(node, &mut successors);
        successors
    }

    // As `successors`, but into a reusable buffer (it is cleared first): searches expand very many nodes.
    fn successors_into(&self, (edge, gap): GapNode, successors: &mut Vec<(GapNode, f64, usize)>) {
        successors.clear();
        let mesh = self.sampler.mesh_ref;
        for (next, weight) in self.graph.outgoing(edge) {
            let cost = *(self.measure)(weight);
            if mesh.twin(edge) == next {
                let n = self.sampler.crossing_count(edge);
                successors.push(((next, n - gap.min(n)), cost, 0));
            } else if mesh.face(edge) == mesh.face(next) {
                let chords = self.chords(mesh.face(edge));
                let p = self.sampler.gap_coord(edge, gap);
                for next_gap in 0..self.gaps(next) {
                    let q = self.sampler.gap_coord(next, next_gap);
                    let (same, other) = count_crossings(&chords, p, q, self.direction);
                    if same == 0 {
                        successors.push(((next, next_gap), cost, other));
                    }
                }
            }
        }
    }

    // A* search from the given start nodes (with initial costs) to the goal half-edge (and optionally a specific gap).
    // Paths are compared by their cost first, and by the number of crossed loops second.
    fn search(
        &self,
        starts: &[(GapNode, f64, usize)],
        goal: EdgeID,
        goal_gap: Option<usize>,
        heuristic_scale: HeuristicScale,
    ) -> Option<(Vec<GapNode>, f64, usize)> {
        let is_goal = |node: GapNode| node.0 == goal && goal_gap.is_none_or(|g| node.1 == g);
        let heuristic = self
            .sampler
            .flow_graph_heuristic(self.graph, goal, heuristic_scale);
        let heuristic = |node: GapNode| heuristic(node.0);

        let mut best: FxHashMap<GapNode, (OrderedFloat<f64>, usize)> = FxHashMap::default();
        let mut previous: FxHashMap<GapNode, GapNode> = FxHashMap::default();
        let mut successors = vec![];
        let mut heap = BinaryHeap::new();
        let mut counter = 0usize;

        for &(node, cost, crossings) in starts {
            let key = (OrderedFloat(cost), crossings);
            if best.get(&node).is_none_or(|&current| key < current) {
                best.insert(node, key);
                previous.remove(&node);
                heap.push(SearchEntry {
                    priority: OrderedFloat(cost + heuristic(node)),
                    crossings,
                    insertion_order: counter,
                    node,
                });
                counter += 1;
            }
        }

        while let Some(SearchEntry {
            priority,
            crossings,
            node,
            ..
        }) = heap.pop()
        {
            let (cost, node_crossings) = best[&node];
            if *priority > *cost + heuristic(node) || crossings > node_crossings {
                continue;
            }

            if is_goal(node) {
                let mut path = vec![node];
                let mut current = node;
                while let Some(&prev) = previous.get(&current) {
                    current = prev;
                    path.push(current);
                }
                path.reverse();
                return Some((path, *cost, node_crossings));
            }

            self.successors_into(node, &mut successors);
            for &(next, step_cost, step_crossings) in &successors {
                let key = (
                    OrderedFloat(*cost + step_cost),
                    node_crossings + step_crossings,
                );
                if best.get(&next).is_none_or(|&current| key < current) {
                    best.insert(next, key);
                    previous.insert(next, node);
                    heap.push(SearchEntry {
                        priority: OrderedFloat(*key.0 + heuristic(next)),
                        crossings: key.1,
                        insertion_order: counter,
                        node: next,
                    });
                    counter += 1;
                }
            }
        }

        None
    }
}

// Loops are kept at a distance from each other in the constructions (see `construct_crossing_loop_avoiding`): moves
// within this many faces of a loop cost more, by up to this factor (per nearby loop).
const PROXIMITY_HOPS: usize = 3;
const PROXIMITY_WEIGHT: f64 = 2.;

pub struct LoopSampler<'a> {
    mesh_ref: &'a Mesh<INPUT>,
    loops: &'a SlotMap<LoopID, Loop>,
    occupied: &'a ids::SecMap<EDGE, INPUT, Vec<LoopID>>,
    flow_graphs: Option<&'a [FlowGraph; 3]>,
    // The proximity of the loops to every face (see `loop_proximity`), computed once (the loops do not change).
    proximity: std::sync::OnceLock<FxHashMap<FaceID, f64>>,
}

impl<'a> LoopSampler<'a> {
    #[must_use]
    pub fn new(
        mesh_ref: &'a Mesh<INPUT>,
        loops: &'a SlotMap<LoopID, Loop>,
        occupied: &'a ids::SecMap<EDGE, INPUT, Vec<LoopID>>,
        flow_graphs: Option<&'a [FlowGraph; 3]>,
    ) -> Self {
        Self {
            mesh_ref,
            loops,
            occupied,
            flow_graphs,
            proximity: std::sync::OnceLock::new(),
        }
    }

    /// Add a cost per face to the constructions (on top of the proximity of the loops; in units of
    /// `PROXIMITY_WEIGHT`, see `loop_proximity`).
    #[must_use]
    pub fn with_extra_proximity(self, extra: &HashMap<FaceID, f64>) -> Self {
        let mut proximity = self.loop_proximity();
        for (&face, &value) in extra {
            *proximity.entry(face).or_default() += value;
        }
        let _ = self.proximity.set(proximity);
        self
    }

    /// The positions of loop `l` on all its edges (in order).
    pub fn get_coordinates_of_loop(&self, l: LoopID) -> Vec<Vector3D> {
        let lewp = &self.loops[l];
        lewp.edges
            .iter()
            .enumerate()
            .map(|(i, &edge)| {
                let offset = lewp.offsets.get(i).copied().unwrap_or(0.5);
                self.mesh_ref.midpoint_offset(edge, offset)
            })
            .collect()
    }

    pub fn get_loops_in_direction(&self, direction: Direction) -> Vec<LoopID> {
        self.loops
            .iter()
            .filter_map(|(id, l)| {
                if l.direction == direction {
                    Some(id)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn loop_to_direction(&self, loop_id: LoopID) -> Direction {
        self.loops[loop_id].direction
    }

    pub fn get_pairs_of_loop(&self, loop_id: LoopID) -> Vec<[EdgeID; 2]> {
        self.get_pairs_of_sequence(&self.loops[loop_id].edges)
    }

    pub fn get_pairs_of_sequence(&self, sequence: &[EdgeID]) -> Vec<[EdgeID; 2]> {
        sequence
            .windows(2)
            .filter_map(|w| {
                if self.mesh_ref.twin(w[0]) == w[1] {
                    None
                } else {
                    Some([w[0], w[1]])
                }
            })
            .collect()
    }

    pub fn cycled_windows(sequence: &[EdgeID]) -> Vec<[EdgeID; 2]> {
        (0..sequence.len())
            .map(|i| {
                let a = sequence[i];
                let b = sequence[(i + 1) % sequence.len()];
                [a, b]
            })
            .collect_vec()
    }

    /// The loops crossing the given half-edge, ordered along the half-edge (from root to tip).
    pub fn loops_on_edge(&self, edge: EdgeID) -> Vec<LoopID> {
        self.loops_on_edge_ref(edge).to_vec()
    }

    fn loops_on_edge_ref(&self, edge: EdgeID) -> &[LoopID] {
        self.occupied.get(&edge).map_or(&[], Vec::as_slice)
    }

    /// The number of loops crossing the given half-edge.
    pub fn crossing_count(&self, edge: EdgeID) -> usize {
        self.loops_on_edge_ref(edge).len()
    }

    // Index of the half-edge in its face.
    fn side_in_face(&self, edge: EdgeID) -> usize {
        self.mesh_ref
            .edges(self.mesh_ref.face(edge))
            .position(|e| e == edge)
            .unwrap()
    }

    // Boundary coordinate (in the face of `edge`) of gap `gap` along `edge`.
    fn gap_coord(&self, edge: EdgeID, gap: usize) -> u64 {
        boundary_coord(self.side_in_face(edge), 2 * gap)
    }

    // All existing loops passing through the face.
    fn face_chords(&self, face: FaceID) -> Vec<FaceChord> {
        let mut ends: HashMap<LoopID, Vec<u64>> = HashMap::new();
        for (side, edge) in self.mesh_ref.edges(face).enumerate() {
            for (rank, &loop_id) in self.loops_on_edge_ref(edge).iter().enumerate() {
                ends.entry(loop_id)
                    .or_default()
                    .push(boundary_coord(side, 2 * rank + 1));
            }
        }
        ends.into_iter()
            .filter_map(|(loop_id, ends)| {
                (ends.len() == 2).then(|| FaceChord {
                    loop_id,
                    direction: self.loops[loop_id].direction,
                    ends: [ends[0], ends[1]],
                })
            })
            .collect()
    }

    /// Decide where to insert a new loop into the existing loop orders. For every crossing of the loop
    /// (given by the index `i` such that `edges[i]` and `edges[i + 1]` are twins), returns the gap along
    /// `edges[i]`. The gaps are chosen to avoid crossing loops of the same direction, and to minimize the
    /// number of crossings with loops of other directions (exactly, by dynamic programming over the cycle).
    pub fn assign_gaps(&self, edges: &[EdgeID], direction: Direction) -> Vec<(usize, usize)> {
        let timer = Instant::now();
        let len = edges.len();
        if len < 2 {
            return vec![];
        }
        let exits = (0..len)
            .filter(|&i| self.mesh_ref.twin(edges[i]) == edges[(i + 1) % len])
            .collect_vec();
        let m = exits.len();
        if m == 0 {
            return vec![];
        }
        let counts = exits
            .iter()
            .map(|&i| self.crossing_count(edges[i]))
            .collect_vec();
        if counts.iter().all(|&n| n == 0) {
            return exits.into_iter().map(|i| (i, 0)).collect();
        }

        // For the face between crossing `k` and crossing `k + 1`: the cost of every combination of gaps.
        let transitions = (0..m)
            .map(|k| {
                let k1 = (k + 1) % m;
                let entry = edges[(exits[k] + 1) % len];
                let exit = edges[exits[k1]];
                let (nk, nk1) = (counts[k], counts[k1]);
                let face = self.mesh_ref.face(entry);
                if face != self.mesh_ref.face(exit) {
                    return vec![vec![0.; nk1 + 1]; nk + 1];
                }
                let chords = self.face_chords(face);
                (0..=nk)
                    .map(|gk| {
                        let p = self.gap_coord(entry, nk - gk);
                        (0..=nk1)
                            .map(|gk1| {
                                let q = self.gap_coord(exit, gk1);
                                let (same, other) = count_crossings(&chords, p, q, Some(direction));
                                same as f64 * SAME_DIRECTION_CROSSING_COST + other as f64
                            })
                            .collect_vec()
                    })
                    .collect_vec()
            })
            .collect_vec();

        let mut best: Option<(f64, Vec<usize>)> = None;
        for g0 in 0..=counts[0] {
            let mut dp = vec![f64::INFINITY; counts[0] + 1];
            dp[g0] = 0.;
            let mut back: Vec<Vec<usize>> = Vec::with_capacity(m);
            for k in 0..m - 1 {
                let mut next = vec![f64::INFINITY; counts[k + 1] + 1];
                let mut arg = vec![0; counts[k + 1] + 1];
                for (g, &cost) in dp.iter().enumerate() {
                    if cost.is_infinite() {
                        continue;
                    }
                    for (g1, &step) in transitions[k][g].iter().enumerate() {
                        if cost + step < next[g1] {
                            next[g1] = cost + step;
                            arg[g1] = g;
                        }
                    }
                }
                back.push(arg);
                dp = next;
            }

            let Some((g_last, total)) = dp
                .iter()
                .enumerate()
                .filter(|(_, cost)| cost.is_finite())
                .map(|(g, &cost)| (g, cost + transitions[m - 1][g][g0]))
                .min_by_key(|&(_, total)| OrderedFloat(total))
            else {
                continue;
            };

            if best
                .as_ref()
                .is_none_or(|(best_total, _)| total < *best_total)
            {
                let mut gaps = vec![0; m];
                gaps[m - 1] = g_last;
                for k in (0..m - 1).rev() {
                    gaps[k] = back[k][gaps[k + 1]];
                }
                gaps[0] = g0;
                best = Some((total, gaps));
            }
        }

        let (total, gaps) = best.unwrap_or_else(|| (f64::INFINITY, vec![0; m]));
        debug!(
            "b_loops::assign_gaps: direction={direction:?} crossings={m} shared_crossings={} cost={total} elapsed_ms={:.3}",
            counts.iter().filter(|&&n| n > 0).count(),
            elapsed_ms(timer)
        );
        if total >= SAME_DIRECTION_CROSSING_COST {
            debug!("assign_gaps: the loop necessarily crosses a loop of the same direction");
        }

        exits.into_iter().zip(gaps).collect()
    }

    /// Where to insert a new loop: the gaps (see `assign_gaps`), and the offsets of the loop along all its
    /// half-edges. Inside its gaps, the loop is pulled taut (straightened to a locally shortest loop in its strip
    /// of faces), instead of crossing the edges at their midpoints.
    pub fn place_loop(
        &self,
        edges: &[EdgeID],
        direction: Direction,
    ) -> (Vec<(usize, usize)>, Vec<f64>) {
        self.place_loop_with(edges, direction, true)
    }

    /// As `place_loop`, but with the given gaps (e.g., from `construct_valid_loop`), such that the loop crosses
    /// exactly the loops it was constructed to cross.
    pub fn place_loop_in_gaps(
        &self,
        edges: &[EdgeID],
        direction: Direction,
        gaps: Vec<(usize, usize)>,
    ) -> (Vec<(usize, usize)>, Vec<f64>) {
        self.place_loop_with_gaps(edges, direction, true, gaps)
    }

    // See `place_loop`; without pulling the loop taut, the loop crosses the middle of its gaps.
    fn place_loop_with(
        &self,
        edges: &[EdgeID],
        direction: Direction,
        taut: bool,
    ) -> (Vec<(usize, usize)>, Vec<f64>) {
        let gaps = self.assign_gaps(edges, direction);
        self.place_loop_with_gaps(edges, direction, taut, gaps)
    }

    fn place_loop_with_gaps(
        &self,
        edges: &[EdgeID],
        direction: Direction,
        taut: bool,
        gaps: Vec<(usize, usize)>,
    ) -> (Vec<(usize, usize)>, Vec<f64>) {
        let len = edges.len();
        let mut offsets = vec![0.5; len];
        if gaps.is_empty() {
            return (gaps, offsets);
        }

        // The interval (along the exiting half-edge) between the neighboring loops of every crossing.
        let intervals = gaps
            .iter()
            .map(|&(exit, gap)| {
                let edge = edges[exit];
                let list = self.loops_on_edge_ref(edge);
                let gap = gap.min(list.len());
                let lo = if gap == 0 {
                    0.
                } else {
                    self.loops[list[gap - 1]].offset(edge).unwrap_or(0.)
                };
                let hi = if gap == list.len() {
                    1.
                } else {
                    self.loops[list[gap]].offset(edge).unwrap_or(1.)
                };
                // Keep some distance to neighboring loops, but only a tiny distance to the mesh vertices (and never
                // more than a quarter of the gap, as neighboring loops can be arbitrarily close).
                let gap_width = (hi - lo).max(0.);
                let margin = |at_vertex: bool| {
                    if at_vertex {
                        VERTEX_MARGIN
                    } else {
                        LOOP_MARGIN * gap_width
                    }
                    .min(0.25 * gap_width)
                };
                (lo + margin(gap == 0), hi - margin(gap == list.len()))
            })
            .collect_vec();
        let exits = gaps.iter().map(|&(exit, _)| exit).collect_vec();
        let mut t = intervals
            .iter()
            .map(|&(lo, hi)| (lo + hi) / 2.)
            .collect_vec();
        if taut {
            self.pull_taut(edges, &exits, &intervals, &mut t, Vector3D::from(direction));
        }

        for (k, &exit) in exits.iter().enumerate() {
            offsets[exit] = t[k];
            offsets[(exit + 1) % len] = 1. - t[k];
        }
        (gaps, offsets)
    }

    /// The positions of a (not yet inserted) loop, as it would be placed by `place_loop`.
    pub fn loop_positions(&self, edges: &[EdgeID], direction: Direction) -> Vec<Vector3D> {
        self.loop_positions_with(edges, direction, true)
    }

    /// Approximate positions of a (not yet inserted) loop: as `loop_positions`, but without pulling the loop taut
    /// (much faster, e.g., for interactive previews).
    pub fn loop_positions_preview(&self, edges: &[EdgeID], direction: Direction) -> Vec<Vector3D> {
        self.loop_positions_with(edges, direction, false)
    }

    fn loop_positions_with(
        &self,
        edges: &[EdgeID],
        direction: Direction,
        taut: bool,
    ) -> Vec<Vector3D> {
        let (_, offsets) = self.place_loop_with(edges, direction, taut);
        edges
            .iter()
            .zip(offsets)
            .map(|(&edge, offset)| self.mesh_ref.midpoint_offset(edge, offset))
            .collect()
    }

    // Shorten a loop by moving its crossings (`t[k]` along half-edge `edges[exits[k]]`) inside their intervals,
    // until it is taut. Lengths are measured in an anisotropic metric that penalizes moving across the target
    // direction of the loop in each face: around the axis (cf. Campen et al., Dual Loops Meshing). Loops thus cross
    // sharp features orthogonally instead of being pulled along them.
    //
    // Every crossing is affine in its parameter, and inside a face the metric is a norm, so the total length is a
    // convex function of `t` (with box constraints). Its Hessian is cyclic tridiagonal (every segment couples two
    // consecutive crossings), so it is minimized with a projected Newton method in O(m) per iteration; see
    // `minimize_taut`. (Coordinate-wise relaxation needs O(m^2) sweeps on long loops.)
    fn pull_taut(
        &self,
        edges: &[EdgeID],
        exits: &[usize],
        intervals: &[(f64, f64)],
        t: &mut [f64],
        axis: Vector3D,
    ) {
        let m = exits.len();
        if m < 3 {
            return;
        }
        let mesh = self.mesh_ref;
        let origins = exits
            .iter()
            .map(|&exit| mesh.position(mesh.root(edges[exit])))
            .collect_vec();
        let directions = exits
            .iter()
            .map(|&exit| mesh.vector(edges[exit]))
            .collect_vec();
        // The segment from crossing `k` to crossing `k + 1` lies in the face of the twin of the crossed half-edge.
        let metrics = exits
            .iter()
            .map(|&exit| {
                if ANISOTROPY <= 1. {
                    return Metric::ISOTROPIC;
                }
                let face = mesh.face(mesh.twin(edges[exit]));
                anisotropic_metric(mesh.normal(face).normalize(), axis)
            })
            .collect_vec();
        minimize_taut(&origins, &directions, &metrics, intervals, t);
    }

    /// The area of the smaller of the parts of the surface that a loop separates (zero if it does not separate).
    /// Whether the loop separates the surface and cuts off a part with less than `min_area` area. Loops that do not
    /// separate the surface (around a handle, for surfaces of higher genus) never cut off a small part.
    pub fn cuts_off_small_part(&self, edges: &[EdgeID], min_area: f64) -> bool {
        let separated = self.separated_area(edges);
        separated > 0. && separated < min_area
    }

    pub fn separated_area(&self, edges: &[EdgeID]) -> f64 {
        let mesh = self.mesh_ref;
        let mut crossed: ids::SecMap<EDGE, INPUT, ()> = ids::SecMap::new();
        for edge in edges {
            crossed.insert(edge, ());
        }
        let mut seen: ids::SecMap<FACE, INPUT, ()> = ids::SecMap::new();
        let mut areas = vec![];
        for start in mesh.face_ids() {
            if seen.contains_key(&start) {
                continue;
            }
            seen.insert(&start, ());
            let mut area = 0.;
            let mut stack = vec![start];
            while let Some(face) = stack.pop() {
                area += mesh.size(face);
                for edge in mesh.edges(face) {
                    if crossed.contains_key(&edge) {
                        continue;
                    }
                    let neighbor = mesh.face(mesh.twin(edge));
                    if !seen.contains_key(&neighbor) {
                        seen.insert(&neighbor, ());
                        stack.push(neighbor);
                    }
                }
            }
            areas.push(area);
        }
        if areas.len() < 2 {
            return 0.;
        }
        let total: f64 = areas.iter().sum();
        let largest = areas.iter().copied().fold(0., f64::max);
        total - largest
    }

    pub fn check_loop(&self, lewp: &[EdgeID]) -> Result<(), PropertyViolationError> {
        let timer = Instant::now();
        let edges = lewp;

        if edges.len() < 2 {
            debug!(
                "b_loops::check_loop: rejected reason=empty elapsed_ms={:.3}",
                elapsed_ms(timer)
            );
            return Err(PropertyViolationError::UnknownError);
        }

        // NOTE: edges (and pairs of edges) that are already occupied by other loops are fine; loops are
        // ordered along the edges they share.

        // Check if the loop contains any duplicates
        let mut seen = HashSet::new();
        for (edge_index, &edge) in edges.iter().enumerate() {
            if !seen.insert(edge) {
                debug!(
                    "b_loops::check_loop: rejected reason=duplicate edges={} edge_index={} elapsed_ms={:.3}",
                    edges.len(),
                    edge_index,
                    elapsed_ms(timer)
                );
                return Err(PropertyViolationError::UnknownError);
            }
        }

        // Check if the loop is valid.
        // A loop should alternate between edges that are twins, and edges that share a face.
        // If `alternate` is true, then the next edge should be a twin of the current edge.
        let mut alternate = self.mesh_ref.twin(edges[0]) == edges[1];
        for (pair_index, edge_pair) in Self::cycled_windows(edges).into_iter().enumerate() {
            if alternate {
                if self.mesh_ref.twin(edge_pair[0]) != edge_pair[1] {
                    debug!(
                        "b_loops::check_loop: rejected reason=not_twins edges={} pair_index={} elapsed_ms={:.3}",
                        edges.len(),
                        pair_index,
                        elapsed_ms(timer)
                    );
                    return Err(PropertyViolationError::UnknownError);
                }
                alternate = false;
            } else {
                if self.mesh_ref.face(edge_pair[0]) != self.mesh_ref.face(edge_pair[1]) {
                    debug!(
                        "b_loops::check_loop: rejected reason=not_same_face edges={} pair_index={} elapsed_ms={:.3}",
                        edges.len(),
                        pair_index,
                        elapsed_ms(timer)
                    );
                    return Err(PropertyViolationError::UnknownError);
                }
                alternate = true;
            }
        }

        debug!(
            "b_loops::check_loop: ok edges={} elapsed_ms={:.3}",
            edges.len(),
            elapsed_ms(timer)
        );
        Ok(())
    }

    // The parameter (from root to tip) of the middle of gap `gap` along the half-edge.
    fn gap_parameter(&self, edge: EdgeID, gap: usize) -> f64 {
        let list = self.loops_on_edge_ref(edge);
        let gap = gap.min(list.len());
        let lo = if gap == 0 {
            0.
        } else {
            self.loops[list[gap - 1]].offset(edge).unwrap_or(0.)
        };
        let hi = if gap == list.len() {
            1.
        } else {
            self.loops[list[gap]].offset(edge).unwrap_or(1.)
        };
        (lo + hi) / 2.
    }

    /// The loop region (of the given dual structure, which must be built from the current loops) of a gap node.
    pub fn region_of_gap(&self, dual: &Dual, (edge, gap): GapNode) -> Option<LoopRegionID> {
        dual.region_on_edge(edge, self.gap_parameter(edge, gap))
    }

    /// Construct a loop of the given axis that follows the given topological structure (a cycle of the paper's
    /// filtered graph, see `Dual::valid_cycles`): starting at the gap node `start` (in the first region of the
    /// cycle), it stays inside every region of the cycle, and leaves it only by crossing the loop of the segment
    /// through which the cycle enters the next region (crossing exactly one loop at a time). The resulting loop is
    /// therefore valid by construction, if it is inserted into the gaps it was constructed in (returned as
    /// `(exit index, gap)` pairs, see `place_loop_in_gaps`). Returns the edges, the cost, and the gaps.
    #[allow(clippy::type_complexity)]
    pub fn construct_valid_loop(
        &self,
        dual: &Dual,
        axis: Direction,
        start: GapNode,
        cycle: &[LoopSegmentID],
        measure: &impl Fn(f64) -> OrderedFloat<f64>,
    ) -> Option<(Vec<EdgeID>, f64, Vec<(usize, usize)>)> {
        let k = cycle.len();
        let structure = &dual.loop_structure;
        let regions = cycle.iter().map(|&s| structure.face(s)).collect_vec();
        let crossings = (0..k)
            .map(|i| dual.segment_to_loop(cycle[(i + 1) % k]))
            .collect_vec();
        self.construct_crossing_loop_retrying(
            axis,
            start,
            &crossings,
            Some((dual, &regions)),
            measure,
        )
    }

    /// Construct a loop of the given axis from the gap node `start` that crosses exactly the given loops, in the given
    /// order (and no other loops), e.g., a loop that crosses one loop twice, or two loops alternately (as needed to
    /// initialize a loop structure). Returns the edges, the cost, and the gaps (see `construct_valid_loop`).
    #[allow(clippy::type_complexity)]
    pub fn construct_crossing_loop(
        &self,
        axis: Direction,
        start: GapNode,
        crossings: &[LoopID],
        measure: &impl Fn(f64) -> OrderedFloat<f64>,
    ) -> Option<(Vec<EdgeID>, f64, Vec<(usize, usize)>)> {
        self.construct_crossing_loop_retrying(axis, start, crossings, None, measure)
    }

    #[allow(clippy::type_complexity)]
    fn construct_crossing_loop_retrying(
        &self,
        axis: Direction,
        start: GapNode,
        crossings: &[LoopID],
        regions: Option<(&Dual, &[LoopRegionID])>,
        measure: &impl Fn(f64) -> OrderedFloat<f64>,
    ) -> Option<(Vec<EdgeID>, f64, Vec<(usize, usize)>)> {
        let proximity = self.proximity.get_or_init(|| self.loop_proximity());
        // A loop cannot cross the same mesh edge twice. If the cheapest path does, block those edges and search again.
        let mut blocked = HashSet::new();
        for _ in 0..3 {
            match self.construct_crossing_loop_avoiding(
                axis, start, crossings, regions, measure, proximity, &blocked,
            ) {
                Ok(found) => return Some(found),
                Err(Some(revisited)) => {
                    let mesh = self.mesh_ref;
                    let before = blocked.len();
                    blocked.extend(revisited.iter().flat_map(|&e| [e, mesh.twin(e)]));
                    if blocked.len() == before {
                        return None;
                    }
                }
                Err(None) => return None,
            }
        }
        None
    }

    // See `construct_valid_loop`; never crosses the blocked half-edges. On failure, returns the mesh edges that the
    // cheapest path crosses more than once (if any).
    #[allow(clippy::type_complexity)]
    // For every face near a loop: how close the loops are, summed over the loops within `PROXIMITY_HOPS` faces (1 for
    // a face that a loop passes through, decreasing linearly to 0 beyond `PROXIMITY_HOPS`).
    fn loop_proximity(&self) -> FxHashMap<FaceID, f64> {
        let mesh = self.mesh_ref;
        let mut proximity: FxHashMap<FaceID, f64> = FxHashMap::default();
        for lewp in self.loops.values() {
            let mut hops: FxHashMap<FaceID, usize> = FxHashMap::default();
            let mut frontier = lewp.edges.iter().map(|&e| mesh.face(e)).collect_vec();
            for &face in &frontier {
                hops.insert(face, 0);
            }
            for hop in 1..=PROXIMITY_HOPS {
                let mut next = vec![];
                for face in frontier {
                    for edge in mesh.edges(face) {
                        let neighbor = mesh.face(mesh.twin(edge));
                        if let std::collections::hash_map::Entry::Vacant(entry) =
                            hops.entry(neighbor)
                        {
                            entry.insert(hop);
                            next.push(neighbor);
                        }
                    }
                }
                frontier = next;
            }
            for (face, hop) in hops {
                *proximity.entry(face).or_default() +=
                    (PROXIMITY_HOPS + 1 - hop) as f64 / (PROXIMITY_HOPS + 1) as f64;
            }
        }
        proximity
    }

    // The search of `construct_crossing_loop` (and, with regions, `construct_valid_loop`): crossing the loops in the
    // given order, entering the given regions (the i-th crossing enters region i + 1). Never crosses the blocked
    // half-edges. Moves near existing loops cost more (by `PROXIMITY_WEIGHT` times their `proximity`), such that loops
    // keep some distance from each other (running alongside a loop, or crossing a loop at an existing intersection,
    // leaves narrow corridors for the layout); every move costs at least its flow cost, so the heuristic remains a
    // lower bound.
    #[allow(clippy::too_many_arguments)]
    fn construct_crossing_loop_avoiding(
        &self,
        axis: Direction,
        start: GapNode,
        crossing_loops: &[LoopID],
        regions: Option<(&Dual, &[LoopRegionID])>,
        measure: &impl Fn(f64) -> OrderedFloat<f64>,
        proximity: &FxHashMap<FaceID, f64>,
        blocked: &HashSet<EdgeID>,
    ) -> Result<(Vec<EdgeID>, f64, Vec<(usize, usize)>), Option<Vec<EdgeID>>> {
        let k = crossing_loops.len();
        if k < 2 {
            return Err(None);
        }
        if let Some((dual, regions)) = regions
            && self.region_of_gap(dual, start) != Some(regions[0])
        {
            return Err(None);
        }
        let mut region_cache: FxHashMap<GapNode, Option<LoopRegionID>> = FxHashMap::default();
        // Progress: the number of crossings so far.
        self.search_loop(
            axis,
            start,
            start,
            0,
            k,
            |index: usize, chord: &FaceChord, target: GapNode| {
                if index >= k || chord.loop_id != crossing_loops[index] {
                    return vec![];
                }
                if let Some((dual, regions)) = regions {
                    let region = *region_cache
                        .entry(target)
                        .or_insert_with(|| self.region_of_gap(dual, target));
                    if region != Some(regions[(index + 1) % k]) {
                        return vec![];
                    }
                }
                vec![index + 1]
            },
            measure,
            proximity,
            blocked,
        )
    }

    /// The cheapest path of the given axis from the gap node `from` to the gap node `to` (ending with the half-edge
    /// through which it enters the face of `to.0`), e.g., to route part of a loop again. It crosses any loops except
    /// those of its own axis (in any order), and never enters the given faces (except the face of `to.0`, through
    /// `to.0`). Returns the edges (from `from.0` to the twin of `to.0`), the cost, and the gaps of its crossings (by the
    /// index of the half-edge through which it leaves a face, see `construct_valid_loop`).
    #[allow(clippy::type_complexity)]
    pub fn construct_open_path(
        &self,
        axis: Direction,
        from: GapNode,
        to: GapNode,
        avoid: &HashSet<FaceID>,
        measure: &impl Fn(f64) -> OrderedFloat<f64>,
    ) -> Option<(Vec<EdgeID>, f64, Vec<(usize, usize)>)> {
        let mesh = self.mesh_ref;
        let proximity = self.proximity.get_or_init(|| self.loop_proximity());
        // Leaving a face through a blocked half-edge would enter an avoided face.
        let mut blocked: HashSet<EdgeID> = avoid
            .iter()
            .flat_map(|&face| mesh.edges(face).map(|edge| mesh.twin(edge)).collect_vec())
            .collect();
        blocked.remove(&mesh.twin(to.0));
        self.search_loop(
            axis,
            from,
            to,
            (),
            (),
            |(), chord: &FaceChord, _| {
                if self.loops[chord.loop_id].direction == axis {
                    vec![]
                } else {
                    vec![()]
                }
            },
            measure,
            proximity,
            &blocked,
        )
        .ok()
    }

    // The cheapest valid loop of the given axis from `start` over all sequences of regions (the free search of
    // `free_valid_loop`): the loop is assumed to have entered the start region through `entry` (a segment of its
    // boundary), crosses a loop only where that is a valid exit (see `Dual::valid_exit`) from the segment it last
    // entered through, and closes by entering the start region through `entry` again.
    #[allow(clippy::too_many_arguments)]
    fn construct_free_loop_avoiding(
        &self,
        dual: &Dual,
        axis: Direction,
        start: GapNode,
        entry: LoopSegmentID,
        measure: &impl Fn(f64) -> OrderedFloat<f64>,
        proximity: &FxHashMap<FaceID, f64>,
        blocked: &HashSet<EdgeID>,
    ) -> Result<(Vec<EdgeID>, f64, Vec<(usize, usize)>), Option<Vec<EdgeID>>> {
        let structure = &dual.loop_structure;
        if self.region_of_gap(dual, start) != Some(structure.face(entry)) {
            return Err(None);
        }
        let mut region_cache: FxHashMap<GapNode, Option<LoopRegionID>> = FxHashMap::default();
        // Progress: the segment through which the loop entered its current region, and whether it crossed a loop yet.
        self.search_loop(
            axis,
            start,
            start,
            (entry, false),
            (entry, true),
            |(current, _): (LoopSegmentID, bool), chord: &FaceChord, target: GapNode| {
                let Some(region) = *region_cache
                    .entry(target)
                    .or_insert_with(|| self.region_of_gap(dual, target))
                else {
                    return vec![];
                };
                structure
                    .edges(structure.face(current))
                    .filter(|&exit| {
                        dual.segment_to_loop(exit) == chord.loop_id
                            && structure.face(structure.twin(exit)) == region
                            && dual.valid_exit(current, exit, axis)
                    })
                    .map(|exit| (structure.twin(exit), true))
                    .collect()
            },
            measure,
            proximity,
            blocked,
        )
    }

    // The search of the loop constructions: the cheapest loop from the gap node `start` back to it (or a path to the gap
    // node `end`, if it differs), where crossing a
    // loop (a chord of a face) advances the progress (`transition` gives the progress after crossing `chord` into the
    // gap node `target`, possibly several or none: not allowed) and the loop closes with progress `goal`. Never crosses
    // the blocked half-edges, or two loops in one move. Moves near existing loops cost more (by `PROXIMITY_WEIGHT` times
    // their `proximity`), such that loops keep some distance from each other (running alongside a loop, or crossing a
    // loop at an existing intersection, leaves narrow corridors for the layout); every move costs at least its flow
    // cost, so the heuristic remains a lower bound.
    #[allow(clippy::too_many_arguments)]
    fn search_loop<P: Copy + Eq + std::hash::Hash>(
        &self,
        axis: Direction,
        start: GapNode,
        end: GapNode,
        initial: P,
        goal: P,
        mut transition: impl FnMut(P, &FaceChord, GapNode) -> Vec<P>,
        measure: &impl Fn(f64) -> OrderedFloat<f64>,
        proximity: &FxHashMap<FaceID, f64>,
        blocked: &HashSet<EdgeID>,
    ) -> Result<(Vec<EdgeID>, f64, Vec<(usize, usize)>), Option<Vec<EdgeID>>> {
        const MAX_EXPANSIONS: usize = 1_000_000;
        let timer = Instant::now();
        let mesh = self.mesh_ref;
        let graph = self.flow_graph(axis).ok_or(None)?;
        if !graph.node_exists(start.0) {
            return Err(None);
        }

        let heuristic_scale = self.flow_graph_heuristic_scale(graph, measure);
        let heuristic = self.flow_graph_heuristic(graph, end.0, heuristic_scale);
        let mut chords_cache: FxHashMap<FaceID, Rc<Vec<FaceChord>>> = FxHashMap::default();

        let mut best: FxHashMap<(GapNode, P), f64> = FxHashMap::default();
        let mut previous: FxHashMap<(GapNode, P), (GapNode, P)> = FxHashMap::default();
        let mut heap = BinaryHeap::new();
        let goal = (end, goal);
        best.insert((start, initial), 0.);
        // Heap entries refer to the states by index (gap nodes are not ordered).
        let mut entries: Vec<(GapNode, P)> = vec![(start, initial)];
        heap.push((std::cmp::Reverse(OrderedFloat(heuristic(start.0))), 0usize));
        let mut expansions = 0usize;
        let mut found = None;

        while let Some((std::cmp::Reverse(priority), entry)) = heap.pop() {
            let (node, progress) = entries[entry];
            let cost = best[&(node, progress)];
            if *priority > cost + heuristic(node.0) + 1e-12 {
                continue;
            }
            if (node, progress) == goal {
                found = Some(cost);
                break;
            }
            expansions += 1;
            if expansions > MAX_EXPANSIONS {
                break;
            }
            let (edge, gap) = node;
            let face = mesh.face(edge);
            let chords = chords_cache
                .entry(face)
                .or_insert_with(|| Rc::new(self.face_chords(face)))
                .clone();
            let p = self.gap_coord(edge, gap);
            let crowding = 1. + PROXIMITY_WEIGHT * proximity.get(&face).copied().unwrap_or(0.);
            for (next, weight) in graph.outgoing(edge) {
                if next == mesh.twin(edge) || mesh.face(next) != face || blocked.contains(&next) {
                    continue;
                }
                let step = *measure(weight) * crowding;
                let n_next = self.crossing_count(next);
                for next_gap in 0..=n_next {
                    let q = self.gap_coord(next, next_gap);
                    let mut crossed = chords.iter().filter(|chord| chord.separates(p, q));
                    let first = crossed.next();
                    if crossed.next().is_some() {
                        continue;
                    }
                    let target = (mesh.twin(next), n_next - next_gap);
                    let next_progress = match first {
                        None => vec![progress],
                        Some(chord) => transition(progress, chord, target),
                    };
                    let new_cost = cost + step;
                    for next_progress in next_progress {
                        let state = (target, next_progress);
                        if best.get(&state).is_none_or(|&c| new_cost < c - 1e-12) {
                            best.insert(state, new_cost);
                            previous.insert(state, (node, progress));
                            heap.push((
                                std::cmp::Reverse(OrderedFloat(new_cost + heuristic(target.0))),
                                entries.len(),
                            ));
                            entries.push(state);
                        }
                    }
                }
            }
        }

        let Some(cost) = found else {
            debug!(
                "b_loops::search_loop: no path axis={axis:?} expansions={expansions} elapsed_ms={:.3}",
                elapsed_ms(timer)
            );
            return Err(None);
        };
        // Reconstruct: every transition is a move inside a face (from the entry half-edge to the exit half-edge, at a
        // gap) followed by crossing the exit half-edge.
        let mut states = vec![goal];
        let mut current = goal;
        while let Some(&prev) = previous.get(&current) {
            current = prev;
            states.push(current);
            if current == (start, initial) {
                break;
            }
        }
        states.reverse();
        let mut edges = vec![];
        let mut gaps = vec![];
        for w in states.windows(2) {
            let ((entry, _), _) = w[0];
            let ((target_edge, target_gap), _) = w[1];
            let exit = mesh.twin(target_edge);
            let exit_gap = self.crossing_count(exit) - target_gap;
            edges.push(entry);
            gaps.push((edges.len(), exit_gap));
            edges.push(exit);
        }
        // (A path is checked for revisited edges only; as part of a loop, it is checked with the rest of that loop.)
        let invalid = if end == start {
            self.check_loop(&edges).is_err()
        } else {
            edges.iter().duplicates().next().is_some()
        };
        if invalid {
            let mut seen = HashSet::new();
            let revisited = edges
                .iter()
                .filter(|&&e| !seen.insert(e))
                .copied()
                .collect_vec();
            debug!(
                "b_loops::construct_valid_loop: rejected (loop revisits {} edges) axis={axis:?}",
                revisited.len()
            );
            return Err((!revisited.is_empty()).then_some(revisited));
        }
        debug!(
            "b_loops::search_loop: ok axis={axis:?} edges={} cost={cost:.4} expansions={expansions} elapsed_ms={:.3}",
            edges.len(),
            elapsed_ms(timer)
        );
        Ok((edges, cost, gaps))
    }

    /// The cheapest valid loop of the given axis from a start edge (random if not given) over all sequences of regions
    /// (instead of a prescribed one, see `construct_valid_loop`): it follows the flow as long as it likes, e.g., along
    /// the outline of a flat shape over all its protrusions. Tries up to `tries` segments through which the loop enters
    /// (and closes) its start region, and keeps the cheapest (cost per length). Valid by construction, except that the
    /// loop may enter a region twice (check the result). The dual structure must be built from the current loops.
    #[allow(clippy::type_complexity)]
    pub fn free_valid_loop(
        &self,
        dual: &Dual,
        axis: Direction,
        start: Option<EdgeID>,
        tries: usize,
        measure: &impl Fn(f64) -> OrderedFloat<f64>,
    ) -> Option<(Vec<EdgeID>, f64, Vec<(usize, usize)>)> {
        let graph = self.flow_graph(axis)?;
        let edge = match start {
            Some(edge) => edge,
            None => self
                .mesh_ref
                .edge_ids_iter()
                .filter(|&e| graph.node_exists(e))
                .choose(&mut rng())?,
        };
        let gap = rand::random_range(0..=self.crossing_count(edge));
        let region = self.region_of_gap(dual, (edge, gap))?;
        let structure = &dual.loop_structure;
        let mut entries = structure
            .edges(region)
            .filter(|&segment| dual.segment_to_direction(segment) != axis)
            .collect_vec();
        entries.shuffle(&mut rng());
        let proximity = self.proximity.get_or_init(|| self.loop_proximity());
        let length = |edges: &[EdgeID]| {
            Self::cycled_windows(edges)
                .into_iter()
                .map(|[a, b]| (self.mesh_ref.position(b) - self.mesh_ref.position(a)).norm())
                .sum::<f64>()
                .max(1e-12)
        };
        let mut best: Option<(Vec<EdgeID>, f64, Vec<(usize, usize)>)> = None;
        for entry in entries.into_iter().take(tries.max(1)) {
            // A loop cannot cross the same mesh edge twice: if the cheapest one does, block those edges and retry.
            let mut blocked = HashSet::new();
            for _ in 0..3 {
                match self.construct_free_loop_avoiding(
                    dual,
                    axis,
                    (edge, gap),
                    entry,
                    measure,
                    proximity,
                    &blocked,
                ) {
                    Ok(candidate) => {
                        let rank = candidate.1 / length(&candidate.0);
                        if best.as_ref().is_none_or(|b| rank < b.1 / length(&b.0)) {
                            best = Some(candidate);
                        }
                        break;
                    }
                    Err(Some(revisited)) => {
                        let before = blocked.len();
                        let mesh = self.mesh_ref;
                        blocked.extend(revisited.iter().flat_map(|&e| [e, mesh.twin(e)]));
                        if blocked.len() == before {
                            break;
                        }
                    }
                    Err(None) => break,
                }
            }
        }
        best
    }

    /// Sample a valid loop of the given axis (see `construct_valid_loop`): from a start edge (random if not given),
    /// try up to `tries` random topological structures through its region, and keep the cheapest loop (cost per
    /// length). The dual structure must be built from the current loops.
    #[allow(clippy::type_complexity)]
    pub fn sample_valid_loop(
        &self,
        dual: &Dual,
        axis: Direction,
        start: Option<EdgeID>,
        tries: usize,
        measure: &impl Fn(f64) -> OrderedFloat<f64>,
    ) -> Option<(Vec<EdgeID>, f64, Vec<(usize, usize)>)> {
        let graph = self.flow_graph(axis)?;
        let edge = match start {
            Some(edge) => edge,
            None => self
                .mesh_ref
                .edge_ids_iter()
                .filter(|&e| graph.node_exists(e))
                .choose(&mut rng())?,
        };
        let gap = rand::random_range(0..=self.crossing_count(edge));
        let region = self.region_of_gap(dual, (edge, gap))?;
        let mut cycles = dual.valid_cycles(region, axis, 2 * tries.max(1));
        cycles.shuffle(&mut rng());
        let length = |edges: &[EdgeID]| {
            Self::cycled_windows(edges)
                .into_iter()
                .map(|[a, b]| (self.mesh_ref.position(b) - self.mesh_ref.position(a)).norm())
                .sum::<f64>()
                .max(1e-12)
        };
        let mut found = 0;
        let mut best: Option<(Vec<EdgeID>, f64, Vec<(usize, usize)>)> = None;
        for cycle in cycles {
            if found >= tries.max(1) {
                break;
            }
            let Some(candidate) =
                self.construct_valid_loop(dual, axis, (edge, gap), &cycle, measure)
            else {
                continue;
            };
            found += 1;
            let rank = candidate.1 / length(&candidate.0);
            if best.as_ref().is_none_or(|b| rank < b.1 / length(&b.0)) {
                best = Some(candidate);
            }
        }
        best
    }

    /// The cheapest (cost per length) valid loop of the given axis through the given gap node, over the first
    /// `limit` topological structures through its region (deterministic, e.g., for interactive use).
    #[allow(clippy::type_complexity)]
    pub fn best_valid_loop(
        &self,
        dual: &Dual,
        axis: Direction,
        start: GapNode,
        limit: usize,
        measure: &impl Fn(f64) -> OrderedFloat<f64>,
    ) -> Option<(Vec<EdgeID>, f64, Vec<(usize, usize)>)> {
        let region = self.region_of_gap(dual, start)?;
        let length = |edges: &[EdgeID]| {
            Self::cycled_windows(edges)
                .into_iter()
                .map(|[a, b]| (self.mesh_ref.position(b) - self.mesh_ref.position(a)).norm())
                .sum::<f64>()
                .max(1e-12)
        };
        dual.valid_cycles(region, axis, limit)
            .into_iter()
            .filter_map(|cycle| self.construct_valid_loop(dual, axis, start, &cycle, measure))
            .min_by_key(|(edges, cost, _)| OrderedFloat(cost / length(edges)))
    }

    fn construct_part_of_loop_cached(
        &self,
        [e1, e2]: [EdgeID; 2],
        domain: &FlowGraph,
        direction: Option<Direction>,
        measure: &impl Fn(f64) -> OrderedFloat<f64>,
        heuristic_scale: HeuristicScale,
        path_cache: &mut HashMap<(EdgeID, EdgeID), Option<(Vec<EdgeID>, f64)>>,
    ) -> Option<(Vec<EdgeID>, f64)> {
        let timer = Instant::now();
        if let Some(cached) = path_cache.get(&(e1, e2)) {
            return cached.clone();
        }

        let result = if !domain.node_exists(e1) || !domain.node_exists(e2) {
            None
        } else if e1 == e2 {
            Some((vec![e1], 0.))
        } else {
            let gap_graph = GapGraph::new(self, domain, direction, measure);
            let starts = (0..gap_graph.gaps(e1))
                .map(|gap| ((e1, gap), 0., 0))
                .collect_vec();
            gap_graph
                .search(&starts, e2, None, heuristic_scale)
                .map(|(path, cost, _)| (path.into_iter().map(|(edge, _)| edge).collect(), cost))
        };

        path_cache.insert((e1, e2), result.clone());
        debug!(
            "b_loops::construct_part_of_loop: from={e1:?} to={e2:?} found={} path_edges={:?} cost={:?} elapsed_ms={:.3}",
            result.is_some(),
            result.as_ref().map(|(path, _)| path.len()),
            result.as_ref().map(|(_, cost)| *cost),
            elapsed_ms(timer)
        );
        result
    }

    // Scales of the lower bounds used by the A* searches in `graph` with the given measure (see
    // `flow_graph_heuristic`): the smallest ratio of measured cost to geometric length (between the midpoints) of the
    // paid (same-face) moves, and the smallest ratio of measured cost to (raw) weight of all moves.
    fn flow_graph_heuristic_scale(
        &self,
        graph: &FlowGraph,
        measure: &impl Fn(f64) -> OrderedFloat<f64>,
    ) -> HeuristicScale {
        let mesh = self.mesh_ref;
        let mut euclidean = f64::INFINITY;
        let mut landmark = f64::INFINITY;
        for &(from, to, weight) in graph.edges_ref() {
            let cost = *measure(weight);
            if !(cost >= 0.) {
                // Negative (or NaN) costs: no lower bound is valid.
                return HeuristicScale::NONE;
            }
            if weight > 0. {
                landmark = landmark.min(cost / weight);
            }
            if mesh.twin(from) == to {
                continue;
            }
            let geometric_length = (mesh.position(to) - mesh.position(from)).norm();
            if geometric_length > 1e-12 {
                euclidean = euclidean.min(cost / geometric_length);
            }
        }
        HeuristicScale {
            euclidean: if euclidean.is_finite() { euclidean } else { 0. },
            landmark: if landmark.is_finite() { landmark } else { 0. },
        }
    }

    /// Admissible (and consistent) lower bound for flow-graph A* searches towards `goal`, as a closure.
    ///
    /// The maximum of two bounds:
    /// - Euclidean: twin hops are free in the geometry (a half-edge and its twin share their midpoint, the position of
    ///   a half-edge), and every paid move has measured cost at least `scale.euclidean` times the distance between
    ///   the midpoints, so by the triangle inequality `scale.euclidean` times the distance between the midpoints of
    ///   the node and the goal never overestimates the remaining cost.
    /// - Landmarks: every move costs at least `scale.landmark` times its raw weight, so `scale.landmark` times a lower
    ///   bound on the raw shortest-path distance in the flow graph is a lower bound (see `grapff::fixed::Landmarks`).
    ///   The searches run on lifts of the flow graph (gap graphs) that only restrict moves, so this remains valid.
    ///   Unlike the Euclidean bound, it captures detours: e.g., the start and goal of a closed-loop search are
    ///   adjacent, while the loop goes around the surface.
    fn flow_graph_heuristic<'g>(
        &'g self,
        graph: &'g FlowGraph,
        goal: EdgeID,
        scale: HeuristicScale,
    ) -> impl Fn(EdgeID) -> f64 + 'g {
        let mesh = self.mesh_ref;
        let midpoint = move |edge: EdgeID| {
            (mesh.position(mesh.root(edge)) + mesh.position(mesh.toor(edge))) * 0.5
        };
        let goal_midpoint = midpoint(goal);
        let landmarks = (scale.landmark > 0.)
            .then(|| {
                graph
                    .node_index(&goal)
                    .map(|goal| (graph.landmarks(), goal))
            })
            .flatten();
        move |node: EdgeID| {
            let mut bound = 0.;
            if scale.euclidean > 0. {
                bound = scale.euclidean * (midpoint(node) - goal_midpoint).norm();
            }
            if let Some((landmarks, goal)) = &landmarks
                && let Some(node) = graph.node_index(&node)
            {
                bound = f64::max(bound, scale.landmark * landmarks.lower_bound(node, *goal));
            }
            bound
        }
    }

    // Shortest closed loop that uses the move between `e1` and `e2` (in the cheaper orientation).
    fn construct_loop_with_heuristic_scale(
        &self,
        [e1, e2]: [EdgeID; 2],
        domain: &FlowGraph,
        direction: Option<Direction>,
        measure: &impl Fn(f64) -> OrderedFloat<f64>,
        heuristic_scale: HeuristicScale,
    ) -> Option<(Vec<EdgeID>, f64)> {
        let timer = Instant::now();
        if !domain.node_exists(e1) || !domain.node_exists(e2) {
            return None;
        }

        // Get the better direction.
        let [n1, n2] = self.orient_anchor_pair([e1, e2], domain, measure)?;

        // A loop must return to the gap it started from.
        let gap_graph = GapGraph::new(self, domain, direction, measure);
        let mut best: Option<(Vec<GapNode>, f64, usize)> = None;
        for gap in 0..gap_graph.gaps(n1) {
            let starts = gap_graph
                .successors((n1, gap))
                .into_iter()
                .filter(|&((edge, _), _, _)| edge == n2)
                .collect_vec();
            if starts.is_empty() {
                continue;
            }
            if let Some(candidate) = gap_graph.search(&starts, n1, Some(gap), heuristic_scale)
                && best.as_ref().is_none_or(|(_, cost, crossings)| {
                    (OrderedFloat(candidate.1), candidate.2) < (OrderedFloat(*cost), *crossings)
                })
            {
                best = Some(candidate);
            }
        }

        let Some((solution, cost, crossings)) = best else {
            debug!(
                "b_loops::construct_loop: rejected reason=no_return_path e1={e1:?} e2={e2:?} elapsed_ms={:.3}",
                elapsed_ms(timer)
            );
            return None;
        };
        let solution = solution.into_iter().map(|(edge, _)| edge).collect_vec();
        let short = self.remove_redundant_same_face_edges(&solution);
        if self.check_loop(&short).is_err() {
            debug!(
                "b_loops::construct_loop: rejected reason=invalid_loop e1={e1:?} e2={e2:?} edges={} elapsed_ms={:.3}",
                short.len(),
                elapsed_ms(timer)
            );
            return None;
        }

        debug!(
            "b_loops::construct_loop: ok e1={e1:?} e2={e2:?} edges={} cost={cost:.6} crossings={crossings} elapsed_ms={:.3}",
            short.len(),
            elapsed_ms(timer)
        );
        Some((short, cost))
    }

    fn remove_redundant_same_face_edges(&self, edges: &[EdgeID]) -> Vec<EdgeID> {
        if edges.len() < 3 {
            return edges.to_vec();
        }

        let mut short = Vec::with_capacity(edges.len());
        for i in 0..edges.len() {
            let prev = edges[(i + edges.len() - 1) % edges.len()];
            let current = edges[i];
            let next = edges[(i + 1) % edges.len()];
            if self.mesh_ref.face(current) == self.mesh_ref.face(prev)
                && self.mesh_ref.face(current) == self.mesh_ref.face(next)
            {
                continue;
            }
            short.push(current);
        }
        short
    }

    fn orient_anchor_pair(
        &self,
        [e1, e2]: [EdgeID; 2],
        graph: &FlowGraph,
        measure: &impl Fn(f64) -> OrderedFloat<f64>,
    ) -> Option<[EdgeID; 2]> {
        let directed_weight = |from, to| -> Option<f64> { graph.get_directed_weight(from, to) };
        let forward = directed_weight(e1, e2).map(measure);
        let backward = directed_weight(e2, e1).map(measure);

        match (forward, backward) {
            (Some(forward), Some(backward)) => {
                if forward <= backward {
                    Some([e1, e2])
                } else {
                    Some([e2, e1])
                }
            }
            (Some(_), None) => Some([e1, e2]),
            (None, Some(_)) => Some([e2, e1]),
            (None, None) => None,
        }
    }

    fn flow_graph(&self, direction: Direction) -> Option<&'a FlowGraph> {
        self.flow_graphs
            .map(|flow_graphs| &flow_graphs[direction as usize])
    }

    // Best loop through a single anchor node.
    fn construct_loop_through_single_anchor(
        &self,
        anchor: EdgeID,
        graph: &FlowGraph,
        direction: Direction,
        measure: &impl Fn(f64) -> OrderedFloat<f64>,
    ) -> Option<(Vec<EdgeID>, f64)> {
        let heuristic_scale = self.flow_graph_heuristic_scale(graph, measure);
        graph
            .neighbors(anchor)
            .into_iter()
            .filter_map(|next| {
                self.construct_loop_with_heuristic_scale(
                    [anchor, next],
                    graph,
                    Some(direction),
                    measure,
                    heuristic_scale,
                )
            })
            .min_by_key(|(_, cost)| OrderedFloat(*cost))
    }

    pub fn construct_loop_with_anchors_and_locked_segments(
        &self,
        anchors: &[[EdgeID; 2]],
        direction: Direction,
        locked_segments: &[(Vec<EdgeID>, f64)],
        measure: impl Fn(f64) -> OrderedFloat<f64>,
    ) -> Option<(Vec<EdgeID>, f64, Vec<(Vec<EdgeID>, f64)>)> {
        let graph = self.flow_graph(direction)?;
        let anchor_nodes = anchors
            .iter()
            .filter_map(|&[e1, e2]| self.orient_anchor_pair([e1, e2], graph, &measure))
            .flatten()
            .collect_vec();

        self.construct_loop_through_anchor_nodes_with_locked_segments_in_graph(
            &anchor_nodes,
            graph,
            direction,
            locked_segments,
            &measure,
        )
    }

    fn construct_loop_through_anchor_nodes_with_locked_segments_in_graph(
        &self,
        anchor_nodes: &[EdgeID],
        graph: &FlowGraph,
        direction: Direction,
        locked_segments: &[(Vec<EdgeID>, f64)],
        measure: &impl Fn(f64) -> OrderedFloat<f64>,
    ) -> Option<(Vec<EdgeID>, f64, Vec<(Vec<EdgeID>, f64)>)> {
        let timer = Instant::now();
        if anchor_nodes.is_empty()
            || anchor_nodes
                .iter()
                .any(|anchor| !graph.node_exists(*anchor))
        {
            return None;
        }

        if anchor_nodes.len() == 1 {
            let (edges, cost) = self.construct_loop_through_single_anchor(
                anchor_nodes[0],
                graph,
                direction,
                measure,
            )?;
            return Some((edges.clone(), cost, vec![(edges, cost)]));
        }

        let mut path_cache = HashMap::new();
        let heuristic_scale = self.flow_graph_heuristic_scale(graph, measure);
        let mut segments = Vec::with_capacity(anchor_nodes.len());
        let mut total_cost = 0.0;

        for i in 0..anchor_nodes.len() {
            let start = anchor_nodes[i];
            let end = anchor_nodes[(i + 1) % anchor_nodes.len()];

            let (part, cost) = if i < locked_segments.len() {
                let (part, cost) = locked_segments[i].clone();
                if part.first().copied() != Some(start) || part.last().copied() != Some(end) {
                    debug!(
                        "b_loops::construct_loop_with_locked_segments: rejected reason=locked_segment_mismatch segment={i}"
                    );
                    return None;
                }
                (part, cost)
            } else {
                self.construct_part_of_loop_cached(
                    [start, end],
                    graph,
                    Some(direction),
                    measure,
                    heuristic_scale,
                    &mut path_cache,
                )?
            };

            if part.len() < 2 {
                return None;
            }
            total_cost += cost;
            segments.push((part, cost));
        }

        let mut loop_edges = Vec::new();
        for (part, _) in &segments {
            let mut part = part.clone();
            part.pop();
            loop_edges.extend(part);
        }

        let loop_edges = self.remove_redundant_same_face_edges(&loop_edges);
        if self.check_loop(&loop_edges).is_err() {
            debug!(
                "b_loops::construct_loop_with_locked_segments: rejected reason=invalid_loop anchor_nodes={} loop_edges={} elapsed_ms={:.3}",
                anchor_nodes.len(),
                loop_edges.len(),
                elapsed_ms(timer)
            );
            return None;
        }

        debug!(
            "b_loops::construct_loop_with_locked_segments: ok anchor_nodes={} locked_segments={} loop_edges={} cost={:.6} elapsed_ms={:.3}",
            anchor_nodes.len(),
            locked_segments.len(),
            loop_edges.len(),
            total_cost,
            elapsed_ms(timer)
        );
        Some((loop_edges, total_cost, segments))
    }

    fn construct_unbounded_loop_scaled(
        &self,
        edges: [EdgeID; 2],
        direction: Direction,
        flow_graph: &FlowGraph,
        measure: impl Fn(f64) -> OrderedFloat<f64>,
        heuristic_scale: HeuristicScale,
    ) -> Option<(Vec<EdgeID>, f64)> {
        self.construct_loop_with_heuristic_scale(
            edges,
            flow_graph,
            Some(direction),
            &measure,
            heuristic_scale,
        )
    }

    pub fn sample_loops(
        &self,
        n: usize,
        axis: Direction,
        measure: impl Fn(f64) -> OrderedFloat<f64> + Sync + Send,
        score: impl Fn((&[EdgeID], f64)) -> f64 + Sync + Send,
    ) -> Vec<Vec<EdgeID>> {
        let timer = Instant::now();
        if n == 0 {
            return Vec::new();
        }

        let Some(flow_graph) = self.flow_graph(axis) else {
            debug!(
                "b_loops::sample_loops: rejected reason=no_flow_graphs axis={axis:?} requested={n}"
            );
            return vec![vec![]; n];
        };

        // Number of bands along the axis (see below), each contributing up to two anchors.
        let selected_target = (3 * n).min(n * n).max(1);
        let candidate_pool = 50 * selected_target;

        // Random anchors (moves inside a face), ranked by alignment: cost per length (not cost, which would favor
        // small faces). Moves against the flow may not exist in the flow graph.
        let edges = self.mesh_ref.edge_ids();
        let axis_vector = Vector3D::from(axis);
        let ranked = (0..candidate_pool)
            .filter_map(|_| {
                let e1 = *edges.iter().choose(&mut rng())?;
                let e2 = self.mesh_ref.next(e1);
                let weight = [(e1, e2), (e2, e1)]
                    .into_iter()
                    .filter_map(|(a, b)| flow_graph.get_directed_weight(a, b))
                    .map(|w| *measure(w))
                    .reduce(f64::min)?;
                let length = (self.mesh_ref.position(e2) - self.mesh_ref.position(e1)).norm();
                Some(([e1, e2], weight / length.max(1e-12)))
            })
            .collect_vec();

        // Spread the anchors along the axis (loops of an axis are roughly level sets along it): one anchor per
        // band of the axis, the best aligned one in its band.
        let coordinate = |edge: EdgeID| self.mesh_ref.position(edge).dot(&axis_vector);
        let (lo, hi) = edges
            .iter()
            .map(|&e| coordinate(e))
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), c| {
                (lo.min(c), hi.max(c))
            });
        let band = |edge: EdgeID| {
            (((coordinate(edge) - lo) / (hi - lo).max(1e-12) * selected_target as f64) as usize)
                .min(selected_target - 1)
        };
        // In every band: the best aligned anchor, and a random anchor among the better aligned half (for variety:
        // the best aligned anchor may lie on a feature that only permits a local loop).
        let mut per_band: HashMap<usize, Vec<([EdgeID; 2], f64)>> = HashMap::new();
        for &(anchor, rank) in &ranked {
            per_band
                .entry(band(anchor[0]))
                .or_default()
                .push((anchor, rank));
        }
        let mut selected = vec![];
        for mut anchors in per_band.into_values() {
            anchors.sort_by_key(|&(_, rank)| OrderedFloat(rank));
            selected.push(anchors[0].0);
            if anchors.len() > 2 {
                let half = anchors.len().div_ceil(2);
                if let Some(&(anchor, _)) = anchors[1..half].iter().choose(&mut rng()) {
                    selected.push(anchor);
                }
            }
        }
        let selected_target = selected_target.max(selected.len());
        // Fill up with the best remaining anchors (if some bands had no anchors).
        for (anchor, _) in ranked
            .into_iter()
            .sorted_by_key(|&(_, rank)| OrderedFloat(rank))
        {
            if selected.len() >= selected_target {
                break;
            }
            if !selected.contains(&anchor) {
                selected.push(anchor);
            }
        }

        let heuristic_scale = self.flow_graph_heuristic_scale(flow_graph, &measure);
        let constructed = selected
            .into_iter()
            .iter_into_par()
            .filter_map(|es| {
                self.construct_unbounded_loop_scaled(
                    es,
                    axis,
                    flow_graph,
                    &measure,
                    heuristic_scale,
                )
            })
            .collect::<Vec<_>>();
        let constructed_count = constructed.len();

        // Different anchors can lead to the same loop.
        let mut seen = HashSet::new();
        let constructed = constructed
            .into_iter()
            .filter(|(path, _)| {
                let mut key = path.clone();
                key.sort_by_key(|e| e.raw());
                seen.insert(key)
            })
            .collect_vec();

        // Loops that cut off only a tiny part of the surface (e.g., loops around a singularity of the flow
        // field, where the flow circulates) are useless.
        let total_area: f64 = self
            .mesh_ref
            .face_ids()
            .into_iter()
            .map(|f| self.mesh_ref.size(f))
            .sum();
        let constructed = constructed
            .into_iter()
            .iter_into_par()
            .filter(|(path, _)| !self.cuts_off_small_part(path, MIN_SEPARATED_AREA * total_area))
            .collect::<Vec<_>>();

        let result = constructed
            .into_iter()
            .sorted_by_key(|(path, s)| OrderedFloat(score((path, *s))))
            .take(n)
            .map(|(x, _)| x)
            .collect::<Vec<_>>();

        debug!(
            "b_loops::sample_loops: axis={axis:?} requested={n} constructed={constructed_count} returned={} elapsed_ms={:.3}",
            result.len(),
            elapsed_ms(timer)
        );

        result
    }
}

// The anisotropic metric of a face with the given (unit) normal, for loops around `axis`: a quadratic form `P` such that
// the length of an in-face vector `v` is `sqrt(v^T P v)`. Moving across the target direction (in the face, orthogonal
// to the plane of the loop) costs `ANISOTROPY` times more (squared) than moving along it, weighted by how well the face
// determines that direction.
fn anisotropic_metric(normal: Vector3D, axis: Vector3D) -> Metric {
    let tangent_axis = axis - normal * axis.dot(&normal);
    let confidence = tangent_axis.norm().min(1.);
    if !(confidence >= 1e-9) || !normal.iter().all(|c| c.is_finite()) {
        return Metric::ISOTROPIC;
    }
    let along = normal.cross(&tangent_axis).normalize();
    let across = normal.cross(&along);
    let weight = 1. + (ANISOTROPY - 1.) * confidence * confidence;
    Metric {
        along,
        across,
        weight,
        isotropic: false,
    }
}

// A quadratic form `P` on (in-face) vectors: `P v = along (along . v) + weight across (across . v)`, or the identity.
#[derive(Clone, Copy, Debug)]
struct Metric {
    along: Vector3D,
    across: Vector3D,
    weight: f64,
    isotropic: bool,
}

impl Metric {
    const ISOTROPIC: Self = Self {
        along: Vector3D::new(0., 0., 0.),
        across: Vector3D::new(0., 0., 0.),
        weight: 1.,
        isotropic: true,
    };

    fn apply(&self, v: Vector3D) -> Vector3D {
        if self.isotropic {
            v
        } else {
            self.along * self.along.dot(&v) + self.across * (self.weight * self.across.dot(&v))
        }
    }
}

// Minimize `sum_k sqrt(d_k^T P_k d_k)` over `t` (inside `intervals`), where `d_k = x_{k+1} - x_k` (cyclically) and
// `x_k = origins[k] + directions[k] * t[k]`. The objective is convex; this is a projected Newton method (Bertsekas,
// 1982) with an Armijo line search along the projection arc. The (slightly regularized) Hessian is cyclic tridiagonal
// and solved in O(m).
fn minimize_taut(
    origins: &[Vector3D],
    directions: &[Vector3D],
    metrics: &[Metric],
    intervals: &[(f64, f64)],
    t: &mut [f64],
) {
    let m = t.len();
    if m < 3 {
        return;
    }
    // Smoothing of the norm at zero-length segments (two crossings at the same point), relative to the edge lengths.
    let scale = directions.iter().map(|d| d.norm()).sum::<f64>() / m as f64;
    if !(scale > 0.) || !scale.is_finite() {
        return;
    }
    let eps2 = (1e-9 * scale).powi(2);
    let point = |k: usize, t: &[f64]| origins[k] + directions[k] * t[k];
    let objective = |t: &[f64]| {
        (0..m)
            .map(|k| {
                let d = point((k + 1) % m, t) - point(k, t);
                (d.dot(&metrics[k].apply(d)) + eps2).sqrt()
            })
            .sum::<f64>()
    };
    let clamp = |k: usize, x: f64| x.clamp(intervals[k].0, intervals[k].1);
    for (k, tk) in t.iter_mut().enumerate() {
        *tk = clamp(k, *tk);
    }

    let mut value = objective(t);
    let mut gradient = vec![0.; m];
    let mut diagonal = vec![0.; m];
    // `off[k]` couples `k` and `k + 1` (cyclically).
    let mut off = vec![0.; m];
    let mut step = vec![0.; m];
    let mut candidate = vec![0.; m];
    let mut system_diagonal = vec![0.; m];
    let mut system_off = vec![0.; m];
    let mut active = vec![false; m];
    for _ in 0..200 {
        gradient.iter_mut().for_each(|g| *g = 0.);
        diagonal.iter_mut().for_each(|h| *h = 0.);
        for k in 0..m {
            let next = (k + 1) % m;
            let d = point(next, t) - point(k, t);
            let pd = metrics[k].apply(d);
            let f = (d.dot(&pd) + eps2).sqrt();
            let q = pd / f;
            // Hessian of the segment length with respect to `d`: H = (P - q q^T) / f (positive semidefinite).
            let (a, b) = (directions[k], directions[next]);
            let (pa, pb) = (metrics[k].apply(a), metrics[k].apply(b));
            let (qa, qb) = (q.dot(&a), q.dot(&b));
            gradient[k] -= qa;
            gradient[next] += qb;
            diagonal[k] += (a.dot(&pa) - qa * qa) / f;
            diagonal[next] += (b.dot(&pb) - qb * qb) / f;
            off[k] = -(a.dot(&pb) - qa * qb) / f;
        }

        // Variables at a bound with the gradient pointing outwards are kept fixed (the active set).
        let width = (0..m)
            .map(|k| (t[k] - clamp(k, t[k] - gradient[k])).abs())
            .fold(0., f64::max);
        if width < 1e-14 {
            break;
        }
        let tolerance = width.min(1e-9);
        for k in 0..m {
            active[k] = (t[k] <= intervals[k].0 + tolerance && gradient[k] > 0.)
                || (t[k] >= intervals[k].1 - tolerance && gradient[k] < 0.);
        }
        let regularization = 1e-10 * diagonal.iter().copied().fold(0., f64::max) + 1e-300;
        system_diagonal.copy_from_slice(&diagonal);
        system_off.copy_from_slice(&off);
        for k in 0..m {
            if active[k] {
                system_diagonal[k] = 1.;
                system_off[k] = 0.;
                system_off[(k + m - 1) % m] = 0.;
                step[k] = 0.;
            } else {
                system_diagonal[k] += regularization;
                step[k] = -gradient[k];
            }
        }
        if !solve_cyclic_tridiagonal(&system_diagonal, &system_off, &mut step) {
            break;
        }
        for k in 0..m {
            if active[k] {
                step[k] = -gradient[k] / diagonal[k].max(regularization);
            }
        }

        // Armijo backtracking along the projection arc.
        let mut alpha = 1.;
        let mut accepted = None;
        for _ in 0..40 {
            for k in 0..m {
                candidate[k] = clamp(k, t[k] + alpha * step[k]);
            }
            let decrease = (0..m)
                .map(|k| gradient[k] * (candidate[k] - t[k]))
                .sum::<f64>();
            let candidate_value = objective(&candidate);
            if candidate_value <= value + 1e-4 * decrease {
                accepted = Some(candidate_value);
                break;
            }
            alpha *= 0.5;
        }
        let Some(new_value) = accepted else {
            break;
        };
        let moved = (0..m)
            .map(|k| (candidate[k] - t[k]).abs())
            .fold(0., f64::max);
        let improvement = value - new_value;
        t.copy_from_slice(&candidate);
        value = new_value;
        if moved < 1e-12 || improvement <= 1e-15 * value {
            break;
        }
    }
}

// Solve `A x = b` in place for a symmetric cyclic tridiagonal matrix `A` with diagonal `diagonal` and `off[k]` at
// `(k, k + 1)` and `(k + 1, k)` (indices modulo `n`), via the Thomas algorithm and the Sherman-Morrison formula.
// Returns false if the system is (numerically) singular.
fn solve_cyclic_tridiagonal(diagonal: &[f64], off: &[f64], b: &mut [f64]) -> bool {
    let n = diagonal.len();
    if n < 3 {
        return false;
    }
    let corner = off[n - 1];
    // A = T + u v^T with u = (gamma, 0, ..., 0, corner) and v = (1, 0, ..., 0, corner / gamma).
    let gamma = -diagonal[0];
    if gamma == 0. {
        return false;
    }
    let mut modified = diagonal.to_vec();
    modified[0] -= gamma;
    modified[n - 1] -= corner * corner / gamma;
    let mut c = vec![0.; n];
    // Thomas algorithm on the tridiagonal T (super- and subdiagonal `off[0..n - 1]`).
    let mut solve = |rhs: &mut [f64]| -> bool {
        let mut denominator = modified[0];
        if denominator.abs() < 1e-300 {
            return false;
        }
        c[0] = off[0] / denominator;
        rhs[0] /= denominator;
        for i in 1..n {
            denominator = modified[i] - off[i - 1] * c[i - 1];
            if denominator.abs() < 1e-300 || !denominator.is_finite() {
                return false;
            }
            if i < n - 1 {
                c[i] = off[i] / denominator;
            }
            rhs[i] = (rhs[i] - off[i - 1] * rhs[i - 1]) / denominator;
        }
        for i in (0..n - 1).rev() {
            rhs[i] -= c[i] * rhs[i + 1];
        }
        true
    };
    if !solve(b) {
        return false;
    }
    let mut z = vec![0.; n];
    z[0] = gamma;
    z[n - 1] = corner;
    if !solve(&mut z) {
        return false;
    }
    let v_dot_y = b[0] + b[n - 1] * corner / gamma;
    let v_dot_z = z[0] + z[n - 1] * corner / gamma;
    let denominator = 1. + v_dot_z;
    if denominator.abs() < 1e-300 {
        return false;
    }
    let factor = v_dot_y / denominator;
    for i in 0..n {
        b[i] -= factor * z[i];
    }
    b.iter().all(|x| x.is_finite())
}

#[cfg(test)]
mod taut_tests {
    use super::*;

    // Coordinate-wise golden-section relaxation (the previous method), run to convergence, as a reference.
    fn relax(
        origins: &[Vector3D],
        directions: &[Vector3D],
        metrics: &[Metric],
        intervals: &[(f64, f64)],
        t: &mut [f64],
    ) {
        let m = t.len();
        let point = |k: usize, s: f64| origins[k] + directions[k] * s;
        let norm = |k: usize, d: Vector3D| d.dot(&metrics[k].apply(d)).sqrt();
        for _ in 0..2_000 {
            let mut change: f64 = 0.;
            for k in 0..m {
                let (prev, next) = ((k + m - 1) % m, (k + 1) % m);
                let (p, q) = (point(prev, t[prev]), point(next, t[next]));
                let cost = |s: f64| norm(prev, point(k, s) - p) + norm(k, q - point(k, s));
                let (mut lo, mut hi) = intervals[k];
                for _ in 0..45 {
                    let (a, b) = (lo + (hi - lo) / 3., hi - (hi - lo) / 3.);
                    if cost(a) <= cost(b) { hi = b } else { lo = a }
                }
                let new = (lo + hi) / 2.;
                change = change.max((new - t[k]).abs());
                t[k] = new;
            }
            if change < 1e-12 {
                break;
            }
        }
    }

    fn length(origins: &[Vector3D], directions: &[Vector3D], metrics: &[Metric], t: &[f64]) -> f64 {
        let m = t.len();
        (0..m)
            .map(|k| {
                let n = (k + 1) % m;
                let d = (origins[n] + directions[n] * t[n]) - (origins[k] + directions[k] * t[k]);
                d.dot(&metrics[k].apply(d)).sqrt()
            })
            .sum()
    }

    #[test]
    fn cyclic_tridiagonal_solver_matches_dense() {
        let n = 7;
        let diagonal = (0..n).map(|i| 4. + i as f64 * 0.3).collect_vec();
        let off = (0..n).map(|i| -1. + 0.1 * i as f64).collect_vec();
        let rhs = (0..n).map(|i| (i as f64).sin()).collect_vec();
        let mut x = rhs.clone();
        assert!(solve_cyclic_tridiagonal(&diagonal, &off, &mut x));
        for i in 0..n {
            let ax = diagonal[i] * x[i]
                + off[i] * x[(i + 1) % n]
                + off[(i + n - 1) % n] * x[(i + n - 1) % n];
            assert!((ax - rhs[i]).abs() < 1e-12, "row {i}: {ax} vs {}", rhs[i]);
        }
    }

    #[test]
    fn taut_matches_relaxation() {
        // Crossings on the edges of a ring of quads around a (perturbed) cylinder, with some tight intervals.
        let m = 16;
        let mut origins = vec![];
        let mut directions = vec![];
        let mut intervals = vec![];
        for k in 0..m {
            let a = std::f64::consts::TAU * k as f64 / m as f64;
            let r = 1. + 0.3 * (3. * a).sin();
            origins.push(Vector3D::new(
                r * a.cos(),
                r * a.sin(),
                -1. + 0.2 * (5. * a).cos(),
            ));
            directions.push(Vector3D::new(0.1 * a.sin(), -0.1 * a.cos(), 2.));
            intervals.push(if k % 5 == 0 { (0.7, 0.9) } else { (0.01, 0.99) });
        }
        for metrics in [
            vec![Metric::ISOTROPIC; m],
            (0..m)
                .map(|k| {
                    let a = std::f64::consts::TAU * (k as f64 + 0.5) / m as f64;
                    anisotropic_metric(
                        Vector3D::new(a.cos(), a.sin(), 0.),
                        Vector3D::new(0., 0., 1.),
                    )
                })
                .collect_vec(),
        ] {
            let start = intervals
                .iter()
                .map(|&(lo, hi)| (lo + hi) / 2.)
                .collect_vec();
            let mut newton = start.clone();
            minimize_taut(&origins, &directions, &metrics, &intervals, &mut newton);
            let mut reference = start.clone();
            relax(&origins, &directions, &metrics, &intervals, &mut reference);
            let (a, b) = (
                length(&origins, &directions, &metrics, &newton),
                length(&origins, &directions, &metrics, &reference),
            );
            assert!(a <= b * (1. + 1e-9), "newton {a} > relaxation {b}");
            for k in 0..m {
                assert!(newton[k] >= intervals[k].0 && newton[k] <= intervals[k].1);
            }
        }
    }
}
