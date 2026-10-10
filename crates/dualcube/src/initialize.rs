//! Initialization of the loop structure (for genus-0 surfaces): three loops, one per axis, forming a cube.
//!
//! For every order of the axes, a few central first loops are sampled. A second loop is constructed that crosses
//! the first exactly twice, and a third loop that crosses the first two alternately (exactly twice each), both with a
//! search that only allows these crossings (see `Solution::construct_crossing_loop`). Three loops that cross pairwise
//! twice (alternately) always form a valid loop structure (a cube) on a genus-0 surface, so this succeeds whenever the
//! searches find paths; these always exist in the unpruned flow graphs, which are used as a fallback. The candidate
//! with the best quality is kept.
//!
//! (Earlier methods, anchoring the second and third loop to points instead of constraining their crossings, all
//! combinations of independently sampled loops, and the paper's random candidates, were removed: they can fail, and
//! were slower.)

use crate::prelude::*;

// Number of first loops sampled per order of the axes.
const SAMPLES: usize = 2;

impl Solution {
    /// Initialize the loop structure (see the module documentation).
    pub fn initialize(&mut self) {
        // Loops are sampled from the flow graphs, so make sure they are ready.
        self.prepare_flow();
        if self.try_initialize_crossing(SAMPLES) {
            return;
        }
        // The pruned flow graphs may lack the paths; the unpruned ones always have them.
        warn!("initialize: no candidate; retrying with the unpruned flow graphs");
        let pruned = self.flow_graphs.clone();
        if let Some(fields) = &self.fields {
            self.flow_graphs = Some(FlowPhase::build_graphs(
                &self.mesh_ref,
                fields,
                GraphParams {
                    max_deviation: std::f64::consts::PI,
                    ..GraphParams::default()
                },
            ));
        }
        let found = self.try_initialize_crossing(2 * SAMPLES);
        self.flow_graphs = pruned;
        if found {
            // Rebuild the derived structures with the regular flow graphs.
            let fast = self.clone();
            if self.reconstruct_solution(false).is_err() {
                *self = fast;
            }
        } else {
            warn!("initialize: no candidate with the unpruned flow graphs either");
        }
    }
    /// Several new initial loop structures (see `initialize`; random, so every call gives others): the best `count` of
    /// the candidates by their quality with a fast layout, best first. The loops of this solution are not used.
    #[must_use]
    pub fn initial_candidates(&self, count: usize) -> Vec<Self> {
        let mut base = self.clone();
        base.loops.clear();
        base.occupied.clear();
        base.clear();
        base.prepare_flow();
        let candidates = base.crossing_candidates(SAMPLES);
        Self::ranked(candidates).into_iter().take(count).collect()
    }

    fn try_initialize_crossing(&mut self, samples: usize) -> bool {
        let candidates = self.crossing_candidates(samples);
        self.keep_best_candidate(candidates)
    }

    // Candidates for the initial loop structure: for every order of the axes and a few sampled first loops, see the
    // module documentation.
    fn crossing_candidates(&self, samples: usize) -> Vec<Self> {
        let orders = [
            [Direction::X, Direction::Y, Direction::Z],
            [Direction::Y, Direction::Z, Direction::X],
            [Direction::Z, Direction::X, Direction::Y],
        ];
        let s = |(p, _): (&[EdgeID], f64)| -self.separated_area(p);
        let firsts = orders
            .iter()
            .flat_map(|&order| {
                self.sample_loops(samples, order[0], OrderedFloat, s)
                    .into_iter()
                    .map(move |first| (order, first))
            })
            .collect_vec();
        let candidates = firsts
            .into_par()
            .flat_map(|(order, first)| {
                (0..2)
                    .filter_map(|rotation| self.complete_crossing(&first, order, rotation))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        info!("initialize: crossing candidates={}", candidates.len());
        candidates
    }
    // Given a first loop, add a second loop that crosses it exactly twice (starting near the first loop at its
    // `rotation` quarter), and a third loop that crosses the first two alternately (starting near the first loop,
    // a quarter further).
    fn complete_crossing(
        &self,
        first: &[EdgeID],
        [first_axis, second_axis, third_axis]: [Direction; 3],
        rotation: usize,
    ) -> Option<Self> {
        let mut solution = self.clone();
        let l1 = solution.add_loop(Loop::new(first.to_vec(), first_axis));
        let exits = solution.loop_exits(first);
        let m = exits.len();
        if m < 4 {
            return None;
        }
        let near = |solution: &Self, quarter: usize, axis: Direction| {
            let exit = first[exits[(quarter * m / 4) % m]];
            solution
                .anchor_near(solution.mesh_ref.toor(exit), axis, first)
                .map(|[e1, _]| e1)
        };

        let start = near(&solution, rotation, second_axis)?;
        let (edges, _, gaps) = solution.construct_crossing_loop(second_axis, start, &[l1, l1])?;
        let l2 = solution.add_loop_in_gaps(Loop::new(edges, second_axis), gaps);

        let start = near(&solution, rotation + 1, third_axis)?;
        let (edges, _, gaps) =
            solution.construct_crossing_loop(third_axis, start, &[l1, l2, l1, l2])?;
        solution.add_loop_in_gaps(Loop::new(edges, third_axis), gaps);
        solution.dual_is_ok().then_some(solution)
    }
    // The given (valid) candidates with a fast layout, best first (those that fail are left out).
    fn ranked(candidates: Vec<Self>) -> Vec<Self> {
        let mut embedded = candidates
            .into_par()
            .filter_map(|mut solution| {
                (solution
                    .reconstruct_solution_with(false, LayoutParams::fast())
                    .is_ok()
                    && solution.get_quality().is_some())
                .then_some(solution)
            })
            .collect::<Vec<_>>();
        embedded.sort_by_key(|solution| {
            std::cmp::Reverse(OrderedFloat(solution.get_quality().unwrap_or(f64::MIN)))
        });
        embedded
    }

    // Keep the best of the given (valid) candidates, scored with a fast layout, and embed it properly.
    fn keep_best_candidate(&mut self, candidates: Vec<Self>) -> bool {
        let Some(best) = Self::ranked(candidates).into_iter().next() else {
            return false;
        };
        *self = best;
        let fast = self.clone();
        if self.reconstruct_solution(false).is_err() {
            *self = fast;
        }
        true
    }
    // Indices of the half-edges through which the loop exits a face (one per crossed edge).
    fn loop_exits(&self, edges: &[EdgeID]) -> Vec<usize> {
        let len = edges.len();
        (0..len)
            .filter(|&i| self.mesh_ref.twin(edges[i]) == edges[(i + 1) % len])
            .collect()
    }
    // An anchor (move inside a face) for a loop of the given axis, in a face around `vertex` that is not crossed by
    // the given loop: the move that is best aligned with the flow.
    pub(crate) fn anchor_near(
        &self,
        vertex: VertID,
        axis: Direction,
        avoid: &[EdgeID],
    ) -> Option<[EdgeID; 2]> {
        let graph = &self.flow_graphs.as_ref()?[axis as usize];
        let mesh = &self.mesh_ref;
        mesh.faces(vertex)
            .filter(|&face| mesh.edges(face).all(|e| !avoid.contains(&e)))
            .flat_map(|face| {
                let edges = mesh.edges(face).collect_vec();
                edges
                    .iter()
                    .copied()
                    .cartesian_product(edges.clone())
                    .filter(|(a, b)| a != b)
                    .collect_vec()
            })
            .filter_map(|(a, b)| {
                let weight = graph.get_directed_weight(a, b)?;
                let length = (mesh.position(b) - mesh.position(a)).norm().max(1e-12);
                Some(([a, b], weight / length))
            })
            .min_by_key(|&(_, rank)| OrderedFloat(rank))
            .map(|(anchor, _)| anchor)
    }
}
