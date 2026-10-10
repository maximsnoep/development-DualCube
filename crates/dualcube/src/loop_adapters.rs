//! Compatibility wrappers for loop state and sampling.
//!
//! The loop data model and algorithms live in `dualcube-dual`. `Solution`
//! still exposes the old methods so GUI/CLI code can migrate gradually.

use crate::prelude::*;

impl Solution {
    fn loop_state_mut(&mut self) -> LoopStateMut<'_> {
        LoopStateMut {
            mesh: self.mesh_ref.as_ref(),
            loops: &mut self.loops,
            occupied: &mut self.occupied,
        }
    }

    fn loop_sampler(&self) -> LoopSampler<'_> {
        LoopSampler::new(
            self.mesh_ref.as_ref(),
            &self.loops,
            &self.occupied,
            self.flow_graphs.as_deref(),
        )
    }

    pub fn del_loop(&mut self, loop_id: LoopID) {
        self.loop_state_mut().del_loop(loop_id);
    }

    pub fn add_loop(&mut self, loop_: Loop) -> LoopID {
        self.loop_state_mut().add_loop(loop_)
    }

    /// Add a loop into the given gaps of the existing loop orders (see `sample_valid_loop`).
    pub fn add_loop_in_gaps(&mut self, loop_: Loop, gaps: Vec<(usize, usize)>) -> LoopID {
        self.loop_state_mut().add_loop_in_gaps(loop_, gaps)
    }

    /// The dual structure of the current loops (with its region locator): the stored one if it is up to date,
    /// otherwise a new one. `None` if the loops do not form a valid loop structure.
    pub fn current_dual(&self) -> Option<std::borrow::Cow<'_, Dual>> {
        if self.dual_is_current()
            && let Ok(dual) = &self.dual
        {
            return Some(std::borrow::Cow::Borrowed(dual));
        }
        Dual::from(self.mesh_ref.clone(), &self.loops)
            .ok()
            .map(std::borrow::Cow::Owned)
    }

    /// Whether the stored dual structure (with its region locator) is built from the current loops.
    pub fn dual_is_current(&self) -> bool {
        self.dual.as_ref().is_ok_and(|dual| {
            dual.has_locator()
                && dual.loops_ref.len() == self.loops.len()
                && self.loops.iter().all(|(id, l)| {
                    dual.loops_ref
                        .get(id)
                        .is_some_and(|d| d.edges == l.edges && d.offsets == l.offsets)
                })
        })
    }

    /// Sample a loop of the given axis that is valid by construction: it follows a topological structure of the
    /// paper's filtered graph G^V (see `LoopSampler::sample_valid_loop`). Starts at the given edge (random if
    /// `None`), tries up to `tries` structures. Returns the edges, cost, and the gaps to insert it into (with
    /// `add_loop_in_gaps`). Requires a valid loop structure.
    #[allow(clippy::type_complexity)]
    pub fn sample_valid_loop(
        &self,
        axis: Direction,
        start: Option<EdgeID>,
        tries: usize,
    ) -> Option<(Vec<EdgeID>, f64, Vec<(usize, usize)>)> {
        let dual = self.current_dual()?;
        self.loop_sampler()
            .sample_valid_loop(&dual, axis, start, tries, &OrderedFloat)
    }

    /// The cheapest valid loop of the given axis from the given edge (random if `None`) over all sequences of regions
    /// (see `LoopSampler::free_valid_loop`): it follows the flow as long as it likes. Returns the edges, cost, and gaps
    /// (for `add_loop_in_gaps`); the result may enter a region twice, so check the resulting loop structure.
    #[allow(clippy::type_complexity)]
    pub fn free_valid_loop(
        &self,
        axis: Direction,
        start: Option<EdgeID>,
        tries: usize,
    ) -> Option<(Vec<EdgeID>, f64, Vec<(usize, usize)>)> {
        let dual = self.current_dual()?;
        self.loop_sampler()
            .free_valid_loop(&dual, axis, start, tries, &OrderedFloat)
    }

    /// The cheapest valid loop of the given axis through the middle gap of the given half-edge (deterministic; see
    /// `LoopSampler::best_valid_loop`). Returns the edges, cost, and gaps (for `add_loop_in_gaps`).
    #[allow(clippy::type_complexity)]
    pub fn best_valid_loop_through(
        &self,
        axis: Direction,
        edge: EdgeID,
        limit: usize,
    ) -> Option<(Vec<EdgeID>, f64, Vec<(usize, usize)>)> {
        let dual = self.current_dual()?;
        let sampler = self.loop_sampler();
        let gap = sampler.crossing_count(edge) / 2;
        sampler.best_valid_loop(&dual, axis, (edge, gap), limit, &OrderedFloat)
    }

    /// Construct a loop of the given axis from the given half-edge (its middle gap) that crosses exactly the given
    /// loops, in the given order (see `LoopSampler::construct_crossing_loop`). Returns the edges, cost, and gaps (for
    /// `add_loop_in_gaps`).
    #[allow(clippy::type_complexity)]
    pub fn construct_crossing_loop(
        &self,
        axis: Direction,
        edge: EdgeID,
        crossings: &[LoopID],
    ) -> Option<(Vec<EdgeID>, f64, Vec<(usize, usize)>)> {
        let sampler = self.loop_sampler();
        let gap = sampler.crossing_count(edge) / 2;
        sampler.construct_crossing_loop(axis, (edge, gap), crossings, &OrderedFloat)
    }

    /// The cheapest path of the given axis between two gap nodes that avoids the given faces (see
    /// `LoopSampler::construct_open_path`). Returns the edges, cost, and gaps (indices into the edges).
    #[allow(clippy::type_complexity)]
    pub fn construct_open_path(
        &self,
        axis: Direction,
        from: GapNode,
        to: GapNode,
        avoid: &HashSet<FaceID>,
    ) -> Option<(Vec<EdgeID>, f64, Vec<(usize, usize)>)> {
        self.loop_sampler()
            .construct_open_path(axis, from, to, avoid, &OrderedFloat)
    }

    /// See `construct_crossing_loop`; with an extra cost per face (see `LoopSampler::with_extra_proximity`).
    pub fn construct_crossing_loop_with_cost(
        &self,
        axis: Direction,
        edge: EdgeID,
        crossings: &[LoopID],
        extra: &HashMap<FaceID, f64>,
    ) -> Option<(Vec<EdgeID>, f64, Vec<(usize, usize)>)> {
        let sampler = self.loop_sampler().with_extra_proximity(extra);
        let gap = sampler.crossing_count(edge) / 2;
        sampler.construct_crossing_loop(axis, (edge, gap), crossings, &OrderedFloat)
    }

    pub fn get_coordinates_of_loop(&self, loop_id: LoopID) -> Vec<Vector3D> {
        self.loop_sampler().get_coordinates_of_loop(loop_id)
    }

    /// The positions of a (not yet inserted) loop, as it would be placed when added.
    pub fn loop_positions(&self, edges: &[EdgeID], direction: Direction) -> Vec<Vector3D> {
        self.loop_sampler().loop_positions(edges, direction)
    }

    /// Approximate positions of a (not yet inserted) loop (fast, for previews).
    pub fn loop_positions_preview(&self, edges: &[EdgeID], direction: Direction) -> Vec<Vector3D> {
        self.loop_sampler().loop_positions_preview(edges, direction)
    }

    /// The area of the smaller part of the surface that a loop separates.
    pub fn separated_area(&self, edges: &[EdgeID]) -> f64 {
        self.loop_sampler().separated_area(edges)
    }

    pub fn get_loops_in_direction(&self, direction: Direction) -> Vec<LoopID> {
        self.loop_sampler().get_loops_in_direction(direction)
    }

    pub fn loop_to_direction(&self, loop_id: LoopID) -> Direction {
        self.loop_sampler().loop_to_direction(loop_id)
    }

    pub fn get_pairs_of_loop(&self, loop_id: LoopID) -> Vec<[EdgeID; 2]> {
        self.loop_sampler().get_pairs_of_loop(loop_id)
    }

    pub fn cycled_windows(sequence: &[EdgeID]) -> Vec<[EdgeID; 2]> {
        LoopSampler::cycled_windows(sequence)
    }

    pub fn loops_on_edge(&self, edge: EdgeID) -> Vec<LoopID> {
        self.loop_sampler().loops_on_edge(edge)
    }

    pub fn check_loop(&self, loop_edges: &[EdgeID]) -> Result<(), PropertyViolationError> {
        self.loop_sampler().check_loop(loop_edges)
    }

    pub fn construct_loop_with_anchors_and_locked_segments(
        &self,
        anchors: &[[EdgeID; 2]],
        direction: Direction,
        locked_segments: &[(Vec<EdgeID>, f64)],
        measure: impl Fn(f64) -> OrderedFloat<f64>,
    ) -> Option<(Vec<EdgeID>, f64, Vec<(Vec<EdgeID>, f64)>)> {
        self.loop_sampler()
            .construct_loop_with_anchors_and_locked_segments(
                anchors,
                direction,
                locked_segments,
                measure,
            )
    }

    pub fn sample_loops(
        &self,
        n: usize,
        axis: Direction,
        measure: impl Fn(f64) -> OrderedFloat<f64> + Sync + Send,
        score: impl Fn((&[EdgeID], f64)) -> f64 + Sync + Send,
    ) -> Vec<Vec<EdgeID>> {
        self.loop_sampler().sample_loops(n, axis, measure, score)
    }
}
