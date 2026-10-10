use crate::sampler::LoopSampler;
use dualcube_types::prelude::*;
use serde::{Deserialize, Serialize};
use slotmap::SlotMap;
use std::time::Instant;

slotmap::new_key_type! {
    pub struct LoopID;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Loop {
    pub edges: Vec<EdgeID>,
    pub direction: Direction,
    // Any number of loops may cross the same mesh edge. For every half-edge in `edges`, this stores
    // the relative position (in (0, 1), from root to tip) at which the loop crosses that half-edge.
    // The positions define the order of all loops crossing the same edge (and the geometry of the loop).
    // They are maintained by `LoopStateMut`, and are empty for loops that have not been inserted yet.
    #[serde(default)]
    pub offsets: Vec<f64>,
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

#[derive(Clone, Debug)]
pub struct LoopState {
    pub loops: SlotMap<LoopID, Loop>,
    // For every half-edge, the loops crossing it, ordered by their position along the half-edge (root to tip).
    pub occupied: ids::SecMap<EDGE, INPUT, Vec<LoopID>>,
}

impl LoopState {
    #[must_use]
    pub fn from_loops(mesh: &Mesh<INPUT>, loops: SlotMap<LoopID, Loop>) -> Self {
        let mut state = Self {
            loops,
            occupied: ids::SecMap::new(),
        };
        state.as_mut(mesh).recompute_occupied();
        state
    }

    pub fn as_mut<'a>(&'a mut self, mesh: &'a Mesh<INPUT>) -> LoopStateMut<'a> {
        LoopStateMut {
            mesh,
            loops: &mut self.loops,
            occupied: &mut self.occupied,
        }
    }
}

pub struct LoopStateMut<'a> {
    pub mesh: &'a Mesh<INPUT>,
    pub loops: &'a mut SlotMap<LoopID, Loop>,
    pub occupied: &'a mut ids::SecMap<EDGE, INPUT, Vec<LoopID>>,
}

impl LoopStateMut<'_> {
    /// Rebuild the per-edge loop orders from the loop offsets. Loops without (valid) offsets are
    /// (re-)inserted into the existing order, as if they were added with `add_loop`.
    pub fn recompute_occupied(&mut self) {
        let timer = Instant::now();
        let loop_count = self.loops.len();
        let mut edge_refs = 0usize;

        self.occupied.clear();

        let (ordered, unordered): (Vec<LoopID>, Vec<LoopID>) =
            self.loops.keys().partition(|&loop_id| {
                self.loops[loop_id].offsets.len() == self.loops[loop_id].edges.len()
            });

        let mut entries: HashMap<EdgeID, Vec<(OrderedFloat<f64>, LoopID)>> = HashMap::new();
        for &loop_id in &ordered {
            let loop_ = &self.loops[loop_id];
            edge_refs += loop_.edges.len();
            for (&edge, &offset) in loop_.edges.iter().zip(&loop_.offsets) {
                entries
                    .entry(edge)
                    .or_default()
                    .push((OrderedFloat(offset), loop_id));
            }
        }
        for (edge, mut list) in entries {
            list.sort();
            self.occupied.insert(
                &edge,
                list.into_iter().map(|(_, loop_id)| loop_id).collect(),
            );
        }

        // Loops without offsets are inserted one by one.
        for loop_id in unordered {
            edge_refs += self.loops[loop_id].edges.len();
            self.place(loop_id, None);
        }

        // Make sure the offsets along every edge are strictly increasing (e.g., for loops from older files).
        let edges = self.occupied.iter().map(|(edge, _)| edge).collect_vec();
        for edge in edges {
            self.ensure_ordered(edge);
        }

        debug!(
            "b_loops::recompute_occupied: loops={} edge_refs={} occupied_edges={} elapsed_ms={:.3}",
            loop_count,
            edge_refs,
            self.occupied.iter().count(),
            elapsed_ms(timer)
        );
    }

    // If the offsets of the loops crossing `edge` are not strictly increasing (in their order along the edge),
    // space them evenly.
    fn ensure_ordered(&mut self, edge: EdgeID) {
        let Some(list) = self.occupied.get(&edge) else {
            return;
        };
        let offsets = list
            .iter()
            .map(|&loop_id| self.loops[loop_id].offset(edge).unwrap_or(0.5))
            .collect_vec();
        let ordered = std::iter::once(0.)
            .chain(offsets)
            .chain(std::iter::once(1.))
            .tuple_windows()
            .all(|(a, b)| b - a > 1e-9);
        if !ordered {
            self.respace(edge);
        }
    }

    // Set the offsets of all loops crossing `edge` (and its twin) to be evenly spaced, following their order.
    fn respace(&mut self, edge: EdgeID) {
        let twin = self.mesh.twin(edge);
        let Some(list) = self.occupied.get(&edge).cloned() else {
            return;
        };
        let n = list.len() as f64;
        for (rank, loop_id) in list.into_iter().enumerate() {
            let offset = (rank as f64 + 1.) / (n + 1.);
            let loop_ = &mut self.loops[loop_id];
            if loop_.offsets.len() != loop_.edges.len() {
                loop_.offsets = vec![0.5; loop_.edges.len()];
            }
            for i in 0..loop_.edges.len() {
                if loop_.edges[i] == edge {
                    loop_.offsets[i] = offset;
                } else if loop_.edges[i] == twin {
                    loop_.offsets[i] = 1. - offset;
                }
            }
        }
    }

    pub fn del_loop(&mut self, loop_id: LoopID) {
        let timer = Instant::now();
        let edges = self.loops[loop_id].edges.clone();

        for &e in &edges {
            if let Some(v) = self.occupied.get_mut(&e) {
                v.retain(|&l| l != loop_id);
                if v.is_empty() {
                    self.occupied.remove(&e);
                }
            }
        }

        self.loops.remove(loop_id);

        debug!(
            "b_loops::del_loop: loop={loop_id:?} edges={} remaining_loops={} elapsed_ms={:.3}",
            edges.len(),
            self.loops.len(),
            elapsed_ms(timer)
        );
    }

    /// Add a loop. The loop is inserted between the loops that already cross the same edges, such that it
    /// does not cross loops of the same direction (if possible), and crosses as few other loops as possible.
    pub fn add_loop(&mut self, l: Loop) -> LoopID {
        let timer = Instant::now();
        let edge_count = l.edges.len();
        let direction = l.direction;

        let loop_id = self.insert(l);

        debug!(
            "b_loops::add_loop: loop={loop_id:?} direction={direction:?} edges={} total_loops={} elapsed_ms={:.3}",
            edge_count,
            self.loops.len(),
            elapsed_ms(timer)
        );

        loop_id
    }

    fn insert(&mut self, l: Loop) -> LoopID {
        let loop_id = self.loops.insert(l);
        self.place(loop_id, None);
        loop_id
    }

    /// Add a loop into the given gaps of the existing loop orders (see `LoopSampler::construct_valid_loop`).
    pub fn add_loop_in_gaps(&mut self, l: Loop, gaps: Vec<(usize, usize)>) -> LoopID {
        let loop_id = self.loops.insert(l);
        self.place(loop_id, Some(gaps));
        loop_id
    }

    // Insert a loop (that is not yet part of `occupied`) into the per-edge loop orders.
    fn place(&mut self, loop_id: LoopID, gaps: Option<Vec<(usize, usize)>>) {
        // Find the gaps (between the loops already crossing the edges) to insert the loop into, and its
        // positions inside these gaps.
        // A loop that already has offsets (e.g., placed earlier) and does not share any edge with other loops keeps them.
        let (gaps, offsets) = {
            let l = &self.loops[loop_id];
            let sampler = LoopSampler::new(self.mesh, self.loops, self.occupied, None);
            if let Some(gaps) = gaps {
                sampler.place_loop_in_gaps(&l.edges, l.direction, gaps)
            } else if l.offsets.len() == l.edges.len()
                && l.edges.iter().all(|e| !self.occupied.contains_key(e))
            {
                let len = l.edges.len();
                let exits = (0..len)
                    .filter(|&i| self.mesh.twin(l.edges[i]) == l.edges[(i + 1) % len])
                    .map(|i| (i, 0))
                    .collect_vec();
                (exits, l.offsets.clone())
            } else {
                sampler.place_loop(&l.edges, l.direction)
            }
        };

        let len = self.loops[loop_id].edges.len();
        self.loops[loop_id].offsets = offsets;

        let mut inserted = HashSet::new();
        for (exit, gap) in gaps {
            let edge = self.loops[loop_id].edges[exit];
            let twin = self.loops[loop_id].edges[(exit + 1) % len];
            let n = self.occupied.get(&edge).map_or(0, Vec::len);
            let gap = gap.min(n);
            for (e, position) in [(edge, gap), (twin, n - gap)] {
                if !self.occupied.contains_key(&e) {
                    self.occupied.insert(&e, vec![]);
                }
                self.occupied.get_mut(&e).unwrap().insert(position, loop_id);
                inserted.insert(e);
            }
            self.ensure_ordered(edge);
        }

        // Half-edges that are not part of a crossing (only for malformed loops) are simply appended.
        for e in self.loops[loop_id].edges.clone() {
            if inserted.insert(e) {
                if !self.occupied.contains_key(&e) {
                    self.occupied.insert(&e, vec![]);
                }
                self.occupied.get_mut(&e).unwrap().push(loop_id);
            }
        }
    }
}

impl Loop {
    #[must_use]
    pub fn new(edges: Vec<EdgeID>, direction: Direction) -> Self {
        Self {
            edges,
            direction,
            offsets: vec![],
        }
    }

    // The position (in (0, 1), from root to tip) at which this loop crosses the given half-edge.
    #[must_use]
    pub fn offset(&self, edge: EdgeID) -> Option<f64> {
        let i = self.edges.iter().position(|&e| e == edge)?;
        Some(self.offsets.get(i).copied().unwrap_or(0.5))
    }
}
