//! Flips for a stuck surgery.
//!
//! When surgery is stuck, every arc at a junction has one patch on both sides and a loop that
//! does not bound. Two junctions joined by an arc e can be pulled together into a four-way
//! point and apart the other way. With spokes counterclockwise (e, a, b) at the start of e and
//! (e, c, d) at its end, the new arc e' separates {b, c} from {d, a}. No patch changes, nor do
//! the classes of a, b, c, d; e' gets κ(b) + κ(c), as a small loop around a junction bounds.
//!
//! A breadth-first search over these combinatorial flips looks for an arc whose loop bounds.
//! Only that sequence is applied to the surface.

use super::*;
use crate::surface::edit::parallel_crossings;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

type Spoke = (usize, usize);

/// Junction spokes counterclockwise, starting with the smallest, and arc classes.
#[derive(Clone, PartialEq, Eq, Hash)]
struct Rotation {
    spokes: Vec<[Spoke; 3]>,
    ends: Vec<[usize; 2]>,
    classes: Vec<Class>,
}

impl Rotation {
    /// `None` when both ends of the arc meet one junction.
    fn flip(&self, e: usize) -> Option<Self> {
        let [j1, j2] = self.ends[e];
        if j1 == j2 {
            return None;
        }
        let [_, a, b] = rotate(self.spokes[j1], (e, 0));
        let [_, c, d] = rotate(self.spokes[j2], (e, 1));
        let mut next = self.clone();
        next.spokes[j1] = smallest_first([(e, 0), d, a]);
        next.spokes[j2] = smallest_first([(e, 1), b, c]);
        next.ends[b.0][b.1] = j2;
        next.ends[d.0][d.1] = j1;
        next.classes[e] = &self.classes[b.0] ^ &self.classes[c.0];
        Some(next)
    }
}

fn rotate<T: Copy + PartialEq>(spokes: [T; 3], first: T) -> [T; 3] {
    let i = spokes.iter().position(|&s| s == first).unwrap();
    [0, 1, 2].map(|k| spokes[(i + k) % 3])
}

fn smallest_first(spokes: [Spoke; 3]) -> [Spoke; 3] {
    rotate(spokes, *spokes.iter().min().unwrap())
}

impl Surgery<'_> {
    /// Shortest flip sequence after which the last flipped arc's loop bounds.
    pub(super) fn search(&mut self) -> Option<Vec<ArcID>> {
        let arcs: Vec<_> = self
            .arcs
            .iter()
            .filter(|(_, a)| a.ends.is_some())
            .map(|(id, _)| id)
            .collect();
        for &arc in &arcs {
            if self.bounds(arc) {
                return Some(Vec::new());
            }
        }
        let arc_index: HashMap<_, _> = arcs.iter().enumerate().map(|(i, &a)| (a, i)).collect();
        let junctions: Vec<_> = self.junctions.keys().collect();
        let junction_index: HashMap<_, _> =
            junctions.iter().enumerate().map(|(i, &j)| (j, i)).collect();
        let start = Rotation {
            spokes: junctions
                .iter()
                .map(|&j| {
                    let spokes: Vec<_> = self
                        .ccw(j)
                        .into_iter()
                        .map(|(arc, end)| (arc_index[&arc], end))
                        .collect();
                    smallest_first(spokes.try_into().expect("junctions have degree three"))
                })
                .collect(),
            ends: arcs
                .iter()
                .map(|&a| self.arcs[a].ends.unwrap().map(|j| junction_index[&j]))
                .collect(),
            classes: arcs
                .iter()
                .map(|&a| self.arcs[a].class.clone().unwrap())
                .collect(),
        };
        let fingerprint = |r: &Rotation| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            r.hash(&mut hasher);
            hasher.finish()
        };
        let mut seen = HashSet::from([fingerprint(&start)]);
        let mut queue = VecDeque::from([(start, Vec::new())]);
        while let Some((rotation, path)) = queue.pop_front() {
            for e in 0..arcs.len() {
                let Some(next) = rotation.flip(e) else {
                    continue;
                };
                let mut sequence = path.clone();
                sequence.push(e);
                if self.homology.bounds(next.classes[e].clone()) {
                    info!(
                        "Connectivity surgery: {} flips free a junction",
                        sequence.len()
                    );
                    return Some(sequence.into_iter().map(|i| arcs[i]).collect());
                }
                if seen.len() < self.options.flip_budget && seen.insert(fingerprint(&next)) {
                    queue.push_back((next, sequence));
                }
            }
        }
        None
    }

    /// Flip an arc between two distinct junctions on the surface: b continues along e, d
    /// returns along a copy of e on a's side, and a copy between them becomes e.
    pub(super) fn flip(&mut self, e: ArcID) {
        let [j1, j2] = self.arcs[e].ends.unwrap();
        assert_ne!(j1, j2, "flips need two junctions");
        let [_, a, b] = rotate(self.ccw_array(j1), (e, 0));
        let [_, c, d] = rotate(self.ccw_array(j2), (e, 1));
        let class = &self.arcs[b.0].class.clone().unwrap() ^ self.arcs[c.0].class.as_ref().unwrap();
        let original: Vec<_> = self.arcs[e].crossings.iter().copied().collect();
        // Spokes counterclockwise from e lie on its left.
        let outer = parallel_crossings(self.mesh, &mut self.surface, &original, 0);
        let inner = parallel_crossings(self.mesh, &mut self.surface, &original, 0);
        self.extend(b, &original, j2);
        let back: Vec<_> = outer.into_iter().rev().collect();
        for &crossing in &back {
            let exit = &mut self.surface.crossings[crossing].exit;
            *exit = self.mesh.twin(*exit);
        }
        self.extend(d, &back, j1);
        let arc = &mut self.arcs[e];
        arc.crossings = inner.into();
        arc.class = Some(class);
        for (junction, old, new) in [(j1, b, d), (j2, d, b)] {
            for spoke in &mut self.junctions[junction].spokes {
                if *spoke == old {
                    *spoke = new;
                }
            }
        }
        debug_assert_eq!(rotate(self.ccw_array(j1), (e, 0)), [(e, 0), d, a]);
        debug_assert_eq!(rotate(self.ccw_array(j2), (e, 1)), [(e, 1), b, c]);
    }

    /// Continue an arc end along crossings oriented away from it, to a new junction.
    fn extend(
        &mut self,
        (arc, end): (ArcID, usize),
        path: &[BoundaryCrossingID],
        junction: JunctionID,
    ) {
        if end == 1 {
            self.arcs[arc].crossings.extend(path);
        } else {
            for &crossing in path {
                let exit = &mut self.surface.crossings[crossing].exit;
                *exit = self.mesh.twin(*exit);
                self.arcs[arc].crossings.push_front(crossing);
            }
        }
        self.arcs[arc].ends.as_mut().unwrap()[end] = junction;
    }

    fn ccw_array(&self, junction: JunctionID) -> [(ArcID, usize); 3] {
        self.ccw(junction)
            .try_into()
            .expect("junctions have degree three")
    }

    /// Spokes in counterclockwise order around the junction's triangle.
    pub(super) fn ccw(&self, junction: JunctionID) -> Vec<(ArcID, usize)> {
        let face = self.junctions[junction].face;
        let sides: Vec<_> = self.mesh.edges(face).collect();
        let mut spokes = self.junctions[junction].spokes.clone();
        spokes.sort_by_cached_key(|&(arc, end)| {
            let arc = &self.arcs[arc];
            let (crossing, side) = if end == 0 {
                (arc.crossings[0], self.exit(arc.crossings[0]))
            } else {
                let last = *arc.crossings.back().unwrap();
                (last, self.mesh.twin(self.exit(last)))
            };
            let key = canonical(self.mesh, side);
            let order = &self.surface.edge_order[&key];
            let rank = order.iter().position(|&c| c == crossing).unwrap();
            // Edge orders run from root to tip of the canonical half-edge.
            let along = if side == key {
                rank
            } else {
                order.len() - 1 - rank
            };
            (sides.iter().position(|&s| s == side).unwrap(), along)
        });
        spokes
    }
}
