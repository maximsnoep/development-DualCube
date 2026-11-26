//! Flips for a stuck surgery.
//!
//! When surgery is stuck, every arc at a junction has one patch on both sides and a loop that
//! does not bound. Two junctions joined by an arc e can be pulled together into a four-way
//! point and apart the other way. With spokes counterclockwise (e, a, b) at the start of e and
//! (e, c, d) at its end, the new arc e' separates {b, c} from {d, a}. No patch changes, nor do
//! the classes of a, b, c, d. A small loop around a junction bounds, so the classes of its
//! arcs sum to zero, each signed by whether the arc starts or ends there ([`signed`]); this
//! gives the class of e'.
//!
//! A breadth-first search over these combinatorial flips looks for an arc whose loop bounds.
//! Only that sequence is applied to the surface.

use super::*;
use crate::surface::edit::parallel_crossings;
use rustc_hash::FxHashSet;
use std::hash::{Hash, Hasher};

pub(super) type Spoke = (usize, usize);

/// Junction spokes counterclockwise, starting with the smallest, and arc classes, with a
/// fingerprint that sums one hash per entry, so that a flip updates it in constant time.
#[derive(Clone)]
pub(super) struct Rotation {
    pub(super) spokes: Vec<[Spoke; 3]>,
    pub(super) ends: Vec<[usize; 2]>,
    pub(super) classes: Vec<Class>,
    pub(super) fingerprint: u64,
}

/// What a flip changed, to undo it.
pub(super) struct Undo {
    e: usize,
    junctions: [(usize, [Spoke; 3]); 2],
    ends: [(Spoke, usize); 2],
    class: Class,
    fingerprint: u64,
}

#[derive(Hash)]
enum Entry<'a> {
    Spokes(usize, &'a [Spoke; 3]),
    Ends(usize, &'a [usize; 2]),
    Class(usize, &'a Class),
}

fn entry_hash(entry: Entry<'_>) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    entry.hash(&mut hasher);
    hasher.finish()
}

impl Rotation {
    pub(super) fn new(spokes: Vec<[Spoke; 3]>, ends: Vec<[usize; 2]>, classes: Vec<Class>) -> Self {
        let mut rotation = Self {
            spokes,
            ends,
            classes,
            fingerprint: 0,
        };
        rotation.fingerprint = (0..rotation.spokes.len())
            .map(|j| rotation.spokes_hash(j))
            .chain((0..rotation.ends.len()).map(|a| rotation.ends_hash(a)))
            .chain((0..rotation.classes.len()).map(|a| rotation.class_hash(a)))
            .fold(0, u64::wrapping_add);
        rotation
    }

    fn spokes_hash(&self, j: usize) -> u64 {
        entry_hash(Entry::Spokes(j, &self.spokes[j]))
    }

    fn ends_hash(&self, a: usize) -> u64 {
        entry_hash(Entry::Ends(a, &self.ends[a]))
    }

    fn class_hash(&self, a: usize) -> u64 {
        entry_hash(Entry::Class(a, &self.classes[a]))
    }

    /// The hash of the entries a flip changes.
    fn touched_hash(&self, [j1, j2]: [usize; 2], b: Spoke, d: Spoke, e: usize) -> u64 {
        let ends = if b.0 == d.0 {
            self.ends_hash(b.0)
        } else {
            self.ends_hash(b.0).wrapping_add(self.ends_hash(d.0))
        };
        [self.spokes_hash(j1), self.spokes_hash(j2), ends, self.class_hash(e)]
            .into_iter()
            .fold(0, u64::wrapping_add)
    }

    /// Flip in place; `None` when both ends of the arc meet one junction.
    pub(super) fn flip(&mut self, e: usize) -> Option<Undo> {
        let [j1, j2] = self.ends[e];
        if j1 == j2 {
            return None;
        }
        let [_, a, b] = rotate(self.spokes[j1], (e, 0));
        let [_, c, d] = rotate(self.spokes[j2], (e, 1));
        let before = self.touched_hash([j1, j2], b, d, e);
        let class = through(&self.classes, b, c);
        let undo = Undo {
            e,
            junctions: [(j1, self.spokes[j1]), (j2, self.spokes[j2])],
            ends: [(b, j1), (d, j2)],
            class: std::mem::replace(&mut self.classes[e], class),
            fingerprint: self.fingerprint,
        };
        self.spokes[j1] = smallest_first([(e, 0), d, a]);
        self.spokes[j2] = smallest_first([(e, 1), b, c]);
        self.ends[b.0][b.1] = j2;
        self.ends[d.0][d.1] = j1;
        let after = self.touched_hash([j1, j2], b, d, e);
        self.fingerprint = self.fingerprint.wrapping_sub(before).wrapping_add(after);
        Some(undo)
    }

    pub(super) fn undo(&mut self, undo: Undo) {
        for (j, spokes) in undo.junctions {
            self.spokes[j] = spokes;
        }
        for ((arc, end), junction) in undo.ends {
            self.ends[arc][end] = junction;
        }
        self.classes[undo.e] = undo.class;
        self.fingerprint = undo.fingerprint;
    }

    /// Shortest sequence of arc indices after which the last flipped arc's class is zero,
    /// trying at most `budget` distinct rotation systems; nodes are u32 indices.
    pub(super) fn search(&mut self, budget: u32) -> Result<Vec<usize>, Stuck> {
        // Breadth-first, each state stored as its parent and the arc flipped from it; the
        // rotation walks between states by undoing to their common ancestor and flipping down.
        const ROOT: u32 = u32::MAX;
        let mut seen = FxHashSet::from_iter([self.fingerprint]);
        let mut nodes: Vec<(u32, u32)> = vec![(ROOT, 0)];
        let mut current: Vec<(u32, Undo)> = vec![];
        let mut truncated = false;
        let mut path = vec![];
        let mut node = 0_usize;
        while node < nodes.len() {
            // Ancestors from the root down to `node`.
            path.clear();
            let mut at = node as u32;
            while at != 0 {
                path.push(at);
                at = nodes[at as usize].0;
            }
            path.reverse();
            let common = current
                .iter()
                .zip(&path)
                .take_while(|((a, _), b)| a == *b)
                .count();
            while current.len() > common {
                self.undo(current.pop().unwrap().1);
            }
            for &step in &path[common..] {
                let undo = self.flip(nodes[step as usize].1 as usize).unwrap();
                current.push((step, undo));
            }
            for e in 0..self.ends.len() {
                let Some(undo) = self.flip(e) else {
                    continue;
                };
                if self.classes[e].is_zero() {
                    return Ok(path
                        .iter()
                        .map(|&step| nodes[step as usize].1 as usize)
                        .chain([e])
                        .collect());
                }
                let key = self.fingerprint;
                self.undo(undo);
                if seen.contains(&key) {
                    continue;
                }
                if seen.len() < budget as usize {
                    seen.insert(key);
                    nodes.push((node as u32, e as u32));
                } else {
                    truncated = true;
                }
            }
            node += 1;
        }
        let states = seen.len();
        Err(if truncated {
            Stuck::Budget { states }
        } else {
            Stuck::Exhausted { states }
        })
    }
}

/// Sign of an arc's class in the relation around a junction: a small loop around it crosses
/// arcs that start there the other way from arcs that end there.
fn signed(class: &Class, end: usize) -> Class {
    if end == 0 { -class } else { class.clone() }
}

/// Class of the arc that ends at the junction with spokes b and c.
fn through(classes: &[Class], b: Spoke, c: Spoke) -> Class {
    -&(&signed(&classes[b.0], b.1) + &signed(&classes[c.0], c.1))
}

fn rotate<T: Copy + PartialEq>(spokes: [T; 3], first: T) -> [T; 3] {
    let i = spokes.iter().position(|&s| s == first).unwrap();
    [0, 1, 2].map(|k| spokes[(i + k) % 3])
}

fn smallest_first(spokes: [Spoke; 3]) -> [Spoke; 3] {
    rotate(spokes, *spokes.iter().min().unwrap())
}

/// Why no flip sequence frees a junction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Stuck {
    /// Every reachable rotation system was tried: no flip sequence exists.
    Exhausted { states: usize },
    /// The budget of distinct rotation systems was reached first.
    Budget { states: usize },
}

impl std::fmt::Display for Stuck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exhausted { states } => write!(
                f,
                "no flip sequence frees a junction (all {states} reachable rotation systems tried)"
            ),
            Self::Budget { states } => write!(
                f,
                "flip budget exhausted ({states} rotation systems tried, more remain)"
            ),
        }
    }
}

impl Surgery<'_> {
    /// The arcs between junctions, and the rotation system with their indices.
    pub(super) fn rotation(&mut self) -> (Vec<ArcID>, Rotation) {
        let arcs: Vec<_> = self
            .arcs
            .iter()
            .filter(|(_, a)| a.ends.is_some())
            .map(|(id, _)| id)
            .collect();
        // Pending arcs were all tested, so none of these is dissolvable.
        debug_assert!(
            arcs.iter()
                .all(|&a| !self.bounds(a) || self.arcs[a].disk == Some(false))
        );
        let arc_index: HashMap<_, _> = arcs.iter().enumerate().map(|(i, &a)| (a, i)).collect();
        let junctions: Vec<_> = self.junctions.keys().collect();
        let junction_index: HashMap<_, _> =
            junctions.iter().enumerate().map(|(i, &j)| (j, i)).collect();
        let rotation = Rotation::new(
            junctions
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
            arcs.iter()
                .map(|&a| self.arcs[a].ends.unwrap().map(|j| junction_index[&j]))
                .collect(),
            arcs.iter()
                .map(|&a| self.arcs[a].class.clone().unwrap())
                .collect(),
        );
        debug_assert!(rotation.spokes.iter().all(|spokes| {
            spokes
                .iter()
                .fold(Class::default(), |sum, &(arc, end)| {
                    &sum + &signed(&rotation.classes[arc], end)
                })
                .is_zero()
        }));
        (arcs, rotation)
    }

    /// Shortest flip sequence after which the last flipped arc's loop bounds.
    pub(super) fn search(&mut self) -> Result<Vec<ArcID>, Stuck> {
        let (arcs, mut rotation) = self.rotation();
        let sequence = rotation.search(self.options.flip_budget)?;
        info!(
            "Connectivity surgery: {} flips free a junction",
            sequence.len()
        );
        Ok(sequence.into_iter().map(|e| arcs[e]).collect())
    }

    /// Flip an arc between two distinct junctions on the surface: b continues along e, d
    /// returns along a copy of e on a's side, and a copy between them becomes e.
    pub(super) fn flip(&mut self, e: ArcID) {
        let [j1, j2] = self.arcs[e].ends.unwrap();
        assert_ne!(j1, j2, "flips need two junctions");
        let [_, a, b] = rotate(self.ccw_array(j1), (e, 0));
        let [_, c, d] = rotate(self.ccw_array(j2), (e, 1));
        let classes = [b.0, c.0].map(|arc| self.arcs[arc].class.clone().unwrap());
        let class = through(&classes, (0, b.1), (1, c.1));
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
        arc.disk = None;
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

    /// A vertex cycle for the loop through `e` once flipped, found before the flip. A stuck
    /// surgery has one patch around every junction. With spokes (e, a, b) and (e, c, d), the
    /// loop runs from between a and b across b, along the side of e that b and c face, and
    /// across c to between c and d. Each crossing goes counterclockwise around its junction:
    /// from right to left of an arc starting there. Other paths in the patch differ by its
    /// loops, which bound.
    pub(super) fn flip_walk(&self, e: ArcID) -> Option<Vec<VertID>> {
        const TRIES: usize = 8;
        let [j1, j2] = self.arcs[e].ends?;
        let [_, _, b] = rotate(self.ccw_array(j1), (e, 0));
        let [_, c, _] = rotate(self.ccw_array(j2), (e, 1));
        let patch = self.arcs[e].sides[0];
        let first = self.crossing_edges(b.0, patch, b.1 == 0);
        let second = self.crossing_edges(c.0, patch, c.1 == 0);
        first.iter().take(TRIES).find_map(|&x| {
            second
                .iter()
                .take(TRIES)
                .filter_map(|&y| self.cycle(patch, &[x, y]))
                .find(|walk| self.walk_class(walk).is_zero())
        })
    }

    /// Class of a closed vertex walk.
    pub(super) fn walk_class(&self, walk: &[VertID]) -> Class {
        (0..walk.len()).fold(Class::default(), |sum, i| {
            let [x, y] = [walk[i], walk[(i + 1) % walk.len()]].map(|v| self.index[&v]);
            &sum + &self.homology.edge(x, y)
        })
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
