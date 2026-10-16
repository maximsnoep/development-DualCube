//! Inserting a loop into a loop map, arc by arc. Arc `j` runs from crossing `j` to crossing
//! `j + 1` of the template itinerary. After every arc the regions must correspond to the
//! template's after the same arcs, bounded by the same wall intervals; regions are planar, so
//! these intervals determine them. A correct prefix can always be completed, so arcs are chosen
//! greedily.
//!
//! An arc returning to the boundary cycle it starts from splits its region. Which side every
//! other boundary of that region ends up on is required through crossing parities with cuts.
//! Everything here is combinatorial; positions only score routes.

use crate::loop_map::{LoopMap, Piece, Point};
use crate::partial::{Correspondence, Insertion, PartialTopology};
use crate::{Loop, LoopID, PropertyViolationError};
use dualcube_types::prelude::*;
use slotmap::SlotMap;
use std::collections::{BTreeSet, BinaryHeap, VecDeque};

/// One prescribed crossing of a routed loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Step {
    pub partner: LoopID,
    /// The partner's crossing preceding the crossed interval; `None` allows any interval.
    pub segment: Option<usize>,
    /// Region entered by crossing.
    pub region: usize,
}

/// Why drawing a loop arc by arc stopped, with the arc's itinerary index.
#[derive(Clone, Copy, Debug)]
pub enum DrawError {
    NoRoute(usize),
    ExpansionLimit(usize),
    /// The regions after this arc differ from the template's.
    Mismatch(usize),
    /// The arc could not be placed as routed; never expected, since arcs are made simple.
    Revisit(usize),
}

fn root(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

fn join(parent: &mut [usize], a: usize, b: usize) {
    let (a, b) = (root(parent, a), root(parent, b));
    parent[a] = b;
}

// Where a wall interval starts along its loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum After {
    // A crossing of the previous arrangement, numbered as in the template.
    Old(usize),
    // A crossing of the inserted loop, numbered along the itinerary.
    New(usize),
    // The wall has no counted crossing.
    Start,
}

// One side of a wall interval.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Mark {
    wall: LoopID,
    after: After,
    // Unset on the inserted loop, whose orientation is decided only on acceptance.
    left: Option<bool>,
}

// The marks bounding each region.
type Partition = Vec<BTreeSet<Mark>>;

// Arcs `start..start + len` of `k` drawn, and crossings `start..start + count` counted.
#[derive(Clone, Copy, Debug)]
struct Window {
    start: usize,
    len: usize,
    count: usize,
    k: usize,
}

impl Window {
    // After the first `len` arcs from `start`.
    fn after(start: usize, len: usize, k: usize) -> Self {
        Self {
            start,
            len,
            count: (len + 1).min(k),
            k,
        }
    }
    fn none(k: usize) -> Self {
        Self {
            start: 0,
            len: 0,
            count: 0,
            k,
        }
    }
    fn offset(&self, j: usize) -> usize {
        (j + self.k - self.start) % self.k
    }
    fn arc(&self, j: usize) -> bool {
        self.offset(j) < self.len
    }
    fn crossing(&self, j: usize) -> bool {
        self.offset(j) < self.count
    }
}

// Crossings of a map in template terms: by vertex, previous crossings by their template index
// and crossings of the inserted loop `id` by itinerary index.
struct Labels {
    id: LoopID,
    old: HashMap<usize, usize>,
    new: HashMap<usize, usize>,
}

impl Labels {
    // Marks of the pieces of one loop's chain in `window`.
    fn chain_marks(
        &self,
        map: &LoopMap,
        wall: LoopID,
        chain: &[usize],
        window: &Window,
        mut put: impl FnMut(usize, Mark),
    ) {
        // A first pass carries the last counted crossing around closed loops.
        let closed = map.next_along(*chain.last().unwrap()) == Some(chain[0]);
        let mut after = After::Start;
        let mut arc = None;
        for pass in 0..2 {
            for &h in chain {
                let v = map.halves[h].origin;
                if wall == self.id {
                    if let Some(&j) = self.new.get(&v) {
                        arc = Some(j);
                    }
                } else if let Some(&o) = self.old.get(&v) {
                    after = After::Old(o);
                } else if let Some(&j) = self.new.get(&v).filter(|&&j| window.crossing(j)) {
                    after = After::New(j);
                }
                if pass == 0 && closed {
                    continue;
                }
                let t = map.halves[h].twin;
                if wall == self.id {
                    if let Some(j) = arc.filter(|&j| window.arc(j)) {
                        let mark = Mark {
                            wall,
                            after: After::New(j),
                            left: None,
                        };
                        put(h, mark);
                        put(t, mark);
                    }
                } else {
                    put(
                        h,
                        Mark {
                            wall,
                            after,
                            left: Some(true),
                        },
                    );
                    put(
                        t,
                        Mark {
                            wall,
                            after,
                            left: Some(false),
                        },
                    );
                }
            }
            if !closed {
                break;
            }
        }
    }

    // The mark of every wall half-edge in `window`; `None` where not a wall there.
    fn marks(&self, map: &LoopMap, window: &Window) -> Vec<Option<Mark>> {
        let mut marks = vec![None; map.halves.len()];
        for (wall, chain) in map.chains() {
            self.chain_marks(map, wall, &chain, window, |h, m| marks[h] = Some(m));
        }
        marks
    }

    // Marks of the loops through the given wall half-edges only.
    fn marks_of(
        &self,
        map: &LoopMap,
        halves: impl IntoIterator<Item = usize>,
        window: &Window,
    ) -> HashMap<usize, Mark> {
        let mut loops = HashMap::new();
        for h in halves {
            if let Some(wall) = map.halves[h].piece.wall() {
                loops.entry(wall).or_insert(h);
            }
        }
        let mut marks = HashMap::new();
        for (wall, h) in loops {
            self.chain_marks(map, wall, &map.chain(h), window, |h, m| {
                marks.insert(h, m);
            });
        }
        marks
    }

    fn partition(&self, map: &LoopMap, window: &Window) -> Partition {
        let marks = self.marks(map, window);
        let mut parent: Vec<usize> = (0..map.faces.len()).collect();
        for (i, h) in map.halves.iter().enumerate() {
            if marks[i].is_none() {
                join(&mut parent, h.face, map.halves[h.twin].face);
            }
        }
        let mut regions = HashMap::<usize, BTreeSet<Mark>>::new();
        for (i, h) in map.halves.iter().enumerate() {
            if let Some(mark) = marks[i] {
                let r = root(&mut parent, h.face);
                regions.entry(r).or_default().insert(mark);
            }
        }
        let mut result: Partition = regions.into_values().collect();
        result.sort();
        result
    }
}

/// A template loop insertion: crossings in order, and the template after every prefix of arcs.
pub struct Plan {
    pub id: LoopID,
    pub itinerary: Vec<Insertion>,
    /// Template topology before insertion; correspondences to drawings start here.
    pub before: PartialTopology,
    template: LoopMap,
    labels: Labels,
    // Template partitions by window (start, arcs, crossings).
    partitions: std::cell::RefCell<HashMap<(usize, usize, usize), std::rc::Rc<Partition>>>,
}

impl Plan {
    pub fn new(
        mesh: &Mesh<INPUT>,
        before: &SlotMap<LoopID, Loop>,
        after: &SlotMap<LoopID, Loop>,
        id: LoopID,
    ) -> Result<Self, PropertyViolationError> {
        let old_map = LoopMap::from_loops(mesh, before)?;
        let topology = PartialTopology::from_map(&old_map);
        let template = LoopMap::from_loops(mesh, after)?;
        // Old crossings keep their loops and position in both maps.
        let mut old = HashMap::new();
        for (c, &v) in topology.vertices.iter().enumerate() {
            let pair = topology.crossings[c];
            let found = (0..template.points.len()).find(|&w| {
                template.points[w] == Point::Crossing
                    && (template.positions[w] - old_map.positions[v]).norm() < 1e-9
                    && pair.iter().all(|&l| {
                        template
                            .outgoing(w)
                            .iter()
                            .any(|&h| template.halves[h].piece.wall() == Some(l))
                    })
            });
            old.insert(found.ok_or(PropertyViolationError::UnknownError)?, c);
        }
        let chain = template
            .chains()
            .remove(&id)
            .ok_or(PropertyViolationError::UnknownError)?;
        let crossings: Vec<_> = chain
            .iter()
            .map(|&h| template.halves[h].origin)
            .filter(|&v| template.points[v] == Point::Crossing)
            .collect();
        let k = crossings.len();
        let new: HashMap<_, _> = crossings.iter().enumerate().map(|(j, &v)| (v, j)).collect();
        let labels = Labels { id, old, new };
        // Previous regions of template faces: matched by their walls, with the loop transparent.
        let none = Window::none(k);
        let regions_before = {
            let previous = Labels {
                id,
                old: topology
                    .vertices
                    .iter()
                    .enumerate()
                    .map(|(c, &v)| (v, c))
                    .collect(),
                new: HashMap::new(),
            };
            let (region, _) = old_map.regions();
            let marks = previous.marks(&old_map, &none);
            let mut by_marks = HashMap::<BTreeSet<Mark>, usize>::new();
            let mut sets = HashMap::<usize, BTreeSet<Mark>>::new();
            for (i, h) in old_map.halves.iter().enumerate() {
                if let Some(m) = marks[i] {
                    sets.entry(region[h.face]).or_default().insert(m);
                }
            }
            for (r, s) in sets {
                by_marks.insert(s, r);
            }
            by_marks
        };
        let template_regions = {
            let marks = labels.marks(&template, &none);
            let mut parent: Vec<usize> = (0..template.faces.len()).collect();
            for (i, h) in template.halves.iter().enumerate() {
                if marks[i].is_none() {
                    join(&mut parent, h.face, template.halves[h.twin].face);
                }
            }
            let mut sets = HashMap::<usize, BTreeSet<Mark>>::new();
            for (i, h) in template.halves.iter().enumerate() {
                if let Some(m) = marks[i] {
                    let r = root(&mut parent, h.face);
                    sets.entry(r).or_default().insert(m);
                }
            }
            let single = topology.regions == 1;
            (0..template.faces.len())
                .map(|f| {
                    let r = root(&mut parent, f);
                    if single {
                        Some(0)
                    } else {
                        sets.get(&r).and_then(|s| regions_before.get(s).copied())
                    }
                })
                .collect::<Vec<_>>()
        };
        let itinerary = chain
            .iter()
            .filter(|&&h| template.points[template.halves[h].origin] == Point::Crossing)
            .map(|&h| {
                let v = template.halves[h].origin;
                let partner = template
                    .outgoing(v)
                    .iter()
                    .find_map(|&o| template.halves[o].piece.wall().filter(|&l| l != id))
                    .ok_or(PropertyViolationError::UnknownError)?;
                // The partner's previous crossing before this one, by its index along the partner.
                let pieces = &topology.along[&partner];
                let mut after = None;
                if !pieces.is_empty() {
                    let mut p = template
                        .outgoing(v)
                        .into_iter()
                        .find(|&o| template.halves[o].piece == Piece::Loop(partner, true))
                        .ok_or(PropertyViolationError::UnknownError)?;
                    loop {
                        p = template
                            .previous_along(p)
                            .ok_or(PropertyViolationError::UnknownError)?;
                        if let Some(&c) = labels.old.get(&template.halves[p].origin) {
                            after = pieces.iter().position(|&x| x == c);
                            break;
                        }
                    }
                }
                let region = template_regions[template.halves[h].face]
                    .ok_or(PropertyViolationError::UnknownError)?;
                Ok(Insertion {
                    partner,
                    after,
                    region,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            id,
            itinerary,
            before: topology,
            template,
            labels,
            partitions: Default::default(),
        })
    }

    /// Router steps in a drawing: corresponding regions and partner segments.
    pub fn steps(&self, correspondence: &Correspondence) -> Vec<Step> {
        self.itinerary
            .iter()
            .map(|i| Step {
                partner: i.partner,
                segment: i
                    .after
                    .map(|k| correspondence.crossings[self.before.along[&i.partner][k]]),
                region: correspondence.regions[i.region],
            })
            .collect()
    }

    fn expected(&self, window: &Window) -> std::rc::Rc<Partition> {
        self.partitions
            .borrow_mut()
            .entry((window.start, window.len, window.count))
            .or_insert_with(|| std::rc::Rc::new(self.labels.partition(&self.template, window)))
            .clone()
    }

    // In `expected`, the region beside arc `j` holding the partner interval arriving at
    // crossing `j`.
    fn arriving(&self, expected: &Partition, window: &Window, j: usize) -> Option<usize> {
        let v = self
            .labels
            .new
            .iter()
            .find_map(|(&v, &i)| (i == j).then_some(v))?;
        let map = &self.template;
        let partner = map
            .outgoing(v)
            .iter()
            .find_map(|&o| map.halves[o].piece.wall().filter(|&l| l != self.id))?;
        let arriving = map
            .outgoing(v)
            .into_iter()
            .find(|&o| map.halves[o].piece == Piece::Loop(partner, false))
            .map(|o| map.halves[o].twin)?;
        let marks = self.labels.marks(map, window);
        let beside = Mark {
            wall: self.id,
            after: After::New(j),
            left: None,
        };
        let sides = [marks[arriving]?, marks[map.halves[arriving].twin]?];
        expected
            .iter()
            .position(|p| p.contains(&beside) && sides.iter().any(|m| p.contains(m)))
    }

    /// Where `steps[j]` may cross in a drawing: wall half-edges of its partner and segment,
    /// leaving the region of the previous step.
    pub fn anchors(
        &self,
        map: &LoopMap,
        actual: &PartialTopology,
        steps: &[Step],
        j: usize,
    ) -> Vec<usize> {
        let k = steps.len();
        let (region, _) = map.regions();
        let segments = segments(map, actual);
        let step = steps[j];
        (0..map.halves.len())
            .filter(|&h| {
                let half = &map.halves[h];
                half.piece.wall() == Some(step.partner)
                    && step.segment.is_none_or(|s| segments[h] == Some(s))
                    && region[half.face] == steps[(j + k - 1) % k].region
                    && region[map.halves[half.twin].face] == step.region
            })
            .collect()
    }

    /// Draw the loop arc by arc into a copy of `map`, starting by crossing `anchor` (an anchor of
    /// `steps[start]`). `actual` and `correspondence` describe the map's previous arrangement.
    /// `cost` scores moves by input triangle, displacement, and arc.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &self,
        map: &LoopMap,
        actual: &PartialTopology,
        correspondence: &Correspondence,
        steps: &[Step],
        start: usize,
        anchor: usize,
        cost: impl Fn(FaceID, Vector3D, usize) -> f64,
    ) -> Result<LoopMap, DrawError> {
        let k = steps.len();
        let inverse: HashMap<_, _> = correspondence
            .crossings
            .iter()
            .enumerate()
            .map(|(t, &a)| (a, t))
            .collect();
        let mut labels = Labels {
            id: self.id,
            old: actual
                .vertices
                .iter()
                .enumerate()
                .filter_map(|(c, &v)| Some((v, *inverse.get(&c)?)))
                .collect(),
            new: HashMap::new(),
        };
        let (region, _) = map.regions();
        let mut drawing = Drawing {
            segment: segments(map, actual),
            region,
            map: map.clone(),
        };
        let (x0, after) = drawing.split(anchor, Point::Crossing);
        labels.new.insert(x0, start);
        // The loop closes into the start from the anchor's side, along the anchor's piece.
        let closing = drawing.map.halves[anchor].piece;
        let mut tip = (x0, drawing.map.halves[after].face);
        let mut last = None;
        for step in 0..k {
            let arc = (start + step) % k;
            let previous = (arc + k - 1) % k;
            // Only the regions beside the last arc changed; they must be the template's.
            if let Some(h) = last {
                let drawn = Window::after(start, step, k);
                let beside = Mark {
                    wall: self.id,
                    after: After::New(previous),
                    left: None,
                };
                let mut want: Vec<_> = self
                    .expected(&drawn)
                    .iter()
                    .filter(|p| p.contains(&beside))
                    .cloned()
                    .collect();
                want.sort();
                if drawing.beside(&labels, h, &drawn) != want {
                    return Err(DrawError::Mismatch(previous));
                }
            }
            let next = Window::after(start, step + 1, k);
            let expected = self.expected(&next);
            let goal = if step + 1 == k {
                // Later crossings may have split the anchor; find the piece arriving now.
                let arriving = drawing
                    .map
                    .outgoing(x0)
                    .into_iter()
                    .map(|o| drawing.map.halves[o].twin)
                    .find(|&i| drawing.map.halves[i].piece == closing);
                Goal::Close(arriving.ok_or(DrawError::Revisit(arc))?)
            } else {
                Goal::Cross(steps[(arc + 1) % k])
            };
            let sides = drawing.sides(
                &labels,
                &mut tip,
                steps[arc].partner,
                &expected,
                self.arriving(&expected, &next, arc),
                &next,
            );
            let path = drawing.route(tip.1, &goal, sides.as_ref(), arc, &cost)?;
            let piece;
            (tip, piece) = drawing.insert(self.id, tip, &path, &goal, arc)?;
            last = Some(piece);
            if let Goal::Cross(_) = goal {
                labels.new.insert(tip.0, (arc + 1) % k);
            }
        }
        Ok(drawing.map)
    }
}

/// Reconnect a curve where two of its chords cross, until none do. Chord `j` joins `points[j]`
/// and `points[j + 1]` in `faces[j]`; `position` orders points around the boundary of a face.
/// The two chords are replaced by the non-crossing pair that keeps one curve, reversing the
/// part between them. Ends and points stay, and crossings strictly decrease.
pub(crate) fn uncross<P: Copy, K: Ord>(
    points: &mut [P],
    faces: &mut [usize],
    position: impl Fn(P, usize) -> K,
) {
    // Whether chords (a, b) and (c, d) of one face interleave around its boundary.
    let interleave = |a: &K, b: &K, c: &K, d: &K| {
        let inside = |x: &K| (a < x && x < b) || (b < a && (x > a || x < b));
        a != b && ![a, b].contains(&c) && ![a, b].contains(&d) && inside(c) != inside(d)
    };
    let crossing = |points: &[P], faces: &[usize]| {
        for j in 0..faces.len() {
            for i in 0..j {
                let f = faces[i];
                if faces[j] != f {
                    continue;
                }
                let [a, b, c, d] = [i, i + 1, j, j + 1].map(|s| position(points[s], f));
                if interleave(&a, &b, &c, &d) {
                    return Some((i, j));
                }
            }
        }
        None
    };
    while let Some((i, j)) = crossing(points, faces) {
        points[i + 1..=j].reverse();
        faces[i + 1..j].reverse();
    }
}

// The previous crossing (by index in `actual`) before every wall half-edge along its loop.
fn segments(map: &LoopMap, actual: &PartialTopology) -> Vec<Option<usize>> {
    let index: HashMap<_, _> = actual
        .vertices
        .iter()
        .enumerate()
        .map(|(c, &v)| (v, c))
        .collect();
    let mut result = vec![None; map.halves.len()];
    for chain in map.chains().values() {
        let mut last = None;
        for _ in 0..2 {
            for &h in chain {
                if let Some(&c) = index.get(&map.halves[h].origin) {
                    last = Some(c);
                }
                result[h] = last;
                result[map.halves[h].twin] = last;
            }
        }
    }
    result
}

// Where an arc ends: across the next crossing, or back in the face of the closing half-edge.
enum Goal {
    Cross(Step),
    Close(usize),
}

// Parity constraints of one arc.
struct Sides {
    // Edges (by half-edge) of every cut.
    cuts: Vec<HashSet<usize>>,
    // Boundary cycle of every vertex of the region, and the arc's own.
    cycle: HashMap<usize, usize>,
    own: usize,
    required: u32,
    // Partner intervals between the cuts' ends and the start, where the arc may not end.
    forbidden: HashSet<usize>,
}

impl Sides {
    fn accepts(&self, vertex: usize, bits: u32) -> bool {
        self.cycle.get(&vertex) != Some(&self.own) || bits == self.required
    }
}

// A map being drawn into, with the previous region of every face and the previous segment of
// every wall half-edge.
struct Drawing {
    map: LoopMap,
    region: Vec<usize>,
    segment: Vec<Option<usize>>,
}

impl Drawing {
    // Split the edge of `h`; returns the new vertex and its half-edge on the twin's side.
    fn split(&mut self, h: usize, point: Point) -> (usize, usize) {
        let t = self.map.halves[h].twin;
        let (a, b) = (self.map.halves[h].origin, self.map.target(h));
        let position = (self.map.positions[a] + self.map.positions[b]) / 2.;
        let x = self.map.split(h, point, position);
        self.segment.push(self.segment[h]);
        self.segment.push(self.segment[t]);
        (x, self.map.halves.len() - 1)
    }

    fn connect(&mut self, a: usize, b: usize, piece: Piece) -> usize {
        let face = self.map.halves[a].face;
        let h = self.map.connect(a, b, piece);
        self.segment.extend([None, None]);
        self.region.push(self.region[face]);
        h
    }

    // A cut of free edges from vertex `from` of face `start` to vertex `to` of face `end`,
    // through faces joined by non-loop edges and across those edges at new points, touching no
    // other vertex. Splitting an edge of an earlier cut extends that cut. Returns its half-edges.
    fn cut(
        &mut self,
        from: usize,
        start: usize,
        to: usize,
        end: usize,
        cuts: &mut [HashSet<usize>],
    ) -> Option<HashSet<usize>> {
        let mut previous = HashMap::from([(start, None)]);
        let mut queue = VecDeque::from([start]);
        while let Some(f) = queue.pop_front() {
            if f == end {
                break;
            }
            for h in self.boundary(f) {
                if self.map.halves[h].piece.wall().is_some() {
                    continue;
                }
                let g = self.map.halves[self.map.halves[h].twin].face;
                if let std::collections::hash_map::Entry::Vacant(e) = previous.entry(g) {
                    e.insert(Some(h));
                    queue.push_back(g);
                }
            }
        }
        previous.get(&end)?;
        let mut path = vec![];
        let mut f = end;
        while let Some(&Some(h)) = previous.get(&f) {
            path.push(h);
            f = self.map.halves[h].face;
        }
        path.reverse();
        let mut edges = HashSet::new();
        let (mut vertex, mut face) = (from, start);
        for h in path {
            let a = self.leaving(vertex, face)?;
            let t = self.map.halves[h].twin;
            let (x, beyond) = self.split(h, Point::Other);
            let continued = self.map.halves.len() - 2;
            for c in cuts.iter_mut() {
                if c.contains(&h) || c.contains(&t) {
                    c.insert(continued);
                    c.insert(beyond);
                }
            }
            let b = self.map.halves[h].next;
            self.connect(a, b, Piece::Free);
            edges.extend([self.map.halves.len() - 2, self.map.halves.len() - 1]);
            vertex = x;
            face = self.map.halves[beyond].face;
        }
        let (a, b) = (self.leaving(vertex, face)?, self.leaving(to, face)?);
        self.connect(a, b, Piece::Free);
        edges.extend([self.map.halves.len() - 2, self.map.halves.len() - 1]);
        Some(edges)
    }

    // Faces of the region of `face`, joined across non-loop edges.
    fn piece(&self, face: usize) -> Vec<usize> {
        let mut seen = HashSet::from([face]);
        let mut queue = vec![face];
        let mut i = 0;
        while i < queue.len() {
            for h in self.boundary(queue[i]) {
                let next = self.map.halves[self.map.halves[h].twin].face;
                if self.map.halves[h].piece.wall().is_none() && seen.insert(next) {
                    queue.push(next);
                }
            }
            i += 1;
        }
        queue
    }

    // The regions beside loop half-edge `h`, as their wall marks in `window`.
    fn beside(&self, labels: &Labels, h: usize, window: &Window) -> Partition {
        let sides = [h, self.map.halves[h].twin].map(|x| self.map.halves[x].face);
        let mut done = HashSet::new();
        let mut result = vec![];
        for face in sides {
            if done.contains(&face) {
                continue;
            }
            let faces = self.piece(face);
            done.extend(faces.iter().copied());
            let walls: Vec<_> = faces
                .iter()
                .flat_map(|&f| self.boundary(f))
                .filter(|&w| self.map.halves[w].piece.wall().is_some())
                .collect();
            let marks = labels.marks_of(&self.map, walls.iter().copied(), window);
            result.push(walls.iter().filter_map(|w| marks.get(w).copied()).collect());
        }
        result.sort();
        result
    }

    fn leaving(&self, v: usize, face: usize) -> Option<usize> {
        self.map
            .outgoing(v)
            .into_iter()
            .find(|&h| self.map.halves[h].face == face)
    }

    fn boundary(&self, face: usize) -> Vec<usize> {
        let start = self.map.faces[face].half;
        let mut result = vec![start];
        let mut h = self.map.halves[start].next;
        while h != start {
            result.push(h);
            h = self.map.halves[h].next;
        }
        result
    }

    fn center(&self, face: usize) -> Vector3D {
        let halves = self.boundary(face);
        halves
            .iter()
            .map(|&h| self.map.positions[self.map.halves[h].origin])
            .sum::<Vector3D>()
            / halves.len() as f64
    }

    fn middle(&self, h: usize) -> Vector3D {
        let half = &self.map.halves[h];
        (self.map.positions[half.origin] + self.map.positions[self.map.target(h)]) / 2.
    }

    // Faces crossed by a shortest admissible route from `start`, as the half-edges crossed.
    fn route(
        &self,
        start: usize,
        goal: &Goal,
        sides: Option<&Sides>,
        arc: usize,
        cost: &impl Fn(FaceID, Vector3D, usize) -> f64,
    ) -> Result<Vec<usize>, DrawError> {
        let map = &self.map;
        // Crossing the next partner ends the arc: such states carry `EXITED` instead of bits.
        const EXITED: u32 = u32::MAX;
        let closing = match goal {
            Goal::Close(h) => Some((map.halves[*h].face, map.target(*h))),
            Goal::Cross(_) => None,
        };
        let accepts = |v: usize, bits: u32| sides.is_none_or(|s| s.accepts(v, bits));
        let mut centers = HashMap::new();
        let mut center = |f: usize| *centers.entry(f).or_insert_with(|| self.center(f));
        let mut heap = BinaryHeap::new();
        let mut best = HashMap::new();
        let mut previous = HashMap::<(usize, u32), ((usize, u32), usize)>::new();
        best.insert((start, 0u32), 0.);
        heap.push((std::cmp::Reverse(OrderedFloat(0.)), (start, 0u32)));
        let mut done = None;
        let mut expansions = 0;
        while let Some((std::cmp::Reverse(distance), state)) = heap.pop() {
            if distance.0 > best[&state] {
                continue;
            }
            let (face, bits) = state;
            if bits == EXITED || closing.is_some_and(|(f, v)| f == face && accepts(v, bits)) {
                done = Some(state);
                break;
            }
            expansions += 1;
            if expansions > 1_000_000 {
                return Err(DrawError::ExpansionLimit(arc));
            }
            let here = center(face);
            for h in self.boundary(face) {
                let half = &map.halves[h];
                let next = map.halves[half.twin].face;
                let portal = self.middle(h);
                let there = center(next);
                let step_cost = cost(map.faces[face].triangle, portal - here, arc)
                    + cost(map.faces[next].triangle, there - portal, arc);
                if !step_cost.is_finite() || step_cost <= 0. {
                    continue;
                }
                let mut next_bits = bits;
                match half.piece {
                    Piece::Loop(wall, _) => {
                        let Goal::Cross(step) = goal else {
                            continue;
                        };
                        if wall != step.partner
                            || sides.is_some_and(|s| s.forbidden.contains(&h))
                            || step.segment.is_some_and(|s| self.segment[h] != Some(s))
                            || self.region[next] != step.region
                            || !accepts(half.origin, bits)
                        {
                            continue;
                        }
                        next_bits = EXITED;
                    }
                    _ => {
                        for (c, cut) in sides.map_or(&[][..], |s| &s.cuts[..]).iter().enumerate() {
                            if cut.contains(&h) {
                                next_bits ^= 1 << c;
                            }
                        }
                    }
                }
                let d = distance.0 + step_cost;
                let state = (next, next_bits);
                if best.get(&state).is_none_or(|&old| d < old) {
                    best.insert(state, d);
                    previous.insert(state, ((face, bits), h));
                    heap.push((std::cmp::Reverse(OrderedFloat(d)), state));
                }
            }
        }
        let mut state = done.ok_or(DrawError::NoRoute(arc))?;
        let mut path = vec![];
        while state != (start, 0) {
            let (before, h) = previous[&state];
            path.push(h);
            state = before;
        }
        path.reverse();
        Ok(path)
    }

    // Add the arc from the tip (vertex, face beyond it) across the half-edges of `path`, as a
    // simple curve. Returns the new tip and the arc's first piece. Each pass through a face is a
    // chord between points on its boundary; points on an edge crossed repeatedly keep a fixed
    // order, and interleaving chords are reconnected by `uncross`.
    fn insert(
        &mut self,
        id: LoopID,
        tip: (usize, usize),
        path: &[usize],
        goal: &Goal,
        arc: usize,
    ) -> Result<((usize, usize), usize), DrawError> {
        let halves = &self.map.halves;
        // Points: the start corner, crossings by index into `path`, the closing corner.
        #[derive(Clone, Copy)]
        enum Spot {
            Corner(usize),
            Cross(usize),
        }
        let start = self.leaving(tip.0, tip.1).ok_or(DrawError::Revisit(arc))?;
        let mut spots = vec![Spot::Corner(start)];
        spots.extend((0..path.len()).map(Spot::Cross));
        // Chord `j` runs from `spots[j]` to `spots[j + 1]` in face `faces[j]`.
        let mut faces = vec![tip.1];
        faces.extend(path.iter().map(|&h| halves[halves[h].twin].face));
        // Beyond the last crossing, where the next arc starts.
        let beyond = match goal {
            Goal::Close(closing) => {
                if halves[*closing].face != *faces.last().unwrap() {
                    return Err(DrawError::Revisit(arc));
                }
                spots.push(Spot::Corner(halves[*closing].next));
                None
            }
            Goal::Cross(_) => faces.pop(),
        };
        // Crossings of every edge (by its lesser half-edge), ordered from that half's origin.
        let key = |h: usize| h.min(halves[h].twin);
        let mut along = HashMap::<usize, Vec<usize>>::new();
        for (o, &h) in path.iter().enumerate() {
            along.entry(key(h)).or_default().push(o);
        }
        let mut rank = vec![0; path.len()];
        for list in along.values() {
            for (r, &o) in list.iter().enumerate() {
                rank[o] = r;
            }
        }
        // Position of a point along the boundary of face `f`.
        let mut indices = HashMap::<usize, HashMap<usize, usize>>::new();
        for &f in &faces {
            indices.entry(f).or_insert_with(|| {
                self.boundary(f)
                    .into_iter()
                    .enumerate()
                    .map(|(i, h)| (h, i))
                    .collect()
            });
        }
        let position = |spot: Spot, f: usize| -> (usize, usize) {
            match spot {
                Spot::Corner(h) => (indices[&f][&h], 0),
                Spot::Cross(o) => {
                    let h = path[o];
                    let side = if halves[h].face == f {
                        h
                    } else {
                        halves[h].twin
                    };
                    let r = rank[o];
                    let n = along[&key(h)].len();
                    (
                        indices[&f][&side],
                        1 + if side == key(h) { r } else { n - 1 - r },
                    )
                }
            }
        };
        uncross(&mut spots, &mut faces, position);
        // Split every edge at its crossings in order.
        let mut vertex = vec![0; path.len()];
        let mut keys: Vec<_> = along.keys().copied().collect();
        keys.sort();
        for e in keys {
            let mut h = e;
            for &o in &along[&e] {
                let point = match self.map.halves[h].piece {
                    Piece::Mesh(_) => Point::OnEdge,
                    Piece::Loop(..) => Point::Crossing,
                    Piece::Free => Point::Other,
                };
                (vertex[o], _) = self.split(h, point);
                // The piece beyond the new vertex, on the same side.
                h = self.map.halves.len() - 2;
            }
        }
        // Connect the chords, each in the part of its face that its ends share.
        // The face of the path that every face added here lies in.
        let mut family: HashMap<usize, usize> = HashMap::new();
        let family_of = |family: &HashMap<usize, usize>, f: usize| *family.get(&f).unwrap_or(&f);
        let corner =
            |map: &LoopMap, family: &HashMap<usize, usize>, spot: Spot, f: usize| match spot {
                Spot::Corner(h) => Some(h),
                Spot::Cross(o) => map
                    .outgoing(vertex[o])
                    .into_iter()
                    .find(|&h| family_of(family, map.halves[h].face) == f),
            };
        let mut first = None;
        for (j, &f) in faces.iter().enumerate() {
            let a = corner(&self.map, &family, spots[j], f).ok_or(DrawError::Revisit(arc))?;
            let b = corner(&self.map, &family, spots[j + 1], f).ok_or(DrawError::Revisit(arc))?;
            let face = self.map.halves[a].face;
            if self.map.halves[b].face != face || family_of(&family, face) != f {
                return Err(DrawError::Revisit(arc));
            }
            let h = self.connect(a, b, Piece::Loop(id, true));
            family.insert(self.map.faces.len() - 1, f);
            first.get_or_insert(h);
        }
        let first = first.ok_or(DrawError::Revisit(arc))?;
        let tip = match goal {
            Goal::Close(_) => tip,
            // Nothing is added beyond the partner, so that face is unchanged.
            Goal::Cross(_) => (
                vertex[path.len() - 1],
                beyond.ok_or(DrawError::Revisit(arc))?,
            ),
        };
        Ok((tip, first))
    }

    // Cuts from every other boundary of the tip's region to the partner interval arriving at
    // the tip, with the parities that put each on the template's side, should the arc return
    // to its own boundary. Cuts are added to the map as free edges; the tip's face is updated.
    fn sides(
        &mut self,
        labels: &Labels,
        tip: &mut (usize, usize),
        wall: LoopID,
        expected: &Partition,
        reference: Option<usize>,
        window: &Window,
    ) -> Option<Sides> {
        let reference = reference?;
        // The partner piece arriving at the tip, on the tip's side; cuts end at its far vertex,
        // in this face's corner, which lies on the interval closing any returning arc.
        let out = self.leaving(tip.0, tip.1)?;
        let into = self.map.halves[out].prev;
        let arriving = match self.map.halves[into].piece {
            Piece::Loop(l, true) if l == wall => into,
            Piece::Loop(l, false) if l == wall => out,
            _ => return None,
        };
        let end = if arriving == into {
            self.map.halves[into].origin
        } else {
            self.map.target(out)
        };
        let faces = self.piece(tip.1);
        let map = &self.map;
        let mut parent: HashMap<usize, usize> = HashMap::new();
        fn find(parent: &mut HashMap<usize, usize>, v: usize) -> usize {
            let p = *parent.entry(v).or_insert(v);
            if p == v {
                return v;
            }
            let r = find(parent, p);
            parent.insert(v, r);
            r
        }
        let mut boundary = vec![];
        for h in faces.iter().flat_map(|&f| self.boundary(f)) {
            if map.halves[h].piece.wall().is_some() {
                let (a, b) = (
                    find(&mut parent, map.halves[h].origin),
                    find(&mut parent, map.target(h)),
                );
                parent.insert(a, b);
                boundary.push(h);
            }
        }
        let own = find(&mut parent, tip.0);
        let marks = labels.marks_of(map, boundary.iter().copied(), window);
        // One boundary half-edge of every other cycle, with its template side.
        let mut others = vec![];
        let mut seen = HashSet::new();
        for &h in &boundary {
            let cycle = find(&mut parent, map.halves[h].origin);
            if cycle == own || seen.contains(&cycle) {
                continue;
            }
            let Some(&mark) = marks.get(&h) else {
                continue;
            };
            seen.insert(cycle);
            // Missing where the template already counts the next crossing: the arc ends there,
            // joining that boundary instead of splitting the region.
            if let Some(found) = expected.iter().position(|p| p.contains(&mark)) {
                others.push((h, found != reference));
            }
        }
        let cycle = boundary
            .iter()
            .flat_map(|&h| [map.halves[h].origin, map.target(h)])
            .map(|v| (v, find(&mut parent, v)))
            .collect();
        if others.len() > 31 {
            return None;
        }
        let mut cuts: Vec<HashSet<usize>> = vec![];
        let mut required = 0u32;
        for (h, odd) in others {
            let (from, start) = (self.map.halves[h].origin, self.map.halves[h].face);
            let corner = self.map.halves[arriving].face;
            let edges = self.cut(from, start, end, corner, &mut cuts)?;
            // Ending on the closing interval, a cut crosses it once: the boundary lies with that
            // interval exactly when the arc crosses the cut an even number of times.
            if odd {
                required |= 1 << cuts.len();
            }
            cuts.push(edges);
        }
        tip.1 = self.map.halves[out].face;
        let forbidden = HashSet::from([arriving, self.map.halves[arriving].twin]);
        Some(Sides {
            cuts,
            cycle,
            own,
            required,
            forbidden,
        })
    }
}

#[cfg(test)]
mod tests;
