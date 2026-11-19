//! Inserting a template loop into a drawing of other template loops, arc by arc.
//!
//! The drawing's crossing vertices carry the template's crossing vertices, so every wall
//! interval and every region has the same name in both: a region is the set of wall intervals
//! bounding it (regions are planar, so these determine it). Arc `j` runs from crossing `j` to
//! crossing `j + 1` of the loop inside one region. It may end only on the template's partner
//! interval, from the template's side; if it returns to the boundary cycle it left, crossing
//! parities with cuts put every other boundary of the region on the template's side. Any such
//! arc keeps the drawing equivalent to the template, and one always exists, so nothing is
//! searched or retried. Everything here is combinatorial; positions only score routes.

use crate::LoopID;
use crate::loop_map::{LoopMap, Piece, Point};
use dualcube_types::prelude::*;
use std::collections::{BTreeSet, BinaryHeap, VecDeque};

/// Why an arc could not be drawn, with its index along the loop. The construction always
/// succeeds on a drawing equivalent to the template, so each of these reports a bug.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DrawError {
    /// No piece of the drawing matches the template's first crossing.
    NoStart,
    NoRoute(usize),
    /// The regions after this arc differ from the template's.
    Mismatch(usize),
    /// The arc could not be placed as routed.
    Unplaced(usize),
}

impl std::fmt::Display for DrawError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoStart => f.write_str("no piece to start from"),
            Self::NoRoute(arc) => write!(f, "arc {arc} has no route"),
            Self::Mismatch(arc) => write!(f, "arc {arc} cut regions unlike the template's"),
            Self::Unplaced(arc) => write!(f, "arc {arc} could not be placed as routed"),
        }
    }
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

// Crossings of a map in template terms: by vertex, crossings of drawn loops by their template
// vertex and crossings of the inserted loop `id` by itinerary index. Loops outside `walls`
// are transparent (template loops not drawn yet).
struct Labels {
    id: LoopID,
    old: HashMap<usize, usize>,
    new: HashMap<usize, usize>,
    walls: HashSet<LoopID>,
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
    fn marks(
        &self,
        map: &LoopMap,
        chains: &HashMap<LoopID, Vec<usize>>,
        window: &Window,
    ) -> Vec<Option<Mark>> {
        let mut marks = vec![None; map.halves.len()];
        for (&wall, chain) in chains {
            if self.walls.contains(&wall) {
                self.chain_marks(map, wall, chain, window, |h, m| marks[h] = Some(m));
            }
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
            if let Some(wall) = map.halves[h].piece.wall()
                && self.walls.contains(&wall)
            {
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

    fn partition(
        &self,
        map: &LoopMap,
        chains: &HashMap<LoopID, Vec<usize>>,
        window: &Window,
    ) -> Partition {
        let marks = self.marks(map, chains, window);
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

/// The complete loop structure to draw, as a loop map of its own surface. Its crossing
/// vertices name the crossings of every drawing.
pub struct Template {
    map: LoopMap,
    chains: HashMap<LoopID, Vec<usize>>,
    // The two loops of every crossing vertex.
    crossings: HashMap<usize, [LoopID; 2]>,
}

impl Template {
    pub fn new(map: LoopMap) -> Self {
        let chains = map.chains();
        let mut crossings = HashMap::new();
        for v in (0..map.points.len()).filter(|&v| map.points[v] == Point::Crossing) {
            let mut walls: Vec<_> = map
                .outgoing(v)
                .iter()
                .filter_map(|&h| map.halves[h].piece.wall())
                .collect();
            walls.sort();
            walls.dedup();
            if let [a, b] = walls[..] {
                crossings.insert(v, [a, b]);
            }
        }
        Self {
            map,
            chains,
            crossings,
        }
    }

    pub fn map(&self) -> &LoopMap {
        &self.map
    }

    /// The loops crossed by loop `id`, in order along it.
    pub fn partners(&self, id: LoopID) -> Vec<LoopID> {
        let chain = self.chains.get(&id).map_or(&[][..], Vec::as_slice);
        chain
            .iter()
            .filter_map(|&h| self.crossings.get(&self.map.halves[h].origin))
            .map(|&[a, b]| if a == id { b } else { a })
            .collect()
    }

    // Labels of the template restricted to the `drawn` loops and `id`.
    fn labels(&self, drawn: &HashSet<LoopID>, id: LoopID) -> Labels {
        let old = self
            .crossings
            .iter()
            .filter(|(_, pair)| pair.iter().all(|l| drawn.contains(l)))
            .map(|(&v, _)| (v, v))
            .collect();
        let mut walls = drawn.clone();
        walls.insert(id);
        Labels {
            id,
            old,
            new: HashMap::new(),
            walls,
        }
    }

    /// Whether `map` holds exactly the `drawn` loops with the template's regions: every region
    /// bounded by the same wall intervals. `known` gives the template vertex of every crossing
    /// vertex of `map`.
    pub fn matches(
        &self,
        map: &LoopMap,
        drawn: &HashSet<LoopID>,
        known: &HashMap<usize, usize>,
    ) -> bool {
        let none = Window::none(0);
        let template = self.labels(drawn, LoopID::default());
        let drawing = Labels {
            old: known.clone(),
            ..self.labels(drawn, LoopID::default())
        };
        template.partition(&self.map, &self.chains, &none)
            == drawing.partition(map, &map.chains(), &none)
    }
}

/// A template loop to insert into a drawing of some of the template's other loops.
pub struct Plan<'a> {
    pub id: LoopID,
    template: &'a Template,
    // Template vertices of the crossings with drawn loops, along the loop.
    crossings: Vec<usize>,
    partners: Vec<LoopID>,
    // Whether the loop leaves each crossing on its partner's left.
    left: Vec<bool>,
    labels: Labels,
    // Template partitions by window (start, arcs, crossings).
    partitions: std::cell::RefCell<HashMap<(usize, usize, usize), std::rc::Rc<Partition>>>,
}

impl<'a> Plan<'a> {
    pub fn new(template: &'a Template, drawn: &HashSet<LoopID>, id: LoopID) -> Self {
        let map = &template.map;
        let mut labels = template.labels(drawn, id);
        let (mut crossings, mut partners, mut left) = (vec![], vec![], vec![]);
        for &h in template.chains.get(&id).map_or(&[][..], Vec::as_slice) {
            let v = map.halves[h].origin;
            let Some(&[a, b]) = template.crossings.get(&v) else {
                continue;
            };
            let partner = if a == id { b } else { a };
            if !drawn.contains(&partner) {
                continue;
            }
            // The loop leaves `v` along `h`, on the partner's left when `h` is the next
            // outgoing half-edge counterclockwise after the partner's forward one.
            let forward = map
                .outgoing(v)
                .into_iter()
                .find(|&o| map.halves[o].piece == Piece::Loop(partner, true))
                .expect("a crossing has its partner's forward piece");
            left.push(map.halves[map.halves[forward].prev].twin == h);
            labels.new.insert(v, crossings.len());
            crossings.push(v);
            partners.push(partner);
        }
        Self {
            id,
            template,
            crossings,
            partners,
            left,
            labels,
            partitions: Default::default(),
        }
    }

    /// The number of crossings with drawn loops.
    pub fn len(&self) -> usize {
        self.crossings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.crossings.is_empty()
    }

    fn expected(&self, window: &Window) -> std::rc::Rc<Partition> {
        self.partitions
            .borrow_mut()
            .entry((window.start, window.len, window.count))
            .or_insert_with(|| {
                let template = self.template;
                std::rc::Rc::new(
                    self.labels
                        .partition(&template.map, &template.chains, window),
                )
            })
            .clone()
    }

    // In `expected`, the region beside arc `j` holding the partner interval arriving at
    // crossing `j`, on the side the arc leaves to. The interval's other side may lie beside
    // the arc too, in the region beyond it.
    fn arriving(&self, expected: &Partition, window: &Window, j: usize) -> Option<usize> {
        let v = self.crossings[j];
        let map = &self.template.map;
        let partner = self.partners[j];
        let arriving = map
            .outgoing(v)
            .into_iter()
            .find(|&o| map.halves[o].piece == Piece::Loop(partner, false))
            .map(|o| map.halves[o].twin)?;
        let marks = self.labels.marks_of(map, [arriving], window);
        let beside = Mark {
            wall: self.id,
            after: After::New(j),
            left: None,
        };
        // A forward piece has its left side in its face.
        let side = if self.left[j] {
            arriving
        } else {
            map.halves[arriving].twin
        };
        let mark = *marks.get(&side)?;
        expected
            .iter()
            .position(|p| p.contains(&beside) && p.contains(&mark))
    }

    /// Partner half-edges of `map` in the template's interval for crossing `j`: after the same
    /// counted crossing, old or new. Old crossings alone cannot order several crossings of one
    /// partner segment.
    fn crossable(
        &self,
        map: &LoopMap,
        labels: &Labels,
        j: usize,
        window: &Window,
    ) -> HashSet<usize> {
        let partner = self.partners[j];
        let template = &self.template.map;
        let into = template
            .outgoing(self.crossings[j])
            .into_iter()
            .find(|&o| template.halves[o].piece == Piece::Loop(partner, false))
            .map(|o| template.halves[o].twin)
            .expect("a crossing has its partner's arriving piece");
        let want = self.labels.marks_of(template, [into], window)[&into].after;
        let walls: Vec<_> = (0..map.halves.len())
            .filter(|&h| map.halves[h].piece.wall() == Some(partner))
            .collect();
        let marks = labels.marks_of(map, walls.iter().copied(), window);
        walls
            .into_iter()
            .filter(|h| marks.get(h).is_some_and(|m| m.after == want))
            .collect()
    }

    // Where to start: the crossing with the fewest pieces to cross first, and the first of
    // them. Every piece of the template's interval, crossed towards the template's side, is
    // equivalent, so this choice only affects the shape.
    fn start(&self, map: &LoopMap, labels: &Labels) -> Option<(usize, usize)> {
        let none = Window::none(self.len());
        (0..self.len())
            .filter_map(|j| {
                // Crossing a piece enters its twin's side: the partner's left when it runs
                // backwards.
                let side = Piece::Loop(self.partners[j], !self.left[j]);
                let anchors: HashSet<_> = self
                    .crossable(map, labels, j, &none)
                    .into_iter()
                    .filter(|&h| map.halves[h].piece == side)
                    .collect();
                let middle = off_middle(map, &anchors, &[])
                    .into_iter()
                    .min_by_key(|&(h, d)| (OrderedFloat(d), h))?;
                Some((anchors.len(), j, middle.0))
            })
            .min()
            .map(|(_, j, anchor)| (j, anchor))
    }

    /// Draw the loop arc by arc into a copy of `map`, a drawing of the plan's drawn loops whose
    /// crossing vertices have the template vertices `known`. Returns the new map and the template
    /// vertex of every new crossing vertex. `cost` scores moves by input triangle, displacement,
    /// and arc.
    pub fn draw(
        &self,
        map: &LoopMap,
        known: &HashMap<usize, usize>,
        cost: impl Fn(FaceID, Vector3D, usize) -> f64,
    ) -> Result<(LoopMap, Vec<(usize, usize)>), DrawError> {
        let k = self.len();
        let mut labels = Labels {
            id: self.id,
            old: known.clone(),
            new: HashMap::new(),
            walls: self.labels.walls.clone(),
        };
        let (start, anchor) = self.start(map, &labels).ok_or(DrawError::NoStart)?;
        let mut drawing = Drawing { map: map.clone() };
        let (x0, after) = drawing.split(anchor, Point::Crossing);
        labels.new.insert(x0, start);
        // The loop closes into the start from the anchor's side, along the anchor's piece.
        let closing = drawing.map.halves[anchor].piece;
        let mut tip = (x0, drawing.map.halves[after].face);
        let mut spread = Spread::new(&drawing.map, self.id);
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
            let target = (arc + 1) % k;
            let goal = if step + 1 == k {
                // Later crossings may have split the anchor; find the piece arriving now.
                let arriving = drawing
                    .map
                    .outgoing(x0)
                    .into_iter()
                    .map(|o| drawing.map.halves[o].twin)
                    .find(|&i| drawing.map.halves[i].piece == closing);
                Goal::Close(arriving.ok_or(DrawError::Unplaced(arc))?)
            } else {
                // Only pieces of the template's interval, crossed towards the template's side.
                let side = Piece::Loop(self.partners[target], !self.left[target]);
                let window = Window::after(start, step, k);
                let mut pieces = self.crossable(&drawing.map, &labels, target, &window);
                pieces.retain(|&h| drawing.map.halves[h].piece == side);
                Goal::Cross(off_middle(&drawing.map, &pieces, &[tip.0, x0]))
            };
            let sides = drawing.sides(
                &labels,
                &mut tip,
                self.partners[arc],
                &expected,
                self.arriving(&expected, &next, arc),
                &next,
            );
            let path = drawing.route(tip.1, &goal, sides.as_ref(), &spread, arc, &cost)?;
            let triangle = |f: usize| drawing.map.faces[f].triangle;
            spread.own.insert(triangle(tip.1));
            for &h in &path {
                spread.own.insert(triangle(drawing.map.halves[drawing.map.halves[h].twin].face));
            }
            if let (Goal::Cross(_), Some(&h)) = (&goal, path.last()) {
                spread.crowded.insert(triangle(drawing.map.halves[h].face));
            }
            let piece;
            (tip, piece) = drawing.insert(self.id, tip, &path, &goal, arc)?;
            last = Some(piece);
            if let Goal::Cross(_) = goal {
                labels.new.insert(tip.0, target);
            }
        }
        let new = labels
            .new
            .into_iter()
            .map(|(v, j)| (v, self.crossings[j]))
            .collect();
        Ok((drawing.map, new))
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

// Where an arc ends: across one of these half-edges of the next partner, each with its distance
// from the middle of the interval, or back in the face of the closing half-edge.
enum Goal {
    Cross(HashMap<usize, f64>),
    Close(usize),
}

/// Cost of ending an arc outside the middle band of its interval, per unit of distance.
const MIDDLE: f64 = 10.;
/// Half-width of the free middle band, as a fraction of the interval.
const BAND: f64 = 0.3;
/// Cost of entering an input triangle that the loop already passes, in triangle sizes.
const REVISIT: f64 = 4.;
/// Cost of a crossing in an input triangle that already holds one, in triangle sizes.
const CROWDED: f64 = 4.;

/// Input triangles a new arc should stay out of. Legacy straight-chord loops (as loop evolution
/// mutates) reproduce a drawing from the order of crossings along the input edges alone when
/// every loop passes a triangle at most once and no triangle holds more than one crossing.
struct Spread {
    /// Triangles the loop being drawn passes.
    own: HashSet<FaceID>,
    /// Triangles holding a crossing.
    crowded: HashSet<FaceID>,
    /// Mean edge length of every triangle.
    size: HashMap<FaceID, f64>,
}

impl Spread {
    fn new(map: &LoopMap, id: LoopID) -> Self {
        let mut corners = HashMap::<FaceID, Vec<usize>>::new();
        let mut own = HashSet::new();
        let mut crowded = HashSet::new();
        for h in &map.halves {
            let triangle = map.faces[h.face].triangle;
            match map.points[h.origin] {
                Point::Input(_) => {
                    let list = corners.entry(triangle).or_default();
                    if !list.contains(&h.origin) {
                        list.push(h.origin);
                    }
                }
                Point::Crossing => {
                    crowded.insert(triangle);
                }
                _ => {}
            }
            if h.piece.wall() == Some(id) {
                own.insert(triangle);
            }
        }
        let size = corners
            .into_iter()
            .map(|(t, list)| {
                let n = list.len();
                let perimeter: f64 = (0..n)
                    .map(|i| (map.positions[list[i]] - map.positions[list[(i + 1) % n]]).norm())
                    .sum();
                (t, perimeter / n.max(1) as f64)
            })
            .collect();
        Self { own, crowded, size }
    }

    fn size(&self, triangle: FaceID) -> f64 {
        self.size.get(&triangle).copied().unwrap_or(0.)
    }
}

/// Distance along the loop from every piece to the middle band of its run of consecutive
/// `pieces`. A run closing on itself is opened at one of the `breaks` vertices if it passes
/// one. Any piece of the interval is valid; a cost on the ends of the interval keeps a loop
/// that is free to shrink from collapsing around a crossing.
fn off_middle(map: &LoopMap, pieces: &HashSet<usize>, breaks: &[usize]) -> HashMap<usize, f64> {
    let from: HashMap<usize, usize> = pieces.iter().map(|&h| (map.halves[h].origin, h)).collect();
    let length = |h: usize| (map.positions[map.target(h)] - map.positions[map.halves[h].origin]).norm();
    let targets: HashSet<usize> = pieces.iter().map(|&h| map.target(h)).collect();
    let mut heads: Vec<usize> = pieces
        .iter()
        .copied()
        .filter(|&h| !targets.contains(&map.halves[h].origin))
        .collect();
    heads.sort_unstable();
    let mut result = HashMap::new();
    let walk = |head: usize, result: &mut HashMap<usize, f64>| {
        let mut run = vec![];
        let (mut h, mut at) = (head, 0.);
        loop {
            let l = length(h);
            run.push((h, at + l / 2.));
            at += l;
            match from.get(&map.target(h)) {
                Some(&next) if next != head && !result.contains_key(&next) => h = next,
                _ => break,
            }
        }
        for (h, s) in run {
            result.insert(h, ((s - at / 2.).abs() - BAND * at).max(0.));
        }
    };
    for head in heads {
        walk(head, &mut result);
    }
    // Closed runs.
    let mut rest: Vec<usize> = pieces.iter().copied().filter(|h| !result.contains_key(h)).collect();
    rest.sort_unstable_by_key(|&h| (!breaks.contains(&map.halves[h].origin), h));
    for head in rest {
        if !result.contains_key(&head) {
            walk(head, &mut result);
        }
    }
    result
}

// Parity constraints of one arc.
struct Sides {
    // Edges (by half-edge) of every cut.
    cuts: Vec<HashSet<usize>>,
    // Boundary cycle of every vertex of the region, and the arc's own.
    cycle: HashMap<usize, usize>,
    own: usize,
    required: u32,
    // The partner piece arriving at the start, between the cuts' end and the start.
    arriving: HashSet<usize>,
}

impl Sides {
    fn accepts(&self, vertex: usize, bits: u32) -> bool {
        self.cycle.get(&vertex) != Some(&self.own) || bits == self.required
    }

    // Crossing `h` from `vertex` ends the arc. On the arriving piece the closing interval runs
    // from that crossing to the start and misses the cuts' end, so every parity flips.
    fn accepts_crossing(&self, h: usize, vertex: usize, bits: u32) -> bool {
        if self.arriving.contains(&h) {
            bits == self.required ^ ((1 << self.cuts.len()) - 1)
        } else {
            self.accepts(vertex, bits)
        }
    }
}

// A map being drawn into.
struct Drawing {
    map: LoopMap,
}

impl Drawing {
    // Split the edge of `h`; returns the new vertex and its half-edge on the twin's side.
    fn split(&mut self, h: usize, point: Point) -> (usize, usize) {
        let (a, b) = (self.map.halves[h].origin, self.map.target(h));
        let position = (self.map.positions[a] + self.map.positions[b]) / 2.;
        let x = self.map.split(h, point, position);
        (x, self.map.halves.len() - 1)
    }

    fn connect(&mut self, a: usize, b: usize, piece: Piece) -> usize {
        self.map.connect(a, b, piece)
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
        spread: &Spread,
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
        while let Some((std::cmp::Reverse(distance), state)) = heap.pop() {
            if distance.0 > best[&state] {
                continue;
            }
            let (face, bits) = state;
            if bits == EXITED || closing.is_some_and(|(f, v)| f == face && accepts(v, bits)) {
                done = Some(state);
                break;
            }
            let here = center(face);
            for h in self.boundary(face) {
                let half = &map.halves[h];
                let next = map.halves[half.twin].face;
                let portal = self.middle(h);
                let there = center(next);
                let step_cost = cost(map.faces[face].triangle, portal - here, arc)
                    + cost(map.faces[next].triangle, there - portal, arc);
                // Geometry only scores: a degenerate step is free, never forbidden.
                let step_cost = if step_cost.is_finite() {
                    step_cost.max(0.)
                } else {
                    0.
                };
                let mut next_bits = bits;
                let mut ending = 0.;
                let (here_t, next_t) = (map.faces[face].triangle, map.faces[next].triangle);
                match half.piece {
                    Piece::Loop(..) => {
                        let Goal::Cross(pieces) = goal else {
                            continue;
                        };
                        let Some(off) = pieces.get(&h) else {
                            continue;
                        };
                        if sides.is_some_and(|s| !s.accepts_crossing(h, half.origin, bits)) {
                            continue;
                        }
                        ending = MIDDLE * off;
                        if spread.crowded.contains(&here_t) {
                            ending += CROWDED * spread.size(here_t);
                        }
                        next_bits = EXITED;
                    }
                    _ => {
                        if here_t != next_t && spread.own.contains(&next_t) {
                            ending += REVISIT * spread.size(next_t);
                        }
                        for (c, cut) in sides.map_or(&[][..], |s| &s.cuts[..]).iter().enumerate() {
                            if cut.contains(&h) {
                                next_bits ^= 1 << c;
                            }
                        }
                    }
                }
                let d = distance.0 + step_cost + ending;
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
        let start = self.leaving(tip.0, tip.1).ok_or(DrawError::Unplaced(arc))?;
        let mut spots = vec![Spot::Corner(start)];
        spots.extend((0..path.len()).map(Spot::Cross));
        // Chord `j` runs from `spots[j]` to `spots[j + 1]` in face `faces[j]`.
        let mut faces = vec![tip.1];
        faces.extend(path.iter().map(|&h| halves[halves[h].twin].face));
        // No chord lies beyond the last crossing.
        match goal {
            Goal::Close(closing) => {
                if halves[*closing].face != *faces.last().unwrap() {
                    return Err(DrawError::Unplaced(arc));
                }
                spots.push(Spot::Corner(halves[*closing].next));
            }
            Goal::Cross(_) => {
                faces.pop();
            }
        }
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
            let a = corner(&self.map, &family, spots[j], f).ok_or(DrawError::Unplaced(arc))?;
            let b = corner(&self.map, &family, spots[j + 1], f).ok_or(DrawError::Unplaced(arc))?;
            let face = self.map.halves[a].face;
            if self.map.halves[b].face != face || family_of(&family, face) != f {
                return Err(DrawError::Unplaced(arc));
            }
            let h = self.connect(a, b, Piece::Loop(id, true));
            family.insert(self.map.faces.len() - 1, f);
            first.get_or_insert(h);
        }
        let first = first.ok_or(DrawError::Unplaced(arc))?;
        let tip = match goal {
            Goal::Close(_) => tip,
            // The corner beyond the partner, between its two pieces. The arc may have passed
            // there before when the partner has one region on both sides, so the face is
            // looked up after the chords are connected.
            Goal::Cross(_) => {
                let x = vertex[path.len() - 1];
                let map = &self.map;
                let other = |h: usize| map.halves[h].piece.wall().is_some_and(|wall| wall != id);
                let beyond = map
                    .outgoing(x)
                    .into_iter()
                    .find(|&o| other(o) && other(map.halves[o].prev))
                    .ok_or(DrawError::Unplaced(arc))?;
                (x, map.halves[beyond].face)
            }
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
        let arriving = HashSet::from([arriving, self.map.halves[arriving].twin]);
        Some(Sides {
            cuts,
            cycle,
            own,
            required,
            arriving,
        })
    }
}

#[cfg(test)]
mod tests;
