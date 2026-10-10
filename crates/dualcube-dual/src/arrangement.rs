//! The arrangement of a collection of loops on the input mesh.
//!
//! Any number of loops may cross the same mesh edge. The loops crossing an edge are ordered along
//! it, at the positions given by their offsets. Inside every face, a loop is a
//! straight chord between its two positions on the boundary of the face. Loop intersections are the
//! crossings of chords inside faces, and loop regions are formed by the pieces (cells) of faces
//! that the chords cut out, glued together across the mesh edges.
//!
//! All geometry inside a face is computed in a (orientation preserving) affine frame, in which the
//! corners of the face are (0, 0), (1, 0), and (0, 1).

use crate::PropertyViolationError;
use crate::loops::{Loop, LoopID};
use dualcube_types::prelude::*;
use slotmap::SlotMap;

pub(crate) type Point2 = [f64; 2];

// The component of every cell, the number of components, and the component of every mesh vertex.
pub(crate) type Components = (Vec<usize>, usize, HashMap<VertID, usize>);

const CORNERS: [Point2; 3] = [[0., 0.], [1., 0.], [0., 1.]];
const EPSILON: f64 = 1e-14;

fn orient(a: Point2, b: Point2, p: Point2) -> f64 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

fn lerp(a: Point2, b: Point2, t: f64) -> Point2 {
    [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
}

fn area(polygon: &[Point2]) -> f64 {
    (0..polygon.len())
        .map(|i| {
            let (a, b) = (polygon[i], polygon[(i + 1) % polygon.len()]);
            a[0] * b[1] - b[0] * a[1]
        })
        .sum::<f64>()
        / 2.
}

fn centroid(polygon: &[Point2]) -> Point2 {
    let a = area(polygon);
    if a.abs() < EPSILON {
        let n = polygon.len() as f64;
        return [
            polygon.iter().map(|p| p[0]).sum::<f64>() / n,
            polygon.iter().map(|p| p[1]).sum::<f64>() / n,
        ];
    }
    let (mut cx, mut cy) = (0., 0.);
    for i in 0..polygon.len() {
        let (p, q) = (polygon[i], polygon[(i + 1) % polygon.len()]);
        let cross = p[0] * q[1] - q[0] * p[1];
        cx += (p[0] + q[0]) * cross;
        cy += (p[1] + q[1]) * cross;
    }
    [cx / (6. * a), cy / (6. * a)]
}

// What a corner of a cell is: a corner of the face, the end of a chord on a side of the face, or the crossing of two
// chords (indices into the chords of the face).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum CellPoint {
    Corner(usize),
    ChordEnd(usize, usize),
    Crossing(usize, usize),
}

// What a side of a cell lies on: a side of the face, or a chord.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CellSide {
    Side(usize),
    Chord(usize),
}

// A corner of a cell: its position (in the frame of the face), what it is, and what the side of the cell from it to
// the next corner lies on.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CellCorner {
    pub position: Point2,
    pub point: CellPoint,
    side: CellSide,
}

// The part of a convex polygon on the given side (`sign`: 1 for left, -1 for right) of chord `j` (from `a` to `b`).
// Corners on the chord belong to both parts. The corners where the chord cuts the sides of the polygon are identified
// by what these sides lie on (not by their positions), so that neighboring cells agree on their shared corners.
fn clip(polygon: &[CellCorner], j: usize, a: Point2, b: Point2, sign: f64) -> Vec<CellCorner> {
    let mut part = vec![];
    for i in 0..polygon.len() {
        let (p, q) = (polygon[i], polygon[(i + 1) % polygon.len()]);
        let (op, oq) = (
            sign * orient(a, b, p.position),
            sign * orient(a, b, q.position),
        );
        let cut = |side| CellCorner {
            position: lerp(p.position, q.position, op / (op - oq)),
            point: match p.side {
                CellSide::Side(s) => CellPoint::ChordEnd(j, s),
                CellSide::Chord(k) => CellPoint::Crossing(j.min(k), j.max(k)),
            },
            side,
        };
        if op > 0. {
            part.push(p);
            if oq < 0. {
                // Leaving the part: along the chord, to where the chord enters it again.
                part.push(cut(CellSide::Chord(j)));
            }
        } else if op == 0. {
            // On the chord: leaving the part along the chord, or continuing along the side of the polygon.
            let side = if oq < 0. { CellSide::Chord(j) } else { p.side };
            part.push(CellCorner { side, ..p });
        } else if oq > 0. {
            // Entering the part.
            part.push(cut(p.side));
        }
    }
    part
}

// A loop passing through a face: a straight chord from where it enters the face to where it exits the face.
#[derive(Clone, Debug)]
pub(crate) struct Chord {
    pub loop_id: LoopID,
    // Index of this chord in the sequence of chords of the loop.
    pub index: usize,
    pub from: Point2,
    pub to: Point2,
    // The sides of the face it enters and exits through.
    pub from_side: usize,
    pub to_side: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct FaceArrangement {
    pub sides: [EdgeID; 3],
    pub chords: Vec<Chord>,
    // The cells (convex polygons) that the chords cut the face into, identified by their side of every chord.
    pub cells: Vec<(Vec<bool>, Vec<Point2>)>,
    // The corners of every cell (as in `cells`, with what they are).
    pub cell_corners: Vec<Vec<CellCorner>>,
    cell_index: HashMap<Vec<bool>, usize>,
}

impl FaceArrangement {
    fn signs(&self, p: Point2, skip: Option<usize>) -> Vec<bool> {
        self.chords
            .iter()
            .enumerate()
            .map(|(j, chord)| Some(j) != skip && orient(chord.from, chord.to, p) > 0.)
            .collect()
    }

    // The cell containing a point (that does not lie on any chord).
    pub fn locate(&self, p: Point2) -> Option<usize> {
        self.cell_index.get(&self.signs(p, None)).copied()
    }

    // The two cells on either side of a point on chord `j`.
    pub fn locate_beside(&self, p: Point2, j: usize) -> Option<[usize; 2]> {
        let mut signs = self.signs(p, Some(j));
        signs[j] = true;
        let left = self.cell_index.get(&signs).copied()?;
        signs[j] = false;
        let right = self.cell_index.get(&signs).copied()?;
        Some([left, right])
    }
}

/// The input mesh refined along the loops (see `Arrangement::refine`).
#[derive(Clone, Debug, Default)]
pub(crate) struct Refinement {
    // Triangles (indices into `positions`).
    pub faces: Vec<Vec<usize>>,
    pub positions: Vec<Vector3D>,
    // For every triangle: the global id of its cell, and its face of the input mesh.
    pub face_cells: Vec<usize>,
    pub face_input: Vec<FaceID>,
    // For every vertex: whether it lies on a loop, and the vertex of the input mesh it is (if any).
    pub on_loop: Vec<bool>,
    pub vertex_input: Vec<Option<VertID>>,
}

// An intersection of two loops: the crossing of two chords inside a face.
#[derive(Clone, Debug)]
pub(crate) struct Crossing {
    pub face: FaceID,
    // The two chords (indices into the chords of the face), and the parameter along each chord.
    pub chords: [usize; 2],
    pub params: [f64; 2],
    pub point: Point2,
}

#[derive(Clone, Debug)]
pub(crate) struct Arrangement {
    pub faces: HashMap<FaceID, FaceArrangement>,
    pub crossings: Vec<Crossing>,
    // For every loop, its crossings in order along the loop.
    pub loop_crossings: HashMap<LoopID, Vec<usize>>,
    // For every edge (by its canonical half-edge), the positions of the loops crossing it along the canonical half-edge.
    edge_positions: HashMap<EdgeID, Vec<f64>>,
    // The position of every loop along every (canonical) edge it crosses.
    loop_positions: HashMap<(EdgeID, LoopID), f64>,
    // Global cell ids: faces without chords have a single cell.
    cell_base: ids::SecMap<FACE, INPUT, usize>,
    cell_count: usize,
}

fn canonical(mesh: &Mesh<INPUT>, edge: EdgeID) -> EdgeID {
    let twin = mesh.twin(edge);
    if edge.raw() < twin.raw() { edge } else { twin }
}

fn sides(mesh: &Mesh<INPUT>, face: FaceID) -> Result<[EdgeID; 3], PropertyViolationError> {
    mesh.edges(face)
        .collect_array::<3>()
        .ok_or(PropertyViolationError::UnknownError)
}

fn side_index(sides: &[EdgeID; 3], edge: EdgeID) -> usize {
    sides.iter().position(|&e| e == edge).unwrap()
}

fn point_on_side(side: usize, t: f64) -> Point2 {
    lerp(CORNERS[side], CORNERS[(side + 1) % 3], t)
}

impl Arrangement {
    pub fn new(
        mesh: &Mesh<INPUT>,
        loops: &SlotMap<LoopID, Loop>,
    ) -> Result<Self, PropertyViolationError> {
        // The crossings of every loop with the mesh edges: (index of the exiting half-edge in the loop).
        let mut loop_exits: HashMap<LoopID, Vec<usize>> = HashMap::new();
        let mut per_edge: HashMap<EdgeID, Vec<(OrderedFloat<f64>, LoopID)>> = HashMap::new();
        for (loop_id, lewp) in loops {
            let len = lewp.edges.len();
            let exits = (0..len)
                .filter(|&i| mesh.twin(lewp.edges[i]) == lewp.edges[(i + 1) % len])
                .collect_vec();
            if exits.len() < 2 {
                return Err(PropertyViolationError::PathEmpty);
            }
            for &i in &exits {
                let edge = lewp.edges[i];
                let offset = lewp.offsets.get(i).copied().unwrap_or(0.5);
                let edge_c = canonical(mesh, edge);
                let offset_c = if edge_c == edge { offset } else { 1. - offset };
                per_edge
                    .entry(edge_c)
                    .or_default()
                    .push((OrderedFloat(offset_c), loop_id));
            }
            loop_exits.insert(loop_id, exits);
        }

        // The positions along the canonical half-edges.
        let mut position: HashMap<(EdgeID, LoopID), f64> = HashMap::new();
        let mut edge_positions = HashMap::new();
        for (edge_c, mut list) in per_edge {
            list.sort();
            let positions = list
                .into_iter()
                .map(|(t, loop_id)| {
                    position.insert((edge_c, loop_id), *t);
                    *t
                })
                .collect_vec();
            edge_positions.insert(edge_c, positions);
        }
        let position_along = |loop_id: LoopID, edge: EdgeID| {
            let edge_c = canonical(mesh, edge);
            let t = position[&(edge_c, loop_id)];
            if edge_c == edge { t } else { 1. - t }
        };

        // The chords of every loop.
        let mut faces: HashMap<FaceID, FaceArrangement> = HashMap::new();
        for (loop_id, lewp) in loops {
            let len = lewp.edges.len();
            let exits = &loop_exits[&loop_id];
            for k in 0..exits.len() {
                let entry = lewp.edges[(exits[k] + 1) % len];
                let exit = lewp.edges[exits[(k + 1) % exits.len()]];
                let face = mesh.face(entry);
                if mesh.face(exit) != face || entry == exit {
                    return Err(PropertyViolationError::UnknownError);
                }

                let arrangement = match faces.entry(face) {
                    std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(FaceArrangement {
                            sides: sides(mesh, face)?,
                            chords: vec![],
                            cells: vec![],
                            cell_corners: vec![],
                            cell_index: HashMap::new(),
                        })
                    }
                };
                // A loop passes through a face at most once.
                if arrangement.chords.iter().any(|c| c.loop_id == loop_id) {
                    return Err(PropertyViolationError::UnknownError);
                }

                let from_side = side_index(&arrangement.sides, entry);
                let to_side = side_index(&arrangement.sides, exit);
                let from = point_on_side(from_side, position_along(loop_id, entry));
                let to = point_on_side(to_side, position_along(loop_id, exit));
                arrangement.chords.push(Chord {
                    loop_id,
                    index: k,
                    from,
                    to,
                    from_side,
                    to_side,
                });
            }
        }

        // The crossings of chords inside faces.
        let mut crossings = vec![];
        for (&face, arrangement) in &faces {
            for [i, j] in (0..arrangement.chords.len()).array_combinations() {
                let (a, b) = (&arrangement.chords[i], &arrangement.chords[j]);
                let d1 = orient(a.from, a.to, b.from);
                let d2 = orient(a.from, a.to, b.to);
                let d3 = orient(b.from, b.to, a.from);
                let d4 = orient(b.from, b.to, a.to);
                if [d1, d2, d3, d4].iter().any(|d| d.abs() < EPSILON) {
                    warn!("Degenerate loop configuration in face {face:?}");
                    return Err(PropertyViolationError::UnknownError);
                }
                if d1.signum() != d2.signum() && d3.signum() != d4.signum() {
                    let s = d3 / (d3 - d4);
                    let u = d1 / (d1 - d2);
                    crossings.push(Crossing {
                        face,
                        chords: [i, j],
                        params: [s, u],
                        point: lerp(a.from, a.to, s),
                    });
                }
            }
        }

        // The crossings along every loop, in order.
        let mut along: HashMap<LoopID, Vec<(usize, OrderedFloat<f64>, usize)>> =
            loops.keys().map(|loop_id| (loop_id, vec![])).collect();
        for (crossing_id, crossing) in crossings.iter().enumerate() {
            for c in 0..2 {
                let chord = &faces[&crossing.face].chords[crossing.chords[c]];
                along.get_mut(&chord.loop_id).unwrap().push((
                    chord.index,
                    OrderedFloat(crossing.params[c]),
                    crossing_id,
                ));
            }
        }
        let loop_crossings = along
            .into_iter()
            .map(|(loop_id, mut list)| {
                list.sort();
                (loop_id, list.into_iter().map(|(_, _, id)| id).collect_vec())
            })
            .collect();

        // The cells of every face with chords.
        let mut cell_base = ids::SecMap::new();
        let mut cell_count = 0;
        for face in mesh.face_ids() {
            cell_base.insert(&face, cell_count);
            if let Some(arrangement) = faces.get_mut(&face) {
                let triangle = (0..3)
                    .map(|i| CellCorner {
                        position: CORNERS[i],
                        point: CellPoint::Corner(i),
                        side: CellSide::Side(i),
                    })
                    .collect_vec();
                let mut cells = vec![(vec![], triangle)];
                for (j, chord) in arrangement.chords.iter().enumerate() {
                    cells = cells
                        .into_iter()
                        .flat_map(|(signs, polygon)| {
                            [(true, 1.), (false, -1.)]
                                .into_iter()
                                .map(|(left, sign)| {
                                    (left, clip(&polygon, j, chord.from, chord.to, sign))
                                })
                                .filter(|(_, part)| {
                                    let positions = part.iter().map(|c| c.position).collect_vec();
                                    part.len() >= 3 && area(&positions) > EPSILON
                                })
                                .map(|(sign, part)| {
                                    let mut signs = signs.clone();
                                    signs.push(sign);
                                    (signs, part)
                                })
                                .collect_vec()
                        })
                        .collect();
                }
                arrangement.cell_index = cells
                    .iter()
                    .enumerate()
                    .map(|(i, (signs, _))| (signs.clone(), i))
                    .collect();
                arrangement.cell_corners =
                    cells.iter().map(|(_, corners)| corners.clone()).collect();
                arrangement.cells = cells
                    .into_iter()
                    .map(|(signs, corners)| (signs, corners.iter().map(|c| c.position).collect()))
                    .collect();
                cell_count += arrangement.cells.len();
            } else {
                cell_count += 1;
            }
        }

        Ok(Self {
            faces,
            crossings,
            loop_crossings,
            edge_positions,
            loop_positions: position,
            cell_base,
            cell_count,
        })
    }

    /// The input mesh refined along the loops: every face is split into its cells (along the chords of the loops),
    /// and the cells are triangulated (fan; cells are convex). Every face of the refined mesh thus lies inside a
    /// single cell (and loop region), and the loops run along edges of the refined mesh. The corners of the cells are
    /// identified by what they are (see `CellPoint`), not by their positions, so nearly coinciding points (e.g., a
    /// loop crossing an edge very close to a vertex) stay apart.
    pub fn refine(&self, mesh: &Mesh<INPUT>) -> Option<Refinement> {
        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        enum Key {
            Vert(VertID),
            EdgePoint(EdgeID, LoopID),
            Crossing(usize),
            // Two chords that cut each other in a cell, but are not a crossing (nearly degenerate configurations).
            Cut(FaceID, usize, usize),
        }
        let mut index: HashMap<Key, usize> = HashMap::new();
        let mut refinement = Refinement::default();
        let mut vertex = |key: Key, position: Vector3D, refinement: &mut Refinement| -> usize {
            *index.entry(key).or_insert_with(|| {
                refinement.positions.push(position);
                refinement.on_loop.push(!matches!(key, Key::Vert(_)));
                refinement.vertex_input.push(match key {
                    Key::Vert(v) => Some(v),
                    _ => None,
                });
                refinement.positions.len() - 1
            })
        };
        let mut crossing_ids: HashMap<(FaceID, usize, usize), usize> = HashMap::new();
        for (id, crossing) in self.crossings.iter().enumerate() {
            let [i, j] = crossing.chords;
            crossing_ids.insert((crossing.face, i.min(j), i.max(j)), id);
        }
        for face in mesh.face_ids() {
            let sides = sides(mesh, face).ok()?;
            let base = *self.cell_base.get(&face)?;
            let Some(arrangement) = self.faces.get(&face) else {
                let triangle = sides
                    .map(|side| {
                        let v = mesh.root(side);
                        vertex(Key::Vert(v), mesh.position(v), &mut refinement)
                    })
                    .to_vec();
                refinement.faces.push(triangle);
                refinement.face_cells.push(base);
                refinement.face_input.push(face);
                continue;
            };
            let key_of = |corner: &CellCorner| -> (Key, Vector3D) {
                match corner.point {
                    CellPoint::Corner(i) => {
                        let v = mesh.root(sides[i]);
                        (Key::Vert(v), mesh.position(v))
                    }
                    CellPoint::ChordEnd(j, side) => {
                        let chord = &arrangement.chords[j];
                        // The end of the chord on that side (or, if the chord only nearly passes through a corner of
                        // the face, the nearest end).
                        let side = if side == chord.from_side || side == chord.to_side {
                            side
                        } else {
                            let distance = |p: Point2| {
                                (p[0] - corner.position[0]).powi(2)
                                    + (p[1] - corner.position[1]).powi(2)
                            };
                            if distance(chord.from) <= distance(chord.to) {
                                chord.from_side
                            } else {
                                chord.to_side
                            }
                        };
                        let edge = canonical(mesh, sides[side]);
                        let position = self
                            .loop_positions
                            .get(&(edge, chord.loop_id))
                            .map_or_else(
                                || Self::to_3d(mesh, &sides, corner.position),
                                |&t| mesh.midpoint_offset(edge, t),
                            );
                        (Key::EdgePoint(edge, chord.loop_id), position)
                    }
                    CellPoint::Crossing(i, j) => match crossing_ids.get(&(face, i, j)) {
                        Some(&id) => (
                            Key::Crossing(id),
                            Self::to_3d(mesh, &sides, self.crossings[id].point),
                        ),
                        None => (
                            Key::Cut(face, i, j),
                            Self::to_3d(mesh, &sides, corner.position),
                        ),
                    },
                }
            };
            for (i, polygon) in arrangement.cell_corners.iter().enumerate() {
                let mut corners: Vec<usize> = vec![];
                for corner in polygon {
                    let (key, position) = key_of(corner);
                    let id = vertex(key, position, &mut refinement);
                    if corners.last() != Some(&id) {
                        corners.push(id);
                    }
                }
                while corners.len() > 1 && corners.first() == corners.last() {
                    corners.pop();
                }
                // A cell that collapses (its corners are not all different) has no area: skip it.
                if corners.len() < 3 || !corners.iter().all_unique() {
                    continue;
                }
                for k in 1..corners.len() - 1 {
                    refinement
                        .faces
                        .push(vec![corners[0], corners[k], corners[k + 1]]);
                    refinement.face_cells.push(base + i);
                    refinement.face_input.push(face);
                }
            }
        }
        Some(refinement)
    }

    // Global id of the cell containing the point (given in the frame of the face).
    fn locate(&self, face: FaceID, p: Point2) -> Option<usize> {
        let base = *self.cell_base.get_or_panic(face);
        match self.faces.get(&face) {
            Some(arrangement) => arrangement.locate(p).map(|i| base + i),
            None => Some(base),
        }
    }

    /// Global id of the cell that contains the point at parameter `t` (from root to tip) along the half-edge, on the
    /// side of its face. The point must not lie on a loop.
    pub fn locate_on_edge(&self, mesh: &Mesh<INPUT>, edge: EdgeID, t: f64) -> Option<usize> {
        let face = mesh.face(edge);
        match self.faces.get(&face) {
            Some(arrangement) => {
                let side = arrangement.sides.iter().position(|&e| e == edge)?;
                self.locate(face, point_on_side(side, t))
            }
            None => self.cell_base.get(&face).copied(),
        }
    }

    /// Map a point (given in the frame of the face) to 3D.
    pub fn to_3d(mesh: &Mesh<INPUT>, sides: &[EdgeID; 3], p: Point2) -> Vector3D {
        let c0 = mesh.position(mesh.root(sides[0]));
        let c1 = mesh.position(mesh.root(sides[1]));
        let c2 = mesh.position(mesh.root(sides[2]));
        c0 + (c1 - c0) * p[0] + (c2 - c0) * p[1]
    }

    /// The two (global) cells on either side of the piece of the loop directly after the given crossing.
    pub fn cells_after_crossing(&self, loop_id: LoopID, crossing_id: usize) -> Option<[usize; 2]> {
        let list = &self.loop_crossings[&loop_id];
        let k = list.iter().position(|&c| c == crossing_id)?;
        let next = list[(k + 1) % list.len()];

        let param = |crossing_id: usize| -> Option<(usize, f64)> {
            let crossing = &self.crossings[crossing_id];
            let arrangement = &self.faces[&crossing.face];
            (0..2).find_map(|c| {
                let chord = &arrangement.chords[crossing.chords[c]];
                (chord.loop_id == loop_id).then_some((crossing.chords[c], crossing.params[c]))
            })
        };

        let face = self.crossings[crossing_id].face;
        let (chord, s) = param(crossing_id)?;
        let s_next = if self.crossings[next].face == face {
            match param(next)? {
                (next_chord, s_next) if next_chord == chord && s_next > s => s_next,
                _ => 1.,
            }
        } else {
            1.
        };

        let arrangement = &self.faces[&face];
        let chord_obj = &arrangement.chords[chord];
        let p = lerp(chord_obj.from, chord_obj.to, (s + s_next) / 2.);
        let base = *self.cell_base.get_or_panic(face);
        arrangement
            .locate_beside(p, chord)
            .map(|[a, b]| [base + a, base + b])
    }

    /// Partition all cells into connected components (loop regions), by gluing cells across the mesh edges.
    /// Returns the component of every cell, the number of components, and the component of every mesh vertex.
    pub fn components(&self, mesh: &Mesh<INPUT>) -> Result<Components, PropertyViolationError> {
        let mut parent = (0..self.cell_count).collect_vec();
        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }

        for edge in mesh.edge_ids() {
            let twin = mesh.twin(edge);
            if canonical(mesh, edge) != edge {
                continue;
            }
            let (face, twin_face) = (mesh.face(edge), mesh.face(twin));
            if !self.faces.contains_key(&face) && !self.faces.contains_key(&twin_face) {
                let (a, b) = (
                    *self.cell_base.get_or_panic(face),
                    *self.cell_base.get_or_panic(twin_face),
                );
                let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                parent[ra] = rb;
                continue;
            }

            let side = self
                .faces
                .get(&face)
                .map_or(0, |arr| side_index(&arr.sides, edge));
            let twin_side = self
                .faces
                .get(&twin_face)
                .map_or(0, |arr| side_index(&arr.sides, twin));

            let positions = self.edge_positions.get(&edge).cloned().unwrap_or_default();
            let bounds = std::iter::once(0.)
                .chain(positions)
                .chain(std::iter::once(1.))
                .collect_vec();
            for w in bounds.windows(2) {
                let m = (w[0] + w[1]) / 2.;
                let a = self
                    .locate(face, point_on_side(side, m))
                    .ok_or(PropertyViolationError::UnknownError)?;
                let b = self
                    .locate(twin_face, point_on_side(twin_side, 1. - m))
                    .ok_or(PropertyViolationError::UnknownError)?;
                let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                parent[ra] = rb;
            }
        }

        let mut component_ids = HashMap::new();
        let components = (0..self.cell_count)
            .map(|cell| {
                let root = find(&mut parent, cell);
                let next = component_ids.len();
                *component_ids.entry(root).or_insert(next)
            })
            .collect_vec();

        let mut vert_components = HashMap::new();
        for vert in mesh.vert_ids() {
            let Some(face) = mesh.faces(vert).next() else {
                continue;
            };
            let face_sides = sides(mesh, face)?;
            let corner = face_sides
                .iter()
                .position(|&e| mesh.root(e) == vert)
                .ok_or(PropertyViolationError::UnknownError)?;
            let cell = self
                .locate(face, CORNERS[corner])
                .ok_or(PropertyViolationError::UnknownError)?;
            vert_components.insert(vert, components[cell]);
        }

        Ok((components, component_ids.len(), vert_components))
    }

    /// For every face with chords, the cells with their global ids and centroids (in 3D).
    pub fn cell_centroids(&self, mesh: &Mesh<INPUT>) -> Vec<(usize, FaceID, Vector3D)> {
        self.faces
            .iter()
            .flat_map(|(&face, arrangement)| {
                let base = *self.cell_base.get_or_panic(face);
                arrangement
                    .cells
                    .iter()
                    .enumerate()
                    .map(move |(i, (_, polygon))| {
                        (
                            base + i,
                            face,
                            Self::to_3d(mesh, &arrangement.sides, centroid(polygon)),
                        )
                    })
            })
            .collect()
    }
}
