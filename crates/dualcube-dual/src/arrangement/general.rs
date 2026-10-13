//! Convex arrangement cells for finite surface segments, including interior bends.

use super::*;
use crate::{SurfaceArc, surface_curve::SurfaceCurveError};

const TOL: f64 = 1e-12;

#[derive(Clone, Debug)]
pub(crate) struct Cell {
    pub face: FaceID,
    pub points: Vec<Point2>,
    pub vertices: Vec<usize>,
}

#[derive(Clone, Debug)]
pub(crate) struct Link {
    pub cells: [usize; 2],
    pub sides: [usize; 2],
    pub wall: Option<LoopID>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Cells {
    pub cells: Vec<Cell>,
    pub links: Vec<Link>,
    positions: Vec<Vector3D>,
    original: Vec<Option<VertID>>,
    on_loop: Vec<bool>,
}

fn root(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}
fn join(parent: &mut [usize], a: usize, b: usize) {
    let a = root(parent, a);
    let b = root(parent, b);
    parent[a] = b;
}
fn close(a: Point2, b: Point2) -> bool {
    (a[0] - b[0]).abs().max((a[1] - b[1]).abs()) < TOL
}
fn on(a: Point2, b: Point2, p: Point2) -> bool {
    let length = (b[0] - a[0]).hypot(b[1] - a[1]);
    orient(a, b, p).abs() <= TOL * length
        && (p[0] - a[0]) * (p[0] - b[0]) + (p[1] - a[1]) * (p[1] - b[1]) <= TOL * TOL
}
pub(crate) fn contains(polygon: &[Point2], p: Point2) -> bool {
    (0..polygon.len()).all(|i| {
        let a = polygon[i];
        let b = polygon[(i + 1) % polygon.len()];
        orient(a, b, p) >= -TOL * (b[0] - a[0]).hypot(b[1] - a[1])
    })
}
fn cut(polygon: &[Point2], a: Point2, b: Point2, sign: f64) -> Vec<Point2> {
    let length = (b[0] - a[0]).hypot(b[1] - a[1]);
    let mut result = Vec::new();
    for i in 0..polygon.len() {
        let p = polygon[i];
        let q = polygon[(i + 1) % polygon.len()];
        let op = sign * orient(a, b, p) / length;
        let oq = sign * orient(a, b, q) / length;
        if op >= -TOL {
            result.push(p);
        }
        if (op > TOL && oq < -TOL) || (op < -TOL && oq > TOL) {
            result.push(lerp(p, q, op / (op - oq)));
        }
    }
    result.dedup_by(|a, b| close(*a, *b));
    if result.len() > 1 && close(result[0], *result.last().unwrap()) {
        result.pop();
    }
    result
}
fn side_at(p: Point2) -> Option<(usize, f64)> {
    if p[1].abs() < TOL {
        Some((0, p[0]))
    } else if (p[0] + p[1] - 1.).abs() < TOL {
        Some((1, p[1]))
    } else if p[0].abs() < TOL {
        Some((2, 1. - p[1]))
    } else {
        None
    }
}

fn bad(reason: SurfaceCurveError) -> PropertyViolationError {
    debug!(?reason, "Invalid explicit surface curve");
    PropertyViolationError::InvalidSurfaceCurve(reason)
}

pub(crate) fn build(
    mesh: &Mesh<INPUT>,
    loops: &SlotMap<LoopID, Loop>,
) -> Result<Arrangement, PropertyViolationError> {
    let mut faces = HashMap::new();
    let mut lengths = HashMap::new();
    for (id, l) in loops {
        let arcs = l.surface_arcs(mesh)?;
        lengths.insert(id, arcs.len());
        for (i, arc) in arcs.iter().enumerate() {
            let next = &arcs[(i + 1) % arcs.len()];
            for p in [arc.from, arc.to] {
                if p.iter().any(|x| !x.is_finite())
                    || p[0] < -TOL
                    || p[1] < -TOL
                    || p[0] + p[1] > 1. + TOL
                {
                    return Err(bad(SurfaceCurveError::PointOutsideTriangle));
                }
                if CORNERS.iter().any(|&c| close(c, p)) {
                    return Err(bad(SurfaceCurveError::MeshVertex));
                }
            }
            if close(arc.from, arc.to) {
                return Err(bad(SurfaceCurveError::ZeroSegment));
            }
            let side = sides(mesh, arc.face)?;
            if arc.face == next.face {
                if !close(arc.to, next.from) {
                    return Err(bad(SurfaceCurveError::DisconnectedSegments));
                }
            } else {
                let (a, t) =
                    side_at(arc.to).ok_or_else(|| bad(SurfaceCurveError::InteriorFaceJump))?;
                let (b, u) =
                    side_at(next.from).ok_or_else(|| bad(SurfaceCurveError::InteriorFaceJump))?;
                if mesh.twin(side[a]) != sides(mesh, next.face)?[b] || (t + u - 1.).abs() > TOL {
                    return Err(bad(SurfaceCurveError::DisconnectedTransition));
                }
            }
            let face = faces.entry(arc.face).or_insert_with(|| FaceArrangement {
                sides: side,
                chords: vec![],
                cells: vec![],
                cell_corners: vec![],
                cell_index: HashMap::new(),
                explicit: true,
            });
            face.chords.push(Chord {
                loop_id: id,
                index: i,
                from: arc.from,
                to: arc.to,
                from_side: side_at(arc.from).map_or(3, |x| x.0),
                to_side: side_at(arc.to).map_or(3, |x| x.0),
            });
        }
    }
    let mut crossings = Vec::new();
    for (&face, arr) in &faces {
        for i in 0..arr.chords.len() {
            for j in i + 1..arr.chords.len() {
                let a = &arr.chords[i];
                let b = &arr.chords[j];
                let d = [
                    orient(a.from, a.to, b.from),
                    orient(a.from, a.to, b.to),
                    orient(b.from, b.to, a.from),
                    orient(b.from, b.to, a.to),
                ];
                let shared = [a.from, a.to]
                    .iter()
                    .any(|&p| [b.from, b.to].iter().any(|&q| close(p, q)));
                let adjacent = a.loop_id == b.loop_id
                    && ((a.index + 1) % lengths[&a.loop_id] == b.index
                        || (b.index + 1) % lengths[&a.loop_id] == a.index);
                if shared && adjacent {
                    if on(a.from, a.to, lerp(b.from, b.to, 0.5))
                        || on(b.from, b.to, lerp(a.from, a.to, 0.5))
                    {
                        return Err(bad(SurfaceCurveError::OverlappingAdjacentSegments));
                    }
                    continue;
                }
                if [a.from, a.to].iter().any(|&p| on(b.from, b.to, p))
                    || [b.from, b.to].iter().any(|&p| on(a.from, a.to, p))
                {
                    return Err(bad(SurfaceCurveError::TouchingCurves));
                }
                if d[0] * d[1] < 0. && d[2] * d[3] < 0. {
                    if a.loop_id == b.loop_id {
                        return Err(bad(SurfaceCurveError::SelfIntersection));
                    }
                    let s = d[2] / (d[2] - d[3]);
                    let t = d[0] / (d[0] - d[1]);
                    let point = lerp(a.from, a.to, s);
                    if crossings
                        .iter()
                        .any(|c: &Crossing| c.face == face && close(c.point, point))
                    {
                        return Err(bad(SurfaceCurveError::TripleIntersection));
                    }
                    crossings.push(Crossing {
                        face,
                        chords: [i, j],
                        params: [s, t],
                        point,
                    });
                }
            }
        }
    }
    let mut along: HashMap<_, Vec<_>> = loops.keys().map(|id| (id, vec![])).collect();
    for (index, c) in crossings.iter().enumerate() {
        for k in 0..2 {
            let a = &faces[&c.face].chords[c.chords[k]];
            along
                .get_mut(&a.loop_id)
                .unwrap()
                .push((a.index, OrderedFloat(c.params[k]), index));
        }
    }
    let loop_crossings = along
        .into_iter()
        .map(|(id, mut values)| {
            values.sort();
            (id, values.into_iter().map(|v| v.2).collect())
        })
        .collect();
    // Supporting lines give convex cells. Their extensions are traversable, never loop barriers.
    // TODO: Use a finite-segment tessellation to avoid quadratic growth from many interior bends.
    for arr in faces.values_mut() {
        let mut polygons = vec![CORNERS.to_vec()];
        for a in &arr.chords {
            let mut parts = vec![];
            for polygon in polygons {
                let length = (a.to[0] - a.from[0]).hypot(a.to[1] - a.from[1]);
                let signs: Vec<_> = polygon
                    .iter()
                    .map(|&p| orient(a.from, a.to, p) / length)
                    .collect();
                if !signs.iter().any(|&v| v > TOL) || !signs.iter().any(|&v| v < -TOL) {
                    parts.push(polygon);
                    continue;
                }
                for sign in [-1., 1.] {
                    let p = cut(&polygon, a.from, a.to, sign);
                    if p.len() >= 3 && area(&p) > TOL * TOL {
                        parts.push(p);
                    }
                }
            }
            polygons = parts;
        }
        arr.cells = polygons.into_iter().map(|p| (vec![], p)).collect();
    }
    // Include cuts from both incident faces so derived tessellations remain conforming.
    let mut edge_cuts = HashMap::<EdgeID, Vec<f64>>::new();
    for arr in faces.values() {
        for (_, polygon) in &arr.cells {
            for &p in polygon {
                if let Some((side, t)) = side_at(p) {
                    let edge = arr.sides[side];
                    let key = canonical(mesh, edge);
                    edge_cuts
                        .entry(key)
                        .or_default()
                        .push(if edge == key { t } else { 1. - t });
                }
            }
        }
    }
    for cuts in edge_cuts.values_mut() {
        cuts.extend([0., 1.]);
        cuts.sort_by(f64::total_cmp);
        cuts.dedup_by(|a, b| (*a - *b).abs() < TOL);
    }
    let mut data = Cells::default();
    let mut vertex_ids = HashMap::new();
    let mut edge_ids = HashMap::<(EdgeID, usize), usize>::new();
    let mut cell_base = ids::SecMap::new();
    for face in mesh.face_ids() {
        let side = sides(mesh, face)?;
        let arr = faces.entry(face).or_insert_with(|| FaceArrangement {
            sides: side,
            chords: vec![],
            cells: vec![(vec![], CORNERS.to_vec())],
            cell_corners: vec![],
            cell_index: HashMap::new(),
            explicit: true,
        });
        cell_base.insert(&face, data.cells.len());
        let mut local = Vec::<(Point2, usize)>::new();
        for (_, polygon) in &mut arr.cells {
            let mut expanded = vec![];
            for i in 0..polygon.len() {
                let p = polygon[i];
                let q = polygon[(i + 1) % polygon.len()];
                expanded.push(p);
                for s in 0..3 {
                    if !on(CORNERS[s], CORNERS[(s + 1) % 3], p)
                        || !on(CORNERS[s], CORNERS[(s + 1) % 3], q)
                    {
                        continue;
                    }
                    let edge = side[s];
                    let key = canonical(mesh, edge);
                    let mut extra = vec![];
                    for &t in edge_cuts.get(&key).into_iter().flatten() {
                        let point = point_on_side(s, if edge == key { t } else { 1. - t });
                        if on(p, q, point) && !close(point, p) && !close(point, q) {
                            extra.push(point);
                        }
                    }
                    extra.sort_by(|a, b| {
                        let d = |x: Point2| (x[0] - p[0]).powi(2) + (x[1] - p[1]).powi(2);
                        d(*a).total_cmp(&d(*b))
                    });
                    expanded.extend(extra);
                    break;
                }
            }
            *polygon = expanded;
            let mut vertices = vec![];
            for &p in polygon.iter() {
                let original = CORNERS
                    .iter()
                    .position(|&c| close(c, p))
                    .map(|i| mesh.root(side[i]));
                let on_loop = arr.chords.iter().any(|c| on(c.from, c.to, p));
                let boundary = side_at(p).map(|(s, t)| {
                    let e = side[s];
                    let key = canonical(mesh, e);
                    let t = if key == e { t } else { 1. - t };
                    let rank = edge_cuts
                        .get(&key)
                        .and_then(|cuts| cuts.iter().position(|&u| (u - t).abs() < TOL))
                        .unwrap_or(0);
                    (key, rank)
                });
                let old = if let Some(v) = original {
                    vertex_ids.get(&v).copied()
                } else if let Some(key) = boundary {
                    edge_ids.get(&key).copied()
                } else {
                    local.iter().find(|(q, _)| close(*q, p)).map(|x| x.1)
                };
                let id = old.unwrap_or_else(|| {
                    let id = data.positions.len();
                    data.positions.push(SurfaceArc::position(mesh, face, p));
                    data.original.push(original);
                    data.on_loop.push(false);
                    if let Some(v) = original {
                        vertex_ids.insert(v, id);
                    } else if let Some(key) = boundary {
                        edge_ids.insert(key, id);
                    } else {
                        local.push((p, id));
                    }
                    id
                });
                data.on_loop[id] |= on_loop;
                vertices.push(id);
            }
            data.cells.push(Cell {
                face,
                points: polygon.clone(),
                vertices,
            });
        }
    }
    let mut paired = HashMap::<(usize, usize), (usize, usize, Option<LoopID>)>::new();
    for (i, cell) in data.cells.iter().enumerate() {
        for j in 0..cell.vertices.len() {
            let a = cell.vertices[j];
            let b = cell.vertices[(j + 1) % cell.vertices.len()];
            if a == b {
                return Err(bad(SurfaceCurveError::CollapsedCellEdge));
            }
            let p = lerp(
                cell.points[j],
                cell.points[(j + 1) % cell.points.len()],
                0.5,
            );
            let wall = faces[&cell.face]
                .chords
                .iter()
                .find(|c| on(c.from, c.to, p))
                .map(|c| c.loop_id);
            let key = (a.min(b), a.max(b));
            if let Some((other, side, other_wall)) = paired.remove(&key) {
                if wall != other_wall {
                    return Err(bad(SurfaceCurveError::InconsistentBarrier));
                }
                data.links.push(Link {
                    cells: [other, i],
                    sides: [side, j],
                    wall,
                });
            } else {
                paired.insert(key, (i, j, wall));
            }
        }
    }
    if !paired.is_empty() {
        return Err(bad(SurfaceCurveError::NonconformingBoundary));
    }
    Ok(Arrangement {
        faces,
        crossings,
        loop_crossings,
        edge_positions: HashMap::new(),
        loop_positions: HashMap::new(),
        cell_base,
        cell_count: data.cells.len(),
        general: Some(data),
    })
}

impl Cells {
    pub fn components(&self) -> Result<Components, PropertyViolationError> {
        let mut parent: Vec<usize> = (0..self.cells.len()).collect();
        for link in &self.links {
            if link.wall.is_none() {
                join(&mut parent, link.cells[0], link.cells[1]);
            }
        }
        let mut ids = HashMap::new();
        let mut components = vec![];
        for i in 0..self.cells.len() {
            let r = root(&mut parent, i);
            let next = ids.len();
            components.push(*ids.entry(r).or_insert(next));
        }
        let mut vertices = HashMap::new();
        for (i, cell) in self.cells.iter().enumerate() {
            for &v in &cell.vertices {
                if let Some(v) = self.original[v] {
                    vertices.insert(v, components[i]);
                }
            }
        }
        Ok((components, ids.len(), vertices))
    }
    pub fn euler(
        &self,
        components: &[usize],
        count: usize,
    ) -> Result<Vec<i64>, PropertyViolationError> {
        let mut base = vec![];
        let mut parent = vec![];
        let mut owners = vec![];
        let mut chi = vec![0; count];
        for (i, cell) in self.cells.iter().enumerate() {
            base.push(parent.len());
            chi[components[i]] += 1 - cell.vertices.len() as i64;
            for _ in &cell.vertices {
                parent.push(parent.len());
                owners.push(components[i]);
            }
        }
        for link in &self.links {
            if link.wall.is_some() {
                continue;
            }
            let [a, b] = link.cells;
            let [u, v] = link.sides;
            join(
                &mut parent,
                base[a] + u,
                base[b] + (v + 1) % self.cells[b].vertices.len(),
            );
            join(
                &mut parent,
                base[a] + (u + 1) % self.cells[a].vertices.len(),
                base[b] + v,
            );
            chi[components[a]] += 1;
        }
        for (i, &owner) in owners.iter().enumerate() {
            if root(&mut parent, i) == i {
                chi[owner] += 1;
            }
        }
        Ok(chi)
    }
    pub fn refine(&self) -> Refinement {
        let mut out = Refinement {
            positions: self.positions.clone(),
            on_loop: self.on_loop.clone(),
            vertex_input: self.original.clone(),
            ..Default::default()
        };
        let mut midpoints = HashMap::new();
        // A traversable interval needs a non-loop vertex even when both endpoints lie on loops.
        for link in &self.links {
            let cell = &self.cells[link.cells[0]];
            let j = link.sides[0];
            let a = cell.vertices[j];
            let b = cell.vertices[(j + 1) % cell.vertices.len()];
            if !self.on_loop[a] || !self.on_loop[b] {
                continue;
            }
            let key = (a.min(b), a.max(b));
            let id = out.positions.len();
            out.positions
                .push((self.positions[a] + self.positions[b]) / 2.);
            out.on_loop.push(link.wall.is_some());
            out.vertex_input.push(None);
            midpoints.insert(key, id);
        }
        for (i, cell) in self.cells.iter().enumerate() {
            let mut perimeter = Vec::new();
            for j in 0..cell.vertices.len() {
                let a = cell.vertices[j];
                let b = cell.vertices[(j + 1) % cell.vertices.len()];
                perimeter.push(a);
                if let Some(&m) = midpoints.get(&(a.min(b), a.max(b))) {
                    perimeter.push(m);
                }
            }
            if perimeter.len() == 3 && !perimeter.iter().all(|&v| out.on_loop[v]) {
                out.faces.push(perimeter);
                out.face_cells.push(i);
                out.face_input.push(cell.face);
                continue;
            }
            let center = cell
                .vertices
                .iter()
                .map(|&v| self.positions[v])
                .sum::<Vector3D>()
                / cell.vertices.len() as f64;
            let c = out.positions.len();
            out.positions.push(center);
            out.on_loop.push(false);
            out.vertex_input.push(None);
            for j in 0..perimeter.len() {
                out.faces
                    .push(vec![perimeter[j], perimeter[(j + 1) % perimeter.len()], c]);
                out.face_cells.push(i);
                out.face_input.push(cell.face);
            }
        }
        out
    }
}
