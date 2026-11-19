//! A new loop as an explicit curve. Its crossings are placed as legacy loops' were (pulled taut
//! inside their gaps). Inside a triangle it is the straight chord between them where that crosses
//! every visit of the triangle as routed (always, among straight chords); otherwise it bends
//! through the convex cells of the arrangement, crossing a loop only where the route does.

use super::*;
use crate::SurfaceArc;
use crate::arrangement::{Point2, general, lerp, orient, point_on_side, sides};

// A visit of an existing loop to a triangle, as a polyline in the triangle's frame.
struct Visit {
    loop_id: LoopID,
    points: Vec<Point2>,
}

// The position of a point on the sides along the boundary (side + parameter, cyclic in [0, 3)).
fn along_boundary(p: Point2) -> Option<f64> {
    general::side_at(p).map(|(side, t)| side as f64 + t)
}

// Whether the chord from `a` to `b` separates `p` from `q` (positions along the boundary).
fn separates(a: f64, b: f64, p: f64, q: f64) -> bool {
    let between = |x: f64| {
        if p < q {
            p < x && x < q
        } else {
            x > p || x < q
        }
    };
    between(a) != between(b)
}

// The number of crossings of the segment from `p` to `q` with a polyline; `None` if they touch.
fn crossings(p: Point2, q: Point2, polyline: &[Point2]) -> Option<usize> {
    const TOL: f64 = 1e-12;
    let sign = |x: f64| {
        if x > TOL {
            1
        } else if x < -TOL {
            -1
        } else {
            0
        }
    };
    let mut count = 0;
    for w in polyline.windows(2) {
        let (a, b) = (w[0], w[1]);
        let (sa, sb) = (sign(orient(p, q, a)), sign(orient(p, q, b)));
        let (sp, sq) = (sign(orient(a, b, p)), sign(orient(a, b, q)));
        if sa * sb < 0 && sp * sq < 0 {
            count += 1;
        } else if (sa * sb == 0 && sp * sq <= 0) || (sp * sq == 0 && sa * sb <= 0) {
            return None;
        }
    }
    Some(count)
}

// The points of a path from `p` to `q` (on the sides of `face`) through the convex cells of
// `face` that crosses `walls` loops (`None` if the fewest possible differ), never a blocked one.
// It crosses every cell edge through its middle, so it stays inside the cells in between.
fn bend(
    cells: &general::Cells,
    face: FaceID,
    p: Point2,
    q: Point2,
    walls: usize,
    blocked: impl Fn(LoopID) -> bool,
) -> Option<Vec<Point2>> {
    let inside = (0..cells.cells.len())
        .filter(|&c| cells.cells[c].face == face)
        .collect_vec();
    let centroid = |c: usize| {
        let points = &cells.cells[c].points;
        let n = points.len() as f64;
        let sum = points
            .iter()
            .fold([0., 0.], |s, p| [s[0] + p[0], s[1] + p[1]]);
        [sum[0] / n, sum[1] / n]
    };
    // The cell of a point on the sides: the one holding a point just inside the triangle.
    let cell_at = |x: Point2| {
        let y = lerp(x, [1. / 3., 1. / 3.], 1e-7);
        inside
            .iter()
            .copied()
            .find(|&c| general::contains(&cells.cells[c].points, y))
    };
    let (from, to) = (cell_at(p)?, cell_at(q)?);
    let mut links: HashMap<usize, Vec<usize>> = HashMap::new();
    for (l, link) in cells.links.iter().enumerate() {
        let [a, b] = link.cells;
        if cells.cells[a].face == face && cells.cells[b].face == face {
            links.entry(a).or_default().push(l);
            links.entry(b).or_default().push(l);
        }
    }
    // Fewest walls first, then the shortest path between cell centroids.
    let distance = |a: Point2, b: Point2| (a[0] - b[0]).hypot(a[1] - b[1]);
    let mut best: HashMap<usize, (usize, OrderedFloat<f64>)> =
        HashMap::from([(from, (0, OrderedFloat(0.)))]);
    let mut previous: HashMap<usize, usize> = HashMap::new();
    let mut heap = BinaryHeap::from([std::cmp::Reverse(((0, OrderedFloat(0.)), from))]);
    while let Some(std::cmp::Reverse((cost, cell))) = heap.pop() {
        if cost > best[&cell] {
            continue;
        }
        if cell == to {
            break;
        }
        for &l in links.get(&cell).map_or(&[][..], Vec::as_slice) {
            let link = &cells.links[l];
            if link.wall.is_some_and(&blocked) {
                continue;
            }
            let next = if link.cells[0] == cell {
                link.cells[1]
            } else {
                link.cells[0]
            };
            let cost = (
                cost.0 + usize::from(link.wall.is_some()),
                OrderedFloat(*cost.1 + distance(centroid(cell), centroid(next))),
            );
            if best.get(&next).is_none_or(|&c| cost < c) {
                best.insert(next, cost);
                previous.insert(next, l);
                heap.push(std::cmp::Reverse((cost, next)));
            }
        }
    }
    if best.get(&to)?.0 != walls {
        return None;
    }
    let mut path = vec![];
    let mut cell = to;
    while cell != from {
        let link = &cells.links[previous[&cell]];
        let s = usize::from(link.cells[1] == cell);
        path.push((link, 1 - s));
        cell = link.cells[1 - s];
    }
    path.reverse();
    let mut points = vec![p];
    for (link, s) in path {
        // Just before and just after the middle of the shared edge, in the two cells.
        let (a, b) = (link.cells[s], link.cells[1 - s]);
        let corners = &cells.cells[a].points;
        let j = link.sides[s];
        let middle = lerp(corners[j], corners[(j + 1) % corners.len()], 0.5);
        let (before, after) = [0.25, 0.1, 0.03, 0.01, 0.003, 0.001]
            .into_iter()
            .map(|lambda| {
                let x = lerp(middle, centroid(a), lambda);
                (x, [2. * middle[0] - x[0], 2. * middle[1] - x[1]])
            })
            .find(|&(x, y)| {
                general::contains(corners, x) && general::contains(&cells.cells[b].points, y)
            })?;
        points.extend([before, after]);
    }
    points.push(q);
    Some(points)
}

impl LoopSampler<'_> {
    // Every visit of an existing loop to the given triangles.
    fn visits_in(&self, faces: &HashSet<FaceID>) -> HashMap<FaceID, Vec<Visit>> {
        let mut result: HashMap<FaceID, Vec<Visit>> = HashMap::new();
        for (loop_id, l) in self.loops {
            let Ok(arcs) = l.surface_arcs(self.mesh_ref) else {
                continue;
            };
            let n = arcs.len();
            let Some(start) = (0..n).find(|&i| arcs[(i + n - 1) % n].face != arcs[i].face) else {
                continue;
            };
            let mut i = 0;
            while i < n {
                let face = arcs[(start + i) % n].face;
                let mut points = vec![arcs[(start + i) % n].from];
                while i < n && arcs[(start + i) % n].face == face {
                    points.push(arcs[(start + i) % n].to);
                    i += 1;
                }
                if faces.contains(&face) {
                    result
                        .entry(face)
                        .or_default()
                        .push(Visit { loop_id, points });
                }
            }
        }
        result
    }

    /// The loop of `direction` along `edges` through `gaps` (see `construct_valid_loop`) as an
    /// explicit curve that crosses exactly the loops it was routed across: its crossings are
    /// placed as by `place_loop_in_gaps`, and inside a triangle it is straight unless that would
    /// cross the triangle's curves otherwise (see the module documentation). `None` if it
    /// cannot be drawn.
    pub fn curve_in_gaps(
        &self,
        edges: &[EdgeID],
        direction: Direction,
        gaps: Vec<(usize, usize)>,
    ) -> Option<Loop> {
        let mesh = self.mesh_ref;
        let (gaps, offsets) = self.place_loop_in_gaps(edges, direction, gaps);
        let m = gaps.len();
        if m < 2 {
            return None;
        }
        // Exit `k` leaves a triangle through `exits[k].0`; visit `k` runs in the next triangle.
        let exits = gaps
            .iter()
            .map(|&(i, _)| (edges[i], offsets[i]))
            .collect_vec();
        let faces = exits
            .iter()
            .map(|&(e, _)| mesh.face(mesh.twin(e)))
            .collect_vec();
        let visits = self.visits_in(&faces.iter().copied().collect());
        let own = |id: LoopID| self.loops.get(id).is_some_and(|l| l.direction == direction);
        let mut cells: Option<Option<general::Cells>> = None;
        let mut arcs = vec![];
        for k in 0..m {
            let face = faces[k];
            let (entry, t) = (mesh.twin(exits[k].0), 1. - exits[k].1);
            let (exit, s) = exits[(k + 1) % m];
            if mesh.face(exit) != face {
                return None;
            }
            let sides = sides(mesh, face).ok()?;
            let at = |e: EdgeID, t: f64| -> Option<Point2> {
                Some(point_on_side(sides.iter().position(|&x| x == e)?, t))
            };
            let (p, q) = (at(entry, t)?, at(exit, s)?);
            let (bp, bq) = (along_boundary(p)?, along_boundary(q)?);
            let here = visits.get(&face).map_or(&[][..], Vec::as_slice);
            // The route crosses the visits that separate its ends, never one of its own axis.
            let mut crossed = vec![];
            for visit in here {
                let a = along_boundary(visit.points[0])?;
                let b = along_boundary(*visit.points.last()?)?;
                crossed.push(separates(a, b, bp, bq));
            }
            if here.iter().zip(&crossed).any(|(v, &c)| c && own(v.loop_id)) {
                return None;
            }
            let straight = here
                .iter()
                .zip(&crossed)
                .all(|(v, &c)| crossings(p, q, &v.points) == Some(usize::from(c)));
            if straight {
                arcs.push(SurfaceArc {
                    face,
                    from: p,
                    to: q,
                });
                continue;
            }
            let cells = cells
                .get_or_insert_with(|| general::build(mesh, self.loops).ok()?.general)
                .as_ref()?;
            let walls = crossed.iter().filter(|&&c| c).count();
            let points = bend(cells, face, p, q, walls, own)?;
            arcs.extend(points.windows(2).map(|w| SurfaceArc {
                face,
                from: w[0],
                to: w[1],
            }));
        }
        Some(Loop::from_surface(direction, arcs))
    }
}

#[cfg(test)]
mod tests;
