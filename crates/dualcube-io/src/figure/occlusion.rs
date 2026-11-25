//! Lines on a surface, occluded as the GPU does it: a line shows where the surface under it on the page does not lie
//! in front of it (by more than a tolerance), and nowhere off the surface. Whether a face lies in front of a line is
//! decided at the line, with the face's plane, so that the faces of the line's own surface never hide it.

use super::{Camera, DepthBuffer};
use dualcube::prelude::*;

// Cells per side of the grid that finds the faces near a segment on the page.
const GRID: usize = 128;

/// How a segment of a line shows.
pub(super) enum Cover {
    /// Not at all.
    Hidden,
    /// Wholly: only over faces that do not hide it, away from the outline.
    Plain,
    /// Through a mask: the faces under it on the page (back to front, see [`Occluders::polygon`]), and whether each
    /// lets it show.
    Masked(Vec<(usize, bool)>),
}

/// The faces of a surface that face the camera, back to front (as drawn), for occluding lines.
pub(super) struct Occluders<'a> {
    corners: Vec<&'a [Vector3D]>,
    normals: Vec<Vector3D>,
    polygons: Vec<Vec<[f64; 2]>>,
    min: [f64; 2],
    cell: f64,
    grid: Vec<Vec<usize>>,
    // Per face, the last query that saw it (to see each face once per query).
    seen: Vec<usize>,
    query: usize,
}

impl<'a> Occluders<'a> {
    pub(super) fn new(
        faces: impl Iterator<Item = (&'a [Vector3D], Vector3D)>,
        camera: &Camera,
    ) -> Self {
        let (corners, normals): (Vec<_>, Vec<_>) = faces.unzip();
        let polygons: Vec<Vec<[f64; 2]>> = corners
            .iter()
            .map(|c| c.iter().map(|&p| camera.project(p)).collect())
            .collect();
        let (mut min, mut max) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for p in polygons.iter().flatten() {
            for k in 0..2 {
                min[k] = min[k].min(p[k]);
                max[k] = max[k].max(p[k]);
            }
        }
        let cell = ((max[0] - min[0]).max(max[1] - min[1]) / GRID as f64).max(1e-300);
        let mut occluders = Self {
            corners,
            normals,
            min,
            cell,
            grid: vec![vec![]; GRID * GRID],
            seen: vec![0; polygons.len()],
            query: 0,
            polygons,
        };
        for (i, polygon) in occluders.polygons.iter().enumerate() {
            let (lo, hi) = bounds(polygon);
            for y in occluders.range(lo[1], hi[1], 1) {
                for x in occluders.range(lo[0], hi[0], 0) {
                    occluders.grid[y * GRID + x].push(i);
                }
            }
        }
        occluders
    }

    // The cells that cover these coordinates on axis `k`.
    fn range(&self, lo: f64, hi: f64, k: usize) -> std::ops::RangeInclusive<usize> {
        let cell =
            |v: f64| (((v - self.min[k]) / self.cell).floor().max(0.) as usize).min(GRID - 1);
        cell(lo)..=cell(hi)
    }

    /// How the segment from `a` to `b` (on the surface) shows when drawn this wide (as a radius on the page), with
    /// faces hiding it when in front of it by more than `tolerance`.
    pub(super) fn cover(
        &mut self,
        camera: &Camera,
        buffer: &DepthBuffer,
        [a, b]: [Vector3D; 2],
        radius: f64,
        tolerance: f64,
    ) -> Cover {
        if !camera.in_front(a) || !camera.in_front(b) {
            return Cover::Hidden;
        }
        let (pa, pb) = (camera.project(a), camera.project(b));
        let (lo, hi) = bounds(&[pa, pb]);
        self.query += 1;
        let mut near = vec![];
        for y in self.range(lo[1] - radius, hi[1] + radius, 1) {
            for x in self.range(lo[0] - radius, hi[0] + radius, 0) {
                for &i in &self.grid[y * GRID + x] {
                    if self.seen[i] != self.query {
                        self.seen[i] = self.query;
                        if distance(&self.polygons[i], pa, pb) <= radius {
                            near.push(i);
                        }
                    }
                }
            }
        }
        near.sort_unstable();
        let shows: Vec<bool> = near
            .iter()
            .map(|&i| !self.hides(camera, i, [a, b], [pa, pb], tolerance))
            .collect();
        if !shows.contains(&true) {
            return Cover::Hidden;
        }
        if !shows.contains(&false) && !off_surface(buffer, pa, pb, radius) {
            return Cover::Plain;
        }
        Cover::Masked(near.into_iter().zip(shows).collect())
    }

    /// Face `i` on the page.
    pub(super) fn polygon(&self, i: usize) -> &[[f64; 2]] {
        &self.polygons[i]
    }

    // Whether face `i` lies in front of the segment, at the segment's point nearest to the face's center on the page.
    fn hides(
        &self,
        camera: &Camera,
        i: usize,
        [a, b]: [Vector3D; 2],
        [pa, pb]: [[f64; 2]; 2],
        tolerance: f64,
    ) -> bool {
        let polygon = &self.polygons[i];
        let center = polygon
            .iter()
            .fold([0.; 2], |c, p| [c[0] + p[0], c[1] + p[1]])
            .map(|v| v / polygon.len() as f64);
        let t = parameter(center, pa, pb);
        let q = a + (b - a) * t;
        camera
            .depth_on_plane(camera.project(q), self.corners[i][0], self.normals[i])
            .is_some_and(|depth| depth < camera.depth(q) - tolerance)
    }
}

// Whether the page around the segment (within the radius) has no surface somewhere (seen in the depth buffer).
fn off_surface(buffer: &DepthBuffer, pa: [f64; 2], pb: [f64; 2], radius: f64) -> bool {
    let length = ((pb[0] - pa[0]).powi(2) + (pb[1] - pa[1]).powi(2)).sqrt();
    let samples = ((length / radius.max(buffer.cell)).ceil() as usize).clamp(1, 16);
    (0..=samples).any(|k| {
        let t = k as f64 / samples as f64;
        let p = [pa[0] + (pb[0] - pa[0]) * t, pa[1] + (pb[1] - pa[1]) * t];
        buffer.void_near(p, radius / buffer.cell)
    })
}

fn bounds(points: &[[f64; 2]]) -> ([f64; 2], [f64; 2]) {
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for p in points {
        for k in 0..2 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    (lo, hi)
}

// The parameter of the point of the segment nearest to `p`.
fn parameter(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let d = [b[0] - a[0], b[1] - a[1]];
    let length = d[0] * d[0] + d[1] * d[1];
    if length < 1e-300 {
        return 0.;
    }
    (((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]) / length).clamp(0., 1.)
}

fn point_distance(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let t = parameter(p, a, b);
    let q = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
    ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt()
}

fn cross(o: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

fn segment_distance(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> f64 {
    let crosses = cross(a, b, c).signum() * cross(a, b, d).signum() < 0.
        && cross(c, d, a).signum() * cross(c, d, b).signum() < 0.;
    if crosses {
        return 0.;
    }
    point_distance(a, c, d)
        .min(point_distance(b, c, d))
        .min(point_distance(c, a, b))
        .min(point_distance(d, a, b))
}

// The distance between a polygon and a segment on the page (zero if they overlap).
fn distance(polygon: &[[f64; 2]], a: [f64; 2], b: [f64; 2]) -> f64 {
    let n = polygon.len();
    let signs: Vec<f64> = (0..n)
        .map(|i| cross(polygon[i], polygon[(i + 1) % n], a))
        .collect();
    if signs.iter().all(|&s| s >= 0.) || signs.iter().all(|&s| s <= 0.) {
        return 0.;
    }
    (0..n)
        .map(|i| segment_distance(a, b, polygon[i], polygon[(i + 1) % n]))
        .fold(f64::INFINITY, f64::min)
}
