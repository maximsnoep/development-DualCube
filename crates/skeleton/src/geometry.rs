//! A shared geometric realization of the combinatorial surface regions.
//!
//! Ordinary arcs are straight. Triangles with returning arcs use a convex reference
//! polygon with extra perimeter samples, giving vertex-free digons positive area.
//! A radial piecewise-affine map preserves the disjoint reference chords.

use crate::surface::{Regions, Side, Vertex, canonical};
use crate::{BoundaryCrossingID, Skeleton, SkeletonEdgeID, SkeletonNodeID, SkeletonResult};
use dualcube_types::prelude::*;
use nalgebra::Vector2;
use slotmap::SecondaryMap;
use std::collections::HashMap;
use std::f64::consts::TAU;

type Point = Vector2<f64>;

#[derive(Clone, Debug)]
pub struct PatchTriangle {
    pub node: SkeletonNodeID,
    pub face: FaceID,
    /// Winding matches the input triangle.
    pub points: [Vector3D; 3],
}

#[derive(Clone, Debug)]
pub struct BoundaryCurve {
    pub edge: SkeletonEdgeID,
    /// Oriented closed polyline, with the first point repeated at the end.
    pub points: Vec<Vector3D>,
    pub arcs: Vec<dualcube_dual::SurfaceArc>,
}

#[derive(Clone, Debug, Default)]
pub struct SurfaceGeometry {
    pub triangles: Vec<PatchTriangle>,
    pub boundaries: Vec<BoundaryCurve>,
}

impl SurfaceGeometry {
    pub fn build(mesh: &Mesh<INPUT>, skeleton: &Skeleton) -> SkeletonResult<Self> {
        let regions = Regions::build(mesh, skeleton)?;
        let mut by_face = HashMap::<_, Vec<usize>>::new();
        for (i, cell) in regions.cells.iter().enumerate() {
            by_face.entry(cell.face).or_default().push(i);
        }
        let mut result = Self::default();
        let mut arcs = HashMap::<BoundaryCrossingID, Vec<Vector3D>>::new();
        for face in mesh.face_ids() {
            let cells = &by_face[&face];
            if cells.len() == 1 && regions.cells[cells[0]].vertices.len() == 3 {
                let points: Vec<_> = mesh.vertices(face).map(|v| mesh.position(v)).collect();
                result.triangles.push(PatchTriangle {
                    node: regions.owners[cells[0]],
                    face,
                    points: [points[0], points[1], points[2]],
                });
                continue;
            }
            let returning = cells.iter().any(|&i| {
                let cell = &regions.cells[i];
                cell.sides.iter().enumerate().any(|(j, side)| {
                    if !matches!(side, Side::Arc { .. }) {
                        return false;
                    }
                    let Vertex::Crossing(a) = cell.vertices[j] else {
                        unreachable!()
                    };
                    let Vertex::Crossing(b) = cell.vertices[(j + 1) % cell.vertices.len()] else {
                        unreachable!()
                    };
                    canonical(mesh, skeleton.surface.crossings[a].exit)
                        == canonical(mesh, skeleton.surface.crossings[b].exit)
                })
            });
            if !returning {
                for &i in cells {
                    let cell = &regions.cells[i];
                    let polygon: Vec<_> = cell
                        .vertices
                        .iter()
                        .map(|v| match *v {
                            Vertex::Input(id) => mesh.position(id),
                            Vertex::Crossing(id) => crate::surface::position(mesh, skeleton, id),
                        })
                        .collect();
                    for (j, side) in cell.sides.iter().enumerate() {
                        if let Side::Arc {
                            start,
                            forward: true,
                        } = *side
                        {
                            arcs.insert(start, vec![polygon[j], polygon[(j + 1) % polygon.len()]]);
                        }
                    }
                    for j in 1..polygon.len() - 1 {
                        result.triangles.push(PatchTriangle {
                            node: regions.owners[i],
                            face,
                            points: [polygon[0], polygon[j], polygon[j + 1]],
                        });
                    }
                }
                continue;
            }
            let chart = Chart::new(mesh, skeleton, face);
            for &i in cells {
                let cell = &regions.cells[i];
                let mut polygon = Vec::new();
                for (j, vertex) in cell.vertices.iter().enumerate() {
                    let index = chart.indices[vertex];
                    polygon.push(chart.reference[index]);
                    if matches!(cell.sides[j], Side::Gap(_)) {
                        polygon.push(chart.reference[(index + 1) % chart.reference.len()]);
                    }
                    if let Side::Arc {
                        start,
                        forward: true,
                    } = cell.sides[j]
                    {
                        let end = cell.vertices[(j + 1) % cell.vertices.len()];
                        arcs.insert(start, chart.arc(index, chart.indices[&end]));
                    }
                }
                for sector in 0..chart.reference.len() {
                    let a = chart.reference[sector];
                    let b = chart.reference[(sector + 1) % chart.reference.len()];
                    let mut clipped = polygon.clone();
                    for (u, v) in [(Point::zeros(), a), (a, b), (b, Point::zeros())] {
                        clipped = clip(&clipped, u, v);
                    }
                    for j in 1..clipped.len().saturating_sub(1) {
                        let p = [clipped[0], clipped[j], clipped[j + 1]];
                        if cross(p[1] - p[0], p[2] - p[0]) <= 1e-15 {
                            continue;
                        }
                        let points = p.map(|p| chart.map(sector, p));
                        if (points[1] - points[0])
                            .cross(&(points[2] - points[0]))
                            .norm()
                            == 0.
                        {
                            continue;
                        }
                        result.triangles.push(PatchTriangle {
                            node: regions.owners[i],
                            face,
                            points,
                        });
                    }
                }
            }
        }
        for (id, edge) in &skeleton.edges {
            let mut points = Vec::new();
            let mut surface = Vec::new();
            for start in &edge.boundary.crossings {
                let arc = &arcs[start];
                points.extend_from_slice(&arc[..arc.len() - 1]);
                let face = mesh.face(mesh.twin(skeleton.surface.crossings[*start].exit));
                let corners: Vec<_> = mesh.vertices(face).map(|v| mesh.position(v)).collect();
                let u = corners[1] - corners[0];
                let v = corners[2] - corners[0];
                let uu = u.dot(&u);
                let uv = u.dot(&v);
                let vv = v.dot(&v);
                let det = uu * vv - uv * uv;
                let coordinates = |p: Vector3D| {
                    let d = p - corners[0];
                    let a = d.dot(&u);
                    let b = d.dot(&v);
                    [(a * vv - b * uv) / det, (b * uu - a * uv) / det]
                };
                for pair in arc.windows(2) {
                    surface.push(dualcube_dual::SurfaceArc {
                        face,
                        from: coordinates(pair[0]),
                        to: coordinates(pair[1]),
                    });
                }
            }
            points.push(points[0]);
            result.boundaries.push(BoundaryCurve {
                edge: id,
                points,
                arcs: surface,
            });
        }
        // Also detects regions too small for this floating-point realization.
        if result.centroids()?.len() != skeleton.nodes.len() {
            return Err("Surface geometry omitted a region".into());
        }
        Ok(result)
    }

    /// Area-weighted surface centroids, not projected back onto the surface.
    pub fn centroids(&self) -> SkeletonResult<SecondaryMap<SkeletonNodeID, Vector3D>> {
        let mut sums = HashMap::new();
        for triangle in &self.triangles {
            let [a, b, c] = triangle.points;
            let weight = (b - a).cross(&(c - a)).norm();
            let (area, moment) = sums.entry(triangle.node).or_insert((0., Vector3D::zeros()));
            *area += weight;
            *moment += (a + (b - a) / 3. + (c - a) / 3.) * weight;
        }
        let mut centroids = SecondaryMap::new();
        for (node, (area, moment)) in sums {
            let center = moment / area;
            if !area.is_finite() || area <= 0. || !center.iter().all(|v| v.is_finite()) {
                return Err("Surface region has no finite positive area".into());
            }
            centroids.insert(node, center);
        }
        Ok(centroids)
    }
}

struct Chart {
    reference: Vec<Point>,
    positions: Vec<Vector3D>,
    indices: HashMap<Vertex, usize>,
    center: Vector3D,
}

impl Chart {
    fn new(mesh: &Mesh<INPUT>, skeleton: &Skeleton, face: FaceID) -> Self {
        let mut vertices = Vec::new();
        let mut positions = Vec::new();
        for edge in mesh.edges(face) {
            vertices.push(Vertex::Input(mesh.root(edge)));
            positions.push(mesh.position(mesh.root(edge)));
            let key = canonical(mesh, edge);
            if let Some(order) = skeleton.surface.edge_order.get(&key) {
                let mut crossings = order.clone();
                if key != edge {
                    crossings.reverse();
                }
                for id in crossings {
                    vertices.push(Vertex::Crossing(id));
                    positions.push(crate::surface::position(mesh, skeleton, id));
                }
            }
        }
        let mut boundary = Vec::new();
        let mut indices = HashMap::new();
        for (i, vertex) in vertices.into_iter().enumerate() {
            indices.insert(vertex, boundary.len());
            boundary.push(positions[i]);
            boundary.push((positions[i] + positions[(i + 1) % positions.len()]) * 0.5);
        }
        // TODO: Optimize this parameterization for surface shape and label quality.
        let reference = (0..boundary.len())
            .map(|i| {
                let theta = TAU * i as f64 / boundary.len() as f64;
                Point::new(theta.cos(), theta.sin())
            })
            .collect();
        let center = mesh
            .vertices(face)
            .map(|v| mesh.position(v))
            .sum::<Vector3D>()
            / 3.;
        Self {
            reference,
            positions: boundary,
            indices,
            center,
        }
    }

    fn map(&self, sector: usize, p: Point) -> Vector3D {
        let next = (sector + 1) % self.reference.len();
        let a = self.reference[sector];
        let b = self.reference[next];
        let determinant = cross(a, b);
        self.center
            + (self.positions[sector] - self.center) * (cross(p, b) / determinant)
            + (self.positions[next] - self.center) * (cross(a, p) / determinant)
    }

    fn arc(&self, from: usize, to: usize) -> Vec<Vector3D> {
        let a = self.reference[from];
        let b = self.reference[to];
        let delta = b - a;
        let mut parameters = vec![0., 1.];
        for &ray in &self.reference {
            let denominator = cross(delta, ray);
            if denominator.abs() > 1e-15 {
                let t = -cross(a, ray) / denominator;
                let r = cross(a, delta) / cross(ray, delta);
                if t > 0. && t < 1. && r >= 0. && r <= 1. {
                    parameters.push(t);
                }
            }
        }
        parameters.sort_by(f64::total_cmp);
        parameters.dedup_by(|a, b| (*a - *b).abs() < 1e-14);
        parameters
            .into_iter()
            .map(|t| {
                if t == 0. {
                    return self.positions[from];
                }
                if t == 1. {
                    return self.positions[to];
                }
                let p = a + delta * t;
                let angle = p.y.atan2(p.x).rem_euclid(TAU);
                let sector = ((angle / TAU * self.reference.len() as f64) as usize)
                    .min(self.reference.len() - 1);
                self.map(sector, p)
            })
            .collect()
    }
}

fn cross(a: Point, b: Point) -> f64 {
    a.x * b.y - a.y * b.x
}

/// Clip a convex polygon to the left of an oriented line.
fn clip(polygon: &[Point], a: Point, b: Point) -> Vec<Point> {
    let mut result = Vec::new();
    if polygon.is_empty() {
        return result;
    }
    for i in 0..polygon.len() {
        let p = polygon[i];
        let q = polygon[(i + 1) % polygon.len()];
        let dp = cross(b - a, p - a);
        let dq = cross(b - a, q - a);
        if dp >= 0. {
            result.push(p);
        }
        if (dp > 0. && dq < 0.) || (dp < 0. && dq > 0.) {
            result.push(p + (q - p) * (dp / (dp - dq)));
        }
    }
    result
}

#[cfg(test)]
mod tests;
