//! The quality of a labeled skeleton, to be maximized; labeling and optimization use its terms.
//!
//! score = (alignment + visibility) / 2 − BETA · #boundaries
//!
//! - Alignment: DualCube's alignment of dual loops, on the boundaries. A segment with direction
//!   d in a triangle with normal n runs along its label's axis a when d × n is parallel to a. The
//!   term is the length-weighted mean of that cosine over all boundaries; signed, so a boundary
//!   labelled against its orientation scores −1.
//! - Visibility: the share of the surface its own patch sees from the patch's centre, the
//!   centroid of the volume the patch bounds with its holes capped. A triangle is seen when the
//!   segment from the centre to its centroid crosses no other mesh triangle.

use crate::geometry::SurfaceGeometry;
use crate::label::Port;
use crate::{
    BoundaryCrossingID, LabeledSkeleton, Skeleton, SkeletonEdgeID, SkeletonNodeID, SkeletonResult,
};
use bvh::aabb::{Aabb, Bounded, IntersectsAabb};
use bvh::bounding_hierarchy::BHShape;
use bvh::bvh::Bvh;
use dualcube_types::prelude::*;
use slotmap::SecondaryMap;
use std::collections::HashMap;

/// Penalty per boundary (a loop pair of the polycube).
pub const BETA: f64 = 0.005;

pub fn score(alignment: f64, visibility: f64, boundaries: usize) -> f64 {
    (alignment + visibility) / 2. - BETA * boundaries as f64
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Quality {
    pub alignment: f64,
    pub visibility: f64,
    pub boundaries: usize,
    pub score: f64,
}

pub fn evaluate(mesh: &Mesh<INPUT>, labeled: &LabeledSkeleton) -> SkeletonResult<Quality> {
    let skeleton = &labeled.skeleton;
    let (mut aligned, mut length) = (0., 0.);
    for (edge, alignment) in alignments(mesh, skeleton) {
        aligned += alignment.length * alignment.with(Port::from_label(labeled.labels[edge]));
        length += alignment.length;
    }
    let alignment = if length > 0. { aligned / length } else { 1. };
    let visibility = Visibility::build(mesh, skeleton)?;
    let patches: Vec<Vec<SkeletonNodeID>> = skeleton.nodes.keys().map(|n| vec![n]).collect();
    let seen: f64 = visibility.visible_all(&patches).iter().sum();
    let visibility = seen / visibility.area;
    let boundaries = skeleton.edges.len();
    Ok(Quality {
        alignment,
        visibility,
        boundaries,
        score: score(alignment, visibility, boundaries),
    })
}

/// A boundary's alignment with every axis.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Alignment {
    pub length: f64,
    /// Length-weighted mean of the unit d × n: its component along an axis is the mean cosine.
    pub direction: Vector3D,
}

impl Alignment {
    /// Mean cosine with `port`'s axis, in [−1, 1].
    pub fn with(&self, port: Port) -> f64 {
        let a = self.direction[port.axis()];
        if port.is_positive() { a } else { -a }
    }

    pub fn reversed(self) -> Self {
        Self {
            direction: -self.direction,
            ..self
        }
    }

    /// Labeling costs: the length times the cosine lost against a perfect fit.
    pub fn costs(&self) -> [f64; 6] {
        Port::ALL.map(|p| self.length * (1. - self.with(p)))
    }
}

/// Of a closed polyline: segment `i` runs from `points[i].0` to the next point in face
/// `points[i].1`.
pub(crate) fn alignment(mesh: &Mesh<INPUT>, points: &[(Vector3D, FaceID)]) -> Alignment {
    let (mut length, mut sum) = (0., Vector3D::zeros());
    for (i, &(a, face)) in points.iter().enumerate() {
        let d = points[(i + 1) % points.len()].0 - a;
        let w = d.cross(&mesh.normal(face));
        if w.norm() > 0. {
            length += d.norm();
            sum += d.norm() * w.normalize();
        }
    }
    Alignment {
        length,
        direction: if length > 0. { sum / length } else { sum },
    }
}

/// [`alignment`] of a boundary, crossings at `positions`; positive along its edge's direction.
pub(crate) fn boundary_alignment(
    mesh: &Mesh<INPUT>,
    skeleton: &Skeleton,
    positions: &HashMap<BoundaryCrossingID, Vector3D>,
    edge: SkeletonEdgeID,
) -> Alignment {
    let points: Vec<_> = skeleton.edges[edge]
        .boundary
        .crossings
        .iter()
        .map(|&c| {
            let exit = skeleton.surface.crossings[c].exit;
            (positions[&c], mesh.face(mesh.twin(exit)))
        })
        .collect();
    alignment(mesh, &points)
}

pub fn alignments(
    mesh: &Mesh<INPUT>,
    skeleton: &Skeleton,
) -> SecondaryMap<SkeletonEdgeID, Alignment> {
    let positions = skeleton.surface.positions(mesh);
    skeleton
        .edges
        .keys()
        .map(|e| (e, boundary_alignment(mesh, skeleton, &positions, e)))
        .collect()
}

#[derive(Clone, Debug, Default)]
struct Patch {
    /// Triangle centroids and areas.
    samples: Vec<(Vector3D, f64)>,
    area: f64,
    /// Area-weighted sum of the sample positions.
    surface: Vector3D,
    /// Signed volume and its first moment, holes capped by fans.
    volume: f64,
    moment: Vector3D,
}

#[derive(Clone, Debug)]
struct Triangle {
    corners: [Vector3D; 3],
    node: usize,
}

impl Bounded<f64, 3> for Triangle {
    fn aabb(&self) -> Aabb<f64, 3> {
        let [a, b, c] = self.corners;
        let (lo, hi) = (a.inf(&b).inf(&c), a.sup(&b).sup(&c));
        Aabb::with_bounds(lo.into(), hi.into())
    }
}

impl BHShape<f64, 3> for Triangle {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node
    }
}

struct Segment {
    from: Vector3D,
    to: Vector3D,
}

impl IntersectsAabb<f64, 3> for Segment {
    fn intersects_aabb(&self, aabb: &Aabb<f64, 3>) -> bool {
        let d = self.to - self.from;
        let (mut lo, mut hi) = (0f64, 1f64);
        for k in 0..3 {
            if d[k].abs() < 1e-300 {
                if self.from[k] < aabb.min[k] || self.from[k] > aabb.max[k] {
                    return false;
                }
                continue;
            }
            let (a, b) = (
                (aabb.min[k] - self.from[k]) / d[k],
                (aabb.max[k] - self.from[k]) / d[k],
            );
            lo = lo.max(a.min(b));
            hi = hi.min(a.max(b));
            if lo > hi {
                return false;
            }
        }
        true
    }
}

impl Segment {
    /// Möller–Trumbore, strictly inside the segment (not at its end on the seen triangle).
    fn hits(&self, [a, b, c]: [Vector3D; 3]) -> bool {
        let d = self.to - self.from;
        let (e1, e2) = (b - a, c - a);
        let p = d.cross(&e2);
        let det = e1.dot(&p);
        if det.abs() < 1e-14 {
            return false;
        }
        let s = self.from - a;
        let u = s.dot(&p) / det;
        if !(0. ..=1.).contains(&u) {
            return false;
        }
        let q = s.cross(&e1);
        let v = d.dot(&q) / det;
        if v < 0. || u + v > 1. {
            return false;
        }
        let t = e2.dot(&q) / det;
        t > 1e-9 && t < 1. - 1e-6
    }
}

/// Visibility of patches, and of unions of them.
pub(crate) struct Visibility {
    bvh: Bvh<f64, 3>,
    triangles: Vec<Triangle>,
    patches: SecondaryMap<SkeletonNodeID, Patch>,
    /// Of the whole surface.
    pub area: f64,
}

impl Visibility {
    pub fn build(mesh: &Mesh<INPUT>, skeleton: &Skeleton) -> SkeletonResult<Self> {
        let geometry = SurfaceGeometry::build(mesh, skeleton)?;
        let mut triangles: Vec<Triangle> = mesh
            .face_ids()
            .into_iter()
            .map(|f| {
                let v: Vec<_> = mesh.vertices(f).map(|v| mesh.position(v)).collect();
                Triangle {
                    corners: [v[0], v[1], v[2]],
                    node: 0,
                }
            })
            .collect();
        let bvh = Bvh::build(&mut triangles);
        // Volumes relative to a point near the surface, for precision.
        let origin =
            triangles.iter().map(|t| t.corners[0]).sum::<Vector3D>() / triangles.len() as f64;
        let mut patches: SecondaryMap<_, Patch> = skeleton
            .nodes
            .keys()
            .map(|n| (n, Patch::default()))
            .collect();
        let tet = |patch: &mut Patch, [a, b, c]: [Vector3D; 3], sign: f64| {
            let (a, b, c) = (a - origin, b - origin, c - origin);
            let volume = sign * a.dot(&b.cross(&c)) / 6.;
            patch.volume += volume;
            patch.moment += volume * ((a + b + c) / 4. + origin);
        };
        for t in &geometry.triangles {
            let [a, b, c] = t.points;
            let area = 0.5 * (b - a).cross(&(c - a)).norm();
            let patch = &mut patches[t.node];
            let centroid = (a + b + c) / 3.;
            patch.samples.push((centroid, area));
            patch.area += area;
            patch.surface += area * centroid;
            tet(patch, t.points, 1.);
        }
        // The curve runs counterclockwise around its edge's first patch, seen from outside.
        for curve in &geometry.boundaries {
            let points = &curve.points[..curve.points.len() - 1];
            let centre = points.iter().sum::<Vector3D>() / points.len() as f64;
            let [first, second] = skeleton.edges[curve.edge].nodes;
            for i in 0..points.len() {
                let fan = [centre, points[i], points[(i + 1) % points.len()]];
                tet(&mut patches[first], fan, -1.);
                tet(&mut patches[second], fan, 1.);
            }
        }
        let area = patches.values().map(|p| p.area).sum();
        Ok(Self {
            bvh,
            triangles,
            patches,
            area,
        })
    }

    /// Signed volume of each patch with its holes capped.
    #[cfg(test)]
    pub fn volumes(&self) -> SecondaryMap<SkeletonNodeID, f64> {
        self.patches.iter().map(|(n, p)| (n, p.volume)).collect()
    }

    fn seen(&self, from: Vector3D, to: Vector3D) -> bool {
        let segment = Segment { from, to };
        !self
            .bvh
            .traverse_iterator(&segment, &self.triangles)
            .any(|t| segment.hits(t.corners))
    }

    /// Area of the union of `nodes` seen from its centre.
    pub fn visible(&self, nodes: &[SkeletonNodeID]) -> f64 {
        let patches: Vec<&Patch> = nodes.iter().map(|&n| &self.patches[n]).collect();
        let area: f64 = patches.iter().map(|p| p.area).sum();
        let volume: f64 = patches.iter().map(|p| p.volume).sum();
        // A patch enclosing (almost) nothing, like a flat disk, is seen from its surface centroid.
        let centre = if volume > 1e-6 * area.powf(1.5) {
            patches.iter().map(|p| p.moment).sum::<Vector3D>() / volume
        } else if area > 0. {
            patches.iter().map(|p| p.surface).sum::<Vector3D>() / area
        } else {
            return 0.;
        };
        patches
            .iter()
            .flat_map(|p| &p.samples)
            .filter(|&&(point, _)| self.seen(centre, point))
            .map(|&(_, area)| area)
            .sum()
    }

    /// [`Self::visible`] of every union, in parallel.
    pub fn visible_all(&self, unions: &[Vec<SkeletonNodeID>]) -> Vec<f64> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let next = AtomicUsize::new(0);
        let mut result = vec![0.; unions.len()];
        // Unions differ a lot in size: threads take the next one when done.
        let parts: Vec<Vec<(usize, f64)>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads.min(unions.len()))
                .map(|_| {
                    scope.spawn(|| {
                        let mut part = Vec::new();
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            let Some(union) = unions.get(i) else {
                                return part;
                            };
                            part.push((i, self.visible(union)));
                        }
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("visibility thread"))
                .collect()
        });
        for (i, value) in parts.into_iter().flatten() {
            result[i] = value;
        }
        result
    }
}

#[cfg(test)]
mod tests;
