//! Loops and paths drawn as bands on the surface: strips of a fixed width, which look the same from every side (unlike
//! lines). A loop gets a strip on each side, telling its positive side from its negative side (as in Polycuber).
//!
//! A loop's negative side is to its left (seen from outside the surface, along the loop): the loop regions to its left
//! get the label (axis, positive), as they lie below it (see `Dual::valid_exit`).

use crate::prelude::*;

/// A point of a polyline on the surface, with the surface normal there and the direction on the surface to its left
/// (for a loop: toward its negative side). The latter is a unit vector, lengthened at a bend (a miter, at most twice),
/// so that the sides of a band of constant width stay parallel to the polyline.
#[derive(Clone, Copy, Debug)]
pub struct LoopFrame {
    pub position: Vector3D,
    pub normal: Vector3D,
    pub negative: Vector3D,
}

/// A path of the layout as a band (see [`band_frames`]); `flat` if it lies between patches with the same normal.
#[derive(Clone, Debug)]
pub struct PathBand {
    pub frames: Vec<LoopFrame>,
    pub flat: bool,
}

impl Solution {
    /// The frames at the points where a loop crosses the mesh edges (see [`LoopFrame`]), in the order of the loop.
    #[must_use]
    pub fn loop_frames(&self, loop_id: LoopID) -> Vec<LoopFrame> {
        let mesh = self.mesh_ref.as_ref();
        let edges = &self.loops[loop_id].edges;
        let positions = self.get_coordinates_of_loop(loop_id);
        let n = positions.len().min(edges.len());
        if n < 2 {
            return vec![];
        }
        let faces_of = |e: EdgeID| [mesh.face(e), mesh.face(mesh.twin(e))];
        // The normal of every segment (from point i to point i + 1): that of the face it lies in.
        let segment_normals = (0..n)
            .map(|i| {
                let j = (i + 1) % n;
                faces_of(edges[i])
                    .into_iter()
                    .find(|f| faces_of(edges[j]).contains(f))
                    .map_or_else(Vector3D::zeros, |f| mesh.normal(f))
            })
            .collect::<Vec<_>>();
        let normals = (0..n).map(|i| mesh.normal(edges[i])).collect::<Vec<_>>();
        frames(&positions[..n], &normals, &segment_normals, true)
    }

    /// The frames of a loop for drawing it as a band (see [`band_frames`]).
    #[must_use]
    pub fn loop_band(&self, loop_id: LoopID, spacing: f64) -> Vec<LoopFrame> {
        let points = self
            .loop_frames(loop_id)
            .iter()
            .map(|f| (f.position, f.normal))
            .collect::<Vec<_>>();
        band_frames(&points, true, spacing)
    }

    /// The paths of the layout (each once) as bands (see [`band_frames`]), on its refined mesh.
    #[must_use]
    pub fn path_bands(&self, spacing: f64) -> Vec<PathBand> {
        match &self.layout {
            Some(layout) => self.path_bands_on(&layout.granulated_mesh, spacing),
            None => vec![],
        }
    }

    /// The paths of the layout as bands (see [`Solution::path_bands`]) on a mesh with the vertices of its refined mesh
    /// elsewhere (e.g., mapped onto the polycube).
    #[must_use]
    pub fn path_bands_on(&self, mesh: &Mesh<INPUT>, spacing: f64) -> Vec<PathBand> {
        let (Some(layout), Some(polycube)) = (&self.layout, &self.polycube) else {
            return vec![];
        };
        let structure = &polycube.structure;
        layout
            .edge_to_path
            .iter()
            .filter(|&(&edge, _)| edge.raw() < structure.twin(edge).raw())
            .map(|(&edge, path)| {
                let points = path
                    .iter()
                    .map(|&v| (mesh.position(v), mesh.normal(v)))
                    .collect::<Vec<_>>();
                PathBand {
                    frames: band_frames(&points, false, spacing),
                    flat: structure.normal(structure.face(edge))
                        == structure.normal(structure.face(structure.twin(edge))),
                }
            })
            .collect()
    }
}

/// The frames of a polyline on a surface (its points with their normals; `closed` if it is a loop) for drawing it as
/// a band, resampled such that consecutive points are at least `spacing` apart (the ends of an open polyline are kept).
/// Where a polyline has points close together (where a loop crosses the edges around a vertex, or a path zigzags along
/// the mesh edges), a band wider than its segments would fold over itself; on the resampled polyline it bends smoothly.
#[must_use]
pub fn band_frames(points: &[(Vector3D, Vector3D)], closed: bool, spacing: f64) -> Vec<LoopFrame> {
    if points.len() < 2 {
        return vec![];
    }
    let mut kept: Vec<(Vector3D, Vector3D)> = vec![];
    let last = points[points.len() - 1];
    for &point in points {
        if kept
            .last()
            .is_none_or(|previous| (point.0 - previous.0).norm() >= spacing)
        {
            kept.push(point);
        }
    }
    let minimum = if closed { 3 } else { 1 };
    if closed {
        // The loop closes: also the last point is at least `spacing` away from the first.
        while kept.len() > minimum && (kept[kept.len() - 1].0 - kept[0].0).norm() < spacing {
            kept.pop();
        }
    } else {
        // The end stays (in place of the last point kept, if that is close to it).
        while kept.len() > minimum && (kept[kept.len() - 1].0 - last.0).norm() < spacing {
            kept.pop();
        }
        kept.push(last);
    }
    if kept.len() < 2 || (closed && kept.len() < 3) {
        kept = points.to_vec();
    }
    let positions = kept.iter().map(|p| p.0).collect::<Vec<_>>();
    let normals = kept.iter().map(|p| p.1).collect::<Vec<_>>();
    let n = positions.len();
    let segment_normals = (0..n)
        .map(|i| normals[i] + normals[(i + 1) % n])
        .collect::<Vec<_>>();
    frames(&positions, &normals, &segment_normals, closed)
}

/// The corners of a regular polygon (`sides` of them) around a point, in the plane of the given normal: the round cap
/// of a band's end (where paths meet).
#[must_use]
pub fn disk(center: Vector3D, normal: Vector3D, radius: f64, sides: usize) -> Vec<Vector3D> {
    let normal = normal.try_normalize(1e-12).unwrap_or_else(Vector3D::z);
    let other = if normal.x.abs() < 0.9 {
        Vector3D::x()
    } else {
        Vector3D::y()
    };
    let u = normal.cross(&other).normalize();
    let v = normal.cross(&u);
    (0..sides)
        .map(|k| {
            let angle = std::f64::consts::TAU * k as f64 / sides as f64;
            center + (u * angle.cos() + v * angle.sin()) * radius
        })
        .collect()
}

// The frames of a polyline on a surface, given the normals at its points and of its segments (from point i to point
// i + 1; for an open polyline, the last is not used).
fn frames(
    positions: &[Vector3D],
    normals: &[Vector3D],
    segment_normals: &[Vector3D],
    closed: bool,
) -> Vec<LoopFrame> {
    let n = positions.len();
    let segments = if closed { n } else { n - 1 };
    // The left of every segment, in the plane of its normal.
    let lefts = (0..segments)
        .map(|i| {
            segment_normals[i]
                .cross(&(positions[(i + 1) % n] - positions[i]))
                .try_normalize(1e-12)
                .unwrap_or_else(Vector3D::zeros)
        })
        .collect::<Vec<_>>();
    (0..n)
        .map(|i| {
            let normal = normals[i];
            let before = if closed || i > 0 {
                lefts[(i + segments - 1) % segments]
            } else {
                Vector3D::zeros()
            };
            let after = if i < segments {
                lefts[i]
            } else {
                Vector3D::zeros()
            };
            let mean = before + after;
            let mean = (mean - normal * mean.dot(&normal)).try_normalize(1e-12);
            let negative = mean.map_or_else(Vector3D::zeros, |m| {
                // The miter: as far out as the sides of both segments (at most twice).
                let reference = if after.norm() > 0. { after } else { before };
                m / m.dot(&reference).max(0.5)
            });
            LoopFrame {
                position: positions[i],
                normal,
                negative,
            }
        })
        .collect()
}
