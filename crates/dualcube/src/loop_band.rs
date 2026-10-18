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

    /// This solution with smooth loops, as post-processing (for the solutions that are shown or returned; the layout, and
    /// so the quality, stay the same): with a layout, the loops traced smoothly through its patches (see
    /// `Solution::smooth_loops_in_patches`), and then (with or without) smoothed within their triangles (see
    /// `smooth_loops`).
    #[must_use]
    pub fn with_smooth_loops(&self) -> Self {
        let mut smooth = self.smooth_loops_in_patches().unwrap_or_else(|| self.clone());
        smooth.smooth_loops();
        smooth
    }

    /// Smooth the loops slightly within their triangles (see [`smoothed_offsets`]): only where they cross the mesh
    /// edges changes, so the loop structure, the dual structure and the layout stay the same.
    pub fn smooth_loops(&mut self) {
        let offsets = smoothed_offsets(&self.mesh_ref, &self.loops);
        for (loop_id, lewp) in &mut self.loops {
            if let Some(smoothed) = offsets.get(&loop_id) {
                lewp.offsets.clone_from(smoothed);
            }
        }
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

// The smoothing of the loops (see `smoothed_offsets`): its maximum number of rounds (it stops earlier when no crossing
// moves more than `SMOOTHING_CONVERGED` of its edge in a round), how far a crossing moves to its target per round, and
// the least distance (relative to the edge) to the ends of the edge and to the other loops on it.
const SMOOTHING_ROUNDS: usize = 500;
const SMOOTHING_CONVERGED: f64 = 1e-4;
const SMOOTHING_STEP: f64 = 0.5;
const SMOOTHING_GAP: f64 = 0.02;

/// The positions where the loops cross the mesh edges (as `Loop::offsets`), slightly smoothed: only along those edges
/// (so every loop stays in its triangles), every round towards the point of the edge closest to the middle of the
/// neighboring crossings (Laplacian smoothing, constrained to the edges). The loops that cross the same edge keep their
/// order on it, so the loop structure stays the same.
#[must_use]
pub fn smoothed_offsets(
    mesh: &Mesh<INPUT>,
    loops: &slotmap::SlotMap<LoopID, Loop>,
) -> HashMap<LoopID, Vec<f64>> {
    let mut offsets: HashMap<LoopID, Vec<f64>> =
        loops.iter().map(|(id, l)| (id, l.offsets.clone())).collect();
    // (Only loops with an offset for every edge, crossing every edge as a pair of half-edges.)
    let usable = |l: &Loop| l.offsets.len() == l.edges.len() && l.edges.len() % 2 == 0;
    if !loops.values().all(usable) {
        return offsets;
    }
    // Every crossing (`edges[2k]` and its twin `edges[2k + 1]`), by the canonical half-edge of its edge, in order
    // along that half-edge.
    let canonical = |edge: EdgeID| {
        let twin = mesh.twin(edge);
        if edge.raw() < twin.raw() { edge } else { twin }
    };
    let along = |offsets: &HashMap<LoopID, Vec<f64>>, (id, k): (LoopID, usize)| {
        let (edge, t) = (loops[id].edges[2 * k], offsets[&id][2 * k]);
        if canonical(edge) == edge { t } else { 1. - t }
    };
    let mut on_edge: HashMap<EdgeID, Vec<(LoopID, usize)>> = HashMap::default();
    for (id, l) in loops {
        for k in 0..l.edges.len() / 2 {
            on_edge.entry(canonical(l.edges[2 * k])).or_default().push((id, k));
        }
    }
    let mut neighbors: HashMap<(LoopID, usize), [Option<(LoopID, usize)>; 2]> = HashMap::default();
    for crossings in on_edge.values_mut() {
        crossings.sort_by(|&a, &b| along(&offsets, a).total_cmp(&along(&offsets, b)));
        for (i, &crossing) in crossings.iter().enumerate() {
            let below = i.checked_sub(1).map(|j| crossings[j]);
            neighbors.insert(crossing, [below, crossings.get(i + 1).copied()]);
        }
    }
    for _ in 0..SMOOTHING_ROUNDS {
        let mut moved: f64 = 0.;
        for (id, l) in loops {
            let n = l.edges.len() / 2;
            if n < 3 {
                continue;
            }
            for k in 0..n {
                let point = |k: usize, offsets: &HashMap<LoopID, Vec<f64>>| {
                    mesh.midpoint_offset(l.edges[2 * k], offsets[&id][2 * k])
                };
                let target = (point((k + n - 1) % n, &offsets) + point((k + 1) % n, &offsets)) / 2.;
                let edge = l.edges[2 * k];
                let (a, b) = (mesh.position(mesh.root(edge)), mesh.position(mesh.toor(edge)));
                let closest =
                    ((target - a).dot(&(b - a)) / (b - a).norm_squared().max(1e-300)).clamp(0., 1.);
                let t = offsets[&id][2 * k];
                let t = t + SMOOTHING_STEP * (closest - t);
                // Between the ends of the edge and the loops below and above on it (along the canonical half-edge).
                let [below, above] = neighbors[&(id, k)];
                let low = below.map_or(0., |c| along(&offsets, c)) + SMOOTHING_GAP;
                let high = above.map_or(1., |c| along(&offsets, c)) - SMOOTHING_GAP;
                if low >= high {
                    continue;
                }
                let flip = |t: f64| if canonical(edge) == edge { t } else { 1. - t };
                let t = flip(flip(t).clamp(low, high));
                let loop_offsets = offsets.get_mut(&id).expect("every loop");
                moved = moved.max((loop_offsets[2 * k] - t).abs());
                loop_offsets[2 * k] = t;
                loop_offsets[2 * k + 1] = 1. - t;
            }
        }
        if moved < SMOOTHING_CONVERGED {
            break;
        }
    }
    offsets
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
