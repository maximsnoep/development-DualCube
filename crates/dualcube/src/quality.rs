//! Quality criteria for polycube segmentations.
//!
//! A [`QualityReport`] collects the quality terms of a solution; they are combined into a single score (to be
//! maximized) with the weights in [`QualityParams`]:
//!
//! penalty P = w_F * fidelity + w_P * path regularity + w_C * coherence + w_R * rectangularity
//!     + w_K * corner regularity + w_N * non-degeneracy + w_S * correspondence + complexity * #loops,
//! score = 1 / (1 + P),
//!
//! such that the score lies in (0, 1] (1: perfect), and a higher score is better.
//!
//! Earlier criteria (fidelity + orthogonality - beta * #loops, and fidelity - beta * #loops as in the paper) were
//! dropped: on organic shapes they hardly correlate with the fidelity of the results (they keep solutions too coarse),
//! and fidelity itself is redundant with distortion (correlation -0.99 over all experiments) while distortion
//! penalizes badly aligned regions much more strongly. Fidelity is still reported (to compare with the paper).
//!
//! The terms:
//!
//! - *alignment* (reported only; the fidelity of the paper): area-weighted mean of n(t) . l(t) over all triangles (normal
//!   vs. label direction)
//!   [LVS13, DPM22].
//! - *fidelity*: area-weighted mean of (1/c - c)^2 with c = n(t) . l(t), the (symmetric Dirichlet) distortion of
//!   flattening a triangle onto the plane of its label (singular values 1 and c). Unlike fidelity, it grows quickly
//!   for badly aligned triangles (0 at 0 degrees, 0.08 at 30, 0.5 at 45, 2.25 at 60, 13 at 75), and is capped for
//!   (nearly) folded triangles. Small but badly aligned features thus justify a loop, while refining well aligned
//!   regions gains little: this suits the coarse-to-fine construction.
//! - *path regularity*: length-weighted mean over all patch boundaries (paths) of (a) the fraction of the path length
//!   that runs backwards along the axis of its polycube edge (non-monotone boundaries / turning points [LVS13]), and
//!   (b) log^2 of the ratio between the path length and the distance between its corners along its axis (straightness).
//! - *coherence*: how much the normals within every patch differ, whatever their direction (unlike fidelity,
//!   which compares them with the label): 1 - |area-weighted mean normal| of a patch (0 for a flat patch), the
//!   area-weighted mean over the patches.
//! - *rectangularity*: how far every patch is from a rectangle: log^2 of the ratios of the lengths of its opposite
//!   sides (paths), averaged over both pairs, plus ((theta - pi/2) / (pi/2))^2 averaged over its corners, with theta the
//!   angle (in the tangent plane of the patch) between the two paths at the corner; the area-weighted mean over the
//!   patches.
//! - *corner regularity*: whether the rows of corners are locally straight. For every three corners a, b, c on a straight
//!   line of the polycube (the edges a-b and b-c in the same direction, so all three on the same levels of the two other
//!   axes), the deviation of b from the line between a and c in the directions of those levels, relative to the length
//!   of the triple, squared; the mean over all such triples. Rows of corners may bend globally (a linear change between
//!   a and c costs nothing), but not kink locally.
//! - *non-degeneracy*: penalizes small patches that are not nice rectangles (they easily degenerate in the map and the
//!   quad mesh), counted per patch (not by area, unlike the other terms, so a small bad patch is not cheap): the mean
//!   over all patches of smallness * badness, with smallness ln(0.25 / s) for an area s relative to the mean patch area
//!   below 0.25 (0 otherwise), and badness the rectangularity of the patch, plus the wiggle of its sides (log^2 of
//!   their length over the distance between their corners), plus 5 * (1 - |area-weighted mean normal|) (its curving).
//!   Small patches that are nice rectangles, and large curved patches, cost nothing here.
//! - *correspondence*: how well the lengths of the polycube edges match the lengths of their paths. All corners of a level of
//!   the polycube (e.g., those of parallel loops, whose edge lengths the polycube couples) get the same coordinate: the
//!   mean position of the corners along its axis (as in the geometric polycube, see `Polycube::resize`). E_S is the
//!   mismatch between the length of every path and the length of its polycube edge (after one global scale), relative
//!   to the total length of the paths.
//! - *complexity* (the number of loops), *corners* (vertices with at least three different labels around them, as in Table 1 of the paper), and
//!   *irregular corners* (polycube vertices of degree other than 4).
//!
//! A cheap estimate of the score can be computed from the dual structure alone ([`QualityReport::estimate`]): every
//! triangle gets the best label among the labels of the polycube corners of its loop regions.

use crate::prelude::*;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

// Triangles whose normal deviates more than this angle from their label count as (nearly) folded.
const MAX_DISTORTION_ANGLE: f64 = 80. * PI / 180.;
// Paths whose corners are closer than this fraction of their length along their axis count as fully stretched.
const MIN_STRETCH_RATIO: f64 = 1. / 20.;

/// Weights of the quality terms (all terms are penalties).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct QualityWeights {
    pub fidelity: f64,
    pub path_regularity: f64,
    pub coherence: f64,
    pub rectangularity: f64,
    pub corner_regularity: f64,
    pub non_degeneracy: f64,
    pub correspondence: f64,
    /// The penalty per loop (beta).
    pub complexity: f64,
}

impl Default for QualityWeights {
    fn default() -> Self {
        Self {
            fidelity: 1.,
            path_regularity: 0.1,
            coherence: 2.,
            rectangularity: 0.2,
            corner_regularity: 1.,
            non_degeneracy: 1.,
            correspondence: 0.5,
            complexity: 0.005,
        }
    }
}

/// The quality criterion.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct QualityParams {
    pub weights: QualityWeights,
}

impl Default for QualityParams {
    fn default() -> Self {
        Self {
            weights: QualityWeights::default(),
        }
    }
}

/// Which terms to compute.
#[derive(Clone, Copy, Debug, Default)]
pub struct QualityTerms {
    pub fidelity: bool,
    pub path_regularity: bool,
    pub coherence: bool,
    pub rectangularity: bool,
    pub corner_regularity: bool,
    pub non_degeneracy: bool,
    pub correspondence: bool,
    pub corners: bool,
}

impl QualityTerms {
    #[must_use]
    pub const fn all() -> Self {
        Self {
            fidelity: true,
            path_regularity: true,
            coherence: true,
            rectangularity: true,
            corner_regularity: true,
            non_degeneracy: true,
            correspondence: true,
            corners: true,
        }
    }

    /// Only the terms with a nonzero weight.
    #[must_use]
    pub fn needed(weights: &QualityWeights) -> Self {
        Self {
            fidelity: weights.fidelity != 0.,
            path_regularity: weights.path_regularity != 0.,
            coherence: weights.coherence != 0.,
            rectangularity: weights.rectangularity != 0.,
            corner_regularity: weights.corner_regularity != 0.,
            non_degeneracy: weights.non_degeneracy != 0.,
            correspondence: weights.correspondence != 0.,
            corners: false,
        }
    }
}

/// All quality terms of a solution (`None` if not computed or not available).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct QualityReport {
    /// The fidelity of the paper (area-weighted mean alignment of the triangles with their labels; reported only).
    pub alignment: Option<f64>,
    pub fidelity: Option<f64>,
    /// Fraction of the path length running backwards along its axis.
    pub path_backtrack: Option<f64>,
    /// Mean log^2 of the path length over the distance between its corners along its axis.
    pub path_stretch: Option<f64>,
    pub coherence: Option<f64>,
    pub rectangularity: Option<f64>,
    pub corner_regularity: Option<f64>,
    pub non_degeneracy: Option<f64>,
    pub correspondence: Option<f64>,
    pub loops: usize,
    pub corners: Option<usize>,
    pub irregular_corners: Option<usize>,
}

impl QualityReport {
    /// The path-regularity term E_P.
    #[must_use]
    pub fn path_regularity(&self) -> Option<f64> {
        Some(self.path_backtrack? + self.path_stretch?)
    }

    /// The score for the given weights (higher is better), or `None` if a term with a nonzero weight is missing.
    #[must_use]
    pub fn score(&self, w: &QualityWeights) -> Option<f64> {
        Some(normalized(self.penalty(w)?))
    }

    /// The penalty P for the given weights (see the module documentation), or `None` if a term with a nonzero weight
    /// is missing.
    #[must_use]
    pub fn penalty(&self, w: &QualityWeights) -> Option<f64> {
        let term = |weight: f64, value: Option<f64>| -> Option<f64> {
            if weight == 0. {
                Some(0.)
            } else {
                value.map(|v| weight * v)
            }
        };
        Some(
            term(w.fidelity, self.fidelity)?
                + term(w.path_regularity, self.path_regularity())?
                + term(w.coherence, self.coherence)?
                + term(w.rectangularity, self.rectangularity)?
                + term(w.corner_regularity, self.corner_regularity)?
                + term(w.non_degeneracy, self.non_degeneracy)?
                + term(w.correspondence, self.correspondence)?
                + w.complexity * self.loops as f64,
        )
    }

    /// The score for the given weights, ignoring terms that are not available (e.g., for estimates).
    #[must_use]
    pub fn partial_score(&self, w: &QualityWeights) -> f64 {
        let term = |weight: f64, value: Option<f64>| value.map_or(0., |v| weight * v);
        normalized(
            term(w.fidelity, self.fidelity)
                + term(w.path_regularity, self.path_regularity())
                + term(w.coherence, self.coherence)
                + term(w.rectangularity, self.rectangularity)
                + term(w.corner_regularity, self.corner_regularity)
                + term(w.non_degeneracy, self.non_degeneracy)
                + term(w.correspondence, self.correspondence)
                + w.complexity * self.loops as f64,
        )
    }

    /// Compute the given terms for a complete layout. Fidelity is always included (it is computed together with the
    /// layout).
    #[must_use]
    pub fn compute(
        loops: usize,
        layout: &Layout,
        _params: &QualityParams,
        terms: QualityTerms,
    ) -> Self {
        let polycube = &layout.polycube_ref;
        let mut report = Self {
            alignment: layout.alignment,
            loops,
            ..Self::default()
        };
        if terms.fidelity {
            report.fidelity = layout_distortion(layout);
        }
        if terms.path_regularity
            && let Some((backtrack, stretch)) = boundary_terms(layout)
        {
            report.path_backtrack = Some(backtrack);
            report.path_stretch = Some(stretch);
        }
        if terms.coherence {
            report.coherence = patch_coherence(layout);
        }
        let shapes = (terms.rectangularity || terms.non_degeneracy)
            .then(|| patch_shapes(layout))
            .flatten();
        if terms.rectangularity {
            report.rectangularity = shapes.as_deref().and_then(patch_rectangularity);
        }
        if terms.corner_regularity {
            report.corner_regularity = corner_flatness(layout);
        }
        if terms.non_degeneracy {
            report.non_degeneracy = shapes.as_deref().and_then(small_bad_patches);
        }
        if terms.correspondence {
            report.correspondence = size_mismatch(layout);
        }
        if terms.corners {
            let (corners, irregular) = corner_counts(polycube);
            report.corners = Some(corners);
            report.irregular_corners = Some(irregular);
        }
        report
    }

    /// Estimate the terms from the dual structure (and its polycube) only, without a layout: every triangle gets the
    /// best label among the labels of the polycube corners of the loop regions of its vertices. Boundary and
    /// features are not available.
    #[must_use]
    pub fn estimate(
        loops: usize,
        dual: &Dual,
        polycube: &Polycube,
        _params: &QualityParams,
        terms: QualityTerms,
    ) -> Self {
        let mesh = &dual.mesh_ref;
        let mut vert_labels: HashMap<VertID, Vec<Vector3D>> = HashMap::new();
        for region in dual.loop_structure.face_ids() {
            let Some(&corner) = polycube.region_to_vertex.get_by_left(&region) else {
                continue;
            };
            let labels = corner_labels(polycube, corner);
            for vert in dual.region_to_verts(region) {
                vert_labels.insert(vert, labels.clone());
            }
        }

        let mut area = 0.;
        let mut fidelity = 0.;
        let mut distortion = 0.;
        for face in mesh.face_ids() {
            let normal = mesh.normal(face).normalize();
            let best = mesh
                .vertices(face)
                .filter_map(|v| vert_labels.get(&v))
                .flatten()
                .map(|label| normal.dot(label))
                .fold(f64::NEG_INFINITY, f64::max);
            if !best.is_finite() {
                continue;
            }
            let a = mesh.size(face);
            area += a;
            fidelity += a * best;
            distortion += a * flattening_distortion(best);
        }

        let mut report = Self {
            loops,
            ..Self::default()
        };
        if area > 0. {
            report.alignment = Some(fidelity / area);
            if terms.fidelity {
                report.fidelity = Some(distortion / area);
            }
        }
        if terms.corners {
            let (corners, irregular) = corner_counts(polycube);
            report.corners = Some(corners);
            report.irregular_corners = Some(irregular);
        }
        report
    }
}

// The correspondence term (see the module documentation).
fn size_mismatch(layout: &Layout) -> Option<f64> {
    let (mismatches, total_path) = correspondence_per_edge(layout)?;
    Some(mismatches.iter().map(|(_, m)| m).sum::<f64>() / total_path)
}

// Per polycube edge (once): the mismatch between the length of its path and its length in the polycube (after the
// global scale), and the total length of the paths (see the correspondence term).
fn correspondence_per_edge(layout: &Layout) -> Option<(Vec<(EdgeKey<POLYCUBE>, f64)>, f64)> {
    let dual = &layout.dual_ref;
    let polycube = &layout.polycube_ref;
    let position = |vert: VertKey<POLYCUBE>| -> Option<Vector3D> {
        Some(
            layout
                .granulated_mesh
                .position(*layout.vert_to_corner.get_by_left(&vert)?),
        )
    };
    // Per axis: the level of every polycube vertex, and the coordinate of every level (the mean of its corners).
    let mut level_of: HashMap<VertKey<POLYCUBE>, [usize; 3]> = HashMap::new();
    let mut coordinates: [Vec<f64>; 3] = Default::default();
    for direction in DIRECTIONS {
        let axis = direction as usize;
        for (level, zones) in dual.level_graphs.levels[axis].iter().enumerate() {
            let (mut sum, mut count) = (0., 0);
            for &zone in zones {
                for region in &dual.level_graphs.zones[zone].regions {
                    let &vert = polycube.region_to_vertex.get_by_left(region)?;
                    level_of.entry(vert).or_default()[axis] = level;
                    sum += position(vert)?[axis];
                    count += 1;
                }
            }
            coordinates[axis].push(if count > 0 {
                sum / f64::from(count)
            } else {
                0.
            });
        }
    }
    // Every polycube edge (once): the length of its path, and its length in the polycube.
    let mesh = &layout.granulated_mesh;
    let mut lengths = vec![];
    for edge in polycube.structure.edge_ids() {
        if edge > polycube.structure.twin(edge) {
            continue;
        }
        let [u, v] = polycube.structure.vertices(edge).collect_array::<2>()?;
        let (level_u, level_v) = (level_of.get(&u)?, level_of.get(&v)?);
        let Some(axis) = (0..3).find(|&axis| level_u[axis] != level_v[axis]) else {
            continue;
        };
        let path = layout.edge_to_path.get(&edge)?;
        let path_length: f64 = path
            .windows(2)
            .map(|w| (mesh.position(w[1]) - mesh.position(w[0])).norm())
            .sum();
        let edge_length =
            (coordinates[axis][level_v[axis]] - coordinates[axis][level_u[axis]]).abs();
        lengths.push((edge, path_length, edge_length));
    }
    // One global scale (the polycube is not the size of the surface), then the mismatch per edge.
    let total_path: f64 = lengths.iter().map(|(_, p, _)| p).sum();
    let total_edge: f64 = lengths.iter().map(|(_, _, e)| e).sum();
    if total_path <= 0. || total_edge <= 0. {
        return None;
    }
    let scale = total_path / total_edge;
    Some((
        lengths
            .into_iter()
            .map(|(edge, p, e)| (edge, (p - scale * e).abs()))
            .collect(),
        total_path,
    ))
}

// The coherence term E_C (see the module documentation), from the normals per patch (see `Layout::patch_normals`).
fn patch_coherence(layout: &Layout) -> Option<f64> {
    let (mut incoherent, mut total) = (0., 0.);
    for &(normal_sum, area) in layout.patch_normals.values() {
        incoherent += area - normal_sum.norm();
        total += area;
    }
    (total > 0.).then(|| incoherent / total)
}

// The corner-regularity term (see the module documentation).
fn corner_flatness(layout: &Layout) -> Option<f64> {
    let (per_corner, count) = corner_regularity_per_corner(layout)?;
    Some(if count > 0 {
        per_corner.values().sum::<f64>() / count as f64
    } else {
        0.
    })
}

// Per middle corner of straight triples: the sum of their squared relative deviations (see the corner-regularity
// term), and the number of triples.
fn corner_regularity_per_corner(
    layout: &Layout,
) -> Option<(HashMap<VertKey<POLYCUBE>, f64>, usize)> {
    let polycube = &layout.polycube_ref.structure;
    let position = |corner: VertKey<POLYCUBE>| -> Option<Vector3D> {
        Some(
            layout
                .granulated_mesh
                .position(*layout.vert_to_corner.get_by_left(&corner)?),
        )
    };
    let mut per_corner: HashMap<VertKey<POLYCUBE>, f64> = HashMap::new();
    let mut count = 0;
    for b in polycube.vert_ids() {
        let neighbors = polycube.neighbors(b).collect_vec();
        for (i, &a) in neighbors.iter().enumerate() {
            for &c in &neighbors[i + 1..] {
                // A straight triple a - b - c of the polycube, along one axis.
                let incoming = (polycube.position(b) - polycube.position(a)).normalize();
                let outgoing = (polycube.position(c) - polycube.position(b)).normalize();
                // (a and c on opposite sides of b: the edges are in the same direction, whatever the order of the pair)
                if incoming.dot(&outgoing) < 0.999 {
                    continue;
                }
                let axis = (0..3)
                    .max_by(|&x, &y| incoming[x].abs().total_cmp(&incoming[y].abs()))
                    .unwrap_or(0);
                let (pa, pb, pc) = (position(a)?, position(b)?, position(c)?);
                let (l1, l2) = ((pb - pa).norm(), (pc - pb).norm());
                if l1 + l2 <= 0. {
                    continue;
                }
                // The deviation of b from the line between a and c, in the directions of the two levels.
                let mut deviation = pb - (pa + (pc - pa) * (l1 / (l1 + l2)));
                deviation[axis] = 0.;
                *per_corner.entry(b).or_default() += (deviation.norm() / (0.5 * (l1 + l2))).powi(2);
                count += 1;
            }
        }
    }
    Some((per_corner, count))
}

// The shape of a patch (see `patch_shapes`).
struct PatchShape {
    patch: FaceKey<POLYCUBE>,
    area: f64,
    // log^2 of the ratios of the lengths of the opposite sides, averaged over both pairs.
    sides: f64,
    // ((theta - pi/2) / (pi/2))^2 averaged over the corners.
    corners: f64,
    // log^2 of the length of a side over the distance between its corners, averaged over the sides.
    wiggle: f64,
    // 1 - |area-weighted mean normal|.
    incoherence: f64,
}

// The shape of every (quadrilateral) patch.
fn patch_shapes(layout: &Layout) -> Option<Vec<PatchShape>> {
    let mesh = &layout.granulated_mesh;
    let structure = &layout.polycube_ref.structure;
    let length = |path: &[VertID]| -> f64 {
        path.windows(2)
            .map(|w| (mesh.position(w[1]) - mesh.position(w[0])).norm())
            .sum()
    };
    // The direction in which a path leaves its first vertex: towards its point at a quarter of its length.
    let direction = |path: &[VertID], total: f64| -> Vector3D {
        let quarter = 0.25 * total;
        let mut walked = 0.;
        for w in path.windows(2) {
            walked += (mesh.position(w[1]) - mesh.position(w[0])).norm();
            if walked >= quarter {
                return mesh.position(w[1]) - mesh.position(path[0]);
            }
        }
        mesh.position(*path.last().unwrap()) - mesh.position(path[0])
    };
    let mut shapes = vec![];
    for (&patch, &(normal_sum, area)) in &layout.patch_normals {
        let edges = structure.edges(patch).collect_vec();
        if edges.len() != 4 || area <= 0. {
            continue;
        }
        let paths = edges
            .iter()
            .map(|edge| layout.edge_to_path.get(edge).map(Vec::as_slice))
            .collect::<Option<Vec<_>>>()?;
        let lengths = paths.iter().map(|p| length(p)).collect_vec();
        if lengths.iter().any(|&l| l <= 0.) {
            continue;
        }
        let sides =
            0.5 * ((lengths[0] / lengths[2]).ln().powi(2) + (lengths[1] / lengths[3]).ln().powi(2));
        // The corner between path i - 1 (ending there) and path i (starting there), in the tangent plane of the patch.
        let normal = normal_sum.normalize();
        let mut corners = 0.;
        for i in 0..4 {
            let incoming = paths[(i + 3) % 4].iter().rev().copied().collect_vec();
            let project = |d: Vector3D| d - normal * d.dot(&normal);
            let (a, b) = (
                project(direction(paths[i], lengths[i])),
                project(direction(&incoming, lengths[(i + 3) % 4])),
            );
            let angle = if a.norm() > 0. && b.norm() > 0. {
                a.angle(&b)
            } else {
                PI / 2.
            };
            corners += ((angle - PI / 2.) / (PI / 2.)).powi(2);
        }
        let wiggle = paths
            .iter()
            .zip(&lengths)
            .map(|(path, &length)| {
                let chord = (mesh.position(*path.last().unwrap()) - mesh.position(path[0])).norm();
                (length / chord.max(length * 1e-3)).ln().powi(2)
            })
            .sum::<f64>()
            / 4.;
        shapes.push(PatchShape {
            patch,
            area,
            sides,
            corners: corners / 4.,
            wiggle,
            incoherence: 1. - normal_sum.norm() / area,
        });
    }
    Some(shapes)
}

// The small-bad-patches penalty of every patch (see the module documentation), in the order of the shapes.
fn small_bad_per_patch(shapes: &[PatchShape]) -> Vec<f64> {
    // Patches smaller than this fraction of the mean area count, the more the smaller.
    const SMALL: f64 = 0.25;
    if shapes.is_empty() {
        return vec![];
    }
    let mean_area = shapes.iter().map(|s| s.area).sum::<f64>() / shapes.len() as f64;
    shapes
        .iter()
        .map(|shape| {
            let relative = shape.area / mean_area;
            let smallness = if relative < SMALL {
                (SMALL / relative.max(1e-6)).ln()
            } else {
                0.
            };
            let badness = shape.sides + shape.corners + shape.wiggle + 5. * shape.incoherence;
            smallness * badness
        })
        .collect()
}

// The small-bad-patches term E_T (see the module documentation).
fn small_bad_patches(shapes: &[PatchShape]) -> Option<f64> {
    let penalties = small_bad_per_patch(shapes);
    (!penalties.is_empty()).then(|| penalties.iter().sum::<f64>() / penalties.len() as f64)
}

/// How much every element of a layout (patch, path, corner) contributes to its penalty, to target the mutations at the
/// elements that need it: its share of every term (with the weights). A path also gets half of the penalties of its
/// two patches, and a corner its own share plus half of its paths' own shares and a quarter of its patches'.
#[derive(Clone, Debug, Default)]
pub struct ElementPenalties {
    pub patches: HashMap<FaceKey<POLYCUBE>, f64>,
    /// Per polycube edge (both orientations).
    pub paths: HashMap<EdgeKey<POLYCUBE>, f64>,
    pub corners: HashMap<VertKey<POLYCUBE>, f64>,
}

/// The penalty of every element of a layout (see `ElementPenalties`).
#[must_use]
pub fn element_penalties(layout: &Layout, weights: &QualityWeights) -> ElementPenalties {
    let structure = &layout.polycube_ref.structure;
    let total_area: f64 = layout
        .patch_normals
        .values()
        .map(|(_, a)| a)
        .sum::<f64>()
        .max(1e-300);
    // Patches: fidelity, coherence, rectangularity, and non-degeneracy.
    let shapes = patch_shapes(layout).unwrap_or_default();
    let small = small_bad_per_patch(&shapes);
    let mut patches: HashMap<FaceKey<POLYCUBE>, f64> = HashMap::new();
    for (patch, triangles) in &layout.patch_triangles {
        let fidelity: f64 = triangles
            .iter()
            .map(|&(a, c)| a * flattening_distortion(c))
            .sum();
        *patches.entry(*patch).or_default() += weights.fidelity * fidelity / total_area;
    }
    for (patch, &(normal_sum, area)) in &layout.patch_normals {
        *patches.entry(*patch).or_default() +=
            weights.coherence * (area - normal_sum.norm()) / total_area;
    }
    for (shape, small) in shapes.iter().zip(&small) {
        *patches.entry(shape.patch).or_default() +=
            weights.rectangularity * shape.area * (shape.sides + shape.corners) / total_area
                + weights.non_degeneracy * small / shapes.len().max(1) as f64;
    }
    // Paths: path regularity and correspondence (their own shares), plus half of their patches.
    let mut own_paths: HashMap<EdgeKey<POLYCUBE>, f64> = HashMap::new();
    let (regularity, total_length) = path_regularity_per_edge(layout);
    for (edge, backwards, stretch) in regularity {
        *own_paths.entry(edge).or_default() +=
            weights.path_regularity * (backwards + stretch) / total_length.max(1e-300);
    }
    if let Some((mismatches, total_path)) = correspondence_per_edge(layout) {
        for (edge, mismatch) in mismatches {
            *own_paths.entry(edge).or_default() += weights.correspondence * mismatch / total_path;
        }
    }
    let own = |edge: EdgeKey<POLYCUBE>| {
        own_paths
            .get(&edge)
            .or_else(|| own_paths.get(&structure.twin(edge)))
            .copied()
            .unwrap_or(0.)
    };
    let mut paths = HashMap::new();
    for edge in structure.edge_ids() {
        let adjacent = [structure.face(edge), structure.face(structure.twin(edge))]
            .iter()
            .map(|patch| patches.get(patch).copied().unwrap_or(0.))
            .sum::<f64>();
        paths.insert(edge, own(edge) + 0.5 * adjacent);
    }
    // Corners: corner regularity (their own share), plus half of their paths' own shares and a quarter of their patches.
    let (regularity, triples) = corner_regularity_per_corner(layout).unwrap_or_default();
    let mut corners = HashMap::new();
    for corner in structure.vert_ids() {
        let mut penalty = weights.corner_regularity
            * regularity.get(&corner).copied().unwrap_or(0.)
            / triples.max(1) as f64;
        for edge in structure.edges(corner) {
            penalty += 0.5 * own(edge);
        }
        for patch in structure.faces(corner) {
            penalty += 0.25 * patches.get(&patch).copied().unwrap_or(0.);
        }
        corners.insert(corner, penalty);
    }
    ElementPenalties {
        patches,
        paths,
        corners,
    }
}

/// The penalty of every input face (the triangles of the refined mesh in it, see `penalty_per_triangle`, times their
/// areas), to target the loops at the faces that need it.
#[must_use]
pub fn penalty_per_input_face(layout: &Layout, weights: &QualityWeights) -> Vec<(FaceID, f64)> {
    let input = &layout.dual_ref.mesh_ref;
    let mut sums: HashMap<FaceID, f64> = HashMap::new();
    for (face, penalty) in penalty_per_triangle(layout, weights) {
        let original = match &layout.regions {
            Some(tracking) => tracking.face_input.get(&face).copied(),
            None => input.face_ids_iter().any(|f| f == face).then_some(face),
        };
        if let Some(original) = original {
            *sums.entry(original).or_default() += layout.granulated_mesh.size(face) * penalty;
        }
    }
    sums.into_iter().filter(|&(_, w)| w > 0.).collect()
}

/// The penalty of every triangle of the layout (of its refined mesh), for showing where the quality is lost: the
/// weighted fidelity penalty of the triangle, plus the weighted coherence, rectangularity, and non-degeneracy penalties
/// of its patch. The terms of paths and corners (path regularity, corner regularity, correspondence) are not included.
#[must_use]
pub fn penalty_per_triangle(layout: &Layout, weights: &QualityWeights) -> HashMap<FaceID, f64> {
    let shapes = patch_shapes(layout).unwrap_or_default();
    let small = small_bad_per_patch(&shapes);
    let per_patch: HashMap<FaceKey<POLYCUBE>, f64> = shapes
        .iter()
        .zip(&small)
        .map(|(shape, &small)| {
            (
                shape.patch,
                weights.coherence * shape.incoherence
                    + weights.rectangularity * (shape.sides + shape.corners)
                    + weights.non_degeneracy * small,
            )
        })
        .collect();
    let mut penalties = HashMap::new();
    for (patch, Patch { faces }) in &layout.face_to_patch {
        let patch_penalty = per_patch.get(patch).copied().unwrap_or(0.);
        for &face in faces {
            let distortion = layout
                .alignment_per_triangle
                .get(&face)
                .map_or(0., |&c| flattening_distortion(c));
            penalties.insert(face, weights.fidelity * distortion + patch_penalty);
        }
    }
    penalties
}

// The rectangularity term E_R (see the module documentation).
fn patch_rectangularity(shapes: &[PatchShape]) -> Option<f64> {
    let (mut sum, mut total) = (0., 0.);
    for shape in shapes {
        sum += shape.area * (shape.sides + shape.corners);
        total += shape.area;
    }
    (total > 0.).then(|| sum / total)
}

// The score of a penalty: in (0, 1], decreasing.
fn normalized(penalty: f64) -> f64 {
    1. / (1. + penalty.max(0.))
}

/// Distortion of flattening a triangle whose normal makes an angle with cosine `c` with its label: singular values
/// 1 and c, symmetric Dirichlet energy (1/c - c)^2. Capped for (nearly) folded triangles.
#[must_use]
pub fn flattening_distortion(c: f64) -> f64 {
    let c = c.max(MAX_DISTORTION_ANGLE.cos());
    (1. / c - c).powi(2)
}

// The (unit) label directions of the polycube faces around a polycube vertex.
fn corner_labels(polycube: &Polycube, corner: VertKey<POLYCUBE>) -> Vec<Vector3D> {
    polycube
        .structure
        .faces(corner)
        .map(|face| {
            let (direction, sign) = to_principal_direction(polycube.structure.normal(face));
            to_vector(direction, sign)
        })
        .unique_by(|v| [v.x as i64, v.y as i64, v.z as i64])
        .collect()
}

// Number of corners (polycube vertices with at least three different labels around them), and number of polycube
// vertices with degree other than 4.
fn corner_counts(polycube: &Polycube) -> (usize, usize) {
    let structure = &polycube.structure;
    let corners = structure
        .vert_ids()
        .into_iter()
        .filter(|&v| corner_labels(polycube, v).len() >= 3)
        .count();
    let irregular = structure
        .vert_ids()
        .into_iter()
        .filter(|&v| structure.degree(v) != 4)
        .count();
    (corners, irregular)
}

fn layout_distortion(layout: &Layout) -> Option<f64> {
    // From the areas and alignments of the triangles (computed with the patches), if available.
    if !layout.patch_triangles.is_empty() {
        let (mut distortion, mut total) = (0., 0.);
        for triangles in layout.patch_triangles.values() {
            for &(area, c) in triangles.iter() {
                distortion += area * flattening_distortion(c);
                total += area;
            }
        }
        return (total > 0.).then(|| distortion / total);
    }
    let mesh = &layout.granulated_mesh;
    let polycube = &layout.polycube_ref.structure;
    let total: f64 = mesh.face_ids().into_iter().map(|f| mesh.size(f)).sum();
    if total <= 0. {
        return None;
    }
    let mut distortion = 0.;
    for (&patch, Patch { faces }) in &layout.face_to_patch {
        let (direction, sign) = to_principal_direction(polycube.normal(patch));
        let label = to_vector(direction, sign);
        for &face in faces {
            let c = mesh.normal(face).normalize().dot(&label);
            distortion += mesh.size(face) * flattening_distortion(c);
        }
    }
    Some(distortion / total)
}

// (backtrack, stretch), both length-weighted over all paths.
fn boundary_terms(layout: &Layout) -> Option<(f64, f64)> {
    let (per_edge, total_length) = path_regularity_per_edge(layout);
    (total_length > 0.).then(|| {
        (
            per_edge.iter().map(|(_, b, _)| b).sum::<f64>() / total_length,
            per_edge.iter().map(|(_, _, s)| s).sum::<f64>() / total_length,
        )
    })
}

// Per polycube edge (once): the length of its path running backwards, and its length times log^2 of its stretch (see
// the path-regularity term); and the total length of the paths.
fn path_regularity_per_edge(layout: &Layout) -> (Vec<(EdgeKey<POLYCUBE>, f64, f64)>, f64) {
    let mesh = &layout.granulated_mesh;
    let polycube = &layout.polycube_ref.structure;
    let mut total_length = 0.;
    let mut per_edge = vec![];
    for edge in polycube.edge_ids() {
        if edge.raw() > polycube.twin(edge).raw() {
            continue;
        }
        let Some(path) = layout.edge_to_path.get(&edge) else {
            continue;
        };
        if path.len() < 2 {
            continue;
        }
        let (direction, sign) = to_principal_direction(polycube.vector(edge));
        let axis = to_vector(direction, sign);
        let mut length = 0.;
        let mut backwards = 0.;
        for w in path.windows(2) {
            let segment = mesh.position(w[1]) - mesh.position(w[0]);
            length += segment.norm();
            backwards += (-segment.dot(&axis)).max(0.);
        }
        if length <= 0. {
            continue;
        }
        let along = (mesh.position(*path.last().unwrap()) - mesh.position(path[0])).dot(&axis);
        let ratio = length / along.max(length * MIN_STRETCH_RATIO);
        total_length += length;
        per_edge.push((edge, backwards, length * ratio.ln().powi(2)));
    }
    (per_edge, total_length)
}
