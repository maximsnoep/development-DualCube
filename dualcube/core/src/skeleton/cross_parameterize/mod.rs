use std::collections::{HashMap, HashSet};

use log::{error, warn};
use mehsh::prelude::{FaceKey, HasFaces, HasPosition, HasVertices, Mesh, SetPosition, Vector2D, Vector3D};
use ordered_float::OrderedFloat;
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};

use crate::prelude::{PrincipalDirection, VertID, INPUT};
use crate::skeleton::orthogonalize::LabeledCurveSkeleton;

mod cutting_plan;
mod disk_boundary;
mod harmonic;

pub use cutting_plan::compute_cutting_plan;
use disk_boundary::{
    build_disk_boundary, find_cut_paths, ordered_boundary_on_our_side, OrderedBoundary,
};
use harmonic::{
    parameterize_disk_to_polygon, polygon_corners, solve_harmonic_2d, solve_harmonic_2d_with_cuts,
};

/// Minimum separation between cut endpoints on the same boundary,
/// measured in [0, 1] normalized arc-length. Two cut endpoints on the
/// same boundary loop must be at least this far apart.
const MIN_CUT_SEPARATION: f64 = 0.15;

/// The parameterization of a single region (skeleton node) onto a canonical 2D domain.
///
/// Both the input mesh region and the polycube mesh region are mapped to the same
/// canonical domain. The composition of these two maps gives the bijection
/// between the input and polycube surfaces for this region.
///
/// The canonical domains are:
/// - degree 0: a sphere (??? TODO)
/// - degree 1: a square, the single boundary maps to the entire boundary of the square
/// - degree 2+: a regular 4(d-1) gon for degree d.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionParameterization {
    /// For each input mesh vertex in this region, its 2D position in the canonical domain.
    pub input_to_canonical: HashMap<VertID, Vector2D>,

    /// For each polycube mesh vertex in this region, its 2D position in the canonical domain.
    /// NOTE: Keys are VertKey<POLYCUBE> stored as VertID via raw key, same convention as
    /// `SkeletonNode::patch_vertices` on the polycube skeleton.
    pub polycube_to_canonical: HashMap<VertID, Vector2D>,
}

/// A bijection between the input mesh surface and the polycube surface,
/// represented as a collection of per-region parameterizations through canonical domains.
///
/// For any input mesh vertex, the map gives a polycube-surface position (and vice versa)
/// by composing: input surface <-> canonical domain <-> polycube surface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolycubeMap {
    /// Per skeleton-node parameterization. The `NodeIndex` keys are valid for both
    /// the input and polycube `LabeledCurveSkeleton` (skeleton-graphs are isomorphic).
    pub regions: HashMap<NodeIndex, RegionParameterization>,
}

/// Describes how to cut a region with multiple boundary loops to disk topology.
///
/// For degree d ≥ 2, we need d-1 cuts forming a spanning tree over the d boundary loops.
/// Each cut is identified by the pair of skeleton `EdgeIndex`es whose boundary loops
/// it connects. The resulting disk has a single boundary, and the canonical domain
/// is a regular 4(d-1)-gon.
///
/// For degree 0 or 1, no cuts are needed (`cuts` is empty).
#[derive(Debug, Clone)]
pub struct CuttingPlan {
    /// Each entry `(edge_a, edge_b)` means: cut from the boundary loop on skeleton
    /// edge `edge_a` to the one on `edge_b`. There are d-1 such cuts.
    pub cuts: Vec<(EdgeIndex, EdgeIndex)>,
}

impl PolycubeMap {
    /// Constructs a `Mesh<INPUT>` whose vertices are the input mesh vertices repositioned
    /// onto the polycube surface. This is the `triangle_mesh_polycube` needed by `Quad`.
    ///
    /// For each input vertex, looks up its canonical-domain coordinates from the input
    /// parameterization, then finds the corresponding polycube-surface position by
    /// interpolating within the polycube parameterization of the same region.
    pub fn to_triangle_mesh_polycube(
        &self,
        input_mesh: &Mesh<INPUT>,
        input_skeleton: &LabeledCurveSkeleton,
        polycube_mesh: &Mesh<INPUT>,
    ) -> Mesh<INPUT> {
        let mut result = input_mesh.clone();

        // Build a lookup: input vertex -> region (NodeIndex).
        let mut vert_to_region: HashMap<VertID, NodeIndex> = HashMap::new();
        for node_idx in input_skeleton.node_indices() {
            for &v in &input_skeleton[node_idx].skeleton_node.patch_vertices {
                vert_to_region.insert(v, node_idx);
            }
        }

        // Pre-build per-region triangulated polycube faces in canonical UV space.
        // Each triangle stores: (uv_a, uv_b, uv_c, pos_a, pos_b, pos_c).
        let mut region_triangles: HashMap<NodeIndex, Vec<UVTriangle>> = HashMap::new();
        for (&node_idx, region) in &self.regions {
            let poly_uv = &region.polycube_to_canonical;
            if poly_uv.is_empty() {
                continue;
            }
            let poly_verts_set: HashSet<VertID> = poly_uv.keys().copied().collect();

            // Collect polycube faces whose vertices are all in this region.
            let mut region_faces: HashSet<FaceKey<INPUT>> = HashSet::new();
            for &v in poly_uv.keys() {
                for f in polycube_mesh.faces(v) {
                    if polycube_mesh.vertices(f).all(|fv| poly_verts_set.contains(&fv)) {
                        region_faces.insert(f);
                    }
                }
            }

            let mut triangles = Vec::new();
            for f in region_faces {
                let face_verts: Vec<VertID> = polycube_mesh.vertices(f).collect();
                // Triangulate: fan from first vertex.
                for i in 1..face_verts.len() - 1 {
                    let va = face_verts[0];
                    let vb = face_verts[i];
                    let vc = face_verts[i + 1];
                    if let (Some(&uv_a), Some(&uv_b), Some(&uv_c)) =
                        (poly_uv.get(&va), poly_uv.get(&vb), poly_uv.get(&vc))
                    {
                        triangles.push(UVTriangle {
                            uv: [uv_a, uv_b, uv_c],
                            pos: [
                                polycube_mesh.position(va),
                                polycube_mesh.position(vb),
                                polycube_mesh.position(vc),
                            ],
                        });
                    }
                }
            }
            region_triangles.insert(node_idx, triangles);
        }

        let mut unmapped = 0usize;
        for &v in input_mesh.vert_ids().iter() {
            let Some(&node_idx) = vert_to_region.get(&v) else { continue };
            let Some(region) = self.regions.get(&node_idx) else { continue };
            let Some(&uv) = region.input_to_canonical.get(&v) else { continue };
            let Some(triangles) = region_triangles.get(&node_idx) else { continue };

            if let Some(pos) = locate_in_triangles(uv, triangles) {
                result.set_position(v, pos);
            } else {
                // Fallback: project onto the nearest triangle.  This should not
                // happen if the triangulation fully covers the canonical domain.
                error!(
                    "to_triangle_mesh_polycube: vertex {:?} in region {:?} \
                     at UV ({:.4}, {:.4}) not inside any polycube triangle \
                     ({} triangles in region). Using nearest fallback.",
                    v, node_idx, uv.x, uv.y, triangles.len()
                );
                if let Some(pos) = nearest_triangle_fallback(uv, triangles) {
                    result.set_position(v, pos);
                } else {
                    unmapped += 1;
                }
            }
        }

        if unmapped > 0 {
            warn!(
                "to_triangle_mesh_polycube: {} input vertices could not be mapped.",
                unmapped
            );
        }

        result
    }
}

/// Performs cross-parameterization between the input and polycube labeled curve skeletons,
/// by mapping all regions individually to a shared canonical domain.
///
/// Both skeletons must be isomorphic (same graph topology, same `NodeIndex`/`EdgeIndex` values).
/// Each region (node) is independently parameterized onto a shared domain,
/// producing a per-vertex 2D coordinate for both the input and polycube sides.
pub fn cross_parameterize(
    input_skeleton: &LabeledCurveSkeleton,
    polycube_skeleton: &LabeledCurveSkeleton,
    input_mesh: &Mesh<INPUT>,
    polycube_mesh: &Mesh<INPUT>, // Polycube mesh passed as Mesh<INPUT> via raw key conversion
) -> PolycubeMap {
    let mut regions = HashMap::new();

    for node_idx in input_skeleton.node_indices() {
        let degree = input_skeleton.edges(node_idx).count();

        let region = parameterize_region(
            node_idx,
            degree,
            input_skeleton,
            polycube_skeleton,
            input_mesh,
            polycube_mesh,
        );

        regions.insert(node_idx, region);
    }

    PolycubeMap { regions }
}

/// Parameterizes a single region (node) on both the input and polycube side onto a shared canonical domain.
fn parameterize_region(
    node_idx: NodeIndex,
    degree: usize,
    input_skeleton: &LabeledCurveSkeleton,
    polycube_skeleton: &LabeledCurveSkeleton,
    input_mesh: &Mesh<INPUT>,
    polycube_mesh: &Mesh<INPUT>,
) -> RegionParameterization {
    // Compute a shared cutting plan using combined geodesic distances from both sides.
    // This determines *which* boundary loops to connect (the topology of the cuts).
    // The actual cut paths (the *how*) are computed later per-side in parameterize_side.
    let cutting_plan = compute_cutting_plan(
        node_idx,
        input_skeleton,
        polycube_skeleton,
        input_mesh,
        polycube_mesh,
    );

    // Parameterize each side independently using the shared cutting plan.
    let input_to_canonical =
        parameterize_side(node_idx, degree, input_skeleton, input_mesh, &cutting_plan);

    let polycube_to_canonical = parameterize_side(
        node_idx,
        degree,
        polycube_skeleton,
        polycube_mesh,
        &cutting_plan,
    );

    RegionParameterization {
        input_to_canonical,
        polycube_to_canonical,
    }
}

/// Parameterizes one side (input or polycube) of a region onto the canonical domain.
///
/// Uses the cutting plan to determine cut topology, then:
/// 1. Finds actual cut paths on this mesh surface
/// 2. Assembles the full disk boundary (boundary arcs + cut paths)
/// 3. Arc-length parameterizes boundary onto the 4(d-1)-gon (or square for d=1)
/// 4. Solves Dirichlet problem for interior vertices
///
/// Returns a map from vertex ID to 2D canonical-domain position.
fn parameterize_side(
    node_idx: NodeIndex,
    degree: usize,
    skeleton: &LabeledCurveSkeleton,
    mesh: &Mesh<INPUT>,
    cutting_plan: &CuttingPlan,
) -> HashMap<VertID, Vector2D> {
    if degree == 0 {
        warn!(
            "TODO: Degree 0 node {:?}, skipping parameterization",
            node_idx
        );
        return HashMap::new();
    }

    let patch_verts = &skeleton[node_idx].skeleton_node.patch_vertices;
    let patch_set: HashSet<VertID> = patch_verts.iter().copied().collect();

    // Build ordered boundary for each incident edge.
    let mut boundaries: HashMap<EdgeIndex, OrderedBoundary> = HashMap::new();
    for edge_ref in skeleton.edges(node_idx) {
        boundaries.insert(
            edge_ref.id(),
            ordered_boundary_on_our_side(&edge_ref.weight().boundary_loop, &patch_set, mesh),
        );
    }

    // Guard: if any boundary loop is empty, we can't parameterize this region.
    let has_empty = boundaries.iter().any(|(_, b)| b.vertices.is_empty());
    if has_empty {
        warn!(
            "Node {:?}: at least one boundary loop has 0 vertices, skipping parameterization.",
            node_idx
        );
        return HashMap::new();
    }

    if degree == 1 {
        let edge_direction = skeleton.edges(node_idx).next().unwrap().weight().direction;
        return parameterize_degree_one(patch_verts, &boundaries, edge_direction, mesh);
    }

    // Degree is at least 2 now
    //Find cut paths with enough distance separation between endpoints on the same boundary loop.
    let cut_paths = find_cut_paths(&boundaries, &patch_set, patch_verts, mesh, cutting_plan);
    if cut_paths.len() != cutting_plan.cuts.len() {
        warn!(
            "Node {:?}: expected {} cut paths but found {}",
            node_idx,
            cutting_plan.cuts.len(),
            cut_paths.len()
        );
    }

    // Build disk boundary via Euler tour over the cut tree.
    let segments = build_disk_boundary(&boundaries, &cut_paths);

    let n_expected = 4 * (degree - 1);
    if segments.len() != n_expected {
        warn!(
            "Node {:?}: disk boundary has {} segments, expected {}",
            node_idx,
            segments.len(),
            n_expected
        );
    }

    // Assign polygon positions to boundary/cut-path vertices.
    let (boundary_positions, cut_dual_values) =
        parameterize_disk_to_polygon(&segments, degree, mesh);

    // Compute UV positions for cross-boundary (other-patch) vertices.
    let cross_uvs = compute_cross_boundary_uvs(&boundaries, &boundary_positions);

    // Solve harmonic 2D with cut-aware boundary handling.
    let mut result = solve_harmonic_2d_with_cuts(
        patch_verts,
        &boundary_positions,
        &cut_dual_values,
        &cut_paths,
        &cross_uvs,
        mesh,
    );

    // Include cross-boundary UVs so that boundary polycube faces (which span two
    // regions) can be fully triangulated in canonical UV space.
    for (v, uv) in cross_uvs {
        result.entry(v).or_insert(uv);
    }
    result
}

/// For each cross-boundary vertex, interpolates a UV position from the UV values
/// of the two bracketing our-side boundary vertices.
fn compute_cross_boundary_uvs(
    boundaries: &HashMap<EdgeIndex, OrderedBoundary>,
    boundary_positions: &HashMap<VertID, Vector2D>,
) -> HashMap<VertID, Vector2D> {
    let mut result = HashMap::new();

    for boundary in boundaries.values() {
        let n = boundary.vertices.len();
        if n == 0 {
            continue;
        }

        // Build sorted anchors: (normalized_position, vertex).
        // cumulative is monotonically increasing by construction.
        let anchors: Vec<(f64, VertID)> = (0..n)
            .map(|i| {
                let t = if boundary.total_length > 0.0 {
                    boundary.cumulative[i] / boundary.total_length
                } else {
                    i as f64 / n as f64
                };
                (t, boundary.vertices[i])
            })
            .collect();

        for (ci, &cross_v) in boundary.cross_vertices.iter().enumerate() {
            if result.contains_key(&cross_v) {
                continue;
            }

            let t = boundary.cross_positions[ci];

            // Binary search for the insertion point in the sorted anchors.
            let pos = anchors.partition_point(|(at, _)| *at <= t);

            let (t_prev, v_prev) = if pos == 0 {
                anchors[n - 1] // wrap to last
            } else {
                anchors[pos - 1]
            };
            let (t_next, v_next) = if pos < n {
                anchors[pos]
            } else {
                anchors[0] // wrap to first
            };

            if let (Some(&uv_prev), Some(&uv_next)) = (
                boundary_positions.get(&v_prev),
                boundary_positions.get(&v_next),
            ) {
                // Cyclic interpolation fraction.
                let dt = {
                    let d = t_next - t_prev;
                    if d <= 0.0 {
                        d + 1.0
                    } else {
                        d
                    }
                };
                let d_cross = {
                    let d = t - t_prev;
                    if d < 0.0 {
                        d + 1.0
                    } else {
                        d
                    }
                };
                let frac = if dt > 0.0 {
                    (d_cross / dt).clamp(0.0, 1.0)
                } else {
                    0.0
                };

                let uv = Vector2D::new(
                    uv_prev.x * (1.0 - frac) + uv_next.x * frac,
                    uv_prev.y * (1.0 - frac) + uv_next.y * frac,
                );
                result.insert(cross_v, uv);
            }
        }
    }

    result
}

/// Returns two tangent-axis indices for a given principal direction.
/// The first tangent axis is used to pick a deterministic starting vertex.
fn tangent_axes(dir: PrincipalDirection) -> (usize, usize) {
    match dir {
        PrincipalDirection::X => (1, 2), // Y, Z
        PrincipalDirection::Y => (2, 0), // Z, X
        PrincipalDirection::Z => (0, 1), // X, Y
    }
}

/// For a region with a single boundary loop: split the boundary into 4 equal arcs,
/// map them to the 4 sides of a square, and solve Dirichlet problem for the interior.
///
/// `edge_direction` is the skeleton edge's principal axis, used to pick a consistent
/// starting vertex on both the input and polycube sides so their canonical-domain
/// orientations match.
fn parameterize_degree_one(
    patch_verts: &[VertID],
    boundaries: &HashMap<EdgeIndex, OrderedBoundary>,
    edge_direction: PrincipalDirection,
    mesh: &Mesh<INPUT>,
) -> HashMap<VertID, Vector2D> {
    let boundary = boundaries.values().next().unwrap();
    let n = boundary.vertices.len();
    if n < 4 {
        warn!(
            "Degree-1 boundary has only {} vertices, too few for square mapping",
            n
        );
        return HashMap::new();
    }

    // Both input and polycube sides use the same tangent-axis criterion so that
    // their canonical-domain orientations match.  The vertex with the maximum
    // coordinate along the first tangent axis is chosen as the start.
    let (axis_a, _) = tangent_axes(edge_direction);
    let start_idx = boundary
        .vertices
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            let pa = mesh.position(**a)[axis_a];
            let pb = mesh.position(**b)[axis_a];
            pa.partial_cmp(&pb).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap()
        .0;

    // Rotate cumulative arc-lengths so that start_idx has arc-length 0.
    let base_al = boundary.cumulative[start_idx];
    let rotated_cumulative: Vec<f64> = (0..n)
        .map(|k| {
            let i = (start_idx + k) % n;
            let al = boundary.cumulative[i] - base_al;
            if al < 0.0 {
                al + boundary.total_length
            } else {
                al
            }
        })
        .collect();

    // Rotated vertex order.
    let rotated_verts: Vec<VertID> = (0..n)
        .map(|k| boundary.vertices[(start_idx + k) % n])
        .collect();

    // Pick 4 corners at equal arc-length spacing from the start.
    let quarter = boundary.total_length / 4.0;
    let corners: Vec<usize> = (0..4)
        .map(|i| {
            let target = quarter * (i as f64);
            rotated_cumulative
                .iter()
                .enumerate()
                .min_by_key(|(_, &al)| OrderedFloat((al - target).abs()))
                .unwrap()
                .0
        })
        .collect();

    let polygon = polygon_corners(4);

    // Assign boundary positions by arc-length interpolation per segment.
    let mut boundary_positions: HashMap<VertID, Vector2D> = HashMap::new();
    for seg in 0..4 {
        let ci_start = corners[seg];
        let ci_end = corners[(seg + 1) % 4];
        let p0 = polygon[seg];
        let p1 = polygon[(seg + 1) % 4];

        let arc_start = rotated_cumulative[ci_start];
        let arc_end_raw = rotated_cumulative[ci_end];
        let seg_length = if ci_end > ci_start {
            arc_end_raw - arc_start
        } else {
            (boundary.total_length - arc_start) + arc_end_raw
        };

        let mut k = ci_start;
        loop {
            let al_adj = {
                let al = rotated_cumulative[k];
                let d = al - arc_start;
                if d < 0.0 {
                    d + boundary.total_length
                } else {
                    d
                }
            };
            let t = if seg_length > 0.0 {
                (al_adj / seg_length).clamp(0.0, 1.0)
            } else {
                0.0
            };
            boundary_positions.insert(
                rotated_verts[k],
                Vector2D::new(p0.x * (1.0 - t) + p1.x * t, p0.y * (1.0 - t) + p1.y * t),
            );
            if k == ci_end {
                break;
            }
            k = (k + 1) % n;
        }
    }

    // Cross-boundary UVs.
    let cross_uvs = compute_cross_boundary_uvs(boundaries, &boundary_positions);

    let mut result = solve_harmonic_2d(patch_verts, &boundary_positions, &cross_uvs, mesh);

    // Include cross-boundary UVs so that boundary polycube faces (which span two
    // regions) can be fully triangulated in canonical UV space.
    for (v, uv) in cross_uvs {
        result.entry(v).or_insert(uv);
    }
    result
}

/// A triangle in canonical UV space with associated 3D polycube positions.
struct UVTriangle {
    uv: [Vector2D; 3],
    pos: [Vector3D; 3],
}

/// Computes 2D barycentric coordinates of `p` in triangle `(a, b, c)`.
/// Returns `(u, v, w)` where `p = u*a + v*b + w*c`.
fn bary_2d(p: Vector2D, a: Vector2D, b: Vector2D, c: Vector2D) -> (f64, f64, f64) {
    let v0 = Vector2D::new(b.x - a.x, b.y - a.y);
    let v1 = Vector2D::new(c.x - a.x, c.y - a.y);
    let v2 = Vector2D::new(p.x - a.x, p.y - a.y);

    let d00 = v0.x * v0.x + v0.y * v0.y;
    let d01 = v0.x * v1.x + v0.y * v1.y;
    let d11 = v1.x * v1.x + v1.y * v1.y;
    let d20 = v2.x * v0.x + v2.y * v0.y;
    let d21 = v2.x * v1.x + v2.y * v1.y;

    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-15 {
        return (1.0, 0.0, 0.0);
    }
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;
    (u, v, w)
}

/// Small tolerance for point-in-triangle test.
const BARY_EPS: f64 = -1e-6;

/// Finds the polycube 3D position for a canonical UV point by locating it in
/// a set of triangulated polycube faces.
fn locate_in_triangles(uv: Vector2D, triangles: &[UVTriangle]) -> Option<Vector3D> {
    for tri in triangles {
        let (u, v, w) = bary_2d(uv, tri.uv[0], tri.uv[1], tri.uv[2]);
        if u >= BARY_EPS && v >= BARY_EPS && w >= BARY_EPS {
            return Some(tri.pos[0] * u + tri.pos[1] * v + tri.pos[2] * w);
        }
    }
    None
}

/// Fallback: project onto the nearest triangle by clamping barycentric coords.
fn nearest_triangle_fallback(uv: Vector2D, triangles: &[UVTriangle]) -> Option<Vector3D> {
    if triangles.is_empty() {
        return None;
    }

    let mut best_dist = f64::INFINITY;
    let mut best_pos = None;
    for tri in triangles {
        let (u, v, w) = bary_2d(uv, tri.uv[0], tri.uv[1], tri.uv[2]);
        // Clamp to triangle
        let (cu, cv, cw) = clamp_bary(u, v, w);
        let projected = Vector2D::new(
            cu * tri.uv[0].x + cv * tri.uv[1].x + cw * tri.uv[2].x,
            cu * tri.uv[0].y + cv * tri.uv[1].y + cw * tri.uv[2].y,
        );
        let dx = uv.x - projected.x;
        let dy = uv.y - projected.y;
        let dist = dx * dx + dy * dy;
        if dist < best_dist {
            best_dist = dist;
            best_pos = Some(tri.pos[0] * cu + tri.pos[1] * cv + tri.pos[2] * cw);
        }
    }
    best_pos
}

/// Clamps barycentric coordinates to the triangle, projecting to the nearest
/// point on the triangle boundary if outside.
fn clamp_bary(u: f64, v: f64, w: f64) -> (f64, f64, f64) {
    let cu = u.max(0.0);
    let cv = v.max(0.0);
    let cw = w.max(0.0);
    let sum = cu + cv + cw;
    if sum < 1e-15 {
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
    }
    (cu / sum, cv / sum, cw / sum)
}
