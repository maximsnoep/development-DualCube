use std::collections::HashMap;
use std::f64::consts::{FRAC_PI_2, TAU};

use itertools::Itertools;
use log::{error, info, warn};
use mehsh::mesh::elem::edge;
use mehsh::prelude::{HasPosition, HasVertices, Mesh, Vector2D, Vector3D};

use petgraph::graph::{EdgeIndex, NodeIndex};
use serde::{Deserialize, Serialize};

use crate::prelude::{EdgeID, VertID, INPUT};
// use crate::skeleton::cross_parameterize::harmonic::solve_dirichlet;
use crate::skeleton::orthogonalize::LabeledCurveSkeleton;

mod boundary_walk;
mod cutting_plan;
mod duplicate_cut_vertices;
mod harmonic;
mod internal_edges;
pub mod virtual_mesh;

use cutting_plan::compute_cutting_plans;
use virtual_mesh::VirtualFlatGeometry;

/// Minimum desired separation between cut endpoints on the same boundary,
/// measured as a proportion of the boundary's total arc-length (i.e. in
/// `[0, 1]` normalized arc-length). Two cut endpoints on the same
/// boundary loop should ideally be at least this far apart. A warning is
/// emitted if this is violated (the cut is still valid, just potentially
/// lower quality for parameterization).
const _MIN_CUT_BOUNDARY_PROPORTION: f64 = 0.05;

/// A path across the mesh surface.
///
/// Consecutive points should share a triangle face.
/// Path starts at an edge midpoint (on some boundary), then traverses using vertices,
/// and ends at another edge midpoint (on another boundary).  The path is simple and does not
/// self-intersect, but may touch the same vertex multiple times.
#[derive(Debug, Clone)]
pub struct SurfacePath {
    pub start: EdgeID,
    pub interior_verts: Vec<VertID>,
    pub end: EdgeID,
}

impl SurfacePath {
    /// Converts this surface path to a sequence of 3D positions (e.g. for visualisation).
    pub fn to_positions(&self, mesh: &Mesh<INPUT>) -> Vec<Vector3D> {
        let mut positions = Vec::new();
        // Start with the start edge midpoint.
        positions.push(edge_id_to_midpoint_pos(self.start, mesh));

        // Then add all interior vertices.
        for vert_id in &self.interior_verts {
            positions.push(mesh.position(*vert_id));
        }

        // End with the end edge midpoint.
        positions.push(edge_id_to_midpoint_pos(self.end, mesh));

        positions
    }
}

/// A single cut connecting two boundary loops, with the exact surface path.
#[derive(Debug, Clone)]
pub struct CutPath {
    /// The boundary loop (skeleton edge) where this cut starts.
    pub start_boundary: EdgeIndex,

    /// The boundary loop (skeleton edge) where this cut ends.
    pub end_boundary: EdgeIndex,

    /// The actual path across the surface, from start boundary to end boundary.
    pub path: SurfacePath,
}

/// Complete cutting plan for one side of a region, including boundary
/// parameterizations and the exact cut paths.
///
/// For degree `d >= 2`, contains `d − 1` cuts forming a spanning tree over the
/// `d` boundary loops.  For degree 0 or 1, `cuts` is empty.
#[derive(Debug, Clone)]
pub struct CuttingPlan {
    /// The `d − 1` cut paths connecting boundary loops.
    pub cuts: Vec<CutPath>,
}

/// The parameterization of a single region (skeleton node) onto a canonical 2D domain.
///
/// Both the input mesh region and the polycube mesh region are mapped to the same
/// canonical domain. The composition of these two maps gives the bijection
/// between the input and polycube surfaces for this region.
///
/// The canonical domains are:
/// - degree 0: a sphere (??? TODO)
/// - degree 1: a square, the single boundary maps to the entire boundary of the square
/// - degree 2+: a regular 4(d-1)-gon for degree d.
///
/// UV coordinates are keyed by `VirtualFlatGeometry` node indices. Cut vertices
/// appear twice in the VFG (one per side of the cut) and therefore have two distinct
/// UV positions. Interior and non-cut boundary vertices appear once.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionParameterization {
    /// Virtual flat geometry for the input mesh side (cut-open disk representation).
    pub input_vfg: VirtualFlatGeometry,

    /// Per-virtual-node canonical-domain UV for the input mesh side.
    /// NOTE: keys are in VFG.
    pub input_uv: HashMap<NodeIndex, Vector2D>,

    /// Virtual flat geometry for the polycube mesh side.
    pub polycube_vfg: VirtualFlatGeometry,

    /// Per-virtual-node canonical-domain UV for the polycube mesh side.
    /// NOTE: keys are in VFG.
    pub polycube_uv: HashMap<NodeIndex, Vector2D>,

    /// The cut paths used to parameterize the input mesh side, stored only for visualisation.
    pub input_cuts: Vec<Vec<Vector3D>>,

    /// The cut paths used to parameterize the polycube mesh side, stored only for visualisation.
    pub polycube_cuts: Vec<Vec<Vector3D>>,
}

/// Calculates the midpoint position of a boundary edge, given its `EdgeID` and the mesh.
pub fn edge_id_to_midpoint_pos(edge_idx: EdgeID, mesh: &Mesh<INPUT>) -> Vector3D {
    let (v1, v2) = mesh
        .vertices(edge_idx)
        .collect_tuple()
        .expect("Expected boundary edge to have exactly two vertices");

    // Compute the midpoint position
    let pos1 = mesh.position(v1);
    let pos2 = mesh.position(v2);
    let midpoint_pos = (pos1 + pos2) / 2.0;
    midpoint_pos
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

impl PolycubeMap {
    /// Constructs a `Mesh<INPUT>` whose vertices are the input mesh vertices repositioned
    /// onto the polycube surface, for the `triangle_mesh_polycube` needed by `Quad`.
    ///
    /// For each input vertex, looks up its canonical-domain coordinates from the input
    /// parameterization, then finds the corresponding polycube-surface position by
    /// interpolating within the polycube parameterization of the same region.
    pub fn to_triangle_mesh_polycube(
        &self,
        input_mesh: &Mesh<INPUT>,
        _input_skeleton: &LabeledCurveSkeleton,
        _polycube_skeleton: &LabeledCurveSkeleton,
        _polycube_mesh: &Mesh<INPUT>,
    ) -> Mesh<INPUT> {
        let result = input_mesh.clone();

        // TODO

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
    patch_node_idx: NodeIndex,
    degree: usize,
    input_skeleton: &LabeledCurveSkeleton,
    polycube_skeleton: &LabeledCurveSkeleton,
    input_mesh: &Mesh<INPUT>,
    polycube_mesh: &Mesh<INPUT>,
) -> RegionParameterization {
    // Compute cutting plans for both sides. Cut paths are found independently on each side, then boundary
    // parameterizations are built so that cut endpoints share t-values.
    let (input_plan, polycube_plan) = compute_cutting_plans(
        patch_node_idx,
        input_skeleton,
        polycube_skeleton,
        input_mesh,
        polycube_mesh,
    );

    // Finalize: save cut positions for visualisation.
    let input_cuts: Vec<Vec<Vector3D>> = input_plan
        .cuts
        .iter()
        .map(|cut| cut.path.to_positions(input_mesh))
        .collect();
    let polycube_cuts: Vec<Vec<Vector3D>> = polycube_plan
        .cuts
        .iter()
        .map(|cut| cut.path.to_positions(polycube_mesh))
        .collect();

    // Build VFG and parameterize each side using its cutting plan.
    let (input_vfg, input_uv) = parameterize_side(
        patch_node_idx,
        degree,
        input_skeleton,
        input_mesh,
        &input_plan,
        true, // base mesh is strict tri, but in cuts we can introduce quads.
    );
    let (polycube_vfg, polycube_uv) = parameterize_side(
        patch_node_idx,
        degree,
        polycube_skeleton,
        polycube_mesh,
        &polycube_plan,
        false, // base mesh is quad, cuts split quads into quads.
    );

    RegionParameterization {
        input_vfg,
        input_uv,
        polycube_vfg,
        polycube_uv,
        input_cuts,
        polycube_cuts,
    }
}

/// Parameterizes one side (input or polycube) of a region onto the canonical domain, by
/// building virtual geometry for the cut-open disk and solving the Dirichlet problem with fixed boundary positions.
///
/// Returns `(vfg, uv_map)` where `uv_map` maps every VFG node index to its
/// 2D canonical-domain position. Cut positions for visualisation are extracted
/// by the caller before this function is called.
fn parameterize_side(
    patch_node_idx: NodeIndex,
    degree: usize,
    skeleton: &LabeledCurveSkeleton,
    mesh: &Mesh<INPUT>,
    cutting_plan: &CuttingPlan,
    is_tri_mesh: bool,
) -> (VirtualFlatGeometry, HashMap<NodeIndex, Vector2D>) {
    if degree == 0 {
        warn!(
            "TODO: Degree 0 node {:?}, skipping parameterization",
            patch_node_idx
        );
        return (VirtualFlatGeometry::empty(), HashMap::new());
    }

    // Build virtual geometry by cutting the mesh open along cut paths,
    // duplicating vertices so the result is a topological disk.
    let vfg = VirtualFlatGeometry::build(patch_node_idx, skeleton, mesh, cutting_plan, is_tri_mesh);

    // Assign 2D positions to every node on the disk boundary.
    // The canonical polygon has n_sides sides:
    //   - degree 1 -> square (4 sides)
    //   - degree d >= 2 -> regular 4(d-1)-gon
    let n_sides = if degree == 1 { 4 } else { 4 * (degree - 1) };
    // let boundary_positions = map_boundary_to_polygon(&vfg, n_sides);

    // TODO temp debug
    if degree >= 2 {
        log::info!(
            "parameterize_side: region {:?} degree {}, n_sides={}, boundary_loop={}",
            patch_node_idx,
            degree,
            n_sides,
            vfg.boundary_loop.len(),
        );
    }

    // Solve the Dirichlet problem on the VFG graph.
    // let uv_map = solve_dirichlet(&vfg, &boundary_positions);
    let uv_map = HashMap::new(); // TODO

    (vfg, uv_map)
}

/// Maps every node in `vfg.boundary_loop` to a 2D position on a regular `n_sides`-gon,
///  where for a region with degree we have `n_sides` = 4 for degree 1, and `n_sides` = 4(d-1) for degree d >= 2.
///
/// For a boundary without cuts, we simply pick 4 points to be corners, and do arc-length parameterization within each of the 4 segments.
///
/// When there are cuts, each segment of the polygon corresponds to a part of the boundary loop.
/// The boundary alternates between cut segments (between cut endpoints) and non-cut segments (though possibly empty when cut endpoints are adjacent).
///
/// The polygon has circumradius 1 with corners at angles `2*pi*k/n_sides` for k'th corner.
fn map_boundary_to_polygon(vfg: &VirtualFlatGeometry) -> HashMap<NodeIndex, Vector2D> {
    let boundary = &vfg.boundary_loop;

    // Traverse the boundary to get the number of different sides
    // Note that it is possible for there to be 0 vertices in a segment, as CutEndpoints can connect directly.
    let segments = 1;

    // TODO: traverse boundary

    info!(
        "{:?} boundary segments (cut/non-cut alternations)",
        segments
    );

    if segments == 1 {
        // No cuts: simple case. Just pick 4 corners and do arc-length parameterization within each of the 4 segments.
        if boundary.len() < 4 {
            // TODO: catch this upstream...
            panic!("Boundary has fewer than 4 vertices, cannot map to polygon with 4 sides");
        }

        // Calculate total length so we can parameterize.
        let mut total_length: f64 = 0.0;
        let mut edge_lengths: Vec<f64> = vec![0.0; boundary.len()];
        for i in 0..boundary.len() {
            let a = vfg.graph[boundary[i]].position;
            let b = vfg.graph[boundary[(i + 1) % boundary.len()]].position;
            let len = (b - a).norm();
            edge_lengths[i] = len;
            total_length += len;
        }

        // We pick index 0 to be the first corner, then place the next 3 corners at cumulative
        // arc-lengths of 1/4, 2/4, and 3/4 around the loop. We enforce strictly increasing indices
        // to guarantee 4 distinct corners, even if edge lengths are disproportionately large.
        let mut corner_indices = Vec::with_capacity(4);
        corner_indices.push(0);

        let mut current_idx = 0;
        let mut cumulative_length = 0.0;

        for k in 1..4 {
            let target_length = total_length * (k as f64 / 4.0);

            // The maximum index we can pick and still leave enough vertices for the remaining corners.
            let max_idx = boundary.len() - (4 - k);

            // Advance by at least one vertex to ensure corners are distinct.
            current_idx += 1;
            cumulative_length += edge_lengths[current_idx - 1];

            // Continue advancing until we reach the target length or the maximum allowed index.
            while current_idx < max_idx && cumulative_length < target_length {
                cumulative_length += edge_lengths[current_idx];
                current_idx += 1;
            }

            corner_indices.push(current_idx);
        }

        // Do arc-length parameterization within each of the 4 segments between corners.
        let mut result = HashMap::new();

        // Process each of the 4 segments formed by the corners.
        for s in 0..4 {
            let start_idx = corner_indices[s];
            let end_idx = corner_indices[(s + 1) % 4];

            // Calculate the total arc-length of the current segment.
            let mut segment_length = 0.0;
            let mut curr = start_idx;
            while curr != end_idx {
                segment_length += edge_lengths[curr];
                curr = (curr + 1) % boundary.len();
            }

            // Map each vertex in this segment to a coordinate on the polygon.
            let mut current_segment_len = 0.0;
            let mut curr = start_idx;
            while curr != end_idx {
                // Prevent division by zero if a segment has 0 length somehow.
                let t = if segment_length > 0.0 {
                    current_segment_len / segment_length
                } else {
                    error!("Segment {} of boundary has zero length, cannot parameterize! Assigning t=0 for all vertices in this segment.", s);
                    0.0
                };

                result.insert(boundary[curr], polygon_point(4, s, t));

                current_segment_len += edge_lengths[curr];
                curr = (curr + 1) % boundary.len();
            }
        }

        return result;
    }

    // Since cuts have at least their two endpoints, we do per segment (alternatingly as the boundary alternates between cut and non-cut):
    // - Per cut-segment: parameterize by arc-length, to [0, 1], so putting the cut endpoints at 0 and 1, and midpoints in between.
    // - Per non-cut segment: parameterize by arc-length, to (0, 1), so not putting any nodes at the corners.

    // Regular polygon vertices.
    // let polygon: Vec<Vector2D> = (0..n_sides)
    //     .map(|k| {
    //         let angle = 2.0 * PI * k as f64 / n_sides as f64;
    //         Vector2D::new(angle.cos(), angle.sin())
    //     })
    //     .collect();

    // let corners = &vfg.corner_indices;

    // if !corners.is_empty() && corners.len() != n_sides {
    //     warn!(
    //         "map_boundary_to_polygon: corner count {} != n_sides {}, falling back to arc-length",
    //         corners.len(),
    //         n_sides,
    //     );
    // }

    // if !corners.is_empty() && corners.len() == n_sides {
    //     // Structured mapping: each segment of boundary nodes maps to one polygon side.
    //     // When adjacent cut endpoints produce empty segments (seg_len == 0), we
    //     // absorb those empty polygon sides into the next non-empty segment so
    //     // that its nodes span a wider polygon arc and no gap is left.
    //     let mut result = HashMap::new();

    //     // Pre-compute segment lengths for all sides.
    //     let seg_lens: Vec<usize> = (0..n_sides)
    //         .map(|side| {
    //             let seg_start = corners[side];
    //             let seg_end = corners[(side + 1) % n_sides];
    //             if seg_end > seg_start {
    //                 seg_end - seg_start
    //             } else if seg_end == seg_start {
    //                 0
    //             } else {
    //                 n - seg_start + seg_end
    //             }
    //         })
    //         .collect();

    //     for side in 0..n_sides {
    //         if seg_lens[side] == 0 {
    //             continue; // Empty side — absorbed by the next non-empty side.
    //         }

    //         // Walk backward through preceding empty sides to find where the
    //         // effective polygon arc starts.
    //         let mut effective_start = side;
    //         loop {
    //             let prev = (effective_start + n_sides - 1) % n_sides;
    //             if seg_lens[prev] != 0 || prev == side {
    //                 break;
    //             }
    //             effective_start = prev;
    //         }
    //         let effective_end = (side + 1) % n_sides;

    //         let seg_start = corners[side];
    //         let seg_len = seg_lens[side];

    //         // Compute arc lengths within this segment.
    //         let mut seg_arc: Vec<f64> = Vec::with_capacity(seg_len);
    //         let mut seg_total: f64 = 0.0;
    //         for j in 0..seg_len {
    //             let idx_a = (seg_start + j) % n;
    //             let idx_b = (seg_start + j + 1) % n;
    //             let a = vfg.graph[boundary[idx_a]].position;
    //             let b = vfg.graph[boundary[idx_b]].position;
    //             let len = (b - a).norm();
    //             seg_arc.push(len);
    //             seg_total += len;
    //         }

    //         // Place nodes along the (possibly extended) polygon arc.
    //         let mut cumulative: f64 = 0.0;
    //         for j in 0..seg_len {
    //             let idx = (seg_start + j) % n;
    //             let node = boundary[idx];
    //             let t = if seg_total > 1e-15 {
    //                 cumulative / seg_total
    //             } else {
    //                 j as f64 / seg_len as f64
    //             };
    //             let pos = polygon_arc_interpolate(&polygon, effective_start, effective_end, t);
    //             result.insert(node, pos);
    //             cumulative += seg_arc[j];
    //         }
    //     }

    //     // Verify all boundary nodes got assigned a position.
    //     let missing = boundary.iter().filter(|n| !result.contains_key(n)).count();
    //     if missing > 0 {
    //         warn!(
    //             "map_boundary_to_polygon: {} of {} boundary nodes missing after corner mapping!",
    //             missing, n,
    //         );
    //     }

    //     return result;
    // }

    todo!()
}

/// Evaluates a 2D position along the boundary of a regular polygon.
///
/// The polygon has `n` vertices, is inscribed within a unit circle, and has its initial vertex located exactly at the top (0.0, 1.0).
/// Vertices are generated in a counter-clockwise direction.
pub fn polygon_point(n: usize, segment: usize, t: f64) -> Vector2D {
    assert!(n >= 3, "A polygon must have at least 3 vertices.");
    assert!(
        segment < n,
        "Segment index must be in the range [0, n-1], got {} for n={}",
        segment,
        n
    );

    let n_f64 = n as f64;
    let s_f64 = segment as f64;

    // Calculate the angles for the start and end of the segment.
    let angle_start = FRAC_PI_2 + (s_f64 * TAU / n_f64);
    let angle_end = FRAC_PI_2 + ((s_f64 + 1.0) * TAU / n_f64);

    let start_x = angle_start.cos();
    let start_y = angle_start.sin();

    let end_x = angle_end.cos();
    let end_y = angle_end.sin();

    // Linearly interpolate between the start and end vertices.
    let x = (1.0 - t) * start_x + t * end_x;
    let y = (1.0 - t) * start_y + t * end_y;

    Vector2D::new(x, y)
}
