use std::collections::HashMap;

use itertools::Itertools;
use log::warn;
use mehsh::prelude::{HasPosition, HasVertices, Mesh, Vector2D, Vector3D};

use petgraph::graph::{EdgeIndex, NodeIndex};
use serde::{Deserialize, Serialize};

use crate::prelude::{EdgeID, VertID, INPUT};
use crate::skeleton::cross_parameterize::harmonic::solve_dirichlet;
use crate::skeleton::orthogonalize::LabeledCurveSkeleton;

mod cutting_plan;
mod harmonic;
pub mod virtual_mesh;

use cutting_plan::compute_cutting_plans;
use virtual_mesh::VirtualFlatGeometry;

/// Minimum desired separation between cut endpoints on the same boundary,
/// measured as a proportion of the boundary's total arc-length (i.e. in
/// `[0, 1]` normalized arc-length). Two cut endpoints on the same
/// boundary loop should ideally be at least this far apart. A warning is
/// emitted if this is violated (the cut is still valid, just potentially
/// lower quality for parameterization).
const MIN_CUT_BOUNDARY_PROPORTION: f64 = 0.05;

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
        input_skeleton: &LabeledCurveSkeleton,
        polycube_skeleton: &LabeledCurveSkeleton,
        polycube_mesh: &Mesh<INPUT>,
    ) -> Mesh<INPUT> {
        let mut result = input_mesh.clone();

        // TODO

        result
    }
}

/// A triangle in UV space with associated 3D positions at each vertex.
struct UvTriangle {
    uv: [Vector2D; 3],
    pos: [Vector3D; 3],
}

/// A quadrilater in UV space with associated 3D positions at each vertex.
struct UvQuad {
    uv: [Vector2D; 4],
    pos: [Vector3D; 4],
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
    );
    let (polycube_vfg, polycube_uv) = parameterize_side(
        patch_node_idx,
        degree,
        polycube_skeleton,
        polycube_mesh,
        &polycube_plan,
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
    let vfg = VirtualFlatGeometry::build(patch_node_idx, skeleton, mesh, cutting_plan);

    // Assign 2D positions to every node on the disk boundary.
    // The canonical polygon has n_sides sides:
    //   - degree 1 -> square (4 sides)
    //   - degree d >= 2 -> regular 4(d-1)-gon
    let n_sides = if degree == 1 { 4 } else { 4 * (degree - 1) };
    let boundary_positions = map_boundary_to_polygon(&vfg, n_sides);

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
    let uv_map = solve_dirichlet(&vfg, &boundary_positions);

    (vfg, uv_map)
}

/// Interpolates a position along the boundary of a regular polygon, from
/// vertex `start` to vertex `end` (going forward through intermediate
/// vertices). `t` in `[0, 1)` maps proportionally across the arc.
///
/// When `start == end` (zero-length arc), returns `polygon[start]`.
/// When the arc spans one edge, this reduces to simple linear interpolation.
fn polygon_arc_interpolate(polygon: &[Vector2D], start: usize, end: usize, t: f64) -> Vector2D {
    let n = polygon.len();
    let n_edges = (end + n - start) % n;
    if n_edges == 0 {
        return polygon[start];
    }
    let total = n_edges as f64;
    let pos = (t * total).clamp(0.0, total - 1e-12);
    let edge = pos as usize;
    let local_t = pos - edge as f64;
    let a = polygon[(start + edge) % n];
    let b = polygon[(start + edge + 1) % n];
    a * (1.0 - local_t) + b * local_t
}

/// Maps every node in `vfg.boundary_loop` to a 2D position on a regular `n_sides`-gon.
///
/// 
/// TODO...
/// 
/// When `corner_indices` is non-empty (degree ≥ 2), each consecutive pair of
/// corners defines a segment that maps to one polygon side. Nodes within a
/// segment are distributed by arc-length along that side. This ensures boundary
/// midpoints and cut vertices land on the correct polygon side.
///
/// When `corner_indices` is empty (degree 1), nodes are distributed by global
/// arc-length around the polygon.
///
/// The polygon has circumradius 1 with vertices at angles `2πk/n_sides`.
fn map_boundary_to_polygon(
    vfg: &VirtualFlatGeometry,
    n_sides: usize,
) -> HashMap<NodeIndex, Vector2D> {
    let boundary = &vfg.boundary_loop;
    let n = boundary.len();
    if n == 0 {
        return HashMap::new();
    }

    warn!("TODO!!!");

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

    // // Fallback: global arc-length distribution (degree 1, no corners).
    // let mut seg_lengths: Vec<f64> = vec![0.0; n];
    // let mut total: f64 = 0.0;
    // for i in 0..n {
    //     let a = vfg.graph[boundary[i]].position;
    //     let b = vfg.graph[boundary[(i + 1) % n]].position;
    //     let len = (b - a).norm();
    //     seg_lengths[i] = len;
    //     total += len;
    // }

    // let mut result = HashMap::new();
    // let mut cumulative: f64 = 0.0;

    // for (i, &node) in boundary.iter().enumerate() {
    //     let t = cumulative / total; // in [0, 1)

    //     let frac = t * n_sides as f64;
    //     let side = (frac as usize).min(n_sides - 1);
    //     let local_t = frac - side as f64;

    //     let p0 = polygon[side];
    //     let p1 = polygon[(side + 1) % n_sides];
    //     let pos = p0 * (1.0 - local_t) + p1 * local_t;

    //     result.insert(node, pos);
    //     cumulative += seg_lengths[i];
    // }

    // result
    todo!()
}
