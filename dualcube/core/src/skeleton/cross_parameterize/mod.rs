use std::collections::HashMap;
use std::f64::consts::{FRAC_PI_2, TAU};

use itertools::Itertools;
use log::{error, warn};
use mehsh::prelude::{HasPosition, HasVertices, Mesh, Vector2D, Vector3D};

use petgraph::graph::{EdgeIndex, NodeIndex};
use serde::{Deserialize, Serialize};

use crate::prelude::{EdgeID, VertID, INPUT};
use crate::skeleton::cross_parameterize::harmonic::solve_dirichlet;
use crate::skeleton::cross_parameterize::virtual_mesh::VirtualNodeOrigin;
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
    // duplicating vertices and edges along the cut so the result is a topological disk.
    let vfg = VirtualFlatGeometry::build(patch_node_idx, skeleton, mesh, cutting_plan, is_tri_mesh);

    // Assign 2D positions to every node on the disk boundary.
    let boundary_positions = map_boundary_to_polygon(&vfg, degree);

    // Solve the Dirichlet problem on the VFG graph.
    let uv_map = solve_dirichlet(&vfg, &boundary_positions);

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
///
/// Degree parameter is the degree of the region being parameterized in the skeleton.
fn map_boundary_to_polygon(
    vfg: &VirtualFlatGeometry,
    degree: usize,
) -> HashMap<NodeIndex, Vector2D> {
    let boundary = &vfg.boundary_loop;

    if degree == 1 {
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

    let origin_first_boundary_node = &vfg.graph[boundary[0]].origin;
    let origin_second_boundary_node = &vfg.graph[boundary[1]].origin;
    // Note that this is not strictly necessary as a result from the other methods, but it is always the case now and simplifies logic here so we just assume it.
    // The assumption is that the first node is always the start of some cut, and the second is along that cut.
    match (origin_first_boundary_node, origin_second_boundary_node) {
        (
            VirtualNodeOrigin::CutEndpointMidpointDuplicate { .. },
            VirtualNodeOrigin::CutDuplicate { .. },
        ) => {
            // This is the expected case: the boundary starts with a cut endpoint, so we can alternate cut/non-cut segments as expected.
        }
        (
            VirtualNodeOrigin::CutEndpointMidpointDuplicate {
                cut_index: cut1, ..
            },
            VirtualNodeOrigin::CutEndpointMidpointDuplicate {
                cut_index: cut2, ..
            },
        ) => {
            // Cut is very short. We check that the endpoints are from the same cut.
            if cut1 != cut2 {
                panic!(
                    "Boundary loop starts with cut endpoints from different cuts: {:?} and {:?}",
                    origin_first_boundary_node, origin_second_boundary_node
                );
            }
        }
        _ => {
            panic!(
                "Unexpected boundary start node types: {:?} and {:?}",
                origin_first_boundary_node, origin_second_boundary_node
            );
        }
    }

    // Because our starting pattern is always set, we know that the order in which we traverse the segments is also set.
    // We can traverse the boundary in order and we will see all segments in the correct order.

    let n_sides = 4 * (degree - 1);
    let mut result = HashMap::new();

    let mut currently_building = CurrentlyBuildingSegment::Cut;
    let mut segment_index = 0;
    let mut buf = vec![boundary[0]];

    for &idx in &boundary[1..] {
        let node = &vfg.graph[idx];
        let origin = &node.origin;
        buf.push(idx);

        match (currently_building, origin) {
            (CurrentlyBuildingSegment::Cut, VirtualNodeOrigin::CutDuplicate { .. }) => {
                // Continue cut segment.
            }
            (
                CurrentlyBuildingSegment::Cut,
                VirtualNodeOrigin::CutEndpointMidpointDuplicate { .. },
            ) => {
                // End the current cut segment.
                parameterize_segment(
                    segment_index,
                    n_sides,
                    &buf,
                    vfg,
                    &mut result,
                );

                // Flush buffer and start new for next segment (Non-Cut).
                buf = vec![idx];
                segment_index += 1;
                currently_building = CurrentlyBuildingSegment::NonCut;
            }
            (CurrentlyBuildingSegment::NonCut, VirtualNodeOrigin::BoundaryMidpoint { .. }) => {
                // Continue building non-cut segment.
            }
            (
                CurrentlyBuildingSegment::NonCut,
                VirtualNodeOrigin::CutEndpointMidpointDuplicate { .. },
            ) => {
                // Non-cut segment ended, now cut starts.
                parameterize_segment(
                    segment_index,
                    n_sides,
                    &buf,
                    vfg,
                    &mut result,
                );

                buf = vec![idx];
                segment_index += 1;
                currently_building = CurrentlyBuildingSegment::Cut;
            }
            _ => {
                panic!(
                    "Unexpected boundary node type {:?} while building segment {} of type {:?}",
                    origin, segment_index, currently_building
                );
            }
        }
    }

    // Last segment always ends back at the start of the first segment.
    buf.push(boundary[0]);
    parameterize_segment(
        segment_index,
        n_sides,
        &buf,
        vfg,
        &mut result,
    );
    segment_index += 1;

    if segment_index != n_sides {
        panic!(
            "Boundary loop tracing failed: expected {} segments for degree {}, but found {}. Boundary len: {}",
            n_sides, degree, segment_index, boundary.len()
        );
    }

    result
}

fn parameterize_segment(
    segment_index: usize,
    n_sides: usize,
    node_indices: &[NodeIndex],
    vfg: &VirtualFlatGeometry,
    result: &mut HashMap<NodeIndex, Vector2D>,
) {
    // Calculate arc-length along the segment nodes.
    let mut lengths = Vec::with_capacity(node_indices.len());
    lengths.push(0.0);
    let mut total_length = 0.0;

    for window in node_indices.windows(2) {
        let a = vfg.graph[window[0]].position;
        let b = vfg.graph[window[1]].position;
        let len = (b - a).norm();
        total_length += len;
        lengths.push(total_length);
    }

    for (i, &idx) in node_indices.iter().enumerate() {
        let t = if total_length > 0.0 {
            lengths[i] / total_length
        } else {
            // Distribute evenly if length is 0 (e.g. coincident endpoints).
            if node_indices.len() > 1 {
                i as f64 / (node_indices.len() - 1) as f64
            } else {
                0.0
            }
        };

        let pos = polygon_point(n_sides, segment_index, t);

        // Check for consistency if the node was already parameterized (corners).
        if let Some(existing) = result.get(&idx) {
            let dist = (pos - *existing).norm();
            if dist > 1e-9 {
                panic!(
                    "Inconsistent UV assignment for node {:?} at segment {}: new {:?} vs existing {:?} (dist={})",
                    idx, segment_index, pos, existing, dist
                );
            }
        }

        result.insert(idx, pos);
    }
}

#[derive(Debug, Clone, Copy)]
enum CurrentlyBuildingSegment {
    Cut,
    NonCut,
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
