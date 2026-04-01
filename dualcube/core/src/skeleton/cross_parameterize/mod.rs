use std::collections::HashMap;
use std::f64::consts::{FRAC_PI_2, TAU};

use itertools::Itertools;
use log::{error, warn};
use mehsh::prelude::{HasPosition, HasVertices, Mesh, SetPosition, Vector2D, Vector3D};

use petgraph::graph::{EdgeIndex, NodeIndex};
use serde::{Deserialize, Serialize};

use crate::prelude::{EdgeID, VertID, INPUT};
use crate::skeleton::cross_parameterize::harmonic::solve_dirichlet;
use crate::skeleton::cross_parameterize::virtual_mesh::{VirtualNode, VirtualNodeOrigin};
use crate::skeleton::orthogonalize::LabeledCurveSkeleton;

use bvh::aabb::{Aabb, Bounded};
use bvh::bounding_hierarchy::BHShape;
use bvh::bvh::Bvh;

mod boundary_walk;
pub mod coordination;
mod cutting_plan;
mod duplicate_cut_vertices;
mod harmonic;
mod internal_edges;
pub mod virtual_mesh;

use coordination::{CutCycleOrder, RegionCoordination};
use cutting_plan::compute_region_coordination;
use virtual_mesh::VirtualFlatGeometry;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UvFace {
    Tri {
        uvs: [Vector2D; 3],
        positions: [Vector3D; 3],
        real_index: usize,
    },
    Quad {
        uvs: [Vector2D; 4],
        positions: [Vector3D; 4],
        real_index: usize,
    },
}

impl UvFace {
    pub fn interpolate(&self, uv: Vector2D) -> Option<Vector3D> {
        match self {
            UvFace::Tri { uvs, positions, .. } => {
                let bc = barycentric_2d(uv, uvs[0], uvs[1], uvs[2]);
                if bc.iter().all(|&c| c >= -1e-9) {
                    Some(positions[0] * bc[0] + positions[1] * bc[1] + positions[2] * bc[2])
                } else {
                    None
                }
            }
            UvFace::Quad { uvs, positions, .. } => {
                // Split into two tris: (0,1,2) and (0,2,3)
                if let Some(pos) = interpolate_tri_if_inside(
                    uv,
                    uvs[0],
                    uvs[1],
                    uvs[2],
                    positions[0],
                    positions[1],
                    positions[2],
                ) {
                    Some(pos)
                } else {
                    interpolate_tri_if_inside(
                        uv,
                        uvs[0],
                        uvs[2],
                        uvs[3],
                        positions[0],
                        positions[2],
                        positions[3],
                    )
                }
            }
        }
    }
}

fn interpolate_tri_if_inside(
    uv: Vector2D,
    a: Vector2D,
    b: Vector2D,
    c: Vector2D,
    pa: Vector3D,
    pb: Vector3D,
    pc: Vector3D,
) -> Option<Vector3D> {
    let bc = barycentric_2d(uv, a, b, c);
    if bc.iter().all(|&c| c >= -1e-4) {
        Some(pa * bc[0] + pb * bc[1] + pc * bc[2])
    } else {
        None
    }
}

fn barycentric_2d(p: Vector2D, a: Vector2D, b: Vector2D, c: Vector2D) -> [f64; 3] {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;

    // 2D cross product: v.x * w.y - v.y * w.x
    let denom = v0.x * v1.y - v0.y * v1.x;

    if denom.abs() < 1e-12 {
        return [-1.0, -1.0, -1.0]; // Degenerate triangle, reject
    }

    let v = (v2.x * v1.y - v2.y * v1.x) / denom;
    let w = (v0.x * v2.y - v0.y * v2.x) / denom;
    let u = 1.0 - v - w;
    [u, v, w]
}

impl Bounded<f64, 3> for UvFace {
    fn aabb(&self) -> Aabb<f64, 3> {
        match self {
            UvFace::Tri { uvs, .. } => {
                let p0 = nalgebra::Point3::new(uvs[0].x, uvs[0].y, 0.0);
                let p1 = nalgebra::Point3::new(uvs[1].x, uvs[1].y, 0.0);
                let p2 = nalgebra::Point3::new(uvs[2].x, uvs[2].y, 0.0);
                Aabb::empty().grow(&p0).grow(&p1).grow(&p2)
            }
            UvFace::Quad { uvs, .. } => {
                let p0 = nalgebra::Point3::new(uvs[0].x, uvs[0].y, 0.0);
                let p1 = nalgebra::Point3::new(uvs[1].x, uvs[1].y, 0.0);
                let p2 = nalgebra::Point3::new(uvs[2].x, uvs[2].y, 0.0);
                let p3 = nalgebra::Point3::new(uvs[3].x, uvs[3].y, 0.0);
                Aabb::empty().grow(&p0).grow(&p1).grow(&p2).grow(&p3)
            }
        }
    }
}

impl BHShape<f64, 3> for UvFace {
    fn set_bh_node_index(&mut self, _index: usize) {}
    fn bh_node_index(&self) -> usize {
        0
    }
}

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

    /// Extracted faces for the input side.
    #[serde(skip)]
    pub input_faces: Vec<UvFace>,

    /// BVH for the input side UV domain.
    #[serde(skip)]
    pub input_bvh: Option<Bvh<f64, 3>>,

    /// Extracted faces for the polycube side.
    #[serde(skip)]
    pub polycube_faces: Vec<UvFace>,

    /// BVH for the polycube side UV domain.
    #[serde(skip)]
    pub polycube_bvh: Option<Bvh<f64, 3>>,
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
        let mut result = input_mesh.clone();

        // Map input VertID to region and VFG NodeIndex
        let mut vert_to_region = HashMap::new();
        for (region_idx, region_param) in &self.regions {
            for vfg_node_idx in region_param.input_vfg.graph.node_indices() {
                let origin = &region_param.input_vfg.graph[vfg_node_idx].origin;
                match origin {
                    VirtualNodeOrigin::MeshVertex(v) => {
                        vert_to_region.insert(*v, (*region_idx, vfg_node_idx));
                    }
                    VirtualNodeOrigin::CutDuplicate { original, .. } => {
                        vert_to_region.insert(*original, (*region_idx, vfg_node_idx));
                    }
                    _ => {}
                }
            }
        }

        for vert_id in input_mesh.vert_ids() {
            if let Some(&(region_idx, vfg_node_idx)) = vert_to_region.get(&vert_id) {
                let region = &self.regions[&region_idx];
                if let Some(uv) = region.input_uv.get(&vfg_node_idx) {
                    if let (Some(bvh), faces) = (&region.polycube_bvh, &region.polycube_faces) {
                        let epsilon = 1e-4;
                        let query_min =
                            nalgebra::Point3::new(uv.x - epsilon, uv.y - epsilon, -epsilon);
                        let query_max =
                            nalgebra::Point3::new(uv.x + epsilon, uv.y + epsilon, epsilon);
                        let query = Aabb::with_bounds(query_min, query_max);
                        let candidates = bvh.traverse(&query, faces);

                        let mut mapped_pos = None;
                        for face in &candidates {
                            if let Some(pos) = face.interpolate(*uv) {
                                mapped_pos = Some(pos);
                                break;
                            }
                        }

                        if let Some(pos) = mapped_pos {
                            result.set_position(vert_id, pos);
                        } else {
                            error!("UV {:?} for vertex {:?} not found in polycube BVH candidates ({} candidates) for region {:?}", uv, vert_id, candidates.len(), region_idx);
                        }
                    }
                }
            }
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
    patch_node_idx: NodeIndex,
    degree: usize,
    input_skeleton: &LabeledCurveSkeleton,
    polycube_skeleton: &LabeledCurveSkeleton,
    input_mesh: &Mesh<INPUT>,
    polycube_mesh: &Mesh<INPUT>,
) -> RegionParameterization {
    // Compute shared coordination: polycube-first slot assignment with input tie-break.
    let coordination = compute_region_coordination(
        patch_node_idx,
        input_skeleton,
        polycube_skeleton,
        input_mesh,
        polycube_mesh,
    );

    // Save cut positions for visualisation.
    let input_cuts: Vec<Vec<Vector3D>> = coordination
        .input_cuts
        .iter()
        .map(|cut| cut.path.to_positions(input_mesh))
        .collect();
    let polycube_cuts: Vec<Vec<Vector3D>> = coordination
        .polycube_cuts
        .iter()
        .map(|cut| cut.path.to_positions(polycube_mesh))
        .collect();

    // Build VFG and parameterize each side using the shared coordination.
    let (input_vfg, input_uv) = parameterize_side(
        patch_node_idx,
        degree,
        input_skeleton,
        input_mesh,
        true,  // is_input_side
        true,  // is_tri_mesh: base mesh is strict tri, cuts can introduce quads
        &coordination,
    );
    let (polycube_vfg, polycube_uv) = parameterize_side(
        patch_node_idx,
        degree,
        polycube_skeleton,
        polycube_mesh,
        false, // is_input_side
        false, // is_tri_mesh: base mesh is quad, cuts split quads into quads
        &coordination,
    );

    // Calculate BVHs
    let mut input_faces = extract_faces(&input_vfg, &input_uv);
    let input_bvh = if !input_faces.is_empty() {
        Some(Bvh::build(&mut input_faces))
    } else {
        None
    };

    let mut polycube_faces = extract_faces(&polycube_vfg, &polycube_uv);
    let polycube_bvh = if !polycube_faces.is_empty() {
        Some(Bvh::build(&mut polycube_faces))
    } else {
        None
    };

    RegionParameterization {
        input_vfg,
        input_uv,
        polycube_vfg,
        polycube_uv,
        input_cuts,
        polycube_cuts,
        input_faces,
        input_bvh,
        polycube_faces,
        polycube_bvh,
    }
}

fn is_empty_tri(
    graph: &petgraph::stable_graph::StableUnGraph<
        VirtualNode,
        crate::skeleton::cross_parameterize::virtual_mesh::VirtualEdgeWeight,
    >,
    uv_map: &HashMap<NodeIndex, Vector2D>,
    tri: &[NodeIndex; 3],
    uvs: &[Vector2D; 3],
) -> bool {
    for node in graph.node_indices() {
        if tri.contains(&node) {
            continue;
        }
        if let Some(&p) = uv_map.get(&node) {
            let bc = barycentric_2d(p, uvs[0], uvs[1], uvs[2]);
            if bc[0] > 1e-5 && bc[1] > 1e-5 && bc[2] > 1e-5 {
                return false;
            }
        }
    }
    true
}

fn is_empty_quad(
    graph: &petgraph::stable_graph::StableUnGraph<
        VirtualNode,
        crate::skeleton::cross_parameterize::virtual_mesh::VirtualEdgeWeight,
    >,
    uv_map: &HashMap<NodeIndex, Vector2D>,
    quad: &[NodeIndex; 4],
    uvs: &[Vector2D; 4],
) -> bool {
    // Check convexity: cross products of adjacent edges must have the same sign
    let cross1 = (uvs[1].x - uvs[0].x) * (uvs[2].y - uvs[1].y)
        - (uvs[1].y - uvs[0].y) * (uvs[2].x - uvs[1].x);
    let cross2 = (uvs[2].x - uvs[0].x) * (uvs[3].y - uvs[2].y)
        - (uvs[2].y - uvs[0].y) * (uvs[3].x - uvs[2].x);
    if cross1 * cross2 <= 0.0 {
        return false; // Not convex or bow-tie
    }

    for node in graph.node_indices() {
        if quad.contains(&node) {
            continue;
        }
        if let Some(&p) = uv_map.get(&node) {
            let bc1 = barycentric_2d(p, uvs[0], uvs[1], uvs[2]);
            if bc1[0] > 1e-5 && bc1[1] > 1e-5 && bc1[2] > 1e-5 {
                return false;
            }
            let bc2 = barycentric_2d(p, uvs[0], uvs[2], uvs[3]);
            if bc2[0] > 1e-5 && bc2[1] > 1e-5 && bc2[2] > 1e-5 {
                return false;
            }
        }
    }
    true
}

/// Extracts triangles and quads from the VFG graph and maps them to UV domain.
fn extract_faces(vfg: &VirtualFlatGeometry, uv_map: &HashMap<NodeIndex, Vector2D>) -> Vec<UvFace> {
    let mut faces = Vec::new();
    let graph = &vfg.graph;
    let mut seen_tris = std::collections::HashSet::new();
    let mut seen_quads = std::collections::HashSet::new();

    for u in graph.node_indices() {
        let neighbors: Vec<NodeIndex> = graph.neighbors(u).collect();

        // Find triangles
        for i in 0..neighbors.len() {
            for j in i + 1..neighbors.len() {
                let v = neighbors[i];
                let w = neighbors[j];
                if graph.find_edge(v, w).is_some() {
                    let mut tri = [u, v, w];
                    tri.sort();
                    if seen_tris.insert(tri) {
                        if let (Some(&uv_u), Some(&uv_v), Some(&uv_w)) =
                            (uv_map.get(&u), uv_map.get(&v), uv_map.get(&w))
                        {
                            let uvs = [uv_u, uv_v, uv_w];
                            if is_empty_tri(graph, uv_map, &[u, v, w], &uvs) {
                                faces.push(UvFace::Tri {
                                    uvs,
                                    positions: [
                                        graph[u].position,
                                        graph[v].position,
                                        graph[w].position,
                                    ],
                                    real_index: faces.len(),
                                });
                            }
                        }
                    }
                }
            }
        }

        // Find quads
        for i in 0..neighbors.len() {
            for j in i + 1..neighbors.len() {
                let v1 = neighbors[i];
                let v2 = neighbors[j];

                let neighbors_v1: std::collections::HashSet<_> = graph.neighbors(v1).collect();
                for w in graph.neighbors(v2) {
                    if w != u && neighbors_v1.contains(&w) {
                        // Check for diagonal
                        if graph.find_edge(v1, v2).is_some() || graph.find_edge(u, w).is_some() {
                            continue;
                        }

                        let mut quad = [u, v1, w, v2];
                        quad.sort();
                        if seen_quads.insert(quad) {
                            if let (Some(&uv_u), Some(&uv_v1), Some(&uv_w), Some(&uv_v2)) = (
                                uv_map.get(&u),
                                uv_map.get(&v1),
                                uv_map.get(&w),
                                uv_map.get(&v2),
                            ) {
                                let uvs = [uv_u, uv_v1, uv_w, uv_v2];
                                if is_empty_quad(graph, uv_map, &[u, v1, w, v2], &uvs) {
                                    // Ordering: u-v1, v1-w, w-v2, v2-u
                                    faces.push(UvFace::Quad {
                                        uvs,
                                        positions: [
                                            graph[u].position,
                                            graph[v1].position,
                                            graph[w].position,
                                            graph[v2].position,
                                        ],
                                        real_index: faces.len(),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    faces
}

/// Parameterizes one side (input or polycube) of a region onto the canonical domain.
///
/// `is_input_side` selects which cut paths and boundary frames from `coordination` to use.
/// Returns `(vfg, uv_map)` where `uv_map` maps every VFG node index to its
/// 2D canonical-domain position.
fn parameterize_side(
    patch_node_idx: NodeIndex,
    degree: usize,
    skeleton: &LabeledCurveSkeleton,
    mesh: &Mesh<INPUT>,
    is_input_side: bool,
    is_tri_mesh: bool,
    coordination: &RegionCoordination,
) -> (VirtualFlatGeometry, HashMap<NodeIndex, Vector2D>) {
    if degree == 0 {
        warn!(
            "TODO: Degree 0 node {:?}, skipping parameterization",
            patch_node_idx
        );
        return (VirtualFlatGeometry::empty(), HashMap::new());
    }

    // Select the side-appropriate cuts and boundary frames.
    let cuts = if is_input_side {
        &coordination.input_cuts
    } else {
        &coordination.polycube_cuts
    };
    let frames = if is_input_side {
        &coordination.input_frames
    } else {
        &coordination.polycube_frames
    };

    // Derive the phase-anchor edge from cycle_order.events[0].
    // Both sides resolve the same anchor spec to their own boundary frame, giving
    // topologically equivalent starting positions.
    let phase_anchor_edge: Option<EdgeID> = coordination
        .cycle_order
        .events
        .first()
        .map(|anchor| {
            let frame = &frames[&anchor.boundary];
            let slot_id = if is_input_side {
                // Proportionally map polycube slot index to input slot index.
                let pc_total = coordination.polycube_frames[&anchor.boundary].num_slots();
                let in_total = frame.num_slots();
                let pos = anchor.slot_id as f64 / pc_total as f64;
                (pos * in_total as f64).round() as usize % in_total
            } else {
                anchor.slot_id
            };
            frame.slot_edge(slot_id)
        });

    // Wrap cuts into a CuttingPlan (VFG builder still uses that type).
    let cutting_plan = CuttingPlan {
        cuts: cuts.to_vec(),
    };

    // Build virtual geometry by cutting the mesh open along cut paths.
    let vfg = VirtualFlatGeometry::build(
        patch_node_idx,
        skeleton,
        mesh,
        &cutting_plan,
        is_tri_mesh,
        phase_anchor_edge,
    );

    // Assign 2D positions to every node on the disk boundary.
    let boundary_positions = map_boundary_to_polygon(&vfg, degree, &coordination.cycle_order);

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
/// `cycle_order` ensures that `segment_index = 0` maps to the arc starting at the phase anchor
/// (Phase F guarantees the walk starts there), giving identical segment→polygon-side
/// assignment on both input and polycube sides.
fn map_boundary_to_polygon(
    vfg: &VirtualFlatGeometry,
    degree: usize,
    cycle_order: &CutCycleOrder,
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

        let basis_index = 0;
        // We pick `basis_index` to be the first corner, then place the next 3 corners at cumulative
        // arc-lengths of 1/4, 2/4, and 3/4 around the loop. We enforce strictly increasing offsets
        // to guarantee 4 distinct corners, even if edge lengths are disproportionately large.
        let mut corner_indices = Vec::with_capacity(4);
        corner_indices.push(basis_index);

        let mut current_offset = 0;
        let mut cumulative_length = 0.0;
        let n = boundary.len();

        for k in 1..4 {
            let target_length = total_length * (k as f64 / 4.0);

            // The maximum offset we can pick and still leave enough vertices for the remaining corners.
            let max_offset = n - (4 - k);

            // Advance by at least one vertex to ensure corners are distinct.
            current_offset += 1;
            cumulative_length += edge_lengths[(basis_index + current_offset - 1) % n];

            // Continue advancing until we reach the target length or the maximum allowed offset.
            while current_offset < max_offset && cumulative_length < target_length {
                cumulative_length += edge_lengths[(basis_index + current_offset) % n];
                current_offset += 1;
            }

            corner_indices.push((basis_index + current_offset) % n);
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
                curr = (curr + 1) % n;
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
                curr = (curr + 1) % n;
            }
        }

        return result;
    }

    let n_sides = 4 * (degree - 1);
    if boundary.len() < 2 {
        panic!(
            "Boundary has fewer than 2 vertices for degree {} region",
            degree
        );
    }

    // Robust against arbitrary cyclic starting index: identify all cut-endpoint duplicates,
    // then parameterize each segment between consecutive endpoints around the cycle.
    let endpoint_positions: Vec<usize> = boundary
        .iter()
        .enumerate()
        .filter_map(|(i, &node_idx)| {
            if matches!(
                vfg.graph[node_idx].origin,
                VirtualNodeOrigin::CutEndpointMidpointDuplicate { .. }
            ) {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    if endpoint_positions.is_empty() {
        panic!(
            "Degree {} boundary has no cut endpoints; cannot build cut/non-cut segments",
            degree
        );
    }
    if endpoint_positions.len() != n_sides {
        panic!(
            "Boundary endpoint count mismatch for degree {}: expected {} cut-endpoint nodes, found {}",
            degree,
            n_sides,
            endpoint_positions.len()
        );
    }

    // Phase G invariant: the boundary walk must start at a cut endpoint (the phase anchor).
    // If Phase F is correct, boundary[0] is always a CutEndpointMidpointDuplicate,
    // so endpoint_positions[0] == 0.
    if !cycle_order.events.is_empty() {
        assert_eq!(
            endpoint_positions[0],
            0,
            "Phase anchor invariant violated: boundary walk does not start at a cut endpoint \
             (first endpoint found at position {}, expected 0). Phase F may be broken.",
            endpoint_positions[0]
        );
    }

    let n = boundary.len();
    let mut result = HashMap::new();

    for segment_index in 0..endpoint_positions.len() {
        let start_pos = endpoint_positions[segment_index];
        let end_pos = endpoint_positions[(segment_index + 1) % endpoint_positions.len()];

        let mut buf = vec![boundary[start_pos]];
        let mut cursor = (start_pos + 1) % n;

        while cursor != end_pos {
            let node_idx = boundary[cursor];
            match vfg.graph[node_idx].origin {
                VirtualNodeOrigin::CutDuplicate { .. }
                | VirtualNodeOrigin::BoundaryMidpoint { .. } => {
                    // Valid interior of a segment between two cut endpoints.
                }
                VirtualNodeOrigin::CutEndpointMidpointDuplicate { .. } => {
                    panic!(
                        "Unexpected cut endpoint inside segment {} while tracing boundary between endpoint positions {} and {}",
                        segment_index, start_pos, end_pos
                    );
                }
                ref other => {
                    panic!(
                        "Unexpected boundary node type {:?} while building segment {}",
                        other, segment_index
                    );
                }
            }
            buf.push(node_idx);
            cursor = (cursor + 1) % n;
        }

        buf.push(boundary[end_pos]);
        parameterize_segment(segment_index, n_sides, &buf, vfg, &mut result);
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
