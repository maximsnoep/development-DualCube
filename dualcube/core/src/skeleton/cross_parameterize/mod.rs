use std::collections::HashMap;
use std::f64::consts::PI;

use log::warn;
use mehsh::prelude::{HasPosition, Mesh, Vector2D, Vector3D};

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

/// Minimum separation between cut endpoints on the same boundary,
/// measured in [0, 1] normalized arc-length. Two cut endpoints on the
/// same boundary loop must be at least this far apart.
const MIN_CUT_SEPARATION: f64 = 0.15;

/// A point on the mesh surface.
///
/// Can represent an arbitrary position along a mesh edge (useful for boundary
/// midpoints and generic surface traversal) or exactly at a mesh vertex.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SurfacePoint {
    /// A point on a mesh edge at interpolation parameter `t in [0, 1]`.
    /// Position = `(1 − t) · root_position + t · toor_position`.
    OnEdge { edge: EdgeID, t: f64 },

    /// Exactly at a mesh vertex.
    OnVertex { vertex: VertID },
}

impl SurfacePoint {
    /// Computes the 3D world-space position of this surface point.
    pub fn position(&self, mesh: &Mesh<INPUT>) -> Vector3D {
        match *self {
            SurfacePoint::OnEdge { edge, t } => {
                let p0 = mesh.position(mesh.root(edge));
                let p1 = mesh.position(mesh.toor(edge));
                p0 * (1.0 - t) + p1 * t
            }
            SurfacePoint::OnVertex { vertex } => mesh.position(vertex),
        }
    }
}

/// A path across the mesh surface, stored as a sequence of [`SurfacePoint`]s.
///
/// Consecutive points should share a triangle face (or be connected by a mesh
/// edge). This representation allows paths to cross faces at arbitrary
/// positions, not just along mesh edges.
#[derive(Debug, Clone)]
pub struct SurfacePath {
    pub points: Vec<SurfacePoint>,
}

impl SurfacePath {
    /// Converts this surface path to a sequence of 3D positions (e.g. for visualisation).
    pub fn to_positions(&self, mesh: &Mesh<INPUT>) -> Vec<Vector3D> {
        self.points.iter().map(|p| p.position(mesh)).collect()
    }
}

/// Arc-length parameterization of a boundary loop into `[0, 1)`.
///
/// Each midpoint of the boundary loop's edges is assigned a `t` value based on
/// cumulative arc length.  The basis point (`t = 0`) is the midpoint with the
/// greatest x coordinate (breaking ties with y, then z).
#[derive(Debug, Clone)]
pub struct BoundaryParameterization {
    /// For each edge midpoint in the boundary loop, its `t` value in `[0, 1)`.
    /// Indices correspond to `BoundaryLoop::edge_midpoints`.
    pub t_values: Vec<f64>,

    /// Total arc length of the boundary loop.
    pub total_length: f64,

    /// Index into `edge_midpoints` that was chosen as the basis (`t = 0`).
    pub basis_index: usize,
}

/// A single cut connecting two boundary loops, with the exact surface path.
#[derive(Debug, Clone)]
pub struct CutPath {
    /// The boundary loop (skeleton edge) where this cut starts.
    pub start_boundary: EdgeIndex,

    /// The `t`-parameter on the start boundary where the cut begins.
    pub start_t: f64,

    /// The boundary loop (skeleton edge) where this cut ends.
    pub end_boundary: EdgeIndex,

    /// The `t`-parameter on the end boundary where the cut ends.
    pub end_t: f64,

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
        polycube_mesh: &Mesh<INPUT>,
    ) -> Mesh<INPUT> {
        let mut result = input_mesh.clone();

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
    node_idx: NodeIndex,
    degree: usize,
    input_skeleton: &LabeledCurveSkeleton,
    polycube_skeleton: &LabeledCurveSkeleton,
    input_mesh: &Mesh<INPUT>,
    polycube_mesh: &Mesh<INPUT>,
) -> RegionParameterization {
    // Compute cutting plans for both sides.
    // The cut topology (which boundary loops to connect) is shared, computed
    // from combined geodesic distances. The actual cut paths are per-side.
    let (input_plan, polycube_plan) = compute_cutting_plans(
        node_idx,
        input_skeleton,
        polycube_skeleton,
        input_mesh,
        polycube_mesh,
    );

    // Parameterize each side independently using its cutting plan.
    let (input_vfg, input_uv, input_cuts) =
        parameterize_side(node_idx, degree, input_skeleton, input_mesh, &input_plan);

    let (polycube_vfg, polycube_uv, polycube_cuts) = parameterize_side(
        node_idx,
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
/// Returns `(vfg, uv_map, cut_positions)` where `uv_map` maps every VFG node
/// index to its 2D canonical-domain position.
fn parameterize_side(
    node_idx: NodeIndex,
    degree: usize,
    skeleton: &LabeledCurveSkeleton,
    mesh: &Mesh<INPUT>,
    cutting_plan: &CuttingPlan,
) -> (VirtualFlatGeometry, HashMap<NodeIndex, Vector2D>, Vec<Vec<Vector3D>>) {
    if degree == 0 {
        warn!(
            "TODO: Degree 0 node {:?}, skipping parameterization",
            node_idx
        );
        return (
            VirtualFlatGeometry::empty(),
            HashMap::new(),
            Vec::new(),
        );
    }

    // Convert cut paths to 3D position sequences for visualisation.
    let cut_positions: Vec<Vec<Vector3D>> = cutting_plan
        .cuts
        .iter()
        .map(|cut| cut.path.to_positions(mesh))
        .collect();

    // Build virtual geometry by cutting the mesh open along cut paths,
    // duplicating vertices so the result is a topological disk.
    // TODO: use this once implemented
    let vfg = VirtualFlatGeometry::build(node_idx, skeleton, mesh, cutting_plan);

    // Assign 2D positions to every node on the disk boundary.
    // The canonical polygon has n_sides sides:
    //   - degree 1 -> square (4 sides)
    //   - degree d >= 2 -> regular 4(d-1)-gon
    let n_sides = if degree == 1 { 4 } else { 4 * (degree - 1) };
    let boundary_positions = map_boundary_to_polygon(&vfg, n_sides);

    // Solve the Dirichlet problem on the VFG graph.
    // let uv_map = solve_dirichlet(&vfg, &boundary_positions);
    let uv_map = HashMap::new(); // TODO implement later

    (vfg, uv_map, cut_positions)
}

/// Maps every node in `vfg.boundary_loop` to a 2D position on a regular `n_sides`-gon.
///
/// Positions are distributed proportionally by the 3D arc-length of consecutive
/// boundary nodes. This ensures a valid Tutte embedding (boundary on a convex polygon)
/// while preserving the structural order produced by the VFG boundary trace.
///
/// The polygon has circumradius 1 with vertices at angles `2πk/n_sides`.
fn map_boundary_to_polygon(vfg: &VirtualFlatGeometry, n_sides: usize) -> HashMap<NodeIndex, Vector2D> {
    let boundary = &vfg.boundary_loop;
    let n = boundary.len();
    if n == 0 {
        return HashMap::new();
    }

    // Regular polygon vertices.
    let polygon: Vec<Vector2D> = (0..n_sides)
        .map(|k| {
            let angle = 2.0 * PI * k as f64 / n_sides as f64;
            Vector2D::new(angle.cos(), angle.sin())
        })
        .collect();

    // Compute 3D arc lengths between consecutive boundary nodes (cyclic).
    let mut seg_lengths: Vec<f64> = vec![0.0; n];
    let mut total: f64 = 0.0;
    for i in 0..n {
        let a = vfg.graph[boundary[i]].position;
        let b = vfg.graph[boundary[(i + 1) % n]].position;
        let len = (b - a).norm();
        seg_lengths[i] = len;
        total += len;
    }

    // Place each boundary node at the polygon position corresponding to its
    // cumulative arc-length fraction.
    let mut result = HashMap::new();
    let mut cumulative: f64 = 0.0;

    for (i, &node) in boundary.iter().enumerate() {
        let t = cumulative / total; // in [0, 1)

        // Locate position on the polygon perimeter at t part.
        let frac = t * n_sides as f64;
        let side = (frac as usize).min(n_sides - 1);
        let local_t = frac - side as f64;

        let p0 = polygon[side];
        let p1 = polygon[(side + 1) % n_sides];
        let pos = p0 * (1.0 - local_t) + p1 * local_t;

        result.insert(node, pos);
        cumulative += seg_lengths[i];
    }

    result
}
