use std::collections::HashMap;

use log::warn;
use mehsh::prelude::{HasPosition, Mesh, Vector2D, Vector3D};

use petgraph::graph::{EdgeIndex, NodeIndex};

use serde::{Deserialize, Serialize};

use crate::prelude::{EdgeID, PrincipalDirection, VertID, INPUT};
use crate::skeleton::orthogonalize::LabeledCurveSkeleton;

mod cutting_plan;
mod harmonic;

pub use cutting_plan::compute_cutting_plans;

/// Minimum separation between cut endpoints on the same boundary,
/// measured in [0, 1] normalized arc-length. Two cut endpoints on the
/// same boundary loop must be at least this far apart.
const MIN_CUT_SEPARATION: f64 = 0.15;

/// A point on the mesh surface.
///
/// Can represent an arbitrary position along a mesh edge (useful for boundary
/// midpoints and generic surface traversal) or exactly at a mesh vertex.
#[derive(Debug, Clone, Copy)]
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
/// For degree `d ≥ 2`, contains `d − 1` cuts forming a spanning tree over the
/// `d` boundary loops.  For degree 0 or 1, `cuts` is empty.
#[derive(Debug, Clone)]
pub struct CuttingPlan {
    /// Arc-length parameterization for every boundary loop of this region.
    pub boundary_params: HashMap<EdgeIndex, BoundaryParameterization>,

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
/// - degree 2+: a regular 4(d-1) gon for degree d.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionParameterization {
    /// For each input mesh vertex in this region, its 2D position in the canonical domain.
    pub input_to_canonical: HashMap<VertID, Vector2D>, // TODO: what to do with cut parts? These should be dual?

    /// For each polycube mesh vertex in this region, its 2D position in the canonical domain.
    /// NOTE: Keys are VertKey<POLYCUBE> stored as VertID via raw key, same convention as
    /// `SkeletonNode::patch_vertices` on the polycube skeleton.
    pub polycube_to_canonical: HashMap<VertID, Vector2D>, // TODO: what to do with cut parts? These should be dual?

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
    let (input_to_canonical, input_cuts) =
        parameterize_side(node_idx, degree, input_skeleton, input_mesh, &input_plan);

    let (polycube_to_canonical, polycube_cuts) = parameterize_side(
        node_idx,
        degree,
        polycube_skeleton,
        polycube_mesh,
        &polycube_plan,
    );

    RegionParameterization {
        input_to_canonical,
        polycube_to_canonical,
        input_cuts,
        polycube_cuts,
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
/// Returns a map from vertex ID to 2D canonical-domain position and the cut paths
/// as 3D position sequences extended to the geometric boundary.
fn parameterize_side(
    node_idx: NodeIndex,
    degree: usize,
    skeleton: &LabeledCurveSkeleton,
    mesh: &Mesh<INPUT>,
    cutting_plan: &CuttingPlan,
) -> (HashMap<VertID, Vector2D>, Vec<Vec<Vector3D>>) {
    if degree == 0 {
        warn!(
            "TODO: Degree 0 node {:?}, skipping parameterization",
            node_idx
        );
        return (HashMap::new(), Vec::new());
    }

    // Convert cut paths to 3D position sequences for visualisation.
    let cut_positions: Vec<Vec<Vector3D>> = cutting_plan
        .cuts
        .iter()
        .map(|cut| cut.path.to_positions(mesh))
        .collect();

    // TODO: harmonic parameterization
    (HashMap::new(), cut_positions)
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


