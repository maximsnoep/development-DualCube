use std::collections::HashMap;

use log::{error, warn};
use mehsh::prelude::{Mesh, Vector2D, Vector3D};

use petgraph::graph::{EdgeIndex, NodeIndex};

use serde::{Deserialize, Serialize};

use crate::prelude::{PrincipalDirection, VertID, INPUT};
use crate::skeleton::orthogonalize::LabeledCurveSkeleton;

mod cutting_plan;
mod harmonic;

pub use cutting_plan::compute_cutting_plan;

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
    /// edge `edge_a` to the one on `edge_b`. There are d-1 such cuts for a degree d region.
    /// // TODO: also needs to specify at what t-parameter along the boundary the cut endpointS are.
    pub cuts: Vec<(EdgeIndex, EdgeIndex)>,
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
    let (input_to_canonical, input_cuts) =
        parameterize_side(node_idx, degree, input_skeleton, input_mesh, &cutting_plan);

    let (polycube_to_canonical, polycube_cuts) = parameterize_side(
        node_idx,
        degree,
        polycube_skeleton,
        polycube_mesh,
        &cutting_plan,
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


    // TODO
    (HashMap::new(), Vec::new())
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


