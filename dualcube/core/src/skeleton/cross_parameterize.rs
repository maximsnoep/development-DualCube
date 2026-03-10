use std::collections::HashMap;

use mehsh::prelude::{Mesh, Vector2D};
use petgraph::graph::NodeIndex;
use serde::{Deserialize, Serialize};

use crate::prelude::{VertID, INPUT};
use crate::skeleton::orthogonalize::LabeledCurveSkeleton;

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
        polycube_skeleton: &LabeledCurveSkeleton,
    ) -> Mesh<INPUT> {
        // TODO: For each region:
        //   1. Clone the input mesh
        //   2. For each input vertex in the region, get its canonical (u, v) coords
        //   3. Find the triangle in the polycube parameterization containing that (u, v)
        //   4. Interpolate to get the 3D polycube-surface position
        //   5. set_position on the cloned mesh
        // NOTE: Step 3-4 requires triangulating the polycube parameterization and building
        //       a point-location structure. A simple approach is barycentric interpolation
        //       over the polycube triangulation in 2D canonical space.
        Mesh::default()
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
    _polycube_mesh: &Mesh<INPUT>,
) -> RegionParameterization {
    // TODO: Steps for each side (input and polycube):
    //
    // 1. Cut both regions in the same way to get to disk topology -> shared domain.
    // 2. Fix boundary parameterization (not sure what to do with multiple distinct point being picked, for 1 point on boundary we can just make it t=0)
    //       I guess if we use paths on the surface which do not necessarily traverse edges this isn't a problem
    // 3. Solve harmonic field (uniform laplacian to make use of Tutte embedding theorem)
    //    for interior vertices to get smooth parameterization.  I think these can be solved independently.

    let input_to_canonical = parameterize_side(node_idx, degree, input_skeleton, input_mesh);

    let polycube_to_canonical =
        parameterize_side(node_idx, degree, polycube_skeleton, _polycube_mesh);

    RegionParameterization {
        input_to_canonical,
        polycube_to_canonical,
    }
}

/// Parameterizes one side (input or polycube) of a region onto the canonical ℓ-gon.
///
/// Returns a map from vertex ID to 2D canonical-domain position.
fn parameterize_side(
    _node_idx: NodeIndex,
    _degree: usize,
    _skeleton: &LabeledCurveSkeleton,
    _mesh: &Mesh<INPUT>,
) -> HashMap<VertID, Vector2D> {
    // TODO

    HashMap::new()
}

/// Solves a 2D Dirichlet problem on a surface mesh region.
///
/// Given a set of vertices, some with fixed 2D positions (boundary) and the rest free (interior),
/// solves the discrete Laplace equation to find 2D positions for the free vertices.
///
/// Uses uniform graph Laplacian weights and direct Cholesky factorization,
/// same as `solve_harmonic_scalar_field` but for 2D coordinates.
fn solve_harmonic_2d(
    all_vertices: &[VertID],
    boundary_positions: &HashMap<VertID, Vector2D>,
    mesh: &Mesh<INPUT>,
) -> HashMap<VertID, Vector2D> {
    // TODO: Essentially solve_harmonic_scalar_field twice (once for u, once for v)
    // but with boundary values not just 0 or 1, but 2d positions (so 0<=x,y<=1, though depending on the domain shape more can be cut off)
    HashMap::new()
}
