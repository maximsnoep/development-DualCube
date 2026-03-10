use std::collections::{HashMap, HashSet};

use log::warn;
use mehsh::prelude::{Mesh, Vector2D};
use ordered_float::OrderedFloat;
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};

use crate::prelude::{VertID, INPUT};
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
        _input_mesh: &Mesh<INPUT>,
        _polycube_skeleton: &LabeledCurveSkeleton,
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

    if degree == 1 {
        return parameterize_degree_one(patch_verts, &boundaries, mesh);
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

    // Solve harmonic 2D with cut-aware boundary handling.
    solve_harmonic_2d_with_cuts(
        patch_verts,
        &boundary_positions,
        &cut_dual_values,
        &cut_paths,
        mesh,
    )
}


/// For a region with a single boundary loop: split the boundary into 4 equal arcs,
/// map them to the 4 sides of a square, and solve Dirichlet problem for the interior.
fn parameterize_degree_one(
    patch_verts: &[VertID],
    boundaries: &HashMap<EdgeIndex, OrderedBoundary>,
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

    // Pick 4 approximately equally-spaced corner vertices by arc-length.
    let quarter = boundary.total_length / 4.0;
    let corners: Vec<usize> = (0..4)
        .map(|i| {
            let target = quarter * (i as f64);
            boundary
                .cumulative
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
        let start_idx = corners[seg];
        let end_idx = corners[(seg + 1) % 4];
        let p0 = polygon[seg];
        let p1 = polygon[(seg + 1) % 4];

        let arc_start = boundary.cumulative[start_idx];
        let arc_end_raw = boundary.cumulative[end_idx];
        let seg_length = if end_idx > start_idx {
            arc_end_raw - arc_start
        } else {
            (boundary.total_length - arc_start) + arc_end_raw
        };

        let mut i = start_idx;
        loop {
            let al = boundary.cumulative[i];
            let al_adj = if i >= start_idx {
                al - arc_start
            } else {
                (boundary.total_length - arc_start) + al
            };
            let t = if seg_length > 0.0 {
                (al_adj / seg_length).clamp(0.0, 1.0)
            } else {
                0.0
            };
            boundary_positions.insert(
                boundary.vertices[i],
                Vector2D::new(p0.x * (1.0 - t) + p1.x * t, p0.y * (1.0 - t) + p1.y * t),
            );
            if i == end_idx {
                break;
            }
            i = (i + 1) % n;
        }
    }

    solve_harmonic_2d(patch_verts, &boundary_positions, mesh)
}
