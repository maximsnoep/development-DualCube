use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;

use log::warn;
use mehsh::prelude::{HasPosition, Mesh, SetPosition, Vector2D, Vector3D};

use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::visit::IntoEdgeReferences;
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

    /// Index into the start boundary's `edge_midpoints` where this cut touches.
    pub start_midpoint_idx: usize,

    /// The `t`-parameter on the start boundary where the cut begins.
    /// Assigned after cut paths are found, based on shared boundary parameterization.
    pub start_t: f64,

    /// The boundary loop (skeleton edge) where this cut ends.
    pub end_boundary: EdgeIndex,

    /// Index into the end boundary's `edge_midpoints` where this cut touches.
    pub end_midpoint_idx: usize,

    /// The `t`-parameter on the end boundary where the cut ends.
    /// Assigned after cut paths are found, based on shared boundary parameterization.
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

    /// Arc-length parameterization for each boundary loop of this region.
    /// On the input side these are natural arc-length; on the polycube side
    /// they are warped so that cut endpoint `t`-values match the input side.
    pub boundary_params: HashMap<EdgeIndex, BoundaryParameterization>,
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
        _polycube_mesh: &Mesh<INPUT>,
    ) -> Mesh<INPUT> {
        let mut result = input_mesh.clone();

        // Build input vertex → region map.
        let mut vert_to_region: HashMap<VertID, NodeIndex> = HashMap::new();
        for node_idx in input_skeleton.node_indices() {
            for &v in &input_skeleton[node_idx].skeleton_node.patch_vertices {
                vert_to_region.insert(v, node_idx);
            }
        }

        // For each region, extract UV triangles from the polycube VFG's planar embedding.
        let mut region_uv_tris: HashMap<NodeIndex, Vec<UvTriangle>> = HashMap::new();
        for (&node_idx, region) in &self.regions {
            let tris = extract_uv_faces(&region.polycube_vfg, &region.polycube_uv);
            region_uv_tris.insert(node_idx, tris);
        }

        // Map each input vertex to its polycube surface position.
        let mut unmapped = 0usize;
        for v in input_mesh.vert_ids() {
            let Some(&region_idx) = vert_to_region.get(&v) else {
                continue;
            };
            let Some(region) = self.regions.get(&region_idx) else {
                continue;
            };
            let Some(vfg_nodes) = region.input_vfg.vert_to_nodes.get(&v) else {
                continue;
            };
            // For cut vertices, both copies map to the same polycube position;
            // pick the first.
            let Some(&uv) = region.input_uv.get(&vfg_nodes[0]) else {
                continue;
            };
            let Some(tris) = region_uv_tris.get(&region_idx) else {
                continue;
            };

            if let Some(pos) = interpolate_in_uv_triangles(uv, tris) {
                result.set_position(v, pos);
            } else {
                unmapped += 1;
            }
        }

        if unmapped > 0 {
            warn!(
                "to_triangle_mesh_polycube: {} input vertices could not be mapped \
                 (UV point not inside any polycube triangle)",
                unmapped
            );
        }

        result
    }
}

/// A triangle in UV space with associated 3D positions at each vertex.
struct UvTriangle {
    uv: [Vector2D; 3],
    pos: [Vector3D; 3],
}

/// Extracts faces from a VFG's planar embedding (using UV coordinates) and
/// returns them as UV triangles suitable for interpolation.
///
/// Uses the standard planar-graph face extraction: for each directed edge,
/// finds the face to its left by following the "next edge in clockwise order"
/// at each vertex. Each face is then fan-triangulated.
fn extract_uv_faces(
    vfg: &VirtualFlatGeometry,
    uv: &HashMap<NodeIndex, Vector2D>,
) -> Vec<UvTriangle> {
    use petgraph::visit::EdgeRef;

    // For each vertex, sort neighbors by angle in CCW order.
    let mut sorted_neighbors: HashMap<NodeIndex, Vec<NodeIndex>> = HashMap::new();
    for node in vfg.graph.node_indices() {
        let node_uv = uv[&node];
        let mut nbrs: Vec<NodeIndex> = vfg.graph.neighbors(node).collect();
        nbrs.sort_by(|&a, &b| {
            let da = uv[&a] - node_uv;
            let db = uv[&b] - node_uv;
            let angle_a = da.y.atan2(da.x);
            let angle_b = db.y.atan2(db.x);
            angle_a.partial_cmp(&angle_b).unwrap_or(std::cmp::Ordering::Equal)
        });
        // Deduplicate neighbors (parallel edges in StableUnGraph).
        nbrs.dedup();
        sorted_neighbors.insert(node, nbrs);
    }

    // For each directed edge (u, v), trace the face to its left.
    // "Next edge" at v after arriving from u: the neighbor of v that is
    // immediately CW from u in v's sorted (CCW) neighbor list.
    let mut seen_faces: HashSet<Vec<NodeIndex>> = HashSet::new();
    let mut tris = Vec::new();

    for edge_ref in vfg.graph.edge_references() {
        let src: NodeIndex = edge_ref.source();
        let tgt: NodeIndex = edge_ref.target();
        for &(start, second) in &[
            (src, tgt),
            (tgt, src),
        ] {
            let face = trace_face_left(start, second, &sorted_neighbors);
            if face.len() < 3 {
                continue;
            }

            // Canonicalize: rotate to start with the minimum NodeIndex.
            let min_idx = face
                .iter()
                .enumerate()
                .min_by_key(|(_, n)| n.index())
                .unwrap()
                .0;
            let mut canonical = face.clone();
            canonical.rotate_left(min_idx);
            if !seen_faces.insert(canonical) {
                continue; // already processed
            }

            // Skip the outer (infinite) face — it winds CW (negative signed area).
            let signed_area: f64 = (0..face.len())
                .map(|i| {
                    let a = uv[&face[i]];
                    let b = uv[&face[(i + 1) % face.len()]];
                    a.x * b.y - b.x * a.y
                })
                .sum();
            if signed_area <= 0.0 {
                continue;
            }

            // Fan-triangulate from the first vertex.
            for i in 1..face.len() - 1 {
                tris.push(UvTriangle {
                    uv: [uv[&face[0]], uv[&face[i]], uv[&face[i + 1]]],
                    pos: [
                        vfg.graph[face[0]].position,
                        vfg.graph[face[i]].position,
                        vfg.graph[face[i + 1]].position,
                    ],
                });
            }
        }
    }

    tris
}

/// Traces the face to the left of directed edge (from → to) by following
/// clockwise-next edges at each vertex.
fn trace_face_left(
    from: NodeIndex,
    to: NodeIndex,
    sorted_neighbors: &HashMap<NodeIndex, Vec<NodeIndex>>,
) -> Vec<NodeIndex> {
    let mut face = vec![from];
    let mut prev = from;
    let mut curr = to;

    let max_iters = sorted_neighbors.len() + 2;
    for _ in 0..max_iters {
        if curr == from {
            break;
        }
        face.push(curr);

        // At curr, find the edge immediately CW from the edge (curr → prev).
        // In our CCW-sorted neighbor list, "immediately CW" = the entry BEFORE
        // prev in the sorted order.
        let nbrs = &sorted_neighbors[&curr];
        let pos = nbrs.iter().position(|&n| n == prev);
        let Some(pos) = pos else {
            break; // shouldn't happen
        };
        let next = if pos == 0 {
            nbrs[nbrs.len() - 1]
        } else {
            nbrs[pos - 1]
        };

        prev = curr;
        curr = next;
    }

    face
}

/// Finds which UV triangle contains `query` and returns the interpolated 3D position.
/// Falls back to the nearest triangle if the point is slightly outside all of them.
fn interpolate_in_uv_triangles(query: Vector2D, tris: &[UvTriangle]) -> Option<Vector3D> {
    let mut best: Option<(f64, Vector3D)> = None; // (min_bary_coord, interpolated_pos)

    for tri in tris {
        if let Some((u, v, w)) = barycentric_2d(query, tri.uv[0], tri.uv[1], tri.uv[2]) {
            let min_coord = u.min(v).min(w);
            let pos = tri.pos[0] * u + tri.pos[1] * v + tri.pos[2] * w;
            if min_coord >= -1e-6 {
                return Some(pos);
            }
            // Track the triangle where the point is "least outside".
            if best.as_ref().map_or(true, |(prev_min, _)| min_coord > *prev_min) {
                best = Some((min_coord, pos));
            }
        }
    }

    // Accept if the point is only slightly outside (within tolerance).
    best.filter(|(min_coord, _)| *min_coord >= -0.1)
        .map(|(_, pos)| pos)
}

/// Computes barycentric coordinates of `p` with respect to triangle `(a, b, c)`.
/// Returns `None` if the triangle is degenerate (zero area).
fn barycentric_2d(
    p: Vector2D,
    a: Vector2D,
    b: Vector2D,
    c: Vector2D,
) -> Option<(f64, f64, f64)> {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;

    let d00 = v0.x * v0.x + v0.y * v0.y;
    let d01 = v0.x * v1.x + v0.y * v1.y;
    let d11 = v1.x * v1.x + v1.y * v1.y;
    let d20 = v2.x * v0.x + v2.y * v0.y;
    let d21 = v2.x * v1.x + v2.y * v1.y;

    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-30 {
        return None;
    }

    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;

    Some((u, v, w))
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
    // Compute cutting plans for both sides. Cut paths are found independently on each side, then boundary
    // parameterizations are built so that cut endpoints share t-values.
    let (input_plan, polycube_plan) = compute_cutting_plans(
        node_idx,
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
    let (input_vfg, input_uv) =
        parameterize_side(node_idx, degree, input_skeleton, input_mesh, &input_plan);

    let (polycube_vfg, polycube_uv) = parameterize_side(
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
/// Returns `(vfg, uv_map)` where `uv_map` maps every VFG node index to its
/// 2D canonical-domain position. Cut positions for visualisation are extracted
/// by the caller before this function is called.
fn parameterize_side(
    node_idx: NodeIndex,
    degree: usize,
    skeleton: &LabeledCurveSkeleton,
    mesh: &Mesh<INPUT>,
    cutting_plan: &CuttingPlan,
) -> (VirtualFlatGeometry, HashMap<NodeIndex, Vector2D>) {
    if degree == 0 {
        warn!(
            "TODO: Degree 0 node {:?}, skipping parameterization",
            node_idx
        );
        return (VirtualFlatGeometry::empty(), HashMap::new());
    }

    // Build virtual geometry by cutting the mesh open along cut paths,
    // duplicating vertices so the result is a topological disk.
    let vfg = VirtualFlatGeometry::build(node_idx, skeleton, mesh, cutting_plan);

    // Assign 2D positions to every node on the disk boundary.
    // The canonical polygon has n_sides sides:
    //   - degree 1 -> square (4 sides)
    //   - degree d >= 2 -> regular 4(d-1)-gon
    let n_sides = if degree == 1 { 4 } else { 4 * (degree - 1) };
    let boundary_positions = map_boundary_to_polygon(&vfg, n_sides);

    // Solve the Dirichlet problem on the VFG graph.
    let uv_map = solve_dirichlet(&vfg, &boundary_positions);
    (vfg, uv_map)
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
