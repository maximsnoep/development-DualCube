use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;

use log::warn;
use mehsh::prelude::{HasEdges, HasNeighbors, HasPosition, HasVertices, Mesh, SetPosition, Vector2D, Vector3D};

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
        polycube_skeleton: &LabeledCurveSkeleton,
        polycube_mesh: &Mesh<INPUT>,
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
        // The VFG faces tile the entire canonical polygon, ensuring complete coverage.
        // Crossing edges in degree 2+ regions may cause some overlap, but every input
        // UV point will be inside at least one polycube UV triangle.
        let mut region_uv_tris: HashMap<NodeIndex, Vec<UvTriangle>> = HashMap::new();
        for (&node_idx, region) in &self.regions {
            let degree = polycube_skeleton.edges(node_idx).count();
            let tris = extract_uv_faces(
                &region.polycube_vfg,
                &region.polycube_uv,
                node_idx,
                degree,
            );
            region_uv_tris.insert(node_idx, tris);
        }

        // Map each input vertex to its polycube surface position.
        let mut unmapped = 0usize;
        let mut mapped_vertices: Vec<(VertID, NodeIndex)> = Vec::new();
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

            if let Some(pos) = interpolate_in_uv_triangles(uv, tris, polycube_mesh) {
                result.set_position(v, pos);
                mapped_vertices.push((v, region_idx));
            } else {
                unmapped += 1;
            }
        }

        if unmapped > 0 {
            warn!(
                "to_triangle_mesh_polycube: {} input vertices could not be mapped",
                unmapped
            );
        }

        // ── Diagnostics: surface distance and edge-through metrics ──
        {
            // Polycube bounding box diameter for threshold.
            let (mut bmin, mut bmax) = (
                Vector3D::new(f64::MAX, f64::MAX, f64::MAX),
                Vector3D::new(f64::MIN, f64::MIN, f64::MIN),
            );
            for pv in polycube_mesh.vert_ids() {
                let p = polycube_mesh.position(pv);
                bmin = Vector3D::new(bmin.x.min(p.x), bmin.y.min(p.y), bmin.z.min(p.z));
                bmax = Vector3D::new(bmax.x.max(p.x), bmax.y.max(p.y), bmax.z.max(p.z));
            }
            let diameter = (bmax - bmin).norm();
            let threshold = diameter * 0.01;

            let mut region_stats: HashMap<NodeIndex, (usize, usize, f64)> = HashMap::new();
            let mut global_max_dist = 0.0f64;
            let mut global_far = 0usize;

            for &(v, region_idx) in &mapped_vertices {
                let pos = result.position(v);
                let dist = distance_to_mesh_surface(pos, polycube_mesh);
                let stats = region_stats.entry(region_idx).or_insert((0, 0, 0.0));
                stats.0 += 1;
                if dist > threshold {
                    stats.1 += 1;
                    global_far += 1;
                }
                if dist > stats.2 { stats.2 = dist; }
                if dist > global_max_dist { global_max_dist = dist; }
            }

            let mut edge_through = 0usize;
            let mut total_edges = 0usize;
            let mut seen_edges: HashSet<(VertID, VertID)> = HashSet::new();
            for face in input_mesh.face_ids() {
                let fv: Vec<VertID> = input_mesh.vertices(face).collect();
                for i in 0..fv.len() {
                    let a = fv[i];
                    let b = fv[(i + 1) % fv.len()];
                    let key = if a < b { (a, b) } else { (b, a) };
                    if !seen_edges.insert(key) { continue; }
                    total_edges += 1;
                    let mid = (result.position(a) + result.position(b)) * 0.5;
                    let dist = distance_to_mesh_surface(mid, polycube_mesh);
                    if dist > threshold { edge_through += 1; }
                }
            }

            log::info!("=== Cross-parameterization diagnostics ===");
            log::info!("  Polycube diameter: {:.4}, threshold (1%): {:.6}", diameter, threshold);
            let mut region_list: Vec<_> = region_stats.iter().collect();
            region_list.sort_by_key(|(idx, _)| idx.index());
            for (&node_idx, &(mapped, far, max_d)) in &region_list {
                let degree = polycube_skeleton.edges(node_idx).count();
                log::info!(
                    "  Region {:?} (deg {}): {} mapped, {} far ({:.1}%), max_dist={:.6} ({:.2}% diam)",
                    node_idx, degree, mapped, far,
                    if mapped > 0 { far as f64 / mapped as f64 * 100.0 } else { 0.0 },
                    max_d, max_d / diameter * 100.0
                );
            }
            log::info!(
                "  Total: {} mapped, {} unmapped, {} far ({:.1}%), max_dist={:.6} ({:.2}% diam)",
                mapped_vertices.len(), unmapped, global_far,
                if !mapped_vertices.is_empty() { global_far as f64 / mapped_vertices.len() as f64 * 100.0 } else { 0.0 },
                global_max_dist, global_max_dist / diameter * 100.0
            );
            log::info!(
                "  Edges through surface: {} of {} ({:.1}%)",
                edge_through, total_edges,
                if total_edges > 0 { edge_through as f64 / total_edges as f64 * 100.0 } else { 0.0 }
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

/// Builds UV triangles by iterating the polycube mesh faces directly,
/// subdividing faces that have boundary midpoints on their edges.
///
/// Boundary midpoints sit on the polygon boundary in UV space but aren't
/// mesh vertices, so raw mesh triangles don't reach the polygon boundary.
/// For each mesh face with boundary midpoints on its edges, we replace
/// the original triangle with sub-triangles that include the midpoints.
///
/// This avoids planar face extraction (which fails on coarse meshes where
/// the Tutte embedding has crossing edges).
fn build_uv_tris_from_mesh(
    polycube_vfg: &VirtualFlatGeometry,
    polycube_uv: &HashMap<NodeIndex, Vector2D>,
    polycube_skeleton: &LabeledCurveSkeleton,
    polycube_mesh: &Mesh<INPUT>,
    node_idx: NodeIndex,
) -> Vec<UvTriangle> {
    let patch_set: HashSet<VertID> = polycube_skeleton[node_idx]
        .skeleton_node
        .patch_vertices
        .iter()
        .copied()
        .collect();

    // Build a map from mesh edge (as unordered vert pair) to VFG midpoint node(s).
    // Includes both regular and cut-endpoint midpoints.
    let mut edge_to_midpoint: HashMap<(VertID, VertID), Vec<NodeIndex>> = HashMap::new();
    for node in polycube_vfg.graph.node_indices() {
        match &polycube_vfg.graph[node].origin {
            virtual_mesh::VirtualNodeOrigin::BoundaryMidpoint { edge, .. } => {
                let r = polycube_mesh.root(*edge);
                let t = polycube_mesh.toor(*edge);
                let key = if r < t { (r, t) } else { (t, r) };
                edge_to_midpoint.entry(key).or_default().push(node);
            }
            virtual_mesh::VirtualNodeOrigin::CutEndpointMidpoint { edge, .. } => {
                let r = polycube_mesh.root(*edge);
                let t = polycube_mesh.toor(*edge);
                let key = if r < t { (r, t) } else { (t, r) };
                edge_to_midpoint.entry(key).or_default().push(node);
            }
            _ => {}
        }
    }

    let mut tris = Vec::new();

    for face in polycube_mesh.face_ids() {
        let fv: Vec<VertID> = polycube_mesh.vertices(face).collect();
        if fv.len() < 3 || !fv.iter().all(|v| patch_set.contains(v)) {
            continue;
        }

        // Fan-triangulate the face (works for tris, quads, and general polygons).
        for i in 1..fv.len() - 1 {
            let idx = [0, i, i + 1];

            // Check which of the 3 sub-triangle edges have boundary midpoints.
            let tri_edges = [
                (fv[idx[0]], fv[idx[1]]),
                (fv[idx[1]], fv[idx[2]]),
                (fv[idx[2]], fv[idx[0]]),
            ];
            let mids: Vec<Option<&Vec<NodeIndex>>> = tri_edges
                .iter()
                .map(|&(a, b)| {
                    let key = if a < b { (a, b) } else { (b, a) };
                    edge_to_midpoint.get(&key)
                })
                .collect();

            let n_mids = mids.iter().filter(|m| m.is_some()).count();

            if n_mids == 0 {
                emit_tri(&mut tris, &fv, &idx, polycube_vfg, polycube_uv, polycube_mesh);
            } else {
                // Emit the base triangle first (covers interior).
                emit_tri(&mut tris, &fv, &idx, polycube_vfg, polycube_uv, polycube_mesh);
                // Also emit midpoint sub-triangles to cover boundary regions.
                for (edge_i, mid_opt) in mids.iter().enumerate() {
                    if let Some(mn_list) = mid_opt {
                        let ei0 = edge_i;
                        let ei1 = (edge_i + 1) % 3;
                        let ei2 = (edge_i + 2) % 3;
                        for &mn in *mn_list {
                            emit_tri_with_midpoint(
                                &mut tris, &fv, idx[ei1], idx[ei2], mn,
                                polycube_vfg, polycube_uv, polycube_mesh,
                            );
                            emit_tri_with_midpoint(
                                &mut tris, &fv, idx[ei0], idx[ei2], mn,
                                polycube_vfg, polycube_uv, polycube_mesh,
                            );
                        }
                    }
                }
            }
        }
    }

    // Step 2: Add border triangles for boundary midpoints.
    // Boundary midpoints sit on mesh edges between regions. The adjacent mesh face
    // on this region's side has vertices that include the edge endpoints (in the patch)
    // and additional vertices. We form triangles from the midpoint to the edge endpoints
    // and any additional patch vertices in the adjacent face.
    for node in polycube_vfg.graph.node_indices() {
        let mesh_edge = match &polycube_vfg.graph[node].origin {
            virtual_mesh::VirtualNodeOrigin::BoundaryMidpoint { edge, .. } => *edge,
            virtual_mesh::VirtualNodeOrigin::CutEndpointMidpoint { edge, .. } => *edge,
            _ => continue,
        };

        let Some(&mid_uv) = polycube_uv.get(&node) else { continue };
        let mid_pos = polycube_vfg.graph[node].position;

        let edge_root = polycube_mesh.root(mesh_edge);
        let edge_toor = polycube_mesh.toor(mesh_edge);

        // Check both faces adjacent to the boundary edge.
        let adj_faces = [
            polycube_mesh.face(mesh_edge),
            polycube_mesh.face(polycube_mesh.twin(mesh_edge)),
        ];
        for face in adj_faces {
            let fv: Vec<VertID> = polycube_mesh.vertices(face).collect();

            // Find vertices in this face that are NOT the edge endpoints AND are in the patch.
            let interior_verts: Vec<VertID> = fv
                .iter()
                .copied()
                .filter(|&v| v != edge_root && v != edge_toor && patch_set.contains(&v))
                .collect();

            if interior_verts.is_empty() {
                continue;
            }

            // For each interior vertex c, emit triangles (mid, root, c) and (mid, c, toor).
            for &c in &interior_verts {
                // (mid, edge_root, c)
                emit_border_tri(
                    &mut tris, node, mid_uv, mid_pos,
                    edge_root, c,
                    polycube_vfg, polycube_uv, polycube_mesh,
                );
                // (mid, c, edge_toor)
                emit_border_tri(
                    &mut tris, node, mid_uv, mid_pos,
                    c, edge_toor,
                    polycube_vfg, polycube_uv, polycube_mesh,
                );
            }
        }
    }

    let degree = polycube_skeleton.edges(node_idx).count();
    log::info!(
        "build_uv_tris_from_mesh: region {:?} degree {}: {} UV tris, {} patch_verts",
        node_idx, degree, tris.len(), patch_set.len(),
    );

    tris
}

/// Emit a UV triangle from mesh face vertices at indices `idx` in `fv`.
fn emit_tri(
    tris: &mut Vec<UvTriangle>,
    fv: &[VertID],
    idx: &[usize; 3],
    vfg: &VirtualFlatGeometry,
    uv: &HashMap<NodeIndex, Vector2D>,
    mesh: &Mesh<INPUT>,
) {
    let nodes: Vec<&Vec<NodeIndex>> = idx
        .iter()
        .filter_map(|&i| vfg.vert_to_nodes.get(&fv[i]))
        .collect();
    if nodes.len() != 3 {
        let missing: Vec<usize> = idx
            .iter()
            .filter(|&&i| !vfg.vert_to_nodes.contains_key(&fv[i]))
            .copied()
            .collect();
        log::warn!(
            "emit_tri: {}/{} verts missing from vert_to_nodes (missing indices: {:?}, fv={:?}, idx={:?})",
            3 - nodes.len(), 3, missing, fv, idx,
        );
        return;
    }
    // Diagnostic: count how many UV lookups succeed
    let mut uv_found = 0usize;
    let mut uv_total = 0usize;
    for &n0 in nodes[0] {
        for &n1 in nodes[1] {
            for &n2 in nodes[2] {
                uv_total += 1;
                if uv.get(&n0).is_some() && uv.get(&n1).is_some() && uv.get(&n2).is_some() {
                    uv_found += 1;
                }
            }
        }
    }
    if uv_found == 0 {
        log::warn!(
            "emit_tri: all {} combos have missing UVs. nodes[0]={:?} nodes[1]={:?} nodes[2]={:?}, uv has {} entries",
            uv_total, nodes[0], nodes[1], nodes[2], uv.len(),
        );
    }
    // Try all combos and pick the one with largest absolute area.
    // Accept either winding — we flip to positive if needed.
    // Track whether we flipped so pos array matches UV order.
    let mut best: Option<(f64, [Vector2D; 3], bool)> = None;
    for &n0 in nodes[0] {
        for &n1 in nodes[1] {
            for &n2 in nodes[2] {
                let Some(&uv0) = uv.get(&n0) else { continue };
                let Some(&uv1) = uv.get(&n1) else { continue };
                let Some(&uv2) = uv.get(&n2) else { continue };
                let area = (uv1.x - uv0.x) * (uv2.y - uv0.y)
                    - (uv2.x - uv0.x) * (uv1.y - uv0.y);
                let abs_area = area.abs();
                if abs_area > 1e-15 && best.as_ref().map_or(true, |(b, _, _)| abs_area > *b) {
                    if area > 0.0 {
                        best = Some((abs_area, [uv0, uv1, uv2], false));
                    } else {
                        best = Some((abs_area, [uv0, uv2, uv1], true)); // flip
                    }
                }
            }
        }
    }
    if let Some((_, uvs, flipped)) = best {
        let pos = if flipped {
            [mesh.position(fv[idx[0]]), mesh.position(fv[idx[2]]), mesh.position(fv[idx[1]])]
        } else {
            [mesh.position(fv[idx[0]]), mesh.position(fv[idx[1]]), mesh.position(fv[idx[2]])]
        };
        tris.push(UvTriangle { uv: uvs, pos });
    }
}

/// Emit a UV triangle with a midpoint node and two mesh face vertices.
fn emit_tri_with_midpoint(
    tris: &mut Vec<UvTriangle>,
    fv: &[VertID],
    vi0: usize,
    vi1: usize,
    mid_node: NodeIndex,
    vfg: &VirtualFlatGeometry,
    uv: &HashMap<NodeIndex, Vector2D>,
    mesh: &Mesh<INPUT>,
) {
    let Some(&mid_uv) = uv.get(&mid_node) else { return };
    let mid_pos = vfg.graph[mid_node].position;

    let nodes0 = match vfg.vert_to_nodes.get(&fv[vi0]) {
        Some(n) => n,
        None => return,
    };
    let nodes1 = match vfg.vert_to_nodes.get(&fv[vi1]) {
        Some(n) => n,
        None => return,
    };

    let mut best: Option<(f64, [Vector2D; 3], bool)> = None;
    for &n0 in nodes0 {
        for &n1 in nodes1 {
            let Some(&uv0) = uv.get(&n0) else { continue };
            let Some(&uv1) = uv.get(&n1) else { continue };
            let area = (uv0.x - mid_uv.x) * (uv1.y - mid_uv.y)
                - (uv1.x - mid_uv.x) * (uv0.y - mid_uv.y);
            let abs_area = area.abs();
            if abs_area > 1e-15 && best.as_ref().map_or(true, |(b, _, _)| abs_area > *b) {
                if area > 0.0 {
                    best = Some((abs_area, [mid_uv, uv0, uv1], false));
                } else {
                    best = Some((abs_area, [mid_uv, uv1, uv0], true));
                }
            }
        }
    }
    if let Some((_, uvs, flipped)) = best {
        let pos = if flipped {
            [mid_pos, mesh.position(fv[vi1]), mesh.position(fv[vi0])]
        } else {
            [mid_pos, mesh.position(fv[vi0]), mesh.position(fv[vi1])]
        };
        tris.push(UvTriangle { uv: uvs, pos });
    }
}

/// Emit a border UV triangle: midpoint + two mesh vertices.
/// Like `emit_tri_with_midpoint` but takes vertex IDs directly instead of face indices.
fn emit_border_tri(
    tris: &mut Vec<UvTriangle>,
    _mid_node: NodeIndex,
    mid_uv: Vector2D,
    mid_pos: Vector3D,
    va: VertID,
    vb: VertID,
    vfg: &VirtualFlatGeometry,
    uv: &HashMap<NodeIndex, Vector2D>,
    mesh: &Mesh<INPUT>,
) {
    let nodes_a = match vfg.vert_to_nodes.get(&va) {
        Some(n) => n,
        None => return,
    };
    let nodes_b = match vfg.vert_to_nodes.get(&vb) {
        Some(n) => n,
        None => return,
    };

    let mut best: Option<(f64, [Vector2D; 3], bool)> = None;
    for &na in nodes_a {
        for &nb in nodes_b {
            let Some(&uva) = uv.get(&na) else { continue };
            let Some(&uvb) = uv.get(&nb) else { continue };
            let area = (uva.x - mid_uv.x) * (uvb.y - mid_uv.y)
                - (uvb.x - mid_uv.x) * (uva.y - mid_uv.y);
            let abs_area = area.abs();
            if abs_area > 1e-15 && best.as_ref().map_or(true, |(b, _, _)| abs_area > *b) {
                if area > 0.0 {
                    best = Some((abs_area, [mid_uv, uva, uvb], false));
                } else {
                    best = Some((abs_area, [mid_uv, uvb, uva], true));
                }
            }
        }
    }
    if let Some((_, uvs, flipped)) = best {
        let pos = if flipped {
            [mid_pos, mesh.position(vb), mesh.position(va)]
        } else {
            [mid_pos, mesh.position(va), mesh.position(vb)]
        };
        tris.push(UvTriangle { uv: uvs, pos });
    }
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
    region_idx: NodeIndex,
    degree: usize,
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

    // Euler formula check: V - E + F = 2 (including outer face).
    // F_inner = F_total - 1 = (2 - V + E) - 1 = 1 - V + E.
    // But we skip the outer face via signed-area filtering, so expected = 1 - V + E.
    // NOTE: the user observed a consistent +1 in practice. This is because
    // the signed-area filter may miss the outer face in some configurations,
    // making the effective count 2 - V + E. We use that here.
    let n_v = vfg.graph.node_count() as i64;
    let n_e = vfg.graph.edge_count() as i64;
    let euler_inner = 2 - n_v + n_e;
    let found_inner = seen_faces.len() as i64;
    if found_inner != euler_inner {
        // Detect crossing edges to diagnose.
        let edges: Vec<(NodeIndex, NodeIndex)> = {
            use petgraph::visit::IntoEdgeReferences;
            vfg.graph
                .edge_references()
                .map(|e| (e.source(), e.target()))
                .collect()
        };
        let mut crossings = 0;
        for i in 0..edges.len() {
            for j in (i + 1)..edges.len() {
                let (a1, a2) = edges[i];
                let (b1, b2) = edges[j];
                // Skip edges that share an endpoint.
                if a1 == b1 || a1 == b2 || a2 == b1 || a2 == b2 {
                    continue;
                }
                if segments_cross(uv[&a1], uv[&a2], uv[&b1], uv[&b2]) {
                    crossings += 1;
                    if crossings <= 3 {
                        let boundary_set: HashSet<NodeIndex> =
                            vfg.boundary_loop.iter().copied().collect();
                        log::warn!(
                            "  crossing: ({:?}[{}]--{:?}[{}]) x ({:?}[{}]--{:?}[{}])",
                            a1,
                            if boundary_set.contains(&a1) { "B" } else { "I" },
                            a2,
                            if boundary_set.contains(&a2) { "B" } else { "I" },
                            b1,
                            if boundary_set.contains(&b1) { "B" } else { "I" },
                            b2,
                            if boundary_set.contains(&b2) { "B" } else { "I" },
                        );
                    }
                }
            }
        }
        log::warn!(
            "extract_uv_faces: region {:?} (degree {}): {} inner faces (Euler: {}), V={}, E={}, boundary={}, {} crossing edge pairs",
            region_idx, degree, found_inner, euler_inner,
            n_v, n_e, vfg.boundary_loop.len(), crossings,
        );
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
/// Returns `None` if the point is not inside any UV triangle (no fallback).
fn interpolate_in_uv_triangles(
    query: Vector2D,
    tris: &[UvTriangle],
    polycube_mesh: &Mesh<INPUT>,
) -> Option<Vector3D> {
    let mut best: Option<(Vector3D, f64)> = None;
    for tri in tris {
        if let Some((u, v, w)) = barycentric_2d(query, tri.uv[0], tri.uv[1], tri.uv[2]) {
            let min_coord = u.min(v).min(w);
            if min_coord >= -1e-6 {
                let pos = tri.pos[0] * u + tri.pos[1] * v + tri.pos[2] * w;
                let dist = distance_to_mesh_surface(pos, polycube_mesh);
                if best.is_none() || dist < best.as_ref().unwrap().1 {
                    best = Some((pos, dist));
                    // Early exit if already on the surface.
                    if dist < 1e-10 {
                        return Some(pos);
                    }
                }
            }
        }
    }
    best.map(|(pos, _)| pos)
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

/// Dot product of two 3D vectors.
fn dot3(a: Vector3D, b: Vector3D) -> f64 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

/// Squared distance from point `p` to triangle `(a, b, c)` in 3D.
/// Uses Voronoi-region based closest point (Ericson, "Real-Time Collision Detection").
fn point_to_triangle_dist_sq(p: Vector3D, a: Vector3D, b: Vector3D, c: Vector3D) -> f64 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = dot3(ab, ap);
    let d2 = dot3(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 { return dot3(ap, ap); }
    let bp = p - b;
    let d3 = dot3(ab, bp);
    let d4 = dot3(ac, bp);
    if d3 >= 0.0 && d4 <= d3 { return dot3(bp, bp); }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        let diff = p - (a + ab * v);
        return dot3(diff, diff);
    }
    let cp = p - c;
    let d5 = dot3(ab, cp);
    let d6 = dot3(ac, cp);
    if d6 >= 0.0 && d5 <= d6 { return dot3(cp, cp); }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        let diff = p - (a + ac * w);
        return dot3(diff, diff);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let diff = p - (b + (c - b) * w);
        return dot3(diff, diff);
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let diff = p - (a + ab * v + ac * w);
    dot3(diff, diff)
}

/// Minimum distance from `point` to the surface of `mesh`.
fn distance_to_mesh_surface(point: Vector3D, mesh: &Mesh<INPUT>) -> f64 {
    let mut min_dist_sq = f64::MAX;
    for face in mesh.face_ids() {
        let verts: Vec<VertID> = mesh.vertices(face).collect();
        for i in 1..verts.len() - 1 {
            let a = mesh.position(verts[0]);
            let b = mesh.position(verts[i]);
            let c = mesh.position(verts[i + 1]);
            min_dist_sq = min_dist_sq.min(point_to_triangle_dist_sq(point, a, b, c));
        }
    }
    min_dist_sq.sqrt()
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

    if degree >= 2 {
        log::info!(
            "parameterize_side: region {:?} degree {}, n_sides={}, boundary_loop={}, corners={}",
            node_idx, degree, n_sides, vfg.boundary_loop.len(), vfg.corner_indices.len(),
        );
    }

    // Solve the Dirichlet problem on the VFG graph.
    let uv_map = solve_dirichlet(&vfg, &boundary_positions);

    if degree >= 2 {
        let signed_area = compute_uv_boundary_signed_area(&vfg, &uv_map);
        let crossings = count_uv_crossings(&vfg, &uv_map);

        // For small graphs (polycube side), dump full structure
        if vfg.graph.node_count() <= 50 {
            dump_polycube_vfg(&vfg, &uv_map, mesh, node_idx, degree);
        }

        log::warn!(
            "Region {:?} (deg {}): signed_area={:.4} ({}) nodes={} edges={} boundary={} crossings={}",
            node_idx, degree, signed_area,
            if signed_area > 0.0 { "CCW" } else { "CW!" },
            vfg.graph.node_count(), vfg.graph.edge_count(),
            vfg.boundary_loop.len(), crossings,
        );
    }

    (vfg, uv_map)
}

/// Dumps the full VFG graph for a small region (polycube side).
fn dump_polycube_vfg(
    vfg: &VirtualFlatGeometry,
    uv: &HashMap<NodeIndex, Vector2D>,
    mesh: &Mesh<INPUT>,
    node_idx: NodeIndex,
    degree: usize,
) {
    use petgraph::visit::EdgeRef;

    let boundary_set: HashSet<NodeIndex> = vfg.boundary_loop.iter().copied().collect();

    log::warn!("=== POLYCUBE VFG DUMP region {:?} deg {} ===", node_idx, degree);

    // Check: are there quad faces in the mesh?
    let patch_verts: HashSet<VertID> = vfg.vert_to_nodes.keys().copied().collect();
    let mut tri_count = 0;
    let mut quad_count = 0;
    let mut other_count = 0;
    for face_id in mesh.face_ids() {
        let verts: Vec<VertID> = mesh.vertices(face_id).collect();
        // Check if face is in our patch (at least one vert in patch)
        if verts.iter().any(|v| patch_verts.contains(v)) {
            match verts.len() {
                3 => tri_count += 1,
                4 => quad_count += 1,
                n => { other_count += 1; log::warn!("  face with {} verts!", n); }
            }
        }
    }
    log::warn!("  Mesh faces touching patch: {} tris, {} quads, {} other", tri_count, quad_count, other_count);

    // Dump all nodes with UV positions
    log::warn!("  Nodes ({}):", vfg.graph.node_count());
    for ni in vfg.graph.node_indices() {
        let on_boundary = if boundary_set.contains(&ni) { "B" } else { "I" };
        let uv_pos = uv.get(&ni).map(|p| format!("({:.3},{:.3})", p.x, p.y)).unwrap_or("NONE".to_string());
        let deg = vfg.graph.edges(ni).count();
        let origin_short = match &vfg.graph[ni].origin {
            virtual_mesh::VirtualNodeOrigin::MeshVertex(v) => format!("MeshVert({:?})", v),
            virtual_mesh::VirtualNodeOrigin::BoundaryMidpoint { edge, boundary } =>
                format!("BdryMid(e={:?},b={:?})", edge, boundary),
            virtual_mesh::VirtualNodeOrigin::CutDuplicate { original, cut_index, side, .. } =>
                format!("CutDup(orig={:?},ci={},s={})", original, cut_index, if *side {"R"} else {"L"}),
            virtual_mesh::VirtualNodeOrigin::CutEndpointMidpoint { edge, cut_index, side, .. } =>
                format!("CutEpMid(e={:?},ci={},s={})", edge, cut_index, if *side {"R"} else {"L"}),
        };
        log::warn!("    {:?}[{}] deg={} uv={} {}", ni, on_boundary, deg, uv_pos, origin_short);
    }

    // Find and log crossing edge pairs
    let edges: Vec<(NodeIndex, NodeIndex)> = {
        use petgraph::visit::IntoEdgeReferences;
        vfg.graph
            .edge_references()
            .map(|e| (e.source(), e.target()))
            .collect()
    };
    let fmt_node = |n: NodeIndex| -> String {
        let b = if boundary_set.contains(&n) { "B" } else { "I" };
        let p = uv.get(&n).map(|p| format!("({:.2},{:.2})", p.x, p.y)).unwrap_or("?".to_string());
        format!("{:?}[{}]{}", n, b, p)
    };
    let mut crossing_count = 0;
    for i in 0..edges.len() {
        for j in (i + 1)..edges.len() {
            let (a1, a2) = edges[i];
            let (b1, b2) = edges[j];
            if a1 == b1 || a1 == b2 || a2 == b1 || a2 == b2 { continue; }
            let Some(&ua1) = uv.get(&a1) else { continue };
            let Some(&ua2) = uv.get(&a2) else { continue };
            let Some(&ub1) = uv.get(&b1) else { continue };
            let Some(&ub2) = uv.get(&b2) else { continue };
            if segments_cross(ua1, ua2, ub1, ub2) {
                crossing_count += 1;
                if crossing_count <= 10 {
                    log::warn!(
                        "  CROSSING: {} -- {} X {} -- {}",
                        fmt_node(a1), fmt_node(a2), fmt_node(b1), fmt_node(b2),
                    );
                }
            }
        }
    }
    log::warn!("  Total crossings: {}", crossing_count);

    // Dump boundary loop
    log::warn!("  Boundary loop ({} nodes): {:?}", vfg.boundary_loop.len(), vfg.boundary_loop);
    log::warn!("  Corner indices ({} corners): {:?}", vfg.corner_indices.len(), vfg.corner_indices);
}

/// Computes the signed area of the UV boundary polygon.
/// Positive = CCW (correct for Tutte), negative = CW (wrong).
fn compute_uv_boundary_signed_area(
    vfg: &VirtualFlatGeometry,
    uv: &HashMap<NodeIndex, Vector2D>,
) -> f64 {
    let bl = &vfg.boundary_loop;
    let n = bl.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..n {
        let Some(&a) = uv.get(&bl[i]) else { continue };
        let Some(&b) = uv.get(&bl[(i + 1) % n]) else { continue };
        area += a.x * b.y - b.x * a.y;
    }
    area * 0.5
}

/// Counts the number of crossing edge pairs in the UV embedding.
fn count_uv_crossings(
    vfg: &VirtualFlatGeometry,
    uv: &HashMap<NodeIndex, Vector2D>,
) -> usize {
    use petgraph::visit::EdgeRef;
    let edges: Vec<(NodeIndex, NodeIndex)> = {
        use petgraph::visit::IntoEdgeReferences;
        vfg.graph
            .edge_references()
            .map(|e| (e.source(), e.target()))
            .collect()
    };
    let mut crossings = 0;
    for i in 0..edges.len() {
        for j in (i + 1)..edges.len() {
            let (a1, a2) = edges[i];
            let (b1, b2) = edges[j];
            if a1 == b1 || a1 == b2 || a2 == b1 || a2 == b2 {
                continue;
            }
            let Some(&uv_a1) = uv.get(&a1) else { continue };
            let Some(&uv_a2) = uv.get(&a2) else { continue };
            let Some(&uv_b1) = uv.get(&b1) else { continue };
            let Some(&uv_b2) = uv.get(&b2) else { continue };
            if segments_cross(uv_a1, uv_a2, uv_b1, uv_b2) {
                crossings += 1;
            }
        }
    }
    crossings
}

/// Diagnoses boundary-to-boundary connections in the VFG.
///
/// For each VFG edge where BOTH endpoints are on the boundary loop, checks:
/// - Do the original mesh vertices share a mesh edge? (real vs synthetic)
/// - Are they on the SAME skeleton boundary or DIFFERENT boundaries?
/// - Are they consecutive in the boundary loop? (expected vs unexpected)
///
/// Also checks: for interior nodes that are very close to a boundary node in UV,
/// could they cause visual "boundary crossing" artifacts?
fn diagnose_boundary_connections(
    vfg: &VirtualFlatGeometry,
    skeleton: &LabeledCurveSkeleton,
    mesh: &Mesh<INPUT>,
    node_idx: NodeIndex,
) {
    use petgraph::visit::EdgeRef;

    let boundary_set: HashSet<NodeIndex> = vfg.boundary_loop.iter().copied().collect();

    // Build map: VFG boundary node -> which skeleton edge (boundary loop) it belongs to
    let mut node_to_skel_edge: HashMap<NodeIndex, EdgeIndex> = HashMap::new();
    for &ni in &vfg.boundary_loop {
        match &vfg.graph[ni].origin {
            virtual_mesh::VirtualNodeOrigin::BoundaryMidpoint { boundary, .. } => {
                node_to_skel_edge.insert(ni, *boundary);
            }
            virtual_mesh::VirtualNodeOrigin::CutEndpointMidpoint { boundary, .. } => {
                node_to_skel_edge.insert(ni, *boundary);
            }
            virtual_mesh::VirtualNodeOrigin::CutDuplicate { cut_index, .. } => {
                // Cut duplicates are on the cut, not on a specific boundary
                // Mark with a sentinel or skip
            }
            virtual_mesh::VirtualNodeOrigin::MeshVertex(_) => {
                // Interior vertex that ended up on boundary? Shouldn't happen for degree>=2
            }
        }
    }

    // Build set of consecutive boundary pairs for quick lookup
    let bl = &vfg.boundary_loop;
    let n = bl.len();
    let mut consecutive_pairs: HashSet<(NodeIndex, NodeIndex)> = HashSet::new();
    for i in 0..n {
        let a = bl[i];
        let b = bl[(i + 1) % n];
        consecutive_pairs.insert((a, b));
        consecutive_pairs.insert((b, a));
    }

    let mut same_boundary = 0usize;
    let mut cross_boundary = 0usize;
    let mut cut_to_boundary = 0usize;
    let mut consecutive_bb = 0usize;
    let mut non_consecutive_bb = 0usize;
    let mut bb_no_mesh_edge = 0usize;
    let mut total_bb = 0usize;

    for edge_ref in vfg.graph.edge_references() {
        let a = edge_ref.source();
        let b = edge_ref.target();
        if !boundary_set.contains(&a) || !boundary_set.contains(&b) {
            continue;
        }
        total_bb += 1;

        let is_consec = consecutive_pairs.contains(&(a, b));
        if is_consec {
            consecutive_bb += 1;
        } else {
            non_consecutive_bb += 1;
        }

        // Check mesh adjacency
        let va = origin_mesh_vertex(&vfg.graph[a].origin);
        let vb = origin_mesh_vertex(&vfg.graph[b].origin);
        let has_mesh_edge = match (va, vb) {
            (Some(va), Some(vb)) if va != vb => mesh.neighbors(va).any(|n| n == vb),
            (Some(_), Some(_)) => true, // same vertex (cut duplicates)
            _ => true, // involves midpoint, synthetic by nature
        };
        if !has_mesh_edge {
            bb_no_mesh_edge += 1;
        }

        // Classify by boundary membership
        let skel_a = node_to_skel_edge.get(&a);
        let skel_b = node_to_skel_edge.get(&b);
        match (skel_a, skel_b) {
            (Some(sa), Some(sb)) if sa == sb => same_boundary += 1,
            (Some(_), Some(_)) => cross_boundary += 1,
            _ => cut_to_boundary += 1, // one or both are cut duplicates
        }
    }

    log::warn!(
        "diagnose_boundary_connections: region {:?} (deg {}): {} total B-B edges",
        node_idx,
        skeleton.edges(node_idx).count(),
        total_bb,
    );
    log::warn!(
        "  consecutive: {}, non-consecutive: {}, no-mesh-edge: {}",
        consecutive_bb, non_consecutive_bb, bb_no_mesh_edge,
    );
    log::warn!(
        "  same-boundary: {}, cross-boundary: {}, cut-to-boundary: {}",
        same_boundary, cross_boundary, cut_to_boundary,
    );

    // Log details of cross-boundary connections (these should NOT exist on input mesh)
    if cross_boundary > 0 {
        let mut count = 0;
        for edge_ref in vfg.graph.edge_references() {
            let a = edge_ref.source();
            let b = edge_ref.target();
            if !boundary_set.contains(&a) || !boundary_set.contains(&b) {
                continue;
            }
            let skel_a = node_to_skel_edge.get(&a);
            let skel_b = node_to_skel_edge.get(&b);
            if let (Some(sa), Some(sb)) = (skel_a, skel_b) {
                if sa != sb {
                    count += 1;
                    if count <= 5 {
                        log::warn!(
                            "  CROSS-BOUNDARY edge: {:?} (skel {:?}) -- {:?} (skel {:?}), origins: {:?} / {:?}",
                            a, sa, b, sb,
                            vfg.graph[a].origin, vfg.graph[b].origin,
                        );
                    }
                }
            }
        }
    }

    // Log non-consecutive B-B edges that DO have mesh backing (these are the suspicious ones)
    if non_consecutive_bb > 0 {
        let mut count = 0;
        for edge_ref in vfg.graph.edge_references() {
            let a = edge_ref.source();
            let b = edge_ref.target();
            if !boundary_set.contains(&a) || !boundary_set.contains(&b) {
                continue;
            }
            if consecutive_pairs.contains(&(a, b)) {
                continue;
            }
            count += 1;
            if count <= 5 {
                log::warn!(
                    "  NON-CONSECUTIVE B-B edge: {:?} -- {:?}, origins: {:?} / {:?}",
                    a, b, vfg.graph[a].origin, vfg.graph[b].origin,
                );
            }
        }
    }
}

/// Interpolates a position along the boundary of a regular polygon, from
/// vertex `start` to vertex `end` (going forward through intermediate
/// vertices). `t` in `[0, 1)` maps proportionally across the arc.
///
/// When `start == end` (zero-length arc), returns `polygon[start]`.
/// When the arc spans one edge, this reduces to simple linear interpolation.
fn polygon_arc_interpolate(
    polygon: &[Vector2D],
    start: usize,
    end: usize,
    t: f64,
) -> Vector2D {
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

/// Checks whether the boundary loop is oriented correctly (CCW around the patch).
///
/// For each BoundaryMidpoint node on the VFG boundary, the underlying mesh half-edge
/// has a face on its left side (`mesh.face(he)`). If this face's vertices are all in the
/// patch, the boundary traversal has the patch interior on the LEFT, which is the correct
/// CCW orientation for Tutte embedding.
fn check_boundary_orientation(
    vfg: &VirtualFlatGeometry,
    skeleton: &LabeledCurveSkeleton,
    mesh: &Mesh<INPUT>,
    node_idx: NodeIndex,
) {
    let patch_set: HashSet<VertID> = skeleton[node_idx]
        .skeleton_node
        .patch_vertices
        .iter()
        .copied()
        .collect();

    let mut interior_left = 0usize; // face on LEFT of traversal is in the patch (CCW = correct)
    let mut interior_right = 0usize; // face on LEFT is NOT in the patch (CW = wrong)

    for &ni in &vfg.boundary_loop {
        if let virtual_mesh::VirtualNodeOrigin::BoundaryMidpoint { edge, .. } =
            &vfg.graph[ni].origin
        {
            // The stored edge is the half-edge as it was during boundary loop construction.
            // mesh.face(edge) gives the face to the LEFT of this half-edge.
            // For boundary faces (triangle mesh): 2 verts from one patch, 1 from other.
            // If majority are in our patch → patch is on the LEFT → CCW (correct).
            let face = mesh.face(*edge);
            let face_verts: Vec<VertID> = mesh.vertices(face).collect();
            let in_patch = face_verts.iter().filter(|v| patch_set.contains(v)).count();
            let out_of_patch = face_verts.len() - in_patch;

            if in_patch > out_of_patch {
                interior_left += 1; // patch majority on LEFT → CCW
            } else if out_of_patch > in_patch {
                interior_right += 1; // other patch majority on LEFT → CW
            }
            // Equal split (ambiguous) → skip
        }
    }

    let total = interior_left + interior_right;
    if total > 0 {
        log::warn!(
            "check_boundary_orientation: region {:?}: {} midpoints with patch on LEFT (CCW), \
             {} with patch on RIGHT (CW) — expect ALL on LEFT for correct Tutte embedding",
            node_idx,
            interior_left,
            interior_right,
        );
        if interior_right > interior_left {
            log::error!(
                "BOUNDARY IS CW (WRONG)! Region {:?} has {} CW vs {} CCW midpoints. \
                 The boundary loop is traversed in the wrong direction!",
                node_idx,
                interior_right,
                interior_left,
            );
        }
    }
}

/// Helper: extract the original mesh vertex from a VFG node origin, if it has one.
fn origin_mesh_vertex(origin: &virtual_mesh::VirtualNodeOrigin) -> Option<VertID> {
    match origin {
        virtual_mesh::VirtualNodeOrigin::MeshVertex(v) => Some(*v),
        virtual_mesh::VirtualNodeOrigin::CutDuplicate {
            original: SurfacePoint::OnVertex { vertex },
            ..
        } => Some(*vertex),
        _ => None, // boundary midpoints / cut-endpoint midpoints don't map to a single mesh vert
    }
}

/// Comprehensive analysis of VFG crossings. For each crossing edge pair,
/// determines WHY the crossing exists: wrong side assignment, synthetic edge,
/// or unexpected boundary-to-boundary connectivity.
fn analyze_vfg_crossings(
    vfg: &VirtualFlatGeometry,
    uv: &HashMap<NodeIndex, Vector2D>,
    mesh: &Mesh<INPUT>,
    node_idx: NodeIndex,
    degree: usize,
) {
    use petgraph::visit::EdgeRef;
    use virtual_mesh::VirtualNodeOrigin;

    let boundary_set: HashSet<NodeIndex> = vfg.boundary_loop.iter().copied().collect();

    // ── Check 1: Every VFG edge should correspond to a mesh adjacency ────
    // (Except for synthetic edges: midpoint-to-opposite-vertex, boundary-consecutive)
    let mut non_mesh_edges = 0usize;
    let mut boundary_boundary_count = 0usize;
    let mut bb_non_mesh = 0usize;

    for edge_ref in vfg.graph.edge_references() {
        let a = edge_ref.source();
        let b = edge_ref.target();
        let orig_a = &vfg.graph[a].origin;
        let orig_b = &vfg.graph[b].origin;

        let a_on_boundary = boundary_set.contains(&a);
        let b_on_boundary = boundary_set.contains(&b);

        // Check mesh adjacency.
        let va = origin_mesh_vertex(orig_a);
        let vb = origin_mesh_vertex(orig_b);
        let has_mesh_edge = if let (Some(va), Some(vb)) = (va, vb) {
            if va == vb {
                true // cut edge between duplicates of the same vertex
            } else {
                mesh.neighbors(va).any(|n| n == vb)
            }
        } else {
            // Involves a midpoint — synthetic by nature, skip mesh check
            true
        };

        if !has_mesh_edge {
            non_mesh_edges += 1;
            log::warn!(
                "  VFG edge {:?}--{:?} has NO mesh counterpart! origins: {:?} / {:?}",
                a, b, orig_a, orig_b
            );
        }

        if a_on_boundary && b_on_boundary {
            boundary_boundary_count += 1;
            if !has_mesh_edge {
                bb_non_mesh += 1;
            }
        }
    }

    // ── Check 2: Full crossing detection with classification ─────────────
    let edges: Vec<(NodeIndex, NodeIndex)> = vfg
        .graph
        .edge_references()
        .map(|e| (e.source(), e.target()))
        .collect();

    let mut crossings = 0usize;
    let mut crossing_involves_non_mesh = 0usize;
    let mut crossing_involves_bb = 0usize;

    // Classify crossing edges by the origin types involved
    let mut crossing_type_counts: HashMap<String, usize> = HashMap::new();

    for i in 0..edges.len() {
        for j in (i + 1)..edges.len() {
            let (a1, a2) = edges[i];
            let (b1, b2) = edges[j];
            if a1 == b1 || a1 == b2 || a2 == b1 || a2 == b2 {
                continue;
            }
            let Some(&uv_a1) = uv.get(&a1) else { continue };
            let Some(&uv_a2) = uv.get(&a2) else { continue };
            let Some(&uv_b1) = uv.get(&b1) else { continue };
            let Some(&uv_b2) = uv.get(&b2) else { continue };

            if !segments_cross(uv_a1, uv_a2, uv_b1, uv_b2) {
                continue;
            }
            crossings += 1;

            // Classify each endpoint
            let classify = |n: NodeIndex| -> &'static str {
                let on_b = boundary_set.contains(&n);
                match &vfg.graph[n].origin {
                    VirtualNodeOrigin::MeshVertex(_) => {
                        if on_b { "MeshVert[B]" } else { "MeshVert[I]" }
                    }
                    VirtualNodeOrigin::CutDuplicate { side, .. } => {
                        if *side { "CutDup(R)[B]" } else { "CutDup(L)[B]" }
                    }
                    VirtualNodeOrigin::BoundaryMidpoint { .. } => "BdryMid[B]",
                    VirtualNodeOrigin::CutEndpointMidpoint { side, .. } => {
                        if *side { "CutEpMid(R)[B]" } else { "CutEpMid(L)[B]" }
                    }
                }
            };

            let types = format!(
                "({}-{}) x ({}-{})",
                classify(a1), classify(a2), classify(b1), classify(b2)
            );
            *crossing_type_counts.entry(types.clone()).or_insert(0) += 1;

            // Check if any edge in this crossing pair is non-mesh
            let edge_a_mesh = {
                let va = origin_mesh_vertex(&vfg.graph[a1].origin);
                let vb = origin_mesh_vertex(&vfg.graph[a2].origin);
                if let (Some(va), Some(vb)) = (va, vb) {
                    va == vb || mesh.neighbors(va).any(|n| n == vb)
                } else {
                    true
                }
            };
            let edge_b_mesh = {
                let va = origin_mesh_vertex(&vfg.graph[b1].origin);
                let vb = origin_mesh_vertex(&vfg.graph[b2].origin);
                if let (Some(va), Some(vb)) = (va, vb) {
                    va == vb || mesh.neighbors(va).any(|n| n == vb)
                } else {
                    true
                }
            };
            if !edge_a_mesh || !edge_b_mesh {
                crossing_involves_non_mesh += 1;
            }

            let a1b = boundary_set.contains(&a1);
            let a2b = boundary_set.contains(&a2);
            let b1b = boundary_set.contains(&b1);
            let b2b = boundary_set.contains(&b2);
            if (a1b && a2b) || (b1b && b2b) {
                crossing_involves_bb += 1;
            }

            // Log first few crossings in detail
            if crossings <= 5 {
                log::warn!(
                    "  CROSSING #{}: {} | mesh_edges: ({}, {})",
                    crossings, types, edge_a_mesh, edge_b_mesh
                );
                // For cut duplicate endpoints, show which side they're on
                for &n in &[a1, a2, b1, b2] {
                    if let VirtualNodeOrigin::CutDuplicate { side, peer, .. } = &vfg.graph[n].origin
                    {
                        let peer_uv = peer.and_then(|p| uv.get(&p));
                        log::warn!(
                            "    {:?} side={} uv={:?} peer_uv={:?}",
                            n,
                            if *side { "R" } else { "L" },
                            uv.get(&n),
                            peer_uv
                        );
                    }
                }
            }
        }
    }

    // ── Summary ──────────────────────────────────────────────────────────
    log::warn!(
        "analyze_vfg_crossings: region {:?} (deg {}): {} crossings, {} non-mesh edges, \
         {} B-B edges ({} non-mesh), {} crossings involve non-mesh, {} involve B-B",
        node_idx,
        degree,
        crossings,
        non_mesh_edges,
        boundary_boundary_count,
        bb_non_mesh,
        crossing_involves_non_mesh,
        crossing_involves_bb,
    );
    for (types, count) in &crossing_type_counts {
        log::warn!("    crossing type: {} x{}", types, count);
    }
}

/// Tests whether two line segments (a1–a2) and (b1–b2) properly cross each other.
/// Returns `true` only for transversal intersections (not touching at endpoints or collinear overlap).
fn segments_cross(a1: Vector2D, a2: Vector2D, b1: Vector2D, b2: Vector2D) -> bool {
    let d1 = a2 - a1;
    let d2 = b2 - b1;

    let denom = d1.x * d2.y - d1.y * d2.x;
    if denom.abs() < 1e-12 {
        return false; // parallel or collinear
    }

    let d = b1 - a1;
    let t = (d.x * d2.y - d.y * d2.x) / denom;
    let u = (d.x * d1.y - d.y * d1.x) / denom;

    // Strictly interior intersection (exclude endpoints).
    let eps = 1e-9;
    t > eps && t < 1.0 - eps && u > eps && u < 1.0 - eps
}

/// Maps every node in `vfg.boundary_loop` to a 2D position on a regular `n_sides`-gon.
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

    let corners = &vfg.corner_indices;

    if !corners.is_empty() && corners.len() != n_sides {
        warn!(
            "map_boundary_to_polygon: corner count {} != n_sides {}, falling back to arc-length",
            corners.len(),
            n_sides,
        );
    }

    if !corners.is_empty() && corners.len() == n_sides {
        // Structured mapping: each segment of boundary nodes maps to one polygon side.
        // When adjacent cut endpoints produce empty segments (seg_len == 0), we
        // absorb those empty polygon sides into the next non-empty segment so
        // that its nodes span a wider polygon arc and no gap is left.
        let mut result = HashMap::new();

        // Pre-compute segment lengths for all sides.
        let seg_lens: Vec<usize> = (0..n_sides)
            .map(|side| {
                let seg_start = corners[side];
                let seg_end = corners[(side + 1) % n_sides];
                if seg_end > seg_start {
                    seg_end - seg_start
                } else if seg_end == seg_start {
                    0
                } else {
                    n - seg_start + seg_end
                }
            })
            .collect();

        for side in 0..n_sides {
            if seg_lens[side] == 0 {
                continue; // Empty side — absorbed by the next non-empty side.
            }

            // Walk backward through preceding empty sides to find where the
            // effective polygon arc starts.
            let mut effective_start = side;
            loop {
                let prev = (effective_start + n_sides - 1) % n_sides;
                if seg_lens[prev] != 0 || prev == side {
                    break;
                }
                effective_start = prev;
            }
            let effective_end = (side + 1) % n_sides;

            let seg_start = corners[side];
            let seg_len = seg_lens[side];

            // Compute arc lengths within this segment.
            let mut seg_arc: Vec<f64> = Vec::with_capacity(seg_len);
            let mut seg_total: f64 = 0.0;
            for j in 0..seg_len {
                let idx_a = (seg_start + j) % n;
                let idx_b = (seg_start + j + 1) % n;
                let a = vfg.graph[boundary[idx_a]].position;
                let b = vfg.graph[boundary[idx_b]].position;
                let len = (b - a).norm();
                seg_arc.push(len);
                seg_total += len;
            }

            // Place nodes along the (possibly extended) polygon arc.
            let mut cumulative: f64 = 0.0;
            for j in 0..seg_len {
                let idx = (seg_start + j) % n;
                let node = boundary[idx];
                let t = if seg_total > 1e-15 {
                    cumulative / seg_total
                } else {
                    j as f64 / seg_len as f64
                };
                let pos = polygon_arc_interpolate(&polygon, effective_start, effective_end, t);
                result.insert(node, pos);
                cumulative += seg_arc[j];
            }
        }

        // Verify all boundary nodes got assigned a position.
        let missing = boundary.iter().filter(|n| !result.contains_key(n)).count();
        if missing > 0 {
            warn!(
                "map_boundary_to_polygon: {} of {} boundary nodes missing after corner mapping!",
                missing, n,
            );
        }

        return result;
    }

    // Fallback: global arc-length distribution (degree 1, no corners).
    let mut seg_lengths: Vec<f64> = vec![0.0; n];
    let mut total: f64 = 0.0;
    for i in 0..n {
        let a = vfg.graph[boundary[i]].position;
        let b = vfg.graph[boundary[(i + 1) % n]].position;
        let len = (b - a).norm();
        seg_lengths[i] = len;
        total += len;
    }

    let mut result = HashMap::new();
    let mut cumulative: f64 = 0.0;

    for (i, &node) in boundary.iter().enumerate() {
        let t = cumulative / total; // in [0, 1)

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
