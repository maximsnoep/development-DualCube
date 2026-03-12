use std::collections::HashSet;

use mehsh::prelude::{HasEdges, HasPosition, HasVertices, Mesh, Vector3D};

use crate::prelude::{EdgeID, FaceID, VertID, INPUT};

use super::{SurfacePath, SurfacePoint};

/// Given a vertex-to-vertex path through the mesh, computes the geodesic
/// (shortest surface path) by:
/// 1. Extracting the triangle strip (channel) the path passes through
/// 2. Unfolding that strip into 2D
/// 3. Computing the straight-line path in the unfolded domain
/// 4. Mapping crossing points back to 3D surface edges
///
/// The result is a [`SurfacePath`] where interior points lie on triangle edges
/// at optimal positions, rather than being pinned to mesh vertices.
///
/// `prefix` and `suffix` are optional boundary-midpoint surface points to
/// prepend/append to the final path (the first/last points of the cut).
pub fn straighten_vertex_path(
    vertex_path: &[VertID],
    mesh: &Mesh<INPUT>,
    patch_set: &HashSet<VertID>,
    prefix: Option<SurfacePoint>,
    suffix: Option<SurfacePoint>,
) -> SurfacePath {
    if vertex_path.len() <= 1 {
        let mut points = Vec::new();
        if let Some(p) = prefix {
            points.push(p);
        }
        for &v in vertex_path {
            points.push(SurfacePoint::OnVertex { vertex: v });
        }
        if let Some(s) = suffix {
            points.push(s);
        }
        return SurfacePath { points };
    }

    // Step 1: Extract the triangle strip.
    let strip = extract_triangle_strip(vertex_path, mesh, patch_set);

    // If the strip extraction fails (degenerate cases), fall back to vertex path.
    if strip.is_empty() {
        return vertex_path_to_surface_path(vertex_path, prefix, suffix);
    }

    // Step 2+3: Unfold the strip and compute the straight-line path.
    let crossing_points = funnel_straighten(&strip, vertex_path, mesh, prefix, suffix);

    SurfacePath {
        points: crossing_points,
    }
}

/// A single triangle in the strip, storing the face and the "portal" edge
/// shared with the next triangle.
#[derive(Debug, Clone, Copy)]
struct StripTriangle {
    face: FaceID,
    /// The edge shared with the *next* triangle in the strip (the portal edge).
    /// This is the directed half-edge on `face`'s side. `None` for the last triangle.
    portal_edge: Option<EdgeID>,
}

/// Extracts the ordered sequence of faces that a vertex path passes through.
///
/// For each consecutive pair of vertices `(v_i, v_{i+1})` in the path, the shared
/// face is found. The result is a strip of triangles where consecutive faces share
/// a "portal" edge.
fn extract_triangle_strip(
    vertex_path: &[VertID],
    mesh: &Mesh<INPUT>,
    patch_set: &HashSet<VertID>,
) -> Vec<StripTriangle> {
    let mut strip = Vec::new();

    for i in 0..vertex_path.len() - 1 {
        let v_curr = vertex_path[i];
        let v_next = vertex_path[i + 1];

        // Find the shared face containing both v_curr and v_next that is *within*
        // the patch. For the triangle strip, we want a face that also contains
        // v_prev or v_next_next to form a connected strip. But the simplest
        // correct approach: pick any shared face in the patch.
        let (edge_ab, _edge_ba) = mesh
            .edge_between_verts(v_curr, v_next)
            .expect("Consecutive path vertices must be connected by an edge");

        let face = mesh.face(edge_ab);

        // Check if this face's vertices are in the patch; if not, try the twin side.
        let face = if face_in_patch(face, mesh, patch_set) {
            face
        } else {
            let twin_face = mesh.face(mesh.twin(edge_ab));
            if face_in_patch(twin_face, mesh, patch_set) {
                twin_face
            } else {
                // Both faces have vertices outside patch — unusual but possible
                // at patch boundary. Use the original face.
                face
            }
        };

        // Determine portal edge: the edge shared with the next segment.
        let portal_edge = if i + 2 < vertex_path.len() {
            // The portal is the edge from v_next that goes toward the next
            // vertex, specifically the half-edge on this face.
            find_portal_edge(face, v_next, v_curr, mesh)
        } else {
            None
        };

        strip.push(StripTriangle { face, portal_edge });
    }

    // For a path of length n, we also need one more triangle at the end if the
    // last vertex has a portal. But the strip already has n-1 entries which is
    // correct for n vertices.
    strip
}

/// Finds the portal edge: the edge of `face` that is incident to `v_next` but is
/// NOT the edge between `v_next` and `v_prev`. Returns `None` if no such edge
/// exists within the face (shouldn't happen for triangles).
fn find_portal_edge(
    face: FaceID,
    v_next: VertID,
    v_prev: VertID,
    mesh: &Mesh<INPUT>,
) -> Option<EdgeID> {
    for edge in mesh.edges(face) {
        let r = mesh.root(edge);
        let t = mesh.toor(edge);
        // We want an edge touching v_next but not the edge between v_prev and v_next.
        if (r == v_next || t == v_next) && r != v_prev && t != v_prev {
            // Return the half-edge on this face that starts at v_next.
            if r == v_next {
                return Some(edge);
            } else {
                // t == v_next, so the half-edge with root=v_next is the twin,
                // but we need the one on this face. Actually, for a triangle,
                // the edge from our face that has v_next as toor() is fine too.
                return Some(edge);
            }
        }
    }
    None
}

/// Checks whether all vertices of a face are in the patch set.
fn face_in_patch(face: FaceID, mesh: &Mesh<INPUT>, patch_set: &HashSet<VertID>) -> bool {
    mesh.vertices(face).all(|v| patch_set.contains(&v))
}


type Vec2 = nalgebra::Vector2<f64>;

/// A triangle unfolded into 2D, with the three vertex positions and their
/// original 3D vertex IDs.
#[derive(Debug, Clone)]
struct UnfoldedTriangle {
    /// 2D positions of the three vertices, in the same order as the face's vertices.
    positions: [Vec2; 3],
    /// The 3D vertex IDs, same order.
    vertex_ids: [VertID; 3],
}

/// Performs the funnel algorithm* on the triangle strip to find the geodesic path.
///
/// * = Instead of the full funnel algorithm (which is elegant but complex to implement
/// correctly), we use an iterative "unfold and straighten" approach:
///
/// 1. Unfold all triangles in the strip into a common 2D plane
/// 2. Draw a straight line from source to target in 2D
/// 3. Find where this line crosses each portal edge
/// 4. Map those crossing points back to 3D as `SurfacePoint::OnEdge`
fn funnel_straighten(
    strip: &[StripTriangle],
    vertex_path: &[VertID],
    mesh: &Mesh<INPUT>,
    prefix: Option<SurfacePoint>,
    suffix: Option<SurfacePoint>,
) -> Vec<SurfacePoint> {
    if strip.is_empty() {
        return vertex_path_to_surface_path(vertex_path, prefix, suffix).points;
    }

    // Unfold the strip into 2D.
    let unfolded = unfold_strip(strip, mesh);
    if unfolded.is_empty() {
        return vertex_path_to_surface_path(vertex_path, prefix, suffix).points;
    }

    // Source: position of the first vertex of the path in the first unfolded triangle.
    // Target: position of the last vertex of the path in the last unfolded triangle.
    let source_2d = find_vertex_in_unfolded(&unfolded[0], vertex_path[0]);
    let target_2d = find_vertex_in_unfolded(
        unfolded.last().unwrap(),
        *vertex_path.last().unwrap(),
    );

    let (Some(source_2d), Some(target_2d)) = (source_2d, target_2d) else {
        return vertex_path_to_surface_path(vertex_path, prefix, suffix).points;
    };

    // Collect portal edges in 2D: for each strip triangle (except the last),
    // the portal edge endpoints in 2D.
    let mut portals: Vec<(Vec2, Vec2, EdgeID)> = Vec::new();
    for (i, tri) in strip.iter().enumerate() {
        if let Some(portal_edge) = tri.portal_edge {
            let r3d = mesh.root(portal_edge);
            let t3d = mesh.toor(portal_edge);
            if let (Some(r2d), Some(t2d)) = (
                find_vertex_in_unfolded(&unfolded[i], r3d),
                find_vertex_in_unfolded(&unfolded[i], t3d),
            ) {
                portals.push((r2d, t2d, portal_edge));
            }
        }
    }

    // Compute straight-line crossings through portals.
    let mut result = Vec::new();

    if let Some(p) = prefix {
        result.push(p);
    }

    result.push(SurfacePoint::OnVertex {
        vertex: vertex_path[0],
    });

    for &(p_left, p_right, portal_edge) in &portals {
        if let Some(t) = line_segment_intersection(source_2d, target_2d, p_left, p_right) {
            // Clamp t to avoid degenerate endpoints exactly at vertices.
            let t_clamped = t.clamp(1e-10, 1.0 - 1e-10);
            result.push(SurfacePoint::OnEdge {
                edge: portal_edge,
                t: t_clamped,
            });
        }
    }

    result.push(SurfacePoint::OnVertex {
        vertex: *vertex_path.last().unwrap(),
    });

    if let Some(s) = suffix {
        result.push(s);
    }

    // If we got fewer crossings than expected, the straight line missed some
    // portals (can happen with non-convex channels). In that case, use a
    // segment-by-segment approach instead.
    if result.len() < portals.len() + 2 {
        return segment_straighten(strip, vertex_path, mesh, prefix, suffix);
    }

    result
}

/// Segment-by-segment straightening: for each triple of consecutive vertices,
/// unfold the two adjacent triangles and find the optimal crossing point on
/// the shared edge. This is less optimal than full-strip unfolding but handles
/// non-convex channels robustly.
fn segment_straighten(
    strip: &[StripTriangle],
    vertex_path: &[VertID],
    mesh: &Mesh<INPUT>,
    prefix: Option<SurfacePoint>,
    suffix: Option<SurfacePoint>,
) -> Vec<SurfacePoint> {
    let mut result = Vec::new();

    if let Some(p) = prefix {
        result.push(p);
    }

    result.push(SurfacePoint::OnVertex {
        vertex: vertex_path[0],
    });

    for i in 0..vertex_path.len().saturating_sub(2) {
        let v_prev = vertex_path[i];
        let v_curr = vertex_path[i + 1];
        let v_next = vertex_path[i + 2];

        if i >= strip.len() {
            // Not enough strip triangles; emit vertex directly.
            result.push(SurfacePoint::OnVertex { vertex: v_curr });
            continue;
        }

        let Some(portal_edge) = strip[i].portal_edge else {
            result.push(SurfacePoint::OnVertex { vertex: v_curr });
            continue;
        };

        // Unfold the two triangles sharing the portal edge into 2D.
        let face_a = strip[i].face;
        let face_b = if i + 1 < strip.len() {
            strip[i + 1].face
        } else {
            // No next triangle, emit vertex.
            result.push(SurfacePoint::OnVertex { vertex: v_curr });
            continue;
        };

        match unfold_two_triangles(face_a, face_b, portal_edge, mesh) {
            Some((tri_a, tri_b)) => {
                let p_prev = find_vertex_in_unfolded(&tri_a, v_prev);
                let p_next = find_vertex_in_unfolded(&tri_b, v_next);

                let r2d = find_vertex_in_unfolded(&tri_a, mesh.root(portal_edge));
                let t2d = find_vertex_in_unfolded(&tri_a, mesh.toor(portal_edge));

                if let (Some(p_prev), Some(p_next), Some(r2d), Some(t2d)) =
                    (p_prev, p_next, r2d, t2d)
                {
                    if let Some(t) = line_segment_intersection(p_prev, p_next, r2d, t2d) {
                        let t_clamped = t.clamp(1e-10, 1.0 - 1e-10);
                        result.push(SurfacePoint::OnEdge {
                            edge: portal_edge,
                            t: t_clamped,
                        });
                        continue;
                    }
                }
                // Intersection failed; fall back to vertex.
                result.push(SurfacePoint::OnVertex { vertex: v_curr });
            }
            None => {
                result.push(SurfacePoint::OnVertex { vertex: v_curr });
            }
        }
    }

    if vertex_path.len() >= 2 {
        result.push(SurfacePoint::OnVertex {
            vertex: *vertex_path.last().unwrap(),
        });
    }

    if let Some(s) = suffix {
        result.push(s);
    }

    result
}


/// Unfolds an entire triangle strip into a common 2D coordinate system.
///
/// The first triangle is placed with its first edge along the x-axis.
/// Each subsequent triangle is unfolded by reflecting across the shared
/// portal edge with the previous triangle.
fn unfold_strip(strip: &[StripTriangle], mesh: &Mesh<INPUT>) -> Vec<UnfoldedTriangle> {
    let mut unfolded = Vec::with_capacity(strip.len());

    if strip.is_empty() {
        return unfolded;
    }

    // Place the first triangle.
    let first = &strip[0];
    let verts: Vec<VertID> = mesh.vertices(first.face).collect();
    if verts.len() != 3 {
        return unfolded;
    }

    let p0 = mesh.position(verts[0]);
    let p1 = mesh.position(verts[1]);
    let p2 = mesh.position(verts[2]);

    let first_unfolded = place_first_triangle(verts[0], verts[1], verts[2], p0, p1, p2);
    unfolded.push(first_unfolded);

    // Unfold subsequent triangles across portal edges.
    for i in 0..strip.len() - 1 {
        let Some(portal_edge) = strip[i].portal_edge else {
            break;
        };

        let next_face = strip[i + 1].face;
        let next_verts: Vec<VertID> = mesh.vertices(next_face).collect();
        if next_verts.len() != 3 {
            break;
        }

        let portal_root = mesh.root(portal_edge);
        let portal_toor = mesh.toor(portal_edge);

        // Find the 2D positions of the portal endpoints in the current unfolded triangle.
        let prev_unfolded = &unfolded[i];
        let pr_2d = find_vertex_in_unfolded(prev_unfolded, portal_root);
        let pt_2d = find_vertex_in_unfolded(prev_unfolded, portal_toor);

        let (Some(pr_2d), Some(pt_2d)) = (pr_2d, pt_2d) else {
            break;
        };

        // The new vertex is the one in next_face that isn't on the portal.
        let apex = next_verts
            .iter()
            .copied()
            .find(|&v| v != portal_root && v != portal_toor);
        let Some(apex) = apex else {
            break;
        };

        // Compute the apex's 2D position by unfolding.
        let apex_3d = mesh.position(apex);
        let pr_3d = mesh.position(portal_root);
        let pt_3d = mesh.position(portal_toor);

        let apex_2d = unfold_point_across_edge(pr_2d, pt_2d, pr_3d, pt_3d, apex_3d);

        // Build the unfolded triangle with correct vertex ordering.
        let mut positions = [Vec2::zeros(); 3];
        let mut vertex_ids = [next_verts[0]; 3];
        for (j, &v) in next_verts.iter().enumerate() {
            vertex_ids[j] = v;
            if v == portal_root {
                positions[j] = pr_2d;
            } else if v == portal_toor {
                positions[j] = pt_2d;
            } else {
                positions[j] = apex_2d;
            }
        }

        unfolded.push(UnfoldedTriangle {
            positions,
            vertex_ids,
        });
    }

    unfolded
}

/// Unfolds two specific triangles sharing an edge into the same 2D space.
fn unfold_two_triangles(
    face_a: FaceID,
    face_b: FaceID,
    shared_edge: EdgeID,
    mesh: &Mesh<INPUT>,
) -> Option<(UnfoldedTriangle, UnfoldedTriangle)> {
    let verts_a: Vec<VertID> = mesh.vertices(face_a).collect();
    let verts_b: Vec<VertID> = mesh.vertices(face_b).collect();
    if verts_a.len() != 3 || verts_b.len() != 3 {
        return None;
    }

    let pr = mesh.root(shared_edge);
    let pt = mesh.toor(shared_edge);

    // Place face_a first.
    let p0 = mesh.position(verts_a[0]);
    let p1 = mesh.position(verts_a[1]);
    let p2 = mesh.position(verts_a[2]);
    let tri_a = place_first_triangle(verts_a[0], verts_a[1], verts_a[2], p0, p1, p2);

    // Find portal endpoints in tri_a.
    let pr_2d = find_vertex_in_unfolded(&tri_a, pr)?;
    let pt_2d = find_vertex_in_unfolded(&tri_a, pt)?;

    // Find the apex of face_b.
    let apex = verts_b.iter().copied().find(|&v| v != pr && v != pt)?;
    let apex_3d = mesh.position(apex);
    let pr_3d = mesh.position(pr);
    let pt_3d = mesh.position(pt);
    let apex_2d = unfold_point_across_edge(pr_2d, pt_2d, pr_3d, pt_3d, apex_3d);

    let mut positions_b = [Vec2::zeros(); 3];
    let mut vertex_ids_b = [verts_b[0]; 3];
    for (j, &v) in verts_b.iter().enumerate() {
        vertex_ids_b[j] = v;
        if v == pr {
            positions_b[j] = pr_2d;
        } else if v == pt {
            positions_b[j] = pt_2d;
        } else {
            positions_b[j] = apex_2d;
        }
    }

    let tri_b = UnfoldedTriangle {
        positions: positions_b,
        vertex_ids: vertex_ids_b,
    };

    Some((tri_a, tri_b))
}

/// Places the first triangle in the 2D coordinate system.
/// v0 is at the origin, v1 is along the positive x-axis.
fn place_first_triangle(
    v0: VertID,
    v1: VertID,
    v2: VertID,
    p0: Vector3D,
    p1: Vector3D,
    p2: Vector3D,
) -> UnfoldedTriangle {
    let d01 = (p1 - p0).norm();
    let d02 = (p2 - p0).norm();
    let d12 = (p2 - p1).norm();

    // Place v0 at origin, v1 along x-axis.
    let u0 = Vec2::new(0.0, 0.0);
    let u1 = Vec2::new(d01, 0.0);

    // v2 via distances (law of cosines).
    let cos_angle = if d01 > 0.0 && d02 > 0.0 {
        ((d01 * d01 + d02 * d02 - d12 * d12) / (2.0 * d01 * d02)).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let sin_angle = (1.0 - cos_angle * cos_angle).sqrt();
    let u2 = Vec2::new(d02 * cos_angle, d02 * sin_angle);

    UnfoldedTriangle {
        positions: [u0, u1, u2],
        vertex_ids: [v0, v1, v2],
    }
}

/// Unfolds a point across a shared edge into the 2D coordinate system.
///
/// Given the 2D positions of the shared edge endpoints (e_a, e_b) and their 3D
/// positions, plus the 3D position of the point to unfold, computes its 2D position
/// such that distances to the shared edge endpoints are preserved and the point is
/// on the opposite side of the edge from the previous triangle.
fn unfold_point_across_edge(
    e_a_2d: Vec2,
    e_b_2d: Vec2,
    e_a_3d: Vector3D,
    e_b_3d: Vector3D,
    point_3d: Vector3D,
) -> Vec2 {
    let d_a = (point_3d - e_a_3d).norm();
    let d_b = (point_3d - e_b_3d).norm();
    let d_ab = (e_b_2d - e_a_2d).norm();

    if d_ab < 1e-15 {
        return e_a_2d;
    }

    // Parameterize along the edge direction.
    let edge_dir = (e_b_2d - e_a_2d) / d_ab;
    let edge_normal = Vec2::new(-edge_dir.y, edge_dir.x);

    // Project: foot of perpendicular from the new point onto the edge line.
    let cos_a = if d_a > 0.0 && d_ab > 0.0 {
        ((d_a * d_a + d_ab * d_ab - d_b * d_b) / (2.0 * d_a * d_ab)).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let foot_dist = d_a * cos_a; // signed distance from e_a along edge
    let perp_dist = d_a * (1.0 - cos_a * cos_a).sqrt();

    // Place on the "negative" side of the edge normal (opposite to the previous
    // triangle's apex) to unfold correctly.
    e_a_2d + edge_dir * foot_dist - edge_normal * perp_dist
}

/// Finds the 2D position of a vertex in an unfolded triangle. Returns `None`
/// if the vertex is not one of the triangle's vertices.
fn find_vertex_in_unfolded(tri: &UnfoldedTriangle, vertex: VertID) -> Option<Vec2> {
    for i in 0..3 {
        if tri.vertex_ids[i] == vertex {
            return Some(tri.positions[i]);
        }
    }
    None
}


/// Computes the intersection of line segment (a→b) with line segment (c→d).
///
/// Returns the parameter `t ∈ [0, 1]` along (c→d) where the intersection occurs,
/// or `None` if the segments don't intersect.
fn line_segment_intersection(a: Vec2, b: Vec2, c: Vec2, d: Vec2) -> Option<f64> {
    let ab = b - a;
    let cd = d - c;
    let ac = c - a;

    let denom = ab.x * cd.y - ab.y * cd.x;
    if denom.abs() < 1e-15 {
        return None; // Parallel or collinear.
    }

    let s = (ac.x * cd.y - ac.y * cd.x) / denom; // Parameter on (a→b)
    let t = (ac.x * ab.y - ac.y * ab.x) / denom; // Parameter on (c→d)

    // We need the line (a→b) to actually cross, but we're more lenient on s
    // since it represents the source-to-target line through the entire strip.
    // t must be in [0, 1] since it's the portal edge parameter.
    if t >= -1e-10 && t <= 1.0 + 1e-10 && s >= -1e-10 && s <= 1.0 + 1e-10 {
        Some(t.clamp(0.0, 1.0))
    } else {
        None
    }
}


fn vertex_path_to_surface_path(
    vertex_path: &[VertID],
    prefix: Option<SurfacePoint>,
    suffix: Option<SurfacePoint>,
) -> SurfacePath {
    let mut points = Vec::new();
    if let Some(p) = prefix {
        points.push(p);
    }
    for &v in vertex_path {
        points.push(SurfacePoint::OnVertex { vertex: v });
    }
    if let Some(s) = suffix {
        points.push(s);
    }
    SurfacePath { points }
}
