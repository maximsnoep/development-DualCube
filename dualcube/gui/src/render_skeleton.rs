use crate::colors;
use crate::render::world_to_view;
use bevy::prelude::{Color, *};
use dualcube::polycube::POLYCUBE;
use dualcube::prelude::*;
use dualcube::skeleton::curve_skeleton::CurveSkeletonSpatial;
use dualcube::skeleton::generate_loops::{CrossingMap, FacePointMap};
use dualcube::skeleton::orthogonalize::{AxisSign, LabeledCurveSkeleton};
use itertools::Itertools;
use mehsh::integrations::bevy::MeshBuilder;
use mehsh::prelude::*;
use std::collections::HashMap;

/// Builds a vertex-to-region map for the input mesh from a CurveSkeleton.
fn build_vertex_to_region_map(curve_skeleton: &CurveSkeleton) -> HashMap<VertKey<INPUT>, usize> {
    let mut map = HashMap::new();
    for (compact_id, node_idx) in curve_skeleton.node_indices().enumerate() {
        for &vert_key in &curve_skeleton[node_idx].patch_vertices {
            map.insert(vert_key, compact_id);
        }
    }
    map
}

/// Builds a vertex-to-region map for the polycube mesh from a LabeledCurveSkeleton.
/// Converts VertID back to VertKey<POLYCUBE> via raw key.
fn build_polycube_vertex_to_region_map(
    labeled_skeleton: &LabeledCurveSkeleton,
) -> HashMap<VertKey<POLYCUBE>, usize> {
    let mut map = HashMap::new();
    for (compact_id, node_idx) in labeled_skeleton.node_indices().enumerate() {
        for &vert_id in &labeled_skeleton[node_idx].skeleton_node.patch_vertices {
            map.insert(VertKey::<POLYCUBE>::new(vert_id.raw()), compact_id);
        }
    }
    map
}

/// Creates gizmos for patch boundaries by connecting midpoints of edges that the boundary crosses.
pub fn create_patch_boundary_gizmos(
    curve_skeleton: &CurveSkeleton,
    mesh: &mehsh::prelude::Mesh<INPUT>,
    translation: Vector3D,
    scale: f64,
) -> GizmoAsset {
    let vertex_to_region = build_vertex_to_region_map(curve_skeleton);
    boundary_gizmos_from_regions(mesh, &vertex_to_region, translation, scale)
}

/// Creates gizmos for visualizing a curve skeleton with spheres for nodes and lines for edges.
pub fn create_skeleton_gizmos(
    curve_skeleton: &CurveSkeleton,
    translation: Vector3D,
    scale: f64,
) -> GizmoAsset {
    let (mut gizmos, skel_color, node_radius) = setup_skeleton_gizmos();

    // Create edges
    create_skeleton_edge_gizmos(curve_skeleton, translation, scale, &mut gizmos, skel_color);

    // Create nodes
    create_skeleton_node_gizmos(
        curve_skeleton,
        translation,
        scale,
        &mut gizmos,
        skel_color,
        node_radius,
    );

    gizmos
}

/// Creates gizmos for visualizing a curve skeleton with spheres for nodes and lines for edges, using labels.
pub fn create_labeled_skeleton_gizmos(
    labeled_skeleton: &LabeledCurveSkeleton,
    translation: Vector3D,
    scale: f64,
) -> GizmoAsset {
    let (mut gizmos, skel_color, node_radius) = setup_skeleton_gizmos();

    // Create edges, use labels for coloring
    create_labeled_skeleton_edge_gizmos(labeled_skeleton, translation, scale, &mut gizmos);

    // Create nodes
    create_skeleton_node_gizmos_from_labeled(
        labeled_skeleton,
        translation,
        scale,
        &mut gizmos,
        skel_color,
        node_radius,
    );

    gizmos
}

fn setup_skeleton_gizmos() -> (GizmoAsset, Color, f32) {
    let gizmos = GizmoAsset::new();
    let skel_color = colors::to_bevy(colors::LIGHT_GRAY);
    let node_radius = 0.2;
    (gizmos, skel_color, node_radius)
}

fn create_labeled_skeleton_edge_gizmos(
    labeled_skeleton: &LabeledCurveSkeleton,
    translation: Vector3D,
    scale: f64,
    gizmos: &mut GizmoAsset,
) {
    for edge in labeled_skeleton.edge_indices() {
        let (a, b) = labeled_skeleton.edge_endpoints(edge).unwrap();
        let pos_a = labeled_skeleton[a].skeleton_node.position;
        let pos_b = labeled_skeleton[b].skeleton_node.position;
        let a_view = world_to_view(pos_a, translation, scale);
        let b_view = world_to_view(pos_b, translation, scale);
        let color = colors::to_bevy(colors::from_direction(
            labeled_skeleton[edge].direction,
            None,
            None,
        ));
        gizmos.line(a_view, b_view, color);
    }
}

fn create_skeleton_edge_gizmos(
    curve_skeleton: &CurveSkeleton,
    translation: Vector3D,
    scale: f64,
    gizmos: &mut GizmoAsset,
    skel_color: Color,
) {
    for edge in curve_skeleton.edge_indices() {
        let (a, b) = curve_skeleton.edge_endpoints(edge).unwrap();
        let pos_a = curve_skeleton[a].position;
        let pos_b = curve_skeleton[b].position;
        let a_view = world_to_view(pos_a, translation, scale);
        let b_view = world_to_view(pos_b, translation, scale);
        gizmos.line(a_view, b_view, skel_color);
    }
}

fn create_skeleton_node_gizmos(
    curve_skeleton: &CurveSkeleton,
    translation: Vector3D,
    scale: f64,
    gizmos: &mut GizmoAsset,
    skel_color: Color,
    node_radius: f32,
) {
    for node_idx in curve_skeleton.node_indices() {
        let pos = curve_skeleton[node_idx].position;
        let center = world_to_view(pos, translation, scale);
        gizmos.sphere(
            Isometry3d::from_translation(center),
            node_radius,
            skel_color,
        );
    }
}

// TODO: could likely just use a trait to unify.
fn create_skeleton_node_gizmos_from_labeled(
    labeled_skeleton: &LabeledCurveSkeleton,
    translation: Vector3D,
    scale: f64,
    gizmos: &mut GizmoAsset,
    skel_color: Color,
    node_radius: f32,
) {
    for node_idx in labeled_skeleton.node_indices() {
        let pos = labeled_skeleton[node_idx].skeleton_node.position;
        let center = world_to_view(pos, translation, scale);
        gizmos.sphere(
            Isometry3d::from_translation(center),
            node_radius,
            skel_color,
        );
    }
}

use bevy::color::palettes::tailwind;

/// Tailwind 500-level colors for patch visualization.
/// Chosen as they should be visually distinct. Grey-like ones removed.
const TAILWIND_500: [bevy::color::Srgba; 17] = [
    tailwind::RED_500,
    tailwind::CYAN_500,
    tailwind::YELLOW_500,
    tailwind::PURPLE_500,
    tailwind::GREEN_500,
    tailwind::PINK_500,
    tailwind::BLUE_500,
    tailwind::ORANGE_500,
    tailwind::TEAL_500,
    tailwind::FUCHSIA_500,
    tailwind::LIME_500,
    tailwind::INDIGO_500,
    tailwind::AMBER_500,
    tailwind::EMERALD_500,
    tailwind::VIOLET_500,
    tailwind::SKY_500,
    tailwind::ROSE_500,
];

/// Gets a color for a region index using Tailwind colors with chroma reduction for cycling.
pub fn get_region_color(region: usize) -> [f32; 3] {
    let base_idx = region % TAILWIND_500.len();
    let cycle = region / TAILWIND_500.len();

    let base = TAILWIND_500[base_idx];

    if cycle == 0 {
        [base.red, base.green, base.blue]
    } else {
        // Reduce chroma and lighten for subsequent cycles
        let mut lch: bevy::color::Lcha = base.into();
        lch.chroma *= 0.5_f32.powi(cycle as i32);
        lch.lightness = (lch.lightness + 0.1 * cycle as f32).min(0.95);
        let srgb: bevy::color::Srgba = lch.into();
        [srgb.red, srgb.green, srgb.blue]
    }
}

/// Shared helper: builds a Bevy mesh coloring patches by a per-region color function.
/// Works for any mesh type — triangles are split at boundary midpoints, quads are split
/// into two sub-quads along the boundary.
fn build_region_mesh<T: Tag, F>(
    mesh: &mehsh::prelude::Mesh<T>,
    vertex_to_region: &HashMap<VertKey<T>, usize>,
    translation: Vector3D,
    scale: f64,
    region_color_fn: F,
) -> bevy::mesh::Mesh
where
    F: Fn(usize) -> [f32; 3],
{
    let mut builder = MeshBuilder::default();

    let mut add_vertex = |pos: Vector3D, normal: Vector3D, color: &[f32; 3]| {
        let transformed_pos = pos * scale + translation;
        builder.add_vertex(&transformed_pos, &normal, color);
    };

    for face_id in mesh.face_ids() {
        let face_verts: Vec<_> = mesh.vertices(face_id).collect();
        let face_normal = mesh.normal(face_id);

        if face_verts.len() == 3 {
            let v0 = face_verts[0];
            let v1 = face_verts[1];
            let v2 = face_verts[2];

            let p0 = mesh.position(v0);
            let p1 = mesh.position(v1);
            let p2 = mesh.position(v2);

            let r0 = vertex_to_region.get(&v0).copied();
            let r1 = vertex_to_region.get(&v1).copied();
            let r2 = vertex_to_region.get(&v2).copied();

            match (r0, r1, r2) {
                (Some(r0), Some(r1), Some(r2)) => {
                    if r0 == r1 && r1 == r2 {
                        let color = region_color_fn(r0);
                        add_vertex(p0, face_normal, &color);
                        add_vertex(p1, face_normal, &color);
                        add_vertex(p2, face_normal, &color);
                    } else if r0 == r1 {
                        split_triangle(
                            &mut add_vertex,
                            p0, p1, p2,
                            face_normal, face_normal, face_normal,
                            r0, r2, &region_color_fn,
                        );
                    } else if r1 == r2 {
                        split_triangle(
                            &mut add_vertex,
                            p1, p2, p0,
                            face_normal, face_normal, face_normal,
                            r1, r0, &region_color_fn,
                        );
                    } else if r0 == r2 {
                        split_triangle(
                            &mut add_vertex,
                            p2, p0, p1,
                            face_normal, face_normal, face_normal,
                            r0, r1, &region_color_fn,
                        );
                    } else {
                        unreachable!(
                            "Triangle with all three vertices in different regions encountered."
                        );
                    }
                }
                _ => {}
            }
        } else if face_verts.len() == 4 {
            let v0 = face_verts[0];
            let v1 = face_verts[1];
            let v2 = face_verts[2];
            let v3 = face_verts[3];
            let p0 = mesh.position(v0);
            let p1 = mesh.position(v1);
            let p2 = mesh.position(v2);
            let p3 = mesh.position(v3);
            let r0 = vertex_to_region.get(&v0).copied();
            let r1 = vertex_to_region.get(&v1).copied();
            let r2 = vertex_to_region.get(&v2).copied();
            let r3 = vertex_to_region.get(&v3).copied();

            match (r0, r1, r2, r3) {
                (Some(a), Some(b), Some(c), Some(d)) if a == b && b == c && c == d => {
                    let color = region_color_fn(a);
                    add_vertex(p0, face_normal, &color);
                    add_vertex(p1, face_normal, &color);
                    add_vertex(p2, face_normal, &color);

                    add_vertex(p0, face_normal, &color);
                    add_vertex(p2, face_normal, &color);
                    add_vertex(p3, face_normal, &color);
                }
                (Some(a), Some(b), Some(c), Some(d)) => {
                    let regions = [(p0, a), (p1, b), (p2, c), (p3, d)];
                    split_quad(&mut add_vertex, &regions, face_normal, &region_color_fn);
                }
                _ => {}
            }
        }
    }

    builder.build()
}

/// Creates a Bevy mesh for visualizing surface patches as filled triangles.
pub fn create_patch_mesh(
    curve_skeleton: &CurveSkeleton,
    mesh: &mehsh::prelude::Mesh<INPUT>,
    translation: Vector3D,
    scale: f64,
) -> bevy::mesh::Mesh {
    let vertex_to_region = build_vertex_to_region_map(curve_skeleton);
    build_region_mesh(mesh, &vertex_to_region, translation, scale, get_region_color)
}

/// Splits a triangle where vertices a,b belong to region_x and vertex c belongs to region_y.
/// Creates:
/// - (a, b, ac) and (b, bc, ac) for region X
/// - (ac, bc, c) for region Y
/// where ac = midpoint(a,c) and bc = midpoint(b,c)
/// Maintains the same winding order as the original triangle (a, b, c).
fn split_triangle<F, C>(
    add_vertex: &mut F,
    pa: Vector3D,
    pb: Vector3D,
    pc: Vector3D,
    na: Vector3D,
    nb: Vector3D,
    nc: Vector3D,
    region_x: usize,
    region_y: usize,
    region_color: &C,
) where
    F: FnMut(Vector3D, Vector3D, &[f32; 3]),
    C: Fn(usize) -> [f32; 3],
{
    // Compute midpoints
    let p_ac = (pa + pc) * 0.5;
    let p_bc = (pb + pc) * 0.5;

    // Interpolate normals at midpoints for mesh
    let n_ac = ((na + nc) * 0.5).normalize();
    let n_bc = ((nb + nc) * 0.5).normalize();

    let color_x = region_color(region_x);
    let color_y = region_color(region_y);

    // Triangle 1 for region X: (a, b, ac)
    add_vertex(pa, na, &color_x);
    add_vertex(pb, nb, &color_x);
    add_vertex(p_ac, n_ac, &color_x);

    // Triangle 2 for region X: (b, bc, ac)
    add_vertex(pb, nb, &color_x);
    add_vertex(p_bc, n_bc, &color_x);
    add_vertex(p_ac, n_ac, &color_x);

    // Triangle for region Y: (ac, bc, c)
    add_vertex(p_ac, n_ac, &color_y);
    add_vertex(p_bc, n_bc, &color_y);
    add_vertex(pc, nc, &color_y);
}

/// Creates a colored Bevy mesh visualizing patch convexity (red=non-convex, green=convex).
pub fn create_patch_convexity_mesh(
    curve_skeleton: &CurveSkeleton,
    mesh: &mehsh::prelude::Mesh<INPUT>,
    translation: Vector3D,
    scale: f64,
) -> bevy::mesh::Mesh {
    let mut region_scores: HashMap<usize, f64> = HashMap::new();
    for (compact_id, node_idx) in curve_skeleton.node_indices().enumerate() {
        let mut score = curve_skeleton.patch_convexity_score(node_idx, mesh);
        // info!("Convexity score for node {:?}: {:.3}", node_idx, score);
        if !score.is_finite() {
            score = 0.0;
        }
        region_scores.insert(compact_id, score.clamp(0.0, 1.0));
    }

    let vertex_to_region = build_vertex_to_region_map(curve_skeleton);
    build_region_mesh(mesh, &vertex_to_region, translation, scale, |region| {
        let score = region_scores.get(&region).copied().unwrap_or(0.0) as f32;
        colors::map(score, &colors::PARULA)
    })
}

/// Creates a Bevy mesh for visualizing polycube surface patches using the labeled skeleton.
pub fn create_polycube_patch_mesh(
    labeled_skeleton: &LabeledCurveSkeleton,
    polycube: &mehsh::prelude::Mesh<POLYCUBE>,
    translation: Vector3D,
    scale: f64,
) -> bevy::mesh::Mesh {
    let vertex_to_region = build_polycube_vertex_to_region_map(labeled_skeleton);
    build_region_mesh(polycube, &vertex_to_region, translation, scale, get_region_color)
}

/// Splits a quad face that crosses a patch boundary.
///
/// Finds the two edges where the region changes, computes their midpoints, and draws
/// two quads (each fan-triangulated) on either side of the boundary line.
fn split_quad<F, C>(
    add_vertex: &mut F,
    regions: &[(Vector3D, usize); 4],
    normal: Vector3D,
    region_color: &C,
) where
    F: FnMut(Vector3D, Vector3D, &[f32; 3]),
    C: Fn(usize) -> [f32; 3],
{
    let mut boundary_edges: Vec<usize> = Vec::new();
    for i in 0..4 {
        let j = (i + 1) % 4;
        if regions[i].1 != regions[j].1 {
            boundary_edges.push(i);
        }
    }

    if boundary_edges.len() == 2 {
        let e0 = boundary_edges[0];
        let e1 = boundary_edges[1];
        let e0_next = (e0 + 1) % 4;
        let e1_next = (e1 + 1) % 4;

        let mid0 = (regions[e0].0 + regions[e0_next].0) * 0.5;
        let mid1 = (regions[e1].0 + regions[e1_next].0) * 0.5;

        let region_a = regions[e0_next].1;
        let region_b = regions[e0].1;

        let color_a = region_color(region_a);
        let color_b = region_color(region_b);

        let mut side_a = vec![mid0];
        let mut idx = e0_next;
        loop {
            side_a.push(regions[idx].0);
            if idx == e1 {
                break;
            }
            idx = (idx + 1) % 4;
        }
        side_a.push(mid1);

        let mut side_b = vec![mid1];
        let mut idx = e1_next;
        loop {
            side_b.push(regions[idx].0);
            if idx == e0 {
                break;
            }
            idx = (idx + 1) % 4;
        }
        side_b.push(mid0);

        fan_triangulate(add_vertex, &side_a, normal, &color_a);
        fan_triangulate(add_vertex, &side_b, normal, &color_b);
    } else {
        // Unexpected pattern (e.g. checkerboard ABAB). Fall back to majority color.
        let majority_region = regions[0].1;
        let color = region_color(majority_region);
        add_vertex(regions[0].0, normal, &color);
        add_vertex(regions[1].0, normal, &color);
        add_vertex(regions[2].0, normal, &color);
        add_vertex(regions[0].0, normal, &color);
        add_vertex(regions[2].0, normal, &color);
        add_vertex(regions[3].0, normal, &color);
    }
}

/// Fan-triangulates a convex polygon (3+ vertices) from vertex 0.
fn fan_triangulate<F>(
    add_vertex: &mut F,
    verts: &[Vector3D],
    normal: Vector3D,
    color: &[f32; 3],
) where
    F: FnMut(Vector3D, Vector3D, &[f32; 3]),
{
    for i in 1..(verts.len() - 1) {
        add_vertex(verts[0], normal, color);
        add_vertex(verts[i], normal, color);
        add_vertex(verts[i + 1], normal, color);
    }
}

/// Creates gizmos for patch boundaries on the polycube mesh.
pub fn create_polycube_patch_boundary_gizmos(
    labeled_skeleton: &LabeledCurveSkeleton,
    polycube: &mehsh::prelude::Mesh<POLYCUBE>,
    translation: Vector3D,
    scale: f64,
) -> GizmoAsset {
    let vertex_to_region = build_polycube_vertex_to_region_map(labeled_skeleton);
    boundary_gizmos_from_regions(polycube, &vertex_to_region, translation, scale)
}

/// Shared helper: draws gray lines at edge midpoints where two adjacent vertices belong to
/// different regions. Works for any mesh type (INPUT triangles or POLYCUBE quads).
fn boundary_gizmos_from_regions<T: Tag>(
    mesh: &mehsh::prelude::Mesh<T>,
    vertex_to_region: &HashMap<VertKey<T>, usize>,
    translation: Vector3D,
    scale: f64,
) -> GizmoAsset {
    let mut gizmos = GizmoAsset::new();
    let boundary_color = colors::to_bevy(colors::GRAY);

    for face_id in mesh.face_ids() {
        let mut boundary_midpoints: Vec<Vec3> = Vec::new();

        for edge_id in mesh.edges(face_id) {
            let Some([v1, v2]) = mesh.vertices(edge_id).collect_array::<2>() else {
                continue;
            };
            let r1 = vertex_to_region.get(&v1);
            let r2 = vertex_to_region.get(&v2);
            if let (Some(r1), Some(r2)) = (r1, r2) {
                if r1 != r2 {
                    let midpoint = (mesh.position(v1) + mesh.position(v2)) * 0.5;
                    boundary_midpoints.push(world_to_view(midpoint, translation, scale));
                }
            }
        }

        if boundary_midpoints.len() == 2 {
            gizmos.line(boundary_midpoints[0], boundary_midpoints[1], boundary_color);
        }
    }

    gizmos
}

// TODO: remove later
/// Creates gizmos for face points as spheres on the input mesh.
///
/// For each node, draws 6 spheres (one per direction/sign slot). Color matches the direction.
/// Boundary face points (loop centroid) use a larger sphere; interior edge midpoints use a smaller one.
pub fn create_face_point_gizmos(
    face_points: &FacePointMap,
    skeleton: &LabeledCurveSkeleton,
    mesh: &mehsh::prelude::Mesh<INPUT>,
    translation: Vector3D,
    scale: f64,
) -> GizmoAsset {
    let mut gizmos = GizmoAsset::new();
    const RADIUS: f32 = 0.25;

    for (&node_idx, per_dir) in face_points {
        let node_pos = world_to_view(
            skeleton[node_idx].skeleton_node.position,
            translation,
            scale,
        );

        for ((dir, _sign), &edge_id) in per_dir {
            let a = mesh.position(mesh.root(edge_id));
            let b = mesh.position(mesh.toor(edge_id));
            let pos = (a + b) * 0.5;
            let center = world_to_view(pos, translation, scale);
            let color = colors::to_bevy(colors::from_direction(*dir, None, None));
            gizmos.sphere(Isometry3d::from_translation(center), RADIUS, color);
            gizmos.line(node_pos, center, color);
        }
    }

    gizmos
}

// TODO: remove later
/// Creates gizmos for boundary loop crossing points as spheres on the input mesh.
///
/// For each boundary loop, draws 4 spheres at the edge midpoints that are most extreme
/// in each orthogonal (direction, sign). Sphere color matches the crossing loop direction.
/// Positive-sign crossings use a larger sphere; negative-sign crossings use a smaller one.
pub fn create_crossing_point_gizmos(
    crossings: &CrossingMap,
    mesh: &mehsh::prelude::Mesh<INPUT>,
    translation: Vector3D,
    scale: f64,
) -> GizmoAsset {
    let mut gizmos = GizmoAsset::new();
    const RADIUS_POSITIVE: f32 = 0.35;
    const RADIUS_NEGATIVE: f32 = 0.18;

    for (_loop_id, per_dir) in crossings {
        for ((ortho_dir, sign), &edge_id) in per_dir {
            let pos = mesh.position(edge_id);
            let center = world_to_view(pos, translation, scale);

            let color = colors::to_bevy(colors::from_direction(*ortho_dir, None, None));
            let radius = match sign {
                AxisSign::Positive => RADIUS_POSITIVE,
                AxisSign::Negative => RADIUS_NEGATIVE,
            };

            gizmos.sphere(Isometry3d::from_translation(center), radius, color);
        }
    }

    gizmos
}
