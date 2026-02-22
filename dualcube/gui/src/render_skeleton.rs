use crate::colors;
use crate::render::world_to_view;
use bevy::prelude::*;
use dualcube::prelude::*;
use dualcube::skeleton::curve_skeleton::CurveSkeletonSpatial;
use itertools::Itertools;
use mehsh::integrations::bevy::MeshBuilder;
use mehsh::prelude::*;
use std::collections::HashMap;

/// Creates gizmos for patch boundaries by connecting midpoints of edges that the boundary crosses.
pub fn create_patch_boundary_gizmos(
    curve_skeleton: &CurveSkeleton,
    mesh: &mehsh::prelude::Mesh<INPUT>,
    translation: Vector3D,
    scale: f64,
) -> GizmoAsset {
    let mut gizmos = GizmoAsset::new();
    let boundary_color = colors::to_bevy(colors::GRAY);

    // Build vertex-to-node mapping
    let mut vertex_to_node: HashMap<VertKey<INPUT>, usize> = HashMap::new();
    for node_idx in curve_skeleton.node_indices() {
        for &vert_key in &curve_skeleton[node_idx].patch_vertices {
            vertex_to_node.insert(vert_key, node_idx.index());
        }
    }

    // For each face, find edges that cross a patch boundary and connect their midpoints
    for face_id in mesh.face_ids() {
        let mut boundary_midpoints: Vec<Vec3> = Vec::new();

        for edge_id in mesh.edges(face_id) {
            let Some([v1, v2]) = mesh.vertices(edge_id).collect_array::<2>() else {
                continue;
            };
            let r1 = vertex_to_node.get(&v1);
            let r2 = vertex_to_node.get(&v2);
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

/// Creates gizmos for visualizing a curve skeleton with spheres for nodes and lines for edges.
pub fn create_skeleton_gizmos(
    curve_skeleton: &CurveSkeleton,
    translation: Vector3D,
    scale: f64,
) -> GizmoAsset {
    let mut gizmos = GizmoAsset::new();
    let skel_color = colors::to_bevy(colors::LIGHT_GRAY);
    let node_radius = 0.2;

    // Draw edges
    for edge in curve_skeleton.edge_indices() {
        let (a, b) = curve_skeleton.edge_endpoints(edge).unwrap();
        let pos_a = curve_skeleton[a].position;
        let pos_b = curve_skeleton[b].position;
        let a_view = world_to_view(pos_a, translation, scale);
        let b_view = world_to_view(pos_b, translation, scale);
        gizmos.line(a_view, b_view, skel_color);
    }

    // Draw nodes
    for node_idx in curve_skeleton.node_indices() {
        let pos = curve_skeleton[node_idx].position;
        let center = world_to_view(pos, translation, scale);
        gizmos.sphere(
            Isometry3d::from_translation(center),
            node_radius,
            skel_color,
        );
    }

    gizmos
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
fn get_region_color(region: usize) -> [f32; 3] {
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
fn build_region_mesh<F>(
    curve_skeleton: &CurveSkeleton,
    mesh: &mehsh::prelude::Mesh<INPUT>,
    translation: Vector3D,
    scale: f64,
    region_color_fn: F,
) -> bevy::mesh::Mesh
where
    F: Fn(usize) -> [f32; 3],
{
    // Build a mapping from vertex to region index
    let mut vertex_to_region: HashMap<VertKey<INPUT>, usize> = HashMap::new();
    for node_idx in curve_skeleton.node_indices() {
        let node = &curve_skeleton[node_idx];
        for &vert_key in &node.patch_vertices {
            vertex_to_region.insert(vert_key, node_idx.index());
        }
    }

    let mut builder = MeshBuilder::default();

    // Helper to transform and add a vertex to the builder
    let mut add_vertex = |pos: Vector3D, normal: Vector3D, color: &[f32; 3]| {
        let transformed_pos = pos * scale + translation;
        builder.add_vertex(&transformed_pos, &normal, color);
    };

    // For each face, handle based on region assignment
    for face_id in mesh.face_ids() {
        let face_verts: Vec<_> = mesh.vertices(face_id).collect();
        if face_verts.len() != 3 {
            continue; // Skip non-triangular faces
        }

        let v0 = face_verts[0];
        let v1 = face_verts[1];
        let v2 = face_verts[2];

        let p0 = mesh.position(v0);
        let p1 = mesh.position(v1);
        let p2 = mesh.position(v2);

        let n0 = mesh.normal(v0);
        let n1 = mesh.normal(v1);
        let n2 = mesh.normal(v2);

        // Get the region for each vertex
        let r0 = vertex_to_region.get(&v0).copied();
        let r1 = vertex_to_region.get(&v1).copied();
        let r2 = vertex_to_region.get(&v2).copied();

        match (r0, r1, r2) {
            // All vertices have regions
            (Some(r0), Some(r1), Some(r2)) => {
                if r0 == r1 && r1 == r2 {
                    // All same region, simply draw the triangle
                    let color = region_color_fn(r0);
                    add_vertex(p0, n0, &color);
                    add_vertex(p1, n1, &color);
                    add_vertex(p2, n2, &color);
                } else if r0 == r1 {
                    // v0, v1 share region X; v2 is region Y
                    split_triangle(
                        &mut add_vertex,
                        p0,
                        p1,
                        p2,
                        n0,
                        n1,
                        n2,
                        r0,
                        r2,
                        &region_color_fn,
                    );
                } else if r1 == r2 {
                    // v1, v2 share region X; v0 is region Y
                    split_triangle(
                        &mut add_vertex,
                        p1,
                        p2,
                        p0,
                        n1,
                        n2,
                        n0,
                        r1,
                        r0,
                        &region_color_fn,
                    );
                } else if r0 == r2 {
                    // v0, v2 share region X; v1 is region Y
                    split_triangle(
                        &mut add_vertex,
                        p2,
                        p0,
                        p1,
                        n2,
                        n0,
                        n1,
                        r0,
                        r1,
                        &region_color_fn,
                    );
                } else {
                    unreachable!(
                        "Triangle with all three vertices in different regions encountered."
                    );
                }
            }
            _ => {}
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
    // Delegate to shared helper using Tailwind region colors
    build_region_mesh(curve_skeleton, mesh, translation, scale, |region| {
        get_region_color(region)
    })
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
    // Compute convexity score per region (clamped between 0 and 1),
    // keyed by node_idx.index() to match build_region_mesh's region keys.
    let mut region_scores: HashMap<usize, f64> = HashMap::new();
    for node_idx in curve_skeleton.node_indices() {
        let mut score = curve_skeleton.patch_convexity_score(node_idx, mesh);
        println!("Convexity score for node {:?}: {}", node_idx, score);
        if !score.is_finite() {
            score = 0.0;
        }
        region_scores.insert(node_idx.index(), score.clamp(0.0, 1.0));
    }

    // Delegate to the shared builder using a score->color mapping
    build_region_mesh(curve_skeleton, mesh, translation, scale, |region| {
        let score = region_scores.get(&region).copied().unwrap_or(0.0) as f32;
        colors::map(score, &colors::SCALE_MAGMA)
    })
}
