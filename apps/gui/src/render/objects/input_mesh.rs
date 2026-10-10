//! Render object for the input (triangle) mesh.

use super::super::gizmos::{
    DirectionalGizmos, add_band, axis_index, band_colors, flow_graph_gizmos, lambert_color_map,
    path_strips, segmentation_color_map, uniform_color_map, world_to_view,
};
use super::super::store::{RenderAsset, RenderObject};
use crate::{colors, resources::Configuration};
use bevy::prelude::*;
use dualcube::prelude::*;
use std::collections::HashMap;

const PRINCIPAL_DIRECTIONS: [Direction; 3] = DIRECTIONS;

/// The INPUT MESH render object, it has:
/// - meshes with gray / black / lambert-shaded faces
/// - segmentation and alignment colorings (if a layout exists)
/// - wireframes (input and granulated mesh)
/// - the dual loops, layout paths, and vector fields
pub(in crate::render) fn build(
    solution: &Solution,
    configuration: &Configuration,
) -> Option<RenderObject> {
    let input = solution.mesh_ref.as_ref();
    let (scale, translation) = input.scale_translation();

    // The dual loops, per principal direction, as bands on the surface.
    // The loops and paths as bands, their widths relative to the diagonal of the bounding box.
    let (min, max) = input.vert_ids().iter().map(|&v| input.position(v)).fold(
        (
            Vector3D::repeat(f64::INFINITY),
            Vector3D::repeat(f64::NEG_INFINITY),
        ),
        |(min, max), p| (min.inf(&p), max.sup(&p)),
    );
    let half = 0.5 * configuration.loop_width * (max - min).norm().max(1e-12);
    let [loops_x, loops_y, loops_z] = loop_bands(solution, half, translation, scale);
    // The paths above the loops, so that they cross them cleanly.
    let [paths, flat_paths] = path_strips(
        |spacing| solution.path_bands(spacing),
        half,
        0.8 * half,
        translation,
        scale,
    );

    // Layout-dependent features.
    let empty_mesh = mehsh::prelude::Mesh::<INPUT>::default();
    let mut granulated_mesh = &empty_mesh;
    let mut granulated_mesh_gizmos = GizmoAsset::new();
    let mut color_map_segmentation = HashMap::new();
    let mut color_map_alignment = HashMap::new();

    if let (Some(layout), Some(polycube)) = (&solution.layout, &solution.polycube) {
        granulated_mesh = &layout.granulated_mesh;
        granulated_mesh_gizmos = mehsh_bevy::gizmos(granulated_mesh, colors::GRAY);
        color_map_segmentation = segmentation_color_map(layout, polycube);

        for triangle_id in granulated_mesh.face_ids() {
            let color = layout
                .alignment_per_triangle
                .get(&triangle_id)
                .map_or(colors::SNOEP_YELLOW, |&score| {
                    colors::map(score as f32, &colors::SCALE_MAGMA)
                });
            color_map_alignment.insert(triangle_id, color);
        }
    }

    // The vector fields, per principal direction.
    let mut field_gizmos = DirectionalGizmos::default();
    if let Some(fields) = &solution.fields {
        // Arrow length relative to the mesh bounding-box diagonal.
        let vert_ids = input.vert_ids();
        let field_scale = 0.015
            * vert_ids.first().map_or(1.0, |&first_id| {
                let first = input.position(first_id);
                let (mut min_p, mut max_p) = (first, first);
                for &vert_id in &vert_ids {
                    let p = input.position(vert_id);
                    min_p = Vector3D::new(min_p.x.min(p.x), min_p.y.min(p.y), min_p.z.min(p.z));
                    max_p = Vector3D::new(max_p.x.max(p.x), max_p.y.max(p.y), max_p.z.max(p.z));
                }
                (max_p - min_p).norm().max(1e-12)
            });

        for (field, dir) in [&fields.field_x, &fields.field_y, &fields.field_z]
            .into_iter()
            .zip(PRINCIPAL_DIRECTIONS)
        {
            let color = colors::to_bevy(colors::from_direction(dir, None, None));
            let gizmos = field_gizmos.get_mut(dir);
            for (&vert_id, &vector_id) in &field.map {
                let Some(v) = field.vectors.get(vector_id) else {
                    continue;
                };
                if v.norm() <= 1e-12 {
                    continue;
                }
                let p = input.position(vert_id);
                gizmos.arrow(
                    world_to_view(p, translation, scale),
                    world_to_view(p + *v * field_scale, translation, scale),
                    color,
                );
            }
        }
    }

    // The flow graphs, per principal direction: edge-to-edge transition weights.
    // Bright = low weight (good), fading to black/transparent = high weight (bad).
    let mut flow_graph_dir = DirectionalGizmos::default();
    if let Some(flow_graphs) = &solution.flow_graphs {
        for (graph, dir) in flow_graphs.iter().zip(PRINCIPAL_DIRECTIONS) {
            let color = colors::from_direction(dir, None, None);
            *flow_graph_dir.get_mut(dir) =
                flow_graph_gizmos(&graph.edges(), input, color, translation, scale);
        }
    }

    Some(
        RenderObject::default()
            .mesh(input, &uniform_color_map(input, colors::LIGHT_GRAY), "gray")
            .mesh(input, &uniform_color_map(input, colors::BLACK), "black")
            .mesh(input, &lambert_color_map(input), "lambert")
            .mesh(granulated_mesh, &color_map_segmentation, "segmentation")
            .mesh(granulated_mesh, &color_map_alignment, "alignment")
            .gizmo(
                mehsh_bevy::gizmos(input, colors::GRAY),
                0.5,
                -0.00001,
                "wireframe",
            )
            .add("x-loops", RenderAsset::Mesh(loops_x))
            .add("y-loops", RenderAsset::Mesh(loops_y))
            .add("z-loops", RenderAsset::Mesh(loops_z))
            .add("paths", RenderAsset::Mesh(paths))
            .add("flat paths", RenderAsset::Mesh(flat_paths))
            .gizmo(granulated_mesh_gizmos, 0.5, -0.00001, "refined wireframe")
            .gizmo(field_gizmos.x, 1., -0.0010, "x-field")
            .gizmo(field_gizmos.y, 1., -0.0011, "y-field")
            .gizmo(field_gizmos.z, 1., -0.0012, "z-field")
            .gizmo(flow_graph_dir.x, 1., -0.0001, "x-graph")
            .gizmo(flow_graph_dir.y, 1., -0.00011, "y-graph")
            .gizmo(flow_graph_dir.z, 1., -0.000111, "z-graph")
            .to_owned(),
    )
}

/// The loops as bands on the surface, per principal direction: a strip of `half` wide on either side of every loop, its
/// positive side in the loop's color and its negative side lighter (see `Solution::loop_band`). The bands float slightly
/// above the surface, so that it does not cover them where it curves.
#[allow(unused_qualifications)]
fn loop_bands(
    solution: &Solution,
    half: f64,
    translation: Vector3D,
    scale: f64,
) -> [bevy::mesh::Mesh; 3] {
    let mut builders = DIRECTIONS.map(|_| mehsh_bevy::MeshBuilder::new());
    for loop_id in solution.loops.keys() {
        let direction = solution.loop_to_direction(loop_id);
        let axis = axis_index(direction);
        // Resampled at about the band's width, so that it bends smoothly (see `band_frames`).
        let frames = solution.loop_band(loop_id, half);
        let segments = frames
            .iter()
            .copied()
            .zip(frames.iter().copied().cycle().skip(1));
        // The axes at slightly different heights, so that crossing bands do not flicker.
        let lift = (0.5 + 0.1 * axis as f64) * half;
        add_band(
            &mut builders[axis],
            segments,
            half,
            lift,
            band_colors(direction),
        );
    }
    builders.map(|mut builder| {
        builder.normalize(scale, translation);
        builder.build()
    })
}
