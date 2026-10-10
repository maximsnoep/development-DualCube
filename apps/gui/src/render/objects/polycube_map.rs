//! Render object for the polycube-map (the input mesh mapped onto the polycube).

use super::super::gizmos::{
    lambert_color_map, loop_half_width, path_strips, segmentation_color_map, uniform_color_map,
};
use super::super::store::{RenderAsset, RenderObject};
use crate::{colors, resources::Configuration};
use dualcube::prelude::*;
use mehsh_bevy;

/// The POLYCUBE-MAP render object, it has:
/// - meshes with black / gray / colored / lambert-shaded faces
/// - quads and triangles mapped on the polycube
/// - the layout paths mapped on the polycube (flat and non-flat)
pub(in crate::render) fn build(
    solution: &Solution,
    configuration: &Configuration,
) -> Option<RenderObject> {
    let (quad, layout, polycube) = (
        solution.quad.as_ref()?,
        solution.layout.as_ref()?,
        solution.polycube.as_ref()?,
    );

    let mesh = &quad.triangle_mesh_polycube;
    let (scale, translation) = mesh.scale_translation();
    // The paths as strips (as on the model), lifted off the polycube (on its edges, they bend around them).
    let half = loop_half_width(mesh, configuration.loop_width);
    let [paths, flat_paths] = path_strips(
        |spacing| solution.path_bands_on(mesh, spacing),
        half,
        half,
        translation,
        scale,
    );

    Some(
        RenderObject::default()
            .mesh(mesh, &uniform_color_map(mesh, colors::BLACK), "black")
            .mesh(mesh, &uniform_color_map(mesh, colors::LIGHT_GRAY), "gray")
            .mesh(mesh, &segmentation_color_map(layout, polycube), "colored")
            // Shading uses the normals of the (world-space) granulated mesh,
            // whose face ids correspond to the polycube-mapped triangle mesh.
            .mesh(mesh, &lambert_color_map(&layout.granulated_mesh), "lambert")
            .gizmo(
                mehsh_bevy::gizmos(&quad.quad_mesh_polycube, colors::GRAY),
                2.,
                -0.01,
                "quads",
            )
            .gizmo(
                mehsh_bevy::gizmos(mesh, colors::GRAY),
                2.,
                -0.01,
                "triangles",
            )
            .add("paths", RenderAsset::Mesh(paths))
            .add("flat paths", RenderAsset::Mesh(flat_paths))
            .to_owned(),
    )
}
