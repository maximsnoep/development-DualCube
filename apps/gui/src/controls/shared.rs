use crate::colors;
use crate::render::gizmos::{PerpetualGizmos, world_to_view};
use crate::resources::InputResource;
use bevy::prelude::*;
use dualcube::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InteractiveMode {
    None,
    LoopModification,
    SegmentationModification,
}

/// Cached interactive control previews.
#[derive(Default, Resource)]
pub struct CacheResource {
    pub loop_preview_key: Option<LoopPreviewKey>,
    pub loop_preview: Option<(Vec<EdgeID>, f64)>,
    // The gaps (in the existing loop orders) of a previewed valid loop.
    pub loop_preview_gaps: Option<Vec<(usize, usize)>>,
    // Where the previewed loop would be placed (pulled taut between the existing loops).
    pub loop_preview_positions: Vec<Vector3D>,
    pub loop_preview_segments: Vec<(Vec<EdgeID>, f64)>,
    pub locked_loop_segments: Vec<(Vec<EdgeID>, f64)>,
    pub locked_loop_direction: Option<Direction>,
    // Vertex lookup of the layout's granulated mesh (rebuilding it every frame is expensive), with the layout it was
    // built for (number of vertices and faces, and alignment).
    pub granulated_lookup: Option<((usize, usize, u64), std::sync::Arc<VertLocation<INPUT>>)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopPreviewKey {
    pub direction: Direction,
    pub hover: [EdgeID; 2],
    pub anchors: Vec<[EdgeID; 2]>,
}

#[allow(unused_qualifications)]
pub fn draw_edgepair_arrow(
    mesh_resmut: &InputResource,
    gizmos: &mut Gizmos<'_, '_, PerpetualGizmos>,
    edgepair: [EdgeID; 2],
    color: bevy::prelude::Color,
) {
    let u = mesh_resmut.mesh.position(edgepair[0]);
    let v = mesh_resmut.mesh.position(edgepair[1]);
    gizmos.arrow(
        world_to_view(
            u,
            mesh_resmut.properties.translation,
            mesh_resmut.properties.scale,
        ),
        world_to_view(
            v,
            mesh_resmut.properties.translation,
            mesh_resmut.properties.scale,
        ),
        color,
    );
}

pub fn draw_polyline_gradient(
    mesh_resmut: &InputResource,
    gizmos: &mut Gizmos<'_, '_, PerpetualGizmos>,
    positions: &[Vector3D],
    color: colors::Kolor,
) {
    if positions.len() < 2 {
        return;
    }
    let last = positions.len().saturating_sub(1).max(1) as f32;
    for i in 0..positions.len() {
        let t = i as f32 / last;
        let segment_color = bevy::color::Color::srgba(color[0], color[1], color[2], 1.0 - t);
        let (u, v) = (positions[i], positions[(i + 1) % positions.len()]);
        gizmos.line(
            world_to_view(
                u,
                mesh_resmut.properties.translation,
                mesh_resmut.properties.scale,
            ),
            world_to_view(
                v,
                mesh_resmut.properties.translation,
                mesh_resmut.properties.scale,
            ),
            segment_color,
        );
    }
}
