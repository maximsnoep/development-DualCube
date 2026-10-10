//! Render layers for the quad mesh (shown on top of the input mesh, see `render::refresh`).

use super::super::gizmos::{edge_endpoints_view, uniform_color_map};
use super::super::store::RenderObject;
use crate::{colors, resources::Configuration};
use bevy::prelude::*;
use dualcube::prelude::*;
use mehsh_bevy;
use std::collections::{HashMap, HashSet};

/// The QUAD MESH render object, it has:
/// - mesh with gray faces
/// - mesh with colored faces
/// - wireframe (the quads)
/// - paths along irregular edge loops (flat and non-flat)
pub(in crate::render) fn build(
    solution: &Solution,
    _configuration: &Configuration,
) -> Option<RenderObject> {
    let quad = solution.quad.as_ref()?;

    let mesh = &quad.quad_mesh;
    let (scale, translation) = mesh.scale_translation();

    // Color each quad by the principal direction of its polycube normal.
    let mut color_map = HashMap::new();
    for face_id in mesh.face_ids() {
        let normal = quad.quad_mesh_polycube.normal(face_id);
        let color = colors::from_direction(
            to_principal_direction(normal).0,
            Some(Perspective::Primal),
            None,
        );
        color_map.insert(face_id, color);
    }

    let mut gizmos_paths = GizmoAsset::new();
    let mut gizmos_flat_paths = GizmoAsset::new();
    if solution.layout.is_some() && solution.polycube.is_some() {
        let c = colors::to_bevy(colors::GRAY);

        // A vertex is irregular if the quads around it carry 3+ distinct labels.
        let mut irregular_vertices = HashSet::new();
        for vert_id in mesh.vert_ids() {
            let labels = mesh
                .faces(vert_id)
                .map(|face_id| to_principal_direction(quad.quad_mesh_polycube.normal(face_id)).0)
                .collect::<HashSet<_>>();
            if labels.len() >= 3 {
                irregular_vertices.insert(vert_id);
            }
        }

        // Trace the edge loops emanating from the irregular vertices.
        let mut irregular_edges = HashSet::new();
        for &vert_id in &irregular_vertices {
            for edge_id in mesh.edges(vert_id) {
                let mut walker = mesh.next(mesh.twin(mesh.next(edge_id)));
                while irregular_edges.insert(walker) {
                    walker = mesh.next(mesh.twin(mesh.next(walker)));
                }
            }
        }

        // Draw all irregular edges; edges between equally-labeled quads are flat.
        for edge_id in irregular_edges {
            let Some([f1, f2]) = mesh.faces(edge_id).collect_array::<2>() else {
                panic!("Expected two faces for edge {edge_id:?}");
            };
            let (u, v) = edge_endpoints_view(mesh, edge_id, translation, scale);
            if quad.quad_mesh_polycube.normal(f1) == quad.quad_mesh_polycube.normal(f2) {
                gizmos_flat_paths.line(u, v, c);
            } else {
                gizmos_paths.line(u, v, c);
            }
        }
    }

    Some(
        RenderObject::default()
            .mesh(mesh, &uniform_color_map(mesh, colors::LIGHT_GRAY), "gray")
            .mesh(mesh, &color_map, "colored")
            .gizmo(
                mehsh_bevy::gizmos(mesh, colors::GRAY),
                1.0,
                -0.001,
                "wireframe",
            )
            .gizmo(gizmos_paths, 4., -0.0001, "paths")
            .gizmo(gizmos_flat_paths, 3., -0.00011, "flat paths")
            .to_owned(),
    )
}
