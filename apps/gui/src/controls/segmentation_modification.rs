use super::shared::CacheResource;
use crate::colors;
use crate::jobs::Job;
use crate::render::gizmos::{PerpetualGizmos, vector3d_to_vec3, world_to_view};
use crate::resources::{Configuration, InputResource, SolutionResource};
use bevy::prelude::*;
use dualcube::prelude::*;

pub fn segmentation_modification_system(
    mouse: Res<'_, ButtonInput<MouseButton>>,
    keyboard: Res<'_, ButtonInput<KeyCode>>,
    mesh_resmut: Res<'_, InputResource>,
    mut solution: ResMut<'_, SolutionResource>,
    mut cache: ResMut<'_, CacheResource>,
    mut gizmos: Gizmos<'_, '_, PerpetualGizmos>,
    configuration: ResMut<'_, Configuration>,
    mut jobs: MessageWriter<'_, Job>,
    position: Vector3D,
    _nearest_face: FaceID,
) -> Result<(), BevyError> {
    if mesh_resmut.mesh.nr_verts() == 0 {
        return Ok(());
    }

    if let Some(layout) = &solution.current_solution.layout {
        let key = (
            layout.granulated_mesh.nr_verts(),
            layout.granulated_mesh.nr_faces(),
            layout.alignment.unwrap_or(f64::NAN).to_bits(),
        );
        if cache
            .granulated_lookup
            .as_ref()
            .is_none_or(|(k, _)| *k != key)
        {
            cache.granulated_lookup =
                Some((key, std::sync::Arc::new(layout.granulated_mesh.kdtree())));
        }
        let granulated_vert_lookup = cache.granulated_lookup.as_ref().unwrap().1.clone();
        let nearest_granulated_vert = granulated_vert_lookup.nearest(&position.into()).1;

        // Look for nearest segmentation corner
        let modification = if (solution.selected_corner).is_none() {
            let (current_polycube_corner, current_segmentation_corner) =
                layout
                    .vert_to_corner
                    .iter()
                    .min_by_key(|&(_, corner)| {
                        OrderedFloat(layout.granulated_mesh.position(*corner).metric_distance(
                            &layout.granulated_mesh.position(nearest_granulated_vert),
                        ))
                    })
                    .map(|(&poly_vert, &seg_vert)| (poly_vert, seg_vert))
                    .unwrap();

            // Highlight this vertex
            let v = layout.granulated_mesh.position(current_segmentation_corner);
            let v_transformed = world_to_view(
                v,
                mesh_resmut.properties.translation,
                mesh_resmut.properties.scale,
            );
            let n = vector3d_to_vec3(layout.granulated_mesh.normal(current_segmentation_corner));

            let isometry = Isometry3d::new(
                v_transformed,
                Quat::from_rotation_arc(Vec3::Z, n.normalize()),
            );
            gizmos.line(
                v_transformed,
                v_transformed + n,
                colors::to_bevy(colors::DARK_GRAY),
            );
            gizmos.circle(isometry, 0.2, colors::to_bevy(colors::DARK_GRAY));

            Some(current_polycube_corner)
        } else {
            None
        };

        if let Some(corner_poly) = solution.selected_corner {
            // Highlight selected corner
            let corner_poly_vert = layout
                .vert_to_corner
                .get_by_left(&corner_poly)
                .unwrap()
                .to_owned();
            let v = layout.granulated_mesh.position(corner_poly_vert);
            let v_transformed = world_to_view(
                v,
                mesh_resmut.properties.translation,
                mesh_resmut.properties.scale,
            );
            let n = vector3d_to_vec3(layout.granulated_mesh.normal(corner_poly_vert));

            let isometry = Isometry3d::new(
                v_transformed,
                Quat::from_rotation_arc(Vec3::Z, n.normalize()),
            );
            gizmos.line(
                v_transformed,
                v_transformed + n,
                colors::to_bevy(colors::BLACK),
            );
            gizmos.circle(isometry, 0.1, colors::to_bevy(colors::BLACK));

            // Highlight the current position (where the vertex would be moved)
            let v1 = layout.granulated_mesh.position(nearest_granulated_vert);
            let v1_transformed = world_to_view(
                v1,
                mesh_resmut.properties.translation,
                mesh_resmut.properties.scale,
            );
            let n1 = vector3d_to_vec3(layout.granulated_mesh.normal(nearest_granulated_vert));

            let isometry1 = Isometry3d::new(
                v1_transformed,
                Quat::from_rotation_arc(Vec3::Z, n1.normalize()),
            );
            gizmos.line(
                v1_transformed,
                v1_transformed + n1,
                colors::to_bevy(colors::BLACK),
            );
            gizmos.circle(isometry1, 0.1, colors::to_bevy(colors::BLACK));

            gizmos.arrow(
                v_transformed + 0.2 * n,
                v1_transformed + 0.1 * n1,
                colors::to_bevy(colors::BLACK),
            );
        }

        // CONTROLS
        //
        // Action1:  Select segmentation corner to be moved (ALT+LMB)
        // Action2:  Move segmentation corner to new location (LMB) (if a corner is selected)
        // Action3:  Deselect segmentation corner (ESCAPE)

        let esc = keyboard.pressed(KeyCode::Escape);
        let lmb = mouse.pressed(MouseButton::Left);
        let alt = keyboard.pressed(KeyCode::AltLeft);

        // Controls
        match (lmb, alt, esc) {
            // Action1:
            (true, true, false) => {
                if let Some(corner_poly) = modification {
                    solution.selected_corner = Some(corner_poly);
                }
            }
            // Action2:
            (true, false, false) => {
                if let Some(corner_poly) = solution.selected_corner {
                    jobs.write(Job::move_corner(
                        solution.current_solution.clone(),
                        configuration.clone(),
                        corner_poly,
                        nearest_granulated_vert,
                    ));
                    solution.selected_corner = None;
                }
            }
            // Action3:
            (_, _, true) => {
                solution.selected_corner = None;
            }
            _ => {}
        }
    }

    Ok(())
}
