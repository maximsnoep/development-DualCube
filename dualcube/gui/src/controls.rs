use crate::jobs::{Job, JobRequest};
use crate::render::{view_to_world, world_to_view};
use crate::{
    colors, vec3_to_vector3d, vector3d_to_vec3, CacheResource, Configuration, InputResource,
    MainMesh, PerpetualGizmos, SolutionResource,
};
use bevy::picking::backend::ray::RayMap;
use bevy::prelude::*;
use dualcube::prelude::*;
use itertools::Itertools;
use mehsh::prelude::*;
use ordered_float::OrderedFloat;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InteractiveMode {
    None,
    LoopModification,
    SegmentationModification,
}

pub fn segmentation_modification_system(
    mouse: Res<ButtonInput<MouseButton>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mesh_resmut: Res<InputResource>,
    mut solution: ResMut<SolutionResource>,
    mut cache: ResMut<CacheResource>,
    mut gizmos: Gizmos<PerpetualGizmos>,
    mut configuration: ResMut<Configuration>,
    mut jobs: EventWriter<JobRequest>,
    position: Vector3D,
    nearest_face: FaceID,
) -> Result<(), BevyError> {
    if mesh_resmut.mesh.nr_verts() == 0 {
        return Ok(());
    }

    if let Some(layout) = &solution.current_solution.layout {
        let granulated_vert_lookup = layout.granulated_mesh.kdtree();
        let nearest_granulated_vert = granulated_vert_lookup.nearest(&position.into()).1;

        // Look for nearest segmentation corner
        let modification = if (solution.selected_corner).is_none() {
            let (current_polycube_corner, current_segmentation_corner) =
                layout
                    .vert_to_corner
                    .iter()
                    .min_by_key(|(_, &corner)| {
                        OrderedFloat(layout.granulated_mesh.position(corner).metric_distance(
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
                    jobs.write(JobRequest::Run(Box::new(Job::MoveCorner {
                        configuration: configuration.clone(),
                        solution: solution.current_solution.clone(),
                        corner: corner_poly,
                        new_vertex: nearest_granulated_vert,
                    })));
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

pub fn loop_modification_system(
    mouse: Res<ButtonInput<MouseButton>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mesh_resmut: Res<InputResource>,
    mut solution: ResMut<SolutionResource>,
    mut cache: ResMut<CacheResource>,
    mut gizmos: Gizmos<PerpetualGizmos>,
    mut configuration: ResMut<Configuration>,
    mut jobs: EventWriter<JobRequest>,

    position: Vector3D,
    nearest_face: FaceID,
) -> Result<(), BevyError> {
    if mesh_resmut.mesh.nr_verts() == 0 {
        return Ok(());
    }

    // get the nearest_vert (one of 3 corners of nearest_face)
    let nearest_vert = mesh_resmut
        .mesh
        .vertices(nearest_face)
        .min_by_key(|&v| OrderedFloat(position.metric_distance(&mesh_resmut.mesh.position(v))))
        .unwrap()
        .to_owned();

    let edgepair = mesh_resmut
        .mesh
        .edges_in_face_with_vert(nearest_face, nearest_vert)
        .unwrap();

    // Render all current solutions  (for currently selected direction)
    for (&edgepair, sol) in &solution.next[configuration.direction as usize] {
        let u = mesh_resmut.mesh.position(edgepair[0]);
        let v = mesh_resmut.mesh.position(edgepair[1]);

        let color = match sol {
            Some(_) => colors::from_direction(
                configuration.direction,
                Some(Perspective::Dual),
                Some(Orientation::Backwards),
            ),
            None => colors::BLACK,
        };

        let u_transformed = world_to_view(
            u,
            mesh_resmut.properties.translation,
            mesh_resmut.properties.scale,
        );
        let v_transformed = world_to_view(
            v,
            mesh_resmut.properties.translation,
            mesh_resmut.properties.scale,
        );
        gizmos.line(u_transformed, v_transformed, colors::to_bevy(color));
    }

    // Render all current anchors
    for anchor in &configuration.loop_anchors {
        let u = mesh_resmut.mesh.position(anchor[0]);
        let v = mesh_resmut.mesh.position(anchor[1]);
        let color = colors::from_direction(configuration.direction, Some(Perspective::Dual), None);
        let u_transformed = world_to_view(
            u,
            mesh_resmut.properties.translation,
            mesh_resmut.properties.scale,
        );
        let v_transformed = world_to_view(
            v,
            mesh_resmut.properties.translation,
            mesh_resmut.properties.scale,
        );
        gizmos.line(u_transformed, v_transformed, colors::to_bevy(color));
    }

    let u = mesh_resmut.mesh.position(edgepair[0]);
    let v = mesh_resmut.mesh.position(edgepair[1]);
    let color = colors::from_direction(configuration.direction, Some(Perspective::Dual), None);

    let u_transformed = world_to_view(
        u,
        mesh_resmut.properties.translation,
        mesh_resmut.properties.scale,
    );
    let v_transformed = world_to_view(
        v,
        mesh_resmut.properties.translation,
        mesh_resmut.properties.scale,
    );

    gizmos.line(u_transformed, v_transformed, colors::to_bevy(color));

    // CONTROLS
    //
    // Action1:  Add loop (LMB, no Shift)
    // Action2:  Show closest solution (Shift, no LMB)
    // Action3:  Add closest solution (LMB + Shift)
    //
    // Action4:  Remove closest valid loop (Delete, no Shift)
    // Action5:  Remove closest loop (Delete + Shift)
    //
    // Action6: Add anchor (Alt + LMB)

    let lmb = mouse.just_pressed(MouseButton::Left);
    let shift = keyboard.pressed(KeyCode::ShiftLeft);
    let delete = keyboard.just_pressed(KeyCode::Delete);
    let alt = keyboard.pressed(KeyCode::AltLeft);

    if keyboard.pressed(KeyCode::Escape) {
        configuration.loop_anchors.clear();
    }

    let closest_solution = solution.next[configuration.direction as usize]
        .iter()
        .map(|(&edgepair, sol)| {
            (
                ((mesh_resmut.mesh.position(edgepair[0]) + mesh_resmut.mesh.position(edgepair[1]))
                    / 2.)
                    .metric_distance(&position),
                sol,
                edgepair,
            )
        })
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    match (shift, lmb, delete, alt) {
        // Action1
        (false, true, false, false) => {
            let selected_edges = mesh_resmut
                .mesh
                .edges_in_face_with_vert(nearest_face, nearest_vert)
                .unwrap();
            let mut anchors = configuration.loop_anchors.clone();
            configuration.loop_anchors.clear();
            anchors.push(selected_edges);
            jobs.write(JobRequest::Run(Box::new(Job::AddLoop {
                anchors,
                direction: configuration.direction,
                flowgraph: mesh_resmut.flow_graphs[configuration.direction as usize].clone(),
                solution: solution.current_solution.clone(),
            })));
        }
        // Action 2
        (true, false, false, _) => {
            if let Some((_, Some(some_solution), signature)) = closest_solution {
                configuration.selected = Some(signature);
                let loop_id = some_solution.last_loop.unwrap();
                let direction = some_solution.loop_to_direction(loop_id);
                let color = colors::from_direction(
                    direction,
                    Some(Perspective::Dual),
                    Some(Orientation::Backwards),
                );
                for &edgepair in &some_solution.get_pairs_of_loop(loop_id) {
                    let u = mesh_resmut.mesh.position(edgepair[0]);
                    let v = mesh_resmut.mesh.position(edgepair[1]);
                    let u_transformed = world_to_view(
                        u,
                        mesh_resmut.properties.translation,
                        mesh_resmut.properties.scale,
                    );
                    let v_transformed = world_to_view(
                        v,
                        mesh_resmut.properties.translation,
                        mesh_resmut.properties.scale,
                    );
                    gizmos.line(u_transformed, v_transformed, colors::to_bevy(color));
                }
            }
        }
        // Action 3
        (true, true, false, _) => {
            if let Some((_, Some(sol), _)) = closest_solution {
                solution.current_solution = sol.clone();
                solution.next[0].clear();
                solution.next[1].clear();
                solution.next[2].clear();
                cache.cache[0].clear();
                cache.cache[1].clear();
                cache.cache[2].clear();
                jobs.write(JobRequest::Run(Box::new(Job::ComputeDual {
                    solution: solution.current_solution.clone(),
                    configuration: configuration.clone(),
                })));
            }
        }
        // Action 4 and Action 5
        (force, _, true, _) => {
            let option_a = [edgepair[0], edgepair[1]];
            let option_b = [edgepair[1], edgepair[0]];

            if let Some(loop_id) = solution.current_solution.loops.keys().find(|&loop_id| {
                let edges = solution.current_solution.get_pairs_of_loop(loop_id);
                edges.contains(&option_a) || edges.contains(&option_b)
            }) {
                jobs.write(JobRequest::Run(Box::new(Job::RemoveLoop {
                    solution: solution.current_solution.clone(),
                    loop_id,
                    force,
                    configuration: configuration.clone(),
                })));
            }
        }
        (_, true, _, true) => {
            configuration.loop_anchors.push(edgepair);
            println!("{:?}", configuration.loop_anchors);
        }

        // No actions
        _ => {}
    }

    Ok(())
}

pub fn system(
    ray_map: Res<RayMap>,
    mut ray_cast: MeshRayCast,
    foo_query: Query<(), With<MainMesh>>,
    mouse: Res<ButtonInput<MouseButton>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mesh_resmut: Res<InputResource>,
    mut solution: ResMut<SolutionResource>,
    mut cache: ResMut<CacheResource>,
    mut gizmos: Gizmos<PerpetualGizmos>,
    mut configuration: ResMut<Configuration>,
    mut jobs: EventWriter<JobRequest>,
) -> Result<(), BevyError> {
    configuration.raycasted = None;
    configuration.selected = None;

    if keyboard.pressed(KeyCode::ControlLeft) || mouse.pressed(MouseButton::Right) {
        return Ok(());
    }

    if configuration.interactive_mode == InteractiveMode::None {
        return Ok(());
    }

    // Only ray cast against entities with the `Foo` component.
    let filter = |entity| foo_query.contains(entity);
    let settings = MeshRayCastSettings::default().with_filter(&filter);

    let Some(&(_, intersection)) = ray_map
        .iter()
        .filter_map(|(_, ray)| {
            let (_, hit) = ray_cast.cast_ray(*ray, &settings).first()?;
            Some((*ray, hit.point))
        })
        .collect_vec()
        .first()
    else {
        return Ok(());
    };

    let position = view_to_world(
        intersection,
        mesh_resmut.properties.translation,
        mesh_resmut.properties.scale,
    );
    let nearest_face = mesh_resmut.triangle_lookup.nearest(&position.into());

    // Draw the ray !
    let isometry1 = Isometry3d::new(
        intersection,
        Quat::from_rotation_arc(
            Vec3::Z,
            vector3d_to_vec3(mesh_resmut.mesh.normal(nearest_face)).normalize(),
        ),
    );
    gizmos.circle(isometry1, 0.1, colors::to_bevy(colors::BLACK));

    match configuration.interactive_mode {
        InteractiveMode::None => Ok(()),
        InteractiveMode::LoopModification => loop_modification_system(
            mouse,
            keyboard,
            mesh_resmut,
            solution,
            cache,
            gizmos,
            configuration,
            jobs,
            position,
            nearest_face,
        ),
        InteractiveMode::SegmentationModification => segmentation_modification_system(
            mouse,
            keyboard,
            mesh_resmut,
            solution,
            cache,
            gizmos,
            configuration,
            jobs,
            position,
            nearest_face,
        ),
    }
}
