use crate::jobs::{Job, JobRequest};
use crate::render::{add_line2, GizmosCache};
use crate::{colors, vec3_to_vector3d, CacheResource, Configuration, InputResource, SolutionResource};
use bevy::picking::backend::ray::RayMap;
use bevy::prelude::*;
use dualcube::prelude::*;
use itertools::Itertools;
use mehsh::prelude::*;
use ordered_float::OrderedFloat;

pub fn system(
    ray_map: Res<RayMap>,
    mut ray_cast: MeshRayCast,
    mouse: Res<ButtonInput<MouseButton>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mesh_resmut: Res<InputResource>,
    mut solution: ResMut<SolutionResource>,
    mut cache: ResMut<CacheResource>,
    mut gizmos_cache: ResMut<GizmosCache>,
    mut configuration: ResMut<Configuration>,
    mut jobs: EventWriter<JobRequest>,
) -> Result<(), BevyError> {
    configuration.raycasted = None;
    configuration.selected = None;
    gizmos_cache.raycaster.clear();

    if !configuration.interactive {
        return Ok(());
    }

    // Render all current solutions  (for currently selected direction)
    for (&edgepair, sol) in &solution.next[configuration.direction as usize] {
        let u = mesh_resmut.mesh.position(edgepair[0]);
        let v = mesh_resmut.mesh.position(edgepair[1]);
        let n = mesh_resmut.mesh.normal(mesh_resmut.mesh.face(edgepair[0]));
        let color = match sol {
            Some(_) => colors::from_direction(configuration.direction, Some(Perspective::Dual), None),
            None => colors::BLACK,
        };
        add_line2(
            &mut gizmos_cache.raycaster,
            u,
            v,
            n * 0.01,
            color,
            mesh_resmut.properties.translation,
            mesh_resmut.properties.scale,
        );
    }

    if keyboard.pressed(KeyCode::ControlLeft) || mouse.pressed(MouseButton::Right) {
        return Ok(());
    }

    let intersections = ray_map
        .iter()
        .filter_map(|(_, ray)| {
            let (_, hit) = ray_cast.cast_ray(*ray, &MeshRayCastSettings::default()).first()?;
            Some(hit.point)
        })
        .collect_vec();

    if intersections.is_empty() {
        return Ok(());
    }

    let position = (vec3_to_vector3d(intersections[0]) - mesh_resmut.properties.translation) / mesh_resmut.properties.scale;

    let nearest_face = mesh_resmut.triangle_lookup.nearest(&position.into());
    // get the nearest_vert (one of 3 corners of nearest_face)
    let nearest_vert = mesh_resmut
        .mesh
        .vertices(nearest_face)
        .iter()
        .min_by_key(|&v| OrderedFloat(position.metric_distance(&mesh_resmut.mesh.position(*v))))
        .unwrap()
        .to_owned();

    let edgepair = mesh_resmut.mesh.edges_in_face_with_vert(nearest_face, nearest_vert).unwrap();

    let u = mesh_resmut.mesh.position(edgepair[0]);
    let v = mesh_resmut.mesh.position(edgepair[1]);
    let n = mesh_resmut.mesh.normal(mesh_resmut.mesh.face(edgepair[0]));
    let color = colors::from_direction(configuration.direction, Some(Perspective::Dual), None);

    add_line2(
        &mut gizmos_cache.raycaster,
        u,
        v,
        n * 0.01,
        color,
        mesh_resmut.properties.translation,
        mesh_resmut.properties.scale,
    );

    // CONTROLS
    //
    // Action1:  Add loop (LMB, no Shift)
    // Action2:  Show closest solution (Shift, no LMB)
    // Action3:  Add closest solution (LMB + Shift)
    //
    // Action4:  Remove closest valid loop (Delete, no Shift)
    // Action5:  Remove closest loop (Delete + Shift)

    let lmb = mouse.pressed(MouseButton::Left);
    let shift = keyboard.pressed(KeyCode::ShiftLeft);
    let delete = keyboard.just_pressed(KeyCode::Delete) || keyboard.just_released(KeyCode::Delete);

    let closest_solution = solution.next[configuration.direction as usize]
        .iter()
        .map(|(&edgepair, sol)| {
            (
                ((mesh_resmut.mesh.position(edgepair[0]) + mesh_resmut.mesh.position(edgepair[1])) / 2.).metric_distance(&position),
                sol,
                edgepair,
            )
        })
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    match (shift, lmb, delete) {
        // Action1
        (false, true, false) => {
            let selected_edges = mesh_resmut.mesh.edges_in_face_with_vert(nearest_face, nearest_vert).unwrap();

            jobs.write(JobRequest::Run(Box::new(Job::AddLoop {
                seed: selected_edges,
                direction: configuration.direction,
                flowgraph: mesh_resmut.flow_graphs[configuration.direction as usize].clone(),
                solution: solution.current_solution.clone(),
            })));
        }
        // Action 2
        (true, false, false) => {
            if let Some((_, Some(some_solution), signature)) = closest_solution {
                configuration.selected = Some(signature);

                for loop_id in some_solution.loops.keys() {
                    let direction = some_solution.loop_to_direction(loop_id);
                    let color = colors::from_direction(direction, Some(Perspective::Dual), None);
                    for &edgepair in &some_solution.get_pairs_of_loop(loop_id) {
                        let u = mesh_resmut.mesh.position(edgepair[0]);
                        let v = mesh_resmut.mesh.position(edgepair[1]);
                        let n = mesh_resmut.mesh.normal(edgepair[0]);
                        add_line2(
                            &mut gizmos_cache.raycaster,
                            u,
                            v,
                            n * 0.05,
                            color,
                            mesh_resmut.properties.translation,
                            mesh_resmut.properties.scale,
                        );
                    }
                }
            }
        }
        // Action 3
        (true, true, false) => {
            if let Some((_, Some(sol), _)) = closest_solution {
                solution.current_solution = sol.clone();
                solution.next[0].clear();
                solution.next[1].clear();
                solution.next[2].clear();
                cache.cache[0].clear();
                cache.cache[1].clear();
                cache.cache[2].clear();
                jobs.write(JobRequest::Run(Box::new(Job::Recompute {
                    solution: solution.current_solution.clone(),
                    unit: true,
                    omega: 1,
                })));
            }
        }
        // Action 4 and Action 5
        (force, _, true) => {
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
                })));
            }
        }
        // No actions
        _ => {}
    }

    Ok(())
}
