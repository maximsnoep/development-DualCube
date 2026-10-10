use super::shared::{CacheResource, LoopPreviewKey, draw_edgepair_arrow, draw_polyline_gradient};
use crate::colors;
use crate::jobs::Job;
use crate::render::gizmos::PerpetualGizmos;
use crate::resources::{Configuration, InputResource, SolutionResource};
use bevy::prelude::*;
use dualcube::prelude::*;
pub fn loop_modification_system(
    mouse: Res<'_, ButtonInput<MouseButton>>,
    keyboard: Res<'_, ButtonInput<KeyCode>>,
    mesh_resmut: Res<'_, InputResource>,
    mut solution: ResMut<'_, SolutionResource>,
    mut cache: ResMut<'_, CacheResource>,
    mut gizmos: Gizmos<'_, '_, PerpetualGizmos>,
    mut configuration: ResMut<'_, Configuration>,
    mut jobs: MessageWriter<'_, Job>,
    position: Vector3D,
    nearest_face: FaceID,
) -> Result<(), BevyError> {
    if mesh_resmut.mesh.nr_verts() == 0 {
        return Ok(());
    }

    // Get the nearest vertex in the hovered face and use its opposite edge pair
    // as the interactive loop anchor/seed.
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

    solution.current_solution.prepare_flow();

    let direction = configuration.direction;
    if cache
        .locked_loop_direction
        .is_some_and(|locked| locked != direction)
    {
        cache.locked_loop_segments.clear();
        cache.locked_loop_direction = None;
        cache.loop_preview = None;
        cache.loop_preview_key = None;
        cache.loop_preview_segments.clear();
    }
    let anchor_color = colors::to_bevy(colors::from_direction(
        direction,
        Some(Perspective::Dual),
        None,
    ));
    let preview_color =
        colors::from_direction(direction, Some(Perspective::Dual), Some(Sign::Negative));

    // Draw already selected anchors.
    for &anchor in &configuration.loop_anchors {
        draw_edgepair_arrow(&mesh_resmut, &mut gizmos, anchor, anchor_color);
    }

    // Draw hovered anchor candidate.
    draw_edgepair_arrow(&mesh_resmut, &mut gizmos, edgepair, anchor_color);

    let mut preview_anchors = configuration.loop_anchors.clone();
    preview_anchors.push(edgepair);
    let preview_key = LoopPreviewKey {
        direction,
        hover: edgepair,
        anchors: configuration.loop_anchors.clone(),
    };

    if cache.loop_preview_key.as_ref() != Some(&preview_key) {
        cache.loop_preview_gaps = None;
        // Valid loops: the cheapest loop through the hovered edge that is valid by construction (only for a single
        // anchor, and if the current loops form a valid loop structure).
        let valid = (configuration.loop_anchors.is_empty()
            && solution.current_solution.dual.is_ok())
        .then(|| {
            solution
                .current_solution
                .best_valid_loop_through(direction, edgepair[0], 6)
        });
        if let Some(found) = valid {
            if let Some((edges, cost, gaps)) = found {
                cache.loop_preview_positions = solution
                    .current_solution
                    .loop_positions_preview(&edges, direction);
                cache.loop_preview = Some((edges, cost));
                cache.loop_preview_gaps = Some(gaps);
            } else {
                cache.loop_preview = None;
            }
            cache.loop_preview_segments.clear();
        } else if let Some((edges, cost, segments)) = solution
            .current_solution
            .construct_loop_with_anchors_and_locked_segments(
                &preview_anchors,
                direction,
                &cache.locked_loop_segments,
                OrderedFloat,
            )
        {
            cache.loop_preview_positions = solution
                .current_solution
                .loop_positions_preview(&edges, direction);
            cache.loop_preview = Some((edges, cost));
            cache.loop_preview_segments = segments;
        } else {
            cache.loop_preview = None;
            cache.loop_preview_segments.clear();
        }
        cache.loop_preview_key = Some(preview_key);
    }

    if cache.loop_preview.is_some() {
        draw_polyline_gradient(
            &mesh_resmut,
            &mut gizmos,
            &cache.loop_preview_positions,
            preview_color,
        );
    }

    let lmb = mouse.just_pressed(MouseButton::Left);
    let enter = keyboard.just_pressed(KeyCode::Enter);
    let delete = keyboard.just_pressed(KeyCode::Delete);
    let force = keyboard.pressed(KeyCode::ShiftLeft);

    if keyboard.just_pressed(KeyCode::Escape) {
        configuration.loop_anchors.clear();
        cache.loop_preview = None;
        cache.loop_preview_key = None;
        cache.loop_preview_segments.clear();
        cache.locked_loop_segments.clear();
        cache.locked_loop_direction = None;
    }

    if lmb {
        configuration.loop_anchors.push(edgepair);
        cache.locked_loop_segments = cache.loop_preview_segments.clone();
        cache.locked_loop_segments.pop();
        cache.locked_loop_direction = Some(direction);
        cache.loop_preview = None;
        cache.loop_preview_key = None;
        cache.loop_preview_segments.clear();
        debug!("loop anchors now: {:?}", configuration.loop_anchors);
    }

    if enter {
        if let Some((edges, _)) = cache.loop_preview.clone() {
            configuration.loop_anchors.clear();
            cache.loop_preview = None;
            cache.loop_preview_key = None;
            cache.loop_preview_segments.clear();
            cache.locked_loop_segments.clear();
            cache.locked_loop_direction = None;
            jobs.write(Job::add_loop(
                solution.current_solution.clone(),
                Loop::new(edges, direction),
                cache.loop_preview_gaps.take(),
                force,
                configuration.clone(),
            ));
        }
    }

    if delete {
        let option_a = [edgepair[0], edgepair[1]];
        let option_b = [edgepair[1], edgepair[0]];

        if let Some(loop_id) = solution.current_solution.loops.keys().find(|&loop_id| {
            let edges = solution.current_solution.get_pairs_of_loop(loop_id);
            edges.contains(&option_a) || edges.contains(&option_b)
        }) {
            jobs.write(Job::remove_loop(
                solution.current_solution.clone(),
                loop_id,
                configuration.clone(),
            ));
        }
    }

    Ok(())
}
