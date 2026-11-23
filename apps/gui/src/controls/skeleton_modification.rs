//! Skeleton edits: the patch boundary nearest the cursor is highlighted (gray when it cannot dissolve), a click
//! selects or deselects it, Escape clears the selection. Leaving the mode dissolves the selected boundaries.

use super::shared::{CacheResource, InteractiveMode, SkeletonBoundaries};
use crate::colors;
use crate::jobs::{Job, JobState};
use crate::render::gizmos::{PerpetualGizmos, world_to_view};
use crate::resources::{Configuration, InputResource, SkeletonResource};
use bevy::prelude::*;
use dualcube::prelude::*;
use dualcube::skeleton::{Skeleton, certify::Certificate, edit};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

pub fn skeleton_modification_system(
    mouse: &ButtonInput<MouseButton>,
    keyboard: &ButtonInput<KeyCode>,
    input: &InputResource,
    skeleton: &SkeletonResource,
    mut cache: ResMut<'_, CacheResource>,
    mut gizmos: Gizmos<'_, '_, PerpetualGizmos>,
    position: Vector3D,
) {
    let Some(current) = skeleton.current_skeleton() else {
        return;
    };
    let boundaries = boundaries(&input.mesh, current, &mut cache);
    let view = |p: Vector3D| world_to_view(p, input.properties.translation, input.properties.scale);
    let mut draw = |points: &[Vector3D], color: colors::Kolor| {
        for i in 0..points.len() {
            gizmos.line(
                view(points[i]),
                view(points[(i + 1) % points.len()]),
                colors::to_bevy(color),
            );
        }
    };
    for (edge, points) in boundaries.iter() {
        if cache.skeleton_selection.contains(edge) {
            draw(points, colors::WARN_RED);
        }
    }

    if keyboard.just_pressed(KeyCode::Escape) {
        cache.skeleton_selection.clear();
    }

    let distance = |points: &[Vector3D]| {
        points
            .iter()
            .map(|p| p.metric_distance(&position))
            .fold(f64::INFINITY, f64::min)
    };
    let Some((edge, points)) = boundaries
        .iter()
        .min_by_key(|(_, points)| OrderedFloat(distance(points)))
    else {
        return;
    };
    let dissolvable = edit::dissolvable(current, *edge);
    draw(
        points,
        if dissolvable {
            colors::BLACK
        } else {
            colors::LIGHT_GRAY
        },
    );

    if mouse.just_pressed(MouseButton::Left) {
        let selection = &mut cache.skeleton_selection;
        if let Some(i) = selection.iter().position(|e| e == edge) {
            selection.remove(i);
        } else if dissolvable {
            selection.push(*edge);
        } else {
            warn!(
                "Skeleton loop not selected: its patches share another loop, which would become a self-loop"
            );
            return;
        }
        info!("Skeleton loops selected: {}", selection.len());
    }
}

/// On leaving the skeleton mode, dissolve the selected boundaries. While another job runs, the selection is kept for
/// the next time the mode is left.
pub fn apply_selection_on_leave(
    configuration: Res<'_, Configuration>,
    mut previous: Local<'_, Option<InteractiveMode>>,
    mut cache: ResMut<'_, CacheResource>,
    input: Res<'_, InputResource>,
    skeleton: Res<'_, SkeletonResource>,
    job_state: Res<'_, JobState>,
    mut jobs: MessageWriter<'_, Job>,
) {
    let mode = configuration.interactive_mode;
    let left = previous.replace(mode) == Some(InteractiveMode::SkeletonModification)
        && mode != InteractiveMode::SkeletonModification;
    if !left || cache.skeleton_selection.is_empty() {
        return;
    }
    let (Some(extraction), Some(current)) = (skeleton.extraction(), skeleton.current_skeleton())
    else {
        cache.skeleton_selection.clear();
        return;
    };
    // Edge keys are local to their skeleton: a selection on another one is dropped.
    if cache.skeleton_boundaries.as_ref().map(|(key, _)| *key) != Some(boundaries_key(current)) {
        warn!("Skeleton loop selection dropped: the skeleton changed");
        cache.skeleton_selection.clear();
        return;
    }
    if let Some(running) = job_state.request {
        warn!(
            "Skeleton loops not dissolved while {running}; the selection is kept, leave the skeleton mode again to apply it"
        );
        return;
    }
    // A certified skeleton stays certified (see `Job::dissolve_skeleton_loops`); a labeled one was certified, a
    // restored save keeps no certificate.
    let certificate = skeleton.progress.certificate.clone().or_else(|| {
        skeleton
            .progress
            .labeled
            .is_some()
            .then(Certificate::default)
    });
    jobs.write(Job::dissolve_skeleton_loops(
        input.mesh.clone(),
        extraction.clone(),
        current.clone(),
        certificate,
        std::mem::take(&mut cache.skeleton_selection),
        skeleton.last_stage(),
    ));
}

/// Identifies a skeleton's boundaries (their keys and crossings).
fn boundaries_key(skeleton: &Skeleton) -> u64 {
    let mut hasher = DefaultHasher::new();
    for (edge, data) in &skeleton.edges {
        edge.hash(&mut hasher);
        data.boundary.crossings.hash(&mut hasher);
    }
    hasher.finish()
}

/// The crossing positions of every boundary, cached while the boundaries stay; other boundaries clear the selection.
fn boundaries(
    mesh: &mehsh::prelude::Mesh<INPUT>,
    skeleton: &Skeleton,
    cache: &mut CacheResource,
) -> Arc<SkeletonBoundaries> {
    let key = boundaries_key(skeleton);
    if let Some((cached, boundaries)) = &cache.skeleton_boundaries
        && *cached == key
    {
        return boundaries.clone();
    }
    let positions = skeleton.surface.positions(mesh);
    let boundaries: Arc<SkeletonBoundaries> = Arc::new(
        skeleton
            .edges
            .iter()
            .map(|(edge, data)| {
                let points = data.boundary.crossings.iter().map(|c| positions[c]);
                (edge, points.collect())
            })
            .collect(),
    );
    cache.skeleton_boundaries = Some((key, boundaries.clone()));
    cache.skeleton_selection.clear();
    boundaries
}

#[cfg(test)]
mod tests;
