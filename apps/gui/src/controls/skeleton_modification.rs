//! Skeleton edits: the patch boundary nearest the cursor is highlighted (gray when it cannot dissolve), Delete
//! dissolves it.

use super::shared::{CacheResource, SkeletonBoundaries};
use crate::colors;
use crate::jobs::Job;
use crate::render::gizmos::{PerpetualGizmos, world_to_view};
use crate::resources::{InputResource, SkeletonResource};
use bevy::prelude::*;
use dualcube::prelude::*;
use dualcube::skeleton::{Skeleton, certify::Certificate, edit};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

pub fn skeleton_modification_system(
    keyboard: &ButtonInput<KeyCode>,
    input: &InputResource,
    skeleton: &SkeletonResource,
    mut cache: ResMut<'_, CacheResource>,
    mut gizmos: Gizmos<'_, '_, PerpetualGizmos>,
    mut jobs: MessageWriter<'_, Job>,
    position: Vector3D,
) {
    let (Some(extraction), Some(current)) = (skeleton.extraction(), skeleton.current_skeleton())
    else {
        return;
    };
    let boundaries = boundaries(&input.mesh, current, &mut cache);
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
    let color = colors::to_bevy(if dissolvable {
        colors::BLACK
    } else {
        colors::LIGHT_GRAY
    });
    let view = |p: Vector3D| world_to_view(p, input.properties.translation, input.properties.scale);
    for i in 0..points.len() {
        gizmos.line(view(points[i]), view(points[(i + 1) % points.len()]), color);
    }

    if keyboard.just_pressed(KeyCode::Delete) {
        if !dissolvable {
            warn!(
                "Skeleton loop not dissolved: its patches share another loop, which would become a self-loop"
            );
            return;
        }
        // A certified skeleton stays certified (see `Job::dissolve_skeleton_loop`); a labeled one was certified, a
        // restored save keeps no certificate.
        let certificate = skeleton.progress.certificate.clone().or_else(|| {
            skeleton
                .progress
                .labeled
                .is_some()
                .then(Certificate::default)
        });
        jobs.write(Job::dissolve_skeleton_loop(
            input.mesh.clone(),
            extraction.clone(),
            current.clone(),
            certificate,
            *edge,
            skeleton.last_stage(),
        ));
    }
}

/// The crossing positions of every boundary, cached while the boundaries stay.
fn boundaries(
    mesh: &mehsh::prelude::Mesh<INPUT>,
    skeleton: &Skeleton,
    cache: &mut CacheResource,
) -> Arc<SkeletonBoundaries> {
    let mut hasher = DefaultHasher::new();
    for (edge, data) in &skeleton.edges {
        edge.hash(&mut hasher);
        data.boundary.crossings.hash(&mut hasher);
    }
    let key = hasher.finish();
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
    boundaries
}
