use bevy::core_pipeline::bloom::{Bloom, BloomCompositeMode, BloomPrefilter};
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::prelude::*;
use bevy::time::common_conditions::on_timer;
use std::time::Duration;

///
fn funky_bloom() -> Bloom {
    Bloom {
        intensity: 0.15,
        high_pass_frequency: 0.35,
        prefilter: BloomPrefilter {
            threshold: 0.3,
            threshold_softness: 0.0,
        },
        composite_mode: BloomCompositeMode::Additive,
        ..Default::default()
    }
}

/// Ensures HDR is on and bloom settings match `funky_bloom()`.
fn ensure_bloom_on_cameras(
    mut commands: Commands,
    mut q: Query<(Entity, &mut Camera, Option<&mut Bloom>), With<Camera3d>>,
) {
    let bloom = funky_bloom();

    for (e, mut camera, maybe_bloom) in &mut q {
        camera.hdr = true;

        // Optional but recommended with bloom
        commands.entity(e).insert(Tonemapping::TonyMcMapface);

        match maybe_bloom {
            Some(mut existing) => {
                // overwrite (or tweak fields)
                *existing = bloom.clone();
            }
            None => {
                // add bloom if missing
                commands.entity(e).insert(bloom.clone());
            }
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct BloomPlugin;

impl Plugin for BloomPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            FixedUpdate,
            ensure_bloom_on_cameras.run_if(on_timer(Duration::from_millis(1000))),
        );
    }
}
