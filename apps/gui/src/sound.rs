//! A "ding" when the optimization finds a new best solution (if switched on, see `OptimizationLive::ding`). The sound
//! is a short drum tick, generated (as WAV) at startup.

use crate::jobs::OptimizationLive;
use bevy::audio::{AudioPlayer, AudioSource, PlaybackSettings};
use bevy::prelude::*;
use std::sync::Arc;

pub struct SoundPlugin;

impl Plugin for SoundPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup)
            .add_systems(Update, ding_on_new_best);
    }
}

#[derive(Resource)]
struct Ding(Handle<AudioSource>);

fn setup(mut commands: Commands<'_, '_>, mut sources: ResMut<'_, Assets<AudioSource>>) {
    let source = AudioSource {
        bytes: Arc::from(tick_wav()),
    };
    commands.insert_resource(Ding(sources.add(source)));
}

// A ding when the best quality of the running optimization improves (not for its start).
fn ding_on_new_best(
    mut commands: Commands<'_, '_>,
    mut live: ResMut<'_, OptimizationLive>,
    ding: Res<'_, Ding>,
) {
    let Some(best) = live
        .monitor
        .as_ref()
        .and_then(dualcube::prelude::OptimizationMonitor::best_quality)
    else {
        return;
    };
    let live = live.bypass_change_detection();
    let improved = live.dinged.is_some_and(|dinged| best > dinged + 1e-12);
    live.dinged = Some(best);
    if improved && live.ding {
        commands.spawn((AudioPlayer(ding.0.clone()), PlaybackSettings::DESPAWN));
    }
}

// A drum tick (like a stick on a rim or a woodblock): a short, quickly fading high knock with a burst of noise, as a
// mono 16-bit WAV file.
fn tick_wav() -> Vec<u8> {
    const RATE: u32 = 44_100;
    const SECONDS: f32 = 0.08;
    let samples = (RATE as f32 * SECONDS) as usize;
    // A fixed pseudo-random noise (a linear congruential generator), so the tick is the same every time.
    let mut seed = 0x1234_5678u32;
    let mut noise = move || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 8) as f32 / (1u32 << 23) as f32 - 1.
    };
    let data = (0..samples)
        .map(|i| {
            let t = i as f32 / RATE as f32;
            let attack = (t / 0.0005).min(1.);
            let knock = (std::f32::consts::TAU * 1800. * t).sin() * (-t * 70.).exp();
            let click = noise() * (-t * 250.).exp();
            let value = attack * (0.8 * knock + 0.3 * click);
            (value * 0.25 * f32::from(i16::MAX)) as i16
        })
        .collect::<Vec<_>>();
    let length = (data.len() * 2) as u32;
    let mut wav = Vec::with_capacity(44 + data.len() * 2);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + length).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    // PCM, mono, the sample rate, the byte rate, the block size, 16 bits.
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&RATE.to_le_bytes());
    wav.extend_from_slice(&(RATE * 2).to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&length.to_le_bytes());
    for sample in data {
        wav.extend_from_slice(&sample.to_le_bytes());
    }
    wav
}
