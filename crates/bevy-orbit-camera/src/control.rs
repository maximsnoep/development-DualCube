use std::f32;

use crate::transform::LookTransform;
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy::window::CursorMoved;

const ROTATION_RADIANS_PER_PIXEL: f32 = 0.01;
const TRANSLATION_UNITS_PER_PIXEL: f32 = 0.01;

// Prevents huge jumps from RDP, focus changes, cursor teleporting, etc.
const MAX_CURSOR_DELTA: f32 = 100.0;

#[derive(Clone, Component, Copy, Debug, Reflect)]
#[reflect(Component, Debug)]
pub(crate) struct OrbitControllerState {
    pub(crate) yaw: f32,
    pitch: f32,
    yaw_zero_forward: Vec3,
    last_cursor_position: Option<Vec2>,
}

/// A 3rd person camera that orbits around the target.
#[derive(Clone, Component, Copy, Debug, Reflect)]
#[reflect(Component, Default, Debug)]
pub struct Controller {
    pub mouse_rotate_sensitivity: Vec2,
    pub mouse_translate_sensitivity: Vec2,
    pub mouse_wheel_zoom_sensitivity: f32,
    pub pixels_per_line: f32,
}

impl Default for Controller {
    fn default() -> Self {
        Self {
            mouse_rotate_sensitivity: Vec2::splat(0.2),
            mouse_translate_sensitivity: Vec2::splat(2.0),
            mouse_wheel_zoom_sensitivity: 0.2,
            pixels_per_line: 53.0,
        }
    }
}

pub fn system(
    mut commands: Commands,
    mut mouse_wheel_reader: MessageReader<MouseWheel>,
    mut cursor_moved_events: MessageReader<CursorMoved>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut cameras: Query<(
        Entity,
        &Controller,
        Option<&mut OrbitControllerState>,
        &mut LookTransform,
    )>,
) {
    // Can only control one camera at a time.
    let (
        entity,
        Controller {
            mouse_rotate_sensitivity,
            mouse_translate_sensitivity,
            mouse_wheel_zoom_sensitivity,
            pixels_per_line,
            ..
        },
        controller_state,
        mut transform,
    ) = match cameras.single_mut() {
        Ok((entity, controller, controller_state, look_transform)) => {
            (entity, controller, controller_state, look_transform)
        }
        Err(_) => return,
    };

    // Amount of scroll since last update.
    let scroll_delta = mouse_wheel_reader.read().fold(1.0, |acc, event| {
        let scroll_amount = match event.unit {
            MouseScrollUnit::Line => event.y,
            MouseScrollUnit::Pixel => event.y / *pixels_per_line,
        };

        acc * (1.0 - scroll_amount * mouse_wheel_zoom_sensitivity)
    });

    // World up vector.
    let up = transform.up.try_normalize().unwrap_or(Vec3::Y);

    let Some(initial_forward) = (transform.target - transform.eye).try_normalize() else {
        return;
    };

    let mut state = controller_state
        .as_deref()
        .copied()
        .unwrap_or_else(|| OrbitControllerState::from_forward(initial_forward, up));

    let rotating =
        keyboard.pressed(KeyCode::ControlLeft) || mouse_buttons.pressed(MouseButton::Middle);
    let translating = mouse_buttons.pressed(MouseButton::Right);
    let using_mouse = rotating || translating;

    // CursorMoved gives absolute cursor positions. We derive a stable delta ourselves.
    let cursor_position = cursor_moved_events
        .read()
        .last()
        .map(|event| event.position);

    let cursor_delta = match cursor_position {
        Some(cursor_position) => {
            let delta = match state.last_cursor_position {
                Some(last_cursor_position) if using_mouse => {
                    let delta = cursor_position - last_cursor_position;

                    // Clamp instead of trusting large raw jumps.
                    if delta.length() > MAX_CURSOR_DELTA {
                        delta.normalize_or_zero() * MAX_CURSOR_DELTA
                    } else {
                        delta
                    }
                }
                _ => Vec2::ZERO,
            };

            state.last_cursor_position = Some(cursor_position);
            delta
        }
        None => Vec2::ZERO,
    };

    // ROTATE / ORBIT
    // Changes the forward vector.
    if rotating {
        let delta = mouse_rotate_sensitivity * cursor_delta * ROTATION_RADIANS_PER_PIXEL;

        state.yaw -= delta.x;
        state.pitch = (state.pitch - delta.y).clamp(
            -f32::consts::FRAC_PI_2 * 0.99,
            f32::consts::FRAC_PI_2 * 0.99,
        );
    }

    let forward = state.forward(up);

    // TRANSLATE
    // Changes the target vector.
    let mut target = transform.target;

    if translating {
        let delta = mouse_translate_sensitivity * cursor_delta * TRANSLATION_UNITS_PER_PIXEL;

        if let Some(right) = up.cross(forward).try_normalize() {
            let upish = forward.cross(right).normalize();
            target += delta.x * right + delta.y * upish;
        }
    }

    match controller_state {
        Some(mut controller_state) => *controller_state = state,
        None => {
            commands.entity(entity).insert(state);
        }
    }

    // ZOOM
    // Changes the radius.
    let radius = transform.radius() * scroll_delta;

    transform.target = target;
    transform.eye = target - forward * radius;
}

impl OrbitControllerState {
    pub(crate) fn from_forward(forward: Vec3, up: Vec3) -> Self {
        let pitch = forward.dot(up).clamp(-1.0, 1.0).asin();

        let yaw_zero_forward = (forward - up * forward.dot(up))
            .try_normalize()
            .unwrap_or_else(|| up.any_orthonormal_vector());

        Self {
            yaw: 0.0,
            pitch: pitch.clamp(
                -f32::consts::FRAC_PI_2 * 0.99,
                f32::consts::FRAC_PI_2 * 0.99,
            ),
            yaw_zero_forward,
            last_cursor_position: None,
        }
    }

    fn forward(&self, up: Vec3) -> Vec3 {
        let horizontal = Quat::from_axis_angle(up, self.yaw) * self.yaw_zero_forward;
        (horizontal * self.pitch.cos() + up * self.pitch.sin()).normalize()
    }
}
