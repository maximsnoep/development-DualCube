use crate::transform::LookTransform;
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy::window::CursorMoved;

const ROTATION_RADIANS_PER_PIXEL: f32 = 0.01;
const TRANSLATION_UNITS_PER_PIXEL: f32 = 0.01;

// Prevents huge jumps from RDP, focus changes, cursor teleporting, etc.
const MAX_CURSOR_DELTA: f32 = 100.0;

#[derive(Clone, Component, Copy, Debug, Default, Reflect)]
#[reflect(Component, Debug)]
pub(crate) struct OrbitControllerState {
    last_cursor_position: Option<Vec2>,
    /// Whether a rotation drag is going on, and the sign of its horizontal turn (fixed for the drag).
    drag_yaw_sign: Option<f32>,
}

/// A 3rd person camera that orbits around the target.
#[derive(Clone, Component, Copy, Debug, Reflect)]
#[reflect(Component, Default, Debug)]
pub struct Controller {
    pub mouse_rotate_sensitivity: Vec2,
    pub mouse_translate_sensitivity: Vec2,
    pub mouse_wheel_zoom_sensitivity: f32,
    pub pixels_per_line: f32,
    /// Horizontal drags orbit about this axis (the world up); vertical drags tilt over it freely.
    pub orbit_axis: Vec3,
}

impl Default for Controller {
    fn default() -> Self {
        Self {
            mouse_rotate_sensitivity: Vec2::splat(0.2),
            mouse_translate_sensitivity: Vec2::splat(2.0),
            mouse_wheel_zoom_sensitivity: 0.2,
            pixels_per_line: 53.0,
            orbit_axis: Vec3::Y,
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
            orbit_axis,
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

    if transform.forward().is_none() {
        return;
    }

    let mut state = controller_state.as_deref().copied().unwrap_or_default();

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
    // Turns the eye and up about the target.
    if rotating {
        let yaw_sign = *state
            .drag_yaw_sign
            .get_or_insert_with(|| yaw_sign(&transform, *orbit_axis));
        let delta = mouse_rotate_sensitivity * cursor_delta * ROTATION_RADIANS_PER_PIXEL;
        if delta != Vec2::ZERO {
            turn(&mut transform, *orbit_axis, -delta.x * yaw_sign, -delta.y);
        }
    } else {
        state.drag_yaw_sign = None;
    }

    // TRANSLATE
    // Changes the target vector.
    if translating && cursor_delta != Vec2::ZERO {
        let delta = mouse_translate_sensitivity * cursor_delta * TRANSLATION_UNITS_PER_PIXEL;
        let forward = transform.forward().unwrap_or(Vec3::NEG_Z);
        let up = transform.screen_up();
        let right = up.cross(forward);
        let shift = delta.x * right + delta.y * up;
        transform.target += shift;
        transform.eye += shift;
    }

    match controller_state {
        Some(mut controller_state) => *controller_state = state,
        None => {
            commands.entity(entity).insert(state);
        }
    }

    // ZOOM
    // Changes the radius.
    if scroll_delta != 1.0 {
        let forward = transform.forward().unwrap_or(Vec3::NEG_Z);
        let radius = transform.radius() * scroll_delta;
        transform.eye = transform.target - forward * radius;
    }
}

/// Orbits `look` by `yaw` about `axis`, then by `pitch` toward its screen up; there is no limit to either.
pub(crate) fn turn(look: &mut LookTransform, axis: Vec3, yaw: f32, pitch: f32) {
    let axis = axis.try_normalize().unwrap_or(Vec3::Y);
    if yaw != 0.0 {
        look.orbit(Quat::from_axis_angle(axis, yaw));
    }
    if pitch != 0.0
        && let Some(forward) = look.forward()
    {
        look.orbit(Quat::from_axis_angle(
            forward.cross(look.screen_up()),
            pitch,
        ));
    }
    look.level(axis);
}

/// -1 while the camera is upside down about `axis`, so horizontal drags still move the model along with the mouse.
pub(crate) fn yaw_sign(look: &LookTransform, axis: Vec3) -> f32 {
    if look.screen_up().dot(axis) < 0.0 {
        -1.0
    } else {
        1.0
    }
}
