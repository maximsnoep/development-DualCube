use crate::{look_angles::LookAngles, look_transform::LookTransform};
use bevy::input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel};
use bevy::prelude::*;

/// A 3rd person camera that orbits around the target.
#[derive(Clone, Component, Copy, Debug, Reflect)]
#[reflect(Component, Default, Debug)]
pub struct Controller {
    pub mouse_rotate_sensitivity: Vec2,
    pub mouse_translate_sensitivity: Vec2,
    pub mouse_wheel_zoom_sensitivity: f32,
    pub pixels_per_line: f32,
    pub smoothing_weight: f32,
}

impl Default for Controller {
    fn default() -> Self {
        Self {
            mouse_rotate_sensitivity: Vec2::splat(0.08),
            mouse_translate_sensitivity: Vec2::splat(0.1),
            mouse_wheel_zoom_sensitivity: 0.2,
            smoothing_weight: 0.8,
            pixels_per_line: 53.0,
        }
    }
}

pub fn control_system(
    time: Res<Time>,
    mut mouse_wheel_reader: MessageReader<MouseWheel>,
    mut mouse_motion_events: MessageReader<MouseMotion>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut cameras: Query<(&Controller, &mut LookTransform, &Transform)>,
) {
    // Can only control one camera at a time.
    let (
        Controller {
            mouse_rotate_sensitivity,
            mouse_translate_sensitivity,
            mouse_wheel_zoom_sensitivity,
            pixels_per_line,
            ..
        },
        mut transform,
        scene_transform,
    ) = match cameras.single_mut() {
        Ok((controller, look_transform, scene_transform)) => (controller, look_transform, scene_transform),
        Err(e) => {
            println!("Error handling bevy-orbit-camera controller: {}", e);
            return;
        }
    };

    let mut look_angles = LookAngles::from_vector(-transform.look_direction().unwrap());
    let mut radius_scalar = 1.0;
    let radius = transform.radius();
    let dt = time.delta_secs();
    let cursor_delta: Vec2 = mouse_motion_events.read().map(|event| event.delta).sum();

    // ORBIT
    if keyboard.pressed(KeyCode::ControlLeft) {
        let delta = mouse_rotate_sensitivity * cursor_delta;
        look_angles.add_yaw(dt * -delta.x);
        look_angles.add_pitch(dt * delta.y);
    }

    // TRANSLATE
    if mouse_buttons.pressed(MouseButton::Right) {
        let delta = mouse_translate_sensitivity * cursor_delta;
        let right_dir = scene_transform.rotation * -Vec3::X;
        let up_dir = scene_transform.rotation * Vec3::Y;
        transform.target += dt * delta.x * right_dir + dt * delta.y * up_dir;
    }

    // ZOOM
    let scalar = mouse_wheel_reader.read().fold(1.0, |acc, event| {
        // scale the event magnitude per pixel or per line
        let scroll_amount = match event.unit {
            MouseScrollUnit::Line => event.y,
            MouseScrollUnit::Pixel => event.y / pixels_per_line,
        };
        acc * (1.0 - scroll_amount * mouse_wheel_zoom_sensitivity)
    });
    radius_scalar *= scalar;

    transform.eye = transform.target + (radius_scalar * radius).clamp(0.001, 1000000.0) * look_angles.unit_vector();
}
