use crate::transform::LookTransform;
use bevy::input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use std::f32::consts::PI;

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
            mouse_rotate_sensitivity: Vec2::splat(0.08),
            mouse_translate_sensitivity: Vec2::splat(0.1),
            mouse_wheel_zoom_sensitivity: 0.2,
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
        Ok((controller, look_transform, scene_transform)) => {
            (controller, look_transform, scene_transform)
        }
        Err(e) => {
            println!("Error handling bevy-orbit-camera controller: {}", e);
            return;
        }
    };

    let mut look_angles = LookAngles::from_vector(-transform.look_direction().unwrap());
    let mut radius_scalar = 1.0;
    let time_delta = time.delta_secs();
    let cursor_delta: Vec2 = mouse_motion_events.read().map(|event| event.delta).sum();

    // ORBIT
    if keyboard.pressed(KeyCode::ControlLeft) {
        let delta = mouse_rotate_sensitivity * cursor_delta;
        look_angles.add_yaw(time_delta * -delta.x);
        look_angles.add_pitch(time_delta * delta.y);
    }

    // TRANSLATE
    if mouse_buttons.pressed(MouseButton::Right) {
        let delta = mouse_translate_sensitivity * cursor_delta;
        let right_dir = scene_transform.rotation * -Vec3::X;
        let up_dir = scene_transform.rotation * Vec3::Y;
        transform.target += time_delta * delta.x * right_dir + time_delta * delta.y * up_dir;
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

    transform.eye = transform.target
        + (radius_scalar * transform.radius()).clamp(0.001, 1000000.0) * look_angles.unit_vector();
}

/// A (yaw, pitch) pair representing a direction.
#[derive(Debug, PartialEq, Clone, Copy, Default, Reflect)]
#[reflect(Default, Debug, PartialEq)]
pub struct LookAngles {
    // The fields are protected to keep them in an allowable range for the camera transform.
    yaw: f32,
    pitch: f32,
}

impl LookAngles {
    pub fn from_vector(v: Vec3) -> Self {
        let mut p = Self::default();
        p.set_direction(v);
        p
    }

    pub fn unit_vector(self) -> Vec3 {
        unit_vector_from_yaw_and_pitch(self.yaw, self.pitch)
    }

    pub fn set_direction(&mut self, v: Vec3) {
        let (yaw, pitch) = yaw_and_pitch_from_vector(v);
        self.set_yaw(yaw);
        self.set_pitch(pitch);
    }

    pub fn set_yaw(&mut self, yaw: f32) {
        self.yaw = yaw % (2.0 * PI);
    }

    pub fn add_yaw(&mut self, delta: f32) {
        self.set_yaw(self.yaw + delta);
    }

    pub fn set_pitch(&mut self, pitch: f32) {
        // Things can get weird if we are parallel to the UP vector.
        let up_eps = 0.01;
        self.pitch = pitch.clamp(-PI / 2.0 + up_eps, PI / 2.0 - up_eps);
    }

    pub fn add_pitch(&mut self, delta: f32) {
        self.set_pitch(self.pitch + delta);
    }
}

/// Returns pitch and yaw angles that rotates z unit vector to v. The yaw is applied first to z about the y axis to get z'. Then
/// the pitch is applied about some axis orthogonal to z' in the XZ plane to get v.
fn yaw_and_pitch_from_vector(v: Vec3) -> (f32, f32) {
    debug_assert_ne!(v, Vec3::ZERO);

    let y = Vec3::Y;
    let z = Vec3::Z;

    let v_xz = Vec3::new(v.x, 0.0, v.z);

    if v_xz == Vec3::ZERO {
        if v.dot(y) > 0.0 {
            return (0.0, PI / 2.0);
        } else {
            return (0.0, -PI / 2.0);
        }
    }

    let mut yaw = v_xz.angle_between(z);
    if v.x < 0.0 {
        yaw *= -1.0;
    }

    let mut pitch = v_xz.angle_between(v);
    if v.y < 0.0 {
        pitch *= -1.0;
    }

    (yaw, pitch)
}

fn unit_vector_from_yaw_and_pitch(yaw: f32, pitch: f32) -> Vec3 {
    let ray = Mat3::from_rotation_y(yaw) * Vec3::Z;
    let pitch_axis = ray.cross(Vec3::Y);
    Mat3::from_axis_angle(pitch_axis, pitch) * ray
}
