pub mod automatic;
mod control;
mod transform;

pub use control::Controller;
pub use transform::LookTransform;

use bevy::prelude::*;

pub struct OrbitCameraPlugin;
impl Plugin for OrbitCameraPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<automatic::AutomaticRotation>()
            .add_systems(
                Update,
                (control::system, automatic::update, transform::system).chain(),
            );
    }
}

#[derive(Bundle)]
pub struct OrbitCameraBundle {
    controller: control::Controller,
    controller_state: control::OrbitControllerState,
    look_transform: transform::LookTransform,
    transform: Transform,
}
impl OrbitCameraBundle {
    pub fn new(controller: control::Controller, eye: Vec3, target: Vec3, up: Vec3) -> Self {
        let look_transform = transform::LookTransform::new(eye, target, up);
        let up = up.try_normalize().unwrap_or(Vec3::Y);
        let controller_state = (target - eye)
            .try_normalize()
            .map(|forward| control::OrbitControllerState::from_forward(forward, up))
            .unwrap_or_else(|| control::OrbitControllerState::from_forward(Vec3::NEG_Z, up));

        Self {
            controller,
            controller_state,
            look_transform,
            transform: look_transform.into(),
        }
    }
}
