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
        Self {
            controller,
            controller_state: orbit_state(look_transform),
            look_transform,
            transform: look_transform.into(),
        }
    }
}

/// Moves an orbit camera to `look`; its orbit restarts from there.
pub fn look_at(camera: &mut EntityCommands<'_>, look: LookTransform) {
    camera.insert((look, Transform::from(look), orbit_state(look)));
}

fn orbit_state(look: LookTransform) -> control::OrbitControllerState {
    let up = look.up.try_normalize().unwrap_or(Vec3::Y);
    let forward = (look.target - look.eye)
        .try_normalize()
        .unwrap_or(Vec3::NEG_Z);
    control::OrbitControllerState::from_forward(forward, up)
}
