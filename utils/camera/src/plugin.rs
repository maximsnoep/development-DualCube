pub struct OrbitCameraPlugin;
use crate::control::*;
use crate::transform::*;
use bevy::prelude::*;

impl Plugin for OrbitCameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, control_system)
            .add_systems(Update, transform_system);
    }
}

#[derive(Bundle)]
pub struct OrbitCameraBundle {
    controller: Controller,
    look_transform: LookTransform,
    transform: Transform,
}

impl OrbitCameraBundle {
    pub fn new(controller: Controller, eye: Vec3, target: Vec3, up: Vec3) -> Self {
        Self {
            controller,
            look_transform: LookTransform::new(eye, target, up),
            transform: Transform::from_translation(eye).looking_at(target, up),
        }
    }
}
