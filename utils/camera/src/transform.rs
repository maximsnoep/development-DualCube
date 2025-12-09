use bevy::{
    ecs::prelude::*, math::prelude::*, prelude::ReflectDefault, reflect::Reflect,
    transform::components::Transform,
};

/// An eye and the target it's looking at. As a component, this can be modified in place of bevy's `Transform`, and the two will
/// stay in sync.
#[derive(Component, Debug, PartialEq, Clone, Copy, Reflect)]
#[reflect(Component, Default, Debug, PartialEq)]
pub struct LookTransform {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
}

impl From<LookTransform> for Transform {
    fn from(t: LookTransform) -> Self {
        Transform::from_translation(t.eye).looking_at(t.eye + (t.target - t.eye).normalize(), t.up)
    }
}

impl Default for LookTransform {
    fn default() -> Self {
        Self {
            eye: Vec3::default(),
            target: Vec3::default(),
            up: Vec3::Y,
        }
    }
}

impl LookTransform {
    pub fn new(eye: Vec3, target: Vec3, up: Vec3) -> Self {
        Self { eye, target, up }
    }

    pub fn radius(&self) -> f32 {
        (self.target - self.eye).length()
    }

    pub fn look_direction(&self) -> Option<Vec3> {
        (self.target - self.eye).try_normalize()
    }
}

pub fn transform_system(mut cameras: Query<(&LookTransform, &mut Transform)>) {
    for (look_transform, mut scene_transform) in cameras.iter_mut() {
        *scene_transform = (*look_transform).into()
    }
}
