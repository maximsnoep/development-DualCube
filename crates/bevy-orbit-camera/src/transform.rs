use bevy::{ecs::prelude::*, math::prelude::*, reflect::Reflect, transform::components::Transform};

/// An eye and the target it's looking at. As a component, this can be modified in place of bevy's `Transform`, and the two will stay in sync.
#[derive(Component, Debug, PartialEq, Clone, Copy, Reflect)]
#[reflect(Component, Debug, PartialEq)]
pub struct LookTransform {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
}

impl LookTransform {
    pub fn new(eye: Vec3, target: Vec3, up: Vec3) -> Self {
        Self { eye, target, up }
    }

    pub fn radius(&self) -> f32 {
        (self.target - self.eye).length()
    }
}

pub fn system(mut cameras: Query<(&LookTransform, &mut Transform), Changed<LookTransform>>) {
    for (look_transform, mut scene_transform) in cameras.iter_mut() {
        *scene_transform = (*look_transform).into();
    }
}

impl From<LookTransform> for Transform {
    fn from(t: LookTransform) -> Self {
        let direction = (t.target - t.eye).try_normalize().unwrap_or(Vec3::NEG_Z);
        let up = valid_up(t.up, direction);
        Transform::from_translation(t.eye).looking_at(t.eye + direction, up)
    }
}

fn valid_up(up: Vec3, direction: Vec3) -> Vec3 {
    let up = up.try_normalize().unwrap_or(Vec3::Y);
    if up.cross(direction).length_squared() > 1e-6 {
        up
    } else {
        direction.any_orthonormal_vector()
    }
}
