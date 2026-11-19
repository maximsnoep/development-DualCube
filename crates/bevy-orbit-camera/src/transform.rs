use bevy::{ecs::prelude::*, math::prelude::*, reflect::Reflect, transform::components::Transform};

/// Within this distance (as `sin` of the angle) of looking along the orbit axis, up is not snapped to the axis.
const POLE: f32 = 1e-2;

/// An eye and the target it's looking at. As a component, this can be modified in place of bevy's `Transform`, and the two will stay in sync.
#[derive(Component, Debug, PartialEq, Clone, Copy, Reflect)]
#[reflect(Component, Debug, PartialEq)]
pub struct LookTransform {
    pub eye: Vec3,
    pub target: Vec3,
    /// Turns with the camera; up on screen is its part orthogonal to the view.
    pub up: Vec3,
}

impl LookTransform {
    pub fn new(eye: Vec3, target: Vec3, up: Vec3) -> Self {
        Self { eye, target, up }
    }

    pub fn radius(&self) -> f32 {
        (self.target - self.eye).length()
    }

    /// The unit direction from the eye to the target.
    pub fn forward(&self) -> Option<Vec3> {
        (self.target - self.eye).try_normalize()
    }

    /// The unit up on screen.
    pub fn screen_up(&self) -> Vec3 {
        let forward = self.forward().unwrap_or(Vec3::NEG_Z);
        valid_up(self.up, forward)
    }

    /// Turns the eye and up about the target by `rotation`.
    pub fn orbit(&mut self, rotation: Quat) {
        let radius = self.radius();
        let Some(forward) = self.forward() else {
            return;
        };
        // Turn up as given (so near the pole it stays near the orbit axis), unless it is too close to the view to say anything.
        let up = match self.up.try_normalize() {
            Some(up) if up.reject_from_normalized(forward).length() > POLE * POLE => up,
            _ => self.screen_up(),
        };
        self.up = (rotation * up).normalize();
        self.eye = self.target - (rotation * forward).normalize() * radius;
    }

    /// Writes up as `±axis` when that shows the same screen: the camera has no roll about `axis` and does not look
    /// (nearly) along it. Keeps the stored up readable; the view does not change.
    pub fn level(&mut self, axis: Vec3) {
        let (Some(forward), Some(axis)) = (self.forward(), axis.try_normalize()) else {
            return;
        };
        if axis.reject_from_normalized(forward).length() < POLE {
            return;
        }
        let screen_up = self.screen_up();
        if forward.cross(screen_up).dot(axis).abs() > 1e-4 {
            return;
        }
        self.up = axis * screen_up.dot(axis).signum();
    }
}

pub fn system(mut cameras: Query<(&LookTransform, &mut Transform), Changed<LookTransform>>) {
    for (look_transform, mut scene_transform) in cameras.iter_mut() {
        *scene_transform = (*look_transform).into();
    }
}

impl From<LookTransform> for Transform {
    fn from(t: LookTransform) -> Self {
        let direction = t.forward().unwrap_or(Vec3::NEG_Z);
        Transform::from_translation(t.eye).looking_at(t.eye + direction, t.screen_up())
    }
}

// `up` made orthogonal to `direction` and unit; any orthogonal direction if they are parallel.
fn valid_up(up: Vec3, direction: Vec3) -> Vec3 {
    up.reject_from_normalized(direction)
        .try_normalize()
        .unwrap_or_else(|| direction.any_orthonormal_vector())
}
