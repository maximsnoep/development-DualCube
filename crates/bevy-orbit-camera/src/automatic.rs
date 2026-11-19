use crate::{control::Controller, transform::LookTransform};
use bevy::prelude::*;

#[derive(Component)]
pub struct Marker;

#[derive(Resource)]
pub struct AutomaticRotation {
    pub enabled: bool,
    pub sensitivity: f32, // amount of radians to be rotated per second
}

impl Default for AutomaticRotation {
    fn default() -> Self {
        Self {
            enabled: false,
            sensitivity: 1.0,
        }
    }
}

pub(crate) fn update(
    mut cameras: Query<(&mut LookTransform, Option<&Controller>), With<Marker>>,
    automatic_rotation: Res<AutomaticRotation>,
    time: Res<Time>,
) {
    if !automatic_rotation.enabled {
        return;
    }

    // radians per second * dt
    let yaw = automatic_rotation.sensitivity * time.delta_secs();
    for (mut t, controller) in cameras.iter_mut() {
        // Spins about the orbit axis, the same way a horizontal drag does.
        let axis = controller.map_or(t.up, |c| c.orbit_axis);
        let sign = crate::control::yaw_sign(&t, axis);
        crate::control::turn(&mut t, axis, yaw * sign, 0.0);
    }
}
