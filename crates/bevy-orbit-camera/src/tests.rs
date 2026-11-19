use crate::control::{turn, yaw_sign};
use crate::transform::LookTransform;
use bevy::prelude::*;
use std::f32::consts::{FRAC_PI_2, TAU};

fn start() -> LookTransform {
    LookTransform::new(Vec3::new(25., 25., 25.), Vec3::ZERO, Vec3::Y)
}

fn close(a: Vec3, b: Vec3) -> bool {
    a.distance(b) < 1e-3
}

#[test]
fn pitch_goes_over_the_pole_smoothly_and_all_the_way_round() {
    let mut look = start();
    let steps = 2000;
    let mut previous = Transform::from(look);
    for _ in 0..steps {
        turn(&mut look, Vec3::Y, 0., TAU / steps as f32);
        let next = Transform::from(look);
        // No step flips the camera (as `looking_at` with a fixed world up does at the pole).
        assert!(previous.rotation.angle_between(next.rotation) < 0.01);
        // Only rounding changes the distance.
        assert!((look.radius() / start().radius() - 1.).abs() < 1e-4);
        previous = next;
    }
    assert!(look.eye.distance(start().eye) < 1e-2, "{look:?}");
    assert!(close(look.screen_up(), start().screen_up()));
}

#[test]
fn looks_straight_down_the_orbit_axis() {
    let mut look = LookTransform::new(Vec3::new(0., 0., 10.), Vec3::ZERO, Vec3::Y);
    turn(&mut look, Vec3::Y, 0., -FRAC_PI_2);
    assert!(close(look.eye, Vec3::new(0., 10., 0.)), "{look:?}");
    // Up on screen is the way the camera was looking.
    assert!(close(look.screen_up(), Vec3::NEG_Z));
    // Spinning there turns the picture instead of locking.
    turn(&mut look, Vec3::Y, FRAC_PI_2, 0.);
    assert!(close(look.eye, Vec3::new(0., 10., 0.)));
    assert!(close(look.screen_up(), Vec3::NEG_X), "{look:?}");
}

#[test]
fn up_reads_as_the_orbit_axis_unless_upside_down_or_at_the_pole() {
    let mut look = start();
    turn(&mut look, Vec3::Y, 0.3, 0.2);
    assert_eq!(look.up, Vec3::Y);
    assert_eq!(yaw_sign(&look, Vec3::Y), 1.);
    // Over the top: the picture is upside down, and horizontal drags turn the other way.
    turn(&mut look, Vec3::Y, 0., 2.);
    assert_eq!(look.up, Vec3::NEG_Y);
    assert_eq!(yaw_sign(&look, Vec3::Y), -1.);
}

#[test]
fn horizontal_turns_keep_the_horizon() {
    let mut look = start();
    for _ in 0..500 {
        turn(&mut look, Vec3::Y, 0.05, 0.013);
        let forward = look.forward().unwrap();
        assert!(forward.cross(look.screen_up()).dot(Vec3::Y).abs() < 1e-4);
    }
}

#[test]
fn a_rolled_view_keeps_its_up() {
    let up = Vec3::new(0.3, 1., 0.).normalize();
    let mut look = LookTransform::new(Vec3::new(0., 0., 10.), Vec3::ZERO, up);
    turn(&mut look, Vec3::Y, 0., 0.);
    assert!(close(look.up, up));
}
