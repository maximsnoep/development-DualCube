mod colors;
mod controls;
mod jobs;
mod render;
mod resources;
mod ui;

use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, SystemInformationDiagnosticsPlugin};
use bevy::log::{Level, LogPlugin};
use bevy::prelude::*;
use bevy::ui::UiScale;
use bevy::window::{WindowMode, WindowResolution};
use dualcube::prelude::*;
use resources::{Configuration, InputResource, SolutionResource};

fn main() {
    let axis_color = |direction| {
        colors::to_bevy(colors::from_direction(
            direction,
            Some(Perspective::Primal),
            None,
        ))
    };

    App::new()
        // Application state
        .init_resource::<Configuration>()
        .init_resource::<InputResource>()
        .init_resource::<SolutionResource>()
        .insert_resource(UiScale(1.0)) // no UI scaling
        .insert_resource(GlobalAmbientLight {
            #[allow(unused_qualifications)]
            color: bevy::color::Color::WHITE,
            brightness: 1.0,
            ..Default::default()
        })
        // Load default plugins
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "DualCube".to_string(),
                        mode: WindowMode::Windowed,
                        resolution: WindowResolution::default().with_scale_factor_override(1.),
                        ..Default::default()
                    }),
                    ..Default::default()
                })
                .set(LogPlugin {
                    level: Level::TRACE,
                    // Quiet in release builds (`RUST_LOG` overrides this).
                    filter: if cfg!(debug_assertions) {
                        "info,dualcube=debug"
                    } else {
                        "warn"
                    }
                    .to_string(),
                    ..default()
                }),
        )
        // Plugin for diagnostics
        .add_plugins((
            FrameTimeDiagnosticsPlugin::default(),
            SystemInformationDiagnosticsPlugin,
        ))
        // My cool plugins c:
        .add_plugins((
            bevy_orbit_camera::OrbitCameraPlugin,
            bevy_toon::ToonPlugin,
            bevy_axes_gizmo::AxesGizmoPlugin {
                colors: [
                    axis_color(Direction::X),
                    axis_color(Direction::Y),
                    axis_color(Direction::Z),
                ],
                width: 3.,
                ..default()
            },
            bevy_wicon::WindowIconPlugin::with_path("apps/gui/assets/logo-32.png"),
        ))
        // The application itself: jobs, rendering, UI, and controls.
        .add_plugins((
            ui::UiPlugin,
            jobs::JobPlugin,
            controls::ControlsPlugin,
            render::RenderPlugin,
        ))
        .add_systems(Startup, maximize)
        .run();
}

// Start maximized (a regular window with its borders, filling the screen).
fn maximize(mut window: Single<'_, '_, &mut Window>) {
    window.set_maximized(true);
}
