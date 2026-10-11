mod cli;
mod colors;
mod controls;
mod jobs;
mod render;
mod resources;
mod ui;

use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, SystemInformationDiagnosticsPlugin};
use bevy::log::{Level, LogPlugin};
use bevy::prelude::*;
use bevy::window::WindowMode;
use bevy::winit::{UpdateMode, WinitSettings};
use dualcube::prelude::*;
use resources::{InputResource, SolutionResource};

fn main() -> AppExit {
    let launch = cli::Launch::from_env_or_exit();
    let axis_color = |direction| {
        colors::to_bevy(colors::from_direction(
            direction,
            Some(Perspective::Primal),
            None,
        ))
    };

    App::new()
        // Application state
        .insert_resource(launch.configuration())
        .insert_resource(launch)
        .init_resource::<InputResource>()
        .init_resource::<SolutionResource>()
        .init_resource::<resources::SkeletonResource>()
        .insert_resource(WinitSettings {
            focused_mode: UpdateMode::Continuous,
            unfocused_mode: UpdateMode::Reactive {
                wait: std::time::Duration::from_millis(200),
                react_to_device_events: false,
                react_to_user_events: false,
                react_to_window_events: false,
            },
        })
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
                        ..Default::default()
                    }),
                    ..Default::default()
                })
                .set(LogPlugin {
                    custom_layer: ui::console::capture,
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
        .add_systems(Startup, (maximize, cli::start))
        .run()
}

// Start maximized (a regular window with its borders, filling the screen).
fn maximize(mut window: Single<'_, '_, &mut Window>) {
    window.set_maximized(true);
}
