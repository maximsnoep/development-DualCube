//! Interactive mouse/keyboard controls for manual algorithm editing.

mod dispatcher;
mod loop_modification;
mod segmentation_modification;
mod shared;

use bevy::prelude::*;
use dispatcher::control_system;

pub use shared::{CacheResource, InteractiveMode};

/// Registers the interactive mouse/keyboard controls.
pub struct ControlsPlugin;

impl Plugin for ControlsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CacheResource>().add_systems(
            Update,
            control_system.run_if(not(bevy_egui::input::egui_wants_any_pointer_input)),
        );
    }
}
