//! The egui user interface: the top menu bar, the footer, the main view, and the floating windows
//! (polycube, evolution).
//!
//! Each panel is its own system (in its own module), run in order via
//! [`UiPlugin`]; `theme` and `widgets` hold the shared styling and widgets.

mod evolution;
mod footer;
mod menu;
mod theme;
pub mod viewport;
mod widgets;

pub use menu::ModelView;

use bevy::prelude::*;
use bevy_egui::{EguiPrimaryContextPass, PrimaryEguiContext};

/// Registers the UI resources and the panel systems.
pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<viewport::ViewResource>()
            .init_resource::<ModelView>()
            .add_plugins(bevy_egui::EguiPlugin::default())
            .add_observer(setup)
            .add_systems(
                EguiPrimaryContextPass,
                (menu::show, footer::show, viewport::show, evolution::show).chain(),
            );
    }
}

/// The ids of the top bar and the footer (panels).
const TOP_BAR: &str = "panel";
const FOOTER: &str = "footer";

/// A `Ui` over the whole window (on the background layer), to show a panel in (see `egui::Panel::show`).
fn root_ui(ctx: &bevy_egui::egui::Context, id: &str) -> bevy_egui::egui::Ui {
    use bevy_egui::egui::{LayerId, Ui, UiBuilder};
    Ui::new(
        ctx.clone(),
        bevy_egui::egui::Id::new(id),
        UiBuilder::new()
            .layer_id(LayerId::background())
            .max_rect(ctx.content_rect()),
    )
}

/// The main view: the window without the top bar and the footer (as of their last frame).
fn main_area(ctx: &bevy_egui::egui::Context) -> bevy_egui::egui::Rect {
    use bevy_egui::egui::{Id, containers::panel::PanelState};
    let mut area = ctx.content_rect();
    if let Some(top) = PanelState::load(ctx, Id::new(TOP_BAR)) {
        area.min.y = area.min.y.max(top.outer_rect.bottom());
    }
    if let Some(bottom) = PanelState::load(ctx, Id::new(FOOTER)) {
        area.max.y = area.max.y.min(bottom.outer_rect.top());
    }
    area
}

/// Sets up egui once the primary context is created.
fn setup(
    _: On<'_, '_, Add<PrimaryEguiContext>>,
    mut ui: bevy_egui::EguiContexts<'_, '_>,
) -> Result<(), BevyError> {
    theme::setup(&mut ui)?;
    Ok(())
}
