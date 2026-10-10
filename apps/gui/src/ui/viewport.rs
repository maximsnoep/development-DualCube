//! The main view and the polycube window.
//!
//! The model (input mesh) is the main view: the main camera renders it into the area that the menus leave free. No
//! egui area covers that area, so the mouse reaches the camera controls and the interactive tools there. The polycube
//! is shown in a floating window, styled like the other windows: its dual view (the loops on the polycube), its
//! primal view (the labelled polycube), or the map (the mesh mapped onto the polycube).

use super::theme::{text, window_frame};
use super::widgets::{collapsed, sep, sleek_button, sleek_button_unfocused, window_title};
use crate::jobs::Job;
use crate::render::Objects;
use crate::render::camera::{CameraFor, CameraHandles};
use crate::render::store::RenderObjectSettingStore;
use crate::resources::{Configuration, SolutionResource};
use bevy::prelude::*;
use bevy_egui::egui::{self, Color32, LayerId, Pos2, Rect, Vec2};

/// What the polycube window shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolycubeMode {
    /// The loops on the polycube.
    Dual,
    /// The labelled polycube and its edges.
    Primal,
    /// The mesh mapped onto the polycube.
    Map,
}

impl PolycubeMode {
    const ALL: [Self; 3] = [Self::Dual, Self::Primal, Self::Map];

    fn name(self) -> &'static str {
        match self {
            Self::Dual => "dual",
            Self::Primal => "primal",
            Self::Map => "map",
        }
    }

    // The visible layers of the polycube in this mode (the map is a different object).
    fn layers(self) -> &'static [&'static str] {
        match self {
            Self::Dual => &[
                "black",
                "x-loops",
                "y-loops",
                "z-loops",
                "paths",
                "flat paths",
            ],
            Self::Primal | Self::Map => &["colored", "paths", "flat paths"],
        }
    }
}

/// The main view's area, and the state of the polycube window.
#[derive(Resource)]
pub struct ViewResource {
    /// The area of the main view, in physical pixels (the main camera's viewport).
    pub main: Option<Rect>,
    pub polycube_mode: PolycubeMode,
}

impl Default for ViewResource {
    fn default() -> Self {
        Self {
            main: None,
            polycube_mode: PolycubeMode::Primal,
        }
    }
}

impl ViewResource {
    /// Whether the given object is currently shown (only shown objects are rendered).
    pub fn shows(&self, object: Objects) -> bool {
        match object {
            Objects::InputMesh => true,
            Objects::Polycube => self.polycube_mode != PolycubeMode::Map,
            Objects::PolycubeMap => self.polycube_mode == PolycubeMode::Map,
        }
    }
}

/// Records the main view's area (for the main camera), draws the axes gizmo in its corner, and shows the polycube
/// window.
pub fn show(
    mut egui_ctx: bevy_egui::EguiContexts<'_, '_>,
    handles: Res<'_, CameraHandles>,
    mut view: ResMut<'_, ViewResource>,
    mut render_settings: ResMut<'_, RenderObjectSettingStore>,
    axes_texture: Res<'_, bevy_axes_gizmo::AxesGizmoTexture>,
    mut conf: ResMut<'_, Configuration>,
    solution: Res<'_, SolutionResource>,
    mut jobs: MessageWriter<'_, Job>,
) -> Result<(), BevyError> {
    let axes = egui_ctx.add_image(bevy_egui::EguiTextureHandle::Strong(axes_texture.0.clone()));
    let object = if view.polycube_mode == PolycubeMode::Map {
        Objects::PolycubeMap
    } else {
        Objects::Polycube
    };
    let texture = handles
        .map
        .get(&CameraFor(object))
        .map(|handle| egui_ctx.add_image(bevy_egui::EguiTextureHandle::Strong(handle.clone())));
    let ctx = egui_ctx.ctx_mut()?;

    // The area left by the panels (drawn before this system) is the main view.
    let area = super::main_area(ctx);
    let scale = ctx.pixels_per_point();
    view.main = Some(Rect::from_min_max(
        (area.min.to_vec2() * scale).to_pos2(),
        (area.max.to_vec2() * scale).to_pos2(),
    ));

    // The axes gizmo, on the background layer (it does not take the mouse).
    let size = 200.;
    ctx.layer_painter(LayerId::background()).image(
        axes,
        Rect::from_min_size(
            Pos2::new(area.left(), area.bottom() - size),
            Vec2::splat(size),
        ),
        Rect::from_min_max(Pos2::ZERO, Pos2::new(1., 1.)),
        Color32::WHITE,
    );

    // The polycube window only shows up once there is a polycube.
    if solution.current_solution.polycube.is_none() {
        return Ok(());
    }
    let mut chosen = None;
    let mut unit = conf.unit;
    let is_collapsed = collapsed(ctx, "Polycube");
    egui::Window::new("polycube")
        .title_bar(false)
        .frame(window_frame())
        .default_size([340., 380.])
        .min_size(if is_collapsed {
            [200., 0.]
        } else {
            [200., 220.]
        })
        // At first at the top left of the main view (movable).
        .default_pos(area.left_top() + Vec2::new(15., 45.))
        .resizable(!is_collapsed)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                // Clicking the title collapses the window to this row.
                window_title(ui, "Polycube");
                let mode = |ui: &mut egui::Ui, name: &str, selected: bool| {
                    if selected {
                        sleek_button(ui, name)
                    } else {
                        sleek_button_unfocused(ui, name)
                    }
                };
                for option in PolycubeMode::ALL {
                    if mode(ui, option.name(), view.polycube_mode == option) {
                        chosen = Some(option);
                    }
                }
                // The edge lengths of the polycube: unit lengths, or the lengths of the layout (geometric).
                ui.checkbox(&mut unit, text("unit"));
            });
            if is_collapsed {
                return;
            }
            sep(ui);
            let Some(texture) = texture else {
                return;
            };
            let [w, h] = ui.available_size().max(Vec2::splat(50.)).into();
            // Crop the square render texture to the window's aspect ratio.
            let (min, max) = if w > h {
                let offset = (1.0 - h / w) / 2.0;
                (Pos2::new(0., offset), Pos2::new(1.0, 1.0 - offset))
            } else {
                let offset = (1.0 - w / h) / 2.0;
                (Pos2::new(offset, 0.), Pos2::new(1.0 - offset, 1.0))
            };
            ui.add(
                egui::Image::new(egui::load::SizedTexture::new(texture, [w, h]))
                    .uv(Rect::from_min_max(min, max)),
            );
        });
    if unit != conf.unit {
        conf.unit = unit;
        if solution.current_solution.polycube.is_some() {
            jobs.write(Job::compute_polycube(
                solution.current_solution.clone(),
                conf.clone(),
            ));
        }
    }
    if let Some(mode) = chosen {
        view.polycube_mode = mode;
        if mode != PolycubeMode::Map
            && let Some(setting) = render_settings.objects.get_mut(&Objects::Polycube)
        {
            for (label, feature) in &mut setting.settings {
                feature.visible = mode.layers().contains(&label.as_str());
            }
        }
    }
    Ok(())
}
