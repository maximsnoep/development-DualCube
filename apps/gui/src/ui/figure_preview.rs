//! The figure preview window: the figure of the chosen view as it would be exported (see `Job::export_figure`),
//! rendered in the background, and only while the window is open.

use super::theme::{TEXT_COLOR2, WARN_RED, window_frame};
use super::widgets::{collapsed, label, window_title};
use crate::resources::{Configuration, InputResource, SkeletonResource, SolutionResource};
use crate::ui::ModelView;
use crate::view_settings::ViewSettings;
use bevy::prelude::*;
use bevy::tasks::futures_lite::future;
use bevy::tasks::{AsyncComputeTaskPool, Task};
use bevy_egui::egui::{self, Color32, Vec2};
use io::figure::Figure;

// Pixels per point of the preview (the figures are 300 points wide).
const PREVIEW_SCALE: f32 = 2.5;

#[derive(Resource, Default)]
pub struct FigurePreview {
    /// Whether the window is shown (and the figure rendered).
    pub open: bool,
    // What the last started rendering was made from; `None` makes the next frame render.
    made_from: Option<Request>,
    // The solution or the skeleton changed since then.
    stale: bool,
    rendering: Option<Task<Rendered>>,
    texture: Option<egui::TextureHandle>,
    // The figure shown, or why there is none.
    shown: Option<Figure>,
    error: Option<String>,
}

#[derive(Clone, Copy, PartialEq)]
struct Request {
    figure: Figure,
    view: ViewSettings,
}

enum Rendered {
    Image([usize; 2], Vec<u8>),
    Nothing(Figure),
    Failed(String),
}

pub fn show(
    mut egui_ctx: bevy_egui::EguiContexts<'_, '_>,
    mut preview: ResMut<'_, FigurePreview>,
    model_view: Res<'_, ModelView>,
    conf: Res<'_, Configuration>,
    solution: Res<'_, SolutionResource>,
    skeleton: Res<'_, SkeletonResource>,
    input: Res<'_, InputResource>,
    main_camera: Query<
        '_,
        '_,
        (&bevy_orbit_camera::LookTransform, &Projection),
        With<bevy_orbit_camera::Controller>,
    >,
) -> Result<(), BevyError> {
    let preview = preview.as_mut();
    if !preview.open {
        // Rendered afresh when opened again.
        preview.made_from = None;
        preview.rendering = None;
        return Ok(());
    }
    let ctx = egui_ctx.ctx_mut()?;
    if solution.is_changed() || skeleton.is_changed() {
        preview.stale = true;
    }

    // A finished rendering replaces the image.
    if let Some(task) = &mut preview.rendering
        && let Some(rendered) = future::block_on(future::poll_once(task))
    {
        preview.rendering = None;
        preview.error = None;
        match rendered {
            Rendered::Image(size, pixels) => {
                let image = egui::ColorImage::from_rgba_premultiplied(size, &pixels);
                let options = egui::TextureOptions::LINEAR;
                match &mut preview.texture {
                    Some(texture) => texture.set(image, options),
                    None => {
                        preview.texture = Some(ctx.load_texture("figure preview", image, options))
                    }
                }
                preview.shown = preview.made_from.map(|request| request.figure);
            }
            Rendered::Nothing(figure) => {
                preview.texture = None;
                preview.error = Some(format!("No {} figure yet", figure.name()));
            }
            Rendered::Failed(error) => preview.error = Some(error),
        }
    }

    // One rendering at a time, of what is current once it is done.
    let request = main_camera.single().ok().and_then(|(look, projection)| {
        Some(Request {
            figure: model_view.active()?.figure(),
            view: ViewSettings::current(look, projection, &conf),
        })
    });
    if let Some(request) = request
        && preview.rendering.is_none()
        && (preview.made_from != Some(request) || preview.stale)
    {
        preview.made_from = Some(request);
        preview.stale = false;
        let solution = solution.current_solution.clone();
        let skeleton = skeleton.save();
        let name = if input.name.is_empty() {
            "model".to_owned()
        } else {
            input.name.clone()
        };
        preview.rendering = Some(AsyncComputeTaskPool::get().spawn(async move {
            let Some(svg) = crate::jobs::figure_as_seen(
                &solution,
                skeleton.as_ref(),
                request.figure,
                &name,
                &request.view,
            ) else {
                return Rendered::Nothing(request.figure);
            };
            match io::figure::rasterize(&svg, PREVIEW_SCALE) {
                Ok(([width, height], pixels)) => {
                    Rendered::Image([width as usize, height as usize], pixels)
                }
                Err(err) => Rendered::Failed(format!("{err:#}")),
            }
        }));
    }

    let is_collapsed = collapsed(ctx, "Figure preview");
    let mut open = true;
    egui::Window::new("figure preview")
        .title_bar(false)
        .frame(window_frame())
        .default_width(420.)
        .min_width(240.)
        // At first at the bottom right of the main view (movable).
        .pivot(egui::Align2::RIGHT_BOTTOM)
        .default_pos(super::main_area(ctx).right_bottom() + Vec2::new(-15., -15.))
        .resizable(!is_collapsed)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                // Clicking the title collapses the window to this row.
                window_title(ui, "Figure preview");
                let status = match (&request, &preview.rendering, preview.shown) {
                    (None, _, _) => "choose a view in the top bar".to_owned(),
                    (_, Some(_), _) => "rendering...".to_owned(),
                    (_, None, Some(figure)) => figure.name().to_owned(),
                    (_, None, None) => String::new(),
                };
                label(ui, &status, TEXT_COLOR2);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("x").on_hover_text("Close").clicked() {
                        open = false;
                    }
                });
            });
            if is_collapsed {
                return;
            }
            if let Some(error) = &preview.error {
                label(ui, error, WARN_RED);
            }
            if let Some(texture) = &preview.texture {
                // On white, as on a page; as wide as the window.
                let [width, height] = texture.size().map(|s| s as f32);
                let shown = ui.available_width();
                egui::Frame::new().fill(Color32::WHITE).show(ui, |ui| {
                    ui.image((texture.id(), Vec2::new(shown, shown * height / width)));
                });
            }
        });
    if !open {
        preview.open = false;
    }
    Ok(())
}
