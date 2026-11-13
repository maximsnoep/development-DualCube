//! The contraction window: a slider over the stages of the mesh contraction (before the first
//! iteration up to the result), shown in the deflated layer. Only shown while that layer is
//! visible.

use super::theme::{TEXT_COLOR2, text, window_frame};
use super::widgets::{collapsed, label, window_title};
use crate::render::Objects;
use crate::render::objects::skeleton::SkeletonLayer;
use crate::render::store::RenderObjectSettingStore;
use crate::resources::SkeletonResource;
use bevy::prelude::*;
use bevy_egui::egui::{self, Vec2};

pub fn show(
    mut egui_ctx: bevy_egui::EguiContexts<'_, '_>,
    mut skeleton: ResMut<'_, SkeletonResource>,
    render_settings: Res<'_, RenderObjectSettingStore>,
) -> Result<(), BevyError> {
    let visible = render_settings
        .objects
        .get(&Objects::InputMesh)
        .and_then(|object| object.settings.get(SkeletonLayer::Deflated.label()))
        .is_some_and(|setting| setting.visible);
    if !visible || skeleton.contracted_mesh().is_none() {
        return Ok(());
    }
    let Some(stages) = skeleton.contraction_cache.stages() else {
        return Ok(());
    };
    let last = stages.len().saturating_sub(1);
    let mut stage = skeleton.contraction_stage.unwrap_or(last).min(last);

    let ctx = egui_ctx.ctx_mut()?;
    let is_collapsed = collapsed(ctx, "Contraction");
    egui::Window::new("contraction")
        .title_bar(false)
        .frame(window_frame())
        .default_width(320.)
        // At first at the bottom left of the main view (movable).
        .pivot(egui::Align2::LEFT_BOTTOM)
        .default_pos(super::main_area(ctx).left_bottom() + Vec2::new(15., -15.))
        .resizable(false)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                // Clicking the title collapses the window to this row.
                window_title(ui, "Contraction");
                let status = match stage {
                    0 => "input".to_owned(),
                    s if s == last => format!("result ({last} iterations)"),
                    s => format!("after iteration {s}"),
                };
                label(ui, &status, TEXT_COLOR2);
            });
            if is_collapsed {
                return;
            }
            ui.add(egui::Slider::new(&mut stage, 0..=last).text(text("iteration")));
        });

    // Only write on change, so the deflated layer is not rebuilt every frame.
    let chosen = (stage != last).then_some(stage);
    if skeleton.contraction_stage != chosen {
        skeleton.contraction_stage = chosen;
    }
    Ok(())
}
