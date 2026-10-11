//! Dismiss menus on press, before the viewport handles that same input.

use bevy::prelude::*;
use bevy_egui::{EguiContext, EguiInput, egui};
use egui::{Context, Event, Id, Order, Popup, PopupCloseBehavior, RawInput, Rect, Response};

#[derive(Clone)]
struct OpenMenu {
    id: Id,
    button: Rect,
}

fn menu_id() -> Id {
    Id::new("open top menu")
}

pub fn track(response: &Response) {
    let id = Popup::default_response_id(response);
    if response.clicked() || Popup::is_id_open(&response.ctx, id) {
        response.ctx.data_mut(|data| {
            data.insert_temp(
                menu_id(),
                OpenMenu {
                    id,
                    button: response.rect,
                },
            );
        });
    }
}

pub fn popup(response: &Response) -> Popup<'_> {
    track(response);
    Popup::menu(response).close_behavior(PopupCloseBehavior::CloseOnClickOutside)
}

#[derive(Resource, Default)]
pub struct MenuInput {
    pub viewport_press: bool,
}

pub fn wants_pointer_input(
    egui: Res<'_, bevy_egui::input::EguiWantsInput>,
    menus: Res<'_, MenuInput>,
) -> bool {
    egui.wants_any_pointer_input() && !menus.viewport_press
}

pub fn dismiss(
    mut contexts: Query<'_, '_, (&mut EguiContext, &EguiInput)>,
    mut menus: ResMut<'_, MenuInput>,
) {
    menus.viewport_press = false;
    for (mut context, input) in &mut contexts {
        menus.viewport_press |= dismiss_on_input(context.get_mut(), &input.0);
    }
}

fn dismiss_on_input(ctx: &Context, input: &RawInput) -> bool {
    let Some(menu) = ctx.data(|data| data.get_temp::<OpenMenu>(menu_id())) else {
        return false;
    };
    if !Popup::is_id_open(ctx, menu.id) {
        return false;
    }
    if !input.focused {
        Popup::close_all(ctx);
        return false;
    }
    for event in &input.events {
        if let Event::PointerButton {
            pos, pressed: true, ..
        } = event
        {
            let inside = menu.button.contains(*pos)
                || ctx.memory(|memory| {
                    memory
                        .area_rect(menu.id)
                        .is_some_and(|rect| rect.contains(*pos))
                });
            if !inside {
                Popup::close_all(ctx);
                // Bevy's egui input flags still describe the previous frame.
                return super::main_area(ctx).contains(*pos)
                    && ctx
                        .layer_id_at(*pos)
                        .is_none_or(|layer| layer.order == Order::Background);
            }
        }
    }
    false
}

#[cfg(test)]
mod tests;
