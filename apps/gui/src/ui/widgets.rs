//! Small reusable egui widgets.

use super::theme::{TEXT_COLOR, TEXT_COLOR2, TEXT_SIZE, WARN_RED, colored_text, text};
use bevy::prelude::Time;
use bevy_egui::egui::{Color32, RichText, Slider, Ui, emath};
use std::ops::RangeInclusive;

pub fn sep(ui: &mut Ui) {
    ui.add_space(5.);
    ui.separator();
    ui.add_space(5.);
}

pub fn space(ui: &mut Ui) {
    ui.add_space(5.);
}

pub fn label(ui: &mut Ui, label: &str, color: Color32) {
    ui.label(colored_text(label, color));
}

pub fn slider<T: emath::Numeric>(
    ui: &mut Ui,
    label: &str,
    value: &mut T,
    range: RangeInclusive<T>,
) {
    ui.add(Slider::new(value, range).text(text(label)));
}

pub fn log_slider<T: emath::Numeric>(ui: &mut Ui, label: &str, value: &mut T, max: T) {
    ui.add(
        Slider::new(value, RangeInclusive::new(T::from_f64(0.), max))
            .logarithmic(true)
            .text(text(label)),
    );
}

pub fn radio<T: PartialEq<T> + std::fmt::Display>(
    ui: &mut Ui,
    item: &mut T,
    value: T,
    color: Color32,
) -> bool {
    if ui
        .radio(*item == value, colored_text(&format!("{value}"), color))
        .clicked()
    {
        *item = value;
        true
    } else {
        false
    }
}

pub fn menu_button(ui: &mut Ui, label: &str, f: impl FnOnce(&mut Ui)) {
    ui.menu_button(RichText::new(label).color(TEXT_COLOR).size(TEXT_SIZE), f);
}

pub fn sleek_button(ui: &mut Ui, label: &str) -> bool {
    click_button(ui, label, TEXT_COLOR)
}

pub fn sleek_button_warn(ui: &mut Ui, label: &str) -> bool {
    click_button(ui, label, WARN_RED)
}

pub fn sleek_button_unfocused(ui: &mut Ui, label: &str) -> bool {
    click_button(ui, label, TEXT_COLOR2)
}

fn click_button(ui: &mut Ui, label: &str, color: Color32) -> bool {
    ui.button(RichText::new(label).color(color).size(TEXT_SIZE))
        .clicked()
}

/// A small looping animation indicating that a job is running.
pub fn timer_animation(time: &Time) -> String {
    let frequency = 6.0;
    let animation = ["●○○○", "○●○○", "○○●○", "○○○●", "○○●○", "○●○○"];
    let index = (time.elapsed_secs() * frequency) as usize % animation.len();
    animation[index].to_string()
}

/// Whether the window with the given title is collapsed (see `window_title`).
pub fn collapsed(ctx: &bevy_egui::egui::Context, title: &str) -> bool {
    ctx.data(|d| d.get_temp(collapse_id(title)).unwrap_or(false))
}

/// The title of a window: clicking it collapses (or expands) the window to its title row (see `collapsed`).
pub fn window_title(ui: &mut Ui, title: &str) {
    let collapsed = collapsed(ui.ctx(), title);
    let response = ui
        .add(
            bevy_egui::egui::Label::new(colored_text(title, TEXT_COLOR))
                .sense(bevy_egui::egui::Sense::click()),
        )
        .on_hover_cursor(bevy_egui::egui::CursorIcon::PointingHand);
    if response.clicked() {
        ui.ctx()
            .data_mut(|d| d.insert_temp(collapse_id(title), !collapsed));
    }
}

fn collapse_id(title: &str) -> bevy_egui::egui::Id {
    bevy_egui::egui::Id::new(("collapsed window", title))
}

/// A label in bold: the font has no bold face, so the text is drawn twice, half a pixel apart.
pub fn bold_label(ui: &mut Ui, text: &str, color: Color32) {
    let galley = ui.painter().layout_job(colored_text(text, color));
    let (rect, _) = ui.allocate_exact_size(
        galley.size() + bevy_egui::egui::vec2(1., 0.),
        bevy_egui::egui::Sense::hover(),
    );
    let painter = ui.painter();
    painter.galley(rect.min, galley.clone(), color);
    painter.galley(rect.min + bevy_egui::egui::vec2(0.6, 0.), galley, color);
}
