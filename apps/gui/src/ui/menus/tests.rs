use super::*;
use egui::{PointerButton, Pos2, pos2, vec2};

struct Harness {
    ctx: Context,
    button: Rect,
    item: Rect,
    other: Rect,
    item_clicks: usize,
    other_clicks: usize,
    combo: bool,
}

impl Harness {
    fn new() -> Self {
        Self::with_combo(false)
    }

    fn with_combo(combo: bool) -> Self {
        let mut harness = Self {
            ctx: Context::default(),
            button: Rect::NOTHING,
            item: Rect::NOTHING,
            other: Rect::NOTHING,
            item_clicks: 0,
            other_clicks: 0,
            combo,
        };
        harness.frame(Vec::new());
        harness.frame(Vec::new());
        harness.click(harness.button.center());
        harness.frame(Vec::new());
        assert!(Popup::is_any_open(&harness.ctx));
        harness
    }

    fn frame(&mut self, events: Vec<Event>) -> bool {
        let input = RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(800., 600.))),
            focused: true,
            events,
            ..Default::default()
        };
        let pass_through = dismiss_on_input(&self.ctx, &input);
        let mut output = self.ctx.run_ui(input, |ui| {
            egui::Panel::top(crate::ui::TOP_BAR).show(ui, |ui| {
                if self.combo {
                    let response = egui::ComboBox::from_id_salt("method")
                        .selected_text("Method")
                        .show_ui(ui, |ui| {
                            ui.label("Choice");
                        })
                        .response;
                    self.button = response.rect;
                    track(&response);
                    return;
                }
                let response = ui.button("Menu");
                self.button = response.rect;
                popup(&response).show(|ui| {
                    let item = ui.button("Inside action");
                    self.item = item.rect;
                    self.item_clicks += usize::from(item.clicked());
                });
            });
            egui::Area::new(Id::new("other control"))
                .fixed_pos(pos2(400., 150.))
                .show(ui.ctx(), |ui| {
                    let button = ui.button("Outside action");
                    self.other = button.rect;
                    self.other_clicks += usize::from(button.clicked());
                });
        });
        output.textures_delta.clear();
        pass_through
    }

    fn pointer(&mut self, pos: Pos2, button: PointerButton, pressed: bool) -> bool {
        self.frame(vec![
            Event::PointerMoved(pos),
            Event::PointerButton {
                pos,
                button,
                pressed,
                modifiers: Default::default(),
            },
        ])
    }

    fn click(&mut self, pos: Pos2) {
        self.pointer(pos, PointerButton::Primary, true);
        self.pointer(pos, PointerButton::Primary, false);
    }
}

#[test]
fn outside_press_closes_menu_and_reaches_viewport_for_each_button() {
    for button in [
        PointerButton::Primary,
        PointerButton::Secondary,
        PointerButton::Middle,
    ] {
        let mut harness = Harness::new();
        assert!(harness.pointer(pos2(700., 500.), button, true));
        assert!(!Popup::is_any_open(&harness.ctx));
        assert!(harness.ctx.input(|i| i.pointer.button_pressed(button)));
    }
}

#[test]
fn inside_action_keeps_menu_open() {
    let mut harness = Harness::new();
    harness.click(harness.item.center());
    assert_eq!(harness.item_clicks, 1);
    assert!(Popup::is_any_open(&harness.ctx));
}

#[test]
fn outside_control_receives_the_dismissing_click() {
    let mut harness = Harness::new();
    harness.click(harness.other.center());
    assert_eq!(harness.other_clicks, 1);
    assert!(!Popup::is_any_open(&harness.ctx));
}

#[test]
fn menu_button_can_toggle_closed_and_focus_loss_dismisses() {
    let mut harness = Harness::new();
    harness.click(harness.button.center());
    assert!(!Popup::is_any_open(&harness.ctx));
    harness.click(harness.button.center());
    assert!(Popup::is_any_open(&harness.ctx));
    dismiss_on_input(
        &harness.ctx,
        &RawInput {
            focused: false,
            ..Default::default()
        },
    );
    assert!(!Popup::is_any_open(&harness.ctx));
}

#[test]
fn initializer_dropdown_also_dismisses_on_press() {
    let mut harness = Harness::with_combo(true);
    assert!(harness.pointer(pos2(700., 500.), PointerButton::Secondary, true));
    assert!(!Popup::is_any_open(&harness.ctx));
}
