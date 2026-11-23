use super::*;
use crate::render::store::{RenderFeatureSetting, RenderObjectSetting};
use egui::{Context, Event, PointerButton, Popup, Pos2, RawInput, Rect, pos2, vec2};

// The view list in a top bar, with every layer of the views in the store.
struct Harness {
    ctx: Context,
    store: RenderObjectSettingStore,
    preview: ModelView,
    button: Rect,
    popup: Option<Rect>,
}

const UNAVAILABLE: ModelViewKind = ModelViewKind::Quad;

impl Harness {
    fn new() -> Self {
        let mut setting = RenderObjectSetting::default();
        for label in ModelViewKind::ALL.iter().flat_map(|view| view.layers()) {
            setting.labels.push((*label).to_owned());
            setting.settings.insert(
                (*label).to_owned(),
                RenderFeatureSetting {
                    label: (*label).to_owned(),
                    visible: false,
                },
            );
        }
        let mut store = RenderObjectSettingStore::default();
        store.objects.insert(Objects::InputMesh, setting);
        let mut harness = Self {
            ctx: Context::default(),
            store,
            preview: ModelView::default(),
            button: Rect::NOTHING,
            popup: None,
        };
        harness
            .preview
            .show(ModelViewKind::Input, &mut harness.store);
        harness.frame(vec![]);
        harness.frame(vec![]);
        harness
    }

    fn frame(&mut self, events: Vec<Event>) {
        let input = RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(800., 800.))),
            focused: true,
            events,
            ..Default::default()
        };
        let mut output = self.ctx.run_ui(input, |ui| {
            Panel::top(crate::ui::TOP_BAR).show(ui, |ui| {
                let response = view_bar(ui, &mut self.store, &mut self.preview, None, |view| {
                    view != UNAVAILABLE
                });
                self.button = response.rect;
                let id = Popup::default_response_id(&response);
                self.popup = Popup::is_id_open(ui.ctx(), id)
                    .then(|| ui.ctx().memory(|m| m.area_rect(id)))
                    .flatten();
            });
        });
        output.textures_delta.clear();
    }

    fn hover(&mut self, pos: Pos2) {
        self.frame(vec![Event::PointerMoved(pos)]);
        self.frame(vec![Event::PointerMoved(pos)]);
    }

    fn click(&mut self, pos: Pos2) {
        for pressed in [true, false] {
            self.frame(vec![
                Event::PointerMoved(pos),
                Event::PointerButton {
                    pos,
                    button: PointerButton::Primary,
                    pressed,
                    modifiers: Default::default(),
                },
            ]);
        }
        self.frame(vec![]);
    }

    fn visible(&self) -> Vec<String> {
        visible(&self.store)
    }

    // The layers a view shows (as its preset applies them to these layers).
    fn shown(&self, view: ModelViewKind) -> Vec<String> {
        let mut store = RenderObjectSettingStore::default();
        store.objects = self.store.objects.clone();
        apply_render_preset(&mut store, view.layers());
        visible(&store)
    }
}

fn visible(store: &RenderObjectSettingStore) -> Vec<String> {
    let mut labels = store.objects[&Objects::InputMesh]
        .settings
        .values()
        .filter(|s| s.visible)
        .map(|s| s.label.clone())
        .collect::<Vec<_>>();
    labels.sort_unstable();
    labels
}

#[test]
fn view_list_previews_on_hover_and_keeps_on_click() {
    let mut harness = Harness::new();
    let input = harness.visible();
    assert_eq!(input, harness.shown(ModelViewKind::Input));
    harness.click(harness.button.center());
    let popup = harness.popup.expect("the view list opens");

    // Down the list: every available view is previewed, in order, and nothing else.
    let x = popup.center().x;
    let mut seen = vec![];
    let mut y = popup.top();
    while y < popup.bottom() {
        harness.hover(pos2(x, y));
        if let Some(view) = harness.preview.hovered
            && seen.last() != Some(&view)
        {
            assert_eq!(harness.visible(), harness.shown(view));
            seen.push(view);
        }
        y += 3.;
    }
    let expected = ModelViewKind::ALL
        .into_iter()
        .filter(|&view| view != UNAVAILABLE)
        .collect::<Vec<_>>();
    assert_eq!(seen, expected);

    // Leaving the list restores the view.
    harness.hover(pos2(700., 700.));
    assert_eq!(harness.preview.hovered, None);
    assert_eq!(harness.visible(), input);

    // A click keeps the view, also after the mouse leaves.
    let mut y = popup.top();
    while harness.preview.hovered != Some(ModelViewKind::CubesPrimal) {
        y += 3.;
        assert!(y < popup.bottom());
        harness.hover(pos2(x, y));
    }
    harness.click(pos2(x, y));
    assert_eq!(harness.preview.active, Some(ModelViewKind::CubesPrimal));
    harness.hover(pos2(700., 700.));
    assert!(harness.popup.is_none());
    assert_eq!(harness.visible(), harness.shown(ModelViewKind::CubesPrimal));
}
