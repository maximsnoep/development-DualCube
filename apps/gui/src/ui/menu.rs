//! The top panel: the main menu bar and the pipeline stage bar.

use super::theme::*;
use super::widgets::{label, log_slider, menu_button, radio, sep, sleek_button, slider, space};
use crate::colors;
use crate::controls::InteractiveMode;
use crate::jobs::{EvolutionLive, Job, JobState};
use crate::render::Objects;
use crate::render::store::{RenderObjectSetting, RenderObjectSettingStore};
use crate::resources::{Configuration, FigureFormat, InputResource, SolutionResource};
use bevy::prelude::*;
use bevy_egui::egui::{
    CollapsingHeader, CursorIcon, MenuBar, Panel, Popup, PopupCloseBehavior, RectAlign, RichText,
    Ui,
};
use bevy_orbit_camera::automatic::AutomaticRotation;
use dualcube::prelude::*;

// Show exactly the given layers of the model (the main view). The polycube window has its own modes.
fn apply_render_preset(store: &mut RenderObjectSettingStore, layers: &[&str]) {
    if let Some(settings) = store.objects.get_mut(&Objects::InputMesh) {
        for (label, setting) in &mut settings.settings {
            setting.visible = layers.contains(&label.as_str());
        }
    }
}

/// Views of the model (in the top bar): the layers of the model that each shows.
const MODEL_VIEWS: [(&str, &[&str]); 4] = [
    ("Input", &["lambert", "wireframe"]),
    (
        "Dual",
        &[
            "black",
            "paths",
            "flat paths",
            "x-loops",
            "y-loops",
            "z-loops",
        ],
    ),
    ("Primal", &["segmentation", "paths", "flat paths"]),
    (
        "Quad",
        &[
            "quad colored",
            "quad wireframe",
            "quad paths",
            "quad flat paths",
        ],
    ),
];

/// The score of the current layout, recomputed only when the layout or the quality criterion changes (computing it
/// takes too long to do every frame).
#[derive(Default)]
pub(crate) struct ScoreCache {
    key: Option<(usize, usize, u64, [u64; 8])>,
    score: Option<f64>,
}

impl ScoreCache {
    pub(crate) fn get(&mut self, solution: &Solution, quality: &QualityParams) -> Option<f64> {
        let layout = solution.layout.as_ref()?;
        let w = quality.weights;
        let key = (
            solution.loops.len(),
            layout.granulated_mesh.nr_faces(),
            layout.alignment.unwrap_or(f64::NAN).to_bits(),
            [
                w.fidelity,
                w.path_regularity,
                w.coherence,
                w.rectangularity,
                w.corner_regularity,
                w.non_degeneracy,
                w.correspondence,
                w.complexity,
            ]
            .map(f64::to_bits),
        );
        if self.key != Some(key) {
            self.key = Some(key);
            self.score = QualityReport::compute(
                solution.loops.len(),
                layout,
                quality,
                QualityTerms::needed(&w),
            )
            .score(&w);
        }
        self.score
    }
}

/// Shows the top panel: the menu bar and the pipeline stage bar.
pub fn show(
    mut egui_ctx: bevy_egui::EguiContexts<'_, '_>,
    mut jobs: MessageWriter<'_, Job>,
    mut conf: ResMut<'_, Configuration>,
    job_state: Res<'_, JobState>,
    solution: Res<'_, SolutionResource>,
    mut render_setting_store: ResMut<'_, RenderObjectSettingStore>,
    mut automatic_rotation: ResMut<'_, AutomaticRotation>,
    mut evolution_live: ResMut<'_, EvolutionLive>,
    mut view_preview: ResMut<'_, ModelView>,
    input: Res<'_, InputResource>,
    main_camera: Query<'_, '_, &Transform, With<bevy_orbit_camera::Controller>>,
) -> Result<(), BevyError> {
    // The figures are seen as the model is (the polycube window shares its rotation).
    let camera = main_camera.single().ok().map(|transform| {
        let (forward, up) = (transform.forward(), transform.up());
        (
            Vector3D::new(
                f64::from(forward.x),
                f64::from(forward.y),
                f64::from(forward.z),
            ),
            Vector3D::new(f64::from(up.x), f64::from(up.y), f64::from(up.z)),
        )
    });
    let mut root = super::root_ui(egui_ctx.ctx_mut()?, "top bar root");
    Panel::top(super::TOP_BAR)
        .show_separator_line(true)
        .show(&mut root, |ui| {
            if job_state.request.is_some() {
                ui.output_mut(|o| o.cursor_icon = CursorIcon::Progress);
            }
            ui.add_space(9.);
            MenuBar::new().ui(ui, |ui| {
                // More room between the buttons.
                ui.spacing_mut().item_spacing.x += 6.;
                ui.spacing_mut().button_padding.y += 2.;
                ui.add_space(8.);
                menu_bar(ui, &mut jobs, &mut conf, &solution, &input.name, camera);
                ui.separator();
                pipeline_bar(
                    ui,
                    &mut jobs,
                    &conf,
                    &solution,
                    &mut evolution_live,
                    &mut view_preview,
                    &mut render_setting_store,
                );
                ui.separator();
                // The view that the running optimization works on (see `ModelView`).
                let followed = evolution_live
                    .monitor
                    .as_ref()
                    .filter(|_| evolution_live.running())
                    .map(|monitor| {
                        if monitor.layout_active() {
                            "Primal"
                        } else {
                            "Dual"
                        }
                    });
                // The views of what has been computed (or is being computed).
                let current = &solution.current_solution;
                let initializing = job_state.request == Some("initializing loops");
                let evolving = evolution_live.running();
                let available = |name: &str| match name {
                    "Dual" => !current.loops.is_empty() || initializing || evolving,
                    "Primal" => current.layout.is_some() || evolving,
                    "Quad" => current.quad.is_some(),
                    _ => true,
                };
                view_bar(
                    ui,
                    &mut render_setting_store,
                    &mut view_preview,
                    followed,
                    available,
                );
                // The View menu at the right (the remaining width, minus the width of the button).
                ui.add_space((ui.available_width() - 45.).max(8.));
                view_menu(
                    ui,
                    &mut jobs,
                    &mut conf,
                    &solution,
                    &mut automatic_rotation,
                    &mut render_setting_store,
                );
            });
            ui.add_space(6.);
        });

    Ok(())
}

/// The View menu (top right, opening to the left): the layers of every view (left), and the camera (right).
fn view_menu(
    ui: &mut Ui,
    jobs: &mut MessageWriter<'_, Job>,
    conf: &mut Configuration,
    solution: &SolutionResource,
    automatic_rotation: &mut AutomaticRotation,
    render_setting_store: &mut RenderObjectSettingStore,
) {
    let response = ui.button(RichText::new("View").color(TEXT_COLOR).size(TEXT_SIZE));
    Popup::menu(&response)
        .align(RectAlign::BOTTOM_END)
        .close_behavior(PopupCloseBehavior::CloseOnClickOutside)
        .show(|ui| {
            ui.horizontal_top(|ui| {
                ui.vertical(|ui| {
                    ui.set_width(200.);
                    // The density of the quad mesh (recomputed when changed).
                    let response = ui.add(
                        bevy_egui::egui::Slider::new(&mut conf.omega, 1..=20)
                            .text(text("quad density")),
                    );
                    if (response.drag_stopped() || (response.changed() && !response.dragged()))
                        && solution.current_solution.polycube.is_some()
                    {
                        jobs.write(Job::compute_quad(
                            solution.current_solution.clone(),
                            conf.clone(),
                        ));
                    }
                    // The width of the loops on the model (redrawn when changed).
                    let response = ui.add(
                        bevy_egui::egui::Slider::new(&mut conf.loop_width, 0.001..=0.03)
                            .logarithmic(true)
                            .text(text("loop width")),
                    );
                    if response.drag_stopped() || (response.changed() && !response.dragged()) {
                        jobs.write(Job::refresh(
                            solution.current_solution.clone(),
                            conf.clone(),
                        ));
                    }
                    space(ui);
                    label(ui, "Layers", TEXT_COLOR);
                    for object in Objects::ALL {
                        let Some(setting) = render_setting_store.objects.get_mut(&object) else {
                            continue;
                        };
                        CollapsingHeader::new(text(&object.to_string()))
                            .default_open(object == Objects::InputMesh)
                            .show(ui, |ui| {
                                for name in setting.labels.clone() {
                                    if let Some(feature) = setting.settings.get_mut(&name) {
                                        ui.checkbox(&mut feature.visible, text(&name));
                                    }
                                }
                            });
                    }
                });
                ui.separator();
                ui.vertical(|ui| {
                    ui.set_width(220.);
                    label(ui, "Camera", TEXT_COLOR);
                    ui.checkbox(&mut automatic_rotation.enabled, text("automatic rotation"));
                    slider(
                        ui,
                        "speed",
                        &mut automatic_rotation.sensitivity,
                        -std::f32::consts::PI..=std::f32::consts::PI,
                    );
                    space(ui);
                    label(ui, "mouse sensitivity", TEXT_COLOR2);
                    log_slider(ui, "rotate", &mut conf.camera_rotate_sensitivity, 1.);
                    log_slider(ui, "translate", &mut conf.camera_translate_sensitivity, 3.);
                    log_slider(ui, "zoom", &mut conf.camera_zoom_sensitivity, 1.);
                    ui.horizontal(|ui| {
                        if sleek_button(ui, "precise") {
                            conf.camera_rotate_sensitivity = 0.01;
                            conf.camera_translate_sensitivity = 0.01;
                            conf.camera_zoom_sensitivity = 0.01;
                        }
                        if sleek_button(ui, "default") {
                            conf.camera_rotate_sensitivity = 0.2;
                            conf.camera_translate_sensitivity = 2.0;
                            conf.camera_zoom_sensitivity = 0.2;
                        }
                    });
                    space(ui);
                    label(ui, "up axis", TEXT_COLOR2);
                    ui.horizontal(|ui| {
                        for &(axis_vec, name, dir) in &[
                            (Vec3::X, "+X", Direction::X),
                            (Vec3::NEG_X, "-X", Direction::X),
                            (Vec3::Y, "+Y", Direction::Y),
                            (Vec3::NEG_Y, "-Y", Direction::Y),
                            (Vec3::Z, "+Z", Direction::Z),
                            (Vec3::NEG_Z, "-Z", Direction::Z),
                        ] {
                            let color = to_color32(colors::from_direction(dir, None, None));
                            let color = if conf.camera_up == axis_vec {
                                color
                            } else {
                                TEXT_COLOR2
                            };
                            if ui
                                .button(RichText::new(name).color(color).size(TEXT_SIZE))
                                .clicked()
                            {
                                conf.camera_up = axis_vec;
                            }
                        }
                    });
                });
            });
        });
}

/// The File / Manual menus.
fn menu_bar(
    ui: &mut Ui,
    jobs: &mut MessageWriter<'_, Job>,
    conf: &mut Configuration,
    solution: &SolutionResource,
    name: &str,
    camera: Option<(Vector3D, Vector3D)>,
) {
    menu_button(ui, "File", |ui| {
        if sleek_button(ui, "Load") {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter(
                    "mesh (obj, stl) OR dualcube save (dc, loops)",
                    &["obj", "stl", "dc", "loops"],
                )
                .pick_file()
            {
                jobs.write(Job::import(path, conf.clone()));
            }
        }
        sep(ui);
        if sleek_button(ui, "Save as") {
            if let Some(path) = rfd::FileDialog::new().save_file() {
                jobs.write(Job::export(
                    solution.current_solution.clone(),
                    path.with_extension("dc"),
                ));
            }
        }
        #[cfg(feature = "nlr")]
        if sleek_button(ui, "Export NLR") {
            if let Some(path) = rfd::FileDialog::new().save_file() {
                jobs.write(Job::export(
                    solution.current_solution.clone(),
                    path.with_extension("nlr"),
                ));
            }
        }
        if sleek_button(ui, "Export graph") {
            if let Some(path) = rfd::FileDialog::new().save_file() {
                jobs.write(Job::export(
                    solution.current_solution.clone(),
                    path.with_extension("apg"),
                ));
            }
        }
        sep(ui);
        // Figures for papers (see `io::figure`): the model, its loops, its segmentation and its polycube, as seen.
        label(ui, "Figures", TEXT_COLOR2);
        ui.horizontal(|ui| {
            for format in FigureFormat::ALL {
                ui.radio_value(&mut conf.figure_format, format, text(format.extension()));
            }
        });
        ui.checkbox(&mut conf.figure_label, text("label"));
        if sleek_button(ui, "Export figures")
            && let Some((view, up)) = camera
            && let Some(dir) = rfd::FileDialog::new().pick_folder()
        {
            jobs.write(Job::export_figures(
                solution.current_solution.clone(),
                conf.clone(),
                dir,
                name.to_owned(),
                view,
                up,
            ));
        }
        sep(ui);
        if sleek_button(ui, "Quit") {
            std::process::exit(0);
        }
    });

    menu_button(ui, "Manual", |ui| {
        let mut loops = conf.interactive_mode == InteractiveMode::LoopModification;
        if ui.checkbox(&mut loops, text("modify loops")).changed() {
            conf.interactive_mode = if loops {
                InteractiveMode::LoopModification
            } else {
                InteractiveMode::None
            };
        }
        ui.horizontal(|ui| {
            for direction in DIRECTIONS {
                let color = to_color32(colors::from_direction(
                    direction,
                    Some(Perspective::Dual),
                    None,
                ));
                radio(ui, &mut conf.direction, direction, color);
            }
        });
        sep(ui);
        let mut segmentation = conf.interactive_mode == InteractiveMode::SegmentationModification;
        if ui
            .checkbox(&mut segmentation, text("modify segmentation"))
            .changed()
        {
            conf.interactive_mode = if segmentation {
                InteractiveMode::SegmentationModification
            } else {
                InteractiveMode::None
            };
        }
    });
}

/// The view of the model chosen in the top bar (see `view_bar`): hovering a view shows it, clicking keeps it. While an
/// optimization runs, the view follows what it works on (the dual while it evolves the loops, the primal while it
/// evolves the layout); a view chosen meanwhile is kept until it switches between the two.
#[derive(Default, Resource)]
pub struct ModelView {
    // The layers of the model before the hovered view was shown (restored when the mouse leaves it).
    saved: Option<RenderObjectSetting>,
    hovered: Option<&'static str>,
    active: Option<&'static str>,
    // The view that the running optimization asked for last.
    followed: Option<&'static str>,
}

impl ModelView {
    /// Show the view with the given name (see `MODEL_VIEWS`).
    pub fn show(&mut self, name: &'static str, store: &mut RenderObjectSettingStore) {
        if let Some(&(name, layers)) = MODEL_VIEWS.iter().find(|(n, _)| *n == name) {
            apply_render_preset(store, layers);
            self.active = Some(name);
            self.saved = None;
        }
    }

    /// Show the input (the first view), e.g., for a new model.
    pub fn show_input(&mut self, store: &mut RenderObjectSettingStore) {
        let (name, layers) = MODEL_VIEWS[0];
        apply_render_preset(store, layers);
        *self = Self {
            active: Some(name),
            ..Self::default()
        };
    }
}

// The views of the model (see `MODEL_VIEWS`): shown while hovered, kept when clicked.
fn view_bar(
    ui: &mut Ui,
    render_setting_store: &mut RenderObjectSettingStore,
    preview: &mut ModelView,
    followed: Option<&'static str>,
    available: impl Fn(&str) -> bool,
) {
    // Follow the optimization (not while a view is hovered; then once the mouse leaves it).
    if followed != preview.followed && preview.hovered.is_none() {
        preview.followed = followed;
        if let Some((name, layers)) =
            followed.and_then(|followed| MODEL_VIEWS.iter().find(|(name, _)| *name == followed))
        {
            apply_render_preset(render_setting_store, layers);
            preview.active = Some(name);
        }
    }
    let mut hovered = None;
    for (name, layers) in MODEL_VIEWS {
        // Not computed yet: shown dimmed, and does nothing.
        if !available(name) {
            ui.add_enabled(
                false,
                bevy_egui::egui::Button::new(RichText::new(name).color(DIM_COLOR).size(TEXT_SIZE)),
            );
            continue;
        }
        let color = if preview.active == Some(name) {
            TEXT_COLOR
        } else {
            TEXT_COLOR2
        };
        let response = ui.button(RichText::new(name).color(color).size(TEXT_SIZE));
        if response.clicked() {
            apply_render_preset(render_setting_store, layers);
            preview.active = Some(name);
            preview.saved = None;
            preview.hovered = Some(name);
            return;
        }
        if response.hovered() {
            hovered = Some((name, layers));
        }
    }
    if hovered.map(|(name, _)| name) != preview.hovered {
        // Restore the layers from before the previous preview, then show the new one.
        if let Some(saved) = preview.saved.take()
            && let Some(settings) = render_setting_store.objects.get_mut(&Objects::InputMesh)
        {
            *settings = saved;
        }
        if let Some((_, layers)) = hovered {
            preview.saved = render_setting_store
                .objects
                .get(&Objects::InputMesh)
                .cloned();
            apply_render_preset(render_setting_store, layers);
        }
        preview.hovered = hovered.map(|(name, _)| name);
    }
}

/// The bar: initialize (new loops), evolve (opens the optimization window), and postprocess (smooth the paths).
fn pipeline_bar(
    ui: &mut Ui,
    jobs: &mut MessageWriter<'_, Job>,
    conf: &Configuration,
    solution: &SolutionResource,
    evolution_live: &mut EvolutionLive,
    view: &mut ModelView,
    render_setting_store: &mut RenderObjectSettingStore,
) {
    let current = &solution.current_solution;
    let button = |ui: &mut Ui, name: &str, enabled: bool| {
        if enabled {
            sleek_button(ui, name)
        } else {
            label(ui, name, DIM_COLOR);
            false
        }
    };

    // A new solution from scratch (and new histories of the optimizations), shown in the dual view.
    if button(ui, "Initialize", current.mesh_ref.nr_verts() > 0) {
        evolution_live.reset();
        view.show("Dual", render_setting_store);
        jobs.write(Job::initialize_loops(current.clone(), conf.clone()));
    }
    // The optimization (its settings, start, and progress) is in its window.
    if button(ui, "Evolve", current.mesh_ref.nr_verts() > 0) {
        evolution_live.open = true;
    }
    // Visual post-processing (smoothing the paths), only on request.
    if button(ui, "Postprocess", current.layout.is_some()) {
        jobs.write(Job::smooth_layout(current.clone(), conf.clone()));
    }
    #[cfg(feature = "hex")]
    if button(ui, "Hex", current.polycube.is_some()) {
        jobs.write(Job::compute_hex(current.clone(), conf.clone()));
    }
}
