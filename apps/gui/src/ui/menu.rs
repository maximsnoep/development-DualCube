//! The top panel: the main menu bar and the pipeline stage bar.

use super::theme::*;
use super::widgets::{label, log_slider, menu_button, radio, sep, sleek_button, slider, space};
use crate::colors;
use crate::controls::InteractiveMode;
use crate::jobs::{EvolutionLive, Job, JobState};
use crate::render::Objects;
use crate::render::objects::skeleton::SkeletonLayer;
use crate::render::store::{RenderObjectSetting, RenderObjectSettingStore};
use crate::resources::{
    Configuration, FigureFormat, InputResource, SkeletonResource, SolutionResource,
};
use crate::view_settings::{ApplyView, ViewSettings};
use bevy::prelude::*;
use bevy_egui::egui::{
    self, Checkbox, CollapsingHeader, CursorIcon, MenuBar, Panel, RectAlign, RichText, Ui,
};
use bevy_orbit_camera::automatic::AutomaticRotation;
use dualcube::prelude::*;

// Show exactly the given layers of the model (the main view). The polycube window has its own modes.
fn apply_render_preset(store: &mut RenderObjectSettingStore, layers: &[&str]) {
    if let Some(settings) = store.objects.get_mut(&Objects::InputMesh) {
        let labeled_combined = layers.contains(&SkeletonLayer::TranslucentPatches.label())
            && layers.contains(&SkeletonLayer::Skeleton.label())
            && settings
                .settings
                .contains_key(SkeletonLayer::Labeled.label());
        for (label, setting) in &mut settings.settings {
            setting.visible = if labeled_combined && label == SkeletonLayer::Skeleton.label() {
                false
            } else if labeled_combined && label == SkeletonLayer::Labeled.label() {
                true
            } else {
                layers.contains(&label.as_str())
            };
        }
    }
}

// The groups of the views, separated in the view list.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ViewGroup {
    Input,
    Skeleton,
    Output,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelViewKind {
    Input,
    Deflated,
    Raw,
    Skeleton,
    /// The skeleton with gray nodes.
    GraySkeleton,
    Patches,
    /// The patch boundaries (the skeleton's loops) on the plain input mesh.
    SkeletonLoops,
    Optimized,
    /// A minimal cut system of the optimized skeleton (a figure only; the pipeline does not use it).
    CutSystem,
    Combined,
    /// The optimized patches with the optimized skeleton.
    OptimizedCombined,
    /// The labeled optimized skeleton on the mesh.
    OptimizedSkeleton,
    /// The optimized skeleton with gray nodes.
    OptimizedGraySkeleton,
    /// The optimized patch boundaries on the plain input mesh.
    OptimizedSkeletonLoops,
    /// The tree labeling, before cycle repairs.
    Tree,
    Labeled,
    /// The labeled skeleton at its placed node positions.
    PlacedSkeleton,
    /// The node cubes on their positions, with the placed skeleton.
    NodeCubes,
    Cubes,
    /// The cubes with only the nodes colored.
    CubesPrimal,
    /// The cubes with only the edges colored.
    CubesDual,
    /// The labeled patch boundaries, before the template's other loops are drawn.
    Lifted,
    Dual,
    Primal,
    Quad,
}

impl ModelViewKind {
    const ALL: [Self; 25] = [
        Self::Input,
        Self::Deflated,
        Self::Raw,
        Self::Skeleton,
        Self::GraySkeleton,
        Self::Patches,
        Self::SkeletonLoops,
        Self::Optimized,
        Self::CutSystem,
        Self::Combined,
        Self::OptimizedCombined,
        Self::OptimizedSkeleton,
        Self::OptimizedGraySkeleton,
        Self::OptimizedSkeletonLoops,
        Self::Tree,
        Self::Labeled,
        Self::PlacedSkeleton,
        Self::NodeCubes,
        Self::Cubes,
        Self::CubesPrimal,
        Self::CubesDual,
        Self::Lifted,
        Self::Dual,
        Self::Primal,
        Self::Quad,
    ];

    // The group it is listed in (see `view_bar`).
    const fn group(self) -> ViewGroup {
        match self {
            Self::Input => ViewGroup::Input,
            Self::Dual | Self::Primal | Self::Quad => ViewGroup::Output,
            _ => ViewGroup::Skeleton,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Input => "Input",
            Self::Deflated => "Deflated",
            Self::Raw => "Raw",
            Self::Skeleton => "Skeleton",
            Self::GraySkeleton => "Skeleton gray",
            Self::Patches => "Patches",
            Self::SkeletonLoops => "Skeleton loops",
            Self::Optimized => "Optimized",
            Self::CutSystem => "Cut system",
            Self::Combined => "Combined",
            Self::OptimizedCombined => "Optimized combined",
            Self::OptimizedSkeleton => "Optimized skeleton",
            Self::OptimizedGraySkeleton => "Optimized skeleton gray",
            Self::OptimizedSkeletonLoops => "Optimized skeleton loops",
            Self::Tree => "Tree",
            Self::Labeled => "Labeled",
            Self::PlacedSkeleton => "Placed skeleton",
            Self::NodeCubes => "Node cubes",
            Self::Cubes => "Cubes",
            Self::CubesPrimal => "Cubes primal",
            Self::CubesDual => "Cubes dual",
            Self::Lifted => "Lifted",
            Self::Dual => "Dual",
            Self::Primal => "Primal",
            Self::Quad => "Quad",
        }
    }

    // Its figure (see `io::figure`).
    pub(crate) const fn figure(self) -> io::figure::Figure {
        use io::figure::Figure;
        match self {
            Self::Input => Figure::InputMesh,
            Self::Deflated => Figure::Deflated,
            Self::Raw => Figure::RawSkeleton,
            Self::Skeleton => Figure::Skeleton,
            Self::GraySkeleton => Figure::GraySkeleton,
            Self::Patches => Figure::Patches,
            Self::SkeletonLoops => Figure::SkeletonLoops,
            Self::Optimized => Figure::OptimizedPatches,
            Self::CutSystem => Figure::CutSystem,
            Self::Combined => Figure::Combined,
            Self::OptimizedCombined => Figure::OptimizedCombined,
            Self::OptimizedSkeleton => Figure::OptimizedSkeleton,
            Self::OptimizedGraySkeleton => Figure::OptimizedGraySkeleton,
            Self::OptimizedSkeletonLoops => Figure::OptimizedSkeletonLoops,
            Self::Tree => Figure::TreeLabeled,
            Self::Labeled => Figure::Labeled,
            Self::PlacedSkeleton => Figure::PlacedSkeleton,
            Self::NodeCubes => Figure::NodeCubes,
            Self::Cubes => Figure::Cubes,
            Self::CubesPrimal => Figure::CubesPrimal,
            Self::CubesDual => Figure::CubesDual,
            Self::Lifted => Figure::LiftedLoops,
            Self::Dual => Figure::LoopStructure,
            Self::Primal => Figure::Segmentation,
            Self::Quad => Figure::Quad,
        }
    }

    fn layers(self) -> &'static [&'static str] {
        match self {
            Self::Input => &["lambert", "wireframe"],
            Self::Deflated => const { &[SkeletonLayer::Deflated.label()] },
            Self::Raw => const { &[SkeletonLayer::Raw.label(), "wireframe"] },
            Self::Skeleton => const { &[SkeletonLayer::Skeleton.label(), "wireframe"] },
            Self::GraySkeleton => const { &[SkeletonLayer::GraySkeleton.label(), "wireframe"] },
            Self::Patches => {
                const {
                    &[
                        SkeletonLayer::Patches.label(),
                        SkeletonLayer::Boundaries.label(),
                    ]
                }
            }
            Self::SkeletonLoops => const { &["lambert", SkeletonLayer::Boundaries.label()] },
            Self::CutSystem => const { &["lambert", SkeletonLayer::CutSystem.label()] },
            Self::Optimized => {
                const {
                    &[
                        SkeletonLayer::OptimizedPatches.label(),
                        SkeletonLayer::OptimizedBoundaries.label(),
                    ]
                }
            }
            Self::Combined => {
                const {
                    &[
                        "wireframe",
                        SkeletonLayer::TranslucentPatches.label(),
                        SkeletonLayer::Skeleton.label(),
                    ]
                }
            }
            Self::OptimizedCombined => {
                const {
                    &[
                        "wireframe",
                        SkeletonLayer::TranslucentOptimizedPatches.label(),
                        SkeletonLayer::OptimizedSkeleton.label(),
                    ]
                }
            }
            Self::PlacedSkeleton => const { &[SkeletonLayer::PlacedSkeleton.label()] },
            Self::NodeCubes => {
                const {
                    &[
                        SkeletonLayer::NodeCubes.label(),
                        SkeletonLayer::NodeCubeEdges.label(),
                        SkeletonLayer::PlacedSkeleton.label(),
                    ]
                }
            }
            Self::Cubes => {
                const {
                    &[
                        SkeletonLayer::Cubes.label(),
                        SkeletonLayer::CubeEdges.label(),
                    ]
                }
            }
            Self::CubesPrimal => {
                const {
                    &[
                        SkeletonLayer::CubesPrimal.label(),
                        SkeletonLayer::CubeEdges.label(),
                    ]
                }
            }
            Self::CubesDual => {
                const {
                    &[
                        SkeletonLayer::CubesDual.label(),
                        SkeletonLayer::CubeEdges.label(),
                    ]
                }
            }
            Self::Tree => const { &[SkeletonLayer::TreeLabeled.label(), "wireframe"] },
            Self::Labeled => const { &[SkeletonLayer::Labeled.label(), "wireframe"] },
            Self::OptimizedSkeleton => {
                const { &[SkeletonLayer::OptimizedSkeleton.label(), "wireframe"] }
            }
            Self::OptimizedGraySkeleton => {
                const { &[SkeletonLayer::OptimizedGraySkeleton.label(), "wireframe"] }
            }
            Self::OptimizedSkeletonLoops => {
                const { &["lambert", SkeletonLayer::OptimizedBoundaries.label()] }
            }
            Self::Lifted => const { &["black", SkeletonLayer::LiftedLoops.label()] },
            Self::Dual => &[
                "black",
                "paths",
                "flat paths",
                "x-loops",
                "y-loops",
                "z-loops",
            ],
            Self::Primal => &["segmentation", "paths", "flat paths"],
            Self::Quad => &[
                "quad colored",
                "quad wireframe",
                "quad paths",
                "quad flat paths",
            ],
        }
    }
}

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
                w.distortion,
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
    skeleton: Res<'_, SkeletonResource>,
    main_camera: Query<
        '_,
        '_,
        (&bevy_orbit_camera::LookTransform, &Projection),
        With<bevy_orbit_camera::Controller>,
    >,
    mut apply_view: MessageWriter<'_, ApplyView>,
    mut figure_preview: ResMut<'_, super::figure_preview::FigurePreview>,
) -> Result<(), BevyError> {
    // The figures are seen as the model is (the polycube window shares its rotation).
    let view = main_camera
        .single()
        .ok()
        .map(|(look, projection)| ViewSettings::current(look, projection, &conf));
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
                menu_bar(
                    ui,
                    &mut jobs,
                    &mut conf,
                    &solution,
                    &input,
                    &skeleton,
                    view,
                    view_preview.active,
                    &mut figure_preview.open,
                );
                ui.separator();
                pipeline_bar(
                    ui,
                    &mut jobs,
                    &mut conf,
                    &solution,
                    &mut evolution_live,
                    &mut view_preview,
                    &mut render_setting_store,
                    &skeleton,
                );
                ui.separator();
                // The view that the running optimization works on (see `ModelView`).
                let followed = evolution_live
                    .monitor
                    .as_ref()
                    .filter(|_| evolution_live.running())
                    .map(|monitor| {
                        if monitor.layout_active() {
                            ModelViewKind::Primal
                        } else {
                            ModelViewKind::Dual
                        }
                    });
                // The views of what has been computed (or is being computed).
                let current = &solution.current_solution;
                let initializing = job_state.request == Some("initializing loops");
                let evolving = evolution_live.running();
                let available = |view| match view {
                    ModelViewKind::Dual => !current.loops.is_empty() || initializing || evolving,
                    ModelViewKind::Primal => current.layout.is_some() || evolving,
                    ModelViewKind::Quad => current.quad.is_some(),
                    ModelViewKind::Skeleton
                    | ModelViewKind::GraySkeleton
                    | ModelViewKind::Raw
                    | ModelViewKind::Combined
                    | ModelViewKind::Patches
                    | ModelViewKind::SkeletonLoops => skeleton.extraction().is_some(),
                    ModelViewKind::Deflated => skeleton.contracted_mesh().is_some(),
                    ModelViewKind::PlacedSkeleton
                    | ModelViewKind::NodeCubes
                    | ModelViewKind::Cubes
                    | ModelViewKind::CubesPrimal
                    | ModelViewKind::CubesDual => skeleton.progress.realized.is_some(),
                    ModelViewKind::Optimized
                    | ModelViewKind::CutSystem
                    | ModelViewKind::OptimizedCombined
                    | ModelViewKind::OptimizedSkeleton
                    | ModelViewKind::OptimizedGraySkeleton
                    | ModelViewKind::OptimizedSkeletonLoops => {
                        skeleton.progress.optimized.is_some()
                    }
                    ModelViewKind::Labeled => {
                        skeleton.progress.labeled.is_some() || skeleton.progress.labeling.is_some()
                    }
                    ModelViewKind::Tree => skeleton
                        .progress
                        .labeling
                        .as_ref()
                        .is_some_and(|a| a.tree_skeleton.is_some()),
                    ModelViewKind::Lifted => skeleton
                        .progress
                        .drawing
                        .as_ref()
                        .is_some_and(|d| !d.lifted.is_empty()),
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
                    view,
                    &mut apply_view,
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
    view: Option<ViewSettings>,
    apply_view: &mut MessageWriter<'_, ApplyView>,
) {
    let response = ui.button(RichText::new("View").color(TEXT_COLOR).size(TEXT_SIZE));
    super::menus::popup(&response)
        .align(RectAlign::BOTTOM_END)
        .show(|ui| {
            ui.horizontal_top(|ui| {
                ui.vertical(|ui| {
                    ui.set_width(200.);
                    // The density of the quad mesh (recomputed when changed).
                    let response = ui
                        .add(egui::Slider::new(&mut conf.omega, 1..=20).text(text("quad density")));
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
                        egui::Slider::new(&mut conf.loop_width, 0.001..=0.03)
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
                    lock_checkbox(ui, conf, automatic_rotation);
                    ui.add_enabled(
                        !conf.camera_locked,
                        Checkbox::new(&mut automatic_rotation.enabled, text("automatic rotation")),
                    );
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
                                .add_enabled(
                                    !conf.camera_locked,
                                    egui::Button::new(
                                        RichText::new(name).color(color).size(TEXT_SIZE),
                                    ),
                                )
                                .clicked()
                            {
                                conf.camera_up = axis_vec;
                            }
                        }
                    });
                    if let Some(view) = view {
                        space(ui);
                        view_fields(ui, view, apply_view);
                    }
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
    input: &InputResource,
    skeleton: &SkeletonResource,
    view: Option<ViewSettings>,
    selected: Option<ModelViewKind>,
    preview: &mut bool,
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
        // With the skeleton stages, so a save restores every view.
        if sleek_button(ui, "Save as") {
            if let Some(path) = rfd::FileDialog::new().save_file() {
                jobs.write(Job::save(
                    solution.current_solution.clone(),
                    skeleton.save(),
                    input.source.clone(),
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
        // Figures for papers (see `io::figure`): one per view preset and the polycube, as seen; with a save and a text
        // to make them again. The camera lock is in the View menu.
        label(ui, "Figures", TEXT_COLOR2);
        ui.horizontal(|ui| {
            for format in FigureFormat::ALL {
                ui.radio_value(&mut conf.figure_format, format, text(format.extension()));
            }
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut conf.figure_label, text("label"));
            // Framed as the camera sees the model, instead of fitted to it.
            ui.checkbox(&mut conf.figure_zoom, text("camera zoom"));
        });
        // A window with the figure of the chosen view, rendered only while it is open.
        ui.checkbox(preview, text("preview"));
        // The figure of the view chosen in the top bar only.
        if let Some(selected) = selected
            && sleek_button(ui, "Export view")
            && let Some(view) = view
        {
            let figure = selected.figure();
            let name = if input.name.is_empty() {
                "model"
            } else {
                &input.name
            };
            if let Some(path) = rfd::FileDialog::new()
                .set_file_name(format!(
                    "{name}_{}.{}",
                    figure.slug(),
                    view.format.extension()
                ))
                .save_file()
            {
                jobs.write(Job::export_figure(
                    solution.current_solution.clone(),
                    skeleton.save(),
                    figure,
                    path,
                    name.to_owned(),
                    view,
                ));
            }
        }
        if sleek_button(ui, "Export figures")
            && let Some(view) = view
            && let Some(dir) = rfd::FileDialog::new().pick_folder()
        {
            jobs.write(Job::export_figures(
                solution.current_solution.clone(),
                conf.clone(),
                dir,
                input.name.clone(),
                view,
                skeleton.save(),
                skeleton.output.clone(),
                input.source.clone(),
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
        sep(ui);
        // Delete dissolves the hovered patch boundary (see `controls::skeleton_modification`).
        let mut skeleton = conf.interactive_mode == InteractiveMode::SkeletonModification;
        if ui
            .checkbox(&mut skeleton, text("modify skeleton"))
            .changed()
        {
            conf.interactive_mode = if skeleton {
                InteractiveMode::SkeletonModification
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
    hovered: Option<ModelViewKind>,
    active: Option<ModelViewKind>,
    // The view that the running optimization asked for last.
    followed: Option<ModelViewKind>,
    pending: Option<ModelViewKind>,
}

impl ModelView {
    /// The view chosen in the top bar.
    pub fn active(&self) -> Option<ModelViewKind> {
        self.active
    }

    pub fn show_when_ready(&mut self, view: ModelViewKind) {
        self.pending = Some(view);
    }

    pub fn show(&mut self, view: ModelViewKind, store: &mut RenderObjectSettingStore) {
        apply_render_preset(store, view.layers());
        self.active = Some(view);
        self.saved = None;
    }

    pub fn show_input(&mut self, store: &mut RenderObjectSettingStore) {
        *self = Self::default();
        self.show(ModelViewKind::Input, store);
    }
}

// The views of the model (see `ModelViewKind`) in a list, grouped: shown while hovered, kept when clicked. Returns the
// list's button.
fn view_bar(
    ui: &mut Ui,
    render_setting_store: &mut RenderObjectSettingStore,
    preview: &mut ModelView,
    followed: Option<ModelViewKind>,
    available: impl Fn(ModelViewKind) -> bool,
) -> egui::Response {
    // Follow the optimization (not while a view is hovered; then once the mouse leaves it).
    if followed != preview.followed && preview.hovered.is_none() {
        preview.followed = followed;
        if let Some(view) = followed {
            apply_render_preset(render_setting_store, view.layers());
            preview.active = Some(view);
        }
    }
    let selected = preview.active.map_or_else(
        || "View".to_owned(),
        |view| format!("View: {}", view.label()),
    );
    let mut hovered = None;
    let mut clicked = None;
    let combo = egui::ComboBox::from_id_salt("model view")
        .selected_text(RichText::new(selected).color(TEXT_COLOR).size(TEXT_SIZE))
        .width(120.)
        // All views, without scrolling.
        .height(f32::INFINITY)
        .show_ui(ui, |ui| {
            let mut group = None;
            for name in ModelViewKind::ALL {
                if group.is_some_and(|g| g != name.group()) {
                    ui.separator();
                }
                group = Some(name.group());
                let text = RichText::new(name.label()).size(TEXT_SIZE);
                // Not computed yet: shown dimmed, and does nothing.
                if !available(name) {
                    ui.add_enabled(
                        false,
                        egui::Button::selectable(false, text.color(DIM_COLOR)),
                    );
                    continue;
                }
                let active = preview.active == Some(name);
                let color = if active { TEXT_COLOR } else { TEXT_COLOR2 };
                let response = ui.add(egui::Button::selectable(active, text.color(color)));
                if response.clicked() {
                    clicked = Some(name);
                } else if response.hovered() {
                    hovered = Some(name);
                }
            }
        });
    super::menus::track(&combo.response);
    if let Some(name) = clicked {
        apply_render_preset(render_setting_store, name.layers());
        preview.active = Some(name);
        preview.saved = None;
        preview.hovered = Some(name);
        return combo.response;
    }
    if hovered != preview.hovered {
        // Restore the layers from before the previous preview, then show the new one.
        if let Some(saved) = preview.saved.take()
            && let Some(settings) = render_setting_store.objects.get_mut(&Objects::InputMesh)
        {
            *settings = saved;
        }
        if let Some(name) = hovered {
            preview.saved = render_setting_store
                .objects
                .get(&Objects::InputMesh)
                .cloned();
            apply_render_preset(render_setting_store, name.layers());
        }
        preview.hovered = hovered;
    }
    combo.response
}

/// The bar: initialize (new loops), evolve (opens the optimization window), and postprocess (smooth the paths).
fn pipeline_bar(
    ui: &mut Ui,
    jobs: &mut MessageWriter<'_, Job>,
    conf: &mut Configuration,
    solution: &SolutionResource,
    evolution_live: &mut EvolutionLive,
    view: &mut ModelView,
    render_setting_store: &mut RenderObjectSettingStore,
    skeleton: &SkeletonResource,
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

    use crate::resources::{Initialization, genus};
    let method = egui::ComboBox::from_id_salt("initialization method")
        .selected_text(match conf.initialization {
            Initialization::Minimal => "Minimal (genus 0)",
            Initialization::Skeleton => "Skeleton",
        })
        .width(85.)
        .show_ui(ui, |ui| {
            ui.add_enabled_ui(genus(&current.mesh_ref) == Some(0), |ui| {
                ui.selectable_value(
                    &mut conf.initialization,
                    Initialization::Minimal,
                    "Minimal (genus 0)",
                );
            })
            .response
            .on_disabled_hover_text("Minimal initialization requires genus zero");
            ui.selectable_value(
                &mut conf.initialization,
                Initialization::Skeleton,
                "Skeleton",
            );
        });
    super::menus::track(&method.response);
    if conf.initialization == Initialization::Skeleton {
        menu_button(ui, "Options", |ui| {
            label(ui, "Skeleton", TEXT_COLOR);
            ui.checkbox(
                &mut conf.skeleton_options.refine_embedding,
                "embedding refinement",
            );
            sep(ui);
            label(ui, "Contraction", TEXT_COLOR);
            ui.add(
                egui::Slider::new(&mut conf.skeleton_options.contraction_step, 1.1..=8.0)
                    .logarithmic(true)
                    .text(text("step")),
            );
            ui.add(
                egui::Slider::new(&mut conf.skeleton_options.max_iterations, 1..=500)
                    .logarithmic(true)
                    .text(text("max iterations")),
            );
            sep(ui);
            label(ui, "Connectivity surgery", TEXT_COLOR);
            let options = &mut conf.skeleton_options.surgery;
            log_slider(ui, "shape weight", &mut options.shape_weight, 100.);
            log_slider(ui, "sampling weight", &mut options.sampling_weight, 100.);
            log_slider(ui, "handle bonus", &mut options.handle_bonus, 1.0e9);
            if sleek_button(ui, "defaults") {
                conf.skeleton_options = Default::default();
            }
            space(ui);
            label(ui, "Settings apply on Initialize.", TEXT_COLOR2);
        });
    }

    // Initialization options
    if button(ui, "Initialize", current.mesh_ref.nr_verts() > 0) {
        evolution_live.reset();
        if conf.initialization == Initialization::Minimal {
            view.show(ModelViewKind::Dual, render_setting_store);
        }
        jobs.write(Job::initialize_loops(
            current.clone(),
            conf.clone(),
            skeleton.contraction_cache.clone(),
        ));
    }
    // The optimization (its settings, start, and progress) is in its window.
    if button(ui, "Evolve", current.mesh_ref.nr_verts() > 0) {
        evolution_live.open = true;
    }
    // Back to what the skeleton initialization produced, to evolve from it anew.
    if button(ui, "Restore output", skeleton.output.is_some())
        && let Some(output) = &skeleton.output
    {
        jobs.write(Job::restore_output(output.clone(), current, conf.clone()));
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

pub fn apply_pending_view(
    mut view: ResMut<'_, ModelView>,
    mut settings: ResMut<'_, RenderObjectSettingStore>,
) {
    if let Some(name) = view.pending.take() {
        view.show(name, &mut settings);
    }
}

/// The camera as numbers (eye, target, up), and a view line (see `ViewSettings`) to copy or paste. Explicit, so they
/// also work while the camera is locked.
fn view_fields(ui: &mut Ui, view: ViewSettings, apply_view: &mut MessageWriter<'_, ApplyView>) {
    let mut edited = view;
    egui::Grid::new("camera position")
        .num_columns(4)
        .show(ui, |ui| {
            for (name, vector) in [
                ("eye", &mut edited.eye),
                ("target", &mut edited.target),
                ("up", &mut edited.up),
            ] {
                label(ui, name, TEXT_COLOR2);
                for component in [&mut vector.x, &mut vector.y, &mut vector.z] {
                    ui.add(egui::DragValue::new(component).speed(0.05).max_decimals(4));
                }
                ui.end_row();
            }
        });
    if edited != view
        && (edited.target - edited.eye).length_squared() > 0.
        && edited.up != Vec3::ZERO
    {
        apply_view.write(ApplyView(edited));
    }

    let id = ui.make_persistent_id("pasted view");
    let (mut pasted, mut error) =
        ui.data_mut(|d| d.get_temp::<(String, String)>(id).unwrap_or_default());
    ui.horizontal(|ui| {
        if sleek_button(ui, "copy view") {
            ui.ctx().copy_text(view.line());
        }
        if sleek_button(ui, "apply pasted") {
            match ViewSettings::parse(&pasted, view) {
                Ok(parsed) => {
                    apply_view.write(ApplyView(parsed));
                    error.clear();
                }
                Err(message) => {
                    warn!("Cannot read the pasted view: {message}");
                    error = message;
                }
            }
        }
    });
    ui.add(
        egui::TextEdit::multiline(&mut pasted)
            .desired_rows(2)
            .desired_width(f32::INFINITY)
            .hint_text("paste a view line or a _view.txt"),
    );
    if !error.is_empty() {
        ui.colored_label(egui::Color32::from_rgb(191, 97, 106), &error);
    }
    ui.data_mut(|d| d.insert_temp(id, (pasted, error)));
}

/// Locks the camera so it cannot move between stills; locking also stops automatic rotation.
fn lock_checkbox(
    ui: &mut Ui,
    conf: &mut Configuration,
    automatic_rotation: &mut AutomaticRotation,
) {
    if ui
        .checkbox(&mut conf.camera_locked, text("lock camera"))
        .changed()
        && conf.camera_locked
    {
        automatic_rotation.enabled = false;
    }
}

#[cfg(test)]
mod tests;
