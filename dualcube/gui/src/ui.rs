use crate::axes::AxesGizmoTexture;
use crate::controls::InteractiveMode;
use crate::jobs::{Job, JobRequest, JobState};
use crate::render::{CameraFor, Objects, RenderFlag, RenderObjectSetting, RenderObjectSettingStore, RenderObjectStore};
use crate::{colors, ActionEvent, CameraHandles, Configuration, InputResource, Perspective, Phase, PrincipalDirection, SolutionResource};
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin, SystemInformationDiagnosticsPlugin};
use bevy::{prelude::*, ui};
use bevy_egui::egui::*;
use egui_dock::{DockArea, DockState, NodeIndex, Style};
use std::collections::HashMap;

#[derive(Resource)]
pub struct UiResource {
    pub tree: DockState<Objects>,
}

impl Default for UiResource {
    fn default() -> Self {
        UiResource {
            tree: {
                let mut tree = DockState::new(vec![Objects::InputMesh]);

                // You can modify the tree before constructing the dock
                let right1 = tree.main_surface_mut().split_right(NodeIndex::root(), 0.8, vec![Objects::PolycubeMap])[1];
                let _right2 = tree.main_surface_mut().split_below(right1, 0.5, vec![Objects::QuadMesh])[1];

                tree
            },
        }
    }
}
struct TabViewer {
    egui_handles: Vec<bevy_egui::egui::TextureId>,
    render_settings: HashMap<Objects, RenderObjectSetting>,
}

impl egui_dock::TabViewer for TabViewer {
    type Tab = Objects;

    fn title(&mut self, tab: &mut Self::Tab) -> bevy_egui::egui::WidgetText {
        tab.to_string().into()
    }

    fn tab_style_override(&self, tab: &Self::Tab, _global_style: &egui_dock::TabStyle) -> Option<egui_dock::TabStyle> {
        let mut default_style = egui_dock::TabStyle::default();
        default_style.tab_body.stroke = Stroke::new(0., Color32::from_rgb(255, 0, 0));
        default_style.tab_body.inner_margin = Margin::same(0);

        match tab {
            Objects::InputMesh => {
                default_style.tab_body.bg_fill = Color32::TRANSPARENT;
            }
            Objects::PolycubeMap | Objects::QuadMesh => {
                default_style.tab_body.bg_fill = Color32::from_gray(27);
            }
        }

        default_style.active.bg_fill = Color32::from_gray(27);
        default_style.active.corner_radius = CornerRadius::same(0);
        default_style.active.outline_color = Color32::from_gray(27);

        default_style.active_with_kb_focus = default_style.active.clone();
        default_style.focused = default_style.active.clone();
        default_style.focused_with_kb_focus = default_style.active.clone();
        default_style.hovered = default_style.active.clone();
        default_style.inactive = default_style.active.clone();
        default_style.inactive_with_kb_focus = default_style.active.clone();

        default_style.active.text_color = Color32::from_gray(255);
        default_style.active_with_kb_focus.text_color = Color32::from_gray(255);

        default_style.inactive.text_color = Color32::from_gray(100);
        default_style.inactive_with_kb_focus.text_color = Color32::from_gray(100);

        default_style.hovered.text_color = Color32::from_gray(150);
        default_style.focused.text_color = Color32::from_gray(255);
        default_style.focused_with_kb_focus.text_color = Color32::from_gray(255);

        Some(default_style)
    }

    fn allowed_in_windows(&self, _tab: &mut Self::Tab) -> bool {
        false
    }

    fn context_menu(&mut self, ui: &mut Ui, tab: &mut Self::Tab, _surface: egui_dock::SurfaceIndex, _node: NodeIndex) {
        if let Some(local_copy) = self.render_settings.get_mut(tab) {
            for (label, setting) in local_copy.settings.iter_mut() {
                ui.checkbox(&mut setting.visible, label.to_owned());
            }
        } else {
            ui.label("o_O");
        }
    }

    fn ui(&mut self, ui: &mut bevy_egui::egui::Ui, tab: &mut Self::Tab) {
        bevy_egui::egui::Frame {
            stroke: bevy_egui::egui::epaint::Stroke {
                width: 5.0,
                color: Color32::from_gray(27),
            },
            ..default()
        }
        .show(ui, |ui| {
            bevy_egui::egui::Frame {
                stroke: bevy_egui::egui::epaint::Stroke {
                    width: 1.0,
                    color: Color32::from_gray(50),
                },
                ..default()
            }
            .show(ui, |ui| match tab {
                Objects::InputMesh => {
                    ui.allocate_exact_size(ui.available_size(), Sense::all());
                }

                Objects::PolycubeMap | Objects::QuadMesh => {
                    let egui_handle = match tab {
                        Objects::PolycubeMap => self.egui_handles[0],
                        Objects::QuadMesh => self.egui_handles[1],
                        _ => unreachable!(),
                    };
                    let [w, h] = ui.available_size().into();

                    if w > h {
                        let offset = (1.0 - h / w) / 2.0;
                        ui.add(
                            bevy_egui::egui::widgets::Image::new(bevy_egui::egui::load::SizedTexture::new(egui_handle, [w, h])).uv(
                                bevy_egui::egui::Rect::from_min_max(bevy_egui::egui::Pos2::new(0., offset), bevy_egui::egui::Pos2::new(1.0, 1.0 - offset)),
                            ),
                        );
                    } else {
                        let offset = (1.0 - w / h) / 2.0;
                        ui.add(
                            bevy_egui::egui::widgets::Image::new(bevy_egui::egui::load::SizedTexture::new(egui_handle, [w, h])).uv(
                                bevy_egui::egui::Rect::from_min_max(bevy_egui::egui::Pos2::new(offset, 0.), bevy_egui::egui::Pos2::new(1.0 - offset, 1.0)),
                            ),
                        );
                    }
                }
            });
        });
    }

    fn closeable(&mut self, _tab: &mut Self::Tab) -> bool {
        false
    }

    fn scroll_bars(&self, _tab: &Self::Tab) -> [bool; 2] {
        [false, false]
    }

    fn id(&mut self, tab: &mut Self::Tab) -> Id {
        Id::new(self.title(tab).text())
    }

    fn on_tab_button(&mut self, _tab: &mut Self::Tab, _response: &bevy_egui::egui::Response) {}

    fn on_close(&mut self, _tab: &mut Self::Tab) -> bool {
        true
    }

    fn on_add(&mut self, _surface: egui_dock::SurfaceIndex, _node: NodeIndex) {}

    fn add_popup(&mut self, _ui: &mut Ui, _surface: egui_dock::SurfaceIndex, _node: NodeIndex) {}

    fn force_close(&mut self, _tab: &mut Self::Tab) -> bool {
        false
    }

    fn clear_background(&self, _tab: &Self::Tab) -> bool {
        true
    }
}

pub fn setup(mut ui: bevy_egui::EguiContexts) {
    // Font
    let mut fonts = bevy_egui::egui::FontDefinitions::default();
    fonts.font_data.insert(
        "font".to_owned(),
        bevy_egui::egui::FontData::from_static(include_bytes!("../assets/font.ttf")).into(),
    );
    fonts
        .families
        .entry(bevy_egui::egui::FontFamily::Proportional)
        .or_default()
        .insert(0, "font".to_owned());
    fonts
        .families
        .entry(bevy_egui::egui::FontFamily::Monospace)
        .or_default()
        .insert(0, "font".to_owned());
    ui.ctx_mut().set_fonts(fonts);

    // Theme
    ui.ctx_mut().style_mut(|style| {
        let zero = CornerRadius::same(0);
        let mut visuals = bevy_egui::egui::Visuals::dark();
        visuals.widgets.open.corner_radius = zero;
        visuals.menu_corner_radius = zero;
        visuals.window_corner_radius = zero;
        visuals.widgets.noninteractive.corner_radius = zero;
        visuals.widgets.hovered.corner_radius = zero;
        visuals.widgets.active.corner_radius = zero;

        visuals.clip_rect_margin = 0.;

        // visuals.widgets.inactive.bg_fill = Color32::from_rgb(27, 27, 27);
        // visuals.widgets.inactive.fg_stroke.color = Color32::from_rgb(50, 0, 0);
        // visuals.widgets.inactive.corner_radius = zero;

        // visuals.widgets.hovered.bg_fill = Color32::from_rgb(0, 0, 50);
        // visuals.widgets.hovered.fg_stroke.color = Color32::from_rgb(255, 255, 255);
        // visuals.widgets.hovered.corner_radius = zero;

        // visuals.widgets.active.bg_fill = Color32::from_rgb(0, 50, 0);
        // visuals.widgets.active.fg_stroke.color = Color32::from_rgb(0, 255, 0);
        // visuals.widgets.active.corner_radius = zero;

        style.visuals = visuals;
    });
}

fn sep(ui: &mut Ui) {
    ui.add_space(5.);
    ui.separator();
    ui.add_space(5.);
}

fn space(ui: &mut Ui) {
    ui.add_space(5.);
}

fn header(
    ui: &mut Ui,
    solution: &SolutionResource,
    ev_w: &mut EventWriter<ActionEvent>,
    jobs: &mut EventWriter<JobRequest>,
    configuration: &mut ResMut<Configuration>,
    render_object_settings_store: &mut ResMut<RenderObjectSettingStore>,
    time: &Res<Time>,
) {
    ui.with_layout(Layout::left_to_right(Align::TOP), |ui| {
        Frame {
            outer_margin: bevy_egui::egui::epaint::Margin::symmetric(15, 0),
            shadow: bevy_egui::egui::epaint::Shadow::NONE,
            ..default()
        }
        .show(ui, |ui| {
            bevy_egui::egui::menu::bar(ui, |ui| {
                menu_button(ui, "File", |ui| {
                    if sleek_button(ui, "Load") {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("triangulated geometry", &["obj", "stl", "save", "flag", "dcube"])
                            .pick_file()
                        {
                            jobs.write(JobRequest::Run(Box::new(Job::Import { path })));
                        }
                    }
                    sep(ui);
                    if sleek_button(ui, "Export") {
                        if let Some(path) = rfd::FileDialog::new().save_file() {
                            jobs.write(JobRequest::Run(Box::new(Job::Export {
                                solution: solution.current_solution.clone(),
                                path,
                            })));
                        }
                    }
                    space(ui);
                    if sleek_button(ui, "Export (NLR)") {
                        if let Some(path) = rfd::FileDialog::new().save_file() {
                            jobs.write(JobRequest::Run(Box::new(Job::ExportNLR {
                                solution: solution.current_solution.clone(),
                                path,
                            })));
                        }
                    }
                    space(ui);
                    if sleek_button(ui, "Export (Honors)") {
                        if let Some(path) = rfd::FileDialog::new().save_file() {
                            jobs.write(JobRequest::Run(Box::new(Job::ExportDotgraph {
                                solution: solution.current_solution.clone(),
                                path,
                            })));
                        }
                    }
                    sep(ui);
                    if sleek_button(ui, "Quit") {
                        std::process::exit(0);
                    }
                });

                menu_button(ui, "Rendering", |ui| {
                    // Select different presets of rendering combinations.

                    if sleek_button(ui, "Dual") {
                        let true_labels = ["black", "X-loops", "Y-loops", "Z-loops", "colored"];
                        for settings in render_object_settings_store.objects.values_mut() {
                            for (label, setting) in settings.settings.iter_mut() {
                                setting.visible = true_labels.contains(&label.as_str());
                            }
                        }
                    }
                    if sleek_button(ui, "Primal") {
                        let true_labels = ["segmentation", "colored", "paths", "flat paths"];
                        for settings in render_object_settings_store.objects.values_mut() {
                            for (label, setting) in settings.settings.iter_mut() {
                                setting.visible = true_labels.contains(&label.as_str());
                            }
                        }
                    }
                    if sleek_button(ui, "Input") {
                        let true_labels = ["gray", "wireframe"];
                        for settings in render_object_settings_store.objects.values_mut() {
                            for (label, setting) in settings.settings.iter_mut() {
                                setting.visible = true_labels.contains(&label.as_str());
                            }
                        }
                    }
                });

                menu_button(ui, "Camera", |ui| {
                    // Select different presets of rendering combinations.

                    // slider for camera configs
                    let m = 2.0;

                    if sleek_button(ui, "reset to default") {
                        configuration.camera_rotate_sensitivity = 0.2;
                        configuration.camera_translate_sensitivity = 2.0;
                        configuration.camera_zoom_sensitivity = 0.2;
                    }

                    let mut rotate = 1. + configuration.camera_rotate_sensitivity.log10() / m;
                    slider(ui, "rotate", &mut rotate, 0.1..=1.);
                    configuration.camera_rotate_sensitivity = 10f32.powf((rotate - 1.) * m);

                    let mut translate = 1. + ((configuration.camera_translate_sensitivity / 3.).log10() / m);
                    slider(ui, "translate", &mut translate, 0.1..=1.);
                    configuration.camera_translate_sensitivity = 10f32.powf((translate - 1.) * m) * 3.;

                    let mut zoom = 1. + configuration.camera_zoom_sensitivity.log10() / m;
                    slider(ui, "zoom", &mut zoom, 0.1..=1.);
                    configuration.camera_zoom_sensitivity = 10f32.powf((zoom - 1.) * m);

                    if sleek_button(ui, "precision mode") {
                        configuration.camera_rotate_sensitivity = 0.01;
                        configuration.camera_translate_sensitivity = 0.01;
                        configuration.camera_zoom_sensitivity = 0.01;
                    }
                });
            });
        });
    });
}

fn footer(
    egui_ctx: &mut bevy_egui::EguiContexts,
    conf: &mut Configuration,
    solution: &SolutionResource,
    diagnostics: &Res<DiagnosticsStore>,
    job_state: &Res<JobState>,
    jobs: &mut EventWriter<JobRequest>,
    time: &Res<Time>,
    axes_texture: TextureId,
) {
    TopBottomPanel::bottom("footer").show_separator_line(false).show(egui_ctx.ctx_mut(), |ui| {
        ui.separator();

        ui.add_space(5.);

        ui.with_layout(Layout::left_to_right(Align::TOP), |ui| {
            // Left side: Display FPS
            ui.with_layout(Layout::left_to_right(Align::TOP), |ui| {
                ui.add_space(15.);

                let size = 8.0;

                let mut job = text::LayoutJob::default();

                ui.add(bevy_egui::egui::widgets::Image::new(bevy_egui::egui::load::SizedTexture::new(
                    axes_texture,
                    [10., 10.],
                )));

                job.append("right-hand: ", 0.0, text_format(size, Color32::LIGHT_GRAY));
                let red = colors::from_direction(PrincipalDirection::X, Some(Perspective::Primal), None);
                job.append(
                    "+X",
                    0.0,
                    text_format(size, Color32::from_rgb((red[0] * 255.) as u8, (red[1] * 255.) as u8, (red[2] * 255.) as u8)),
                );

                job.append(", ", 0.0, text_format(size, Color32::GRAY));

                let yellow = colors::from_direction(PrincipalDirection::Y, Some(Perspective::Primal), None);
                job.append(
                    "+Y",
                    0.0,
                    text_format(
                        size,
                        Color32::from_rgb((yellow[0] * 255.) as u8, (yellow[1] * 255.) as u8, (yellow[2] * 255.) as u8),
                    ),
                );

                job.append(", ", 0.0, text_format(size, Color32::GRAY));

                let green = colors::from_direction(PrincipalDirection::Z, Some(Perspective::Primal), None);
                job.append(
                    "+Z",
                    0.0,
                    text_format(
                        size,
                        Color32::from_rgb((green[0] * 255.) as u8, (green[1] * 255.) as u8, (green[2] * 255.) as u8),
                    ),
                );

                ui.label(job);

                let mut job = text::LayoutJob::default();

                fn usage_color(value: f64) -> Color32 {
                    if value < 70.0 {
                        Color32::LIGHT_GRAY // neutral
                    } else if value < 90.0 {
                        Color32::from_rgb(180, 100, 100) // faded red
                    } else {
                        Color32::RED // critical
                    }
                }

                fn fps_color(fps: f64) -> Color32 {
                    if fps < 30.0 {
                        Color32::RED // very bad
                    } else if fps < 50.0 {
                        Color32::from_rgb(180, 100, 100) // warning
                    } else {
                        Color32::LIGHT_GRAY // normal
                    }
                }

                let fps = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS).and_then(|d| d.smoothed()).unwrap_or(0.0);

                let sys_cpu = diagnostics
                    .get(&SystemInformationDiagnosticsPlugin::SYSTEM_CPU_USAGE)
                    .and_then(|d| d.smoothed())
                    .unwrap_or(0.0);

                let sys_mem = diagnostics
                    .get(&SystemInformationDiagnosticsPlugin::SYSTEM_MEM_USAGE)
                    .and_then(|d| d.smoothed())
                    .unwrap_or(0.0);

                let proc_cpu = diagnostics
                    .get(&SystemInformationDiagnosticsPlugin::PROCESS_CPU_USAGE)
                    .and_then(|d| d.smoothed())
                    .unwrap_or(0.0);

                let proc_mem = diagnostics
                    .get(&SystemInformationDiagnosticsPlugin::PROCESS_MEM_USAGE)
                    .and_then(|d| d.smoothed())
                    .unwrap_or(0.0);

                job.append("  |  ", 0.0, text_format(9.0, Color32::GRAY));

                job.append(&format!("fps {:>3.0}", fps), 0.0, text_format(size, fps_color(fps)));

                job.append("  |  ", 0.0, text_format(9.0, Color32::GRAY));

                job.append(&format!("scpu {:>3.0}%", sys_cpu), 0.0, text_format(size, usage_color(sys_cpu)));

                job.append("  |  ", 0.0, text_format(9.0, Color32::GRAY));

                job.append(&format!("smem {:>3.0}%", sys_mem), 0.0, text_format(size, usage_color(sys_mem)));

                job.append("  |  ", 0.0, text_format(9.0, Color32::GRAY));

                job.append(&format!("pcpu {:>3.0}%", proc_cpu), 0.0, text_format(size, usage_color(proc_cpu)));

                job.append("  |  ", 0.0, text_format(9.0, Color32::GRAY));

                job.append(&format!("pmem {:>3.0}%", proc_mem), 0.0, text_format(size, usage_color(proc_mem)));

                job.append("  |  ", 0.0, text_format(9.0, Color32::GRAY));

                if let Some(request) = &job_state.request {
                    job.append(&format!("{}  {}", request, &timer_animation(time)), 0.0, text_format(size, Color32::LIGHT_GRAY));
                } else {
                    job.append("idle", 0.0, text_format(size, Color32::LIGHT_GRAY));
                }

                ui.label(job);
            });

            // Right side: Display fixed label
            ui.with_layout(Layout::right_to_left(Align::TOP), |ui| {
                ui.add_space(15.);
                let mut job = text::LayoutJob::default();
                display_label(&mut job, "DualCube by snoep");
                ui.label(job);
            });
        });

        ui.add_space(5.);
    });
}

fn display_label(job: &mut text::LayoutJob, label: &str) {
    job.append(label, 0.0, text_format(9.0, Color32::WHITE));
}

pub fn update(
    mut egui_ctx: bevy_egui::EguiContexts,
    mut ev_w: EventWriter<ActionEvent>,
    mut jobs: EventWriter<JobRequest>,
    mut conf: ResMut<Configuration>,
    job_state: Res<JobState>,
    solution: Res<SolutionResource>,
    mut render_setting_store: ResMut<RenderObjectSettingStore>,
    time: Res<Time>,
    image_handle: Res<CameraHandles>,
    mut ui_resource: ResMut<UiResource>,
    diagnostics: Res<DiagnosticsStore>,
    mesh_ref: Res<InputResource>,
    axes_texture: Res<AxesGizmoTexture>,
) {
    let axes_texture = egui_ctx.add_image(axes_texture.0.clone());

    TopBottomPanel::top("panel").show_separator_line(false).show(egui_ctx.ctx_mut(), |ui| {
        ui.add_space(10.);

        if job_state.request.is_some() {
            ui.output_mut(|o| o.cursor_icon = bevy_egui::egui::CursorIcon::Progress);
        }

        ui.horizontal(|ui| {
            ui.with_layout(Layout::top_down(Align::TOP), |ui| {
                // FIRST ROW
                header(ui, &solution, &mut ev_w, &mut jobs, &mut conf, &mut render_setting_store, &time);

                ui.add_space(5.);

                ui.separator();

                ui.add_space(5.);

                ui.with_layout(Layout::left_to_right(Align::TOP), |ui| {
                    // Center: Display status of dual, embd, alignment, and orthogonality

                    ui.add_space(5.);

                    let text_size = 12.5;

                    bevy_egui::egui::menu::bar(ui, |ui| {
                        // ****************
                        // INPUT
                        // ****************
                        label(ui, "Input", text_size, Color32::WHITE);
                        label(ui, &format!("({})", solution.current_solution.mesh_ref.nr_verts()), text_size, Color32::GRAY);

                        let mut stopped = false;

                        let stop_label = "  stop  ";
                        let continue_label = "  --->  ";

                        if conf.stop == Phase::Input {
                            stopped = true;
                            if sleek_button_warn(ui, stop_label) {
                                conf.stop = Phase::None;
                            }
                        } else if sleek_button_unfocused(ui, continue_label) {
                            conf.stop = Phase::Input;
                        }

                        // ****************
                        // LOOPS
                        // ****************
                        if solution.current_solution.mesh_ref.nr_verts() == 0 || stopped {
                            label(ui, "Loops", text_size, Color32::GRAY);
                        } else {
                            menu_button(ui, "Loops", |ui| {
                                if sleek_button(ui, "initialize") {
                                    jobs.write(JobRequest::Run(Box::new(Job::InitializeLoops {
                                        solution: solution.current_solution.clone(),
                                        flowgraphs: mesh_ref.flow_graphs.clone(),
                                        configuration: conf.clone(),
                                    })));
                                    ui.close_menu();
                                }

                                if sleek_button(ui, "evolve") {
                                    jobs.write(JobRequest::Run(Box::new(Job::Evolve {
                                        solution: solution.current_solution.clone(),
                                        configuration: conf.clone(),
                                        flowgraphs: mesh_ref.flow_graphs.clone(),
                                    })));
                                    ui.close_menu();
                                }
                                slider(ui, "iterations", &mut conf.iterations, 1..=20);
                                slider(ui, "pool1", &mut conf.pool1, 1..=20);
                                slider(ui, "pool2", &mut conf.pool2, 1..=50);
                            });
                            label(ui, &format!("({})", solution.current_solution.loops.len()), 13., Color32::GRAY);
                        }

                        if conf.stop == Phase::Loops {
                            stopped = true;
                            if sleek_button_warn(ui, stop_label) {
                                conf.stop = Phase::None;
                            }
                        } else if sleek_button_unfocused(ui, continue_label) {
                            conf.stop = Phase::Loops;
                        }

                        // ****************
                        // DUAL
                        // ****************
                        match (&solution.current_solution.loops.len(), stopped) {
                            (0, _) | (_, true) => {
                                label(ui, "Dual", text_size, Color32::GRAY);
                            }
                            _ => {
                                menu_button(ui, "Dual", |ui| {
                                    if sleek_button(ui, "(re)compute") {
                                        jobs.write(JobRequest::Run(Box::new(Job::ComputeDual {
                                            solution: solution.current_solution.clone(),
                                            configuration: conf.clone(),
                                        })));
                                        ui.close_menu();
                                    }
                                });
                                let status = match solution.current_solution.dual {
                                    Ok(_) => "(Ok)",
                                    Err(_) => "(err)",
                                };
                                label(ui, status, text_size, Color32::GRAY);
                            }
                        }

                        if conf.stop == Phase::Dual {
                            stopped = true;
                            if sleek_button_warn(ui, stop_label) {
                                conf.stop = Phase::None;
                            }
                        } else if sleek_button_unfocused(ui, continue_label) {
                            conf.stop = Phase::Dual;
                        }

                        // ****************
                        // LAYOUT
                        // ****************
                        match (&solution.current_solution.dual, stopped) {
                            (Err(_), _) | (_, true) => {
                                label(ui, "Layout", text_size, Color32::GRAY);
                            }
                            (Ok(_), _) => {
                                menu_button(ui, "Layout", |ui| {
                                    // Place corners
                                    if sleek_button(ui, "(re)compute corners") {
                                        jobs.write(JobRequest::Run(Box::new(Job::PlaceCorners {
                                            solution: solution.current_solution.clone(),
                                            configuration: conf.clone(),
                                        })));
                                        ui.close_menu();
                                    }
                                    // Place paths
                                    if sleek_button(ui, "(re)compute paths") {
                                        jobs.write(JobRequest::Run(Box::new(Job::PlacePaths {
                                            solution: solution.current_solution.clone(),
                                            configuration: conf.clone(),
                                        })));
                                        ui.close_menu();
                                    }
                                    // Optimize corners
                                    if sleek_button(ui, "optimize") {
                                        jobs.write(JobRequest::Run(Box::new(Job::SmoothenLayout {
                                            solution: solution.current_solution.clone(),
                                            configuration: conf.clone(),
                                        })));
                                        ui.close_menu();
                                    }
                                });

                                label(
                                    ui,
                                    &format!("({:.3})", solution.current_solution.get_quality().unwrap_or(0.0)),
                                    13.,
                                    Color32::GRAY,
                                );
                            }
                        }

                        if conf.stop == Phase::Layout {
                            stopped = true;
                            if sleek_button_warn(ui, stop_label) {
                                conf.stop = Phase::None;
                            }
                        } else if sleek_button_unfocused(ui, continue_label) {
                            conf.stop = Phase::Layout;
                        }

                        // ****************
                        // POLYCUBE
                        // ****************
                        match (&solution.current_solution.layout, stopped) {
                            (Err(_), _) | (_, true) => {
                                label(ui, "Polycube", text_size, Color32::GRAY);
                            }
                            (Ok(_), _) => {
                                menu_button(ui, "Polycube", |ui| {
                                    ui.checkbox(&mut conf.unit, "unit");
                                    if sleek_button(ui, "(re)compute") {
                                        jobs.write(JobRequest::Run(Box::new(Job::ComputePolycube {
                                            solution: solution.current_solution.clone(),
                                            configuration: conf.clone(),
                                        })));
                                        ui.close_menu();
                                    }
                                });
                            }
                        }

                        if conf.stop == Phase::Polycube {
                            stopped = true;
                            if sleek_button_warn(ui, stop_label) {
                                conf.stop = Phase::None;
                            }
                        } else if sleek_button_unfocused(ui, continue_label) {
                            conf.stop = Phase::Polycube;
                        }

                        // ****************
                        // QUAD
                        // ****************
                        match (&solution.current_solution.quad, stopped) {
                            (None, _) | (_, true) => {
                                label(ui, "Quad", text_size, Color32::GRAY);
                            }
                            (Some(_quad), _) => {
                                menu_button(ui, "Quad", |ui| {
                                    if sleek_button(ui, "(re)compute") {
                                        jobs.write(JobRequest::Run(Box::new(Job::ComputeQuad {
                                            solution: solution.current_solution.clone(),
                                            configuration: conf.clone(),
                                        })));
                                        ui.close_menu();
                                    }

                                    slider(ui, "omega", &mut conf.omega, 1..=20);

                                    if sleek_button(ui, "optimize") {
                                        jobs.write(JobRequest::Run(Box::new(Job::SmoothenQuad {
                                            solution: solution.current_solution.clone(),
                                            configuration: conf.clone(),
                                        })));
                                        ui.close_menu();
                                    }
                                });

                                label(ui, "(Ok)", 13., Color32::GRAY);
                            }
                        }
                    });
                });

                ui.add_space(15.);

                // NEXT ROW
                ui.with_layout(Layout::left_to_right(Align::TOP), |ui| {
                    ui.add_space(15.);

                    bevy_egui::egui::menu::bar(ui, |ui| {
                        match conf.interactive_mode {
                            InteractiveMode::None => {
                                if sleek_button_unfocused(ui, "edit loops") {
                                    conf.interactive_mode = InteractiveMode::LoopModification;
                                }
                                if sleek_button_unfocused(ui, "edit segm") {
                                    conf.interactive_mode = InteractiveMode::SegmentationModification;
                                }
                            }
                            InteractiveMode::LoopModification => {
                                if sleek_button(ui, "edit loops") {
                                    conf.interactive_mode = InteractiveMode::LoopModification;
                                }
                                if sleek_button_unfocused(ui, "edit segm") {
                                    conf.interactive_mode = InteractiveMode::SegmentationModification;
                                }
                            }
                            InteractiveMode::SegmentationModification => {
                                if sleek_button_unfocused(ui, "edit loops") {
                                    conf.interactive_mode = InteractiveMode::LoopModification;
                                }
                                if sleek_button(ui, "edit segm") {
                                    conf.interactive_mode = InteractiveMode::SegmentationModification;
                                }
                            }
                        }

                        ui.add_space(15.);

                        match conf.interactive_mode {
                            InteractiveMode::LoopModification => {
                                for direction in [PrincipalDirection::X, PrincipalDirection::Y, PrincipalDirection::Z] {
                                    radio(
                                        ui,
                                        &mut conf.direction,
                                        direction,
                                        Color32::from_rgb(
                                            (colors::from_direction(direction, Some(Perspective::Dual), None)[0] * 255.) as u8,
                                            (colors::from_direction(direction, Some(Perspective::Dual), None)[1] * 255.) as u8,
                                            (colors::from_direction(direction, Some(Perspective::Dual), None)[2] * 255.) as u8,
                                        ),
                                    );
                                }

                                // add slider for alpha (or 1-beta)
                                slider(ui, "alpha", &mut conf.alpha, 0.0..=1.0);

                                if let Some(edgepair) = conf.selected {
                                    if let Some(Some(sol)) = solution.next[conf.direction as usize].get(&edgepair) {
                                        ui.label("DUAL[");
                                        if sol.dual.is_ok() {
                                            ui.label(colored_text("Ok", Color32::GREEN));
                                        } else {
                                            ui.label(colored_text(&format!("{:?}", sol.dual.as_ref().err()), Color32::RED));
                                        }
                                        ui.label("]");

                                        ui.label("EMBD[");
                                        if sol.layout.is_ok() {
                                            ui.label(colored_text("Ok", Color32::GREEN));
                                        } else {
                                            ui.label(colored_text(&format!("{:?}", sol.layout.as_ref().err()), Color32::RED));
                                        }

                                        ui.label("]");

                                        if let Some(alignment) = sol.alignment {
                                            ui.label("ALIGN[");
                                            ui.label(format!("{alignment:.3}"));
                                            ui.label("]");
                                        }
                                    }
                                }
                            }
                            InteractiveMode::None => {}
                            InteractiveMode::SegmentationModification => {
                                ui.label(" ! click to move corners ! ");
                            }
                        }
                    })
                });

                ui.add_space(5.);
            });
        });
    });

    footer(&mut egui_ctx, &mut conf, &solution, &diagnostics, &job_state, &mut jobs, &time, axes_texture);

    let mut egui_handles = vec![];
    for obj in conf.window_shows_object.iter() {
        let egui_handle = egui_ctx.add_image(image_handle.map.get(&CameraFor(*obj)).unwrap().clone());
        egui_handles.push(egui_handle);
    }

    bevy_egui::egui::CentralPanel::default()
        .frame(Frame {
            stroke: bevy_egui::egui::epaint::Stroke {
                width: 20.0,
                color: Color32::from_gray(27),
            },
            fill: Color32::TRANSPARENT,
            ..default()
        })
        .show(egui_ctx.ctx_mut(), |ui| {
            let dock_area = DockArea::new(&mut ui_resource.tree)
                .show_leaf_collapse_buttons(false)
                .show_leaf_close_all_buttons(false);
            let mut dock_area_style = Style::from_egui(ui.style());
            dock_area_style.dock_area_padding = Some(bevy_egui::egui::epaint::Margin::same(20));
            dock_area_style.tab_bar.corner_radius = CornerRadius::same(0);
            dock_area_style.tab_bar.bg_fill = Color32::from_gray(27);
            dock_area_style.tab_bar.hline_color = Color32::from_gray(27);
            dock_area_style.separator.width = 1.;
            dock_area_style.separator.color_dragged = Color32::from_gray(27);
            dock_area_style.separator.color_hovered = Color32::from_gray(27);
            dock_area_style.separator.color_idle = Color32::from_gray(27);

            dock_area_style.overlay.selection_color = Color32::from_rgba_unmultiplied(50, 50, 50, 100);
            dock_area_style.overlay.overlay_type = egui_dock::OverlayType::HighlightedAreas;

            let settings_copy = &render_setting_store.objects;

            let mut tab_viewer = TabViewer {
                egui_handles: egui_handles.clone(),
                render_settings: settings_copy.clone(),
            };
            dock_area.style(dock_area_style).show(ui.ctx(), &mut tab_viewer);

            if settings_copy != &tab_viewer.render_settings {
                render_setting_store.objects = tab_viewer.render_settings.clone();
            }
        });
}

fn slider<T: emath::Numeric>(ui: &mut Ui, label: &str, value: &mut T, range: std::ops::RangeInclusive<T>) {
    ui.add(Slider::new(value, range).text(text(label)));
}

#[allow(dead_code)]
fn stepper(ui: &mut Ui, label: &str, value: &mut u32, min: u32, max: u32) -> bool {
    ui.horizontal(|ui| {
        if ui.button("<<").clicked() {
            let new_value = *value - 1;
            if new_value >= min && new_value <= max {
                *value = new_value;
            } else {
                *value = max;
            };
            return true;
        }
        ui.label(format!("{label}: {value} [{min}-{max}]"));
        if ui.button(">>").clicked() {
            let new_value = *value + 1;
            if new_value <= max && new_value >= min {
                *value = new_value;
            } else {
                *value = min;
            };
            return true;
        }
        false
    })
    .inner
}

fn radio<T: PartialEq<T> + std::fmt::Display>(ui: &mut Ui, item: &mut T, value: T, color: Color32) -> bool {
    if ui.radio(*item == value, colored_text(&format!("{value}"), color)).clicked() {
        *item = value;
        true
    } else {
        false
    }
}

pub fn text(string: &str) -> text::LayoutJob {
    colored_text(string, Color32::WHITE)
}

pub fn colored_text(string: &str, color: Color32) -> text::LayoutJob {
    let mut job = text::LayoutJob::default();
    job.append(string, 0.0, text_format(13.0, color));
    job
}

pub fn text_format(size: f32, color: Color32) -> TextFormat {
    TextFormat {
        font_id: FontId {
            size,
            family: bevy_egui::egui::FontFamily::Monospace,
        },
        color,
        ..Default::default()
    }
}

pub fn menu_button(ui: &mut Ui, label: &str, f: impl FnOnce(&mut Ui)) {
    bevy_egui::egui::menu::menu_button(ui, RichText::new(label).color(Color32::WHITE), f);
}

#[allow(dead_code)]
pub fn menu_button_unfocused(ui: &mut Ui, label: &str, f: impl FnOnce(&mut Ui)) {
    bevy_egui::egui::menu::menu_button(ui, RichText::new(label).color(Color32::GRAY), f);
}

pub fn sleek_button(ui: &mut Ui, label: &str) -> bool {
    bevy_egui::egui::menu::menu_button(ui, RichText::new(label).color(Color32::WHITE), |ui| {
        ui.close_menu();
    })
    .response
    .clicked()
}

pub fn sleek_button_warn(ui: &mut Ui, label: &str) -> bool {
    bevy_egui::egui::menu::menu_button(ui, RichText::new(label).color(Color32::RED), |ui| {
        ui.close_menu();
    })
    .response
    .clicked()
}

pub fn sleek_button_unfocused(ui: &mut Ui, label: &str) -> bool {
    bevy_egui::egui::menu::menu_button(ui, RichText::new(label).color(Color32::GRAY), |ui| {
        ui.close_menu();
    })
    .response
    .clicked()
}

pub fn label(ui: &mut Ui, label: &str, size: f32, color: Color32) {
    let mut job = text::LayoutJob::default();
    job.append(label, 0.0, text_format(size, color));
    ui.label(job);
}

// pub fn timer_animation(time: &Time) -> String {
//     let frequency = 2.0;
//     let animation = ["(^_^)", "(o_o)", "(o_O)", "(>_<)", "(-_-)", "(._.)", "(X_X)"];
//     let index = (time.elapsed_secs() * frequency) as usize % animation.len();
//     animation[index].to_string()
// }

pub fn timer_animation(time: &Time) -> String {
    let frequency = 6.0;
    let animation = ["●○○○", "○●○○", "○○●○", "○○○●", "○○●○", "○●○○"];
    let index = (time.elapsed_secs() * frequency) as usize % animation.len();
    animation[index].to_string()
}
