use crate::jobs::{Job, JobRequest, JobState};
use crate::render::{CameraFor, Objects, RenderFlag, RenderObjectStore};
use crate::{colors, ActionEvent, CameraHandles, Configuration, InputResource, Perspective, PrincipalDirection, SolutionResource};
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin, SystemInformationDiagnosticsPlugin};
use bevy::prelude::*;
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
    render_objects: HashMap<Objects, Vec<RenderFlag>>,
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
        if let Some(local_copy) = self.render_objects.get_mut(tab) {
            for flag in local_copy.iter_mut() {
                ui.checkbox(&mut flag.visible, flag.label.clone());
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
                    sep(ui);
                    if sleek_button(ui, "Quit") {
                        std::process::exit(0);
                    }
                });

                sep(ui);

                menu_button(ui, "Solution", |ui| {
                    ui.checkbox(&mut configuration.unit_cubes, "constrain polycube edges to 1 unit size");
                    slider(ui, "omega (quad mesh detail)", &mut configuration.omega, 1..=20);

                    ui.add_space(5.);
                    if sleek_button(ui, "Recompute solution") {
                        jobs.write(JobRequest::Run(Box::new(Job::Recompute {
                            solution: solution.current_solution.clone(),
                            unit: configuration.unit_cubes,
                            omega: configuration.omega,
                        })));
                    }
                });

                ui.separator();

                menu_button(ui, "EXPERIMENTAL", |ui| {
                    ui.with_layout(Layout::left_to_right(Align::TOP), |ui| {
                        if sleek_button(ui, "Quad mesh: Laplacian Smoothing") {
                            jobs.write(JobRequest::Run(Box::new(Job::SmoothenQuad {
                                solution: solution.current_solution.clone(),
                            })));
                        }
                    });

                    ui.add_space(5.);

                    // ui.with_layout(Layout::left_to_right(Align::TOP), |ui| {
                    //     if sleek_button(ui, "Hex mesh: Run 'RobustPolycube'") && !matches!(&configuration.hex_mesh_status, HexMeshStatus::Loading) {
                    //         ev_w.write(ActionEvent::ToHexmesh);
                    //     };
                    //     if matches!(configuration.hex_mesh_status, HexMeshStatus::Loading) {
                    //         ui.add_space(5.);
                    //         ui.label(text(&timer_animation(time)));
                    //     }
                    // });

                    // if let HexMeshStatus::Done(score) = &configuration.hex_mesh_status {
                    //     ui.add_space(5.);
                    //     ui.label("Hex-meshing results:");
                    //     ui.add_space(5.);
                    //     ui.label(format!(
                    //         "hd: {hd:.3}\nsJ: {sj:.3} ({sjmin:.3}-{sjmax:.3})\nirr: {irr:.3}",
                    //         hd = score.hausdorff,
                    //         sj = score.avg_jacob,
                    //         sjmin = score.min_jacob,
                    //         sjmax = score.max_jacob,
                    //         irr = score.irregular,
                    //     ));
                    // }
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
) {
    TopBottomPanel::bottom("footer").show_separator_line(false).show(egui_ctx.ctx_mut(), |ui| {
        ui.separator();

        ui.add_space(5.);

        ui.with_layout(Layout::left_to_right(Align::TOP), |ui| {
            // Left side: Display FPS
            ui.with_layout(Layout::left_to_right(Align::TOP), |ui| {
                ui.add_space(15.);
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

                let size = 8.0;

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

            // Center: Display status of dual, embd, alignment, and orthogonality
            ui.vertical_centered(|ui| {
                let mut job = text::LayoutJob::default();

                let default = || text_format(9.0, Color32::WHITE);
                let off = || text_format(9.0, Color32::GRAY);
                let ok = || text_format(9.0, Color32::GREEN);
                let error = || text_format(9.0, Color32::RED);

                let mut active = true; // all systems go until proven otherwise

                // Input
                job.append("Input: ", 0.0, default());
                if solution.current_solution.mesh_ref.nr_verts() == 0 {
                    job.append("Uninitialized", 0.0, error());
                    active = false;
                } else {
                    job.append(
                        &format!(
                            "V: {}, E: {}, F: {}",
                            solution.current_solution.mesh_ref.nr_verts(),
                            solution.current_solution.mesh_ref.nr_edges() / 2,
                            solution.current_solution.mesh_ref.nr_faces()
                        ),
                        0.0,
                        if active { ok() } else { off() },
                    );
                }

                job.append(" | ", 0.0, off());

                // Loops
                job.append("Loops: ", 0.0, default());
                if active {
                    if solution.current_solution.loops.is_empty() {
                        job.append("Empty", 0.0, error());
                        active = false;
                    } else {
                        job.append(&format!("{}", solution.current_solution.loops.len()), 0.0, ok());
                    }
                } else {
                    job.append("...", 0.0, off());
                }

                job.append(" | ", 0.0, off());

                // Dual
                job.append("Dual: ", 0.0, default());
                if active {
                    match &solution.current_solution.dual {
                        Ok(dual) => {
                            job.append("Ok", 0.0, ok());
                            job.append(
                                &format!(
                                    " (zones: {} , {} , {})",
                                    dual.level_graphs.levels[0].len(),
                                    dual.level_graphs.levels[1].len(),
                                    dual.level_graphs.levels[2].len()
                                ),
                                0.0,
                                default(),
                            );
                        }
                        Err(err) => {
                            job.append("Error", 0.0, error());
                            job.append(&format!(" ({:?})", err), 0.0, error());
                            active = false;
                        }
                    }
                } else {
                    job.append("...", 0.0, off());
                }

                job.append(" | ", 0.0, off());

                // Layout
                job.append("Seg: ", 0.0, default());
                if active {
                    match &solution.current_solution.layout {
                        Ok(_layout) => {
                            job.append("Ok", 0.0, ok());
                            job.append(
                                &format!(" (alignment: {:.5})", solution.current_solution.get_quality().unwrap_or(0.0)),
                                0.0,
                                default(),
                            );
                        }
                        Err(err) => {
                            job.append("Error", 0.0, error());
                            job.append(&format!(" ({:?})", err), 0.0, error());
                            active = false;
                        }
                    }
                } else {
                    job.append("...", 0.0, off());
                }

                job.append(" | ", 0.0, off());

                // Quad
                job.append("Quad: ", 0.0, default());
                if active {
                    if let Some(_quad) = &solution.current_solution.quad {
                        job.append("Ok", 0.0, ok());
                    } else {
                        job.append("None / Error?", 0.0, error());
                    }
                } else {
                    job.append("...", 0.0, off());
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
    mut render_object_store: ResMut<RenderObjectStore>,
    time: Res<Time>,
    image_handle: Res<CameraHandles>,
    mut ui_resource: ResMut<UiResource>,
    diagnostics: Res<DiagnosticsStore>,
    mesh_ref: Res<InputResource>,
) {
    TopBottomPanel::top("panel").show_separator_line(false).show(egui_ctx.ctx_mut(), |ui| {
        ui.add_space(10.);

        if job_state.request.is_some() {
            ui.output_mut(|o| o.cursor_icon = bevy_egui::egui::CursorIcon::Progress);
        }

        ui.horizontal(|ui| {
            ui.with_layout(Layout::top_down(Align::TOP), |ui| {
                // FIRST ROW
                header(ui, &solution, &mut ev_w, &mut jobs, &mut conf, &time);

                ui.add_space(5.);

                ui.separator();

                ui.add_space(5.);

                // NEXT ROW
                ui.with_layout(Layout::left_to_right(Align::TOP), |ui| {
                    ui.add_space(15.);
                });

                ui.add_space(5.);

                // THIRD ROW
                ui.with_layout(Layout::left_to_right(Align::TOP), |ui| {
                    ui.add_space(15.);

                    bevy_egui::egui::menu::bar(ui, |ui| {
                        if if conf.automatic {
                            sleek_button(ui, "AUTO")
                        } else {
                            sleek_button_unfocused(ui, "AUTO")
                        } {
                            if conf.automatic {
                                conf.automatic = false;
                            } else {
                                conf.automatic = true;
                                conf.interactive = false;
                            }
                        };

                        let rt: RichText = RichText::new("|").color(Color32::GRAY);
                        ui.label(rt);

                        if if conf.interactive {
                            sleek_button(ui, "MANUAL")
                        } else {
                            sleek_button_unfocused(ui, "MANUAL")
                        } {
                            if conf.interactive {
                                conf.interactive = false;
                            } else {
                                conf.interactive = true;
                                conf.automatic = false;
                            }
                        };

                        ui.add_space(15.);

                        if conf.automatic {
                            if sleek_button(ui, "initialize") {
                                jobs.write(JobRequest::Run(Box::new(Job::InitializeLoops {
                                    solution: solution.current_solution.clone(),
                                    flowgraphs: mesh_ref.flow_graphs.clone(),
                                })));
                            }

                            if sleek_button(ui, "mutate") {
                                jobs.write(JobRequest::Run(Box::new(Job::Evolve {
                                    solution: solution.current_solution.clone(),
                                    iterations: 10,
                                    pool1: 10,
                                    pool2: 30,
                                    flowgraphs: mesh_ref.flow_graphs.clone(),
                                })));
                            }
                        }

                        if conf.interactive {
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
                    });
                });

                ui.add_space(5.);
            });
        });
    });

    footer(&mut egui_ctx, &mut conf, &solution, &diagnostics, &job_state, &mut jobs, &time);

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

            let flags: HashMap<Objects, Vec<RenderFlag>> = render_object_store
                .objects
                .iter()
                .map(|(&obj, data)| (obj, data.features.iter().map(|f| f.flag()).collect()))
                .collect();
            let mut tab_viewer = TabViewer {
                egui_handles: egui_handles.clone(),
                render_objects: flags.clone(),
            };
            dock_area.style(dock_area_style).show(ui.ctx(), &mut tab_viewer);

            if flags != tab_viewer.render_objects {
                for obj in flags.keys() {
                    for i in 0..flags[obj].len() {
                        render_object_store.objects.get_mut(obj).unwrap().features[i].visible = tab_viewer.render_objects[obj][i].visible;
                    }
                }
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

pub fn sleek_button_unfocused(ui: &mut Ui, label: &str) -> bool {
    bevy_egui::egui::menu::menu_button(ui, RichText::new(label).color(Color32::GRAY), |ui| {
        ui.close_menu();
    })
    .response
    .clicked()
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
