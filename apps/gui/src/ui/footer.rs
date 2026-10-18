use super::menu::ScoreCache;
use super::theme::{TEXT_COLOR, WARN_RED, sized_text, sized_text_format, to_color32};
use super::widgets::timer_animation;
use crate::colors;
use crate::controls::InteractiveMode;
use crate::jobs::JobState;
use crate::resources::{Configuration, SolutionResource};
use bevy::diagnostic::{
    DiagnosticPath, DiagnosticsStore, FrameTimeDiagnosticsPlugin,
    SystemInformationDiagnosticsPlugin,
};
use bevy::prelude::*;
use bevy_egui::egui::{Align, Layout, Panel, text};
use dualcube::prelude::*;

const STAT_SIZE: f32 = 8.0;
// Warn above this share of the memory of the system in use (%), and above this share of the processors used by other
// programs (%).
const MEMORY_WARNING: f64 = 85.;
const OTHERS_WARNING: f64 = 25.;

/// Shows the bottom panel.
pub fn show(
    mut egui_ctx: bevy_egui::EguiContexts<'_, '_>,
    conf: Res<'_, Configuration>,
    diagnostics: Res<'_, DiagnosticsStore>,
    job_state: Res<'_, JobState>,
    time: Res<'_, Time>,
    solution: Res<'_, SolutionResource>,
    mut score_cache: Local<'_, ScoreCache>,
) -> Result<(), BevyError> {
    // The model (vertices, edges, faces, genus) and the solution (loops, quality score).
    let current = &solution.current_solution;
    let mesh = &current.mesh_ref;
    let (v, e, f) = (
        mesh.nr_verts() as i64,
        mesh.nr_edges() as i64 / 2,
        mesh.nr_faces() as i64,
    );
    let score = score_cache
        .get(current, &conf.quality)
        .filter(|_| current.polycube.is_some());
    let model = [
        format!("V {v}"),
        format!("E {e}"),
        format!("F {f}"),
        format!("genus {}", (2 - (v - e + f)) / 2),
        format!("loops {}", current.loops.len()),
        score.map_or_else(|| "score -".to_owned(), |score| format!("score {score:.4}")),
    ];
    let mut root = super::root_ui(egui_ctx.ctx_mut()?, "footer root");
    Panel::bottom(super::FOOTER)
        .show_separator_line(false)
        .show(&mut root, |ui| {
            ui.add_space(5.);
            ui.separator();
            ui.add_space(5.);

            ui.with_layout(Layout::left_to_right(Align::TOP), |ui| {
                ui.with_layout(Layout::left_to_right(Align::TOP), |ui| {
                    ui.add_space(30.);

                    // The right-handed coordinate tripod, colored per axis.
                    let mut tripod = text::LayoutJob::default();
                    tripod.append(
                        "right-hand: ",
                        0.0,
                        sized_text_format(STAT_SIZE, TEXT_COLOR),
                    );
                    for (i, direction) in DIRECTIONS.into_iter().enumerate() {
                        if i > 0 {
                            tripod.append(", ", 0.0, sized_text_format(STAT_SIZE, TEXT_COLOR));
                        }
                        let color = to_color32(colors::from_direction(
                            direction,
                            Some(Perspective::Primal),
                            None,
                        ));
                        tripod.append(
                            ["+X", "+Y", "+Z"][i],
                            0.0,
                            sized_text_format(STAT_SIZE, color),
                        );
                    }
                    ui.label(tripod);

                    // The usage of this program (the processors it keeps busy, its memory), warnings when the memory
                    // of the system is nearly full or other programs take much of the processors, the mode, and the
                    // job status. All measurements (and the frame rate) on hover.
                    let measure = |path: &DiagnosticPath| {
                        diagnostics
                            .get(path)
                            .and_then(|d| d.smoothed())
                            .unwrap_or(0.0)
                    };
                    let system_cpu = measure(&SystemInformationDiagnosticsPlugin::SYSTEM_CPU_USAGE);
                    let system_memory = measure(&SystemInformationDiagnosticsPlugin::SYSTEM_MEM_USAGE);
                    let process_cpu = measure(&SystemInformationDiagnosticsPlugin::PROCESS_CPU_USAGE);
                    let process_memory = measure(&SystemInformationDiagnosticsPlugin::PROCESS_MEM_USAGE);
                    let processors =
                        std::thread::available_parallelism().map_or(1, usize::from);
                    let others = (system_cpu - process_cpu).max(0.);
                    let fps = measure(&FrameTimeDiagnosticsPlugin::FPS);
                    let details = format!(
                        "this program: {process_cpu:.0}% of the processors ({:.1} of {processors}), \
                         {process_memory:.2} GiB of memory\n\
                         system: {system_cpu:.0}% of the processors, {system_memory:.0}% of the memory in use\n\
                         {fps:.0} frames per second",
                        process_cpu / 100. * processors as f64
                    );

                    let mode = match conf.interactive_mode {
                        InteractiveMode::None => "automatic",
                        InteractiveMode::LoopModification => "manual loops",
                        InteractiveMode::SegmentationModification => "manual seg",
                    };
                    let job_status = job_state.request.map_or_else(
                        || "idle".to_string(),
                        |request| format!("{request}  {}", timer_animation(&time)),
                    );

                    let mut entries = vec![
                        (
                            format!(
                                "CPU {:.0}/{processors}",
                                process_cpu / 100. * processors as f64
                            ),
                            TEXT_COLOR,
                        ),
                        (format!("{process_memory:.1} GiB"), TEXT_COLOR),
                    ];
                    if system_memory > MEMORY_WARNING {
                        entries.push((format!("memory {system_memory:.0}% full"), WARN_RED));
                    }
                    if others > OTHERS_WARNING {
                        entries.push((format!("other programs {others:.0}% CPU"), WARN_RED));
                    }
                    entries.push((mode.to_string(), TEXT_COLOR));
                    entries.push((job_status, TEXT_COLOR));

                    let mut stats = text::LayoutJob::default();
                    if v > 0 {
                        for entry in &model {
                            stats.append("  |  ", 0.0, sized_text_format(9.0, TEXT_COLOR));
                            stats.append(entry, 0.0, sized_text_format(STAT_SIZE, TEXT_COLOR));
                        }
                    }
                    for (entry, color) in entries {
                        stats.append("  |  ", 0.0, sized_text_format(9.0, TEXT_COLOR));
                        stats.append(&entry, 0.0, sized_text_format(STAT_SIZE, color));
                    }
                    ui.label(stats).on_hover_text(details);
                });

                ui.with_layout(Layout::right_to_left(Align::TOP), |ui| {
                    ui.add_space(30.);
                    ui.label(sized_text("DualCube by snoep", 9.0, TEXT_COLOR));
                });
            });

            ui.add_space(5.);
        });

    Ok(())
}
