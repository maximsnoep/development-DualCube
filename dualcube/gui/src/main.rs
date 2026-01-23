mod bloom;
mod colors;
mod controls;
mod jobs;
mod render;
mod toons;
mod ui;

use crate::controls::InteractiveMode;
use crate::render::RenderObjectSettingStore;
use crate::toons::ToonsMaterialPlugin;
use crate::ui::UiResource;
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, SystemInformationDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::time::common_conditions::on_timer;
use bevy::window::WindowMode;
use bevy::winit::WinitWindows;
use bevy_egui::EguiPlugin;
use dualcube::polycube::POLYCUBE;
use dualcube::prelude::*;
use dualcube::solutions::Solution;
use itertools::Itertools;
use mehsh::prelude::*;
use ordered_float::OrderedFloat;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use render::{CameraFor, MeshProperties, Objects, RenderObjectStore};
use smooth_bevy_cameras::controllers::orbit::OrbitCameraPlugin;
use smooth_bevy_cameras::LookTransformPlugin;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use winit::window::Icon;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Phase {
    None,
    Input,
    Loops,
    Dual,
    Layout,
    Polycube,
    Quad,
}

#[derive(Resource, Debug, Clone)]
pub struct Configuration {
    pub direction: PrincipalDirection,
    pub alpha: f64,

    pub unit: bool,
    pub omega: usize,
    pub iterations: usize,
    pub pool1: usize,
    pub pool2: usize,

    pub raycasted: Option<[EdgeID; 2]>,
    pub selected: Option<[EdgeID; 2]>,

    pub loop_anchors: Vec<[EdgeID; 2]>,

    pub automatic: bool,

    pub interactive_mode: InteractiveMode,

    pub window_shows_object: [Objects; 3],

    pub camera_rotate_sensitivity: f32,
    pub camera_translate_sensitivity: f32,
    pub camera_zoom_sensitivity: f32,
    pub automatic_rotation_camera: bool,

    pub stop: Phase,

    pub clear_color: [u8; 3],
}

impl Default for Configuration {
    fn default() -> Self {
        Self {
            direction: PrincipalDirection::X,
            alpha: 0.5,

            unit: true,
            omega: 5,
            iterations: 10,
            pool1: 10,
            pool2: 30,

            loop_anchors: vec![],

            stop: Phase::None,

            raycasted: None,
            selected: None,
            automatic: false,
            interactive_mode: InteractiveMode::None,
            window_shows_object: [Objects::PolycubeMap, Objects::QuadMesh, Objects::Polycube],
            clear_color: [27, 27, 27],
            // clear_color: [255, 255, 255],
            camera_rotate_sensitivity: 0.2,
            camera_translate_sensitivity: 2.,
            camera_zoom_sensitivity: 0.2,
            automatic_rotation_camera: true,
        }
    }
}

#[derive(Resource, Default)]
pub struct CameraHandles {
    pub map: HashMap<CameraFor, Handle<Image>>,
}

#[derive(Component)]
pub struct RenderedMesh;

#[derive(Component)]
pub struct Rendered;

#[derive(Component)]
pub struct MainMesh;

#[derive(Event, Debug, Eq, Hash, PartialEq)]
pub enum ActionEvent {
    LoadFile(PathBuf),
    ExportAll,
    ExportState,
    ExportSolution,
    ExportNLR,
    ToHexmesh,
    ResetCamera,
    Mutate,
    SmoothenQuad,
    Recompute,
    Initialize,
}

#[derive(Debug, Clone)]
pub enum ActionEventStatus {
    None,
    Loading,
    Done(String),
}

#[derive(Default, Resource)]
pub struct CacheResource {
    cache: [HashMap<[EdgeID; 2], Vec<([EdgeID; 2], OrderedFloat<f64>)>>; 3],
}

#[derive(Default, Debug, Clone, Resource)]
pub struct InputResource {
    mesh: Arc<mehsh::prelude::Mesh<INPUT>>,
    properties: MeshProperties,
    vertex_lookup: mehsh::prelude::VertLocation<INPUT>,
    triangle_lookup: mehsh::prelude::FaceLocation<INPUT>,
    flow_graphs: [grapff::fixed::FixedGraph<EdgeID, f64>; 3],
}

impl InputResource {
    pub fn new(mesh: Arc<mehsh::prelude::Mesh<INPUT>>) -> Self {
        if mesh.nr_verts() == 0 {
            return InputResource::default();
        }
        let vertex_lookup = mesh.kdtree();
        let triangle_lookup = mesh.bvh();
        let mut flow_graphs = [
            grapff::fixed::FixedGraph::default(),
            grapff::fixed::FixedGraph::default(),
            grapff::fixed::FixedGraph::default(),
        ];

        let mut properties = MeshProperties::default();
        (properties.scale, properties.translation) = mesh.scale_translation();
        properties.source = String::from("im blue dabadee dabada");

        let nodes = mesh.edge_ids();

        for axis in [
            PrincipalDirection::X,
            PrincipalDirection::Y,
            PrincipalDirection::Z,
        ] {
            let edges = nodes
                .clone()
                .into_par_iter()
                .flat_map(|node| {
                    mesh.neighbor_function_edgegraph()(node)
                        .into_iter()
                        .map(|neighbor| {
                            let face1 = mesh.face(node);
                            let face2 = mesh.face(neighbor);

                            if face1 == face2 {
                                let normal = mesh.normal(face1);
                                let m1 = mesh.position(node);
                                let m2 = mesh.position(neighbor);
                                let direction = m2 - m1;
                                let cross = direction.cross(&normal);
                                let angle = cross.angle(&axis.into());

                                (node, neighbor, angle)
                            } else {
                                assert!(mesh.twin(node) == neighbor);
                                (node, neighbor, 0.)
                            }
                        })
                        .collect_vec()
                })
                .collect::<Vec<_>>();

            flow_graphs[axis as usize] = grapff::fixed::FixedGraph::from(nodes.clone(), edges);
        }

        Self {
            mesh,
            properties,
            vertex_lookup,
            triangle_lookup,
            flow_graphs,
        }
    }
}

#[derive(Debug, Clone, Resource)]
pub struct SolutionResource {
    current_solution: Solution,
    next: [HashMap<[EdgeID; 2], Option<Solution>>; 3],
    selected_corner: Option<VertKey<POLYCUBE>>,
}

impl Default for SolutionResource {
    fn default() -> Self {
        Self {
            current_solution: Solution::new(Arc::new(mehsh::mesh::connectivity::Mesh::default())),
            next: [HashMap::new(), HashMap::new(), HashMap::new()],
            selected_corner: None,
        }
    }
}

// We can create our own gizmo config group!
#[derive(Default, Reflect, GizmoConfigGroup)]
struct PerpetualGizmos {}

fn main() {
    App::new()
        .init_resource::<UiResource>()
        .init_resource::<InputResource>()
        .init_resource::<CacheResource>()
        .init_resource::<SolutionResource>()
        .init_resource::<Configuration>()
        .init_resource::<RenderObjectSettingStore>()
        .init_resource::<RenderObjectStore>()
        .init_resource::<CameraHandles>()
        .init_gizmo_group::<PerpetualGizmos>()
        .insert_resource(AmbientLight {
            color: bevy::color::Color::WHITE,
            brightness: 1.0,
            affects_lightmapped_meshes: true,
        })
        // Load default plugins
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "DualCube".to_string(),
                mode: WindowMode::Windowed,
                ..Default::default()
            }),
            ..Default::default()
        }))
        // Plugin for diagnostics
        .add_plugins((
            FrameTimeDiagnosticsPlugin::default(),
            SystemInformationDiagnosticsPlugin,
        ))
        // Plugin for GUI
        .add_plugins(EguiPlugin {
            enable_multipass_for_primary_context: false,
        })
        .add_plugins((bloom::BloomPlugin))
        // Plugin for smooth camera
        .add_plugins((LookTransformPlugin, OrbitCameraPlugin::default()))
        // Material
        .add_plugins(ToonsMaterialPlugin)
        // Jobs system
        .add_plugins(jobs::JobPlugin)
        // Axes gizmo
        .add_plugins(bevy_axes_gizmo::AxesGizmoPlugin {
            colors: [
                colors::to_bevy(colors::from_direction(
                    PrincipalDirection::X,
                    Some(Perspective::Primal),
                    None,
                )),
                colors::to_bevy(colors::from_direction(
                    PrincipalDirection::Y,
                    Some(Perspective::Primal),
                    None,
                )),
                colors::to_bevy(colors::from_direction(
                    PrincipalDirection::Z,
                    Some(Perspective::Primal),
                    None,
                )),
            ],
            width: 5.,
            ..default()
        })
        // Setups
        .add_systems(Startup, ui::setup)
        .add_systems(Startup, render::setup)
        .add_systems(Startup, set_window_icon)
        // Updates
        .add_systems(Update, ui::update)
        .add_systems(Update, render::update)
        .add_systems(
            FixedUpdate,
            render::respawn_renders.run_if(on_timer(Duration::from_millis(100))),
        )
        .add_systems(Update, render::update_camera_settings)
        .add_systems(Update, render::update_render_settings)
        .add_systems(Update, controls::system)
        .add_systems(Update, update_field)
        .add_systems(
            FixedUpdate,
            render::automatic_rotation_camera.run_if(on_timer(Duration::from_millis(10))),
        )
        // .add_systems(Update, handle_tasks)
        .add_event::<ActionEvent>()
        .run();
}

fn set_window_icon(windows: NonSend<WinitWindows>) {
    let path = "dualcube/gui/assets/logo-32.png";
    let window = windows.windows.iter().next().unwrap().1;
    let image = image::open(path)
        .expect("Failed to open icon path")
        .into_rgba8();
    winit::window::Window::set_window_icon(
        window,
        Some(Icon::from_rgba(image.clone().into_raw(), image.width(), image.height()).unwrap()),
    );
}

#[inline]
fn vec3_to_vector3d(v: Vec3) -> Vector3D {
    Vector3D::new(v.x.into(), v.y.into(), v.z.into())
}

#[inline]
fn vector3d_to_vec3(v: Vector3D) -> Vec3 {
    Vec3::new(v.x as f32, v.y as f32, v.z as f32)
}

pub fn update_field(mut sol_res: ResMut<SolutionResource>) {
    // If field_res is empty (None), then initialize it
    if sol_res.current_solution.fields.is_none() {
        let field_x = dualcube::field::Field::from_mesh(&sol_res.current_solution.mesh_ref);
        let field_y = dualcube::field::Field::from_mesh(&sol_res.current_solution.mesh_ref);
        let field_z = dualcube::field::Field::from_mesh(&sol_res.current_solution.mesh_ref);
        sol_res.current_solution.fields = Some(dualcube::field::Fields {
            field_x,
            field_y,
            field_z,
        });
    }
}
