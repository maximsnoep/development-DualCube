use crate::render_skeleton::{
    create_crossing_point_gizmos, create_failed_surgery_face_mesh,
    create_failed_surgery_skeleton_gizmos, create_labeled_skeleton_gizmos,
    create_patch_boundary_gizmos, create_patch_convexity_mesh,
    create_patch_mesh, create_polycube_patch_boundary_gizmos, create_polycube_patch_mesh,
    create_invalid_region_gizmos, create_routing_diagnostics_gizmos, create_skeleton_gizmos,
};
use crate::ui::UiResource;
use crate::{colors, InputResource, MainMesh, PerpetualGizmos};
use crate::{
    to_principal_direction, vector3d_to_vec3, CameraHandles, Configuration, Perspective,
    PrincipalDirection, Rendered,
};
use bevy::camera::RenderTarget;
use bevy::camera::ScalingMode;
use bevy::camera::Viewport;
use bevy::camera::{visibility::RenderLayers, CameraOutputMode};
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::prelude::*;
use bevy::render::render_resource::{
    Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
};
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured};
use bevy_axes_gizmo::AxesGizmoSyncCamera;
use bevy_egui::EguiContexts;
use bevy_egui::EguiGlobalSettings;
use bevy_egui::PrimaryEguiContext;
use bevy_orbit_camera::*;
use bevy_toon::ToonMaterial;
use core::f32;
use dualcube::prelude::*;
use egui_dock::LeafNode;
use enum_iterator::{all, Sequence};
use itertools::Itertools;
use mehsh::prelude::*;
use std::collections::{HashMap, HashSet};
use std::ops::Index;
use std::path::PathBuf;
use std::sync::Mutex;
use wgpu_types::BlendState;

pub static PENDING_SCREENSHOT: Mutex<Option<PathBuf>> = Mutex::new(None);

#[derive(Component)]
pub struct ScreenshotCamera;

#[derive(Resource)]
pub struct ScreenshotHandle(pub Handle<Image>);

/// Number of views captured by a "comprehensive" screenshot (also the number
/// of quadrants in the preview window).
pub const COMPREHENSIVE_VIEW_COUNT: usize = 4;

/// The four comprehensive-screenshot views: which object the offscreen camera
/// frames, and which render features are made visible for it. These mirror the
/// "Patches", "Dual", "Primal", and a segmentation-with-paths preset.
const COMPREHENSIVE_VIEWS: [(Objects, &[&str]); COMPREHENSIVE_VIEW_COUNT] = [
    (Objects::InputMesh, &["patches"]),
    (Objects::InputMesh, &["black", "x-loops", "y-loops", "z-loops"]),
    (Objects::Polycube, &["colored", "paths", "flat paths"]),
    (Objects::InputMesh, &["segmentation", "paths", "flat paths"]),
];

/// When `Some`, the offscreen screenshot camera frames this object instead of
/// the largest panel's object. Used to drive the comprehensive capture through
/// the four views.
#[derive(Resource, Default)]
pub struct ScreenshotCameraOverride(pub Option<Objects>);

/// Handles for the [`COMPREHENSIVE_VIEW_COUNT`] preview tile images shown in the
/// comprehensive preview window's quadrants. Captured frames are copied in here.
#[derive(Resource, Default)]
pub struct PreviewTileHandles(pub Vec<Handle<Image>>);

/// What a comprehensive capture run should do with each captured view.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum ComprehensiveMode {
    /// Only refresh the preview tiles.
    #[default]
    Preview,
    /// Refresh the preview tiles and write each view to disk.
    Save,
}

/// A request from the UI to run a comprehensive capture. Saved into
/// [`PENDING_COMPREHENSIVE`]; consumed by [`drive_comprehensive_capture`].
pub struct ComprehensiveRequest {
    pub mode: ComprehensiveMode,
    /// Output directory for `Save` mode (created if needed).
    pub save_dir: Option<PathBuf>,
    /// Base file name for `Save` mode (the model name); files are
    /// `{base_name}-{n}.png`.
    pub base_name: String,
    /// Contents of `stats.txt` written once at the start of a `Save` run.
    pub stats_text: String,
}

pub static PENDING_COMPREHENSIVE: Mutex<Option<ComprehensiveRequest>> = Mutex::new(None);

/// A captured view handed off from the screenshot observer (which can't touch
/// ECS resources) to [`apply_captured_tiles`].
struct CapturedTile {
    index: usize,
    image: Image,
    save_path: Option<PathBuf>,
}

static CAPTURED_TILES: Mutex<Vec<CapturedTile>> = Mutex::new(Vec::new());

/// Phases of capturing one comprehensive view.
#[derive(Default, PartialEq, Eq, Clone, Copy)]
enum CapturePhase {
    /// Apply the view's render settings + camera override.
    #[default]
    ApplyConfig,
    /// Let the scene respawn and render settle.
    Wait,
    /// Spawn the screenshot capture.
    Capture,
    /// Wait until the captured tile has been processed.
    WaitCapture,
    /// Restore settings and finish.
    Finish,
}

/// State machine driving a comprehensive capture across the four views.
#[derive(Resource, Default)]
pub struct ComprehensiveState {
    active: bool,
    mode: ComprehensiveMode,
    save_dir: Option<PathBuf>,
    base_name: String,
    view: usize,
    phase: CapturePhase,
    /// Seconds left to wait for the scene to respawn + render before capturing.
    wait_secs: f32,
    captures_done: usize,
    /// Snapshot of render settings taken at the start, restored at the end.
    saved_settings: Option<HashMap<Objects, RenderObjectSetting>>,
}

impl ComprehensiveState {
    /// Whether a comprehensive capture run is currently in progress.
    pub fn is_running(&self) -> bool {
        self.active
    }
}

const DEFAULT_CAMERA_EYE: Vec3 = Vec3::new(25.0, 25.0, 25.0);
const DEFAULT_CAMERA_TARGET: Vec3 = Vec3::new(0., 0., 0.);
const DEFAULT_CAMERA_TEXTURE_SIZE: u32 = 640 * 2;

// (p * s) + t = p'
#[must_use]
pub fn transform_coordinates(position: Vector3D, translation: Vector3D, scale: f64) -> Vector3D {
    position * scale + translation
}

// (p' - t) / s = p
#[must_use]
pub fn invert_transform_coordinates(
    position: Vector3D,
    translation: Vector3D,
    scale: f64,
) -> Vector3D {
    (position - translation) / scale
}

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone, Default, Sequence)]
pub enum Objects {
    InputMesh,
    #[default]
    Polycube,
    PolycubeMap,
    QuadMesh,
    ContractedMesh,
}

impl std::fmt::Display for Objects {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Objects::InputMesh => "input mesh",
            Objects::Polycube => "polycube",
            Objects::PolycubeMap => "polycube-map",
            Objects::QuadMesh => "quad mesh",
            Objects::ContractedMesh => "contracted mesh",
        };
        write!(f, "{}", s)
    }
}

#[derive(Clone)]
pub enum RenderAsset {
    Mesh(bevy::mesh::Mesh),
    Gizmo((GizmoAsset, f32, f32)),
}

#[derive(Clone, PartialEq)]
pub struct MeshBundle(Mesh3d);
impl MeshBundle {
    pub const fn new(handle: Handle<bevy::mesh::Mesh>) -> Self {
        Self(Mesh3d(handle))
    }
}

#[derive(Clone)]
pub struct GizmoBundle(Gizmo);

impl PartialEq for GizmoBundle {
    fn eq(&self, other: &Self) -> bool {
        self.0.handle == other.0.handle
    }
}

impl GizmoBundle {
    pub fn new(handle: Handle<GizmoAsset>, width: f32, depth: f32) -> Self {
        Self(Gizmo {
            handle,
            line_config: GizmoLineConfig {
                width,
                joints: GizmoLineJoint::Round(4),
                ..Default::default()
            },
            depth_bias: depth,
        })
    }
}

#[derive(Clone)]
pub struct RenderFeature {
    pub assets: Vec<RenderAsset>,
}

#[derive(Clone)]
pub struct RenderFeatureSetting {
    pub label: String,
    pub visible: bool,
}

impl PartialEq for RenderFeatureSetting {
    fn eq(&self, other: &Self) -> bool {
        self.label == other.label && self.visible == other.visible
    }
}

impl RenderFeature {
    pub fn new(asset: RenderAsset) -> Self {
        Self {
            assets: vec![asset],
        }
    }
}

#[derive(Clone, Default)]
pub struct RenderObject {
    pub labels: Vec<String>,
    pub features: HashMap<String, RenderFeature>,
}

impl RenderObject {
    pub fn add(&mut self, label: &str, feature: RenderFeature) -> &mut Self {
        if let Some(existing) = self.features.get_mut(label) {
            existing.assets.extend(feature.assets);
        } else {
            self.labels.push(label.to_owned());
            self.features.insert(label.to_owned(), feature);
        }
        self
    }

    pub fn mesh<M: Tag>(
        &mut self,
        mesh: &mehsh::prelude::Mesh<M>,
        color_map: &HashMap<FaceKey<M>, colors::Color>,
        label: &str,
    ) -> &mut Self {
        self.add(
            label,
            RenderFeature::new(RenderAsset::Mesh(mesh.bevy(color_map).0)),
        )
    }

    pub fn gizmo(&mut self, gizmo: GizmoAsset, width: f32, depth: f32, label: &str) -> &mut Self {
        self.add(
            label,
            RenderFeature::new(RenderAsset::Gizmo((gizmo, width, depth))),
        )
    }

    pub fn bevy_mesh(&mut self, bevy_mesh: bevy::mesh::Mesh, label: &str) -> &mut Self {
        self.add(label, RenderFeature::new(RenderAsset::Mesh(bevy_mesh)))
    }
}

#[derive(Default, Resource)]
pub struct RenderObjectSettingStore {
    pub objects: HashMap<Objects, RenderObjectSetting>,
}

#[derive(Clone, Default, PartialEq)]
pub struct RenderObjectSetting {
    pub labels: Vec<String>,
    pub settings: HashMap<String, RenderFeatureSetting>,
}

#[derive(Default, Resource)]
pub struct RenderObjectStore {
    pub objects: HashMap<Objects, RenderObject>,
}

impl RenderObjectStore {
    pub fn add_object(&mut self, object: Objects, render_object: RenderObject) {
        self.objects.insert(object, render_object);
    }
}

impl From<Objects> for Vec3 {
    fn from(val: Objects) -> Self {
        match val {
            Objects::InputMesh => Self::new(0., 0., 0.),
            Objects::Polycube => Self::new(0., 0., 1_000.),
            Objects::PolycubeMap => Self::new(0., 1_000., 1_000.),
            Objects::QuadMesh => Self::new(1_000., 0., 1_000.),
            Objects::ContractedMesh => Self::new(2_000., 0., 1_000.),
        }
    }
}

pub fn update_camera_settings(
    mut camera_controller: Query<&mut Controller>,
    configuration: ResMut<Configuration>,
) {
    let Ok(mut main_camera) = camera_controller.single_mut() else {
        warn!("No main camera controller.");
        return;
    };

    *main_camera = Controller {
        mouse_rotate_sensitivity: Vec2::splat(configuration.camera_rotate_sensitivity),
        mouse_translate_sensitivity: Vec2::splat(configuration.camera_translate_sensitivity),
        mouse_wheel_zoom_sensitivity: configuration.camera_zoom_sensitivity,
        ..Default::default()
    };
}

#[derive(Component, PartialEq, Eq, Hash, Debug, Copy, Clone, Default)]
pub struct CameraFor(pub Objects);

pub fn reset(
    commands: &mut Commands,

    cameras: &Query<Entity, With<Camera>>,
    images: &mut ResMut<Assets<Image>>,
    handles: &mut ResMut<CameraHandles>,
    configuration: &ResMut<Configuration>,
) {
    for camera in cameras.iter() {
        commands.entity(camera).despawn();
    }

    // Egui camera.
    commands.spawn((
        // The `PrimaryEguiContext` component requires everything needed to render a primary context.
        PrimaryEguiContext,
        Camera2d,
        // Setting RenderLayers to none makes sure we won't render anything apart from the UI.
        RenderLayers::none(),
        Camera {
            order: 1,
            output_mode: CameraOutputMode::Write {
                blend_state: Some(BlendState::ALPHA_BLENDING),
                clear_color: ClearColorConfig::None,
            },
            clear_color: ClearColorConfig::Custom(bevy::color::Color::NONE),
            ..default()
        },
    ));

    // Main camera. This is the camera that the user can control.
    commands
        .spawn((
            Camera3d::default(),
            Camera {
                clear_color: ClearColorConfig::Custom(bevy::prelude::Color::srgb_u8(
                    configuration.clear_color[0],
                    configuration.clear_color[1],
                    configuration.clear_color[2],
                )),
                ..Default::default()
            },
            // RenderTarget::Window()
            AxesGizmoSyncCamera,
            Tonemapping::None,
            bevy_blossom::CameraMarker,
            bevy_orbit_camera::automatic::Marker,
            // Layer 1 carries the background spheres (see respawn_renders); the
            // dock cameras show them, the screenshot camera does not.
            RenderLayers::from_layers(&[0, 1]),
        ))
        .insert((OrbitCameraBundle::new(
            Controller {
                mouse_rotate_sensitivity: Vec2::splat(0.2),
                mouse_translate_sensitivity: Vec2::splat(2.),
                mouse_wheel_zoom_sensitivity: 0.2,
                ..Default::default()
            },
            DEFAULT_CAMERA_EYE + Vec3::from(Objects::InputMesh),
            DEFAULT_CAMERA_TARGET + Vec3::from(Objects::InputMesh),
            Vec3::Y,
        ),))
        .insert(CameraFor(Objects::InputMesh));

    // Sub cameras. These cameras render to a texture.
    let mut image = Image {
        texture_descriptor: TextureDescriptor {
            label: None,
            size: Extent3d {
                width: DEFAULT_CAMERA_TEXTURE_SIZE,
                height: DEFAULT_CAMERA_TEXTURE_SIZE,
                ..default()
            },
            dimension: TextureDimension::D2,
            format: TextureFormat::Bgra8UnormSrgb,
            mip_level_count: 1,
            sample_count: 1,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        },
        ..default()
    };
    image.resize(image.texture_descriptor.size);

    for object in all::<Objects>() {
        let handle = images.add(image.clone());
        handles.map.insert(CameraFor(object), handle.clone());
        let projection = if object == Objects::PolycubeMap || object == Objects::Polycube {
            let mut proj = OrthographicProjection::default_3d();
            proj.scaling_mode = ScalingMode::FixedVertical {
                viewport_height: 30.,
            };
            Projection::Orthographic(proj)
        } else {
            bevy::prelude::Projection::default()
        };

        commands.spawn((
            Camera3d::default(),
            RenderTarget::Image(handle.into()),
            bevy_blossom::CameraMarker,
            Camera {
                clear_color: ClearColorConfig::Custom(bevy::prelude::Color::srgb_u8(
                    configuration.clear_color[0],
                    configuration.clear_color[1],
                    configuration.clear_color[2],
                )),
                ..Default::default()
            },
            projection,
            Tonemapping::None,
            CameraFor(object),
            RenderLayers::from_layers(&[0, 1]),
        ));
    }

    // Off-screen camera for transparent-background screenshots.
    let mut screenshot_image = Image {
        texture_descriptor: TextureDescriptor {
            label: None,
            size: Extent3d {
                width: 2048,
                height: 2048,
                ..default()
            },
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            mip_level_count: 1,
            sample_count: 1,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        },
        ..default()
    };
    screenshot_image.resize(screenshot_image.texture_descriptor.size);
    let screenshot_handle = images.add(screenshot_image);
    commands.insert_resource(ScreenshotHandle(screenshot_handle.clone()));

    // Preview tiles for the comprehensive 4-quadrant preview. Same format/size
    // as the screenshot target so captured pixels can be copied in directly.
    let mut tile_handles = Vec::with_capacity(COMPREHENSIVE_VIEW_COUNT);
    for _ in 0..COMPREHENSIVE_VIEW_COUNT {
        let mut tile = Image {
            texture_descriptor: TextureDescriptor {
                label: None,
                size: Extent3d {
                    width: 2048,
                    height: 2048,
                    ..default()
                },
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8UnormSrgb,
                mip_level_count: 1,
                sample_count: 1,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            },
            ..default()
        };
        tile.resize(tile.texture_descriptor.size);
        tile_handles.push(images.add(tile));
    }
    commands.insert_resource(PreviewTileHandles(tile_handles));
    commands.spawn((
        Camera3d::default(),
        Projection::default(),
        RenderTarget::Image(screenshot_handle.into()),
        bevy_blossom::CameraMarker,
        Camera {
            clear_color: ClearColorConfig::Custom(bevy::prelude::Color::NONE),
            ..Default::default()
        },
        Tonemapping::None,
        ScreenshotCamera,
        // Layer 0 only: excludes the background spheres (layer 1) so screenshots
        // keep a transparent background even when the camera is outside a sphere.
        RenderLayers::layer(0),
    ));
}

pub fn setup(
    mut commands: Commands,
    mut egui_global_settings: ResMut<EguiGlobalSettings>,
    mut images: ResMut<Assets<Image>>,
    mut handles: ResMut<CameraHandles>,
    cameras: Query<Entity, With<Camera>>,
    configuration: ResMut<Configuration>,
    mut config_store: ResMut<GizmoConfigStore>,
) {
    // Disable the automatic creation of a primary context to set it up manually for the camera we need.
    egui_global_settings.auto_create_primary_context = false;

    let (perp_gizmos, _) = config_store.config_mut::<PerpetualGizmos>();
    perp_gizmos.depth_bias = -1.0;

    self::reset(
        &mut commands,
        &cameras,
        &mut images,
        &mut handles,
        &configuration,
    );
}

pub fn update_render_settings(
    render_object_store: Res<RenderObjectStore>,
    mut render_settings_store: ResMut<RenderObjectSettingStore>,
) {
    // default overlays
    let default = |object: &Objects, label: &str| {
        matches!(
            (object, label),
            (Objects::InputMesh, "gray")
            | (Objects::InputMesh, "wireframe")
                // | (Objects::InputMesh, "x-loops")
                // | (Objects::InputMesh, "y-loops")
                // | (Objects::InputMesh, "z-loops")
                // | (Objects::InputMesh, "patches")
                | (Objects::InputMesh, "cuts")
                | (Objects::InputMesh, "routing failures")
                | (Objects::InputMesh, "invalid regions")
                // | (Objects::InputMesh, "virtual mesh debug")
                | (Objects::InputMesh, "uv long edges")
                // | (Objects::InputMesh, "uv patches")
                // | (Objects::Polycube, "gray")
                // | (Objects::Polycube, "patches")
                | (Objects::Polycube, "cuts")
                | (Objects::Polycube, "paths")
                | (Objects::Polycube, "flat paths")
                | (Objects::PolycubeMap, "colored")
                | (Objects::PolycubeMap, "triangles")
                | (Objects::QuadMesh, "gray")
                | (Objects::QuadMesh, "wireframe")
                | (Objects::ContractedMesh, "gray")
                | (Objects::ContractedMesh, "wireframe")
        )
    };

    if render_object_store.is_changed() {
        for (object, render_object) in &render_object_store.objects {
            let labels = render_object.labels.clone();
            let mut settings = render_settings_store
                .objects
                .get(object)
                .map_or_else(HashMap::new, |s| s.settings.clone());
            for feature_label in render_object.features.keys() {
                settings
                    .entry(feature_label.clone())
                    .or_insert(RenderFeatureSetting {
                        label: feature_label.clone(),
                        visible: default(object, feature_label),
                    });
            }

            render_settings_store
                .objects
                .insert(object.to_owned(), RenderObjectSetting { labels, settings });
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct MeshProperties {
    pub source: String,
    pub scale: f64,
    pub translation: Vector3D,
}

pub fn respawn_renders(
    mut commands: Commands,
    mut meshes: ResMut<Assets<bevy::mesh::Mesh>>,
    mut gizmos: ResMut<Assets<GizmoAsset>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut custom_materials: ResMut<Assets<ToonMaterial>>,
    configuration: Res<Configuration>,
    render_object_store: Res<RenderObjectStore>,
    render_settings_store: Res<RenderObjectSettingStore>,
    rendered_mesh_query: Query<Entity, With<Rendered>>,
) {
    if render_settings_store.is_changed() {
        info!("render_settings_store has been changed.");

        info!("Despawn all objects.");
        for entity in rendered_mesh_query.iter() {
            commands.entity(entity).despawn();
        }

        for material in custom_materials.iter().map(|x| x.0).collect_vec() {
            custom_materials.remove(material);
        }
        for material in materials.iter().map(|x| x.0).collect_vec() {
            materials.remove(material);
        }

        let flat_material = materials.add(StandardMaterial {
            unlit: true,
            ..default()
        });
        let toon_material = custom_materials.add(ToonMaterial {
            view_dir: Vec3::new(0.0, 0.0, 1.0),
        });
        let background_material = materials.add(StandardMaterial {
            base_color: bevy::prelude::Color::srgb_u8(
                configuration.clear_color[0],
                configuration.clear_color[1],
                configuration.clear_color[2],
            ),
            unlit: true,
            ..default()
        });

        // Go through render_object_store and spawn all objects (if they are visible).
        info!("Spawn all objects.");
        for &object in render_object_store.objects.keys() {
            let features = &render_object_store.objects.get(&object).unwrap().features;
            let settings = &render_settings_store.objects.get(&object).unwrap().settings;
            for feature in features.keys() {
                let visible = settings.get(feature).unwrap().visible;
                let assets = &features.get(feature).unwrap().assets;
                if visible {
                    for asset in assets {
                        match asset {
                            RenderAsset::Mesh(mesh) => {
                                let mesh_handle = MeshBundle::new(meshes.add(mesh.clone())).0;
                                match object {
                                    Objects::InputMesh => {
                                        commands.spawn((
                                            mesh_handle,
                                            MeshMaterial3d(toon_material.clone()),
                                            Transform {
                                                translation: Vec3::from(object),
                                                ..Default::default()
                                            },
                                            Rendered,
                                            MainMesh,
                                        ));
                                    }
                                    Objects::QuadMesh | Objects::ContractedMesh => {
                                        commands.spawn((
                                            mesh_handle,
                                            MeshMaterial3d(toon_material.clone()),
                                            Transform {
                                                translation: Vec3::from(object),
                                                ..Default::default()
                                            },
                                            Rendered,
                                        ));
                                    }
                                    Objects::PolycubeMap | Objects::Polycube => {
                                        commands.spawn((
                                            mesh_handle,
                                            MeshMaterial3d(flat_material.clone()),
                                            Transform {
                                                translation: Vec3::from(object),
                                                ..Default::default()
                                            },
                                            Rendered,
                                        ));
                                    }
                                }
                            }
                            RenderAsset::Gizmo(gizmo) => {
                                let gizmo_handle =
                                    GizmoBundle::new(gizmos.add(gizmo.0.clone()), gizmo.1, gizmo.2)
                                        .0;
                                commands.spawn((
                                    (
                                        gizmo_handle,
                                        Transform {
                                            translation: Vec3::from(object),
                                            ..Default::default()
                                        },
                                    ),
                                    Rendered,
                                ));
                            }
                        }
                    }
                }
            }
            // Spawning covers such that the objects are view-blocked.
            // On render layer 1 so only the dock cameras show them; the
            // screenshot camera (layer 0) keeps a transparent background.
            commands.spawn((
                Mesh3d(meshes.add(Sphere::new(400.))),
                MeshMaterial3d(background_material.clone()),
                Transform::from_translation(Vec3::from(object)),
                Rendered,
                RenderLayers::layer(1),
            ));
        }
    }
}

pub fn update(
    ui_resource: Res<UiResource>,
    mut custom_materials: ResMut<Assets<ToonMaterial>>,
    window: Single<&Window>,
    mut main_camera: Query<(&LookTransform, &Transform, &mut Camera), With<Controller>>,
    mut other_cameras: Query<
        (&mut Transform, &mut Projection, &mut Camera, &CameraFor),
        (Without<Controller>, Without<ScreenshotCamera>),
    >,
    mut screenshot_cameras: Query<
        (&mut Transform, &mut Projection),
        (With<ScreenshotCamera>, Without<Controller>),
    >,
    screenshot_override: Res<ScreenshotCameraOverride>,
) {
    let (_, main_transform, mut main_camera) = main_camera.single_mut().unwrap();

    let (_, node_index, _) = ui_resource.tree.find_tab(&Objects::InputMesh).unwrap();
    let main_surface = ui_resource.tree.main_surface().clone();
    let main_node = main_surface.index(node_index);
    let main_surface_viewport = match main_node {
        egui_dock::Node::Leaf(LeafNode { viewport, .. }) => *viewport,
        _ => unreachable!(),
    };

    let viewport_width = main_surface_viewport.max[0] - main_surface_viewport.min[0];
    let viewport_height = main_surface_viewport.max[1] - main_surface_viewport.min[1];
    let scale_factor = window.scale_factor() as f32;

    let physical_x = (main_surface_viewport.min[0] * scale_factor)
        .round()
        .max(0.0) as u32;
    let physical_y = (main_surface_viewport.min[1] * scale_factor)
        .round()
        .max(0.0) as u32;
    let physical_width = (viewport_width * scale_factor).round().max(0.0) as u32;
    let physical_height = (viewport_height * scale_factor).round().max(0.0) as u32;

    if window.physical_size().x == 0
        || window.physical_size().y == 0
        || physical_width == 0
        || physical_height == 0
        || viewport_width.is_infinite()
        || viewport_height.is_infinite()
    {
        main_camera.is_active = false;
    } else {
        main_camera.is_active = true;
        main_camera.viewport = Some(Viewport {
            physical_position: UVec2 {
                x: physical_x,
                y: physical_y,
            },
            physical_size: UVec2 {
                x: physical_width,
                y: physical_height,
            },
            ..Default::default()
        });
    }

    let normalized_translation = main_transform.translation - Vec3::from(Objects::InputMesh);
    let normalized_rotation = main_transform.rotation;

    let distance = normalized_translation.length();

    for (mut sub_transform, mut sub_projection, _sub_camera, sub_object) in &mut other_cameras {
        sub_transform.translation = normalized_translation + Vec3::from(sub_object.0);
        sub_transform.rotation = normalized_rotation;
        if let Projection::Orthographic(orthographic) = sub_projection.as_mut() {
            orthographic.scaling_mode = ScalingMode::FixedVertical {
                viewport_height: distance,
            };
        }
    }

    // Find which object the largest panel is currently showing.
    let largest_object = {
        let mut best = Objects::InputMesh;
        let mut best_area: f32 = 0.0;
        for node in ui_resource.tree.main_surface().iter() {
            if let egui_dock::Node::Leaf(leaf) = node {
                let area = leaf.viewport.width() * leaf.viewport.height();
                if area > best_area {
                    best_area = area;
                    if let Some(tab) = leaf.tabs.get(leaf.active.0) {
                        best = *tab;
                    }
                }
            }
        }
        best
    };

    // During a comprehensive capture the screenshot camera is pointed at a
    // specific object per view; otherwise it follows the largest panel.
    let screenshot_object = screenshot_override.0.unwrap_or(largest_object);

    for (mut transform, mut projection) in &mut screenshot_cameras {
        transform.translation = normalized_translation + Vec3::from(screenshot_object);
        transform.rotation = normalized_rotation;
        if matches!(screenshot_object, Objects::PolycubeMap | Objects::Polycube) {
            let mut proj = OrthographicProjection::default_3d();
            proj.scaling_mode = ScalingMode::FixedVertical {
                viewport_height: distance,
            };
            *projection = Projection::Orthographic(proj);
        } else {
            *projection = Projection::default();
        }
    }

    for material in custom_materials.iter_mut() {
        // current location of the camera, to (0, 0, 0)
        material.1.view_dir = Vec3::new(
            normalized_translation.x,
            normalized_translation.y,
            normalized_translation.z,
        )
        .normalize();
    }
}

pub fn refresh(solution: &Solution, configuration: &Configuration) -> RenderObjectStore {
    let mut render_object_store = RenderObjectStore::default();
    for object in all::<Objects>() {
        match object {
            // Adds the QUAD MESH to our RenderObjectStore, it has:
            // mesh with black faces
            // mesh with colored faces
            // wireframe (the quads)
            Objects::QuadMesh => {
                if let Some(quad) = &solution.quad {
                    let mut default_color_map = HashMap::new();
                    for face_id in quad.quad_mesh.face_ids() {
                        default_color_map.insert(face_id, colors::LIGHT_GRAY);
                    }

                    let (scale, translation) = quad.quad_mesh.scale_translation();
                    let mut color_map = HashMap::new();
                    for face_id in quad.quad_mesh.face_ids() {
                        let normal = quad.quad_mesh_polycube.normal(face_id);
                        let color = colors::from_direction(
                            to_principal_direction(normal).0,
                            Some(Perspective::Primal),
                            None,
                        );
                        color_map
                            .insert(face_id, [color[0] as f32, color[1] as f32, color[2] as f32]);
                    }

                    let mut gizmos_paths = GizmoAsset::new();
                    let mut gizmos_flat_paths = GizmoAsset::new();
                    if let (Some(_), Some(_)) = (&solution.layout, &solution.polycube) {
                        let color = colors::GRAY;
                        let c = bevy::color::Color::srgb(color[0], color[1], color[2]);

                        let mut irregular_vertices = HashSet::new();
                        for vert_id in quad.quad_mesh.vert_ids() {
                            // Get the faces around the vertex
                            let faces = quad.quad_mesh.faces(vert_id);
                            // Get the labels of the faces around
                            let labels = faces
                                .map(|face_id| {
                                    to_principal_direction(quad.quad_mesh_polycube.normal(face_id))
                                        .0
                                })
                                .collect::<HashSet<_>>();
                            // If 3+ labels, its irregular
                            if labels.len() >= 3 {
                                irregular_vertices.insert(vert_id);
                            }
                        }

                        // Get all edges going out irregular vertices
                        let mut irregular_edges = HashSet::new();
                        for &vert_id in &irregular_vertices {
                            for edge_id in quad.quad_mesh.edges(vert_id) {
                                let mut next_twin_next = quad
                                    .quad_mesh
                                    .next(quad.quad_mesh.twin(quad.quad_mesh.next(edge_id)));
                                while !irregular_edges.contains(&next_twin_next) {
                                    irregular_edges.insert(next_twin_next);
                                    next_twin_next = quad.quad_mesh.next(
                                        quad.quad_mesh.twin(quad.quad_mesh.next(next_twin_next)),
                                    );
                                }
                            }
                        }

                        // Draw all irregular edges
                        for edge_id in irregular_edges {
                            let Some([f1, f2]) = quad.quad_mesh.faces(edge_id).collect_array::<2>()
                            else {
                                panic!("Expected two faces for edge {edge_id:?}");
                            };
                            let n1 = quad.quad_mesh_polycube.normal(f1);
                            let n2 = quad.quad_mesh_polycube.normal(f2);
                            let Some([e1, e2]) =
                                quad.quad_mesh.vertices(edge_id).collect_array::<2>()
                            else {
                                panic!("Expected two vertices for edge {edge_id:?}");
                            };
                            let u = quad.quad_mesh.position(e1);
                            let v = quad.quad_mesh.position(e2);
                            let u_transformed = world_to_view(u, translation, scale);
                            let v_transformed = world_to_view(v, translation, scale);
                            if n1 == n2 {
                                gizmos_flat_paths.line(u_transformed, v_transformed, c);
                            } else {
                                gizmos_paths.line(u_transformed, v_transformed, c);
                            }
                        }
                    }

                    render_object_store.add_object(
                        object,
                        RenderObject::default()
                            .mesh(&quad.quad_mesh, &default_color_map, "gray")
                            .mesh(&quad.quad_mesh, &color_map, "colored")
                            .gizmo(
                                quad.quad_mesh.gizmos(colors::GRAY),
                                1.0,
                                -0.001,
                                "wireframe",
                            )
                            .gizmo(gizmos_paths, 4., -0.0001, "paths")
                            .gizmo(gizmos_flat_paths, 3., -0.00011, "flat paths")
                            .to_owned(),
                    );
                }
            }
            Objects::Polycube => {
                if let Some(polycube) = &solution.polycube {
                    let mut gray_color_map = HashMap::new();
                    let mut black_color_map = HashMap::new();
                    let mut colored_color_map = HashMap::new();
                    let mut gizmos_xloops = GizmoAsset::new();
                    let mut gizmos_yloops = GizmoAsset::new();
                    let mut gizmos_zloops = GizmoAsset::new();

                    let (scale, translation) = polycube.structure.scale_translation();

                    for face_id in polycube.structure.face_ids() {
                        let normal = polycube.structure.normal(face_id);

                        black_color_map.insert(face_id, colors::BLACK);
                        gray_color_map.insert(face_id, colors::LIGHT_GRAY);
                        colored_color_map.insert(face_id, {
                            colors::from_direction(
                                to_principal_direction(normal).0,
                                Some(Perspective::Primal),
                                None,
                            )
                        });

                        // draw loops
                        let Some([e1, e2, e3, e4]) =
                            polycube.structure.edges(face_id).collect_array::<4>()
                        else {
                            panic!("Expected four edges for face {face_id:?}");
                        };
                        let edge1_pos = polycube.structure.position(e1);
                        let edge1_pos_view = world_to_view(edge1_pos, translation, scale);
                        let edge2_pos = polycube.structure.position(e2);
                        let edge2_pos_view = world_to_view(edge2_pos, translation, scale);
                        let edge3_pos = polycube.structure.position(e3);
                        let edge3_pos_view = world_to_view(edge3_pos, translation, scale);
                        let edge4_pos = polycube.structure.position(e4);
                        let edge4_pos_view = world_to_view(edge4_pos, translation, scale);

                        // loop 1, from edge 1 to edge 3
                        let dir = to_principal_direction(edge2_pos - edge4_pos).0;
                        let c = colors::to_bevy(colors::from_direction(
                            dir,
                            Some(Perspective::Dual),
                            None,
                        ));
                        match dir {
                            PrincipalDirection::X => {
                                gizmos_xloops.line(edge1_pos_view, edge3_pos_view, c)
                            }
                            PrincipalDirection::Y => {
                                gizmos_yloops.line(edge1_pos_view, edge3_pos_view, c)
                            }
                            PrincipalDirection::Z => {
                                gizmos_zloops.line(edge1_pos_view, edge3_pos_view, c)
                            }
                        }

                        // loop 2, from edge 2 to edge 4
                        let dir = to_principal_direction(edge1_pos - edge3_pos).0;
                        let c = colors::to_bevy(colors::from_direction(
                            dir,
                            Some(Perspective::Dual),
                            None,
                        ));

                        match dir {
                            PrincipalDirection::X => {
                                gizmos_xloops.line(edge2_pos_view, edge4_pos_view, c)
                            }
                            PrincipalDirection::Y => {
                                gizmos_yloops.line(edge2_pos_view, edge4_pos_view, c)
                            }
                            PrincipalDirection::Z => {
                                gizmos_zloops.line(edge2_pos_view, edge4_pos_view, c)
                            }
                        }
                    }

                    let mut gizmos_paths = GizmoAsset::new();
                    let mut gizmos_flat_paths = GizmoAsset::new();

                    let color = colors::GRAY;
                    let c = bevy::color::Color::srgb(color[0], color[1], color[2]);

                    for pedge_id in polycube.structure.edge_ids() {
                        let f1 = polycube.structure.normal(polycube.structure.face(pedge_id));
                        let f2 = polycube
                            .structure
                            .normal(polycube.structure.face(polycube.structure.twin(pedge_id)));
                        let Some([e1, e2]) =
                            polycube.structure.vertices(pedge_id).collect_array::<2>()
                        else {
                            panic!("Expected two vertices for edge {pedge_id:?}");
                        };
                        let u = polycube.structure.position(e1);
                        let v = polycube.structure.position(e2);
                        let u_transformed = world_to_view(u, translation, scale);
                        let v_transformed = world_to_view(v, translation, scale);
                        gizmos_flat_paths.line(u_transformed, v_transformed, c);
                        if f1 != f2 {
                            gizmos_paths.line(u_transformed, v_transformed, c);
                        }
                    }

                    let mut render_obj = RenderObject::default();
                    render_obj
                        .mesh(&polycube.structure, &black_color_map, "black")
                        .mesh(&polycube.structure, &gray_color_map, "gray")
                        .mesh(&polycube.structure, &colored_color_map, "colored")
                        .gizmo(gizmos_xloops, 6., -0.001, "x-loops")
                        .gizmo(gizmos_yloops, 6., -0.0011, "y-loops")
                        .gizmo(gizmos_zloops, 6., -0.00111, "z-loops")
                        .gizmo(gizmos_paths, 7., -0.001, "paths")
                        .gizmo(gizmos_flat_paths, 5., -0.0011, "flat paths");

                    // Add polycube patch visualization if polycube skeleton is available.
                    if let Some(polycube_skeleton) = solution
                        .skeleton
                        .as_ref()
                        .and_then(|s| s.polycube_skeleton())
                    {
                        let patch_mesh = create_polycube_patch_mesh(
                            polycube_skeleton,
                            &polycube.structure,
                            translation,
                            scale,
                        );
                        render_obj.bevy_mesh(patch_mesh, "patches");

                        let boundary_gizmos = create_polycube_patch_boundary_gizmos(
                            polycube_skeleton,
                            &polycube.structure,
                            translation,
                            scale,
                        );
                        render_obj.gizmo(boundary_gizmos, 1.0, -0.00016, "patches");
                    }

                    render_object_store.add_object(object, render_obj);
                }
            }
            // Adds the POLYCUBE to our RenderObjectStore, it has:
            // mesh with black faces
            // mesh with colored faces
            // quads mapped on the polycube
            // triangles mapped on the polycube
            Objects::PolycubeMap => {
                if let Some(quad) = &solution.quad {
                    let mut color_map = HashMap::new();
                    for face_id in quad.quad_mesh_polycube.face_ids() {
                        let color = colors::LIGHT_GRAY;
                        color_map.insert(face_id, color);
                    }

                    let mut render_obj = RenderObject::default();
                    render_obj
                        .mesh(&quad.quad_mesh_polycube, &color_map, "colored")
                        .gizmo(
                            quad.quad_mesh_polycube.gizmos(colors::GRAY),
                            2.,
                            -0.01,
                            "quads",
                        )
                        .gizmo(
                            quad.triangle_mesh_polycube.gizmos(colors::GRAY),
                            2.,
                            -0.01,
                            "triangles",
                        );

                    render_object_store.add_object(object, render_obj);
                }
            }
            Objects::InputMesh => {
                let input = solution.mesh_ref.as_ref();
                let (scale, translation) = input.scale_translation();
                let mut gizmos_xloops = GizmoAsset::new();
                let mut gizmos_yloops = GizmoAsset::new();
                let mut gizmos_zloops = GizmoAsset::new();
                let mut gizmos_paths = GizmoAsset::new();
                let mut gizmos_flat_paths = GizmoAsset::new();
                let mut gizmos_raw_skeleton = GizmoAsset::new();
                let mut gizmos_cleaned_skeleton = GizmoAsset::new();
                let mut gizmos_cleaned_skeleton_gray = GizmoAsset::new();
                let mut gizmos_failed_surgery_skeleton = GizmoAsset::new();
                let mut patch_mesh: Option<bevy::mesh::Mesh> = None;
                let mut raw_patch_mesh: Option<bevy::mesh::Mesh> = None;
                let mut failed_surgery_patch_mesh: Option<bevy::mesh::Mesh> = None;
                let mut failed_surgery_face_mesh: Option<bevy::mesh::Mesh> = None;
                let mut granulated_mesh = &mehsh::prelude::Mesh::<INPUT>::default();
                let mut default_color_map = HashMap::new();
                let mut black_color_map = HashMap::new();
                for face_id in input.face_ids() {
                    default_color_map.insert(face_id, colors::LIGHT_GRAY);
                    black_color_map.insert(face_id, colors::BLACK);
                }
                let mut color_map_segmentation = HashMap::new();
                let mut color_map_alignment = HashMap::new();

                let color = colors::GRAY;
                let c = bevy::color::Color::srgb(color[0], color[1], color[2]);

                for (lewp_id, lewp) in &solution.loops {
                    let direction = solution.loop_to_direction(lewp_id);

                    let mut positions = vec![];
                    for u in [lewp.edges.clone(), vec![lewp.edges[0]], vec![lewp.edges[1]]].concat()
                    {
                        let ut = transform_coordinates(input.position(u), translation, scale);
                        // gizmos_loop.line(line.u, line.v, c);
                        positions.push(vector3d_to_vec3(ut));
                    }

                    let color = colors::from_direction(
                        direction,
                        Some(Perspective::Dual),
                        Some(Orientation::Forwards),
                    );
                    let c = bevy::color::Color::srgb(color[0], color[1], color[2]);
                    match direction {
                        PrincipalDirection::X => gizmos_xloops.linestrip(positions, c),
                        PrincipalDirection::Y => gizmos_yloops.linestrip(positions, c),
                        PrincipalDirection::Z => gizmos_zloops.linestrip(positions, c),
                    }
                }

                if let (Some(lay), Some(polycube)) = (&solution.layout, &solution.polycube) {
                    granulated_mesh = &lay.granulated_mesh;

                    for (&pedge_id, path) in &lay.edge_to_path {
                        let f1 = polycube.structure.normal(polycube.structure.face(pedge_id));
                        let f2 = polycube
                            .structure
                            .normal(polycube.structure.face(polycube.structure.twin(pedge_id)));
                        for vertexpair in path.windows(2) {
                            if granulated_mesh
                                .edge_between_verts(vertexpair[0], vertexpair[1])
                                .is_none()
                            {
                                println!(
                                    "Edge between {:?} and {:?} does not exist",
                                    vertexpair[0], vertexpair[1]
                                );
                                continue;
                            }
                            let edge_id = granulated_mesh
                                .edge_between_verts(vertexpair[0], vertexpair[1])
                                .unwrap()
                                .0;
                            let Some([e1, e2]) =
                                granulated_mesh.vertices(edge_id).collect_array::<2>()
                            else {
                                panic!("Expected two vertices for edge {edge_id:?}");
                            };
                            let u = granulated_mesh.position(e1);
                            let v = granulated_mesh.position(e2);
                            let u_transformed = world_to_view(u, translation, scale);
                            let v_transformed = world_to_view(v, translation, scale);
                            gizmos_flat_paths.line(u_transformed, v_transformed, c);
                            if f1 != f2 {
                                gizmos_paths.line(u_transformed, v_transformed, c);
                            }
                        }
                    }

                    for &face_id in &polycube.structure.face_ids() {
                        let normal = (polycube.structure.normal(face_id) as Vector3D).normalize();

                        let (dir, side) = to_principal_direction(normal);
                        let color =
                            colors::from_direction(dir, Some(Perspective::Primal), Some(side));
                        for &triangle_id in &lay.face_to_patch[&face_id].faces {
                            color_map_segmentation.insert(triangle_id, color);
                        }
                    }

                    for triangle_id in lay.granulated_mesh.face_ids() {
                        if let Some(&score) = solution
                            .layout
                            .as_ref()
                            .unwrap()
                            .alignment_per_triangle
                            .get(&triangle_id)
                        {
                            color_map_alignment.insert(
                                triangle_id,
                                colors::map(score as f32, &colors::SCALE_MAGMA),
                            );
                        } else {
                            color_map_alignment.insert(triangle_id, colors::SNOEP_YELLOW);
                        }
                    }
                }

                let features =
                    dualcube::feature::feature_extraction(input, std::f64::consts::FRAC_PI_3, 1);
                let mut gizmos_features = GizmoAsset::new();
                let cs = [
                    colors::to_bevy(colors::from_direction(PrincipalDirection::X, None, None)),
                    colors::to_bevy(colors::from_direction(PrincipalDirection::Y, None, None)),
                    colors::to_bevy(colors::from_direction(PrincipalDirection::Z, None, None)),
                ];
                for i in 0..3 {
                    let color = cs[i];
                    let feature_edges = &features[i];
                    for &edge_id in feature_edges {
                        let Some([v1, v2]) = input.vertices(edge_id).collect_array::<2>() else {
                            panic!("Expected two vertices for edge {edge_id:?}");
                        };
                        let u = input.position(v1);
                        let v = input.position(v2);
                        let u_transformed = world_to_view(u, translation, scale);
                        let v_transformed = world_to_view(v, translation, scale);
                        gizmos_features.line(u_transformed, v_transformed, color);
                    }
                }

                // Visualize skeleton(s) if available. Similarly, show labelling if available.
                if let Some(skeleton_data) = &solution.skeleton {
                    if let Some(curve_skeleton) = skeleton_data.curve_skeleton() {
                        gizmos_raw_skeleton =
                            create_skeleton_gizmos(curve_skeleton, translation, scale);
                        raw_patch_mesh =
                            Some(create_patch_mesh(curve_skeleton, input, translation, scale));
                    }
                    // Diagnostic: when connectivity surgery couldn't reduce
                    // away every face, render the partial skeleton AND the
                    // remaining face triangles in red so the user can see
                    // exactly what got stuck.
                    if let Some(failed) = skeleton_data.failed_surgery() {
                        gizmos_failed_surgery_skeleton = create_failed_surgery_skeleton_gizmos(
                            &failed.skeleton,
                            &failed.remaining_face_positions,
                            translation,
                            scale,
                        );
                        failed_surgery_patch_mesh = Some(create_patch_mesh(
                            &failed.skeleton,
                            input,
                            translation,
                            scale,
                        ));
                        failed_surgery_face_mesh = Some(create_failed_surgery_face_mesh(
                            &failed.remaining_face_positions,
                            translation,
                            scale,
                        ));
                    }
                    if let Some(cleaned_skeleton) = skeleton_data.cleaned_skeleton() {
                        // Check for labeled skeleton
                        if let Some(labeled_skeleton) = skeleton_data.labeled_skeleton() {
                            // Add coloring based on labels
                            gizmos_cleaned_skeleton = create_labeled_skeleton_gizmos(
                                labeled_skeleton,
                                translation,
                                scale,
                            );
                        } else {
                            // Labelling failed, just show cleaned skeleton
                            gizmos_cleaned_skeleton =
                                create_skeleton_gizmos(cleaned_skeleton, translation, scale);
                        }
                        gizmos_cleaned_skeleton_gray =
                            create_skeleton_gizmos(cleaned_skeleton, translation, scale);

                        patch_mesh = Some(create_patch_mesh(
                            cleaned_skeleton,
                            input,
                            translation,
                            scale,
                        ));
                    }
                }

                let mut granulated_mesh_gizmos = GizmoAsset::new();
                if let Some(layout) = &solution.layout {
                    granulated_mesh_gizmos = layout.granulated_mesh.gizmos(colors::GRAY);
                }

                // Visualize the vector fields
                let mut gizmos_xfield: GizmoAsset = GizmoAsset::new();
                let mut gizmos_yfield: GizmoAsset = GizmoAsset::new();
                let mut gizmos_zfield: GizmoAsset = GizmoAsset::new();
                if let Some(fields) = &solution.fields {
                    let field_scale = 0.01;

                    for (&vert_id, &vector_id) in &fields.field_x.map {
                        // println!("Drawing vector at vert_id: {:?}", vert_id);
                        let vector = fields.field_x.vectors.get(vector_id).unwrap();
                        let vert_pos = input.position(vert_id);
                        let start = world_to_view(vert_pos, translation, scale);
                        let end_world = vert_pos + vector.normalize() * field_scale;
                        let end = world_to_view(end_world, translation, scale);
                        gizmos_xfield.arrow(
                            start,
                            end,
                            colors::to_bevy(colors::from_direction(
                                PrincipalDirection::X,
                                None,
                                None,
                            )),
                        );
                    }

                    for (&vert_id, &vector_id) in &fields.field_y.map {
                        let vector = fields.field_y.vectors.get(vector_id).unwrap();
                        let vert_pos = input.position(vert_id);
                        let start = world_to_view(vert_pos, translation, scale);
                        let end_world = vert_pos + vector.normalize() * field_scale;
                        let end = world_to_view(end_world, translation, scale);
                        gizmos_yfield.arrow(
                            start,
                            end,
                            colors::to_bevy(colors::from_direction(
                                PrincipalDirection::Y,
                                None,
                                None,
                            )),
                        );
                    }

                    for (&vert_id, &vector_id) in &fields.field_z.map {
                        let vector = fields.field_z.vectors.get(vector_id).unwrap();
                        let vert_pos = input.position(vert_id);
                        let start = world_to_view(vert_pos, translation, scale);
                        let end_world = vert_pos + vector.normalize() * field_scale;
                        let end = world_to_view(end_world, translation, scale);
                        gizmos_zfield.arrow(
                            start,
                            end,
                            colors::to_bevy(colors::from_direction(
                                PrincipalDirection::Z,
                                None,
                                None,
                            )),
                        );
                    }
                }

                // Visualize principal curvature (direction + resolution-robust magnitude)
                let mut gizmos_curvature_max = GizmoAsset::new();
                let mut gizmos_curvature_min = GizmoAsset::new();

                // Tunables for glyph sizing
                let s: f64 = 2.0; // sensitivity of length to curvature (dimensionless)
                let base_frac: f64 = 0.5; // glyph base length as fraction of local edge scale
                let min_h: f64 = 1e-9; // avoid divide-by-zero / degenerate neighborhoods

                for vert_id in input.vert_ids() {
                    let v = input.position(vert_id);

                    // Tangent frame (force orthonormal)
                    let (t1_raw, _t2_raw, n_raw) = input.tangent_frame(vert_id);
                    let n = n_raw.normalize();
                    let t1 = (t1_raw - n * t1_raw.dot(&n)).normalize();
                    let t2 = n.cross(&t1); // guarantees orthonormal + right-handed

                    // ---------
                    // Local scale h(v): mean 1-ring edge length (resolution proxy)
                    // ---------
                    let mut h_sum = 0.0;
                    let mut h_cnt = 0.0;
                    for neighbor_id in input.neighbors(vert_id) {
                        let p = input.position(neighbor_id);
                        h_sum += (p - v).norm();
                        h_cnt += 1.0;
                    }
                    if h_cnt < 1.0 {
                        continue;
                    }
                    let h = (h_sum / h_cnt).max(min_h);

                    // Normal equations for 4 unknowns: [a11 a12 a21 a22]
                    let mut ata = nalgebra::Matrix4::<f64>::zeros();
                    let mut atb = nalgebra::Vector4::<f64>::zeros();

                    for neighbor_id in input.neighbors(vert_id) {
                        let p = input.position(neighbor_id);

                        let e = p - v;
                        let e_t = e - n * e.dot(&n);
                        let len2 = e_t.dot(&e_t);
                        if len2 < 1e-12 {
                            continue;
                        }

                        let nj = input.normal(neighbor_id).normalize();
                        let dn = nj - n;
                        let dn_t = dn - n * dn.dot(&n);

                        let u = nalgebra::Vector2::new(t1.dot(&e_t), t2.dot(&e_t));
                        let dn2 = nalgebra::Vector2::new(t1.dot(&dn_t), t2.dot(&dn_t));

                        // Weight (simple + works well)
                        let alpha = (1.0 / len2).min(1e6);

                        // dn2.x = -(a11*u.x + a12*u.y)
                        // dn2.y = -(a21*u.x + a22*u.y)
                        let r1 = nalgebra::Vector4::new(u.x, u.y, 0.0, 0.0);
                        let r2 = nalgebra::Vector4::new(0.0, 0.0, u.x, u.y);

                        ata += alpha * (r1 * r1.transpose() + r2 * r2.transpose());
                        atb += alpha * (r1 * (-dn2.x) + r2 * (-dn2.y));
                    }

                    // Solve ATA * a = ATb (skip degenerate vertices)
                    let inv = match ata.try_inverse() {
                        Some(inv) => inv,
                        None => continue,
                    };
                    let a = inv * atb;

                    // Build 2x2 A (shape operator in tangent coordinates) and symmetrize
                    #[allow(non_snake_case)]
                    let mut A = nalgebra::Matrix2::new(a[0], a[1], a[2], a[3]);
                    A = 0.5 * (A + A.transpose());

                    // Eigen-decompose symmetric 2x2
                    let eig = nalgebra::SymmetricEigen::new(A);
                    let eigenvalues = eig.eigenvalues;
                    let eigenvectors = eig.eigenvectors;

                    // SymmetricEigen eigenvalues ascending: [0]=min, [1]=max
                    let k_min = eigenvalues[0];
                    let k_max = eigenvalues[1];

                    // Avoid column() inference issues: index directly
                    // let dmin_x: f64 = eigenvectors[(0, 0)];
                    // let dmin_y: f64 = eigenvectors[(1, 0)];
                    let dmax_x: f64 = eigenvectors[(0, 1)];
                    let dmax_y: f64 = eigenvectors[(1, 1)];

                    // Map 2D eigenvectors back to 3D tangent vectors
                    // let dir_min = (t1 * dmin_x + t2 * dmin_y).normalize();

                    let mut dir_max = (t1 * dmax_x + t2 * dmax_y).normalize();

                    // Enforce perfect tangency + orthogonality (cleaner field)
                    dir_max = (dir_max - n * dir_max.dot(&n)).normalize();
                    let dir_min = n.cross(&dir_max).normalize();

                    // -------------------------
                    // Sanity checks (debug-friendly)
                    // -------------------------
                    let eps_tangent = 1e-6;
                    let eps_ortho = 1e-5;
                    let eps_unit = 1e-5;

                    debug_assert!((dir_max.norm() - 1.0).abs() < eps_unit);
                    debug_assert!((dir_min.norm() - 1.0).abs() < eps_unit);
                    debug_assert!(dir_max.dot(&n).abs() < eps_tangent);
                    debug_assert!(dir_min.dot(&n).abs() < eps_tangent);
                    debug_assert!(dir_max.dot(&dir_min).abs() < eps_ortho);

                    // -------------------------
                    // Resolution-robust glyph lengths
                    //
                    // Curvature k has units 1/length. Make a dimensionless "bending per step":
                    //   c = |k| * h
                    // Then map to a length in world units using a saturating function.
                    // -------------------------
                    let cmax = k_max.abs() * h;
                    let cmin = k_min.abs() * h;

                    // Base glyph size in world units (tied to local resolution)
                    let base = base_frac * h;

                    // Saturating mapping to keep things readable
                    let lmax = base * (s * cmax).tanh();
                    let lmin = base * (s * cmin).tanh();

                    // Add to gizmos (convert endpoints to view space)
                    let v_transformed = world_to_view(v, translation, scale);

                    let u_max = v + dir_max * lmax;
                    let u_max_neg = v - dir_max * lmax;
                    gizmos_curvature_max.line(
                        v_transformed,
                        world_to_view(u_max, translation, scale),
                        colors::to_bevy(colors::from_direction(PrincipalDirection::X, None, None)),
                    );
                    gizmos_curvature_max.line(
                        v_transformed,
                        world_to_view(u_max_neg, translation, scale),
                        colors::to_bevy(colors::from_direction(PrincipalDirection::X, None, None)),
                    );

                    let u_min = v + dir_min * lmin;
                    let u_min_neg = v - dir_min * lmin;
                    gizmos_curvature_min.line(
                        v_transformed,
                        world_to_view(u_min, translation, scale),
                        colors::to_bevy(colors::from_direction(PrincipalDirection::Z, None, None)),
                    );
                    gizmos_curvature_min.line(
                        v_transformed,
                        world_to_view(u_min_neg, translation, scale),
                        colors::to_bevy(colors::from_direction(PrincipalDirection::Z, None, None)),
                    );
                }

                let mut render_obj = RenderObject::default();
                render_obj
                    .mesh(input, &default_color_map, "gray")
                    .mesh(input, &black_color_map, "black")
                    .mesh(granulated_mesh, &color_map_segmentation, "segmentation")
                    .mesh(granulated_mesh, &color_map_alignment, "alignment")
                    .gizmo(input.gizmos(colors::GRAY), 0.5, -0.00001, "wireframe")
                    .gizmo(gizmos_xloops, 3., -0.0001, "x-loops")
                    .gizmo(gizmos_yloops, 3., -0.00011, "y-loops")
                    .gizmo(gizmos_zloops, 3., -0.000111, "z-loops")
                    .gizmo(gizmos_paths, 4., -0.0001, "paths")
                    .gizmo(gizmos_flat_paths, 2., -0.00011, "flat paths")
                    // .mesh(input, &color_map_flag, "flag")
                    // .gizmo(gizmos_flag_paths, 2., -1e-4, "flag paths")
                    .gizmo(gizmos_features, 5., -0.00012, "features")
                    .gizmo(granulated_mesh_gizmos, 0.5, -0.00001, "refined wireframe")
                    .gizmo(gizmos_raw_skeleton, 25., -0.00014, "raw skeleton")
                    .gizmo(gizmos_cleaned_skeleton, 25., -0.00015, "cleaned skeleton")
                    .gizmo(gizmos_cleaned_skeleton_gray, 25., -0.000155, "cleaned skeleton (gray)")
                    .gizmo(
                        gizmos_failed_surgery_skeleton,
                        25.,
                        -0.000145,
                        "failed surgery skeleton",
                    );

                // TODO: remove later
                if let Some(crossings) = &solution.loop_crossings {
                    let crossing_gizmos =
                        create_crossing_point_gizmos(crossings, input, translation, scale);
                    render_obj.gizmo(crossing_gizmos, 25., -0.00016, "loop crossings");
                }

                // Routing diagnostics: dropped loops + failed-segment markers, so the user
                // can see where loop generation got stuck.
                if let Some(diagnostics) = &solution.routing_diagnostics {
                    let diag_gizmos =
                        create_routing_diagnostics_gizmos(diagnostics, input, translation, scale);
                    render_obj.gizmo(diag_gizmos, 25., -0.00018, "routing failures");
                }

                // Invalid loop regions (Property 3 violations): malformed regions surfaced when
                // dual reconstruction fails with "Invalid face boundary".
                if !solution.invalid_regions.is_empty() {
                    let invalid_gizmos =
                        create_invalid_region_gizmos(&solution.invalid_regions, input, translation, scale);
                    render_obj.gizmo(invalid_gizmos, 25., -0.000181, "invalid regions");
                }

                let mut patch_convexity_mesh: Option<bevy::mesh::Mesh> = None;
                if let Some(pm) = patch_mesh {
                    render_obj.bevy_mesh(pm, "patches");
                }
                // Patch boundary edges for cleaned skeleton
                if let Some(cleaned) = solution
                    .skeleton
                    .as_ref()
                    .and_then(|s| s.cleaned_skeleton())
                {
                    let boundary_gizmos =
                        create_patch_boundary_gizmos(cleaned, input, translation, scale);
                    render_obj.gizmo(boundary_gizmos, 1.0, -0.00016, "patches");
                }
                // Raw patch mesh and boundary edges from raw skeleton
                if let Some(rpm) = raw_patch_mesh {
                    render_obj.bevy_mesh(rpm, "raw patches");
                }
                if let Some(raw) = solution
                    .skeleton
                    .as_ref()
                    .and_then(|s| s.curve_skeleton())
                {
                    let raw_boundary_gizmos =
                        create_patch_boundary_gizmos(raw, input, translation, scale);
                    render_obj.gizmo(raw_boundary_gizmos, 1.0, -0.000165, "raw patches");
                }
                // Failed-surgery patch overlay
                if let Some(fpm) = failed_surgery_patch_mesh {
                    render_obj.bevy_mesh(fpm, "failed surgery patches");
                }
                // Failed-surgery face overlay: the literal triangles still
                // in the stuck simplicial complex, rendered in red.
                if let Some(ffm) = failed_surgery_face_mesh {
                    render_obj.bevy_mesh(ffm, "failed surgery faces (red)");
                }
                if let Some(failed) = solution
                    .skeleton
                    .as_ref()
                    .and_then(|s| s.failed_surgery())
                {
                    let failed_boundary_gizmos =
                        create_patch_boundary_gizmos(&failed.skeleton, input, translation, scale);
                    render_obj.gizmo(
                        failed_boundary_gizmos,
                        1.0,
                        -0.000167,
                        "failed surgery patches",
                    );
                }
                if let Some(cleaned) = &solution
                    .skeleton
                    .as_ref()
                    .and_then(|s| s.cleaned_skeleton())
                {
                    // Build convexity overlay mesh
                    patch_convexity_mesh = Some(create_patch_convexity_mesh(
                        cleaned,
                        input,
                        translation,
                        scale,
                    ));
                }
                if let Some(pc) = patch_convexity_mesh {
                    render_obj.bevy_mesh(pc, "patch convexity");
                }

                render_obj
                    .gizmo(gizmos_xfield, 1., -0.0001, "x-vector field")
                    .gizmo(gizmos_yfield, 1., -0.00011, "y-vector field")
                    .gizmo(gizmos_zfield, 1., -0.000111, "z-vector field")
                    .gizmo(
                        gizmos_curvature_max,
                        2.,
                        -0.00012,
                        "maximum principal curvature",
                    )
                    .gizmo(
                        gizmos_curvature_min,
                        2.,
                        -0.00013,
                        "minimum principal curvature",
                    );

                render_object_store.add_object(object, render_obj);
            }
            // Adds the CONTRACTED MESH to our RenderObjectStore, it has:
            // - gray mesh
            // - wireframe
            // - raw skeleton
            // - cleaned skeleton
            Objects::ContractedMesh => {
                if let Some(skeleton_data) = &solution.skeleton {
                    let contracted = skeleton_data.contraction_mesh();
                    let (scale, translation) = contracted.scale_translation();
                    let mut default_color_map = HashMap::new();
                    for face_id in contracted.face_ids() {
                        default_color_map.insert(face_id, colors::LIGHT_GRAY);
                    }

                    // Build skeleton gizmos
                    // Raw skeleton
                    let raw_gizmos_skeleton = skeleton_data
                        .curve_skeleton()
                        .map(|skel| create_skeleton_gizmos(skel, translation, scale))
                        .unwrap_or_else(GizmoAsset::new);
                    // Cleaned skeleton
                    let cleaned_gizmos_skeleton = skeleton_data
                        .cleaned_skeleton()
                        .map(|skel| create_skeleton_gizmos(skel, translation, scale))
                        .unwrap_or_else(GizmoAsset::new);

                    render_object_store.add_object(
                        object,
                        RenderObject::default()
                            .mesh(contracted, &default_color_map, "gray")
                            .gizmo(
                                contracted.gizmos(colors::WHITE),
                                0.75,
                                -0.00001,
                                "wireframe",
                            )
                            .gizmo(raw_gizmos_skeleton, 25., -0.00014, "raw skeleton")
                            .gizmo(cleaned_gizmos_skeleton, 25., -0.00015, "cleaned skeleton")
                            .to_owned(),
                    );
                }
            }
        }
    }

    render_object_store
}

pub fn world_to_view(v: Vector3D, translation: Vector3D, scale: f64) -> Vec3 {
    let vt = transform_coordinates(v, translation, scale);
    Vec3::new(vt.x as f32, vt.y as f32, vt.z as f32)
}

pub fn view_to_world(v: Vec3, translation: Vector3D, scale: f64) -> Vector3D {
    let v_world = Vector3D::new(v.x as f64, v.y as f64, v.z as f64);
    invert_transform_coordinates(v_world, translation, scale)
}

pub fn take_screenshot(mut commands: Commands, handle: Option<Res<ScreenshotHandle>>) {
    let Some(path) = PENDING_SCREENSHOT.lock().unwrap().take() else {
        return;
    };
    let Some(handle) = handle else {
        return;
    };
    commands
        .spawn(Screenshot::image(handle.0.clone()))
        .observe(move |screenshot: On<ScreenshotCaptured>| {
            let img = screenshot.image.clone();
            match img.try_into_dynamic() {
                Ok(dyn_img) => {
                    let rgba = dyn_img.to_rgba8();
                    match rgba.save(&path) {
                        Ok(()) => info!("Screenshot saved to {}", path.display()),
                        Err(e) => error!("Failed to save screenshot: {e}"),
                    }
                }
                Err(e) => error!("Failed to convert screenshot image: {e:?}"),
            }
        });
}

/// Sets exactly the listed features of `object` visible (all others hidden),
/// leaving other objects' settings untouched. Mutating the store triggers
/// [`respawn_renders`] so the next frame reflects the change.
fn apply_view_config(store: &mut RenderObjectSettingStore, object: Objects, visible: &[&str]) {
    if let Some(setting) = store.objects.get_mut(&object) {
        for (label, feature) in setting.settings.iter_mut() {
            feature.visible = visible.contains(&label.as_str());
        }
    }
}

/// Drives the comprehensive capture state machine. Picks up requests from
/// [`PENDING_COMPREHENSIVE`], then for each of the four views: applies the
/// view's render config + camera override, waits for the scene to settle,
/// captures the offscreen image, and waits for the tile to be processed. On
/// finish it restores the original render settings.
pub fn drive_comprehensive_capture(
    mut commands: Commands,
    mut state: ResMut<ComprehensiveState>,
    mut settings: ResMut<RenderObjectSettingStore>,
    mut cam_override: ResMut<ScreenshotCameraOverride>,
    screenshot_handle: Option<Res<ScreenshotHandle>>,
    time: Res<Time>,
) {
    if !state.active {
        let Some(request) = PENDING_COMPREHENSIVE.lock().unwrap().take() else {
            return;
        };
        if screenshot_handle.is_none() {
            return;
        }

        // Save mode: create the output directory and write stats.txt up front.
        if request.mode == ComprehensiveMode::Save {
            if let Some(dir) = &request.save_dir {
                if let Err(e) = std::fs::create_dir_all(dir) {
                    error!("Failed to create screenshot directory {}: {e}", dir.display());
                    return;
                }
                let stats_path = dir.join("stats.txt");
                if let Err(e) = std::fs::write(&stats_path, &request.stats_text) {
                    error!("Failed to write {}: {e}", stats_path.display());
                }
            }
        }

        state.active = true;
        state.mode = request.mode;
        state.save_dir = request.save_dir;
        state.base_name = request.base_name;
        state.view = 0;
        state.phase = CapturePhase::ApplyConfig;
        state.wait_secs = 0.0;
        state.captures_done = 0;
        state.saved_settings = Some(settings.objects.clone());
        return;
    }

    match state.phase {
        CapturePhase::ApplyConfig => {
            let (object, visible) = COMPREHENSIVE_VIEWS[state.view];
            apply_view_config(&mut settings, object, visible);
            cam_override.0 = Some(object);
            // `respawn_renders` runs on a 100ms timer, so wait comfortably past
            // that (plus a render frame) before capturing.
            state.wait_secs = 0.3;
            state.phase = CapturePhase::Wait;
        }
        CapturePhase::Wait => {
            state.wait_secs -= time.delta_secs();
            if state.wait_secs <= 0.0 {
                state.phase = CapturePhase::Capture;
            }
        }
        CapturePhase::Capture => {
            if let Some(handle) = screenshot_handle.as_ref() {
                let index = state.view;
                let save_path = if state.mode == ComprehensiveMode::Save {
                    state
                        .save_dir
                        .as_ref()
                        .map(|dir| dir.join(format!("{}-{}.png", state.base_name, index + 1)))
                } else {
                    None
                };
                commands.spawn(Screenshot::image(handle.0.clone())).observe(
                    move |screenshot: On<ScreenshotCaptured>| {
                        CAPTURED_TILES.lock().unwrap().push(CapturedTile {
                            index,
                            image: screenshot.image.clone(),
                            save_path: save_path.clone(),
                        });
                    },
                );
            }
            state.phase = CapturePhase::WaitCapture;
        }
        CapturePhase::WaitCapture => {
            // `apply_captured_tiles` bumps `captures_done` once this view's tile
            // has been written to disk / copied into the preview tile.
            if state.captures_done > state.view {
                state.view += 1;
                state.phase = if state.view < COMPREHENSIVE_VIEW_COUNT {
                    CapturePhase::ApplyConfig
                } else {
                    CapturePhase::Finish
                };
            }
        }
        CapturePhase::Finish => {
            if let Some(saved) = state.saved_settings.take() {
                settings.objects = saved;
            }
            cam_override.0 = None;
            state.active = false;
            if state.mode == ComprehensiveMode::Save {
                if let Some(dir) = &state.save_dir {
                    info!("Comprehensive screenshot saved to {}", dir.display());
                }
            }
        }
    }
}

/// Drains captured comprehensive tiles: saves each to disk (Save mode) and
/// copies its pixels into the matching preview tile image for the 4-quadrant
/// preview, then advances the capture state machine.
pub fn apply_captured_tiles(
    mut images: ResMut<Assets<Image>>,
    tiles: Res<PreviewTileHandles>,
    mut state: ResMut<ComprehensiveState>,
) {
    let drained: Vec<CapturedTile> = {
        let mut queue = CAPTURED_TILES.lock().unwrap();
        if queue.is_empty() {
            return;
        }
        queue.drain(..).collect()
    };

    for tile in drained {
        if let Some(path) = &tile.save_path {
            match tile.image.clone().try_into_dynamic() {
                Ok(dyn_img) => match dyn_img.to_rgba8().save(path) {
                    Ok(()) => info!("Saved {}", path.display()),
                    Err(e) => error!("Failed to save screenshot tile: {e}"),
                },
                Err(e) => error!("Failed to convert tile image: {e:?}"),
            }
        }

        // Copy the captured pixels into the preview tile (same size/format),
        // preserving its texture descriptor so egui keeps sampling it.
        if let Some(handle) = tiles.0.get(tile.index) {
            if let Some(target) = images.get_mut(handle) {
                target.data = tile.image.data.clone();
            }
        }

        state.captures_done += 1;
    }
}

/// Set by the "Fit to view" button to request framing the model.
pub static PENDING_FIT: Mutex<bool> = Mutex::new(false);

/// Latest captured screenshot frame handed from the capture observer to the fit
/// state machine for pixel analysis.
static FIT_CAPTURE: Mutex<Option<Image>> = Mutex::new(None);

/// Fraction of the frame left as a gap between the model and the edge on the
/// tightest axis.
const FIT_MARGIN: f32 = 0.03;

/// Max camera-adjustment passes per fit.
const FIT_MAX_ITERATIONS: u32 = 8;

/// Phases of one fit iteration.
#[derive(Default, PartialEq, Eq, Clone, Copy)]
enum FitPhase {
    /// Let the screenshot camera render the current pose.
    #[default]
    Wait,
    /// Capture the screenshot texture.
    Capture,
    /// Wait for the captured frame, then analyze + adjust.
    WaitCapture,
    /// Restore state and finish.
    Finish,
}

/// State machine that frames the model by repeatedly rendering it to the
/// offscreen screenshot texture, measuring where it actually lands, and nudging
/// the camera until centred with [`FIT_MARGIN`].
#[derive(Resource, Default)]
pub struct FitState {
    active: bool,
    phase: FitPhase,
    iterations_left: u32,
    wait_frames: u32,
    /// View-space bounding-box center of the input model (depth reference).
    model_center: Vec3,
}

/// Frames the input model by measuring it in the offscreen screenshot texture
/// (a clean, square, UI-free render) and iterating the camera until the model is
/// centred with a [`FIT_MARGIN`] gap on the tightest edge. Orientation never
/// changes — only the camera's target and distance. Triggered by the "Fit to
/// view" button (via [`PENDING_FIT`]) or the `F` key (unless typing into the UI).
///
/// Image-based on purpose: it sidesteps viewport / projection / perspective-skew
/// math by optimizing exactly what gets rendered.
pub fn fit_camera_to_view(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut egui_ctx: EguiContexts,
    mut commands: Commands,
    input: Res<InputResource>,
    screenshot_handle: Option<Res<ScreenshotHandle>>,
    comprehensive: Res<ComprehensiveState>,
    mut cam_override: ResMut<ScreenshotCameraOverride>,
    mut state: ResMut<FitState>,
    mut cameras: Query<(&mut LookTransform, Option<&Projection>), With<Controller>>,
) {
    if !state.active {
        let button = std::mem::take(&mut *PENDING_FIT.lock().unwrap());
        let typing = egui_ctx
            .ctx_mut()
            .map(|c| c.wants_keyboard_input())
            .unwrap_or(false);
        let key = !typing && keyboard.just_pressed(KeyCode::KeyF);
        if !button && !key {
            return;
        }
        // Both the fit and the comprehensive capture drive the screenshot camera;
        // don't start a fit while a comprehensive run is active.
        if input.mesh.nr_verts() == 0 || comprehensive.is_running() || screenshot_handle.is_none() {
            return;
        }

        // View-space bbox center of the input model, for depth estimation.
        let translation = input.properties.translation;
        let scale = input.properties.scale;
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        for v in input.mesh.vert_ids() {
            let p = world_to_view(input.mesh.position(v), translation, scale);
            min = min.min(p);
            max = max.max(p);
        }
        if !min.is_finite() || !max.is_finite() {
            return;
        }
        state.model_center = (min + max) * 0.5;
        state.active = true;
        state.phase = FitPhase::Wait;
        state.wait_frames = 2;
        state.iterations_left = FIT_MAX_ITERATIONS;
        *FIT_CAPTURE.lock().unwrap() = None;
        // Point the screenshot camera at the model regardless of the largest panel.
        cam_override.0 = Some(Objects::InputMesh);
        return;
    }

    match state.phase {
        FitPhase::Wait => {
            if state.wait_frames > 0 {
                state.wait_frames -= 1;
            } else {
                state.phase = FitPhase::Capture;
            }
        }
        FitPhase::Capture => {
            if let Some(handle) = screenshot_handle.as_ref() {
                commands.spawn(Screenshot::image(handle.0.clone())).observe(
                    move |screenshot: On<ScreenshotCaptured>| {
                        *FIT_CAPTURE.lock().unwrap() = Some(screenshot.image.clone());
                    },
                );
            }
            // Safety timeout (frames) so a missed capture can't hang the fit.
            state.wait_frames = 120;
            state.phase = FitPhase::WaitCapture;
        }
        FitPhase::WaitCapture => {
            let Some(img) = FIT_CAPTURE.lock().unwrap().take() else {
                state.wait_frames = state.wait_frames.saturating_sub(1);
                if state.wait_frames == 0 {
                    state.phase = FitPhase::Finish;
                }
                return;
            };
            let converged = match cameras.single_mut() {
                Ok((mut look, projection)) => {
                    let vfov = match projection {
                        Some(Projection::Perspective(p)) => p.fov,
                        _ => std::f32::consts::FRAC_PI_4,
                    };
                    fit_adjust(&img, &mut look, state.model_center, vfov)
                }
                Err(_) => true,
            };
            state.iterations_left = state.iterations_left.saturating_sub(1);
            if converged || state.iterations_left == 0 {
                state.phase = FitPhase::Finish;
            } else {
                state.wait_frames = 2;
                state.phase = FitPhase::Wait;
            }
        }
        FitPhase::Finish => {
            cam_override.0 = None;
            state.active = false;
        }
    }
}

/// Measures the model's bounding box in the captured frame (opaque pixels over
/// the transparent background) and nudges the camera to centre it with
/// [`FIT_MARGIN`]. Returns whether the framing has converged.
fn fit_adjust(img: &Image, look: &mut LookTransform, model_center: Vec3, vfov: f32) -> bool {
    let Some(data) = img.data.as_ref() else {
        return true;
    };
    let w = img.width() as usize;
    let h = img.height() as usize;
    if w == 0 || h == 0 || data.len() < w * h * 4 {
        return true;
    }

    // Bounding box of model pixels. The screenshot camera renders the model
    // opaque over a transparent background, so alpha distinguishes them (a
    // colour test would fail on a black-feature model over a black-transparent
    // background).
    const ALPHA_THRESHOLD: u8 = 16;
    const STEP: usize = 2;
    let (mut xmin, mut xmax, mut ymin, mut ymax) = (usize::MAX, 0usize, usize::MAX, 0usize);
    let mut found = false;
    let mut y = 0;
    while y < h {
        let mut x = 0;
        while x < w {
            let alpha = data[(y * w + x) * 4 + 3];
            if alpha > ALPHA_THRESHOLD {
                found = true;
                xmin = xmin.min(x);
                xmax = xmax.max(x);
                ymin = ymin.min(y);
                ymax = ymax.max(y);
            }
            x += STEP;
        }
        y += STEP;
    }
    if !found {
        return true;
    }

    // Screen-space offset of the model's center and its half-extent, in pixels.
    let off_x = (xmin + xmax) as f32 * 0.5 - w as f32 * 0.5;
    let off_y = (ymin + ymax) as f32 * 0.5 - h as f32 * 0.5;
    let half = (((xmax - xmin) as f32) * 0.5)
        .max(((ymax - ymin) as f32) * 0.5)
        .max(1.0);

    // Camera basis from the (unchanged) orientation.
    let forward = (look.target - look.eye).normalize_or_zero();
    if forward == Vec3::ZERO {
        return true;
    }
    let mut right = forward.cross(Vec3::Y);
    if right.length_squared() < 1e-9 {
        right = forward.cross(Vec3::Z);
    }
    let right = right.normalize();
    let up = right.cross(forward).normalize();

    // The screenshot texture is square, so one world-per-pixel scale (derived
    // from the vertical FOV at the model's depth) covers both axes.
    let depth = (model_center - look.eye).dot(forward).max(1e-3);
    let world_per_px = 2.0 * depth * (vfov * 0.5).tan() / h as f32;

    // Pan the whole camera so the model's pixel center moves to the frame center
    // (image Y is top-down, hence the negated up component).
    let pan = right * (off_x * world_per_px) - up * (off_y * world_per_px);
    look.eye += pan;
    look.target += pan;

    // Scale distance so the larger half-extent fills to `1 - FIT_MARGIN`.
    let desired_half = (h as f32 * 0.5) * (1.0 - FIT_MARGIN);
    let ratio = half / desired_half;
    let dir = look.eye - look.target;
    look.eye = look.target + dir * ratio;

    off_x.abs() < 2.0 && off_y.abs() < 2.0 && (ratio - 1.0).abs() < 0.01
}
