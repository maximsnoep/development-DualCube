use crate::ui::UiResource;
use crate::{colors, MainMesh, PerpetualGizmos};
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
use bevy_axes_gizmo::AxesGizmoSyncCamera;
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
use mehsh::integrations::bevy::MeshBuilder;
use std::collections::{HashMap, HashSet};
use std::ops::Index;
use wgpu_types::BlendState;


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
    pub asset: RenderAsset,
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
        Self { asset }
    }
}

#[derive(Clone, Default)]
pub struct RenderObject {
    pub labels: Vec<String>,
    pub features: HashMap<String, RenderFeature>,
}

impl RenderObject {
    pub fn add(&mut self, label: &str, feature: RenderFeature) -> &mut Self {
        self.labels.push(label.to_owned());
        self.features.insert(label.to_owned(), feature);
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
        ));
    }
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
    let default = |object: &Objects, label: &str| {
        matches!(
            (object, label),
            // (Objects::InputMesh, "gray")
                | (Objects::InputMesh, "wireframe")
                | (Objects::InputMesh, "patches")
                | (Objects::Polycube, "gray")
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
                let asset = &features.get(feature).unwrap().asset;
                if visible {
                    match &asset {
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
                                GizmoBundle::new(gizmos.add(gizmo.0.clone()), gizmo.1, gizmo.2).0;
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
            // Spawning covers such that the objects are view-blocked.
            commands.spawn((
                Mesh3d(meshes.add(Sphere::new(400.))),
                MeshMaterial3d(background_material.clone()),
                Transform::from_translation(Vec3::from(object)),
                Rendered,
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
        Without<Controller>,
    >,
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

    if window.physical_size().x == 0
        || window.physical_size().y == 0
        || viewport_width == 0.
        || viewport_height == 0.
        || viewport_width.is_infinite()
        || viewport_height.is_infinite()
    {
        main_camera.is_active = false;
    } else {
        main_camera.is_active = true;
        main_camera.viewport = Some(Viewport {
            physical_position: UVec2 {
                x: main_surface_viewport.min[0] as u32,
                y: main_surface_viewport.min[1] as u32,
            },
            physical_size: UVec2 {
                x: viewport_width as u32,
                y: viewport_height as u32,
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

pub fn refresh(solution: &Solution) -> RenderObjectStore {
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

                    render_object_store.add_object(
                        object,
                        RenderObject::default()
                            .mesh(&polycube.structure, &black_color_map, "black")
                            .mesh(&polycube.structure, &gray_color_map, "gray")
                            .mesh(&polycube.structure, &colored_color_map, "colored")
                            .gizmo(gizmos_xloops, 6., -0.001, "x-loops")
                            .gizmo(gizmos_yloops, 6., -0.0011, "y-loops")
                            .gizmo(gizmos_zloops, 6., -0.00111, "z-loops")
                            .gizmo(gizmos_paths, 7., -0.001, "paths")
                            .gizmo(gizmos_flat_paths, 5., -0.0011, "flat paths")
                            .to_owned(),
                    );
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

                    render_object_store.add_object(
                        object,
                        RenderObject::default()
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
                            )
                            .to_owned(),
                    );
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
                let mut patch_mesh: Option<bevy::mesh::Mesh> = None;
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

                // Visualize skeleton(s) if available
                if let Some(skeleton_data) = &solution.skeleton {
                    if let Some(curve_skeleton) = skeleton_data.curve_skeleton() {
                        gizmos_raw_skeleton = create_skeleton_gizmos(curve_skeleton, translation, scale);
                    }
                    if let Some(cleaned_skeleton) = skeleton_data.cleaned_skeleton() {
                        gizmos_cleaned_skeleton =
                            create_skeleton_gizmos(cleaned_skeleton, translation, scale);
                        // Create patch visualization using the cleaned skeleton
                        patch_mesh = Some(create_patch_mesh(cleaned_skeleton, input, translation, scale));
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
                        println!("Drawing vector at vert_id: {:?}", vert_id);
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
                    let dmin_x: f64 = eigenvectors[(0, 0)];
                    let dmin_y: f64 = eigenvectors[(1, 0)];
                    let dmax_x: f64 = eigenvectors[(0, 1)];
                    let dmax_y: f64 = eigenvectors[(1, 1)];

                    // Map 2D eigenvectors back to 3D tangent vectors
                    let mut dir_min = (t1 * dmin_x + t2 * dmin_y).normalize();
                    let mut dir_max = (t1 * dmax_x + t2 * dmax_y).normalize();

                    // Enforce perfect tangency + orthogonality (cleaner field)
                    dir_max = (dir_max - n * dir_max.dot(&n)).normalize();
                    dir_min = n.cross(&dir_max).normalize();

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
                    .gizmo(gizmos_cleaned_skeleton, 25., -0.00015, "cleaned skeleton");
                
                if let Some(pm) = patch_mesh {
                    render_obj.bevy_mesh(pm, "patches");
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
                            .gizmo(contracted.gizmos(colors::WHITE), 0.75, -0.00001, "wireframe")
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

/// Creates gizmos for visualizing a curve skeleton with spheres for nodes and lines for edges.
pub fn create_skeleton_gizmos(
    curve_skeleton: &CurveSkeleton,
    translation: Vector3D,
    scale: f64,
) -> GizmoAsset {
    let mut gizmos = GizmoAsset::new();
    let skel_color = colors::to_bevy(colors::LIGHT_GRAY);
    let node_radius = 0.2;

    // Draw edges
    for edge in curve_skeleton.edge_indices() {
        let (a, b) = curve_skeleton.edge_endpoints(edge).unwrap();
        let pos_a = curve_skeleton[a].0;
        let pos_b = curve_skeleton[b].0;
        let a_view = world_to_view(pos_a, translation, scale);
        let b_view = world_to_view(pos_b, translation, scale);
        gizmos.line(a_view, b_view, skel_color);
    }

    // Draw nodes
    for node_idx in curve_skeleton.node_indices() {
        let pos = curve_skeleton[node_idx].0;
        let center = world_to_view(pos, translation, scale);
        gizmos.sphere(Isometry3d::from_translation(center), node_radius, skel_color);
    }

    gizmos
}

use bevy::color::palettes::tailwind;

/// Tailwind 500-level colors for patch visualization.
const TAILWIND_500: [bevy::color::Srgba; 22] = [
    tailwind::RED_500,
    tailwind::CYAN_500,
    tailwind::YELLOW_500,
    tailwind::PURPLE_500,
    tailwind::GREEN_500,
    tailwind::PINK_500,
    tailwind::BLUE_500,
    tailwind::ORANGE_500,
    tailwind::TEAL_500,
    tailwind::FUCHSIA_500,
    tailwind::LIME_500,
    tailwind::INDIGO_500,
    tailwind::AMBER_500,
    tailwind::EMERALD_500,
    tailwind::VIOLET_500,
    tailwind::SKY_500,
    tailwind::ROSE_500,
    tailwind::SLATE_500,
    tailwind::ZINC_500,
    tailwind::GRAY_500,
    tailwind::NEUTRAL_500,
    tailwind::STONE_500,
];

/// Gets a color for a region index using Tailwind colors with chroma reduction for cycling.
fn get_region_color(region: usize) -> [f32; 3] {
    let base_idx = region % TAILWIND_500.len();
    let cycle = region / TAILWIND_500.len();

    let base = TAILWIND_500[base_idx];

    if cycle == 0 {
        [base.red, base.green, base.blue]
    } else {
        // Reduce chroma and lighten for subsequent cycles
        let mut lch: bevy::color::Lcha = base.into();
        lch.chroma *= 0.5_f32.powi(cycle as i32);
        lch.lightness = (lch.lightness + 0.1 * cycle as f32).min(0.95);
        let srgb: bevy::color::Srgba = lch.into();
        [srgb.red, srgb.green, srgb.blue]
    }
}

/// Creates a Bevy mesh for visualizing surface patches as filled triangles.
pub fn create_patch_mesh(
    curve_skeleton: &CurveSkeleton,
    mesh: &mehsh::prelude::Mesh<INPUT>,
    translation: Vector3D,
    scale: f64,
) -> bevy::mesh::Mesh {
    // Build a mapping from vertex to region index
    let mut vertex_to_region: HashMap<VertKey<INPUT>, usize> = HashMap::new();
    for (region_idx, node_idx) in curve_skeleton.node_indices().enumerate() {
        let (_, vertices) = &curve_skeleton[node_idx];
        for &vert_key in vertices {
            vertex_to_region.insert(vert_key, region_idx);
        }
    }

    let mut builder = MeshBuilder::default();

    // Helper to get color for a region
    let region_color = |region: usize| -> [f32; 3] {
        get_region_color(region)
    };

    // Helper to transform and add a vertex to the builder
    let mut add_vertex = |pos: Vector3D, normal: Vector3D, color: &[f32; 3]| {
        let transformed_pos = pos * scale + translation;
        builder.add_vertex(&transformed_pos, &normal, color);
    };

    // For each face, handle based on region assignment
    for face_id in mesh.face_ids() {
        let face_verts: Vec<_> = mesh.vertices(face_id).collect();
        if face_verts.len() != 3 {
            continue; // Skip non-triangular faces
        }

        let v0 = face_verts[0];
        let v1 = face_verts[1];
        let v2 = face_verts[2];

        let p0 = mesh.position(v0);
        let p1 = mesh.position(v1);
        let p2 = mesh.position(v2);

        let n0 = mesh.normal(v0);
        let n1 = mesh.normal(v1);
        let n2 = mesh.normal(v2);

        // Get the region for each vertex
        let r0 = vertex_to_region.get(&v0).copied();
        let r1 = vertex_to_region.get(&v1).copied();
        let r2 = vertex_to_region.get(&v2).copied();

        match (r0, r1, r2) {
            // All vertices have regions
            (Some(r0), Some(r1), Some(r2)) => {
                if r0 == r1 && r1 == r2 {
                    // All same region, simply draw the triangle
                    let color = region_color(r0);
                    add_vertex(p0, n0, &color);
                    add_vertex(p1, n1, &color);
                    add_vertex(p2, n2, &color);
                } else if r0 == r1 {
                    // v0, v1 share region X; v2 is region Y
                    // a=v0, b=v1, c=v2
                    split_triangle(
                        &mut add_vertex,
                        p0, p1, p2, n0, n1, n2, r0, r2, &region_color,
                    );
                } else if r1 == r2 {
                    // v1, v2 share region X; v0 is region Y
                    // a=v1, b=v2, c=v0
                    split_triangle(
                        &mut add_vertex,
                        p1, p2, p0, n1, n2, n0, r1, r0, &region_color,
                    );
                } else if r0 == r2 {
                    // v0, v2 share region X; v1 is region Y
                    // Use cyclic rotation (v2, v0, v1) to maintain winding order
                    // a=v2, b=v0, c=v1
                    split_triangle(
                        &mut add_vertex,
                        p2, p0, p1, n2, n0, n1, r0, r1, &region_color,
                    );
                } else {
                    unreachable!("Triangle with all three vertices in different regions encountered.");
                    // This case is impossible for a valid skeleton.
                }
            }
            _ => {}
        }
    }

    builder.build()
}

/// Splits a triangle where vertices a,b belong to region_x and vertex c belongs to region_y.
/// Creates:
/// - (a, b, ac) and (b, bc, ac) for region X
/// - (ac, bc, c) for region Y
/// where ac = midpoint(a,c) and bc = midpoint(b,c)
/// Maintains the same winding order as the original triangle (a, b, c).
fn split_triangle<F, C>(
    add_vertex: &mut F,
    pa: Vector3D, pb: Vector3D, pc: Vector3D,
    na: Vector3D, nb: Vector3D, nc: Vector3D,
    region_x: usize, region_y: usize,
    region_color: &C,
)
where
    F: FnMut(Vector3D, Vector3D, &[f32; 3]),
    C: Fn(usize) -> [f32; 3],
{
    // Compute midpoints
    let p_ac = (pa + pc) * 0.5;
    let p_bc = (pb + pc) * 0.5;
    
    // Interpolate normals at midpoints for mesh
    let n_ac = ((na + nc) * 0.5).normalize();
    let n_bc = ((nb + nc) * 0.5).normalize();

    let color_x = region_color(region_x);
    let color_y = region_color(region_y);

    // Triangle 1 for region X: (a, b, ac)
    add_vertex(pa, na, &color_x);
    add_vertex(pb, nb, &color_x);
    add_vertex(p_ac, n_ac, &color_x);

    // Triangle 2 for region X: (b, bc, ac)
    add_vertex(pb, nb, &color_x);
    add_vertex(p_bc, n_bc, &color_x);
    add_vertex(p_ac, n_ac, &color_x);

    // Triangle for region Y: (ac, bc, c)
    add_vertex(p_ac, n_ac, &color_y);
    add_vertex(p_bc, n_bc, &color_y);
    add_vertex(pc, nc, &color_y);
}
