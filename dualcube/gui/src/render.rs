use crate::toons::ToonsMaterial;
use crate::ui::UiResource;
use crate::{colors, MainMesh, PerpetualGizmos};
use crate::{
    to_principal_direction, vector3d_to_vec3, CameraHandles, Configuration, Perspective,
    PrincipalDirection, Rendered,
};
use bevy::camera::Viewport;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::prelude::*;
use bevy::render::render_resource::{
    Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
};
use bevy_axes_gizmo::AxesGizmoSyncCamera;
use core::f32;
use dualcube::prelude::*;
use egui_dock::LeafNode;
use enum_iterator::{all, Sequence};
use itertools::Itertools;
use mehsh::prelude::*;
use smooth_bevy_cameras::*;
use std::collections::{HashMap, HashSet};
use std::ops::Index;

const DEFAULT_CAMERA_EYE: Vec3 = Vec3::new(25.0, 25.0, 35.0);
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
}

use std::fmt;

impl fmt::Display for Objects {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Objects::InputMesh => "input mesh",
            Objects::Polycube => "polycube",
            Objects::PolycubeMap => "polycube-map",
            Objects::QuadMesh => "quad mesh",
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
        }
    }
}

pub fn update_camera_settings(
    mut cameras: Query<(&mut Controller, &mut Smoother)>,
    configuration: ResMut<Configuration>,
) {
    if let Ok((mut main_camera, mut smoother)) = cameras.single_mut() {
        *main_camera = Controller {
            mouse_rotate_sensitivity: Vec2::splat(configuration.camera_rotate_sensitivity),
            mouse_translate_sensitivity: Vec2::splat(configuration.camera_translate_sensitivity),
            mouse_wheel_zoom_sensitivity: configuration.camera_zoom_sensitivity,
            smoothing_weight: 0.8,
            ..Default::default()
        };

        *smoother = Smoother::new(0.8);
    }
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
            AxesGizmoSyncCamera,
            Tonemapping::None,
            // PrimaryEguiContext,
        ))
        .insert((OrbitCameraBundle::new(
            Controller {
                mouse_rotate_sensitivity: Vec2::splat(0.2),
                mouse_translate_sensitivity: Vec2::splat(2.),
                mouse_wheel_zoom_sensitivity: 0.2,
                smoothing_weight: 0.8,
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
            proj.scale = 0.1;
            Projection::Orthographic(proj)
        } else {
            bevy::prelude::Projection::default()
        };

        commands.spawn((
            Camera3d::default(),
            Camera {
                target: handle.into(),
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
    mut images: ResMut<Assets<Image>>,
    mut handles: ResMut<CameraHandles>,
    cameras: Query<Entity, With<Camera>>,
    configuration: ResMut<Configuration>,
    mut config_store: ResMut<GizmoConfigStore>,
) {
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
            (Objects::InputMesh, "gray")
                | (Objects::InputMesh, "wireframe")
                | (Objects::Polycube, "gray")
                | (Objects::Polycube, "paths")
                | (Objects::Polycube, "flat paths")
                | (Objects::PolycubeMap, "colored")
                | (Objects::PolycubeMap, "triangles")
                | (Objects::QuadMesh, "gray")
                | (Objects::QuadMesh, "wireframe")
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
    mut custom_materials: ResMut<Assets<ToonsMaterial>>,
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
        let toon_material = custom_materials.add(ToonsMaterial {
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
                                Objects::QuadMesh => {
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
    mut custom_materials: ResMut<Assets<ToonsMaterial>>,
    window: Single<&Window>,
    mut cameras: Query<(&mut Transform, &mut Projection, &mut Camera, &CameraFor)>,
) {
    let mut main_camera = cameras
        .iter_mut()
        .find(|(_, _, _, camera_for)| camera_for.0 == Objects::InputMesh)
        .unwrap()
        .2;

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

    let main_transform = cameras
        .iter()
        .find(|(_, _, _, camera_for)| camera_for.0 == Objects::InputMesh)
        .unwrap()
        .0;
    let normalized_translation = main_transform.translation - Vec3::from(Objects::InputMesh);
    let normalized_rotation = main_transform.rotation;

    let distance = normalized_translation.length();

    for (mut sub_transform, mut sub_projection, _sub_camera, sub_object) in &mut cameras {
        sub_transform.translation = normalized_translation + Vec3::from(sub_object.0);
        // println!("translate: {:?}", sub_transform.translation);
        sub_transform.rotation = normalized_rotation;
        if let Projection::Orthographic(orthographic) = sub_projection.as_mut() {
            dbg!(distance);
            orthographic.scale = 1. / distance.max(0.1);
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

                let mut granulated_mesh_gizmos = GizmoAsset::new();
                if let Some(layout) = &solution.layout {
                    granulated_mesh_gizmos = layout.granulated_mesh.gizmos(colors::GRAY);
                }

                render_object_store.add_object(
                    object,
                    RenderObject::default()
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
                        .to_owned(),
                );
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

pub fn automatic_rotation_camera(
    mut cameras: Query<(&mut LookTransform, &mut Projection, &mut Camera, &CameraFor)>,
    configuration: Res<Configuration>,
) {
    if !configuration.automatic_rotation_camera {
        return;
    }
    let (mut transform, _, _, _) = cameras
        .iter_mut()
        .find(|(_, _, _, camera_for)| camera_for.0 == Objects::InputMesh)
        .unwrap();
    let mut look_angles = LookAngles::from_vector(-transform.look_direction().unwrap());
    let rotation_speed = configuration.camera_rotate_sensitivity / 100.; // radians per 10ms
    look_angles.add_yaw(-rotation_speed);
    transform.eye = transform.target + transform.radius() * look_angles.unit_vector();
}
