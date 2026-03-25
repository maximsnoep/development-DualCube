use crate::render_skeleton::{
    create_labeled_skeleton_gizmos, create_patch_boundary_gizmos, create_patch_convexity_mesh,
    create_patch_mesh, create_polycube_patch_boundary_gizmos, create_polycube_patch_mesh,
    create_skeleton_gizmos, get_region_color,
};
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
use dualcube::skeleton::cross_parameterize::virtual_mesh::{VertexToVirtual, VirtualNodeOrigin};
use dualcube::skeleton::cross_parameterize::PolycubeMap;
use dualcube::skeleton::orthogonalize::LabeledCurveSkeleton;
use egui_dock::LeafNode;
use enum_iterator::{all, Sequence};
use itertools::Itertools;
use mehsh::integrations::bevy::MeshBuilder;
use mehsh::prelude::*;
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
    UvDomain,
    QuadMesh,
    ContractedMesh,
}

impl std::fmt::Display for Objects {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Objects::InputMesh => "input mesh",
            Objects::Polycube => "polycube",
            Objects::PolycubeMap => "polycube-map",
            Objects::UvDomain => "UV domain",
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
            Objects::UvDomain => Self::new(0., 2_000., 1_000.),
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
        let projection = if object == Objects::PolycubeMap
            || object == Objects::Polycube
            || object == Objects::UvDomain
        {
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
    // default overlays
    let default = |object: &Objects, label: &str| {
        matches!(
            (object, label),
            // (Objects::InputMesh, "gray")
            (Objects::InputMesh, "wireframe")
                | (Objects::InputMesh, "patches")
                | (Objects::InputMesh, "cuts")
                // | (Objects::InputMesh, "virtual mesh debug")
                | (Objects::InputMesh, "uv long edges")
                // | (Objects::InputMesh, "uv patches")
                // | (Objects::Polycube, "gray")
                | (Objects::Polycube, "patches")
                | (Objects::Polycube, "cuts")
                | (Objects::Polycube, "paths")
                | (Objects::Polycube, "flat paths")
                | (Objects::PolycubeMap, "colored")
                | (Objects::PolycubeMap, "triangles")
                | (Objects::UvDomain, "input edges")
                | (Objects::UvDomain, "polycube edges")
                | (Objects::UvDomain, "input vertices")
                | (Objects::UvDomain, "polycube vertices")
                | (Objects::UvDomain, "uv background")
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
                                    Objects::PolycubeMap
                                    | Objects::Polycube
                                    | Objects::UvDomain => {
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
        if sub_object.0 == Objects::UvDomain {
            // UV domain: top-down view coupled to main camera's distance (zoom)
            // and XY translation (pan). Always looks down -Z.
            let uv_center = Vec3::from(Objects::UvDomain);
            // Use main camera's XY offset for panning, scaled down since UV domain is smaller.
            let pan_x = normalized_translation.x / 25.0 * 2.0;
            let pan_y = normalized_translation.y / 25.0 * 2.0;
            sub_transform.translation = uv_center + Vec3::new(pan_x, pan_y, 50.0);
            sub_transform.rotation = Quat::IDENTITY;
            if let Projection::Orthographic(orthographic) = sub_projection.as_mut() {
                // Zoom coupled to main camera distance: closer = more zoomed in.
                let viewport_height = (distance / 25.0 * 4.0).clamp(0.5, 40.0);
                orthographic.scaling_mode = ScalingMode::FixedVertical { viewport_height };
            }
        } else {
            sub_transform.translation = normalized_translation + Vec3::from(sub_object.0);
            sub_transform.rotation = normalized_rotation;
            if let Projection::Orthographic(orthographic) = sub_projection.as_mut() {
                orthographic.scaling_mode = ScalingMode::FixedVertical {
                    viewport_height: distance,
                };
            }
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

fn create_virtual_mesh_debug_gizmos(
    solution: &Solution,
    translation: Vector3D,
    scale: f64,
    use_polycube_vfg: bool,
) -> (GizmoAsset, GizmoAsset) {
    let mut edge_gizmos = GizmoAsset::new();
    let mut vertex_gizmos = GizmoAsset::new();

    let Some(pmap) = solution.skeleton.as_ref().and_then(|s| s.polycube_map()) else {
        return (edge_gizmos, vertex_gizmos);
    };

    let boundary_midpoint_color = bevy::prelude::Color::srgb(0.15, 0.85, 0.95);
    let cut_vertex_color = bevy::prelude::Color::srgb(1.0, 0.45, 0.2);

    let cut_cut_edge_color = bevy::prelude::Color::srgb(1.0, 0.2, 0.2);
    let boundary_boundary_edge_color = bevy::prelude::Color::srgb(0.2, 0.8, 0.9);
    let cut_boundary_edge_color = bevy::prelude::Color::srgb(1.0, 0.85, 0.2);

    let duplicate_left_edge_color = bevy::prelude::Color::srgb(0.2, 1.0, 0.35);
    let duplicate_right_edge_color = bevy::prelude::Color::srgb(0.9, 0.25, 1.0);

    let vertex_radius = 0.12;

    for region in pmap.regions.values() {
        let vfg = if use_polycube_vfg {
            &region.polycube_vfg
        } else {
            &region.input_vfg
        };

        let boundary_original_nodes: HashSet<_> = vfg
            .graph
            .node_indices()
            .filter(|&node| {
                matches!(vfg.graph[node].origin, VirtualNodeOrigin::MeshVertex(_))
                    && vfg.graph.neighbors(node).any(|neighbor| {
                        matches!(
                            vfg.graph[neighbor].origin,
                            VirtualNodeOrigin::BoundaryMidpoint { .. }
                        )
                    })
            })
            .collect();

        for node in vfg.graph.node_indices() {
            let pos = world_to_view(vfg.graph[node].position, translation, scale);
            match vfg.graph[node].origin {
                VirtualNodeOrigin::BoundaryMidpoint { .. } => {
                    vertex_gizmos.sphere(
                        Isometry3d::from_translation(pos),
                        vertex_radius,
                        boundary_midpoint_color,
                    );
                }
                VirtualNodeOrigin::CutDuplicate { .. } => {
                    vertex_gizmos.sphere(
                        Isometry3d::from_translation(pos),
                        vertex_radius,
                        cut_vertex_color,
                    );
                }
                _ => {}
            }
        }

        for edge_idx in vfg.graph.edge_indices() {
            let Some((a, b)) = vfg.graph.edge_endpoints(edge_idx) else {
                continue;
            };

            let a_origin = &vfg.graph[a].origin;
            let b_origin = &vfg.graph[b].origin;

            let a_is_cut = matches!(a_origin, VirtualNodeOrigin::CutDuplicate { .. });
            let b_is_cut = matches!(b_origin, VirtualNodeOrigin::CutDuplicate { .. });

            let a_is_boundary_midpoint =
                matches!(a_origin, VirtualNodeOrigin::BoundaryMidpoint { .. });
            let b_is_boundary_midpoint =
                matches!(b_origin, VirtualNodeOrigin::BoundaryMidpoint { .. });
            let a_is_boundary_vertex = boundary_original_nodes.contains(&a);
            let b_is_boundary_vertex = boundary_original_nodes.contains(&b);

            let is_midpoint_neighbor_edge = (a_is_boundary_midpoint && !b_is_boundary_midpoint)
                || (b_is_boundary_midpoint && !a_is_boundary_midpoint);

            let edge_color = if a_is_cut && b_is_cut {
                Some(cut_cut_edge_color)
            } else if is_midpoint_neighbor_edge {
                Some(boundary_boundary_edge_color)
            } else if (a_is_cut && (b_is_boundary_midpoint || b_is_boundary_vertex))
                || ((a_is_boundary_midpoint || a_is_boundary_vertex) && b_is_cut)
            {
                Some(cut_boundary_edge_color)
            } else {
                None
            };

            if let Some(color) = edge_color {
                let pa = world_to_view(vfg.graph[a].position, translation, scale);
                let pb = world_to_view(vfg.graph[b].position, translation, scale);
                edge_gizmos.line(pa, pb, color);
            }
        }

        for node in vfg.graph.node_indices() {
            let VirtualNodeOrigin::CutDuplicate { side, .. } = vfg.graph[node].origin else {
                continue;
            };

            let color = if side {
                duplicate_right_edge_color
            } else {
                duplicate_left_edge_color
            };

            let from = world_to_view(vfg.graph[node].position, translation, scale);
            for neighbor in vfg.graph.neighbors(node) {
                let to = world_to_view(vfg.graph[neighbor].position, translation, scale);
                edge_gizmos.line(from, to, color);
            }
        }
    }

    (edge_gizmos, vertex_gizmos)
}

fn uv_to_color(uv: Vector2D, u_min: f64, u_range: f64, v_min: f64, v_range: f64) -> [f32; 3] {
    let u = ((uv.x - u_min) / u_range).clamp(0.0, 1.0) as f32;
    let v = ((uv.y - v_min) / v_range).clamp(0.0, 1.0) as f32;
    [v, 0.0, u]
}

fn create_input_uv_patch_mesh(
    solution: &Solution,
    input: &mehsh::prelude::Mesh<INPUT>,
) -> Option<bevy::mesh::Mesh> {
    let pmap = solution.skeleton.as_ref().and_then(|s| s.polycube_map())?;
    let cleaned = solution
        .skeleton
        .as_ref()
        .and_then(|s| s.cleaned_skeleton())?;

    let mut vertex_to_region: HashMap<VertID, usize> = HashMap::new();
    for (compact_id, node_idx) in cleaned.node_indices().enumerate() {
        for &vert_key in &cleaned[node_idx].patch_vertices {
            vertex_to_region.insert(vert_key, compact_id);
        }
    }

    let mut vert_uv_avg: HashMap<VertID, Vector2D> = HashMap::new();

    for region in pmap.regions.values() {
        for (&vert_id, vfg_nodes) in &region.input_vfg.vert_to_nodes {
            let mut uv_sum = Vector2D::new(0.0, 0.0);
            let mut uv_count = 0.0;
            match vfg_nodes {
                VertexToVirtual::Unique(node) => {
                    if let Some(&uv) = region.input_uv.get(&node) {
                        uv_sum += uv;
                        uv_count += 1.0;
                    }
                }
                VertexToVirtual::CutPair { left, right } => {
                    if let Some(&uv) = region.input_uv.get(&left) {
                        uv_sum += uv;
                        uv_count += 1.0;
                    }
                    if let Some(&uv) = region.input_uv.get(&right) {
                        uv_sum += uv;
                        uv_count += 1.0;
                    }
                }
            }

            if uv_count > 0.0 {
                vert_uv_avg.insert(vert_id, uv_sum / uv_count);
            }
        }
    }

    if vert_uv_avg.is_empty() {
        return None;
    }

    let mut u_min = f64::INFINITY;
    let mut u_max = f64::NEG_INFINITY;
    let mut v_min = f64::INFINITY;
    let mut v_max = f64::NEG_INFINITY;

    for uv in vert_uv_avg.values() {
        u_min = u_min.min(uv.x);
        u_max = u_max.max(uv.x);
        v_min = v_min.min(uv.y);
        v_max = v_max.max(uv.y);
    }

    let u_range = (u_max - u_min).max(1e-12);
    let v_range = (v_max - v_min).max(1e-12);

    let (scale, translation) = input.scale_translation();

    let mut builder = MeshBuilder::default();

    let mut add_colored_triangle = |a: Vector3D,
                                    b: Vector3D,
                                    c: Vector3D,
                                    normal: Vector3D,
                                    uv_a: Vector2D,
                                    uv_b: Vector2D,
                                    uv_c: Vector2D,
                                    a_is_real: bool,
                                    b_is_real: bool,
                                    c_is_real: bool| {
        let mut uv_sum = Vector2D::new(0.0, 0.0);
        let mut uv_count = 0.0;

        if a_is_real {
            uv_sum += uv_a;
            uv_count += 1.0;
        }
        if b_is_real {
            uv_sum += uv_b;
            uv_count += 1.0;
        }
        if c_is_real {
            uv_sum += uv_c;
            uv_count += 1.0;
        }

        let color = if uv_count > 0.0 {
            uv_to_color(uv_sum / uv_count, u_min, u_range, v_min, v_range)
        } else {
            colors::BLACK
        };

        for p in [a, b, c] {
            let transformed_pos = p * scale + translation;
            builder.add_vertex(&transformed_pos, &normal, &color);
        }
    };

    let mut add_triangle_with_boundary_split =
        |a: (VertID, Vector3D), b: (VertID, Vector3D), c: (VertID, Vector3D), normal: Vector3D| {
            let (va, pa) = a;
            let (vb, pb) = b;
            let (vc, pc) = c;

            let (Some(&ra), Some(&rb), Some(&rc)) = (
                vertex_to_region.get(&va),
                vertex_to_region.get(&vb),
                vertex_to_region.get(&vc),
            ) else {
                return;
            };

            let (Some(&uv_a), Some(&uv_b), Some(&uv_c)) = (
                vert_uv_avg.get(&va),
                vert_uv_avg.get(&vb),
                vert_uv_avg.get(&vc),
            ) else {
                return;
            };

            if ra == rb && rb == rc {
                add_colored_triangle(pa, pb, pc, normal, uv_a, uv_b, uv_c, true, true, true);
                return;
            }

            let mut split_two_plus_one = |pa: Vector3D,
                                          pb: Vector3D,
                                          pc: Vector3D,
                                          uv_a: Vector2D,
                                          uv_b: Vector2D,
                                          uv_c: Vector2D| {
                let p_ac = (pa + pc) * 0.5;
                let p_bc = (pb + pc) * 0.5;
                let uv_ac = (uv_a + uv_c) / 2.0;
                let uv_bc = (uv_b + uv_c) / 2.0;

                add_colored_triangle(pa, pb, p_ac, normal, uv_a, uv_b, uv_ac, true, true, false);
                add_colored_triangle(
                    pb, p_bc, p_ac, normal, uv_b, uv_bc, uv_ac, true, false, false,
                );
                add_colored_triangle(
                    p_ac, p_bc, pc, normal, uv_ac, uv_bc, uv_c, false, false, true,
                );
            };

            if ra == rb {
                split_two_plus_one(pa, pb, pc, uv_a, uv_b, uv_c);
            } else if rb == rc {
                split_two_plus_one(pb, pc, pa, uv_b, uv_c, uv_a);
            } else if ra == rc {
                split_two_plus_one(pc, pa, pb, uv_c, uv_a, uv_b);
            } else {
                // Rare fallback: all three vertices assigned different regions.
                add_colored_triangle(pa, pb, pc, normal, uv_a, uv_b, uv_c, true, true, true);
            }
        };

    for face_id in input.face_ids() {
        let normal = input.normal(face_id);
        let verts: Vec<_> = input.vertices(face_id).collect();
        match verts.as_slice() {
            [v0, v1, v2] => {
                add_triangle_with_boundary_split(
                    (*v0, input.position(*v0)),
                    (*v1, input.position(*v1)),
                    (*v2, input.position(*v2)),
                    normal,
                );
            }
            [v0, v1, v2, v3] => {
                // Triangulate quads and apply the same boundary-aware split logic.
                add_triangle_with_boundary_split(
                    (*v0, input.position(*v0)),
                    (*v1, input.position(*v1)),
                    (*v2, input.position(*v2)),
                    normal,
                );
                add_triangle_with_boundary_split(
                    (*v0, input.position(*v0)),
                    (*v2, input.position(*v2)),
                    (*v3, input.position(*v3)),
                    normal,
                );
            }
            _ => {}
        }
    }

    Some(builder.build())
}

fn create_input_uv_long_edge_overlay(
    solution: &Solution,
    skeleton: &LabeledCurveSkeleton,
    selected_region: usize,
    translation: Vector3D,
    scale: f64,
    threshold: f64,
) -> Option<GizmoAsset> {
    let pmap = solution.skeleton.as_ref().and_then(|s| s.polycube_map())?;

    let mut sorted_regions: Vec<_> = pmap.regions.iter().collect();
    sorted_regions.sort_by_key(|(idx, _)| idx.index());
    if sorted_regions.is_empty() {
        return None;
    }
    let region_i = selected_region.min(sorted_regions.len().saturating_sub(1));
    let (region_idx, region) = sorted_regions[region_i];

    let boundary_from_origin = |origin: &VirtualNodeOrigin| match origin {
        VirtualNodeOrigin::BoundaryMidpoint {
            boundary_edge: boundary,
            ..
        }
        | VirtualNodeOrigin::CutEndpointMidpointDuplicate { boundary, .. } => Some(*boundary),
        _ => None,
    };

    let other_region_color = |boundary| -> Option<bevy::prelude::Color> {
        let (a, b) = skeleton.edge_endpoints(boundary)?;
        let other = if a == *region_idx {
            b
        } else if b == *region_idx {
            a
        } else {
            return None;
        };
        let c = get_region_color(other.index());
        Some(bevy::prelude::Color::srgb(c[0], c[1], c[2]))
    };

    // Save exact edge endpoints in input-mesh space first; UV is only used
    // to classify whether an edge is unexpectedly long.
    let mut bad_edges_mesh_space: Vec<(Vector3D, Vector3D, bevy::prelude::Color)> = Vec::new();

    let mut count = 0usize;
    for edge_idx in region.input_vfg.graph.edge_indices() {
        let Some((a, b)) = region.input_vfg.graph.edge_endpoints(edge_idx) else {
            continue;
        };

        let (Some(&uv_a), Some(&uv_b)) = (region.input_uv.get(&a), region.input_uv.get(&b)) else {
            continue;
        };

        let uv_len = (uv_b - uv_a).norm();
        if uv_len > threshold {
            count += 1;

            let ba = boundary_from_origin(&region.input_vfg.graph[a].origin);
            let bb = boundary_from_origin(&region.input_vfg.graph[b].origin);
            let boundary_count = usize::from(ba.is_some()) + usize::from(bb.is_some());

            let color = match boundary_count {
                2 => bevy::prelude::Color::BLACK,
                1 => {
                    let boundary = ba
                        .or(bb)
                        .expect("boundary must exist when boundary_count=1");
                    other_region_color(boundary).unwrap_or(bevy::prelude::Color::WHITE)
                }
                _ => bevy::prelude::Color::WHITE,
            };

            bad_edges_mesh_space.push((
                region.input_vfg.graph[a].position,
                region.input_vfg.graph[b].position,
                color,
            ));
        }
    }

    if count > 0 {
        error!(
            "UV long-edge detector: region {:?} has {} long UV edges (threshold={:.3})",
            region_idx, count, threshold
        );
    }

    let mut gizmos = GizmoAsset::new();

    for (pa_mesh, pb_mesh, color) in bad_edges_mesh_space {
        let pa = world_to_view(pa_mesh, translation, scale);
        let pb = world_to_view(pb_mesh, translation, scale);
        gizmos.line(pa, pb, color);
    }

    Some(gizmos)
}

/// Creates a flat 2D UV-domain visualization showing both input and polycube
/// VFG embeddings overlaid on the canonical domain, one region at a time.
/// Regions are laid out in a grid. Each region shows:
/// - A UV-colored background quad
/// - Input mesh edges in blue (boundary in cyan)
/// - Polycube mesh edges in red (boundary in yellow)
/// - Boundary and interior vertices at different sizes
/// Returns the number of regions in the polycube map (for UI selectors).
pub fn uv_domain_region_count(solution: &Solution) -> usize {
    solution
        .skeleton
        .as_ref()
        .and_then(|s| s.polycube_map())
        .map(|pmap| pmap.regions.len())
        .unwrap_or(0)
}

fn create_uv_domain_view(
    pmap: &PolycubeMap,
    skeleton: &LabeledCurveSkeleton,
    selected_region: usize,
) -> RenderObject {
    let mut render_obj = RenderObject::default();

    // Sort regions by node index for stable ordering (matches UI selector).
    let mut sorted_regions: Vec<_> = pmap.regions.iter().collect();
    sorted_regions.sort_by_key(|(idx, _)| idx.index());

    let region_i = selected_region.min(sorted_regions.len().saturating_sub(1));
    let (node_idx, region) = sorted_regions[region_i];

    let mut input_edge_gizmos = GizmoAsset::new();
    let mut polycube_edge_gizmos = GizmoAsset::new();
    let mut input_vertex_gizmos = GizmoAsset::new();
    let mut polycube_vertex_gizmos = GizmoAsset::new();
    let mut builder = MeshBuilder::default();

    let input_color = bevy::prelude::Color::srgb(0.3, 0.5, 1.0);
    let polycube_color = bevy::prelude::Color::srgb(1.0, 0.3, 0.3);
    let input_boundary_color = bevy::prelude::Color::srgb(0.0, 0.9, 0.9);
    let polycube_boundary_color = bevy::prelude::Color::srgb(1.0, 0.9, 0.0);
    let cut_boundary_color = bevy::prelude::Color::srgb(1.0, 0.5, 0.0); // orange: cut boundary
    let input_interior_color = bevy::prelude::Color::srgb(0.5, 0.5, 0.5);

    let boundary_from_origin = |origin: &VirtualNodeOrigin| match origin {
        VirtualNodeOrigin::BoundaryMidpoint {
            boundary_edge: boundary,
            ..
        }
        | VirtualNodeOrigin::CutEndpointMidpointDuplicate { boundary, .. } => Some(*boundary),
        _ => None,
    };

    let other_region_color = |boundary| -> Option<bevy::prelude::Color> {
        let (a, b) = skeleton.edge_endpoints(boundary)?;
        let other = if a == *node_idx {
            b
        } else if b == *node_idx {
            a
        } else {
            return None;
        };
        let c = get_region_color(other.index());
        Some(bevy::prelude::Color::srgb(c[0], c[1], c[2]))
    };

    // Precompute cut node sets for both VFGs to avoid closure type-inference issues.
    let input_cut_set: HashSet<_> = region
        .input_vfg
        .graph
        .node_indices()
        .filter(|&n| {
            matches!(
                region.input_vfg.graph[n].origin,
                VirtualNodeOrigin::CutDuplicate { .. }
                    | VirtualNodeOrigin::CutEndpointMidpointDuplicate { .. }
            )
        })
        .collect();
    let polycube_cut_set: HashSet<_> = region
        .polycube_vfg
        .graph
        .node_indices()
        .filter(|&n| {
            matches!(
                region.polycube_vfg.graph[n].origin,
                VirtualNodeOrigin::CutDuplicate { .. }
                    | VirtualNodeOrigin::CutEndpointMidpointDuplicate { .. }
            )
        })
        .collect();

    // UV to flat XY position (centered at origin).
    let uv_to_pos = |uv: &Vector2D| -> Vec3 { Vec3::new(uv.x as f32, uv.y as f32, 0.0) };

    // Build the background polygon from the actual boundary UV positions of the
    // input VFG — these are the vertices that were placed on the canonical polygon
    // by map_boundary_to_polygon, so they define the exact shape used.
    {
        let polygon: Vec<Vector2D> = region
            .input_vfg
            .boundary_loop
            .iter()
            .filter_map(|n| region.input_uv.get(n).copied())
            .collect();

        if polygon.len() >= 3 {
            let normal = Vector3D::new(0.0, 0.0, 1.0);
            let centroid = polygon.iter().fold(Vector2D::new(0.0, 0.0), |acc, p| {
                Vector2D::new(acc.x + p.x, acc.y + p.y)
            }) / polygon.len() as f64;
            // Fan-triangulate from centroid so we handle non-convex boundaries.
            for i in 0..polygon.len() {
                let a = &centroid;
                let b = &polygon[i];
                let c = &polygon[(i + 1) % polygon.len()];
                for uv in [a, b, c] {
                    let pos = Vector3D::new(uv.x, uv.y, -0.01);
                    let u_norm = ((uv.x + 1.0) / 2.0).clamp(0.0, 1.0) as f32;
                    let v_norm = ((uv.y + 1.0) / 2.0).clamp(0.0, 1.0) as f32;
                    let color = [v_norm * 0.4, 0.0, u_norm * 0.4];
                    builder.add_vertex(&pos, &normal, &color);
                }
            }
        }
    }

    // Input VFG edges.
    let input_boundary_set: HashSet<_> = region.input_vfg.boundary_loop.iter().copied().collect();
    for edge_idx in region.input_vfg.graph.edge_indices() {
        let Some((a, b)) = region.input_vfg.graph.edge_endpoints(edge_idx) else {
            continue;
        };
        let (Some(uv_a), Some(uv_b)) = (region.input_uv.get(&a), region.input_uv.get(&b)) else {
            continue;
        };
        let on_boundary = input_boundary_set.contains(&a) && input_boundary_set.contains(&b);
        let side_segment_color = if on_boundary {
            match (
                boundary_from_origin(&region.input_vfg.graph[a].origin),
                boundary_from_origin(&region.input_vfg.graph[b].origin),
            ) {
                (Some(ea), Some(eb)) if ea == eb => other_region_color(ea),
                _ => None,
            }
        } else {
            None
        };
        let on_cut = input_cut_set.contains(&a) || input_cut_set.contains(&b);
        let color = if let Some(c) = side_segment_color {
            c
        } else if on_boundary && on_cut {
            cut_boundary_color
        } else if on_boundary {
            input_boundary_color
        } else {
            input_color
        };
        input_edge_gizmos.line(uv_to_pos(uv_a), uv_to_pos(uv_b), color);
    }

    // Polycube VFG edges.
    let polycube_boundary_set: HashSet<_> =
        region.polycube_vfg.boundary_loop.iter().copied().collect();
    for edge_idx in region.polycube_vfg.graph.edge_indices() {
        let Some((a, b)) = region.polycube_vfg.graph.edge_endpoints(edge_idx) else {
            continue;
        };
        let (Some(uv_a), Some(uv_b)) = (region.polycube_uv.get(&a), region.polycube_uv.get(&b))
        else {
            continue;
        };
        let on_boundary = polycube_boundary_set.contains(&a) && polycube_boundary_set.contains(&b);
        let side_segment_color = if on_boundary {
            match (
                boundary_from_origin(&region.polycube_vfg.graph[a].origin),
                boundary_from_origin(&region.polycube_vfg.graph[b].origin),
            ) {
                (Some(ea), Some(eb)) if ea == eb => other_region_color(ea),
                _ => None,
            }
        } else {
            None
        };
        let on_cut = polycube_cut_set.contains(&a) || polycube_cut_set.contains(&b);
        let color = if let Some(c) = side_segment_color {
            c
        } else if on_boundary && on_cut {
            cut_boundary_color
        } else if on_boundary {
            polycube_boundary_color
        } else {
            polycube_color
        };
        polycube_edge_gizmos.line(uv_to_pos(uv_a), uv_to_pos(uv_b), color);
    }

    // Input vertices: cyan boundary (large), gray interior (small), orange if on cut.
    for node in region.input_vfg.graph.node_indices() {
        let Some(uv) = region.input_uv.get(&node) else {
            continue;
        };
        let pos = uv_to_pos(uv);
        if input_boundary_set.contains(&node) {
            let color = boundary_from_origin(&region.input_vfg.graph[node].origin)
                .and_then(other_region_color)
                .unwrap_or_else(|| {
                    if input_cut_set.contains(&node) {
                        cut_boundary_color
                    } else {
                        input_boundary_color
                    }
                });
            input_vertex_gizmos.sphere(Isometry3d::from_translation(pos), 0.03, color);
        } else {
            input_vertex_gizmos.sphere(
                Isometry3d::from_translation(pos),
                0.015,
                input_interior_color,
            );
        }
    }

    // Polycube vertices: yellow boundary (large), red interior (small), orange if on cut.
    // Higher Z so they draw on top of input vertices.
    for node in region.polycube_vfg.graph.node_indices() {
        let Some(uv) = region.polycube_uv.get(&node) else {
            continue;
        };
        let mut pos = uv_to_pos(uv);
        pos.z += 0.002;
        if polycube_boundary_set.contains(&node) {
            let color = boundary_from_origin(&region.polycube_vfg.graph[node].origin)
                .and_then(other_region_color)
                .unwrap_or_else(|| {
                    if polycube_cut_set.contains(&node) {
                        cut_boundary_color
                    } else {
                        polycube_boundary_color
                    }
                });
            polycube_vertex_gizmos.sphere(Isometry3d::from_translation(pos), 0.035, color);
        } else {
            polycube_vertex_gizmos.sphere(Isometry3d::from_translation(pos), 0.02, polycube_color);
        }
    }

    render_obj
        .bevy_mesh(builder.build(), "uv background")
        .gizmo(input_edge_gizmos, 2.0, -0.001, "input edges")
        .gizmo(polycube_edge_gizmos, 2.0, -0.002, "polycube edges")
        .gizmo(input_vertex_gizmos, 1.0, -0.003, "input vertices")
        .gizmo(polycube_vertex_gizmos, 1.0, -0.004, "polycube vertices");

    render_obj
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

                    if let Some(pmap) = solution.skeleton.as_ref().and_then(|s| s.polycube_map()) {
                        let mut gizmos_cuts = GizmoAsset::new();
                        let cut_color = colors::to_bevy(colors::SNOEP_YELLOW);
                        for region in pmap.regions.values() {
                            for cut_path in &region.polycube_cuts {
                                let positions: Vec<Vec3> = cut_path
                                    .iter()
                                    .map(|&p| world_to_view(p, translation, scale))
                                    .collect::<Vec<Vec3>>();
                                gizmos_cuts.linestrip(positions, cut_color);
                            }
                        }
                        // Make cuts more visible by drawing them thicker and slightly above the surface.
                        render_obj.gizmo(gizmos_cuts, 10.0, -0.01, "cuts");
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
                    let (scale, translation) = quad.quad_mesh_polycube.scale_translation();
                    for face_id in quad.quad_mesh_polycube.face_ids() {
                        let color = colors::LIGHT_GRAY;
                        color_map.insert(face_id, color);
                    }

                    let mut render_obj = RenderObject::default();
                    let (virtual_mesh_edge_gizmos, virtual_mesh_vertex_gizmos) =
                        create_virtual_mesh_debug_gizmos(solution, translation, scale, true);
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
                        )
                        .gizmo(virtual_mesh_edge_gizmos, 3., -0.0102, "virtual mesh debug")
                        .gizmo(
                            virtual_mesh_vertex_gizmos,
                            1.,
                            -0.0103,
                            "virtual mesh debug",
                        );

                    render_object_store.add_object(object, render_obj);
                }
            }
            Objects::UvDomain => {
                if let (Some(pmap), Some(labeled)) = (
                    solution.skeleton.as_ref().and_then(|s| s.polycube_map()),
                    solution
                        .skeleton
                        .as_ref()
                        .and_then(|s| s.labeled_skeleton()),
                ) {
                    let render_obj =
                        create_uv_domain_view(pmap, labeled, configuration.uv_domain_region);
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

                // Visualize skeleton(s) if available. Similarly, show labelling if available.
                if let Some(skeleton_data) = &solution.skeleton {
                    if let Some(curve_skeleton) = skeleton_data.curve_skeleton() {
                        gizmos_raw_skeleton =
                            create_skeleton_gizmos(curve_skeleton, translation, scale);
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
                    .gizmo(gizmos_cleaned_skeleton, 25., -0.00015, "cleaned skeleton");

                let mut patch_convexity_mesh: Option<bevy::mesh::Mesh> = None;
                let mut collapse_history_mesh: Option<bevy::mesh::Mesh> = None;
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
                // Build collapse history patch overlay if history is available
                if let Some(skeleton_data) = &solution.skeleton {
                    if let Some(history_skeleton) = skeleton_data
                        .reconstruct_skeleton_from_collapse_history(
                            configuration.collapse_history_step,
                        )
                    {
                        collapse_history_mesh = Some(create_patch_mesh(
                            &history_skeleton,
                            input,
                            translation,
                            scale,
                        ));
                        let history_boundary_gizmos = create_patch_boundary_gizmos(
                            &history_skeleton,
                            input,
                            translation,
                            scale,
                        );
                        render_obj.gizmo(
                            history_boundary_gizmos,
                            1.2,
                            -0.00017,
                            "collapse history",
                        );
                    }
                }
                if let Some(chm) = collapse_history_mesh {
                    render_obj.bevy_mesh(chm, "collapse history");
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

                if let (Some(pmap), Some(labeled)) = (
                    solution.skeleton.as_ref().and_then(|s| s.polycube_map()),
                    solution
                        .skeleton
                        .as_ref()
                        .and_then(|s| s.labeled_skeleton()),
                ) {
                    let mut gizmos_cuts = GizmoAsset::new();
                    let cut_color = colors::to_bevy(colors::SNOEP_YELLOW);
                    for region in pmap.regions.values() {
                        for cut_path in &region.input_cuts {
                            let positions: Vec<Vec3> = cut_path
                                .iter()
                                .map(|&p| world_to_view(p, translation, scale))
                                .collect::<Vec<Vec3>>();
                            gizmos_cuts.linestrip(positions, cut_color);
                        }
                    }
                    render_obj.gizmo(gizmos_cuts, 5.0, -0.001, "cuts");

                    let (virtual_mesh_edge_gizmos, virtual_mesh_vertex_gizmos) =
                        create_virtual_mesh_debug_gizmos(solution, translation, scale, false);
                    render_obj
                        .gizmo(virtual_mesh_edge_gizmos, 2.0, -0.0012, "virtual mesh debug")
                        .gizmo(
                            virtual_mesh_vertex_gizmos,
                            1.0,
                            -0.0013,
                            "virtual mesh debug",
                        );

                    if let Some(long_edge_overlay) = create_input_uv_long_edge_overlay(
                        solution,
                        labeled,
                        configuration.uv_domain_region,
                        translation,
                        scale,
                        1.0,
                    ) {
                        render_obj.gizmo(long_edge_overlay, 4.0, -0.00125, "uv long edges");
                    }
                }

                if let Some(uv_patch_mesh) = create_input_uv_patch_mesh(solution, input) {
                    render_obj.bevy_mesh(uv_patch_mesh, "uv patches");
                }

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
