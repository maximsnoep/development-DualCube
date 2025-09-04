use crate::colors;
use crate::ui::UiResource;
use crate::{to_principal_direction, vector3d_to_vec3, CameraHandles, Configuration, FlatMaterial, Perspective, PrincipalDirection, Rendered};
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::prelude::*;
use bevy::render::camera::ScalingMode;
use bevy::render::camera::Viewport;
use bevy::render::render_resource::{Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages};
use dualcube::prelude::*;
use enum_iterator::{all, Sequence};
use itertools::Itertools;
use mehsh::prelude::*;
use smooth_bevy_cameras::controllers::orbit::{OrbitCameraBundle, OrbitCameraController};
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
#[allow(dead_code)]
#[must_use]
pub fn invert_transform_coordinates(position: Vector3D, translation: Vector3D, scale: f64) -> Vector3D {
    (position - translation) / scale
}

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone, Default, Sequence)]
pub enum Objects {
    InputMesh,
    #[default]
    PolycubeMap,
    QuadMesh,
}

use std::fmt;

impl fmt::Display for Objects {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Objects::InputMesh => "input mesh",
            Objects::PolycubeMap => "polycube-map",
            Objects::QuadMesh => "quad mesh",
        };
        write!(f, "{}", s)
    }
}

#[derive(Clone)]
pub enum RenderAsset {
    Mesh(bevy::render::mesh::Mesh),
    Gizmo((GizmoAsset, f32, f32)),
}

#[derive(Clone, PartialEq)]
pub struct MeshBundle(Mesh3d);
impl MeshBundle {
    pub const fn new(handle: Handle<bevy::render::mesh::Mesh>) -> Self {
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
    pub label: String,
    pub visible: bool,
    pub asset: RenderAsset,
}

#[derive(Clone, PartialEq)]
pub struct RenderFlag {
    pub label: String,
    pub visible: bool,
}

impl RenderFeature {
    pub fn flag(&self) -> RenderFlag {
        RenderFlag {
            label: self.label.clone(),
            visible: self.visible,
        }
    }
}

impl PartialEq for RenderFeature {
    fn eq(&self, other: &Self) -> bool {
        self.label == other.label && self.visible == other.visible
    }
}

impl RenderFeature {
    pub fn new(label: String, visible: bool, asset: RenderAsset) -> Self {
        Self { label, visible, asset }
    }
}

#[derive(Clone, Default, PartialEq)]
pub struct RenderObject {
    pub features: Vec<RenderFeature>,
}

impl RenderObject {
    pub fn add(&mut self, feature: RenderFeature) -> &mut Self {
        self.features.push(feature);
        self
    }

    pub fn mesh<M: Tag>(&mut self, mesh: &mehsh::prelude::Mesh<M>, color_map: &HashMap<FaceKey<M>, colors::Color>, label: &str, visible: bool) -> &mut Self {
        self.add(RenderFeature::new(label.to_owned(), visible, RenderAsset::Mesh(mesh.bevy(color_map).0)))
    }

    pub fn gizmo(&mut self, gizmo: GizmoAsset, width: f32, depth: f32, label: &str, visible: bool) -> &mut Self {
        self.add(RenderFeature::new(label.to_owned(), visible, RenderAsset::Gizmo((gizmo, width, depth))))
    }
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

impl From<Objects> for String {
    fn from(val: Objects) -> Self {
        match val {
            Objects::InputMesh => "input mesh",
            Objects::PolycubeMap => "polycube-map",
            Objects::QuadMesh => "quad mesh",
        }
        .to_owned()
    }
}

impl From<Objects> for Vec3 {
    fn from(val: Objects) -> Self {
        match val {
            Objects::InputMesh => Self::new(0., 0., 0.),
            Objects::PolycubeMap => Self::new(0., 1_000., 1_000.),
            Objects::QuadMesh => Self::new(1_000., 0., 1_000.),
        }
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
            Tonemapping::None,
        ))
        .insert((OrbitCameraBundle::new(
            OrbitCameraController {
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
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        },
        ..default()
    };
    image.resize(image.texture_descriptor.size);

    for object in all::<Objects>() {
        let handle = images.add(image.clone());
        handles.map.insert(CameraFor(object), handle.clone());
        let projection = if object == Objects::PolycubeMap {
            let mut proj = OrthographicProjection::default_3d();
            proj.scaling_mode = ScalingMode::FixedVertical { viewport_height: 30. };
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
) {
    self::reset(&mut commands, &cameras, &mut images, &mut handles, &configuration);
}

#[derive(Default, Debug, Clone)]
pub struct MeshProperties {
    pub source: String,
    pub scale: f64,
    pub translation: Vector3D,
}

pub fn respawn_renders(
    mut commands: Commands,
    mut meshes: ResMut<Assets<bevy::render::mesh::Mesh>>,
    mut gizmos: ResMut<Assets<GizmoAsset>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut custom_materials: ResMut<Assets<FlatMaterial>>,
    configuration: Res<Configuration>,
    render_object_store: Res<RenderObjectStore>,
    rendered_mesh_query: Query<Entity, With<Rendered>>,
) {
    if render_object_store.is_changed() {
        info!("render_object_store has been changed.");

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

        let flat_material = materials.add(StandardMaterial { unlit: true, ..default() });
        let toon_material = custom_materials.add(FlatMaterial {
            view_dir: Vec3::new(0.0, 0.0, 1.0),
        });
        let background_material = materials.add(StandardMaterial {
            base_color: bevy::prelude::Color::srgb_u8(configuration.clear_color[0], configuration.clear_color[1], configuration.clear_color[2]),
            unlit: true,
            ..default()
        });

        // Go through render_object_store and spawn all objects (if they are visible).
        info!("Spawn all objects.");
        for (&object, render_object) in &render_object_store.objects {
            for feature in &render_object.features {
                if feature.visible {
                    match &feature.asset {
                        RenderAsset::Mesh(mesh) => {
                            let mesh_handle = MeshBundle::new(meshes.add(mesh.clone())).0;
                            match object {
                                Objects::InputMesh | Objects::QuadMesh => {
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
                                Objects::PolycubeMap => {
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
                            let gizmo_handle = GizmoBundle::new(gizmos.add(gizmo.0.clone()), gizmo.1, gizmo.2).0;
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
    mut custom_materials: ResMut<Assets<FlatMaterial>>,
    window: Single<&Window>,
    mut cameras: Query<(&mut Transform, &mut Projection, &mut Camera, &CameraFor)>,
) {
    let mut main_camera = cameras.iter_mut().find(|(_, _, _, camera_for)| camera_for.0 == Objects::InputMesh).unwrap().2;

    let (_, node_index, _) = ui_resource.tree.find_tab(&Objects::InputMesh).unwrap();
    let main_surface = ui_resource.tree.main_surface().clone();
    let main_node = main_surface.index(node_index);
    let main_surface_viewport = match main_node {
        egui_dock::Node::Leaf { viewport, .. } => *viewport,
        _ => unreachable!(),
    };

    let viewport_width = main_surface_viewport.max[0] - main_surface_viewport.min[0];
    let viewport_height = main_surface_viewport.max[1] - main_surface_viewport.min[1];

    if window.physical_size().x == 0 || window.physical_size().y == 0 || viewport_width == 0. || viewport_height == 0. {
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

    let main_transform = cameras.iter().find(|(_, _, _, camera_for)| camera_for.0 == Objects::InputMesh).unwrap().0;
    let normalized_translation = main_transform.translation - Vec3::from(Objects::InputMesh);
    let normalized_rotation = main_transform.rotation;

    let distance = normalized_translation.length();

    for (mut sub_transform, mut sub_projection, _sub_camera, sub_object) in &mut cameras {
        sub_transform.translation = normalized_translation + Vec3::from(sub_object.0);
        // println!("translate: {:?}", sub_transform.translation);
        sub_transform.rotation = normalized_rotation;
        if let Projection::Orthographic(orthographic) = sub_projection.as_mut() {
            orthographic.scaling_mode = ScalingMode::FixedVertical { viewport_height: distance };
        }
    }

    for material in custom_materials.iter_mut() {
        // current location of the camera, to (0, 0, 0)
        material.1.view_dir = Vec3::new(normalized_translation.x, normalized_translation.y, normalized_translation.z).normalize();
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
                        let color = colors::from_direction(to_principal_direction(normal).0, Some(Perspective::Primal), None);
                        color_map.insert(face_id, [color[0] as f32, color[1] as f32, color[2] as f32]);
                    }

                    let mut gizmos_paths = GizmoAsset::new();
                    let mut gizmos_flat_paths = GizmoAsset::new();
                    if let (Ok(_), Some(_)) = (&solution.layout, &solution.polycube) {
                        let color = colors::GRAY;
                        let c = bevy::color::Color::srgb(color[0], color[1], color[2]);

                        let mut irregular_vertices = HashSet::new();
                        for vert_id in quad.quad_mesh.vert_ids() {
                            // Get the faces around the vertex
                            let faces = quad.quad_mesh.faces(vert_id);
                            // Get the labels of the faces around
                            let labels = faces
                                .iter()
                                .map(|&face_id| to_principal_direction(quad.quad_mesh_polycube.normal(face_id)).0)
                                .collect::<HashSet<_>>();
                            // If 3+ labels, its irregular
                            if labels.len() >= 3 {
                                irregular_vertices.insert(vert_id);
                            }
                        }

                        // Get all edges going out irregular vertices
                        let mut irregular_edges = HashSet::new();
                        for &vert_id in &irregular_vertices {
                            let edges = quad.quad_mesh.edges(vert_id);
                            for &edge_id in &edges {
                                let mut next_twin_next = quad.quad_mesh.next(quad.quad_mesh.twin(quad.quad_mesh.next(edge_id)));
                                while !irregular_edges.contains(&next_twin_next) {
                                    irregular_edges.insert(next_twin_next);
                                    next_twin_next = quad.quad_mesh.next(quad.quad_mesh.twin(quad.quad_mesh.next(next_twin_next)));
                                }
                            }
                        }

                        // Draw all irregular edges
                        for edge_id in irregular_edges {
                            let faces = quad.quad_mesh.faces(edge_id);
                            let n1 = quad.quad_mesh_polycube.normal(faces[0]);
                            let n2 = quad.quad_mesh_polycube.normal(faces[1]);
                            let endpoints = quad.quad_mesh.vertices(edge_id);
                            let u = quad.quad_mesh.position(endpoints[0]);
                            let v = quad.quad_mesh.position(endpoints[1]);
                            let line = DrawableLine::from_line(u, v, Vector3D::new(0., 0., 0.), translation, scale);
                            if n1 == n2 {
                                gizmos_flat_paths.line(line.u, line.v, c);
                            } else {
                                gizmos_paths.line(line.u, line.v, c);
                            }
                        }
                    }

                    render_object_store.add_object(
                        object,
                        RenderObject::default()
                            .mesh(&quad.quad_mesh, &default_color_map, "default", false)
                            .mesh(&quad.quad_mesh, &color_map, "colored", true)
                            .gizmo(quad.quad_mesh.gizmos(colors::GRAY), 1.0, -0.001, "wireframe", false)
                            .gizmo(gizmos_paths, 4., -0.0001, "paths", true)
                            .gizmo(gizmos_flat_paths, 3., -0.00011, "flat paths", false)
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
                    let mut default_color_map = HashMap::new();
                    for face_id in quad.quad_mesh_polycube.face_ids() {
                        default_color_map.insert(face_id, colors::LIGHT_GRAY);
                    }

                    let (scale, translation) = quad.quad_mesh_polycube.scale_translation();

                    let mut color_map = HashMap::new();
                    for face_id in quad.quad_mesh_polycube.face_ids() {
                        let normal = quad.quad_mesh_polycube.normal(face_id);
                        let color = colors::from_direction(to_principal_direction(normal).0, Some(Perspective::Primal), None);
                        color_map.insert(face_id, [color[0] as f32, color[1] as f32, color[2] as f32]);
                    }

                    let mut gizmos_paths = GizmoAsset::new();
                    let mut gizmos_flat_paths = GizmoAsset::new();
                    if let (Ok(lay), Some(polycube)) = (&solution.layout, &solution.polycube) {
                        let color = colors::GRAY;
                        let c = bevy::color::Color::srgb(color[0], color[1], color[2]);

                        for &pedge_id in lay.edge_to_path.keys() {
                            let f1 = polycube.structure.normal(polycube.structure.face(pedge_id));
                            let f2 = polycube.structure.normal(polycube.structure.face(polycube.structure.twin(pedge_id)));
                            let endpoints = polycube.structure.vertices(pedge_id);
                            let u = polycube.structure.position(endpoints[0]);
                            let v = polycube.structure.position(endpoints[1]);
                            let line = DrawableLine::from_line(u, v, Vector3D::new(0., 0., 0.), translation, scale);
                            gizmos_flat_paths.line(line.u, line.v, c);
                            if f1 != f2 {
                                gizmos_paths.line(line.u, line.v, c);
                            }
                        }
                    }

                    render_object_store.add_object(
                        object,
                        RenderObject::default()
                            // .mesh(&mut meshes, &quad.quad_mesh_polycube, &default_color_map, "default", false)
                            .mesh(&quad.quad_mesh_polycube, &color_map, "colored", true)
                            .gizmo(quad.quad_mesh_polycube.gizmos(colors::GRAY), 2., -0.01, "quads", false)
                            .gizmo(quad.triangle_mesh_polycube.gizmos(colors::GRAY), 2., -0.01, "triangles", false)
                            .gizmo(gizmos_paths, 5., -0.001, "paths", true)
                            .gizmo(gizmos_flat_paths, 4., -0.0011, "flat paths", false)
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
                    for u in [lewp.edges.clone(), vec![lewp.edges[0]], vec![lewp.edges[1]]].concat() {
                        let ut = transform_coordinates(input.position(u), translation, scale);
                        // gizmos_loop.line(line.u, line.v, c);
                        positions.push(vector3d_to_vec3(ut));
                    }

                    let color = colors::from_direction(direction, Some(Perspective::Dual), Some(Orientation::Forwards));
                    let c = bevy::color::Color::srgb(color[0], color[1], color[2]);
                    match direction {
                        PrincipalDirection::X => gizmos_xloops.linestrip(positions, c),
                        PrincipalDirection::Y => gizmos_yloops.linestrip(positions, c),
                        PrincipalDirection::Z => gizmos_zloops.linestrip(positions, c),
                    }
                }

                if let (Ok(lay), Some(polycube)) = (&solution.layout, &solution.polycube) {
                    granulated_mesh = &lay.granulated_mesh;

                    for (&pedge_id, path) in &lay.edge_to_path {
                        let f1 = polycube.structure.normal(polycube.structure.face(pedge_id));
                        let f2 = polycube.structure.normal(polycube.structure.face(polycube.structure.twin(pedge_id)));
                        for vertexpair in path.windows(2) {
                            if granulated_mesh.edge_between_verts(vertexpair[0], vertexpair[1]).is_none() {
                                println!("Edge between {:?} and {:?} does not exist", vertexpair[0], vertexpair[1]);
                                continue;
                            }
                            let edge_id = granulated_mesh.edge_between_verts(vertexpair[0], vertexpair[1]).unwrap().0;
                            let endpoints = granulated_mesh.vertices(edge_id);
                            let u = granulated_mesh.position(endpoints[0]);
                            let v = granulated_mesh.position(endpoints[1]);
                            let line = DrawableLine::from_line(u, v, Vector3D::new(0., 0., 0.), translation, scale);
                            gizmos_flat_paths.line(line.u, line.v, c);
                            if f1 != f2 {
                                gizmos_paths.line(line.u, line.v, c);
                            }
                        }
                    }

                    for &face_id in &polycube.structure.face_ids() {
                        let normal = (polycube.structure.normal(face_id) as Vector3D).normalize();
                        let (dir, side) = to_principal_direction(normal);
                        let color = colors::from_direction(dir, Some(Perspective::Primal), Some(side));
                        for &triangle_id in &lay.face_to_patch[&face_id].faces {
                            color_map_segmentation.insert(triangle_id, color);
                        }
                    }

                    for &triangle_id in &granulated_mesh.face_ids() {
                        let score = *solution.alignment_per_triangle.get_or_panic(triangle_id);
                        let color = colors::map(score as f32, &colors::SCALE_MAGMA);
                        color_map_alignment.insert(triangle_id, color);
                    }
                }

                let mut color_map_flag = HashMap::new();
                let mut gizmos_flag_paths = GizmoAsset::new();
                if let Some(flags) = &solution.external_flag {
                    for (face_id, label) in flags.iter() {
                        let color = match label {
                            0 => colors::RED,
                            1 => colors::RED,
                            4 => colors::YELLOW,
                            5 => colors::YELLOW,
                            2 => colors::BLUE,
                            3 => colors::BLUE,
                            _ => colors::BLACK,
                        };
                        color_map_flag.insert(face_id, color);
                    }

                    for edge_id in input.edge_ids() {
                        let f1 = flags.get(&input.face(edge_id));
                        let f2 = flags.get(&input.face(input.twin(edge_id)));
                        if f1 != f2 {
                            let endpoints = input.vertices(edge_id);
                            let u = input.position(endpoints[0]);
                            let v = input.position(endpoints[1]);
                            let line = DrawableLine::from_line(u, v, Vector3D::new(0., 0., 0.), translation, scale);
                            gizmos_flag_paths.line(line.u, line.v, c);
                        }
                    }
                }

                render_object_store.add_object(
                    object,
                    RenderObject::default()
                        .mesh(input, &default_color_map, "gray", false)
                        .mesh(input, &black_color_map, "black", true)
                        .mesh(granulated_mesh, &color_map_segmentation, "segmentation", false)
                        // .mesh(&mut meshes, granulated_mesh, &color_map_alignment, "alignment", false)
                        .gizmo(input.gizmos(colors::GRAY), 0.5, -0.00001, "wireframe", true)
                        .gizmo(gizmos_xloops, 3., -0.0001, "X-loops", true)
                        .gizmo(gizmos_yloops, 3., -0.00011, "Y-loops", true)
                        .gizmo(gizmos_zloops, 3., -0.000111, "Z-loops", true)
                        .gizmo(gizmos_paths, 4., -0.0001, "paths", false)
                        .gizmo(gizmos_flat_paths, 2., -0.00011, "flat paths", false)
                        .mesh(input, &color_map_flag, "flag", false)
                        .gizmo(gizmos_flag_paths, 2., -1e-4, "flag paths", false)
                        .to_owned(),
                );
            }
        }
    }

    render_object_store
}

/// TODO:

#[derive(Default, Resource)]
pub struct GizmosCache {
    pub raycaster: Vec<Line>,
}

type Line = (Vec3, Vec3, colors::Color);

// Draws the gizmos. This includes all wireframes, vertices, normals, raycasts, etc.
pub fn gizmos(mut gizmos: Gizmos, gizmos_cache: Res<GizmosCache>, configuration: Res<Configuration>) {
    // println!("Drawing gizmos");
    // println!("Gizmos cache: {:?}", gizmos_cache.raycaster);
    if configuration.interactive {
        for &(u, v, c) in &gizmos_cache.raycaster {
            gizmos.line(u, v, bevy::color::Color::srgb(c[0], c[1], c[2]));
        }
    }
}

pub fn add_line2(lines: &mut Vec<Line>, position_a: Vector3D, position_b: Vector3D, offset: Vector3D, color: colors::Color, translation: Vector3D, scale: f64) {
    let line = DrawableLine::from_line(position_a, position_b, offset, translation, scale);
    lines.push((line.u, line.v, color));
}

pub struct DrawableLine {
    pub u: Vec3,
    pub v: Vec3,
}

impl DrawableLine {
    pub fn new(u: Vector3D, v: Vector3D) -> Self {
        Self {
            u: Vec3::new(u.x as f32, u.y as f32, u.z as f32),
            v: Vec3::new(v.x as f32, v.y as f32, v.z as f32),
        }
    }

    pub fn from_line(u: Vector3D, v: Vector3D, offset: Vector3D, translation: Vector3D, scale: f64) -> Self {
        Self::new(
            transform_coordinates(u, translation, scale) + offset,
            transform_coordinates(v, translation, scale) + offset,
        )
    }
}
