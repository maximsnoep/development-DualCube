use crate::colors;
use crate::render::{CameraFor, Objects};
use crate::{Perspective, PrincipalDirection};
use bevy::prelude::*;
use bevy::render::camera::ScalingMode;
use bevy::render::render_resource::{Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages};
use bevy::render::view::RenderLayers;

// Plugin
pub struct AxesGizmoPlugin;

impl Plugin for AxesGizmoPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(AxesGizmoTexture(Handle::default()))
            .add_systems(Startup, setup)
            .add_systems(Update, update);
    }
}

// Public handle to the offscreen texture so you can use it in your UI.
#[derive(Resource, Clone)]
pub struct AxesGizmoTexture(pub Handle<Image>);

/// Tag for the gizmo's offscreen camera.
#[derive(Component)]
struct AxesCamera;

// Dedicated render layer for the axis mini-scene
const AXIS_LAYER: usize = 9;

// Create the axis mini-scene, the offscreen render target, the axis camera, and a UI image to display it.
fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<bevy::render::mesh::Mesh>>,
    mut mats: ResMut<Assets<StandardMaterial>>,
) {
    // We'll use unlit materials so we don't need a light in the mini-scene.
    let mat_x = mats.add(StandardMaterial {
        base_color: colors::to_bevy(colors::from_direction(PrincipalDirection::X, Some(Perspective::Primal), None)),
        unlit: true,
        ..Default::default()
    });
    let mat_y = mats.add(StandardMaterial {
        base_color: colors::to_bevy(colors::from_direction(PrincipalDirection::Y, Some(Perspective::Primal), None)),
        unlit: true,
        ..Default::default()
    });
    let mat_z = mats.add(StandardMaterial {
        base_color: colors::to_bevy(colors::from_direction(PrincipalDirection::Z, Some(Perspective::Primal), None)),
        unlit: true,
        ..Default::default()
    });

    let length = 10.;
    let diameter = 2.;
    let axis_mesh = meshes.add(Cuboid::new(length, diameter, diameter));

    // +X axis
    commands.spawn((
        Mesh3d(axis_mesh.clone()),
        MeshMaterial3d(mat_x.clone()),
        Transform::from_translation(Vec3::X * (length * 0.5)),
        RenderLayers::layer(AXIS_LAYER),
    ));

    // +Y axis
    commands.spawn((
        Mesh3d(axis_mesh.clone()),
        MeshMaterial3d(mat_y.clone()),
        Transform::from_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)).with_translation(Vec3::Y * (length * 0.5)),
        RenderLayers::layer(AXIS_LAYER),
    ));

    // +Z axis
    commands.spawn((
        Mesh3d(axis_mesh.clone()),
        MeshMaterial3d(mat_z.clone()),
        Transform::from_rotation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2)).with_translation(Vec3::Z * (length * 0.5)),
        RenderLayers::layer(AXIS_LAYER),
    ));

    // ---- 3) Axis camera rendering to the offscreen image ----
    let mut image = Image {
        texture_descriptor: TextureDescriptor {
            label: None,
            size: Extent3d {
                width: 256,
                height: 256,
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
    let handle = images.add(image.clone());

    // Expose to the rest of the app
    commands.insert_resource(AxesGizmoTexture(handle.clone()));

    let mut proj = OrthographicProjection::default_3d();
    proj.scale = 0.05;
    // proj.scaling_mode = ScalingMode::FixedVertical { viewport_height: 500. };

    commands.spawn((
        Camera3d::default(),
        bevy::render::camera::Projection::Orthographic(proj),
        Camera {
            target: handle.into(),
            clear_color: bevy::render::camera::ClearColorConfig::Custom(bevy::color::Color::NONE),
            ..Default::default()
        },
        // Only render the axis layer
        RenderLayers::layer(AXIS_LAYER),
        // Place the cam some distance away; its rotation will be synced to main camera
        Transform::from_xyz(0.0, 0.0, 0.0).looking_at(Vec3::ZERO, Vec3::Y),
        AxesCamera,
    ));
}

// This camera is synchronized with main camera (should rotate with it)
fn update(mut axis_cam_q: Query<&mut Transform, With<AxesCamera>>, main_cameras_q: Query<(&GlobalTransform, &CameraFor), With<Camera3d>>) {
    let Ok(mut axis_t) = axis_cam_q.single_mut() else { return };

    // Find your "main" camera based on the user's enum.
    if let Some((main_gt, _)) = main_cameras_q.iter().find(|(_, camera_for)| camera_for.0 == Objects::InputMesh) {
        // Derive forward/up from main camera's rotation.
        let rot = main_gt.rotation();
        let forward = rot * -Vec3::Z;
        let up = rot * Vec3::Y;

        // Keep a consistent distance so the gizmo size is stable.
        let eye = -forward * 500.;

        *axis_t = Transform::from_translation(eye);
        axis_t.look_at(Vec3::ZERO, up);
    }
}
