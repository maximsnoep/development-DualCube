use bevy::color::Color;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages};
use bevy::render::view::RenderLayers;

// Public handle to the offscreen texture so you can use it in your UI.
#[derive(Resource, Clone)]
pub struct AxesGizmoTexture(pub Handle<Image>);

/// Synchronize AxesGizmoCamera with AxesGizmoSyncCamera
#[derive(Component)]
pub struct AxesGizmoSyncCamera;

/// Plugin for the axes gizmo
#[derive(Resource, Clone)]
pub struct AxesGizmoPlugin {
    pub colors: [Color; 3],
    pub length: f32,
    pub width: f32,
}

//////////////////// Don't look down. ////////////////////
//////////////////// Don't look down. ////////////////////
//////////////////// Don't look down. ////////////////////
//////////////////// Don't look down. ////////////////////
//////////////////// Don't look down. ////////////////////
//////////////////// Don't look down. ////////////////////
//////////////////// Don't look down. ////////////////////
//////////////////// Don't look down. ////////////////////
//////////////////// Don't look down. ////////////////////
//////////////////// Don't look down. ////////////////////
//////////////////// Don't look down. ////////////////////
//////////////////// Don't look down. ////////////////////
//////////////////// Don't look down. ////////////////////
//////////////////// Don't look down. ////////////////////
//////////////////// Don't look down. ////////////////////
//////////////////// Don't look down. ////////////////////
//////////////////// Don't look down. ////////////////////
//////////////////// Don't look down. ////////////////////
//////////////////// Don't look down. ////////////////////
//////////////////// Don't look down. ////////////////////

impl Plugin for AxesGizmoPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(AxesGizmoTexture(Handle::default()))
            .insert_resource(self.clone())
            .add_systems(Startup, setup)
            .add_systems(Update, sync);
    }
}

impl Default for AxesGizmoPlugin {
    fn default() -> Self {
        Self {
            colors: [
                Color::linear_rgb(1.0, 0.0, 0.0),
                Color::linear_rgb(0.0, 1.0, 0.0),
                Color::linear_rgb(0.0, 0.0, 1.0),
            ],
            length: 10.0,
            width: 0.5,
        }
    }
}

#[derive(Component)]
struct AxesGizmoCamera;

// Synchronize AxesGizmoCamera with AxesGizmoSyncCamera
fn sync(mut axis_cam_q: Query<&mut Transform, With<AxesGizmoCamera>>, main_cameras_q: Query<&GlobalTransform, With<AxesGizmoSyncCamera>>) {
    if let Ok(mut axis_t) = axis_cam_q.single_mut() {
        if let Ok(main_gt) = main_cameras_q.single() {
            let rot = main_gt.rotation();
            let forward = rot * -Vec3::Z;
            let up = rot * Vec3::Y;
            let eye = -forward * 500.;

            *axis_t = Transform::from_translation(eye);
            axis_t.look_at(Vec3::ZERO, up);
        }
    }
}

// Dedicated render layer for the axis mini-scene
const AXIS_LAYER: usize = 13;

// Setup the axis mini-scene
fn setup(
    mut commands: Commands,
    plugin_config: Res<AxesGizmoPlugin>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<bevy::render::mesh::Mesh>>,
    mut mats: ResMut<Assets<StandardMaterial>>,
) {
    // Mesh of one axis
    let axis_mesh = meshes.add(Cuboid::new(plugin_config.length, plugin_config.width, plugin_config.width));

    // Spawn the three axes
    // +X axis
    let mat_x = mats.add(StandardMaterial {
        base_color: plugin_config.colors[0],
        unlit: true,
        ..Default::default()
    });
    commands.spawn((
        Mesh3d(axis_mesh.clone()),
        MeshMaterial3d(mat_x.clone()),
        Transform::from_translation(Vec3::X * (plugin_config.length * 0.5)),
        RenderLayers::layer(AXIS_LAYER),
    ));

    // +Y axis
    let mat_y = mats.add(StandardMaterial {
        base_color: plugin_config.colors[1],
        unlit: true,
        ..Default::default()
    });
    commands.spawn((
        Mesh3d(axis_mesh.clone()),
        MeshMaterial3d(mat_y.clone()),
        Transform::from_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)).with_translation(Vec3::Y * (plugin_config.length * 0.5)),
        RenderLayers::layer(AXIS_LAYER),
    ));

    // +Z axis
    let mat_z = mats.add(StandardMaterial {
        base_color: plugin_config.colors[2],
        unlit: true,
        ..Default::default()
    });
    commands.spawn((
        Mesh3d(axis_mesh.clone()),
        MeshMaterial3d(mat_z.clone()),
        Transform::from_rotation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2)).with_translation(Vec3::Z * (plugin_config.length * 0.5)),
        RenderLayers::layer(AXIS_LAYER),
    ));

    // Create the texture
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
    commands.insert_resource(AxesGizmoTexture(handle.clone()));

    // Spawn the camera
    commands.spawn((
        Camera3d::default(),
        bevy::render::camera::Projection::Orthographic(OrthographicProjection {
            scale: 0.1,
            near: 0.0,
            far: 1000.0,
            viewport_origin: Vec2::new(0.5, 0.5),
            scaling_mode: bevy::render::camera::ScalingMode::WindowSize,
            area: Rect::new(-1.0, -1.0, 1.0, 1.0),
        }),
        Camera {
            target: handle.into(),
            clear_color: bevy::render::camera::ClearColorConfig::Custom(bevy::color::Color::NONE),
            ..Default::default()
        },
        RenderLayers::layer(AXIS_LAYER),
        Transform::from_xyz(0.0, 0.0, 0.0).looking_at(Vec3::ZERO, Vec3::Y),
        AxesGizmoCamera,
    ));
}
