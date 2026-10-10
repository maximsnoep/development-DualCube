use super::Objects;
use crate::resources::Configuration;
use crate::ui::viewport::ViewResource;
use bevy::camera::visibility::RenderLayers;
use bevy::camera::{CameraOutputMode, RenderTarget, ScalingMode, Viewport};
use bevy::core_pipeline::tonemapping::{DebandDither, Tonemapping};
use bevy::prelude::*;
use bevy::render::render_resource::{
    BlendState, Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
};

use bevy_axes_gizmo::AxesGizmoSyncCamera;
use bevy_egui::{EguiGlobalSettings, PrimaryEguiContext, egui::Rect};
use bevy_orbit_camera::*;
use bevy_toon::ToonMaterial;

const DEFAULT_CAMERA_EYE: Vec3 = Vec3::new(25.0, 25.0, 25.0);
const DEFAULT_CAMERA_TARGET: Vec3 = Vec3::new(0., 0., 0.);
const DEFAULT_CAMERA_TEXTURE_SIZE: u32 = 640 * 2;

#[derive(Component, PartialEq, Eq, Hash, Debug, Copy, Clone, Default)]
pub struct CameraFor(pub Objects);

#[derive(Resource, Default)]
pub struct CameraHandles {
    pub map: std::collections::HashMap<CameraFor, Handle<Image>>,
}

pub fn setup(
    mut commands: Commands<'_, '_>,
    mut egui_global_settings: ResMut<'_, EguiGlobalSettings>,
    mut images: ResMut<'_, Assets<Image>>,
    mut handles: ResMut<'_, CameraHandles>,
    cameras: Query<'_, '_, Entity, With<Camera>>,
    configuration: Res<'_, Configuration>,
) {
    for camera in cameras.iter() {
        commands.entity(camera).despawn();
    }

    egui_global_settings.auto_create_primary_context = false;

    let clear_color = ClearColorConfig::Custom(super::clear_color(&configuration));

    // The UI camera owns the primary egui context and renders no world layers.
    commands.spawn((
        PrimaryEguiContext,
        Camera2d,
        RenderLayers::none(),
        Camera {
            order: 1,
            output_mode: CameraOutputMode::Write {
                blend_state: Some(BlendState::ALPHA_BLENDING),
                clear_color: ClearColorConfig::None,
            },
            clear_color: ClearColorConfig::Custom(Color::NONE),
            ..default()
        },
    ));

    commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: clear_color.clone(),
            ..Default::default()
        },
        AxesGizmoSyncCamera,
        Tonemapping::None,
        DebandDither::Disabled,
        automatic::Marker,
        OrbitCameraBundle::new(
            Controller {
                mouse_rotate_sensitivity: Vec2::splat(0.2),
                mouse_translate_sensitivity: Vec2::splat(2.),
                mouse_wheel_zoom_sensitivity: 0.2,
                ..Default::default()
            },
            DEFAULT_CAMERA_EYE + Vec3::from(Objects::InputMesh),
            DEFAULT_CAMERA_TARGET + Vec3::from(Objects::InputMesh),
            Vec3::Y,
        ),
        CameraFor(Objects::InputMesh),
    ));

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

    for object in configuration.window_shows_object {
        let handle = images.add(image.clone());
        handles.map.insert(CameraFor(object), handle.clone());

        let projection = if uses_orthographic_camera(object) {
            orthographic_projection(30.)
        } else {
            Projection::default()
        };

        commands.spawn((
            Camera3d::default(),
            RenderTarget::Image(handle.into()),
            Camera {
                clear_color: clear_color.clone(),
                ..Default::default()
            },
            projection,
            Tonemapping::None,
            DebandDither::Disabled,
            CameraFor(object),
        ));
    }
}

pub fn update_camera_settings(
    mut camera_controller: Query<'_, '_, &mut Controller>,
    configuration: Res<'_, Configuration>,
    egui_input: Res<'_, bevy_egui::input::EguiWantsInput>,
) {
    let Ok(mut main_camera) = camera_controller.single_mut() else {
        warn_once!("No main camera controller");
        return;
    };

    // The camera does not react to the mouse while it is over the UI (menus, windows).
    let active = if egui_input.wants_any_pointer_input() {
        0.
    } else {
        1.
    };
    main_camera.mouse_rotate_sensitivity =
        Vec2::splat(active * configuration.camera_rotate_sensitivity);
    main_camera.mouse_translate_sensitivity =
        Vec2::splat(active * configuration.camera_translate_sensitivity);
    main_camera.mouse_wheel_zoom_sensitivity = active * configuration.camera_zoom_sensitivity;
}

pub fn update(
    configuration: Res<'_, Configuration>,
    view: Res<'_, ViewResource>,
    mut custom_materials: ResMut<'_, Assets<ToonMaterial>>,
    window: Single<'_, '_, &Window>,
    mut main_camera: Query<'_, '_, (&mut LookTransform, &Transform, &mut Camera), With<Controller>>,
    mut other_cameras: Query<
        '_,
        '_,
        (&mut Transform, &mut Projection, &mut Camera, &CameraFor),
        Without<Controller>,
    >,
) {
    let (mut look, main_transform, mut main_camera) = main_camera.single_mut().unwrap();

    main_camera.is_active = false;
    if let Some(viewport) = view
        .main
        .and_then(|rect| viewport_from_rect(rect, window.physical_size()))
    {
        main_camera.is_active = true;
        if viewport_changed(main_camera.viewport.as_ref(), &viewport) {
            main_camera.viewport = Some(viewport);
        }
    }

    look.up = configuration.camera_up.normalize();

    // Store sub-camera transforms relative to their object origins so every view matches the main camera.
    let normalized_translation = main_transform.translation - Vec3::from(Objects::InputMesh);
    let normalized_rotation = main_transform.rotation;
    let distance = normalized_translation.length().max(0.001);

    for (mut sub_transform, mut sub_projection, mut sub_camera, sub_object) in &mut other_cameras {
        // Only render the views whose tab is shown (rendering every view each frame is expensive).
        let visible = view.shows(sub_object.0);
        if sub_camera.is_active != visible {
            sub_camera.is_active = visible;
        }
        sub_transform.translation = normalized_translation + Vec3::from(sub_object.0);
        sub_transform.rotation = normalized_rotation;
        if let Projection::Orthographic(orthographic) = sub_projection.as_mut() {
            orthographic.scaling_mode = ScalingMode::FixedVertical {
                viewport_height: distance,
            };
        }
    }

    // Toon shading depends on view direction, so keep materials in sync with the camera.
    // `Assets::iter_mut` marks every yielded asset as modified (forcing a GPU re-upload), so only
    // touch the materials when the view direction actually changed.
    if let Some(view_dir) = normalized_translation.try_normalize() {
        let stale = custom_materials
            .iter()
            .any(|(_, material)| material.view_dir.distance_squared(view_dir) > 1e-12);
        if stale {
            for (_, material) in custom_materials.iter_mut() {
                material.view_dir = view_dir;
            }
        }
    }
}

fn uses_orthographic_camera(object: Objects) -> bool {
    matches!(object, Objects::PolycubeMap | Objects::Polycube)
}

fn orthographic_projection(viewport_height: f32) -> Projection {
    Projection::Orthographic(OrthographicProjection {
        scaling_mode: ScalingMode::FixedVertical { viewport_height },
        ..OrthographicProjection::default_3d()
    })
}

fn viewport_from_rect(rect: Rect, window_size: UVec2) -> Option<Viewport> {
    // Bevy panics on zero-sized viewports during window minimization/resizing.
    if window_size.x == 0 || window_size.y == 0 {
        return None;
    }

    let pos = UVec2::new(rect.min[0] as u32, rect.min[1] as u32);
    let end = UVec2::new(rect.max[0] as u32, rect.max[1] as u32);

    // Clamp to the window bounds so the viewport never exceeds the render target.
    let clamped_pos = pos.min(window_size.saturating_sub(UVec2::ONE));
    let clamped_end = end.min(window_size);

    let size = clamped_end.saturating_sub(clamped_pos);

    valid_viewport_size(size.x as f32)?;
    valid_viewport_size(size.y as f32)?;

    Some(Viewport {
        physical_position: clamped_pos,
        physical_size: size,
        ..Default::default()
    })
}

fn valid_viewport_size(value: f32) -> Option<u32> {
    (value > 0. && value.is_finite()).then_some(value as u32)
}

fn viewport_changed(current: Option<&Viewport>, next: &Viewport) -> bool {
    current.is_none_or(|current| {
        current.physical_position != next.physical_position
            || current.physical_size != next.physical_size
            || current.depth != next.depth
    })
}
