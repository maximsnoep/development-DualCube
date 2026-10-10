//! Render objects (meshes and gizmos per scene), their visibility settings,
//! and the systems that keep the spawned entities in sync with them.

use super::Objects;
use crate::colors::Kolor;
use crate::resources::Configuration;
use bevy::prelude::*;
use bevy_toon::ToonMaterial;
use mehsh::prelude::*;
use mehsh_bevy;
use std::collections::HashMap;

#[derive(Default, Debug, Clone)]
pub struct MeshProperties {
    pub scale: f64,
    pub translation: Vector3D,
}

/// Marks the input mesh, the target for raycasting in the interactive modes.
#[derive(Component)]
pub struct MainMesh;

/// A single renderable feature of a [`RenderObject`].
#[allow(unused_qualifications)]
#[derive(Clone)]
pub enum RenderAsset {
    Mesh(bevy::mesh::Mesh),
    Gizmo {
        asset: GizmoAsset,
        line_width: f32,
        depth_bias: f32,
    },
}

/// All renderable features of one of the [`Objects`], keyed by label.
#[derive(Clone, Default)]
pub struct RenderObject {
    pub labels: Vec<String>,
    pub features: HashMap<String, RenderAsset>,
}

impl RenderObject {
    pub fn add(&mut self, label: &str, asset: RenderAsset) -> &mut Self {
        self.labels.push(label.to_owned());
        self.features.insert(label.to_owned(), asset);
        self
    }

    pub fn mesh<M: Tag>(
        &mut self,
        mesh: &mehsh::prelude::Mesh<M>,
        color_map: &HashMap<FaceKey<M>, Kolor>,
        label: &str,
    ) -> &mut Self {
        self.add(
            label,
            RenderAsset::Mesh(mehsh_bevy::to_bevy(mesh, color_map).0),
        )
    }

    pub fn gizmo(&mut self, gizmo: GizmoAsset, width: f32, depth: f32, label: &str) -> &mut Self {
        self.add(
            label,
            RenderAsset::Gizmo {
                asset: gizmo,
                line_width: width,
                depth_bias: depth,
            },
        )
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

#[derive(Clone, Default, PartialEq)]
pub struct RenderObjectSetting {
    pub labels: Vec<String>,
    pub settings: HashMap<String, RenderFeatureSetting>,
}

#[derive(Default, Resource)]
pub struct RenderObjectSettingStore {
    pub objects: HashMap<Objects, RenderObjectSetting>,
    last_applied_objects: HashMap<Objects, RenderObjectSetting>,
}

/// Syncs the settings store with the object store: every feature gets a
/// visibility toggle, newly seen features start with their default visibility.
pub fn update_render_settings(
    render_object_store: Res<'_, RenderObjectStore>,
    mut render_settings_store: ResMut<'_, RenderObjectSettingStore>,
) {
    let default = |object: &Objects, label: &str| {
        matches!(
            (object, label),
            (Objects::InputMesh, "lambert")
                | (Objects::InputMesh, "wireframe")
                | (Objects::Polycube, "colored")
                | (Objects::Polycube, "paths")
                | (Objects::Polycube, "flat paths")
                | (Objects::PolycubeMap, "colored")
                | (Objects::PolycubeMap, "triangles")
                | (Objects::PolycubeMap, "paths")
                | (Objects::PolycubeMap, "flat paths")
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

            let new_setting = RenderObjectSetting { labels, settings };
            if render_settings_store.objects.get(object) != Some(&new_setting) {
                render_settings_store.objects.insert(*object, new_setting);
            }
        }
    }
}

/// The handles of a spawned feature (its asset is replaced in place when the feature changes).
#[allow(unused_qualifications)]
enum SpawnedAsset {
    Mesh(Handle<bevy::mesh::Mesh>),
    Gizmo(Handle<GizmoAsset>),
}

/// The spawned entities of the visible features and the covers of the objects, and the shared materials.
#[derive(Default)]
pub struct Spawned {
    features: HashMap<(Objects, String), (Entity, SpawnedAsset)>,
    covers: HashMap<Objects, Entity>,
    materials: Option<[Handle<StandardMaterial>; 2]>,
    toon_material: Option<Handle<ToonMaterial>>,
}

/// Keeps the spawned entities in sync with the object store and the visibility settings. Features that stay visible
/// keep their entities: on new data their assets are replaced in place, so the view never shows an empty frame
/// (despawning and respawning everything made the renders flicker on every update, e.g. during an evolution).
#[allow(unused_qualifications)]
pub fn sync_renders(
    mut commands: Commands<'_, '_>,
    mut meshes: ResMut<'_, Assets<bevy::mesh::Mesh>>,
    mut gizmos: ResMut<'_, Assets<GizmoAsset>>,
    mut materials: ResMut<'_, Assets<StandardMaterial>>,
    mut custom_materials: ResMut<'_, Assets<ToonMaterial>>,
    configuration: Res<'_, Configuration>,
    render_object_store: Res<'_, RenderObjectStore>,
    mut render_settings_store: ResMut<'_, RenderObjectSettingStore>,
    mut spawned: Local<'_, Spawned>,
) {
    let content_changed = render_object_store.is_changed();
    let settings_changed =
        render_settings_store.objects != render_settings_store.last_applied_objects;
    if !content_changed && !settings_changed && !configuration.is_changed() {
        return;
    }

    // The shared materials (created once; the background follows the configured color).
    let [flat_material, background_material] = spawned
        .materials
        .get_or_insert_with(|| {
            [
                materials.add(StandardMaterial {
                    unlit: true,
                    ..default()
                }),
                materials.add(StandardMaterial {
                    unlit: true,
                    ..default()
                }),
            ]
        })
        .clone();
    let background = super::clear_color(&configuration);
    if let Some(mut material) = materials.get_mut(&background_material)
        && material.base_color != background
    {
        material.base_color = background;
    }
    let toon_material = spawned
        .toon_material
        .get_or_insert_with(|| {
            custom_materials.add(ToonMaterial {
                view_dir: Vec3::new(0.0, 0.0, 1.0),
            })
        })
        .clone();
    if !content_changed && !settings_changed {
        return;
    }

    let visible = |object: Objects, label: &str| {
        render_settings_store
            .objects
            .get(&object)
            .and_then(|s| s.settings.get(label))
            .is_some_and(|s| s.visible)
    };

    // Despawn the features that are gone or hidden, and the covers of the objects that are gone.
    spawned.features.retain(|(object, label), (entity, _)| {
        let keep = render_object_store
            .objects
            .get(object)
            .is_some_and(|o| o.features.contains_key(label))
            && visible(*object, label);
        if !keep {
            commands.entity(*entity).despawn();
        }
        keep
    });
    spawned.covers.retain(|object, entity| {
        let keep = render_object_store.objects.contains_key(object);
        if !keep {
            commands.entity(*entity).despawn();
        }
        keep
    });

    for (&object, render_object) in &render_object_store.objects {
        let translation = Vec3::from(object);
        for (label, asset) in &render_object.features {
            if !visible(object, label) {
                continue;
            }
            let key = (object, label.clone());
            // On new data, replace the assets of the spawned features in place.
            match (spawned.features.get(&key), asset) {
                (Some((_, SpawnedAsset::Mesh(handle))), RenderAsset::Mesh(mesh)) => {
                    if content_changed && let Some(mut current) = meshes.get_mut(handle) {
                        *current = mesh.clone();
                    }
                    continue;
                }
                (
                    Some((entity, SpawnedAsset::Gizmo(handle))),
                    RenderAsset::Gizmo {
                        asset,
                        line_width,
                        depth_bias,
                    },
                ) => {
                    if content_changed {
                        if let Some(mut current) = gizmos.get_mut(handle) {
                            *current = asset.clone();
                        }
                        commands.entity(*entity).insert(gizmo(
                            handle.clone(),
                            *line_width,
                            *depth_bias,
                        ));
                    }
                    continue;
                }
                // A feature of another kind under the same label: respawn it.
                (Some((entity, _)), _) => {
                    commands.entity(*entity).despawn();
                }
                (None, _) => {}
            }
            let spawned_feature = match asset {
                RenderAsset::Mesh(mesh) => {
                    let handle = meshes.add(mesh.clone());
                    let mut entity = commands.spawn((
                        Mesh3d(handle.clone()),
                        Transform::from_translation(translation),
                    ));
                    // The polycube-like objects are unlit; the surface meshes are toon-shaded.
                    if matches!(object, Objects::Polycube | Objects::PolycubeMap) {
                        entity.insert(MeshMaterial3d(flat_material.clone()));
                    } else {
                        entity.insert(MeshMaterial3d(toon_material.clone()));
                    }
                    if object == Objects::InputMesh {
                        entity.insert(MainMesh);
                    }
                    (entity.id(), SpawnedAsset::Mesh(handle))
                }
                RenderAsset::Gizmo {
                    asset,
                    line_width,
                    depth_bias,
                } => {
                    let handle = gizmos.add(asset.clone());
                    let entity = commands.spawn((
                        gizmo(handle.clone(), *line_width, *depth_bias),
                        Transform::from_translation(translation),
                    ));
                    (entity.id(), SpawnedAsset::Gizmo(handle))
                }
            };
            spawned.features.insert(key, spawned_feature);
        }

        // A cover such that the object is view-blocked from the others.
        if !spawned.covers.contains_key(&object) {
            let cover = commands
                .spawn((
                    Mesh3d(meshes.add(Sphere::new(400.))),
                    MeshMaterial3d(background_material.clone()),
                    Transform::from_translation(translation),
                ))
                .id();
            spawned.covers.insert(object, cover);
        }
    }

    render_settings_store.last_applied_objects = render_settings_store.objects.clone();
}

fn gizmo(handle: Handle<GizmoAsset>, line_width: f32, depth_bias: f32) -> Gizmo {
    Gizmo {
        handle,
        line_config: GizmoLineConfig {
            width: line_width,
            joints: GizmoLineJoint::Round(4),
            ..Default::default()
        },
        depth_bias,
    }
}
