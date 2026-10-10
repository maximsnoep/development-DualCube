//! Rendering: cameras, render-object stores, and per-object scene construction.

pub mod camera;
pub mod gizmos;
pub mod objects {
    pub mod input_mesh;
    pub mod polycube;
    pub mod polycube_map;
    pub mod quad_mesh;
}
pub mod store;

use crate::resources::Configuration;
use bevy::prelude::*;
use dualcube::prelude::*;
use store::{RenderObject, RenderObjectStore};

/// Registers the render resources, the cameras, and the systems that keep the renders in sync.
pub struct RenderPlugin;

impl Plugin for RenderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RenderObjectStore>()
            .init_resource::<store::RenderObjectSettingStore>()
            .init_resource::<camera::CameraHandles>()
            .init_gizmo_group::<gizmos::PerpetualGizmos>()
            .add_systems(Startup, (camera::setup, gizmos::setup))
            .add_systems(
                Update,
                (
                    camera::update,
                    camera::update_camera_settings,
                    // Settings must be derived from the store before (re)spawning from them.
                    (store::update_render_settings, store::sync_renders).chain(),
                ),
            );
    }
}

/// The scenes the application can show.
///
/// To add a new scene: add a variant, one line in [`Objects::spec`], and a
/// module with a `build` function that constructs its [`RenderObject`].
#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone, Default)]
pub(crate) enum Objects {
    InputMesh,
    #[default]
    Polycube,
    PolycubeMap,
}

impl Objects {
    pub const ALL: [Self; 3] = [Self::InputMesh, Self::Polycube, Self::PolycubeMap];

    /// The display name and scene builder of each object.
    fn spec(
        self,
    ) -> (
        &'static str,
        fn(&Solution, &Configuration) -> Option<RenderObject>,
    ) {
        match self {
            Self::InputMesh => ("input mesh", objects::input_mesh::build),
            Self::Polycube => ("polycube", objects::polycube::build),
            Self::PolycubeMap => ("polycube-map", objects::polycube_map::build),
        }
    }
}

impl std::fmt::Display for Objects {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.spec().0)
    }
}

/// World-space offset of each scene, spaced far enough apart that the scenes
/// never overlap.
impl From<Objects> for Vec3 {
    fn from(object: Objects) -> Self {
        Self::new(0., 0., 1_000. * object as u8 as f32)
    }
}

/// Builds the complete [`RenderObjectStore`] for the given solution. With `smooth_paths`, the paths of the layout are
/// shown smoothed (see `Solution::smooth_layout`): only in the renders, as the optimizations need the paths of the
/// solution as they are.
#[must_use]
pub fn refresh(solution: &Solution, configuration: &Configuration) -> RenderObjectStore {
    let smoothed = if configuration.smooth_paths {
        smoothed(solution)
    } else {
        None
    };
    let mut store = RenderObjectStore::default();
    for object in Objects::ALL {
        let (_, build) = object.spec();
        // The polycube map is built from the quad mesh, which belongs to the paths of the solution (not smoothed).
        let shown = match (object, &smoothed) {
            (Objects::InputMesh | Objects::Polycube, Some(smoothed)) => smoothed,
            _ => solution,
        };
        if let Some(mut render_object) = build(shown, configuration) {
            // The quad mesh is shown on top of the model (its features are layers of the input mesh).
            if object == Objects::InputMesh
                && let Some(quad) = objects::quad_mesh::build(solution, configuration)
            {
                for label in &quad.labels {
                    render_object.add(&format!("quad {label}"), quad.features[label].clone());
                }
            }
            store.add_object(object, render_object);
        }
    }
    store
}

// The solution with smoothed paths (without a quad mesh), or `None` if it has no layout or the smoothing fails.
pub(crate) fn smoothed(solution: &Solution) -> Option<Solution> {
    solution.layout.as_ref()?;
    let mut smoothed = solution.clone();
    smoothed.smooth_layout().ok()?;
    Some(smoothed)
}

/// How far the negative side of a loop is lightened toward white (as in Polycuber).
pub(crate) const LIGHT_MIX: f32 = 0.4;

/// The configured background color as a Bevy color.
#[allow(unused_qualifications)]
pub(crate) fn clear_color(configuration: &Configuration) -> bevy::color::Color {
    bevy::color::Color::srgb_u8(
        configuration.clear_color[0],
        configuration.clear_color[1],
        configuration.clear_color[2],
    )
}
