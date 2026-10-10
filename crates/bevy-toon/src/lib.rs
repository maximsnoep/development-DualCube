use bevy::asset;
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;

// The shader (WESL), embedded in the plugin.
const SHADER: &str = "embedded://bevy_toon/shader.wesl";

pub struct ToonPlugin;
impl Plugin for ToonPlugin {
    fn build(&self, app: &mut App) {
        asset::embedded_asset!(app, "shader.wesl");
        app.add_plugins(MaterialPlugin::<ToonMaterial>::default());
    }
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct ToonMaterial {
    #[uniform(0)]
    pub view_dir: Vec3,
}

impl Material for ToonMaterial {
    fn fragment_shader() -> ShaderRef {
        SHADER.into()
    }
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }
}
