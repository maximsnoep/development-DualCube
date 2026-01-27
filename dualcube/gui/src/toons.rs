use bevy::asset::{load_internal_asset, uuid_handle};
use bevy::prelude::*;

#[derive(
    Asset, bevy::reflect::TypePath, bevy::render::render_resource::AsBindGroup, Debug, Clone,
)]
pub struct ToonsMaterial {
    #[uniform(0)]
    pub view_dir: Vec3,
}

const SHADER_HANDLE: Handle<Shader> = uuid_handle!("ddeed264-efde-495e-9159-4ac3db07f9f8");

impl Material for ToonsMaterial {
    fn fragment_shader() -> bevy::shader::ShaderRef {
        SHADER_HANDLE.into()
    }
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ToonsMaterialPlugin;

impl Plugin for ToonsMaterialPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            SHADER_HANDLE,
            "../assets/toons.wgsl",
            Shader::from_wgsl
        );
        app.add_plugins(MaterialPlugin::<ToonsMaterial>::default());
    }
}
