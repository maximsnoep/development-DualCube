use bevy::ecs::system::NonSendMarker;
use bevy::prelude::*;

#[derive(Debug, Default, Clone)]
pub struct WindowIconPlugin {
    path: String,
}

#[derive(Resource)]
struct WindowIconResource {
    path: String,
}

impl Plugin for WindowIconPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(WindowIconResource {
            path: self.path.clone(),
        });
        app.add_systems(Startup, set);
    }
}

impl WindowIconPlugin {
    pub fn with_path(path: &str) -> Self {
        Self {
            path: path.to_string(),
        }
    }
}

fn set(window_icon_resource: Res<WindowIconResource>, _: NonSendMarker) -> Result {
    let icon = image::open(window_icon_resource.path.as_str())?.into_rgba8();
    bevy::winit::WINIT_WINDOWS.with_borrow_mut(|winit_windows| -> Result {
        if winit_windows.windows.is_empty() {
            return Ok(());
        }
        for window in winit_windows.windows.values() {
            window.set_window_icon(Some(winit::window::Icon::from_rgba(
                icon.clone().into_raw(),
                icon.width(),
                icon.height(),
            )?));
        }
        Ok(())
    })
}
