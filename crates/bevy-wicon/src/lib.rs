//! The icon of the window (title bar and taskbar), from PNG images that are built into the program (so it does not
//! depend on the working directory). It is set as soon as the window exists: the window is created by winit after
//! the startup systems may have run, so the plugin tries every frame until it succeeds.

use bevy::ecs::system::NonSendMarker;
use bevy::prelude::*;

#[derive(Debug, Clone)]
pub struct WindowIconPlugin {
    // PNG images: the small icon (title bar), and the big one (taskbar, Alt+Tab).
    small: &'static [u8],
    big: &'static [u8],
}

impl WindowIconPlugin {
    /// The icon from PNG images (e.g., `include_bytes!`): a small one (e.g., 32 x 32) and a big one.
    #[must_use]
    pub const fn new(small: &'static [u8], big: &'static [u8]) -> Self {
        Self { small, big }
    }
}

#[derive(Resource)]
struct WindowIcon {
    small: &'static [u8],
    big: &'static [u8],
    done: bool,
}

impl Plugin for WindowIconPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(WindowIcon {
            small: self.small,
            big: self.big,
            done: false,
        });
        app.add_systems(Update, set.run_if(|icon: Res<WindowIcon>| !icon.done));
    }
}

fn icon(png: &[u8]) -> Result<winit::window::Icon> {
    let image = image::load_from_memory_with_format(png, image::ImageFormat::Png)?.into_rgba8();
    let (width, height) = image.dimensions();
    Ok(winit::window::Icon::from_rgba(image.into_raw(), width, height)?)
}

fn set(mut window_icon: ResMut<WindowIcon>, _: NonSendMarker) -> Result {
    let (small, big) = (icon(window_icon.small)?, icon(window_icon.big)?);
    let set = bevy::winit::WINIT_WINDOWS.with_borrow(|winit_windows| {
        for window in winit_windows.windows.values() {
            window.set_window_icon(Some(small.clone()));
            #[cfg(target_os = "windows")]
            winit::platform::windows::WindowExtWindows::set_taskbar_icon(&**window, Some(big.clone()));
            #[cfg(not(target_os = "windows"))]
            let _ = &big;
        }
        !winit_windows.windows.is_empty()
    });
    // (Not yet if there is no window yet.)
    window_icon.done = set;
    Ok(())
}
