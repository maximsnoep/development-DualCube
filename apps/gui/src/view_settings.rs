//! The camera and the figure settings, as one line of `key=value` pairs to copy and paste, and the report saved next
//! to exported figures.

use crate::resources::{Configuration, FigureFormat};
use bevy::prelude::*;
use bevy_orbit_camera::LookTransform;
use dualcube::prelude::*;

/// What the line starts with; a pasted text uses its last line starting with it, or else all of it.
const PREFIX: &str = "view:";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewSettings {
    pub eye: Vec3,
    pub target: Vec3,
    /// The camera's up (the screen up is its part orthogonal to the view); applying a view also makes it the orbit
    /// axis. It reads as the orbit axis, or its opposite once the camera has gone over the top.
    pub up: Vec3,
    pub loop_width: f64,
    pub format: FigureFormat,
    pub label: bool,
    pub smooth_paths: bool,
    /// Frame the figures on the model as the camera does (its eye, field of view and window), instead of fitting the
    /// model.
    pub zoom: bool,
    /// The camera's vertical field of view (radians) and the width over the height of its window.
    pub fov: f32,
    pub aspect: f32,
}

/// Asks to move the main camera and adopt the figure settings.
#[derive(Message, Clone, Copy, Debug)]
pub struct ApplyView(pub ViewSettings);

impl ViewSettings {
    pub fn current(look: &LookTransform, projection: &Projection, conf: &Configuration) -> Self {
        let perspective = match projection {
            Projection::Perspective(perspective) => perspective.clone(),
            _ => PerspectiveProjection::default(),
        };
        Self {
            eye: look.eye,
            target: look.target,
            up: look.up,
            loop_width: conf.loop_width,
            format: conf.figure_format,
            label: conf.figure_label,
            smooth_paths: conf.smooth_paths,
            zoom: conf.figure_zoom,
            fov: perspective.fov,
            aspect: perspective.aspect_ratio,
        }
    }

    /// The direction the figures look in, and up on their page.
    pub fn figure_camera(&self) -> (Vector3D, Vector3D) {
        let forward = (self.target - self.eye).normalize_or(Vec3::NEG_Z);
        (vector(forward), vector(self.up))
    }

    pub fn line(&self) -> String {
        let mut line = format!(
            "{PREFIX} eye={} target={} up={} loop_width={} format={} label={} smooth={} zoom={}",
            triple(self.eye),
            triple(self.target),
            triple(self.up),
            self.loop_width,
            self.format.extension(),
            self.label,
            self.smooth_paths,
            self.zoom,
        );
        // The window only matters when zoomed.
        if self.zoom {
            line.push_str(&format!(" fov={} aspect={}", self.fov, self.aspect));
        }
        line
    }

    /// The settings of `text` (a line, or a report containing one); keys it omits keep their value in `base`.
    pub fn parse(text: &str, base: Self) -> Result<Self, String> {
        let line = text
            .lines()
            .rev()
            .find_map(|line| line.trim().strip_prefix(PREFIX))
            .unwrap_or(text);
        let mut settings = base;
        let mut any = false;
        for token in tokens(line) {
            let token = token.as_str();
            let (key, value) = token
                .split_once('=')
                .ok_or_else(|| format!("`{token}` is not key=value"))?;
            match key {
                "eye" => settings.eye = parse_vec3(value)?,
                "target" => settings.target = parse_vec3(value)?,
                "up" => settings.up = parse_vec3(value)?,
                "loop_width" => settings.loop_width = parse_number(value)?,
                "format" => {
                    settings.format = FigureFormat::ALL
                        .into_iter()
                        .find(|f| f.extension().eq_ignore_ascii_case(value))
                        .ok_or_else(|| format!("unknown format `{value}` (pdf, svg or png)"))?;
                }
                "label" => settings.label = parse_bool(value)?,
                "smooth" => settings.smooth_paths = parse_bool(value)?,
                "zoom" => settings.zoom = parse_bool(value)?,
                "fov" => settings.fov = parse_number(value)?,
                "aspect" => settings.aspect = parse_number(value)?,
                _ => return Err(format!("unknown key `{key}`")),
            }
            any = true;
        }
        if !any {
            return Err("nothing to read".to_owned());
        }
        if (settings.target - settings.eye).length_squared() == 0. {
            return Err("eye and target coincide".to_owned());
        }
        if settings.up.length_squared() == 0. {
            return Err("up is zero".to_owned());
        }
        if !(settings.fov > 0. && settings.fov < std::f32::consts::PI) {
            return Err("fov must lie between 0 and pi".to_owned());
        }
        if !(settings.aspect > 0.) {
            return Err("aspect must be positive".to_owned());
        }
        Ok(settings)
    }
}

/// What the report says about the model and the saved files.
pub struct Report<'a> {
    pub name: &'a str,
    pub source: Option<&'a std::path::Path>,
    pub save: Option<&'a std::path::Path>,
    pub figures: &'a [std::path::PathBuf],
    pub vertices: usize,
    pub faces: usize,
    pub date: &'a str,
    pub configuration: &'a Configuration,
}

impl ViewSettings {
    /// A text describing how the figures were made; pasting all of it restores the view.
    pub fn report(&self, report: &Report<'_>) -> String {
        let file_name = |path: &std::path::Path| {
            path.file_name().map_or_else(
                || path.display().to_string(),
                |n| n.to_string_lossy().into_owned(),
            )
        };
        let (view, _) = self.figure_camera();
        let lines = vec![
            "DualCube figures. Load the save, paste this text (or its last line) into the box under View > Camera, and click apply pasted."
                .to_owned(),
            String::new(),
            format!("model: {}", report.name),
            format!(
                "source: {}",
                report.source.map_or("unknown".to_owned(), |p| p.display().to_string())
            ),
            format!("mesh: {} vertices, {} faces", report.vertices, report.faces),
            format!("save: {}", report.save.map_or("not written".to_owned(), file_name)),
            format!(
                "figures: {}",
                report.figures.iter().map(|p| file_name(p)).collect::<Vec<_>>().join(", ")
            ),
            format!("date: {} (UTC)", report.date),
            String::new(),
            format!("camera eye: {}", spaced(self.eye)),
            format!("camera target: {}", spaced(self.target)),
            format!("camera up: {}", spaced(self.up)),
            format!("figure view direction: {} {} {}", view.x, view.y, view.z),
            format!("loop width: {}", self.loop_width),
            format!("format: {}", self.format.extension()),
            format!("label: {}", self.label),
            format!("smooth paths: {}", self.smooth_paths),
            format!(
                "camera zoom: {}",
                if self.zoom {
                    format!("on (fov {} rad, aspect {})", self.fov, self.aspect)
                } else {
                    "off (fitted to the model)".to_owned()
                }
            ),
            format!("initialization: {:?}", report.configuration.initialization),
            format!("skeleton options: {:?}", report.configuration.skeleton_options),
            String::new(),
            self.line(),
            String::new(),
        ];
        lines.join("\n")
    }
}

// The `key=value` pairs, allowing spaces inside a vector (`eye=(1, 2, 3)`).
fn tokens(line: &str) -> Vec<String> {
    let mut tokens: Vec<String> = vec![];
    let mut current = String::new();
    for c in line.chars() {
        if c.is_whitespace() {
            if !current.is_empty() && !current.ends_with([',', '=', '(', '[']) {
                tokens.push(std::mem::take(&mut current));
            }
        } else if matches!(c, ',' | ')' | ']') && current.is_empty() && !tokens.is_empty() {
            current = tokens.pop().unwrap_or_default();
            current.push(c);
        } else {
            current.push(c);
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn vector(v: Vec3) -> Vector3D {
    Vector3D::new(f64::from(v.x), f64::from(v.y), f64::from(v.z))
}

fn triple(v: Vec3) -> String {
    format!("{},{},{}", v.x, v.y, v.z)
}

fn spaced(v: Vec3) -> String {
    format!("{} {} {}", v.x, v.y, v.z)
}

fn parse_number<T: std::str::FromStr>(value: &str) -> Result<T, String> {
    value
        .parse()
        .map_err(|_| format!("`{value}` is not a number"))
}

fn parse_vec3(value: &str) -> Result<Vec3, String> {
    let parts = value
        .trim_matches(|c| matches!(c, '(' | ')' | '[' | ']'))
        .split(',')
        .map(|part| parse_number::<f32>(part.trim()))
        .collect::<Result<Vec<_>, _>>()?;
    match parts[..] {
        [x, y, z] if x.is_finite() && y.is_finite() && z.is_finite() => Ok(Vec3::new(x, y, z)),
        _ => Err(format!("`{value}` is not three finite numbers x,y,z")),
    }
}

fn parse_bool(value: &str) -> Result<bool, String> {
    match value {
        "true" | "1" | "yes" => Ok(true),
        "false" | "0" | "no" => Ok(false),
        _ => Err(format!("`{value}` is not true or false")),
    }
}

#[cfg(test)]
mod tests;
