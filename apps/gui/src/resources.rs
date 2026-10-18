//! Application-wide state: the configuration, the input mesh, and the
//! current solution.

use crate::controls::InteractiveMode;
use crate::render::Objects;
use crate::render::store::MeshProperties;
use bevy::prelude::*;
use dualcube::prelude::*;
use std::collections::BTreeSet;
use std::sync::Arc;

#[derive(Resource, Debug, Clone)]
pub struct Configuration {
    pub direction: Direction,

    pub omega: usize,
    /// Show the paths of the layout smoothed (see `render::refresh`).
    pub smooth_paths: bool,
    /// The width of the loops (drawn as bands on the model), as a fraction of the diagonal of its bounding box.
    pub loop_width: f64,

    /// The format of the exported figures, and whether they are labelled (see `Job::export_figures`).
    pub figure_format: FigureFormat,
    pub figure_label: bool,

    /// The criterion used to compare solutions (initialization, optimization).
    pub quality: QualityParams,
    /// The optimization (see `Solution::optimize`): the annealing of the loops (its steps per cycle), and the
    /// optimization of the layout and its generations per cycle; the mutations of both that are switched off, by name
    /// (see `LiveSettings`). They can be changed while it runs.
    pub anneal: AnnealParams,
    pub anneal_disabled: BTreeSet<String>,
    pub layout_evolution: LayoutEvolutionParams,
    pub layout_generations: usize,
    pub layout_disabled: BTreeSet<String>,
    /// Cycles without a new best solution after which the optimization stops (0: never).
    pub patience: usize,

    pub loop_anchors: Vec<[EdgeID; 2]>,

    pub interactive_mode: InteractiveMode,

    pub window_shows_object: [Objects; 2],

    pub camera_rotate_sensitivity: f32,
    pub camera_translate_sensitivity: f32,
    pub camera_zoom_sensitivity: f32,

    pub camera_up: Vec3,

    pub clear_color: [u8; 3],
}

impl Default for Configuration {
    fn default() -> Self {
        Self {
            direction: Direction::X,

            omega: 5,
            smooth_paths: true,
            loop_width: 0.006,

            figure_format: FigureFormat::Pdf,
            figure_label: false,

            quality: QualityParams::default(),
            // All mutations on (the adaptive selection chooses among them).
            anneal: AnnealParams::default(),
            anneal_disabled: BTreeSet::new(),
            layout_evolution: LayoutEvolutionParams::default(),
            layout_generations: CoupledParams::default().layout_generations,
            layout_disabled: BTreeSet::new(),
            patience: CoupledParams::default().patience,

            loop_anchors: vec![],

            camera_up: Vec3::Y,

            interactive_mode: InteractiveMode::None,
            window_shows_object: [Objects::Polycube, Objects::PolycubeMap],
            clear_color: if cfg!(feature = "light_mode") {
                [255, 255, 255]
            } else {
                [27, 27, 27]
            },
            camera_rotate_sensitivity: 0.2,
            camera_translate_sensitivity: 2.,
            camera_zoom_sensitivity: 0.2,
        }
    }
}

/// The format of the exported figures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FigureFormat {
    Pdf,
    Svg,
    Png,
}

impl FigureFormat {
    pub const ALL: [Self; 3] = [Self::Pdf, Self::Svg, Self::Png];

    pub const fn extension(self) -> &'static str {
        match self {
            Self::Pdf => "pdf",
            Self::Svg => "svg",
            Self::Png => "png",
        }
    }
}

impl std::fmt::Display for FigureFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.extension())
    }
}

/// The input mesh with its lookup structures and per-axis flow graphs.
#[derive(Default, Debug, Clone, Resource)]
pub struct InputResource {
    /// The name of the model (its file name, without the extension).
    pub name: String,
    pub mesh: Arc<mehsh::prelude::Mesh<INPUT>>,
    pub properties: MeshProperties,
    pub triangle_lookup: FaceLocation<INPUT>,
}

impl InputResource {
    pub fn new(mesh: Arc<mehsh::prelude::Mesh<INPUT>>) -> Self {
        if mesh.nr_verts() == 0 {
            return InputResource::default();
        }
        let triangle_lookup = mesh.bvh();

        let mut properties = MeshProperties::default();
        (properties.scale, properties.translation) = mesh.scale_translation();

        Self {
            name: String::new(),
            mesh,
            properties,
            triangle_lookup,
        }
    }
}

/// The current solution and the candidate solutions per loop seed.
#[derive(Debug, Clone, Resource)]
pub struct SolutionResource {
    pub current_solution: Solution,
    pub selected_corner: Option<VertKey<POLYCUBE>>,
}

impl Default for SolutionResource {
    fn default() -> Self {
        Self {
            current_solution: Solution::new(Arc::new(mehsh::mesh::connectivity::Mesh::default())),
            selected_corner: None,
        }
    }
}
