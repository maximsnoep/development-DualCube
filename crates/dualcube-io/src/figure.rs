//! Figures of a solution for papers, one per view preset of the GUI (the input mesh, the skeleton stages, the loop
//! structure, the polycube segmentation, the quad mesh) and the polycube, as SVG, PDF (vector) or PNG.
//!
//! All figures share one view direction. The polycube and the cubes are drawn in a parallel (axonometric) projection,
//! the rest in a perspective projection fitted to the input mesh, so that those line up exactly. Every figure is a
//! square of the same size with the shape fitted inside it; optionally with a label below it (the input, the object,
//! the score, and the date). Skeleton graphs are drawn over a light rendering of the input mesh, as if seen through it.
//!
//! Hidden surfaces are removed with the painter's algorithm: the faces that face the camera are drawn back to front,
//! and every line on the surface is drawn right after the (last drawn) face it lies on, so the faces in front of it
//! cover it. The polycube's faces are first cut on the lattice of all its coordinates: for axis-aligned faces in a
//! parallel projection, the order of the cells along the view is then exact (as in Polycuber).
//!
//! The PDF and PNG are converted from the SVG, so all formats show the same.

use crate::SkeletonSave;
use anyhow::{Context, anyhow, bail};
use dualcube::prelude::*;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

/// A color (sRGB, components in [0, 1]).
pub type Rgb = [f32; 3];

/// The figures of a solution: one per view preset of the GUI, in its order, and the polycube.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Figure {
    /// The input mesh (shaded).
    InputMesh,
    /// The contracted (deflated) mesh.
    Deflated,
    /// The skeleton before refinement.
    RawSkeleton,
    /// The skeleton as labeled (with its junction splits and repairs).
    Skeleton,
    /// The skeleton with gray nodes.
    GraySkeleton,
    /// The input mesh colored by the skeleton's patches, with their boundaries.
    Patches,
    /// The boundaries of the skeleton's patches (its loops) on the plain input mesh.
    SkeletonLoops,
    /// The patches after optimization.
    OptimizedPatches,
    /// A minimal cut system: g boundary loops of the optimized skeleton off a spanning tree (spread out), on the plain
    /// input mesh. Cutting along their disks leaves a ball.
    CutSystem,
    /// The patches (light) with the skeleton graph.
    Combined,
    /// The optimized patches (light) with the optimized skeleton graph.
    OptimizedCombined,
    /// The optimized skeleton graph with the axis labels of its edges.
    OptimizedSkeleton,
    /// The tree labeling, before cycle repairs.
    TreeLabeled,
    /// The skeleton graph with the axis labels of its edges.
    Labeled,
    /// The labeled skeleton at its placed node positions (the cube centers), in a parallel projection.
    PlacedSkeleton,
    /// The node cubes on their positions, with the labeled edges between them, as in the placed skeleton.
    NodeCubes,
    /// The cube placement, in a parallel projection.
    Cubes,
    /// The cube placement with only the node cubes colored (by patch).
    CubesPrimal,
    /// The cube placement with only the edge cubes colored (by axis).
    CubesDual,
    /// The labeled patch boundaries on the input mesh, before the template's other loops are drawn.
    LiftedLoops,
    /// The loops on the input mesh.
    LoopStructure,
    /// The input mesh colored by the patches of the layout, with their boundaries (the paths).
    Segmentation,
    /// The quad mesh, colored by the axes of its polycube faces, with its edges and irregular edge loops.
    Quad,
    /// The polycube, in a parallel projection.
    Polycube,
}

impl Figure {
    pub const ALL: [Self; 24] = [
        Self::InputMesh,
        Self::Deflated,
        Self::RawSkeleton,
        Self::Skeleton,
        Self::GraySkeleton,
        Self::Patches,
        Self::SkeletonLoops,
        Self::OptimizedPatches,
        Self::CutSystem,
        Self::Combined,
        Self::OptimizedCombined,
        Self::OptimizedSkeleton,
        Self::TreeLabeled,
        Self::Labeled,
        Self::PlacedSkeleton,
        Self::NodeCubes,
        Self::Cubes,
        Self::CubesPrimal,
        Self::CubesDual,
        Self::LiftedLoops,
        Self::LoopStructure,
        Self::Segmentation,
        Self::Quad,
        Self::Polycube,
    ];

    /// The name of the object (in the label).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::InputMesh => "input mesh",
            Self::Deflated => "deflated mesh",
            Self::RawSkeleton => "raw skeleton",
            Self::Skeleton => "skeleton",
            Self::GraySkeleton => "gray skeleton",
            Self::Patches => "skeleton patches",
            Self::SkeletonLoops => "skeleton loops",
            Self::OptimizedPatches => "optimized patches",
            Self::CutSystem => "minimal cut system",
            Self::Combined => "patches and skeleton",
            Self::OptimizedCombined => "optimized patches and skeleton",
            Self::OptimizedSkeleton => "optimized skeleton",
            Self::TreeLabeled => "tree labeled skeleton",
            Self::Labeled => "labeled skeleton",
            Self::PlacedSkeleton => "placed skeleton",
            Self::NodeCubes => "node cubes",
            Self::Cubes => "cube placement",
            Self::CubesPrimal => "cube placement (primal)",
            Self::CubesDual => "cube placement (dual)",
            Self::LiftedLoops => "lifted loops",
            Self::LoopStructure => "loop structure",
            Self::Segmentation => "polycube segmentation",
            Self::Quad => "quad mesh",
            Self::Polycube => "polycube",
        }
    }

    /// A short name (in file names).
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::InputMesh => "mesh",
            Self::Deflated => "deflated",
            Self::RawSkeleton => "raw",
            Self::Skeleton => "skeleton",
            Self::GraySkeleton => "skeleton_gray",
            Self::Patches => "patches",
            Self::SkeletonLoops => "skeleton_loops",
            Self::OptimizedPatches => "optimized",
            Self::CutSystem => "cut_system",
            Self::Combined => "combined",
            Self::OptimizedCombined => "optimized_combined",
            Self::OptimizedSkeleton => "optimized_skeleton",
            Self::TreeLabeled => "tree",
            Self::Labeled => "labeled",
            Self::PlacedSkeleton => "placed_skeleton",
            Self::NodeCubes => "node_cubes",
            Self::Cubes => "cubes",
            Self::CubesPrimal => "cubes_primal",
            Self::CubesDual => "cubes_dual",
            Self::LiftedLoops => "lifted",
            Self::LoopStructure => "loops",
            Self::Segmentation => "segmentation",
            Self::Quad => "quad",
            Self::Polycube => "polycube",
        }
    }

    // Drawn in perspective, fitted to the input mesh.
    const fn on_model(self) -> bool {
        !self.on_placement() && !matches!(self, Self::Polycube)
    }

    // Drawn in a parallel projection, fitted to the cubes (so that they line up).
    const fn on_placement(self) -> bool {
        matches!(
            self,
            Self::PlacedSkeleton
                | Self::NodeCubes
                | Self::Cubes
                | Self::CubesPrimal
                | Self::CubesDual
        )
    }
}

/// Which cubes of the cube placement are colored; the others are gray.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CubeColoring {
    /// The node cubes by patch, the edge cubes by axis.
    Combined,
    /// Only the node cubes, by patch.
    Primal,
    /// Only the edge cubes, by axis.
    Dual,
}

impl CubeColoring {
    #[must_use]
    pub const fn colors_nodes(self) -> bool {
        !matches!(self, Self::Dual)
    }

    #[must_use]
    pub const fn colors_edges(self) -> bool {
        !matches!(self, Self::Primal)
    }
}

/// The colors of the figures.
#[derive(Clone, Debug)]
pub struct FigureStyle {
    /// The patches and polycube faces, per axis (X, Y, Z) of their normal.
    pub primal: [Rgb; 3],
    /// The loops, per axis.
    pub dual: [Rgb; 3],
    /// The input mesh.
    pub surface: Rgb,
    /// The input mesh under the loops (dark charcoal, as in the GUI's dual view).
    pub dark: Rgb,
    /// The paths (patch boundaries) and the edges of the polycube.
    pub line: Rgb,
    /// How far the negative side of a loop is lightened toward white (as in Polycuber).
    pub light_mix: f32,
}

const fn rgb(r: u8, g: u8, b: u8) -> Rgb {
    [r as f32 / 255., g as f32 / 255., b as f32 / 255.]
}

impl Default for FigureStyle {
    /// The default colors of the GUI (Nord).
    fn default() -> Self {
        let nord = [rgb(191, 97, 106), rgb(129, 161, 193), rgb(235, 203, 139)];
        Self {
            primal: nord,
            dual: nord,
            surface: [0.6, 0.6, 0.6],
            dark: [0.2, 0.2, 0.2],
            line: [0.145, 0.145, 0.145],
            light_mix: 0.4,
        }
    }
}

/// The label below a figure (the object's name is added per figure).
#[derive(Clone, Debug)]
pub struct Annotation {
    /// The name of the input (e.g., its file name).
    pub input: String,
    /// The score (quality) of the solution.
    pub score: Option<f64>,
    /// The date (see [`today`]).
    pub date: String,
}

/// How the figures are drawn.
#[derive(Clone, Debug)]
pub struct FigureParams {
    /// The direction the camera looks in.
    pub view: Vector3D,
    /// The direction that is up on the page (made orthogonal to `view`).
    pub up: Vector3D,
    /// The width and height of the figure (without the label), in points.
    pub size: f64,
    /// The width of the loops and of the paths (bands), as fractions of the diagonal of the bounding box of the mesh.
    pub loop_width: f64,
    pub path_width: f64,
    pub annotation: Option<Annotation>,
}

impl Default for FigureParams {
    /// An isometric view (as the GUI's default camera), 300 points wide, without a label.
    fn default() -> Self {
        Self {
            view: -Vector3D::new(1., 1., 1.).normalize(),
            up: Vector3D::new(0., 1., 0.),
            size: 300.,
            loop_width: 0.008,
            path_width: 0.004,
            annotation: None,
        }
    }
}

// The margin around the shape, the widths of the lines, and the font size, as fractions of the figure's size.
const MARGIN: f64 = 0.04;
const POLYCUBE_EDGE_WIDTH: f64 = 0.004;
const FONT_SIZE: f64 = 0.032;
// The vertical field of view of the perspective projection (degrees).
const FIELD_OF_VIEW: f64 = 30.;
// The width of the outline of a face in its own color (points), so that neighboring faces meet without hairline gaps.
const SEAM: f64 = 0.25;
// Pixels per point of a PNG.
const PNG_SCALE: f32 = 4.;

/// The SVG of a figure of the solution (and of the skeleton stages, if given), or `None` if it does not have it yet
/// (e.g., no skeleton, loops, layout, quad mesh or polycube).
#[must_use]
pub fn figure_svg(
    solution: &Solution,
    skeleton: Option<&SkeletonSave>,
    figure: Figure,
    params: &FigureParams,
    style: &FigureStyle,
) -> Option<String> {
    let model = Model {
        solution,
        skeleton: skeleton.unwrap_or(&NO_SKELETON),
        camera: model_camera(solution, params),
        params,
        style,
    };
    let items = match figure {
        Figure::InputMesh => input_mesh(&model),
        Figure::Deflated => skeleton::deflated(&model)?,
        Figure::RawSkeleton => skeleton::graph(&model, skeleton::Graph::Raw)?,
        Figure::Skeleton => skeleton::graph(&model, skeleton::Graph::Current)?,
        Figure::GraySkeleton => skeleton::graph(&model, skeleton::Graph::Gray)?,
        Figure::Patches => skeleton::patches(&model, false)?,
        Figure::SkeletonLoops => skeleton::patch_boundaries(&model)?,
        Figure::OptimizedPatches => skeleton::patches(&model, true)?,
        Figure::CutSystem => skeleton::cut_system(&model)?,
        Figure::Combined => skeleton::combined(&model, false)?,
        Figure::OptimizedCombined => skeleton::combined(&model, true)?,
        Figure::OptimizedSkeleton => skeleton::graph(&model, skeleton::Graph::Optimized)?,
        Figure::TreeLabeled => skeleton::graph(&model, skeleton::Graph::Tree)?,
        Figure::Labeled => skeleton::graph(&model, skeleton::Graph::Labeled)?,
        Figure::PlacedSkeleton => skeleton::placed_skeleton(&model)?,
        Figure::NodeCubes => skeleton::node_cubes(&model)?,
        Figure::Cubes => skeleton::cubes(&model, CubeColoring::Combined)?,
        Figure::CubesPrimal => skeleton::cubes(&model, CubeColoring::Primal)?,
        Figure::CubesDual => skeleton::cubes(&model, CubeColoring::Dual)?,
        Figure::LiftedLoops => lifted_loops(&model)?,
        Figure::LoopStructure => loop_structure(&model)?,
        Figure::Segmentation => segmentation(&model)?,
        Figure::Quad => skeleton::quad(&model)?,
        Figure::Polycube => polycube(solution, params, style)?,
    };
    if items.is_empty() {
        return None;
    }
    let frame = if figure.on_model() {
        let mesh = &*solution.mesh_ref;
        mesh.vert_ids()
            .into_iter()
            .map(|v| model.camera.project(mesh.position(v)))
            .collect()
    } else if figure.on_placement() {
        skeleton::placement_frame(&model)
    } else {
        vec![]
    };
    let label = params.annotation.as_ref().map(|a| {
        let mut parts = vec![a.input.clone(), figure.name().to_owned()];
        if let Some(score) = a.score {
            parts.push(format!("score {score:.4}"));
        }
        parts.push(a.date.clone());
        parts.join(" \u{b7} ")
    });
    Some(to_svg(items, &frame, params.size, label.as_deref()))
}

/// Saves a figure (an SVG, see [`figure_svg`]) in the format given by the extension of the path: SVG, PDF or PNG.
pub fn save_figure(svg: &str, path: &Path) -> anyhow::Result<()> {
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    let bytes = match extension.as_deref() {
        Some("svg") => svg.as_bytes().to_vec(),
        Some("pdf") => svg2pdf::to_pdf(
            &parse(svg)?,
            svg2pdf::ConversionOptions::default(),
            svg2pdf::PageOptions::default(),
        )
        .map_err(|e| anyhow!("PDF conversion failed: {e}"))?,
        Some("png") => {
            let tree = parse(svg)?;
            let size = tree
                .size()
                .to_int_size()
                .scale_by(PNG_SCALE)
                .context("empty figure")?;
            let mut pixmap = resvg::tiny_skia::Pixmap::new(size.width(), size.height())
                .context("figure too large")?;
            resvg::render(
                &tree,
                resvg::tiny_skia::Transform::from_scale(PNG_SCALE, PNG_SCALE),
                &mut pixmap.as_mut(),
            );
            pixmap.encode_png()?
        }
        _ => bail!("Unknown figure format: {path:?} (use svg, pdf or png)"),
    };
    std::fs::write(path, bytes).with_context(|| format!("writing {path:?}"))
}

/// Saves all figures the solution (with the skeleton stages, if given) has, as `<dir>/<stem>_<figure>.<extension>`;
/// returns their paths.
pub fn export_figures(
    solution: &Solution,
    skeleton: Option<&SkeletonSave>,
    dir: &Path,
    stem: &str,
    extension: &str,
    params: &FigureParams,
    style: &FigureStyle,
) -> anyhow::Result<Vec<PathBuf>> {
    let mut saved = vec![];
    for figure in Figure::ALL {
        let Some(svg) = figure_svg(solution, skeleton, figure, params, style) else {
            continue;
        };
        let path = dir.join(format!("{stem}_{}.{extension}", figure.slug()));
        save_figure(&svg, &path)?;
        saved.push(path);
    }
    Ok(saved)
}

/// Today's date (UTC), as YYYY-MM-DD.
#[must_use]
pub fn today() -> String {
    date(seconds_now() / 86_400)
}

/// The current time (UTC), as YYYY-MM-DD_HHMMSS (sorts chronologically, valid in file names).
#[must_use]
pub fn timestamp() -> String {
    let seconds = seconds_now();
    let time = seconds % 86_400;
    format!(
        "{}_{:02}{:02}{:02}",
        date(seconds / 86_400),
        time / 3600,
        time / 60 % 60,
        time % 60
    )
}

fn seconds_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs()) as i64
}

// The date of a day since the Unix epoch, as YYYY-MM-DD.
fn date(days: i64) -> String {
    // Civil from days (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

// The SVG parsed for conversion, with the system fonts (for the label; loaded once).
fn parse(svg: &str) -> anyhow::Result<resvg::usvg::Tree> {
    static FONTS: OnceLock<Arc<resvg::usvg::fontdb::Database>> = OnceLock::new();
    let fonts = FONTS.get_or_init(|| {
        let mut database = resvg::usvg::fontdb::Database::new();
        database.load_system_fonts();
        Arc::new(database)
    });
    let options = resvg::usvg::Options {
        fontdb: fonts.clone(),
        ..Default::default()
    };
    Ok(resvg::usvg::Tree::from_str(svg, &options)?)
}

// ---------------------------------------------------------------------------------------------------------------------
// Projection.

// A camera: orthonormal axes (`forward` is the viewing direction), and the eye of a perspective projection (`None`:
// parallel).
struct Camera {
    right: Vector3D,
    up: Vector3D,
    forward: Vector3D,
    eye: Option<Vector3D>,
}

impl Camera {
    // A camera along the view of the parameters; in perspective, it sees all the points (in its field of view).
    fn new(params: &FigureParams, perspective: Option<&[Vector3D]>) -> Self {
        let forward = params.view.try_normalize(1e-12).unwrap_or(-Vector3D::z());
        let mut up = params.up - forward * params.up.dot(&forward);
        if up.norm() < 1e-9 {
            // The view is along `up`: any other direction will do.
            let other = if forward.x.abs() < 0.9 {
                Vector3D::x()
            } else {
                Vector3D::y()
            };
            up = other - forward * other.dot(&forward);
        }
        let up = up.normalize();
        let right = forward.cross(&up);
        let eye = perspective.map(|points| {
            let (min, max) = bounds(points);
            let center = (min + max) / 2.;
            let radius = points
                .iter()
                .map(|p| (p - center).norm())
                .fold(0., f64::max)
                .max(1e-12);
            center - forward * (radius / (FIELD_OF_VIEW.to_radians() / 2.).sin())
        });
        Self {
            right,
            up,
            forward,
            eye,
        }
    }

    // A point on the page (y up, before the figure is fitted).
    fn project(&self, p: Vector3D) -> [f64; 2] {
        match self.eye {
            Some(eye) => {
                let q = p - eye;
                let z = q.dot(&self.forward).max(1e-12);
                [q.dot(&self.right) / z, q.dot(&self.up) / z]
            }
            None => [p.dot(&self.right), p.dot(&self.up)],
        }
    }

    // The distance along the view (larger is farther).
    fn depth(&self, p: Vector3D) -> f64 {
        match self.eye {
            Some(eye) => (p - eye).norm(),
            None => p.dot(&self.forward),
        }
    }

    // Whether a face at `p` with this normal faces the camera.
    fn faces(&self, p: Vector3D, normal: Vector3D) -> bool {
        match self.eye {
            Some(eye) => normal.dot(&(eye - p)) > 0.,
            None => normal.dot(&self.forward) < 0.,
        }
    }

    // The brightness of a face with this normal: a light above and to the left of the camera.
    fn shade(&self, normal: Vector3D) -> f32 {
        let light = (-self.forward + self.up * 0.6 - self.right * 0.4).normalize();
        let diffuse = normal
            .try_normalize(1e-12)
            .map_or(0., |n| n.dot(&light).max(0.));
        (0.55 + 0.45 * diffuse) as f32
    }
}

fn bounds(points: &[Vector3D]) -> (Vector3D, Vector3D) {
    points.iter().fold(
        (
            Vector3D::repeat(f64::INFINITY),
            Vector3D::repeat(f64::NEG_INFINITY),
        ),
        |(min, max), p| (min.inf(p), max.sup(p)),
    )
}

fn scaled(color: Rgb, factor: f32) -> Rgb {
    color.map(|c| (c * factor).clamp(0., 1.))
}

// ---------------------------------------------------------------------------------------------------------------------
// Drawing: polygons and lines on the page (before fitting), in the order they are drawn.

enum Item {
    Polygon {
        points: Vec<[f64; 2]>,
        fill: Rgb,
    },
    // Width as a fraction of the figure's size.
    Line {
        from: [f64; 2],
        to: [f64; 2],
        color: Rgb,
        width: f64,
    },
    // A dot with a ring around it; radius and ring width as fractions of the figure's size.
    Disk {
        center: [f64; 2],
        radius: f64,
        fill: Rgb,
        ring: Rgb,
    },
}

// What a figure is drawn from: the solution, the skeleton stages, and the perspective camera of the input mesh.
struct Model<'a> {
    solution: &'a Solution,
    skeleton: &'a SkeletonSave,
    camera: Camera,
    params: &'a FigureParams,
    style: &'a FigureStyle,
}

static NO_SKELETON: SkeletonSave = SkeletonSave {
    extraction: None,
    labeling: None,
    labeled: None,
    realized: None,
    optimized: None,
    drawing: None,
    loops: None,
    failure: None,
    output: None,
};

// A face of a surface; one without a color hides what is behind it but is not drawn.
struct Face {
    corners: Vec<Vector3D>,
    normal: Vector3D,
    color: Option<Rgb>,
}

fn mesh_faces<M: Tag>(mesh: &Mesh<M>, color: impl Fn(FaceKey<M>) -> Option<Rgb>) -> Vec<Face> {
    mesh.face_ids()
        .into_iter()
        .map(|face| Face {
            corners: mesh.vertices(face).map(|v| mesh.position(v)).collect(),
            normal: mesh.normal(face),
            color: color(face),
        })
        .collect()
}

// How the faces of a surface are drawn: also those facing away (for surfaces that are not closed and consistently
// oriented), how far their shaded colors are lightened toward white, and how far their shading is flattened (1: one
// uniform color).
#[derive(Clone, Copy, Default)]
struct Look {
    two_sided: bool,
    lighten: f32,
    flatten: f32,
}

// A line on a surface (width as a fraction of the figure's size); drawn where the surface does not hide it.
struct SurfaceLine {
    points: Vec<Vector3D>,
    color: Rgb,
    width: f64,
}

fn mix(color: Rgb, toward: Rgb, t: f32) -> Rgb {
    [0, 1, 2].map(|k| mix_scalar(color[k], toward[k], t))
}

fn mix_scalar(value: f32, toward: f32, t: f32) -> f32 {
    value + (toward - value) * t
}

/// Muted colors for the regions (patches, nodes).
pub const PASTEL: [Rgb; 10] = [
    [0.561, 0.725, 0.788], // dusty blue
    [0.890, 0.710, 0.561], // sand
    [0.663, 0.780, 0.608], // sage
    [0.831, 0.604, 0.604], // dusty rose
    [0.718, 0.651, 0.820], // lavender
    [0.902, 0.835, 0.549], // straw
    [0.561, 0.776, 0.706], // seafoam
    [0.780, 0.635, 0.486], // tan
    [0.616, 0.663, 0.788], // slate
    [0.839, 0.706, 0.784], // mauve
];

/// A color per node of the skeleton stages, shared by all views (as in the GUI). Node IDs carry over between stages,
/// so the stages are colored together, the first (the optimized skeleton, else the labeled one) before the nodes only
/// the later ones have. Greedily: none of the colors of its already colored neighbors in any stage (the fewest of them
/// if all are taken), then the least used, then the next in the cycle after the previous pick.
#[must_use]
pub fn region_colors(
    stages: &[&dualcube::skeleton::Skeleton],
) -> HashMap<dualcube::skeleton::SkeletonNodeID, Rgb> {
    let mut neighbors: HashMap<_, Vec<_>> = HashMap::new();
    for edge in stages.iter().flat_map(|s| s.edges.values()) {
        let [a, b] = edge.nodes;
        if a != b {
            neighbors.entry(a).or_default().push(b);
            neighbors.entry(b).or_default().push(a);
        }
    }
    let mut chosen = HashMap::new();
    let mut used = [0usize; PASTEL.len()];
    let mut next = 0;
    for id in stages.iter().flat_map(|s| s.nodes.keys()) {
        if chosen.contains_key(&id) {
            continue;
        }
        let mut clashes = [0usize; PASTEL.len()];
        for n in neighbors.get(&id).into_iter().flatten() {
            if let Some(&c) = chosen.get(n) {
                clashes[c] += 1;
            }
        }
        let color = (0..PASTEL.len())
            .min_by_key(|&c| {
                (
                    clashes[c],
                    used[c],
                    (c + PASTEL.len() - next) % PASTEL.len(),
                )
            })
            .unwrap();
        chosen.insert(id, color);
        used[color] += 1;
        next = (color + 1) % PASTEL.len();
    }
    chosen.into_iter().map(|(id, c)| (id, PASTEL[c])).collect()
}

// A band on a surface (flat colored): a polyline (closed: a loop) and the offsets of its other side. Drawn as one
// polygon per stretch that shows, so that it has no seams between its segments.
struct SurfaceMark {
    points: Vec<Vector3D>,
    offsets: Vec<Vector3D>,
    closed: bool,
    color: Rgb,
}

// The resolution of the depth buffer (pixels across), and the depth (as a fraction of the size of the shape) by which a
// mark may lie behind the surface and still be drawn.
const DEPTH_RESOLUTION: usize = 2048;
const DEPTH_TOLERANCE: f64 = 0.01;

// The depth of the nearest face per pixel (of the projected faces).
struct DepthBuffer {
    min: [f64; 2],
    cell: f64,
    depth: Vec<f32>,
}

impl DepthBuffer {
    // The faces as projected points with their depths (fanned into triangles).
    fn new(faces: &[Vec<([f64; 2], f64)>]) -> Self {
        let (mut min, mut max) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for (p, _) in faces.iter().flatten() {
            for k in 0..2 {
                min[k] = min[k].min(p[k]);
                max[k] = max[k].max(p[k]);
            }
        }
        let n = DEPTH_RESOLUTION;
        let cell = ((max[0] - min[0]).max(max[1] - min[1]) / n as f64).max(1e-300);
        let mut buffer = Self {
            min,
            cell,
            depth: vec![f32::INFINITY; n * n],
        };
        for face in faces {
            for i in 1..face.len().saturating_sub(1) {
                buffer.rasterize([face[0], face[i], face[i + 1]]);
            }
        }
        buffer
    }

    fn pixel(&self, p: [f64; 2]) -> [f64; 2] {
        [
            (p[0] - self.min[0]) / self.cell,
            (p[1] - self.min[1]) / self.cell,
        ]
    }

    // A triangle, with its depth interpolated linearly on the page.
    fn rasterize(&mut self, corners: [([f64; 2], f64); 3]) {
        let n = DEPTH_RESOLUTION;
        let q = corners.map(|(p, _)| self.pixel(p));
        let area =
            (q[1][0] - q[0][0]) * (q[2][1] - q[0][1]) - (q[2][0] - q[0][0]) * (q[1][1] - q[0][1]);
        if area.abs() < 1e-12 {
            return;
        }
        let range = |k: usize| {
            let lo = q
                .iter()
                .map(|p| p[k])
                .fold(f64::INFINITY, f64::min)
                .floor()
                .max(0.) as usize;
            let hi = (q
                .iter()
                .map(|p| p[k])
                .fold(f64::NEG_INFINITY, f64::max)
                .ceil() as usize)
                .min(n - 1);
            lo..=hi
        };
        for y in range(1) {
            for x in range(0) {
                let c = [x as f64 + 0.5, y as f64 + 0.5];
                let w = [1, 2, 0].map(|i| {
                    let (a, b) = (q[i], q[(i + 1) % 3]);
                    ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / area
                });
                if w.iter().any(|&w| w < 0.) {
                    continue;
                }
                // w[k] weighs the corner opposite to the side it was measured on.
                let d = (w[0] * corners[0].1 + w[1] * corners[1].1 + w[2] * corners[2].1) as f32;
                let slot = &mut self.depth[y * n + x];
                *slot = slot.min(d);
            }
        }
    }

    // Whether a point (projected, with its depth) is not behind the surface.
    fn shows(&self, p: [f64; 2], depth: f64, tolerance: f64) -> bool {
        let n = DEPTH_RESOLUTION;
        let [x, y] = self.pixel(p);
        if x < 0. || y < 0. || x >= n as f64 || y >= n as f64 {
            return true;
        }
        depth <= f64::from(self.depth[y as usize * n + x as usize]) + tolerance
    }
}

// The faces of a surface that face the camera (or all, if two-sided), back to front, then the marks and lines on it,
// back to front, where the surface does not hide them (seen through a depth buffer of the faces).
fn surface(
    faces: Vec<Face>,
    camera: &Camera,
    look: Look,
    marks: &[SurfaceMark],
    lines: &[SurfaceLine],
) -> Vec<Item> {
    let mut faces = faces
        .into_iter()
        .filter_map(|face| {
            let Face {
                corners,
                normal,
                color,
            } = face;
            let centroid = corners.iter().sum::<Vector3D>() / corners.len().max(1) as f64;
            let normal = if camera.faces(centroid, normal) {
                normal
            } else if look.two_sided {
                -normal
            } else {
                return None;
            };
            Some((camera.depth(centroid), color, corners, normal))
        })
        .collect::<Vec<_>>();
    faces.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut items = vec![];
    for (_, color, corners, normal) in &faces {
        if let Some(base) = color {
            items.push(Item::Polygon {
                points: corners.iter().map(|&p| camera.project(p)).collect(),
                fill: mix(
                    scaled(*base, mix_scalar(camera.shade(*normal), 1., look.flatten)),
                    [1.; 3],
                    look.lighten,
                ),
            });
        }
    }
    if marks.is_empty() && lines.is_empty() {
        return items;
    }

    let projected = faces
        .iter()
        .map(|(_, _, corners, _)| {
            corners
                .iter()
                .map(|&p| (camera.project(p), camera.depth(p)))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let buffer = DepthBuffer::new(&projected);
    let points = faces
        .iter()
        .flat_map(|(_, _, corners, _)| corners.iter().copied())
        .collect::<Vec<_>>();
    let (min, max) = bounds(&points);
    let tolerance = DEPTH_TOLERANCE * (max - min).norm();
    let shows = |p: Vector3D| buffer.shows(camera.project(p), camera.depth(p), tolerance);
    let project = |points: &[Vector3D]| {
        points
            .iter()
            .map(|&p| camera.project(p))
            .collect::<Vec<_>>()
    };
    // The marks that show, with their depths (drawn back to front).
    let mut shown = vec![];
    for mark in marks {
        let SurfaceMark {
            points,
            offsets,
            closed,
            color,
        } = mark;
        let n = points.len();
        let segments = if *closed { n } else { n.saturating_sub(1) };
        let quad = |i: usize| {
            let j = (i + 1) % n;
            [
                points[i],
                points[j],
                points[j] + offsets[j],
                points[i] + offsets[i],
            ]
        };
        let visible = (0..segments)
            .map(|i| shows(quad(i).iter().sum::<Vector3D>() / 4.))
            .collect::<Vec<_>>();
        // The stretches of consecutive segments that show; around a loop, starting after one that does not
        // (if any).
        let start = if *closed {
            (0..segments).find(|&i| !visible[i]).map_or(0, |i| i + 1)
        } else {
            0
        };
        let mut runs: Vec<Vec<usize>> = vec![];
        let mut run = vec![];
        for k in 0..segments {
            let i = (start + k) % n;
            if visible[i] {
                run.push(i);
            } else if !run.is_empty() {
                runs.push(std::mem::take(&mut run));
            }
        }
        if !run.is_empty() {
            runs.push(run);
        }
        for run in runs {
            // Along the loop, then back along the other side.
            let mut outline = run.iter().map(|&i| points[i]).collect::<Vec<_>>();
            let last = (run[run.len() - 1] + 1) % n;
            outline.push(points[last]);
            outline.push(points[last] + offsets[last]);
            outline.extend(run.iter().rev().map(|&i| points[i] + offsets[i]));
            let depth =
                run.iter().map(|&i| camera.depth(points[i])).sum::<f64>() / run.len() as f64;
            let item = Item::Polygon {
                points: project(&outline),
                fill: *color,
            };
            shown.push((depth, item));
        }
    }
    for line in lines {
        for pair in line.points.windows(2) {
            if shows((pair[0] + pair[1]) / 2.) {
                let item = Item::Line {
                    from: camera.project(pair[0]),
                    to: camera.project(pair[1]),
                    color: line.color,
                    width: line.width,
                };
                shown.push((camera.depth((pair[0] + pair[1]) / 2.), item));
            }
        }
    }
    shown.sort_by(|a, b| b.0.total_cmp(&a.0));
    items.extend(shown.into_iter().map(|(_, item)| item));
    items
}

// The perspective camera of all figures on the model: it sees the whole input mesh.
fn model_camera(solution: &Solution, params: &FigureParams) -> Camera {
    let mesh = &*solution.mesh_ref;
    let points = mesh
        .vert_ids()
        .into_iter()
        .map(|v| mesh.position(v))
        .collect::<Vec<_>>();
    Camera::new(params, Some(&points))
}

fn input_mesh(model: &Model<'_>) -> Vec<Item> {
    let faces = mesh_faces(&model.solution.mesh_ref, |_| Some(model.style.surface));
    surface(faces, &model.camera, Look::default(), &[], &[])
}

fn axis(direction: Direction) -> usize {
    match direction {
        Direction::X => 0,
        Direction::Y => 1,
        Direction::Z => 2,
    }
}

// The loops on the input mesh, as bands (see `Solution::loop_frames`): between consecutive crossings, a strip on either
// side of the loop, its positive side in the loop's color and its negative side lighter (as in the GUI).
fn loop_structure(model: &Model<'_>) -> Option<Vec<Item>> {
    loop_bands(model, model.solution)
}

// The lifted patch boundaries of the drawing, as the loop structure.
fn lifted_loops(model: &Model<'_>) -> Option<Vec<Item>> {
    let drawing = model.skeleton.drawing.as_ref()?;
    let mut solution = Solution::new(model.solution.mesh_ref.clone());
    solution.loops = drawing.lifted.clone();
    loop_bands(model, &solution)
}

fn loop_bands(model: &Model<'_>, solution: &Solution) -> Option<Vec<Item>> {
    let (params, style) = (model.params, model.style);
    if solution.loops.is_empty() {
        return None;
    }
    let mesh = &*solution.mesh_ref;
    let points = mesh
        .vert_ids()
        .into_iter()
        .map(|v| mesh.position(v))
        .collect::<Vec<_>>();
    let (min, max) = bounds(&points);
    let half = 0.5 * params.loop_width * (max - min).norm();
    let mut lines = vec![];
    for (loop_id, lewp) in &solution.loops {
        let positive = style.dual[axis(lewp.direction)];
        let negative = positive.map(|c| c + (1. - c) * style.light_mix);
        // Resampled at about the band's width, so that it bends smoothly (see `Solution::loop_band`).
        let frames = solution.loop_band(loop_id, half);
        let points = frames.iter().map(|f| f.position).collect::<Vec<_>>();
        for (side, color) in [(-1., positive), (1., negative)] {
            lines.push(SurfaceMark {
                points: points.clone(),
                offsets: frames.iter().map(|f| f.negative * (side * half)).collect(),
                closed: true,
                color,
            });
        }
    }
    let faces = mesh_faces(mesh, |_| Some(style.dark));
    Some(surface(faces, &model.camera, Look::default(), &lines, &[]))
}

// The input mesh (refined by the layout) colored by patch, with the paths between the patches.
fn segmentation(model: &Model<'_>) -> Option<Vec<Item>> {
    let (solution, params, style) = (model.solution, model.params, model.style);
    let (layout, polycube) = (solution.layout.as_ref()?, solution.polycube.as_ref()?);
    let mesh = &layout.granulated_mesh;
    let structure = &polycube.structure;
    let mut colors = HashMap::new();
    for (&patch_id, patch) in &layout.face_to_patch {
        let (direction, _) = to_principal_direction(structure.normal(patch_id));
        for &face in &patch.faces {
            colors.insert(face, style.primal[axis(direction)]);
        }
    }
    // The paths as bands (centered on them), resampled at about their width (see `band_frames`).
    let points = mesh
        .vert_ids()
        .into_iter()
        .map(|v| mesh.position(v))
        .collect::<Vec<_>>();
    let (min, max) = bounds(&points);
    let half = 0.5 * params.path_width * (max - min).norm();
    let mut lines = vec![];
    for band in solution.path_bands(half) {
        lines.push(SurfaceMark {
            points: band
                .frames
                .iter()
                .map(|f| f.position - f.negative * half)
                .collect(),
            offsets: band
                .frames
                .iter()
                .map(|f| f.negative * (2. * half))
                .collect(),
            closed: false,
            color: style.line,
        });
        // Round ends, where the paths meet (at the corners).
        for end in [band.frames.first(), band.frames.last()]
            .into_iter()
            .flatten()
        {
            let points = disk(end.position, end.normal, half, 16);
            lines.push(SurfaceMark {
                offsets: points.iter().map(|p| end.position - p).collect(),
                points,
                closed: true,
                color: style.line,
            });
        }
    }
    let faces = mesh_faces(mesh, |face| {
        Some(colors.get(&face).copied().unwrap_or(style.surface))
    });
    Some(surface(faces, &model.camera, Look::default(), &lines, &[]))
}

// The polycube in a parallel projection: its (axis-aligned, rectangular) faces cut on the lattice of all their
// coordinates, drawn back to front by their position in the lattice; every cell with the sides of its face that it
// has.
fn polycube(solution: &Solution, params: &FigureParams, style: &FigureStyle) -> Option<Vec<Item>> {
    let structure = &solution.polycube.as_ref()?.structure;
    let camera = Camera::new(params, None);

    let mut faces = vec![];
    for face in structure.face_ids() {
        let corners = structure
            .vertices(face)
            .map(|v| structure.position(v))
            .collect::<Vec<_>>();
        let normal = structure.normal(face);
        let centroid = corners.iter().sum::<Vector3D>() / corners.len().max(1) as f64;
        if corners.is_empty() || !camera.faces(centroid, normal) {
            continue;
        }
        let n = (0..3).max_by(|&a, &b| normal[a].abs().total_cmp(&normal[b].abs()))?;
        let (ia, ib) = ((n + 1) % 3, (n + 2) % 3);
        let lo = |k: usize| corners.iter().map(|p| p[k]).fold(f64::INFINITY, f64::min);
        let hi = |k: usize| {
            corners
                .iter()
                .map(|p| p[k])
                .fold(f64::NEG_INFINITY, f64::max)
        };
        // Flat colors (as in the GUI): the axes are told apart by their colors.
        let (direction, _) = to_principal_direction(normal);
        faces.push(AxisFace {
            n,
            c: centroid[n],
            rect: [lo(ia), hi(ia), lo(ib), hi(ib)],
            fill: style.primal[axis(direction)],
        });
    }
    boxes(&faces, &[], &camera, style)
}

// An axis-aligned rectangle facing the camera: normal axis, plane coordinate, and extent on the two other axes (in
// their cyclic order).
struct AxisFace {
    n: usize,
    c: f64,
    rect: [f64; 4],
    fill: Rgb,
}

// An axis-aligned line segment: its axis, its ends, and its color and width (as for `Item::Line`).
struct AxisSegment {
    axis: usize,
    from: Vector3D,
    to: Vector3D,
    color: Rgb,
    width: f64,
}

// Axis-aligned faces and segments in a parallel projection: cut on the lattice of all face coordinates, drawn back to
// front by their position in the lattice; every face cell with the sides of its face that it has. A segment piece
// is drawn over a face cell at the same depth.
fn boxes(
    faces: &[AxisFace],
    segments: &[AxisSegment],
    camera: &Camera,
    style: &FigureStyle,
) -> Option<Vec<Item>> {
    if faces.is_empty() {
        return None;
    }
    let view = [camera.forward.x, camera.forward.y, camera.forward.z];
    let extent = faces
        .iter()
        .map(|f| (f.rect[1] - f.rect[0]).max(f.rect[3] - f.rect[2]))
        .fold(0., f64::max);

    // The lattice: the distinct coordinates per axis.
    let eps = 1e-9 * extent.max(1e-12);
    let levels = (0..3)
        .map(|k| {
            let mut values = vec![];
            for f in faces {
                let (ia, ib) = ((f.n + 1) % 3, (f.n + 2) % 3);
                if f.n == k {
                    values.push(f.c);
                }
                if ia == k {
                    values.extend([f.rect[0], f.rect[1]]);
                }
                if ib == k {
                    values.extend([f.rect[2], f.rect[3]]);
                }
            }
            values.sort_by(f64::total_cmp);
            values.dedup_by(|a, b| (*a - *b).abs() <= eps);
            values
        })
        .collect::<Vec<_>>();
    let index = |k: usize, x: f64| {
        levels[k]
            .partition_point(|&l| l <= x + eps)
            .saturating_sub(1)
    };
    let sign = view.map(|v| if v.abs() < 1e-9 { 0. } else { v.signum() });

    let key = |d: [f64; 3]| (0..3).map(|k| sign[k] * d[k]).sum::<f64>();

    enum Piece<'a> {
        Face {
            face: &'a AxisFace,
            a: [f64; 2],
            b: [f64; 2],
        },
        Segment {
            segment: &'a AxisSegment,
            from: Vector3D,
            to: Vector3D,
        },
    }
    struct Cell<'a> {
        piece: Piece<'a>,
        key: f64,
    }
    let mut cells = vec![];
    for f in faces {
        let (ia, ib) = ((f.n + 1) % 3, (f.n + 2) % 3);
        let (i0, i1) = (index(ia, f.rect[0]), index(ia, f.rect[1]));
        let (j0, j1) = (index(ib, f.rect[2]), index(ib, f.rect[3]));
        let kc = index(f.n, f.c);
        for i in i0..i1 {
            for j in j0..j1 {
                // The cell's doubled lattice coordinates, along the view.
                let mut d = [0.; 3];
                d[f.n] = 2. * kc as f64;
                d[ia] = 2. * i as f64 + 1.;
                d[ib] = 2. * j as f64 + 1.;
                cells.push(Cell {
                    piece: Piece::Face {
                        face: f,
                        a: [levels[ia][i], levels[ia][i + 1]],
                        b: [levels[ib][j], levels[ib][j + 1]],
                    },
                    key: key(d),
                });
            }
        }
    }
    // The doubled lattice coordinate of a value: even on a level, odd between two.
    let doubled = |k: usize, x: f64| {
        let i = index(k, x);
        let on = levels[k].get(i).is_some_and(|&l| (l - x).abs() <= eps);
        2. * i as f64 + if on { 0. } else { 1. }
    };
    for segment in segments {
        let k = segment.axis;
        let (lo, hi) = {
            let (a, b) = (segment.from[k], segment.to[k]);
            (a.min(b), a.max(b))
        };
        // Cut where the segment crosses a level.
        let mut cuts = vec![lo];
        cuts.extend(
            levels[k]
                .iter()
                .copied()
                .filter(|&l| l > lo + eps && l < hi - eps),
        );
        cuts.push(hi);
        for pair in cuts.windows(2) {
            let (mut from, mut to) = (segment.from, segment.from);
            from[k] = pair[0];
            to[k] = pair[1];
            let middle = (from + to) / 2.;
            let d = [0, 1, 2].map(|j| doubled(j, middle[j]));
            cells.push(Cell {
                piece: Piece::Segment { segment, from, to },
                key: key(d),
            });
        }
    }
    // Farthest first (stable: segments after the faces at the same depth).
    cells.sort_by(|x, y| y.key.total_cmp(&x.key));

    let mut items = vec![];
    for cell in &cells {
        let (f, [a0, a1], [b0, b1]) = match cell.piece {
            Piece::Face { face, a, b } => (face, a, b),
            Piece::Segment { segment, from, to } => {
                items.push(Item::Line {
                    from: camera.project(from),
                    to: camera.project(to),
                    color: segment.color,
                    width: segment.width,
                });
                continue;
            }
        };
        let (ia, ib) = ((f.n + 1) % 3, (f.n + 2) % 3);
        let point = |a: f64, b: f64| {
            let mut p = Vector3D::zeros();
            p[f.n] = f.c;
            p[ia] = a;
            p[ib] = b;
            camera.project(p)
        };
        items.push(Item::Polygon {
            points: vec![point(a0, b0), point(a1, b0), point(a1, b1), point(a0, b1)],
            fill: f.fill,
        });
        // The sides of the cell on the boundary of its face.
        let on = |x: f64, y: f64| (x - y).abs() <= eps;
        let sides = [
            (on(a0, f.rect[0]), (a0, b0, a0, b1)),
            (on(a1, f.rect[1]), (a1, b0, a1, b1)),
            (on(b0, f.rect[2]), (a0, b0, a1, b0)),
            (on(b1, f.rect[3]), (a0, b1, a1, b1)),
        ];
        for (_, (pa, pb, qa, qb)) in sides.into_iter().filter(|(side, _)| *side) {
            items.push(Item::Line {
                from: point(pa, pb),
                to: point(qa, qb),
                color: style.line,
                width: POLYCUBE_EDGE_WIDTH,
            });
        }
    }
    Some(items)
}

// ---------------------------------------------------------------------------------------------------------------------
// Output.

fn hex(color: Rgb) -> String {
    let [r, g, b] = color.map(|c| (c.clamp(0., 1.) * 255.).round() as u8);
    format!("#{r:02x}{g:02x}{b:02x}")
}

fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

// The items, and the frame (projected points), fitted (centered) in a square of the given size, with the label below
// it.
fn to_svg(items: Vec<Item>, frame: &[[f64; 2]], size: f64, label: Option<&str>) -> String {
    let (mut min, mut max) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    let mut extend = |p: &[f64; 2]| {
        for k in 0..2 {
            min[k] = min[k].min(p[k]);
            max[k] = max[k].max(p[k]);
        }
    };
    for item in &items {
        match item {
            Item::Polygon { points, .. } => points.iter().for_each(&mut extend),
            Item::Line { from, to, .. } => {
                extend(from);
                extend(to);
            }
            Item::Disk { center, .. } => extend(center),
        }
    }
    frame.iter().for_each(&mut extend);
    let span = (max[0] - min[0]).max(max[1] - min[1]).max(1e-12);
    let scale = size * (1. - 2. * MARGIN) / span;
    let center = [(min[0] + max[0]) / 2., (min[1] + max[1]) / 2.];
    // On the page: y down.
    let page = |p: &[f64; 2]| {
        (
            size / 2. + (p[0] - center[0]) * scale,
            size / 2. - (p[1] - center[1]) * scale,
        )
    };

    let font = FONT_SIZE * size;
    let height = if label.is_some() {
        size + 2. * font
    } else {
        size
    };
    let mut svg = String::new();
    let _ = writeln!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{size:.1}" height="{height:.1}" viewBox="0 0 {size:.1} {height:.1}">"#
    );
    for item in &items {
        match item {
            Item::Polygon { points, fill } => {
                let color = hex(*fill);
                let _ = write!(svg, r#"<path d=""#);
                for (i, p) in points.iter().enumerate() {
                    let (x, y) = page(p);
                    let _ = write!(svg, "{}{x:.2} {y:.2}", if i == 0 { 'M' } else { 'L' });
                }
                let _ = writeln!(
                    svg,
                    r#"Z" fill="{color}" stroke="{color}" stroke-width="{SEAM}" stroke-linejoin="round"/>"#
                );
            }
            Item::Line {
                from,
                to,
                color,
                width,
            } => {
                let ((x0, y0), (x1, y1)) = (page(from), page(to));
                let _ = writeln!(
                    svg,
                    r#"<path d="M{x0:.2} {y0:.2}L{x1:.2} {y1:.2}" fill="none" stroke="{}" stroke-width="{:.2}" stroke-linecap="round"/>"#,
                    hex(*color),
                    width * size
                );
            }
            Item::Disk {
                center,
                radius,
                fill,
                ring,
            } => {
                let (x, y) = page(center);
                let _ = writeln!(
                    svg,
                    r#"<circle cx="{x:.2}" cy="{y:.2}" r="{:.2}" fill="{}" stroke="{}" stroke-width="{:.2}"/>"#,
                    radius * size,
                    hex(*fill),
                    hex(*ring),
                    0.25 * radius * size
                );
            }
        }
    }
    if let Some(label) = label {
        let _ = writeln!(
            svg,
            r#"<text x="{:.2}" y="{:.2}" font-family="Helvetica, Arial, sans-serif" font-size="{font:.2}" fill="{}" text-anchor="middle">{}</text>"#,
            size / 2.,
            size + 1.4 * font,
            hex([0.2, 0.2, 0.2]),
            escape(label)
        );
    }
    svg.push_str("</svg>\n");
    svg
}

mod skeleton;

#[cfg(test)]
mod tests {
    use super::today;

    #[test]
    fn today_is_a_date() {
        let date = today();
        assert_eq!(date.len(), 10);
        assert!(date.starts_with("20"));
    }
}
