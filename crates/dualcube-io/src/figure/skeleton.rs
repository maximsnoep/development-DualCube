//! The figures of the skeleton stages and of the quad mesh (see the view presets of the GUI).

use super::{
    AxisFace, AxisSegment, Camera, CubeColoring, Face, Item, Look, Model, Rgb, SurfaceLine, axis,
    boxes, mesh_faces, region_colors, surface,
};
use crate::SkeletonExtraction;
use dualcube::prelude::*;
use dualcube::skeleton::geometry::SurfaceGeometry;
use dualcube::skeleton::skeletonize::Extraction;
use dualcube::skeleton::{
    CuboidOwner, EdgeLabel, RealizedSkeleton, Skeleton, SkeletonEdgeID, SkeletonNodeID,
};
use std::collections::{HashMap, HashSet};

// Sizes as fractions of the figure's size.
const NODE_RADIUS: f64 = 0.011;
const EDGE_WIDTH: f64 = 0.009;
// Edges outside the labeling's spanning tree (thicker, as in the GUI).
const NON_TREE_WIDTH: f64 = 0.02;
const BOUNDARY_WIDTH: f64 = 0.004;
// The loops of a cut system (few, so wider).
const CUT_WIDTH: f64 = 0.012;
// How far the shading of the plain input mesh under loops is flattened, so that its far side is not dark.
const FLAT_SURFACE: f32 = 0.7;
const QUAD_EDGE_WIDTH: f64 = 0.0012;
const QUAD_PATH_WIDTH: f64 = 0.004;

// Unlabeled edges, and the nodes of a labeled graph.
const GRAY: Rgb = [0.45, 0.45, 0.45];
const QUAD_EDGE: Rgb = [0.35, 0.35, 0.35];
// How far the input mesh under a graph, and the patches of the combined figure, are lightened.
const GHOST: f32 = 0.3;
const LIGHT_PATCHES: f32 = 0.55;

/// Which skeleton a graph figure shows.
pub(super) enum Graph {
    Raw,
    Current,
    /// The current skeleton with gray nodes.
    Gray,
    Tree,
    Labeled,
    /// The labeled optimized skeleton.
    Optimized,
    /// The optimized skeleton with gray nodes.
    OptimizedGray,
}

fn extraction<'a>(model: &Model<'a>) -> Option<&'a Extraction> {
    match &model.skeleton.extraction {
        Some(SkeletonExtraction::Extracted(extraction)) => Some(extraction),
        _ => None,
    }
}

// The skeleton as labeled (with its junction splits and repairs), else as extracted (as in the GUI).
fn current<'a>(model: &Model<'a>) -> Option<&'a Skeleton> {
    let save = model.skeleton;
    save.labeling
        .as_ref()
        .map(|attempt| &attempt.skeleton)
        .or_else(|| save.labeled.as_ref().map(|labeled| &labeled.skeleton))
        .or_else(|| match &save.extraction {
            Some(SkeletonExtraction::Extracted(extraction)) => Some(&extraction.skeleton),
            _ => None,
        })
}

// The node colors of all stages, the optimized skeleton first (as in the GUI).
fn palette(model: &Model<'_>) -> HashMap<SkeletonNodeID, Rgb> {
    let optimized = model
        .skeleton
        .optimized
        .as_ref()
        .map(|o| &o.labeled.skeleton);
    let raw = extraction(model).map(|e| &e.raw);
    let stages: Vec<_> = [optimized, current(model), raw]
        .into_iter()
        .flatten()
        .collect();
    region_colors(&stages)
}

pub(super) fn deflated(model: &Model<'_>) -> Option<Vec<Item>> {
    let mesh = match model.skeleton.extraction.as_ref()? {
        SkeletonExtraction::Deflated(mesh) => mesh,
        SkeletonExtraction::Extracted(extraction) => &extraction.contracted_mesh,
    };
    // Inside the input mesh, seen through it; contraction folds the surface, so its faces are seen from both sides.
    let mut items = ghost(model);
    let look = Look {
        two_sided: true,
        ..Look::default()
    };
    let faces = mesh_faces(mesh, |_| Some(GRAY));
    items.extend(surface(faces, &model.camera, look, &[], &[]));
    Some(items)
}

// The input mesh, light, to draw what lies inside it over.
fn ghost(model: &Model<'_>) -> Vec<Item> {
    let faces = mesh_faces(&model.solution.mesh_ref, |_| Some(model.style.surface));
    let look = Look {
        lighten: GHOST,
        ..Look::default()
    };
    surface(faces, &model.camera, look, &[], &[])
}

pub(super) fn graph(model: &Model<'_>, graph: Graph) -> Option<Vec<Item>> {
    let edges = match graph {
        Graph::Raw => plain_graph(model, &extraction(model)?.raw),
        Graph::Current => plain_graph(model, current(model)?),
        Graph::Gray => gray_graph(model, current(model)?),
        Graph::OptimizedGray => {
            gray_graph(model, &model.skeleton.optimized.as_ref()?.labeled.skeleton)
        }
        Graph::Tree => tree_graph(model)?,
        Graph::Labeled => labeled_graph(model)?,
        Graph::Optimized => optimized_graph(model)?,
    };
    let mut items = ghost(model);
    items.extend(edges);
    Some(items)
}

pub(super) fn patches(model: &Model<'_>, optimized: bool) -> Option<Vec<Item>> {
    let current = current(model)?;
    let skeleton = if optimized {
        &model.skeleton.optimized.as_ref()?.labeled.skeleton
    } else {
        current
    };
    let (faces, lines) = patch_surface(model, skeleton, &palette(model), |_| true)?;
    Some(surface(faces, &model.camera, Look::default(), &[], &lines))
}

// The patch boundaries of the (optimized) skeleton on the input mesh in its plain color.
pub(super) fn patch_boundaries(model: &Model<'_>, optimized: bool) -> Option<Vec<Item>> {
    let skeleton = if optimized {
        &model.skeleton.optimized.as_ref()?.labeled.skeleton
    } else {
        current(model)?
    };
    plain_boundaries(model, skeleton, |_| true, BOUNDARY_WIDTH)
}

// A minimal cut system of the optimized skeleton, spread out (see `dualcube::skeleton::cut_system`).
pub(super) fn cut_system(model: &Model<'_>) -> Option<Vec<Item>> {
    let skeleton = &model.skeleton.optimized.as_ref()?.labeled.skeleton;
    let cuts = dualcube::skeleton::cut_system::run(&model.solution.mesh_ref, skeleton);
    plain_boundaries(model, skeleton, |edge| cuts.contains(&edge), CUT_WIDTH)
}

// The boundaries it keeps of the skeleton's patches (this wide), on the input mesh in its plain color.
fn plain_boundaries(
    model: &Model<'_>,
    skeleton: &Skeleton,
    keep: impl Fn(SkeletonEdgeID) -> bool,
    width: f64,
) -> Option<Vec<Item>> {
    let (mut faces, mut lines) = patch_surface(model, skeleton, &HashMap::new(), keep)?;
    for face in &mut faces {
        face.color = Some(model.style.surface);
    }
    for line in &mut lines {
        line.width = width;
    }
    let look = Look {
        flatten: FLAT_SURFACE,
        ..Look::default()
    };
    Some(surface(faces, &model.camera, look, &[], &lines))
}

// The (optimized) patches, light, with the (optimized) skeleton graph; labeled when it is.
pub(super) fn combined(model: &Model<'_>, optimized: bool) -> Option<Vec<Item>> {
    let current = current(model)?;
    let optimized = match optimized {
        true => Some(&model.skeleton.optimized.as_ref()?.labeled),
        false => None,
    };
    let skeleton = optimized.map_or(current, |labeled| &labeled.skeleton);
    let (faces, _) = patch_surface(model, skeleton, &palette(model), |_| true)?;
    let look = Look {
        lighten: LIGHT_PATCHES,
        ..Look::default()
    };
    let mut items = surface(faces, &model.camera, look, &[], &[]);
    let graph = match optimized {
        Some(_) => optimized_graph(model),
        None => labeled_graph(model),
    };
    items.extend(graph.unwrap_or_else(|| plain_graph(model, current)));
    Some(items)
}

// The (optimized) cube placement.
fn placement<'a>(model: &Model<'a>) -> Option<&'a RealizedSkeleton> {
    let save = model.skeleton;
    save.optimized.as_ref().or(save.realized.as_ref())
}

// The corners of all cuboids of the placement, projected: the frame of its figures, so that they line up.
pub(super) fn placement_frame(model: &Model<'_>) -> Vec<[f64; 2]> {
    let Some(realized) = placement(model) else {
        return vec![];
    };
    let camera = Camera::new(model.params, None);
    let mut frame = vec![];
    for cube in realized.cuboids().cuboids.values() {
        let (lo, hi) = (cube.center - cube.size / 2., cube.center + cube.size / 2.);
        for corner in 0..8 {
            let p = Vector3D::from_fn(|k, _| if corner >> k & 1 == 0 { lo[k] } else { hi[k] });
            frame.push(camera.project(p));
        }
    }
    frame
}

// The labeled skeleton with its nodes at their placed positions (the centers of their cubes).
pub(super) fn placed_skeleton(model: &Model<'_>) -> Option<Vec<Item>> {
    let realized = placement(model)?;
    let labeled = &realized.labeled;
    let mut placed = labeled.skeleton.clone();
    for (id, node) in &mut placed.nodes {
        node.position = realized.node_positions[id];
    }
    let camera = Camera::new(model.params, None);
    Some(nodes_and_edges(
        model,
        &camera,
        &placed,
        |id| {
            let color = labeled
                .labels
                .get(id)
                .map_or(GRAY, |label| model.style.primal[axis(label.direction)]);
            (color, EDGE_WIDTH)
        },
        |_| GRAY,
        Layering::NodesOnTop,
    ))
}

// The faces of an axis-aligned box that face the camera, but those `hidden` (axis, positive side) leaves out.
fn box_faces(
    faces: &mut Vec<AxisFace>,
    camera: &Camera,
    (lo, hi): (Vector3D, Vector3D),
    fill: Rgb,
    hidden: impl Fn(usize, bool) -> bool,
) {
    for n in 0..3 {
        for positive in [false, true] {
            let toward = if positive { 1. } else { -1. };
            if hidden(n, positive) || toward * camera.forward[n] >= 0. {
                continue;
            }
            let (ia, ib) = ((n + 1) % 3, (n + 2) % 3);
            faces.push(AxisFace {
                n,
                c: if positive { hi[n] } else { lo[n] },
                rect: [lo[ia], hi[ia], lo[ib], hi[ib]],
                fill,
            });
        }
    }
}

// The cuboids of the (optimized) placement as boxes, with the faces where they connect hidden as in the GUI; the
// bevels of the edge cuboids are left out. Cuboids the coloring leaves out are gray.
pub(super) fn cubes(model: &Model<'_>, coloring: CubeColoring) -> Option<Vec<Item>> {
    let realized = placement(model)?;
    let layout = realized.cuboids();
    let palette = palette(model);
    let hidden = layout.hidden_faces();
    let camera = Camera::new(model.params, None);
    let mut faces = vec![];
    for (id, cube) in &layout.cuboids {
        let fill = match cube.owner {
            CuboidOwner::Node(_) if !coloring.colors_nodes() => GRAY,
            CuboidOwner::Edge(_) if !coloring.colors_edges() => GRAY,
            CuboidOwner::Node(node) => palette.get(&node).copied().unwrap_or(GRAY),
            CuboidOwner::Edge(edge) => realized
                .labeled
                .labels
                .get(edge)
                .map_or(GRAY, |label| model.style.primal[axis(label.direction)]),
        };
        let bounds = (cube.center - cube.size / 2., cube.center + cube.size / 2.);
        box_faces(&mut faces, &camera, bounds, fill, |n, positive| {
            hidden.contains(&(id, n, positive))
        });
    }
    boxes(&faces, &[], &camera, model.style)
}

// The node cubes of the placement (in their colors), with the labeled edges between them as in the placed skeleton.
pub(super) fn node_cubes(model: &Model<'_>) -> Option<Vec<Item>> {
    let realized = placement(model)?;
    let palette = palette(model);
    let camera = Camera::new(model.params, None);
    let (mut faces, mut segments) = (vec![], vec![]);
    for cube in realized.cuboids().cuboids.values() {
        let (lo, hi) = (cube.center - cube.size / 2., cube.center + cube.size / 2.);
        match cube.owner {
            CuboidOwner::Node(node) => {
                let fill = palette.get(&node).copied().unwrap_or(GRAY);
                box_faces(&mut faces, &camera, (lo, hi), fill, |_, _| false);
            }
            CuboidOwner::Edge(edge) => {
                let Some(label) = realized.labeled.labels.get(edge) else {
                    continue;
                };
                // Along the axis of the edge, its cuboid spans between the node cubes.
                let along = axis(label.direction);
                let (mut from, mut to) = (cube.center, cube.center);
                from[along] = lo[along];
                to[along] = hi[along];
                segments.push(AxisSegment {
                    axis: along,
                    from,
                    to,
                    color: model.style.primal[along],
                    width: EDGE_WIDTH,
                });
            }
        }
    }
    boxes(&faces, &segments, &camera, model.style)
}

// The quad mesh colored by the axes of its polycube faces, with its edges, and the edge loops from its irregular
// vertices (as in the GUI).
pub(super) fn quad(model: &Model<'_>) -> Option<Vec<Item>> {
    let (quad, style) = (model.solution.quad.as_ref()?, model.style);
    let mesh = &quad.quad_mesh;
    let direction = |face| to_principal_direction(quad.quad_mesh_polycube.normal(face)).0;
    let faces = mesh_faces(mesh, |face| Some(style.primal[axis(direction(face))]));

    let irregular = mesh
        .vert_ids()
        .into_iter()
        .filter(|&vert| {
            mesh.faces(vert)
                .map(direction)
                .collect::<HashSet<_>>()
                .len()
                >= 3
        })
        .collect::<Vec<_>>();
    let mut paths = HashSet::new();
    for vert in irregular {
        for edge in mesh.edges(vert) {
            let mut walker = mesh.next(mesh.twin(mesh.next(edge)));
            while paths.insert(walker) {
                walker = mesh.next(mesh.twin(mesh.next(walker)));
            }
        }
    }
    let mut seen = HashSet::new();
    let mut lines = vec![];
    for edge in mesh.edge_ids() {
        if !seen.insert(edge) || !seen.insert(mesh.twin(edge)) {
            continue;
        }
        let path = paths.contains(&edge) || paths.contains(&mesh.twin(edge));
        lines.push(SurfaceLine {
            points: vec![
                mesh.position(mesh.root(edge)),
                mesh.position(mesh.toor(edge)),
            ],
            color: if path { style.line } else { QUAD_EDGE },
            width: if path {
                QUAD_PATH_WIDTH
            } else {
                QUAD_EDGE_WIDTH
            },
        });
    }
    Some(surface(faces, &model.camera, Look::default(), &[], &lines))
}

// The input mesh cut into the skeleton's patches (colored by node), and the boundaries it keeps.
fn patch_surface(
    model: &Model<'_>,
    skeleton: &Skeleton,
    colors: &HashMap<SkeletonNodeID, Rgb>,
    keep: impl Fn(SkeletonEdgeID) -> bool,
) -> Option<(Vec<Face>, Vec<SurfaceLine>)> {
    let geometry = SurfaceGeometry::build(&model.solution.mesh_ref, skeleton).ok()?;
    let faces = geometry
        .triangles
        .iter()
        .map(|triangle| {
            let [a, b, c] = triangle.points;
            Face {
                corners: triangle.points.to_vec(),
                normal: (b - a)
                    .cross(&(c - a))
                    .try_normalize(1e-300)
                    .unwrap_or_else(Vector3D::zeros),
                color: Some(colors.get(&triangle.node).copied().unwrap_or(GRAY)),
            }
        })
        .collect();
    let lines = geometry
        .boundaries
        .into_iter()
        .filter(|boundary| keep(boundary.edge))
        .map(|boundary| SurfaceLine {
            points: boundary.points,
            color: model.style.line,
            width: BOUNDARY_WIDTH,
        })
        .collect();
    Some((faces, lines))
}

// A graph without labels: dark edges, nodes in their region colors.
fn plain_graph(model: &Model<'_>, skeleton: &Skeleton) -> Vec<Item> {
    let palette = palette(model);
    nodes_and_edges(
        model,
        &model.camera,
        skeleton,
        |_| (model.style.line, EDGE_WIDTH),
        |node| palette.get(&node).copied().unwrap_or(GRAY),
        Layering::EdgesOnTop,
    )
}

// The labeling (edges in their axis colors, non-tree edges thick), else the labeled skeleton.
fn labeled_graph(model: &Model<'_>) -> Option<Vec<Item>> {
    let save = model.skeleton;
    let edge = |label, non_tree| labeled_edge(model, label, non_tree);
    if let Some(attempt) = &save.labeling {
        return Some(nodes_and_edges(
            model,
            &model.camera,
            &attempt.skeleton,
            |id| edge(attempt.labels.get(id), attempt.outside_tree(id)),
            |_| GRAY,
            Layering::EdgesOnTop,
        ));
    }
    let labeled = save.labeled.as_ref()?;
    Some(nodes_and_edges(
        model,
        &model.camera,
        &labeled.skeleton,
        |id| edge(labeled.labels.get(id), false),
        |_| GRAY,
        Layering::EdgesOnTop,
    ))
}

// The skeleton in the line color, with gray nodes.
fn gray_graph(model: &Model<'_>, skeleton: &Skeleton) -> Vec<Item> {
    nodes_and_edges(
        model,
        &model.camera,
        skeleton,
        |_| (model.style.line, EDGE_WIDTH),
        |_| GRAY,
        Layering::EdgesOnTop,
    )
}

// The optimized skeleton, edges in their axis colors.
fn optimized_graph(model: &Model<'_>) -> Option<Vec<Item>> {
    let labeled = &model.skeleton.optimized.as_ref()?.labeled;
    Some(nodes_and_edges(
        model,
        &model.camera,
        &labeled.skeleton,
        |id| labeled_edge(model, labeled.labels.get(id), false),
        |_| GRAY,
        Layering::EdgesOnTop,
    ))
}

// A labeled edge in its axis color (unlabeled gray), non-tree edges thick.
fn labeled_edge(model: &Model<'_>, label: Option<&EdgeLabel>, non_tree: bool) -> (Rgb, f64) {
    (
        label.map_or(GRAY, |label| model.style.primal[axis(label.direction)]),
        if non_tree { NON_TREE_WIDTH } else { EDGE_WIDTH },
    )
}

// The tree labeling before cycle repairs: the edges it leaves unlabeled gray and thick.
fn tree_graph(model: &Model<'_>) -> Option<Vec<Item>> {
    let attempt = model.skeleton.labeling.as_ref()?;
    let skeleton = attempt.tree_skeleton.as_ref()?;
    Some(nodes_and_edges(
        model,
        &model.camera,
        skeleton,
        |id| match attempt.tree_labels.get(id) {
            Some(label) => (model.style.primal[axis(label.direction)], EDGE_WIDTH),
            None => (GRAY, NON_TREE_WIDTH),
        },
        |_| GRAY,
        Layering::EdgesOnTop,
    ))
}

// Which of nodes and edges are drawn over the other.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Layering {
    EdgesOnTop,
    NodesOnTop,
}

// The nodes as dots and the edges as lines, one layer over the other; each back to front.
fn nodes_and_edges(
    model: &Model<'_>,
    camera: &Camera,
    skeleton: &Skeleton,
    edge: impl Fn(SkeletonEdgeID) -> (Rgb, f64),
    node: impl Fn(SkeletonNodeID) -> Rgb,
    layering: Layering,
) -> Vec<Item> {
    let position = |id: SkeletonNodeID| skeleton.nodes[id].position;
    let mut disks = vec![];
    for id in skeleton.nodes.keys() {
        let p = position(id);
        let disk = Item::Disk {
            center: camera.project(p),
            radius: NODE_RADIUS,
            fill: node(id),
            ring: model.style.line,
        };
        disks.push((camera.depth(p), disk));
    }
    let mut lines = vec![];
    for (id, e) in &skeleton.edges {
        let (a, b) = (position(e.nodes[0]), position(e.nodes[1]));
        let (color, width) = edge(id);
        let line = Item::Line {
            from: camera.project(a),
            to: camera.project(b),
            color,
            width,
        };
        lines.push((camera.depth((a + b) / 2.), line));
    }
    let groups = match layering {
        Layering::EdgesOnTop => [disks, lines],
        Layering::NodesOnTop => [lines, disks],
    };
    let mut items = vec![];
    for mut group in groups {
        // Farthest first.
        group.sort_by(|a, b| b.0.total_cmp(&a.0));
        items.extend(group.into_iter().map(|(_, item)| item));
    }
    items
}
