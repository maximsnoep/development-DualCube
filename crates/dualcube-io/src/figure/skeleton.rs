//! The figures of the skeleton stages and of the quad mesh (see the view presets of the GUI).

use super::{
    AxisFace, Camera, Face, Item, Look, Model, Rgb, SurfaceLine, axis, boxes, mesh_faces,
    region_colors, surface,
};
use crate::SkeletonExtraction;
use dualcube::prelude::*;
use dualcube::skeleton::geometry::SurfaceGeometry;
use dualcube::skeleton::skeletonize::Extraction;
use dualcube::skeleton::{CuboidOwner, EdgeLabel, Skeleton, SkeletonEdgeID, SkeletonNodeID};
use std::collections::{HashMap, HashSet};

// Sizes as fractions of the figure's size.
const NODE_RADIUS: f64 = 0.011;
const EDGE_WIDTH: f64 = 0.009;
// Edges outside the labeling's spanning tree (thicker, as in the GUI).
const NON_TREE_WIDTH: f64 = 0.02;
const BOUNDARY_WIDTH: f64 = 0.004;
const QUAD_EDGE_WIDTH: f64 = 0.0012;
const QUAD_PATH_WIDTH: f64 = 0.004;

// Unlabeled edges, and the nodes of a labeled graph.
const GRAY: Rgb = [0.45, 0.45, 0.45];
const QUAD_EDGE: Rgb = [0.35, 0.35, 0.35];
// How far the input mesh under a graph, and the patches of the combined figure, are lightened.
const GHOST: f32 = 0.6;
const LIGHT_PATCHES: f32 = 0.55;

/// Which skeleton a graph figure shows.
pub(super) enum Graph {
    Raw,
    Current,
    Tree,
    Labeled,
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
        lighten: 0.,
    };
    let faces = mesh_faces(mesh, |_| Some(GRAY));
    items.extend(surface(faces, &model.camera, look, &[], &[]));
    Some(items)
}

// The input mesh, light, to draw what lies inside it over.
fn ghost(model: &Model<'_>) -> Vec<Item> {
    let faces = mesh_faces(&model.solution.mesh_ref, |_| Some(model.style.surface));
    let look = Look {
        two_sided: false,
        lighten: GHOST,
    };
    surface(faces, &model.camera, look, &[], &[])
}

pub(super) fn graph(model: &Model<'_>, graph: Graph) -> Option<Vec<Item>> {
    let edges = match graph {
        Graph::Raw => plain_graph(model, &extraction(model)?.raw),
        Graph::Current => plain_graph(model, current(model)?),
        Graph::Tree => tree_graph(model)?,
        Graph::Labeled => labeled_graph(model)?,
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
    let (faces, lines) = patch_surface(model, skeleton, &palette(model))?;
    Some(surface(faces, &model.camera, Look::default(), &[], &lines))
}

pub(super) fn combined(model: &Model<'_>) -> Option<Vec<Item>> {
    let current = current(model)?;
    let (faces, _) = patch_surface(model, current, &palette(model))?;
    let look = Look {
        two_sided: false,
        lighten: LIGHT_PATCHES,
    };
    let mut items = surface(faces, &model.camera, look, &[], &[]);
    items.extend(labeled_graph(model).unwrap_or_else(|| plain_graph(model, current)));
    Some(items)
}

// The cuboids of the (optimized) placement as boxes, with the faces where they connect hidden as in the GUI; the
// bevels of the edge cuboids are left out.
pub(super) fn cubes(model: &Model<'_>) -> Option<Vec<Item>> {
    let save = model.skeleton;
    let realized = save.optimized.as_ref().or(save.realized.as_ref())?;
    let layout = &realized.cuboids;
    let palette = palette(model);
    let beveled = |owner| matches!(owner, CuboidOwner::Edge(_));
    let mut hidden = HashSet::new();
    for &[a, b] in &layout.connections {
        let delta = layout.cuboids[b].center - layout.cuboids[a].center;
        let Some(axis) = (0..3).find(|&i| delta[i] != 0.) else {
            continue;
        };
        let positive = delta[axis] > 0.;
        // A square node face supplies the shoulder around an edge cuboid.
        if beveled(layout.cuboids[a].owner) || !beveled(layout.cuboids[b].owner) {
            hidden.insert((a, axis, positive));
        }
        if beveled(layout.cuboids[b].owner) || !beveled(layout.cuboids[a].owner) {
            hidden.insert((b, axis, !positive));
        }
    }
    let camera = Camera::new(model.params, None);
    let mut faces = vec![];
    for (id, cube) in &layout.cuboids {
        let fill = match cube.owner {
            CuboidOwner::Node(node) => palette.get(&node).copied().unwrap_or(GRAY),
            CuboidOwner::Edge(edge) => realized
                .labeled
                .labels
                .get(edge)
                .map_or(GRAY, |label| model.style.primal[axis(label.direction)]),
        };
        let (lo, hi) = (cube.center - cube.size / 2., cube.center + cube.size / 2.);
        for n in 0..3 {
            for positive in [false, true] {
                let toward = if positive { 1. } else { -1. };
                if hidden.contains(&(id, n, positive)) || toward * camera.forward[n] >= 0. {
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
    boxes(&faces, &camera, model.style)
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

// The input mesh cut into the skeleton's patches (colored by node), and their boundaries.
fn patch_surface(
    model: &Model<'_>,
    skeleton: &Skeleton,
    colors: &HashMap<SkeletonNodeID, Rgb>,
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
        .map(|boundary| SurfaceLine {
            points: boundary.points,
            color: model.style.line,
            width: BOUNDARY_WIDTH,
        })
        .collect();
    Some((faces, lines))
}

// A graph without labels: gray edges, nodes in their region colors.
fn plain_graph(model: &Model<'_>, skeleton: &Skeleton) -> Vec<Item> {
    let palette = palette(model);
    nodes_and_edges(
        model,
        skeleton,
        |_| (GRAY, EDGE_WIDTH),
        |node| palette.get(&node).copied().unwrap_or(GRAY),
    )
}

// The labeling (edges in their axis colors, non-tree edges thick), else the labeled skeleton.
fn labeled_graph(model: &Model<'_>) -> Option<Vec<Item>> {
    let save = model.skeleton;
    let edge = |label: Option<&EdgeLabel>, non_tree: bool| {
        (
            label.map_or(GRAY, |label| model.style.primal[axis(label.direction)]),
            if non_tree { NON_TREE_WIDTH } else { EDGE_WIDTH },
        )
    };
    if let Some(attempt) = &save.labeling {
        let non_tree = |id: SkeletonEdgeID| {
            !attempt.tree_edges.is_empty() && !attempt.tree_labels.contains_key(id)
        };
        return Some(nodes_and_edges(
            model,
            &attempt.skeleton,
            |id| edge(attempt.labels.get(id), non_tree(id)),
            |_| GRAY,
        ));
    }
    let labeled = save.labeled.as_ref()?;
    Some(nodes_and_edges(
        model,
        &labeled.skeleton,
        |id| edge(labeled.labels.get(id), false),
        |_| GRAY,
    ))
}

// The tree labeling before cycle repairs: the edges it leaves unlabeled gray and thick.
fn tree_graph(model: &Model<'_>) -> Option<Vec<Item>> {
    let attempt = model.skeleton.labeling.as_ref()?;
    let skeleton = attempt.tree_skeleton.as_ref()?;
    Some(nodes_and_edges(
        model,
        skeleton,
        |id| match attempt.tree_labels.get(id) {
            Some(label) => (model.style.primal[axis(label.direction)], EDGE_WIDTH),
            None => (GRAY, NON_TREE_WIDTH),
        },
        |_| GRAY,
    ))
}

// The edges as lines and the nodes as dots, back to front; a node in front of its edges.
fn nodes_and_edges(
    model: &Model<'_>,
    skeleton: &Skeleton,
    edge: impl Fn(SkeletonEdgeID) -> (Rgb, f64),
    node: impl Fn(SkeletonNodeID) -> Rgb,
) -> Vec<Item> {
    let camera = &model.camera;
    let position = |id: SkeletonNodeID| skeleton.nodes[id].position;
    let mut items = vec![];
    for (id, e) in &skeleton.edges {
        let (a, b) = (position(e.nodes[0]), position(e.nodes[1]));
        let (color, width) = edge(id);
        let depth = camera.depth((a + b) / 2.);
        let line = Item::Line {
            from: camera.project(a),
            to: camera.project(b),
            color,
            width,
        };
        items.push((depth, 1, line));
    }
    for id in skeleton.nodes.keys() {
        let p = position(id);
        let disk = Item::Disk {
            center: camera.project(p),
            radius: NODE_RADIUS,
            fill: node(id),
            ring: model.style.line,
        };
        items.push((camera.depth(p), 0, disk));
    }
    // Farthest first; at equal depth, edges before nodes.
    items.sort_by(|a, b| b.0.total_cmp(&a.0).then(b.1.cmp(&a.1)));
    items.into_iter().map(|(_, _, item)| item).collect()
}
