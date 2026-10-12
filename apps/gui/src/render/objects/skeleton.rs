//! Skeleton graphs and boundary-defined surface patches.

use super::super::store::{RenderAsset, RenderObject, RenderObjectStore};
use crate::colors;
use crate::render::Objects;
use crate::resources::{InputResource, SkeletonResource};
use bevy::mesh::{Mesh as BevyMesh, VertexAttributeValues};
use bevy::prelude::*;
use dualcube::prelude::*;
use dualcube::skeleton::{LabeledSkeleton, Skeleton, SkeletonEdgeID};
use mehsh_bevy::MeshBuilder;
use std::f64::consts::{PI, TAU};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkeletonLayer {
    Skeleton,
    Raw,
    Labeled,
    Patches,
    TranslucentPatches,
    Boundaries,
    Deflated,
}

impl SkeletonLayer {
    pub const ALL: [Self; 7] = [
        Self::Skeleton,
        Self::Raw,
        Self::Labeled,
        Self::Patches,
        Self::TranslucentPatches,
        Self::Boundaries,
        Self::Deflated,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Skeleton => "skeleton",
            Self::Raw => "raw skeleton",
            Self::Labeled => "labeled skeleton",
            Self::Patches => "patches",
            Self::TranslucentPatches => "translucent patches",
            Self::Boundaries => "patch boundaries",
            Self::Deflated => "deflated",
        }
    }

    pub fn from_label(label: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|layer| layer.label() == label)
    }

    pub fn is_surface(self) -> bool {
        matches!(
            self,
            Self::Patches | Self::TranslucentPatches | Self::Boundaries
        )
    }
}

pub fn sync(
    state: Res<'_, SkeletonResource>,
    input: Res<'_, InputResource>,
    mut store: ResMut<'_, RenderObjectStore>,
    mut cached: Local<'_, RenderObject>,
) {
    let changed = state.is_changed() || input.is_changed();
    if !changed && !store.is_changed() {
        return;
    }
    if changed {
        *cached = build(&input.mesh, &state);
    }
    if let Some(object) = store.objects.get_mut(&Objects::InputMesh) {
        object
            .labels
            .retain(|label| SkeletonLayer::from_label(label).is_none());
        object
            .features
            .retain(|label, _| SkeletonLayer::from_label(label).is_none());
        for label in &cached.labels {
            object.add(label, cached.features[label].clone());
        }
    }
}

fn build(mesh: &mehsh::prelude::Mesh<INPUT>, state: &SkeletonResource) -> RenderObject {
    let mut object = RenderObject::default();
    let Some(contracted) = state.contracted_mesh() else {
        return object;
    };
    let (scale, translation) = mesh.scale_translation();
    let mut deflated = MeshBuilder::new();
    for f in contracted.face_ids() {
        let points: Vec<_> = contracted
            .vertices(f)
            .map(|v| contracted.position(v))
            .collect();
        let p = [points[0], points[1], points[2]];
        let normal = (p[1] - p[0])
            .cross(&(p[2] - p[0]))
            .try_normalize(1e-20)
            .unwrap_or(Vector3D::z());
        deflated.add_triangle(p, [normal; 3], colors::LIGHT_GRAY);
    }
    deflated.normalize(scale, translation);
    object.add(
        SkeletonLayer::Deflated.label(),
        RenderAsset::Mesh(deflated.build()),
    );
    let Some(extraction) = state.extraction() else {
        return object;
    };
    let current = state
        .labeling
        .as_ref()
        .map(|attempt| &attempt.skeleton)
        .or_else(|| state.labeled.as_ref().map(|labeled| &labeled.skeleton))
        .unwrap_or(&extraction.skeleton);
    object.add(
        SkeletonLayer::Skeleton.label(),
        RenderAsset::Mesh(graph(current, None, scale, translation)),
    );
    object.add(
        SkeletonLayer::Raw.label(),
        RenderAsset::Mesh(graph(&extraction.raw, None, scale, translation)),
    );
    if let Some(attempt) = &state.labeling {
        object.add(
            SkeletonLayer::Labeled.label(),
            RenderAsset::Mesh(graph_with_labels(
                &attempt.skeleton,
                |e| attempt.labels.get(e).copied(),
                |e| !attempt.tree_edges.is_empty() && !attempt.tree_labels.contains_key(e),
                true,
                scale,
                translation,
            )),
        );
    } else if let Some(labeled) = &state.labeled {
        object.add(
            SkeletonLayer::Labeled.label(),
            RenderAsset::Mesh(graph(&labeled.skeleton, Some(labeled), scale, translation)),
        );
    }
    match dualcube::skeleton::geometry::SurfaceGeometry::build(mesh, current) {
        Ok(geometry) => add_surface(&mut object, current, &geometry, scale, translation),
        Err(error) => warn!("Could not build skeleton surface geometry: {error}"),
    }
    object
}

fn add_surface(
    object: &mut RenderObject,
    skeleton: &Skeleton,
    geometry: &dualcube::skeleton::geometry::SurfaceGeometry,
    scale: f64,
    translation: Vector3D,
) {
    let colors: HashMap<_, _> = skeleton
        .nodes
        .keys()
        .enumerate()
        .map(|(i, id)| (id, region_color(i)))
        .collect();
    let mut patches = MeshBuilder::new();
    for triangle in &geometry.triangles {
        let [a, b, c] = triangle.points;
        let normal = (b - a).cross(&(c - a)).normalize();
        patches.add_triangle(triangle.points, [normal; 3], colors[&triangle.node]);
    }
    patches.normalize(scale, translation);
    let mut patches = patches.build();
    object.add(
        SkeletonLayer::Patches.label(),
        RenderAsset::Mesh(patches.clone()),
    );
    if let Some(VertexAttributeValues::Float32x4(colors)) =
        patches.attribute_mut(BevyMesh::ATTRIBUTE_COLOR)
    {
        for color in colors {
            color[3] = 0.3;
        }
    }
    object.add(
        SkeletonLayer::TranslucentPatches.label(),
        RenderAsset::TranslucentMesh(patches),
    );
    let mut boundaries = GizmoAsset::new();
    for boundary in &geometry.boundaries {
        for pair in boundary.points.windows(2) {
            boundaries.line(
                crate::render::gizmos::world_to_view(pair[0], translation, scale),
                crate::render::gizmos::world_to_view(pair[1], translation, scale),
                colors::to_bevy(colors::DARK_GRAY),
            );
        }
    }
    object.gizmo(boundaries, 1.5, -0.001, SkeletonLayer::Boundaries.label());
}

fn graph(
    skeleton: &Skeleton,
    labeled: Option<&LabeledSkeleton>,
    scale: f64,
    translation: Vector3D,
) -> BevyMesh {
    graph_with_labels(
        skeleton,
        |e| labeled.and_then(|s| s.labels.get(e)).copied(),
        |_| false,
        labeled.is_some(),
        scale,
        translation,
    )
}

fn graph_with_labels(
    skeleton: &Skeleton,
    label: impl Fn(SkeletonEdgeID) -> Option<dualcube::skeleton::EdgeLabel>,
    non_tree: impl Fn(SkeletonEdgeID) -> bool,
    labeled: bool,
    scale: f64,
    translation: Vector3D,
) -> BevyMesh {
    let mut builder = MeshBuilder::new();
    let position = |id| skeleton.nodes[id].position * scale + translation;
    let edge_color = |id: SkeletonEdgeID| {
        label(id).map_or(colors::LIGHT_GRAY, |label| {
            colors::from_direction(label.direction, Some(Perspective::Primal), None)
        })
    };
    for (id, edge) in &skeleton.edges {
        tube(
            &mut builder,
            position(edge.nodes[0]),
            position(edge.nodes[1]),
            // Repair subdivisions inherit non-MST status; unresolved cycle edges stay gray.
            if non_tree(id) { 0.2 } else { 0.09 },
            edge_color(id),
        );
    }
    for (region, id) in skeleton.nodes.keys().enumerate() {
        let color = if labeled {
            colors::LIGHT_GRAY
        } else {
            region_color(region)
        };
        sphere(&mut builder, position(id), 0.2, color);
    }
    builder.build()
}

fn sphere(builder: &mut MeshBuilder, center: Vector3D, radius: f64, color: colors::Kolor) {
    let normal = |i: usize, j: usize| {
        let theta = PI * i as f64 / 8.;
        let phi = TAU * j as f64 / 12.;
        Vector3D::new(
            theta.sin() * phi.cos(),
            theta.cos(),
            theta.sin() * phi.sin(),
        )
    };
    for i in 0..8 {
        for j in 0..12 {
            let [a, b, c, d] = [
                normal(i, j),
                normal(i, j + 1),
                normal(i + 1, j + 1),
                normal(i + 1, j),
            ];
            for normals in [[a, b, c], [a, c, d]] {
                if (normals[1] - normals[0])
                    .cross(&(normals[2] - normals[0]))
                    .norm_squared()
                    > 1e-20
                {
                    builder.add_triangle(normals.map(|n| center + radius * n), normals, color);
                }
            }
        }
    }
}

fn tube(builder: &mut MeshBuilder, a: Vector3D, b: Vector3D, radius: f64, color: colors::Kolor) {
    let Some(axis) = (b - a).try_normalize(1e-12) else {
        return;
    };
    let reference = if axis.x.abs() < 0.9 {
        Vector3D::x()
    } else {
        Vector3D::y()
    };
    let u = axis.cross(&reference).normalize();
    let v = axis.cross(&u);
    for i in 0..12 {
        let normal = |j| {
            let theta = TAU * j as f64 / 12.;
            u * theta.cos() + v * theta.sin()
        };
        let (n, m) = (normal(i), normal(i + 1));
        builder.add_triangle(
            [a + radius * n, a + radius * m, b + radius * m],
            [n, m, m],
            color,
        );
        builder.add_triangle(
            [a + radius * n, b + radius * m, b + radius * n],
            [n, m, n],
            color,
        );
    }
}

/// Spread successive regions around the hue wheel without fading larger patch sets.
fn region_color(region: usize) -> [f32; 3] {
    let hue = ((0.58 + region as f64 * 0.618_033_988_749_895).fract() * 360.) as f32;
    let (saturation, value) = match region % 3 {
        0 => (0.78, 0.90),
        1 => (0.65, 0.76),
        _ => (0.85, 0.98),
    };
    let srgb = Srgba::from(Hsva::new(hue, saturation, value, 1.));
    [srgb.red, srgb.green, srgb.blue]
}

#[cfg(test)]
mod tests;
