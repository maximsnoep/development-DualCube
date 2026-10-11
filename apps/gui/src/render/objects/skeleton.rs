//! Skeleton geometry and midpoint-split surface patches.

use super::super::gizmos::world_to_view;
use super::super::store::{RenderAsset, RenderObject, RenderObjectStore};
use crate::colors;
use crate::render::Objects;
use crate::resources::{InputResource, SkeletonResource};
use bevy::mesh::{Mesh as BevyMesh, VertexAttributeValues};
use bevy::prelude::*;
use dualcube::prelude::*;
use dualcube::skeleton::{LabeledSkeleton, Skeleton, SkeletonEdgeID};
use mehsh_bevy::MeshBuilder;
use std::collections::HashMap;
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

const PATCH_OPACITY: f32 = 0.3;

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
    object.add(
        SkeletonLayer::Skeleton.label(),
        RenderAsset::Mesh(graph(&extraction.skeleton, None, scale, translation)),
    );
    object.add(
        SkeletonLayer::Raw.label(),
        RenderAsset::Mesh(graph(&extraction.raw, None, scale, translation)),
    );
    if let Some(labeled) = &state.labeled {
        object.add(
            SkeletonLayer::Labeled.label(),
            RenderAsset::Mesh(graph(&labeled.skeleton, Some(labeled), scale, translation)),
        );
    }
    let mut translucent = patches(mesh, &extraction.skeleton, scale, translation);
    object.add(
        SkeletonLayer::Patches.label(),
        RenderAsset::Mesh(translucent.clone()),
    );
    if let Some(VertexAttributeValues::Float32x4(colors)) =
        translucent.attribute_mut(BevyMesh::ATTRIBUTE_COLOR)
    {
        for color in colors {
            color[3] = PATCH_OPACITY;
        }
    }
    object.add(
        SkeletonLayer::TranslucentPatches.label(),
        RenderAsset::TranslucentMesh(translucent),
    );
    let mut boundaries = GizmoAsset::new();
    for edge in extraction.skeleton.edges.values() {
        let points: Vec<_> = edge
            .boundary
            .edge_midpoints
            .iter()
            .map(|&e| {
                world_to_view(
                    (mesh.position(mesh.root(e)) + mesh.position(mesh.root(mesh.twin(e)))) * 0.5,
                    translation,
                    scale,
                )
            })
            .collect();
        if !points.is_empty() {
            boundaries.linestrip(
                points.iter().copied().chain(points.first().copied()),
                colors::to_bevy(colors::WHITE),
            );
        }
    }
    object.gizmo(boundaries, 2.0, -0.001, SkeletonLayer::Boundaries.label());
    object
}

fn graph(
    skeleton: &Skeleton,
    labeled: Option<&LabeledSkeleton>,
    scale: f64,
    translation: Vector3D,
) -> BevyMesh {
    let mut builder = MeshBuilder::new();
    let position = |id| skeleton.nodes[id].position * scale + translation;
    let edge_color = |id: SkeletonEdgeID| {
        labeled
            .and_then(|s| s.labels.get(id))
            .map_or(colors::LIGHT_GRAY, |label| {
                colors::from_direction(label.direction, Some(Perspective::Primal), None)
            })
    };
    for (id, edge) in &skeleton.edges {
        tube(
            &mut builder,
            position(edge.nodes[0]),
            position(edge.nodes[1]),
            0.09,
            edge_color(id),
        );
    }
    for (region, id) in skeleton.nodes.keys().enumerate() {
        let color = if labeled.is_some() {
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

// The two equal labels form a quad; the remaining label forms a triangle.
fn split_triangle(points: [Vector3D; 3], regions: [usize; 3]) -> Vec<([Vector3D; 3], usize)> {
    if regions[0] == regions[1] && regions[1] == regions[2] {
        return vec![(points, regions[0])];
    }
    for i in 0..3 {
        let j = (i + 1) % 3;
        let k = (i + 2) % 3;
        if regions[i] == regions[j] {
            let [a, b, c] = [points[i], points[j], points[k]];
            let ac = (a + c) * 0.5;
            let bc = (b + c) * 0.5;
            return vec![
                ([a, b, ac], regions[i]),
                ([b, bc, ac], regions[i]),
                ([ac, bc, c], regions[k]),
            ];
        }
    }
    // Diagnostic partitions can have a three-way junction.
    let center = (points[0] + points[1] + points[2]) / 3.;
    (0..3)
        .flat_map(|i| {
            let a = points[i];
            let next = (a + points[(i + 1) % 3]) * 0.5;
            let prev = (a + points[(i + 2) % 3]) * 0.5;
            [
                ([a, next, center], regions[i]),
                ([a, center, prev], regions[i]),
            ]
        })
        .collect()
}

fn patches(
    mesh: &mehsh::prelude::Mesh<INPUT>,
    skeleton: &Skeleton,
    scale: f64,
    translation: Vector3D,
) -> BevyMesh {
    let owners: HashMap<_, _> = skeleton
        .nodes
        .values()
        .enumerate()
        .flat_map(|(i, node)| node.patch_vertices.iter().map(move |&v| (v, i)))
        .collect();
    let mut builder = MeshBuilder::new();
    for f in mesh.face_ids() {
        let vertices: Vec<_> = mesh.vertices(f).collect();
        let points = [vertices[0], vertices[1], vertices[2]].map(|v| mesh.position(v));
        let regions = [vertices[0], vertices[1], vertices[2]].map(|v| owners[&v]);
        for (triangle, region) in split_triangle(points, regions) {
            builder.add_triangle(triangle, [mesh.normal(f); 3], region_color(region));
        }
    }
    builder.normalize(scale, translation);
    builder.build()
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
