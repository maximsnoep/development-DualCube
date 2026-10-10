use crate::colors;
use crate::colors::Kolor;
use bevy::prelude::*;
use dualcube::prelude::*;
use std::collections::HashMap;

#[derive(Default, Reflect, GizmoConfigGroup)]
pub struct PerpetualGizmos {}

pub fn setup(mut config_store: ResMut<'_, GizmoConfigStore>) {
    let (perp_gizmos, _) = config_store.config_mut::<PerpetualGizmos>();
    perp_gizmos.depth_bias = -1.0;
}

#[inline]
#[must_use]
pub fn vector3d_to_vec3(v: Vector3D) -> Vec3 {
    Vec3::new(v.x as f32, v.y as f32, v.z as f32)
}

#[must_use]
pub fn world_to_view(v: Vector3D, translation: Vector3D, scale: f64) -> Vec3 {
    vector3d_to_vec3(v * scale + translation)
}

#[must_use]
pub fn view_to_world(v: Vec3, translation: Vector3D, scale: f64) -> Vector3D {
    (Vector3D::new(f64::from(v.x), f64::from(v.y), f64::from(v.z)) - translation) / scale
}

pub struct DirectionalGizmos {
    pub x: GizmoAsset,
    pub y: GizmoAsset,
    pub z: GizmoAsset,
}

impl Default for DirectionalGizmos {
    fn default() -> Self {
        Self {
            x: GizmoAsset::new(),
            y: GizmoAsset::new(),
            z: GizmoAsset::new(),
        }
    }
}

impl DirectionalGizmos {
    pub fn get_mut(&mut self, direction: Direction) -> &mut GizmoAsset {
        match direction {
            Direction::X => &mut self.x,
            Direction::Y => &mut self.y,
            Direction::Z => &mut self.z,
        }
    }
}

#[must_use]
pub fn uniform_color_map<M: Tag>(
    mesh: &mehsh::prelude::Mesh<M>,
    color: Kolor,
) -> HashMap<FaceKey<M>, Kolor> {
    mesh.face_ids()
        .into_iter()
        .map(|face_id| (face_id, color))
        .collect()
}

#[must_use]
pub fn lambert_color_map<M: Tag>(mesh: &mehsh::prelude::Mesh<M>) -> HashMap<FaceKey<M>, Kolor> {
    let light_dir = Vector3D::new(-25.0, 25.0, 25.0).normalize();
    let wrap = 0.5;

    mesh.face_ids()
        .into_iter()
        .map(|face_id| {
            let shade = lambert_shade(mesh.normal(face_id).normalize(), light_dir, wrap);
            (face_id, [shade, shade, shade])
        })
        .collect()
}

#[must_use]
pub fn segmentation_color_map(layout: &Layout, polycube: &Polycube) -> HashMap<FaceID, Kolor> {
    let mut color_map = HashMap::new();
    for &patch_id in &polycube.structure.face_ids() {
        let normal = (polycube.structure.normal(patch_id) as Vector3D).normalize();
        let (dir, side) = to_principal_direction(normal);
        let color = colors::from_direction(dir, Some(Perspective::Primal), Some(side));
        // The layout may be incomplete (e.g., when placing the paths failed).
        let Some(patch) = layout.face_to_patch.get(&patch_id) else {
            continue;
        };
        for &triangle_id in &patch.faces {
            color_map.insert(triangle_id, color);
        }
    }
    color_map
}

#[must_use]
pub fn edge_endpoints_view<M: Tag>(
    mesh: &mehsh::prelude::Mesh<M>,
    edge_id: EdgeKey<M>,
    translation: Vector3D,
    scale: f64,
) -> (Vec3, Vec3) {
    let Some([v1, v2]) = mesh.vertices(edge_id).collect_array::<2>() else {
        panic!("Expected two vertices for edge {edge_id:?}");
    };
    (
        world_to_view(mesh.position(v1), translation, scale),
        world_to_view(mesh.position(v2), translation, scale),
    )
}

#[must_use]
pub fn flow_graph_gizmos(
    edges: &[(EdgeID, EdgeID, f64)],
    mesh: &mehsh::prelude::Mesh<INPUT>,
    base_color: Kolor,
    translation: Vector3D,
    scale: f64,
) -> GizmoAsset {
    let mut gizmos = GizmoAsset::new();

    let Some(threshold) = flow_weight_threshold(edges, FLOW_GRAPH_DRAWN_PERCENT) else {
        return gizmos;
    };

    let color = colors::to_bevy(base_color);

    for &(from, to, weight) in edges {
        if weight <= 1e-9 || weight > threshold {
            continue;
        }

        gizmos.arrow(
            world_to_view(mesh.position(from), translation, scale),
            world_to_view(mesh.position(to), translation, scale),
            color,
        );
    }

    gizmos
}

fn lambert_shade(normal: Vector3D, light_dir: Vector3D, wrap: f64) -> f32 {
    let diffuse =
        ((normal.dot(&light_dir) as f32 + wrap as f32) / (1.0 + wrap as f32)).clamp(0.0, 1.0);
    let hemi = 0.2 + 0.15 * ((normal.y as f32) * 0.5 + 0.5);
    (0.75 * diffuse + 0.25 * hemi).clamp(0.0, 1.0)
}

// Only the cheapest moves of the flow graphs are drawn (all of them would hide the mesh).
const FLOW_GRAPH_DRAWN_PERCENT: f32 = 20.0;

fn flow_weight_threshold(edges: &[(EdgeID, EdgeID, f64)], top_percent: f32) -> Option<f64> {
    let mut weights = edges
        .iter()
        .filter_map(|&(_, _, weight)| (weight > 1e-9).then_some(weight))
        .collect::<Vec<_>>();

    weights.sort_by(f64::total_cmp);
    let keep_count = (weights.len() as f32 * top_percent.clamp(0.0, 100.0) / 100.0).ceil() as usize;
    (keep_count > 0).then(|| weights[keep_count.saturating_sub(1).min(weights.len() - 1)])
}

/// The index of a principal direction (X, Y, Z).
#[must_use]
pub const fn axis_index(direction: Direction) -> usize {
    match direction {
        Direction::X => 0,
        Direction::Y => 1,
        Direction::Z => 2,
    }
}

/// Adds a loop's band (see `Solution::loop_frames`) along the given segments: a strip of `half` wide on either side,
/// its positive side in `positive` and its negative side in `negative`, lifted by `lift` along the normals (so that the
/// surface does not cover it). The segments share their frames, so the strips connect without gaps.
pub fn add_band(
    builder: &mut mehsh_bevy::MeshBuilder,
    segments: impl IntoIterator<Item = (LoopFrame, LoopFrame)>,
    half: f64,
    lift: f64,
    [positive, negative]: [Kolor; 2],
) {
    for (a, b) in segments {
        let (pa, pb) = (a.position + a.normal * lift, b.position + b.normal * lift);
        for (side, color) in [(-1., positive), (1., negative)] {
            let (qa, qb) = (
                pa + a.negative * (side * half),
                pb + b.negative * (side * half),
            );
            // Two triangles, facing outward.
            for [x, y, z] in [[pa, pb, qb], [pa, qb, qa]] {
                let outward = (y - x).cross(&(z - x)).dot(&(a.normal + b.normal)) >= 0.;
                let corners = if outward { [x, y, z] } else { [x, z, y] };
                builder.add_triangle(corners, [a.normal, b.normal, b.normal], color);
            }
        }
    }
}

/// The colors of a loop's band: its positive side, and its (lighter) negative side.
#[must_use]
pub fn band_colors(direction: Direction) -> [Kolor; 2] {
    let positive = colors::from_direction(direction, Some(Perspective::Dual), None);
    [positive, positive.map(|c| c + (1. - c) * super::LIGHT_MIX)]
}

/// Adds a round cap (a disk of radius `half`) at a point of the surface, lifted by `lift` along its normal: at the ends
/// of a band, so that bands that meet there join smoothly.
pub fn add_cap(
    builder: &mut mehsh_bevy::MeshBuilder,
    frame: &LoopFrame,
    half: f64,
    lift: f64,
    color: Kolor,
) {
    let center = frame.position + frame.normal * lift;
    let corners = disk(center, frame.normal, half, 16);
    for (k, &a) in corners.iter().enumerate() {
        let b = corners[(k + 1) % corners.len()];
        builder.add_triangle([center, a, b], [frame.normal; 3], color);
    }
}

/// Adds a flat quadrilateral (corners in order) facing along `normal`.
pub fn add_quad(
    builder: &mut mehsh_bevy::MeshBuilder,
    corners: [Vector3D; 4],
    normal: Vector3D,
    color: Kolor,
) {
    let [a, b, c, d] = corners;
    for [x, y, z] in [[a, b, c], [a, c, d]] {
        let outward = (y - x).cross(&(z - x)).dot(&normal) >= 0.;
        let corners = if outward { [x, y, z] } else { [x, z, y] };
        builder.add_triangle(corners, [normal; 3], color);
    }
}

/// Paths as strips (see `Solution::path_bands`): those between patches with different normals (the edges of the
/// polycube), and all paths (thinner), relative to the half width of the loops, lifted by `lift`; with round ends where
/// they meet.
#[allow(unused_qualifications)]
pub fn path_strips(
    bands: impl Fn(f64) -> Vec<PathBand>,
    loop_half: f64,
    lift: f64,
    translation: Vector3D,
    scale: f64,
) -> [bevy::mesh::Mesh; 2] {
    let mut builders = [(); 2].map(|()| mehsh_bevy::MeshBuilder::new());
    for (builder, factor, all) in [(0, 0.6, false), (1, 0.35, true)] {
        let half = factor * loop_half;
        for band in bands(half) {
            if band.flat && !all {
                continue;
            }
            let segments = band.frames.windows(2).map(|w| (w[0], w[1]));
            add_band(
                &mut builders[builder],
                segments,
                half,
                lift,
                [colors::GRAY; 2],
            );
            for end in [band.frames.first(), band.frames.last()]
                .into_iter()
                .flatten()
            {
                add_cap(&mut builders[builder], end, half, lift, colors::GRAY);
            }
        }
    }
    builders.map(|mut builder| {
        builder.normalize(scale, translation);
        builder.build()
    })
}

/// Half the width of the loops (`width`: a fraction of the diagonal of the mesh's bounding box).
#[must_use]
pub fn loop_half_width<M: Tag>(mesh: &mehsh::prelude::Mesh<M>, width: f64) -> f64 {
    let (min, max) = mesh.vert_ids().iter().map(|&v| mesh.position(v)).fold(
        (
            Vector3D::repeat(f64::INFINITY),
            Vector3D::repeat(f64::NEG_INFINITY),
        ),
        |(min, max), p| (min.inf(&p), max.sup(&p)),
    );
    0.5 * width * (max - min).norm().max(1e-12)
}
