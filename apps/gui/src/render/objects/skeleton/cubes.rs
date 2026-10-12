use super::*;
use dualcube::skeleton::{CubeOwner, RealizedSkeleton};
use std::collections::HashSet;

pub(super) fn add(object: &mut RenderObject, realized: &RealizedSkeleton) {
    let layout = &realized.cubes;
    if layout.cubes.is_empty() {
        return;
    }
    let mut lower = Vector3D::repeat(f64::INFINITY);
    let mut upper = Vector3D::repeat(f64::NEG_INFINITY);
    for cube in layout.cubes.values() {
        lower = lower.inf(&(cube.center - Vector3D::repeat(cube.size / 2.)));
        upper = upper.sup(&(cube.center + Vector3D::repeat(cube.size / 2.)));
    }
    let scale = 32. / (upper - lower).amax();
    let translation = -(lower + upper) * (scale / 2.);
    let palette: HashMap<_, _> = realized
        .labeled
        .skeleton
        .nodes
        .keys()
        .enumerate()
        .map(|(i, id)| (id, region_color(i)))
        .collect();
    let mut hidden = HashSet::new();
    for &[a, b] in &layout.connections {
        let delta = layout.cubes[b].center - layout.cubes[a].center;
        let axis = (0..3).find(|&i| delta[i] != 0.).unwrap();
        let positive = delta[axis] > 0.;
        // A square node face supplies the shoulder around a beveled neighbor.
        if beveled(layout.cubes[a].owner) || !beveled(layout.cubes[b].owner) {
            hidden.insert((a, axis, positive));
        }
        if beveled(layout.cubes[b].owner) || !beveled(layout.cubes[a].owner) {
            hidden.insert((b, axis, !positive));
        }
    }
    let mut builder = MeshBuilder::new();
    let mut lines = GizmoAsset::new();
    for (id, cube) in &layout.cubes {
        let color = match cube.owner {
            CubeOwner::Node(node) | CubeOwner::Edge { patch: node, .. } => palette[&node],
            CubeOwner::Midpoint(edge) => colors::from_direction(
                realized.labeled.labels[edge].direction,
                Some(Perspective::Primal),
                None,
            ),
        };
        let edge_axis = match cube.owner {
            CubeOwner::Node(_) => None,
            CubeOwner::Edge { edge, .. } | CubeOwner::Midpoint(edge) => {
                Some(match realized.labeled.labels[edge].direction {
                    Direction::X => 0,
                    Direction::Y => 1,
                    Direction::Z => 2,
                })
            }
        };
        let half = cube.size / 2.;
        let inner = if edge_axis.is_some() {
            half - cube.size * 0.16
        } else {
            half
        };
        let mut face = |mut corners: Vec<Vector3D>, normal: Vector3D, color| {
            if (corners[1] - corners[0])
                .cross(&(corners[2] - corners[0]))
                .dot(&normal)
                < 0.
            {
                corners.reverse();
            }
            for p in &mut corners {
                *p += cube.center;
            }
            for i in 1..corners.len() - 1 {
                builder.add_triangle([corners[0], corners[i], corners[i + 1]], [normal; 3], color);
            }
            for i in 0..corners.len() {
                lines.line(
                    crate::render::gizmos::world_to_view(corners[i], translation, scale),
                    crate::render::gizmos::world_to_view(
                        corners[(i + 1) % corners.len()],
                        translation,
                        scale,
                    ),
                    colors::to_bevy(colors::DARK_GRAY),
                );
            }
        };
        for axis in 0..3 {
            for positive in [false, true] {
                if hidden.contains(&(id, axis, positive)) {
                    continue;
                }
                let mut normal = Vector3D::zeros();
                normal[axis] = if positive { 1. } else { -1. };
                let mut u = Vector3D::zeros();
                let u_axis = (axis + 1) % 3;
                u[u_axis] = if edge_axis == Some(u_axis) {
                    half
                } else {
                    inner
                };
                let mut v = Vector3D::zeros();
                let v_axis = (axis + 2) % 3;
                v[v_axis] = if edge_axis == Some(v_axis) {
                    half
                } else {
                    inner
                };
                let center = normal * half;
                face(
                    vec![
                        center - u - v,
                        center + u - v,
                        center + u + v,
                        center - u + v,
                    ],
                    normal,
                    color,
                );
            }
        }
        let Some(axis) = edge_axis else {
            continue;
        };
        let bevel_color = color.map(|c| c * 0.8);
        // Bevel only the four longitudinal edges; cube-to-cube joints stay flush.
        let u = (axis + 1) % 3;
        let v = (axis + 2) % 3;
        for a in [-1., 1.] {
            for b in [-1., 1.] {
                let mut p = Vector3D::zeros();
                p[u] = a * half;
                p[v] = b * inner;
                p[axis] = -half;
                let mut q = p;
                q[u] = a * inner;
                q[v] = b * half;
                let mut r = q;
                r[axis] = half;
                let mut s = p;
                s[axis] = half;
                let mut normal = Vector3D::zeros();
                normal[u] = a;
                normal[v] = b;
                face(vec![p, q, r, s], normal.normalize(), bevel_color);
            }
        }
    }
    builder.normalize(scale, translation);
    object.add(
        SkeletonLayer::Cubes.label(),
        RenderAsset::Mesh(builder.build()),
    );
    object.gizmo(lines, 1., -0.00001, SkeletonLayer::CubeEdges.label());
}

fn beveled(owner: CubeOwner) -> bool {
    matches!(owner, CubeOwner::Edge { .. } | CubeOwner::Midpoint(_))
}
