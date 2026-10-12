use super::*;
use dualcube::skeleton::{CuboidOwner, RealizedSkeleton};
use std::collections::HashSet;

pub(super) fn add(object: &mut RenderObject, realized: &RealizedSkeleton) {
    let layout = &realized.cuboids;
    if layout.cuboids.is_empty() {
        return;
    }
    let mut lower = Vector3D::repeat(f64::INFINITY);
    let mut upper = Vector3D::repeat(f64::NEG_INFINITY);
    for cube in layout.cuboids.values() {
        lower = lower.inf(&(cube.center - cube.size / 2.));
        upper = upper.sup(&(cube.center + cube.size / 2.));
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
        let delta = layout.cuboids[b].center - layout.cuboids[a].center;
        let axis = (0..3).find(|&i| delta[i] != 0.).unwrap();
        let positive = delta[axis] > 0.;
        // A square node face supplies the shoulder around a beveled neighbor.
        if beveled(layout.cuboids[a].owner) || !beveled(layout.cuboids[b].owner) {
            hidden.insert((a, axis, positive));
        }
        if beveled(layout.cuboids[b].owner) || !beveled(layout.cuboids[a].owner) {
            hidden.insert((b, axis, !positive));
        }
    }
    let mut builder = MeshBuilder::new();
    let mut lines = GizmoAsset::new();
    for (id, cube) in &layout.cuboids {
        let color = match cube.owner {
            CuboidOwner::Node(node) => palette[&node],
            CuboidOwner::Edge(edge) => colors::from_direction(
                realized.labeled.labels[edge].direction,
                Some(Perspective::Primal),
                None,
            ),
        };
        let edge_axis = match cube.owner {
            CuboidOwner::Node(_) => None,
            CuboidOwner::Edge(edge) => Some(match realized.labeled.labels[edge].direction {
                Direction::X => 0,
                Direction::Y => 1,
                Direction::Z => 2,
            }),
        };
        let half = cube.size / 2.;
        let mut inner = half;
        if let Some(axis) = edge_axis {
            let width = cube.size[(axis + 1) % 3].min(cube.size[(axis + 2) % 3]) * 0.16;
            for d in 0..3 {
                if d != axis {
                    inner[d] -= width;
                }
            }
        }
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
                u[u_axis] = inner[u_axis];
                let mut v = Vector3D::zeros();
                let v_axis = (axis + 2) % 3;
                v[v_axis] = inner[v_axis];
                let center = normal * half[axis];
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
        // Bevel only the four longitudinal edges; endpoint joints stay flush.
        let u = (axis + 1) % 3;
        let v = (axis + 2) % 3;
        for a in [-1., 1.] {
            for b in [-1., 1.] {
                let mut p = Vector3D::zeros();
                p[u] = a * half[u];
                p[v] = b * inner[v];
                p[axis] = -half[axis];
                let mut q = p;
                q[u] = a * inner[u];
                q[v] = b * half[v];
                let mut r = q;
                r[axis] = half[axis];
                let mut s = p;
                s[axis] = half[axis];
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

fn beveled(owner: CuboidOwner) -> bool {
    matches!(owner, CuboidOwner::Edge(_))
}
