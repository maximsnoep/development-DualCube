use super::*;
use dualcube::skeleton::{CuboidOwner, RealizedSkeleton};
use io::figure::CubeColoring;

// One mesh per coloring (the cuboids it leaves out gray); the edges are shared.
const COLORINGS: [(SkeletonLayer, CubeColoring); 3] = [
    (SkeletonLayer::Cubes, CubeColoring::Combined),
    (SkeletonLayer::CubesPrimal, CubeColoring::Primal),
    (SkeletonLayer::CubesDual, CubeColoring::Dual),
];

pub(super) fn add(
    object: &mut RenderObject,
    realized: &RealizedSkeleton,
    palette: &HashMap<SkeletonNodeID, colors::Kolor>,
) {
    let layout = realized.cuboids();
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
    let hidden = layout.hidden_faces();
    let mut builders = COLORINGS.map(|_| MeshBuilder::new());
    let mut lines = GizmoAsset::new();
    // The node cubes alone (in their colors), with their edges.
    let mut nodes = MeshBuilder::new();
    let mut node_lines = GizmoAsset::new();
    for (id, cube) in &layout.cuboids {
        let node = matches!(cube.owner, CuboidOwner::Node(_));
        let fills = COLORINGS.map(|(_, coloring)| match cube.owner {
            CuboidOwner::Node(_) if !coloring.colors_nodes() => colors::LIGHT_GRAY,
            CuboidOwner::Edge(_) if !coloring.colors_edges() => colors::LIGHT_GRAY,
            CuboidOwner::Node(node) => palette.get(&node).copied().unwrap_or(colors::LIGHT_GRAY),
            CuboidOwner::Edge(edge) => colors::from_direction(
                realized.labeled.labels[edge].direction,
                Some(Perspective::Primal),
                None,
            ),
        });
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
        let mut face = |mut corners: Vec<Vector3D>, normal: Vector3D, fills: [colors::Kolor; 3]| {
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
            let node_builder = node.then_some((&mut nodes, fills[0]));
            for (builder, color) in builders.iter_mut().zip(fills).chain(node_builder) {
                for i in 1..corners.len() - 1 {
                    builder.add_triangle(
                        [corners[0], corners[i], corners[i + 1]],
                        [normal; 3],
                        color,
                    );
                }
            }
            for i in 0..corners.len() {
                let view = |p| crate::render::gizmos::world_to_view(p, translation, scale);
                let (from, to) = (view(corners[i]), view(corners[(i + 1) % corners.len()]));
                let color = colors::to_bevy(colors::DARK_GRAY);
                lines.line(from, to, color);
                if node {
                    node_lines.line(from, to, color);
                }
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
                    fills,
                );
            }
        }
        let Some(axis) = edge_axis else {
            continue;
        };
        let bevel = fills.map(|color| color.map(|c| c * 0.8));
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
                face(vec![p, q, r, s], normal.normalize(), bevel);
            }
        }
    }
    for ((layer, _), mut builder) in COLORINGS.into_iter().zip(builders) {
        builder.normalize(scale, translation);
        object.add(layer.label(), RenderAsset::Mesh(builder.build()));
    }
    object.gizmo(lines, 1., -0.00001, SkeletonLayer::CubeEdges.label());
    nodes.normalize(scale, translation);
    object.add(
        SkeletonLayer::NodeCubes.label(),
        RenderAsset::Mesh(nodes.build()),
    );
    object.gizmo(
        node_lines,
        1.,
        -0.00001,
        SkeletonLayer::NodeCubeEdges.label(),
    );
    template(object, realized, scale, translation);
    // The labeled skeleton at the centers of its node cubes.
    let mut placed = realized.labeled.skeleton.clone();
    for (id, node) in &mut placed.nodes {
        node.position = realized.node_positions[id];
    }
    object.add(
        SkeletonLayer::PlacedSkeleton.label(),
        RenderAsset::Mesh(graph(
            &placed,
            Some(&realized.labeled),
            &HashMap::new(),
            scale,
            translation,
        )),
    );
}

// The cube surface of the loop drawing's template in gray, with its loops: all of them, and the lifted ones (the patch
// boundaries) alone.
fn template(
    object: &mut RenderObject,
    realized: &RealizedSkeleton,
    scale: f64,
    translation: Vector3D,
) {
    // Lifted off the faces, so that the lines are not hidden in them.
    const LIFT: f64 = 0.02;
    let view = |p| crate::render::gizmos::world_to_view(p, translation, scale);
    let mut faces = MeshBuilder::new();
    let mut outlines = GizmoAsset::new();
    let mut loops = GizmoAsset::new();
    let mut lifted = GizmoAsset::new();
    let outline = colors::to_bevy(colors::DARK_GRAY);
    for solid in realized.template_solids() {
        for normal in 0..3 {
            for positive in [false, true] {
                if solid.hidden[normal][usize::from(positive)] {
                    continue;
                }
                let (u, v) = ((normal + 1) % 3, (normal + 2) % 3);
                let corner = |a: bool, b: bool| {
                    let mut p = if positive { solid.hi } else { solid.lo };
                    p[u] = if a { solid.hi[u] } else { solid.lo[u] };
                    p[v] = if b { solid.hi[v] } else { solid.lo[v] };
                    p
                };
                // Counterclockwise seen from outside.
                let mut corners = [
                    corner(false, false),
                    corner(true, false),
                    corner(true, true),
                    corner(false, true),
                ];
                if !positive {
                    corners.reverse();
                }
                let mut n = Vector3D::zeros();
                n[normal] = if positive { 1. } else { -1. };
                for i in 1..3 {
                    faces.add_triangle(
                        [corners[0], corners[i], corners[i + 1]],
                        [n; 3],
                        colors::LIGHT_GRAY,
                    );
                }
                for i in 0..4 {
                    outlines.line(view(corners[i]), view(corners[(i + 1) % 4]), outline);
                }
            }
        }
        for section in solid.sections() {
            let mut lift = Vector3D::zeros();
            lift[section.normal] = if section.positive { LIFT } else { -LIFT };
            let [from, to] = section.ends.map(|p| view(p + lift));
            let color = colors::to_bevy(colors::from_direction(
                DIRECTIONS[section.axis],
                Some(Perspective::Dual),
                None,
            ));
            loops.line(from, to, color);
            if section.boundary {
                lifted.line(from, to, color);
            }
        }
    }
    faces.normalize(scale, translation);
    object.add(
        SkeletonLayer::TemplateCubes.label(),
        RenderAsset::Mesh(faces.build()),
    );
    object.gizmo(
        outlines,
        1.,
        -0.00001,
        SkeletonLayer::TemplateCubeEdges.label(),
    );
    object.gizmo(loops, 4., -0.0001, SkeletonLayer::CubeLoops.label());
    object.gizmo(lifted, 4., -0.0001, SkeletonLayer::CubeLiftedLoops.label());
}
