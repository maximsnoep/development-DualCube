//! Abstract cuboid face gluing. Coordinates never identify unrelated solids.

use super::*;
use crate::{CuboidID, CuboidOwner, SkeletonEdgeID};
use petgraph::unionfind::UnionFind;

pub(super) struct Template {
    pub mesh: Mesh<INPUT>,
    pub loops: DualLoops,
    pub boundaries: HashMap<SkeletonEdgeID, LoopID>,
}

impl Template {
    pub fn build(realized: &RealizedSkeleton) -> SkeletonResult<Self> {
        let layout = &realized.cuboids;
        let ids: Vec<_> = layout.cuboids.keys().collect();
        let indices: HashMap<CuboidID, usize> =
            ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let mut corners = UnionFind::new(8 * ids.len());
        let mut hidden = HashSet::new();
        for &[a, b] in &layout.connections {
            let delta = layout.cuboids[b].center - layout.cuboids[a].center;
            let axis = (0..3)
                .find(|&i| delta[i] != 0.)
                .ok_or("Coincident connected cuboids")?;
            let positive = delta[axis] > 0.;
            hidden.insert((a, axis, positive));
            hidden.insert((b, axis, !positive));
            for c in 0..8 {
                if (c & (1 << axis) != 0) == positive {
                    corners.union(8 * indices[&a] + c, 8 * indices[&b] + (c ^ (1 << axis)));
                }
            }
        }
        let mut positions = Vec::new();
        let mut vertex = HashMap::new();
        let mut triangles = Vec::new();
        let mut owners = Vec::new();
        for &id in &ids {
            let cuboid = &layout.cuboids[id];
            for axis in 0..3 {
                for positive in [false, true] {
                    if hidden.contains(&(id, axis, positive)) {
                        continue;
                    }
                    let u = (axis + 1) % 3;
                    let v = (axis + 2) % 3;
                    let base = usize::from(positive) << axis;
                    let mut codes = [base, base | 1 << u, base | 1 << u | 1 << v, base | 1 << v];
                    if !positive {
                        codes.reverse();
                    }
                    let mut quad = [0usize; 4];
                    for (j, c) in codes.into_iter().enumerate() {
                        let root = corners.find(8 * indices[&id] + c);
                        quad[j] = *vertex.entry(root).or_insert_with(|| {
                            let p = Vector3D::from_fn(|d, _| {
                                cuboid.center[d]
                                    + cuboid.size[d] * if c & (1 << d) == 0 { -0.5 } else { 0.5 }
                            });
                            positions.push(p);
                            positions.len() - 1
                        });
                    }
                    // Off-center fan keeps the two face midlines' intersection inside a triangle.
                    let center = positions.len();
                    positions.push(
                        positions[quad[0]] * 0.46
                            + positions[quad[1]] * 0.23
                            + positions[quad[3]] * 0.31,
                    );
                    for j in 0..4 {
                        triangles.push(vec![quad[j], quad[(j + 1) % 4], center]);
                        owners.push((id, axis));
                    }
                }
            }
        }
        let (mesh, _, face_map) = Mesh::<INPUT>::from(&triangles, &positions)
            .map_err(|_| "Cuboid surface is not manifold")?;
        let mut arcs = HashMap::new();
        let mut entries = Vec::new();
        for face in mesh.face_ids() {
            let index = *face_map.id(&face).ok_or("Missing template face")?;
            let (id, normal_axis) = owners[index];
            let cuboid = &layout.cuboids[id];
            for axis in 0..3 {
                if axis == normal_axis {
                    continue;
                }
                let mut points = Vec::new();
                for edge in mesh.edges(face) {
                    let p = mesh.position(mesh.root(edge));
                    let q = mesh.position(mesh.toor(edge));
                    let a = p[axis] - cuboid.center[axis];
                    let b = q[axis] - cuboid.center[axis];
                    if a * b < 0. {
                        let t = a / (a - b);
                        points.push((edge, t, p + (q - p) * t));
                    }
                }
                if points.is_empty() {
                    continue;
                }
                if points.len() != 2 {
                    return Err("Degenerate template section".into());
                }
                let direction = DIRECTIONS[axis];
                let tangent = mesh.normal(face).cross(&Vector3D::from(direction));
                if (points[1].2 - points[0].2).dot(&tangent) < 0. {
                    points.swap(0, 1);
                }
                let start = (direction, points[0].0);
                entries.push(start);
                arcs.insert(start, (points[0].1, points[1].0, points[1].1, cuboid.owner));
            }
        }
        let mut loops = DualLoops::with_key();
        let mut boundaries = HashMap::new();
        for start in entries {
            if !arcs.contains_key(&start) {
                continue;
            }
            let mut current = start;
            let mut edges = Vec::new();
            let mut offsets = Vec::new();
            let mut boundary = None;
            loop {
                let (entry_offset, exit, exit_offset, owner) = arcs
                    .remove(&current)
                    .ok_or("Template trace does not close")?;
                edges.extend([current.1, exit]);
                offsets.extend([entry_offset, exit_offset]);
                if let CuboidOwner::Edge(edge) = owner {
                    if realized.labeled.labels[edge].direction == start.0 {
                        if boundary.is_some_and(|old| old != edge) {
                            return Err("Ambiguous boundary trace".into());
                        }
                        boundary = Some(edge);
                    }
                }
                current = (start.0, mesh.twin(exit));
                if current == start {
                    break;
                }
            }
            let id = loops.insert(Loop {
                edges,
                offsets,
                direction: start.0,
            });
            if let Some(edge) = boundary {
                boundaries.insert(edge, id);
            }
        }
        if boundaries.len() != realized.labeled.skeleton.edges.len() {
            return Err("Missing template boundaries".into());
        }
        dualcube_dual::Dual::from(std::sync::Arc::new(mesh.clone()), &loops)?;
        Ok(Self {
            mesh,
            loops,
            boundaries,
        })
    }
}
