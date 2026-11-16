//! Insert a parallel boundary and the annular patch between the two loops.

use super::*;
use crate::{SkeletonEdge, SkeletonNode};

pub(crate) fn collar(
    mesh: &Mesh<INPUT>,
    skeleton: &Skeleton,
    edge: SkeletonEdgeID,
    side: usize,
) -> SkeletonResult<(Skeleton, SkeletonEdgeID)> {
    if side > 1 {
        return Err("Invalid collar side".into());
    }
    let mut split = skeleton.clone();
    let old = &skeleton.edges[edge];
    let [a, b] = old.nodes;
    let middle = split.nodes.insert(SkeletonNode {
        // This position only supplies initial angle costs; it is not a surface centroid.
        position: (skeleton.nodes[a].position + skeleton.nodes[b].position) * 0.5,
    });
    let copy = crate::surface::edit::parallel(mesh, &mut split, &old.boundary, side);
    split.edges[edge].nodes = [a, middle];
    let boundary = if side == 0 {
        std::mem::replace(&mut split.edges[edge].boundary, copy)
    } else {
        copy
    };
    let added = split.edges.insert(SkeletonEdge {
        nodes: [middle, b],
        boundary,
    });
    // TODO: Compare alternative surface routes jointly with path labeling costs.
    split.validate(mesh)?;
    Ok((split, added))
}

/// A new loop through the middle of `node`'s patch, between `edge` and the patch's other
/// boundaries: a minimum cut that is cheap far from every boundary.
pub(crate) struct Middle {
    path: Vec<(crate::surface::Gap, dualcube_types::prelude::EdgeID)>,
    /// Normal of the loop's plane, pointing from `edge` towards the rest of the patch.
    pub normal: Vector3D,
}

/// Surface cells and positions shared by every [`middle`] of one skeleton.
pub(crate) struct Ground {
    regions: crate::surface::Regions,
    positions: HashMap<crate::BoundaryCrossingID, Vector3D>,
    cells: Vec<crate::optimize::fidelity::CellGeometry>,
}

impl Ground {
    pub fn build(mesh: &Mesh<INPUT>, skeleton: &Skeleton) -> SkeletonResult<Self> {
        let regions = crate::surface::Regions::build(mesh, skeleton)?;
        let positions = crate::optimize::fidelity::crossing_positions(mesh, skeleton);
        let cells = crate::optimize::fidelity::cells(mesh, &regions, &positions);
        Ok(Self {
            regions,
            positions,
            cells,
        })
    }
}


/// Vertex sides in `owner`'s patch by a minimum cut between the crossings pinned `true` and
/// those pinned `false`, where cutting is cheap far from every boundary of the patch.
fn clearance_cut(
    mesh: &Mesh<INPUT>,
    ground: &Ground,
    node: SkeletonNodeID,
    pinned: &HashMap<crate::BoundaryCrossingID, bool>,
) -> Option<(HashMap<dualcube_types::prelude::VertID, usize>, Vec<bool>)> {
    use crate::surface::{Side, Vertex};
    let position = |v: Vertex| match v {
        Vertex::Input(v) => mesh.position(v),
        Vertex::Crossing(c) => ground.positions[&c],
    };
    // Clearance: distance of every patch vertex to the nearest boundary, along intervals.
    let mut links = HashMap::<dualcube_types::prelude::VertID, Vec<(Vertex, f64)>>::new();
    for (cell, _) in ground
        .regions
        .cells
        .iter()
        .zip(&ground.regions.owners)
        .filter(|(_, owner)| **owner == node)
    {
        let n = cell.vertices.len();
        for j in 0..n {
            if !matches!(cell.sides[j], Side::Gap(_)) {
                continue;
            }
            let ends = [cell.vertices[j], cell.vertices[(j + 1) % n]];
            let length = (position(ends[1]) - position(ends[0])).norm();
            for (a, b) in [(ends[0], ends[1]), (ends[1], ends[0])] {
                if let Vertex::Input(a) = a {
                    links.entry(a).or_default().push((b, length));
                }
            }
        }
    }
    let vertices: Vec<_> = links.keys().copied().collect();
    let number: HashMap<_, _> = vertices.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let mut clearance = vec![f64::INFINITY; vertices.len()];
    let mut heap = std::collections::BinaryHeap::new();
    for (i, v) in vertices.iter().enumerate() {
        clearance[i] = links[v]
            .iter()
            .filter(|(other, _)| matches!(other, Vertex::Crossing(_)))
            .map(|&(_, length)| length)
            .fold(f64::INFINITY, f64::min);
        if clearance[i].is_finite() {
            heap.push((std::cmp::Reverse(dualcube_types::prelude::OrderedFloat(clearance[i])), i));
        }
    }
    while let Some((std::cmp::Reverse(distance), i)) = heap.pop() {
        if distance.0 > clearance[i] {
            continue;
        }
        for &(other, length) in &links[&vertices[i]] {
            if let Vertex::Input(other) = other {
                let (j, next) = (number[&other], distance.0 + length);
                if next < clearance[j] {
                    clearance[j] = next;
                    heap.push((std::cmp::Reverse(dualcube_types::prelude::OrderedFloat(next)), j));
                }
            }
        }
    }
    let widest = clearance.iter().copied().filter(|d| d.is_finite()).fold(0., f64::max);
    let clear = |v: Vertex| match v {
        Vertex::Input(v) => number.get(&v).map_or(0., |&i| clearance[i].min(widest)),
        Vertex::Crossing(_) => 0.,
    };
    // Cutting an interval costs its area over its squared clearance.
    let cutting = |ends: [Vertex; 2], area: f64| {
        let distance = (clear(ends[0]) + clear(ends[1])) / 2. + 0.01 * widest;
        area / (distance * distance)
    };
    crate::optimize::redraw::cut_sides(
        mesh,
        &ground.regions,
        &ground.cells,
        &ground.positions,
        node,
        pinned,
        &|_| [0.; 2],
        &cutting,
    )
}

/// `None` when the patch has no other boundary, or no vertices to put the loop between.
pub(crate) fn middle(
    mesh: &Mesh<INPUT>,
    skeleton: &Skeleton,
    ground: &Ground,
    edge: SkeletonEdgeID,
    node: SkeletonNodeID,
) -> Option<Middle> {
    use crate::surface::{Side, Vertex};
    let mut pinned = HashMap::new();
    for (id, data) in skeleton.incident(node) {
        for &c in &data.boundary.crossings {
            pinned.insert(c, id == edge);
        }
    }
    if pinned.values().all(|&near| near) {
        return None;
    }
    let position = |v: Vertex| match v {
        Vertex::Input(v) => mesh.position(v),
        Vertex::Crossing(c) => ground.positions[&c],
    };
    let (index, sides) = clearance_cut(mesh, ground, node, &pinned)?;
    let side = |v: Vertex| match v {
        Vertex::Input(v) => sides[index[&v]],
        Vertex::Crossing(c) => pinned.get(&c).copied().unwrap_or(false),
    };
    let path = crate::optimize::redraw::trace(mesh, &ground.regions, node, &side)?;
    // The loop's plane, with its normal along the steps from the edge's side to the other.
    let points: Vec<Vector3D> = path
        .iter()
        .map(|&(_, exit)| (mesh.position(mesh.root(exit)) + mesh.position(mesh.toor(exit))) / 2.)
        .collect();
    let centre: Vector3D = points.iter().sum::<Vector3D>() / points.len() as f64;
    let mut normal = Vector3D::zeros();
    for i in 0..points.len() {
        normal += (points[i] - centre).cross(&(points[(i + 1) % points.len()] - centre));
    }
    let cut: std::collections::HashSet<_> = path.iter().map(|&(gap, _)| gap).collect();
    let mut across = Vector3D::zeros();
    for (cell, _) in ground
        .regions
        .cells
        .iter()
        .zip(&ground.regions.owners)
        .filter(|(_, owner)| **owner == node)
    {
        let n = cell.vertices.len();
        for j in 0..n {
            if let Side::Gap(gap) = cell.sides[j]
                && cut.contains(&gap)
            {
                let ends = [cell.vertices[j], cell.vertices[(j + 1) % n]];
                let step = position(ends[1]) - position(ends[0]);
                across += if side(ends[0]) { step } else { -step };
            }
        }
    }
    if normal.dot(&across) < 0. {
        normal = -normal;
    }
    Some(Middle { path, normal })
}

/// Split `node`'s patch along the middle loop: a new patch between `edge` and the loop, joined
/// to `node` by a new edge directed like `edge`. `None` as [`middle`], or if the loop is invalid.
pub(crate) fn split(
    mesh: &Mesh<INPUT>,
    skeleton: &Skeleton,
    edge: SkeletonEdgeID,
    node: SkeletonNodeID,
) -> Option<(Skeleton, SkeletonEdgeID)> {
    if !skeleton.edges[edge].nodes.contains(&node) {
        return None;
    }
    let ground = Ground::build(mesh, skeleton).ok()?;
    let middle = middle(mesh, skeleton, &ground, edge, node)?;
    let mut result = skeleton.clone();
    let end = usize::from(skeleton.edges[edge].nodes[1] == node);
    let between = result.nodes.insert(SkeletonNode {
        position: (skeleton.nodes[skeleton.edges[edge].nodes[0]].position
            + skeleton.nodes[skeleton.edges[edge].nodes[1]].position)
            * 0.5,
    });
    result.edges[edge].nodes[end] = between;
    let crossings = crate::optimize::redraw::insert(&mut result, &middle.path);
    let added = result.edges.insert(SkeletonEdge {
        nodes: if end == 1 {
            [between, node]
        } else {
            [node, between]
        },
        boundary: crate::PatchBoundary { crossings },
    });
    if result.validate(mesh).is_err() {
        let mut boundary = result.edges[added].boundary.clone();
        crate::surface::edit::reverse(mesh, &mut result, &mut boundary);
        result.edges[added].boundary = boundary;
        result.validate(mesh).ok()?;
    }
    Some((result, added))
}

/// Move a boundary to the middle of the union of its two patches, between the other boundaries
/// of the one and of the other. `None` if that changes nothing valid.
pub(crate) fn recentre(
    mesh: &Mesh<INPUT>,
    skeleton: &Skeleton,
    edge: SkeletonEdgeID,
) -> Option<Skeleton> {
    use crate::optimize::redraw;
    use crate::surface::Vertex;
    let [a, b] = skeleton.edges[edge].nodes;
    if a == b {
        return None;
    }
    // The union of both patches as one region, with the boundary removed.
    let mut merged = skeleton.clone();
    let old = merged.edges.remove(edge)?;
    redraw::erase(mesh, &mut merged, &old.boundary.crossings);
    let mut pinned = HashMap::new();
    for data in merged.edges.values_mut() {
        let first = data.nodes.contains(&a);
        if first || data.nodes.contains(&b) {
            for &c in &data.boundary.crossings {
                pinned.insert(c, first);
            }
        }
        for n in &mut data.nodes {
            if *n == b {
                *n = a;
            }
        }
    }
    merged.nodes.remove(b);
    if pinned.values().all(|&s| s) || !pinned.values().any(|&s| s) {
        return None;
    }
    let ground = Ground::build(mesh, &merged).ok()?;
    let (index, sides) = clearance_cut(mesh, &ground, a, &pinned)?;
    let side = |v: Vertex| match v {
        Vertex::Input(v) => sides[index[&v]],
        Vertex::Crossing(c) => pinned.get(&c).copied().unwrap_or(true),
    };
    let path = redraw::trace(mesh, &ground.regions, a, &side)?;
    redraw::replace(mesh, skeleton, edge, &path)
}
