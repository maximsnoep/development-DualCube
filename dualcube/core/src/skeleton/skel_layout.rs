use crate::{
    layout::Layout,
    prelude::{INPUT, PrincipalDirection, VertID},
    skeleton::SkeletonData,
};
use log::debug;
use mehsh::prelude::*;
use ordered_float::OrderedFloat;
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
};

#[derive(Clone, Copy, Debug)]
struct IncidentEdge {
    direction: PrincipalDirection,
    length: f64,
    vector: Vector3D,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PartialLayout {
    pub corner_vertices: HashMap<usize, [VertID; 8]>,
    pub corner_paths: HashMap<usize, Vec<Vec<VertID>>>,
}

fn direction_index(direction: PrincipalDirection) -> usize {
    match direction {
        PrincipalDirection::X => 0,
        PrincipalDirection::Y => 1,
        PrincipalDirection::Z => 2,
    }
}

fn index_axis(index: usize) -> Vector3D {
    match index {
        0 => Vector3D::new(1.0, 0.0, 0.0),
        1 => Vector3D::new(0.0, 1.0, 0.0),
        _ => Vector3D::new(0.0, 0.0, 1.0),
    }
}

fn direction_axis(direction: PrincipalDirection) -> Vector3D {
    index_axis(direction_index(direction))
}

fn normalize_or(v: Vector3D, fallback: Vector3D) -> Vector3D {
    if v.norm_squared() <= EPS {
        fallback.normalize()
    } else {
        v.normalize()
    }
}

fn orthonormalize_basis(mut basis: [Vector3D; 3]) -> [Vector3D; 3] {
    basis[0] = normalize_or(basis[0], Vector3D::new(1.0, 0.0, 0.0));

    let v1 = basis[1] - basis[0] * basis[1].dot(&basis[0]);
    basis[1] = if v1.norm_squared() <= EPS {
        if basis[0].x.abs() < 0.9 {
            Vector3D::new(1.0, 0.0, 0.0)
        } else {
            Vector3D::new(0.0, 1.0, 0.0)
        }
    } else {
        v1.normalize()
    };

    let v2 = basis[0].cross(&basis[1]);
    if v2.norm_squared() <= EPS {
        basis[2] = Vector3D::new(0.0, 0.0, 1.0);
    } else {
        basis[2] = v2.normalize();
    }

    basis
}

fn octant_direction(octant: usize, basis: &[Vector3D; 3]) -> Vector3D {
    let signs = [
        if (octant & 0b001) != 0 { 1.0 } else { -1.0 },
        if (octant & 0b010) != 0 { 1.0 } else { -1.0 },
        if (octant & 0b100) != 0 { 1.0 } else { -1.0 },
    ];

    normalize_or(
        basis[0] * signs[0] + basis[1] * signs[1] + basis[2] * signs[2],
        Vector3D::new(1.0, 0.0, 0.0),
    )
}

fn canonical_patch_edge(a: VertID, b: VertID) -> (VertID, VertID) {
    if a.raw() <= b.raw() {
        (a, b)
    } else {
        (b, a)
    }
}

fn cube_corner_pairs() -> Vec<(usize, usize)> {
    let mut pairs = Vec::with_capacity(12);
    for corner in 0..8 {
        for bit in [1usize, 2, 4] {
            let other = corner ^ bit;
            if corner < other {
                pairs.push((corner, other));
            }
        }
    }
    pairs
}

fn restricted_patch_dijkstra(
    mesh: &Mesh<INPUT>,
    start: VertID,
    end: VertID,
    patch_vertices: &HashSet<VertID>,
    blocked_edges: &HashSet<(VertID, VertID)>,
) -> Option<Vec<VertID>> {
    if !patch_vertices.contains(&start) || !patch_vertices.contains(&end) {
        return None;
    }

    let mut dist = patch_vertices
        .iter()
        .copied()
        .map(|v| (v, f64::INFINITY))
        .collect::<HashMap<_, _>>();
    let mut prev = HashMap::<VertID, VertID>::new();
    let mut unvisited = patch_vertices.clone();

    dist.insert(start, 0.0);

    while !unvisited.is_empty() {
        let current = *unvisited.iter().min_by_key(|&&v| {
            let d = dist.get(&v).copied().unwrap_or(f64::INFINITY);
            OrderedFloat(d)
        })?;

        let current_dist = dist.get(&current).copied().unwrap_or(f64::INFINITY);
        if !current_dist.is_finite() {
            break;
        }

        unvisited.remove(&current);
        if current == end {
            break;
        }

        for neighbor in mesh.neighbors(current) {
            if !patch_vertices.contains(&neighbor) || !unvisited.contains(&neighbor) {
                continue;
            }

            let edge = canonical_patch_edge(current, neighbor);
            if blocked_edges.contains(&edge) {
                continue;
            }

            let step_cost = mesh
                .position(current)
                .metric_distance(&mesh.position(neighbor));
            let next_dist = current_dist + step_cost;
            let old_dist = dist.get(&neighbor).copied().unwrap_or(f64::INFINITY);
            if next_dist + f64::EPSILON < old_dist {
                dist.insert(neighbor, next_dist);
                prev.insert(neighbor, current);
            }
        }
    }

    if !dist.get(&end).copied().is_some_and(f64::is_finite) {
        return None;
    }

    let mut path = vec![end];
    let mut current = end;
    while current != start {
        let parent = *prev.get(&current)?;
        path.push(parent);
        current = parent;
    }

    path.reverse();
    Some(path)
}

pub fn populate_layout_from_skeleton(
    skeleton_data: &SkeletonData,
    mesh: &Mesh<INPUT>,
) -> (Option<Layout>, Option<PartialLayout>) {
    let ortho_skeleton = if skeleton_data.labeled_skeleton().is_some() {
        skeleton_data.labeled_skeleton().unwrap()
    } else {
        return (None, None);
    };

    let mut partial_layout = PartialLayout::default();

    // For each region, project a cube from its position outwards
    for node in ortho_skeleton.node_indices() {
        let node_weight = &ortho_skeleton[node];
        let centroid = node_weight.skeleton_node.position;

        let mut incident_edges = ortho_skeleton
            .edges(node)
            .map(|edge_ref| {
                let source = edge_ref.source();
                let target = edge_ref.target();
                let other = if source == node { target } else { source };

                let mut vector = ortho_skeleton[other].skeleton_node.position - centroid;
                if vector.norm_squared() <= EPS {
                    vector = direction_axis(edge_ref.weight().direction)
                        * f64::from(edge_ref.weight().length.max(1));
                }

                IncidentEdge {
                    direction: edge_ref.weight().direction,
                    length: f64::from(edge_ref.weight().length.max(1)),
                    vector,
                }
            })
            .collect::<Vec<_>>();

        incident_edges.sort_by(|a, b| b.length.partial_cmp(&a.length).unwrap_or(Ordering::Equal));
        let mut basis = [
            Vector3D::new(1.0, 0.0, 0.0),
            Vector3D::new(0.0, 1.0, 0.0),
            Vector3D::new(0.0, 0.0, 1.0),
        ];

        if incident_edges.is_empty() {
            basis = [
                Vector3D::new(1.0, 0.0, 0.0),
                Vector3D::new(0.0, 1.0, 0.0),
                Vector3D::new(0.0, 0.0, 1.0),
            ];
        }


        if incident_edges.len() == 1 {
            let edge = incident_edges[0];
            let axis = direction_index(edge.direction);
            basis[axis] = normalize_or(edge.vector, direction_axis(edge.direction));

            let other_axes = (0..3).filter(|&idx| idx != axis).collect::<Vec<_>>();
            basis[other_axes[0]] = index_axis(other_axes[0]);
            basis[other_axes[1]] = index_axis(other_axes[1]);
        }
        if incident_edges.len() >= 2 {
            let first = incident_edges[0];
            let second = incident_edges
                .iter()
                .copied()
                .skip(1)
                .find(|edge| edge.direction != first.direction)
                .unwrap_or(incident_edges[1]);

            let axis1 = direction_index(first.direction);
            let axis2 = direction_index(second.direction);

            basis[axis1] = normalize_or(first.vector, direction_axis(first.direction));

            if axis2 == axis1 {
                let fallback_axis = (0..3).find(|&idx| idx != axis1).unwrap_or(1);
                basis[fallback_axis] = normalize_or(second.vector, index_axis(fallback_axis));
            } else {
                basis[axis2] = normalize_or(second.vector, direction_axis(second.direction));
            }
        }

        basis = orthonormalize_basis(basis);

        let patch_vertices = node_weight
            .skeleton_node
            .patch_vertices
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        if patch_vertices.is_empty() {
            continue;
        }

        let fallback_corner = patch_vertices
            .iter()
            .copied()
            .min_by_key(|&v| OrderedFloat(mesh.position(v).metric_distance(&centroid)))
            .unwrap();


        let corner_vertices = std::array::from_fn(|octant| {
            let preferred_dir = octant_direction(octant, &basis);
            let mut best_score = f64::NEG_INFINITY;
            let mut best_corner = fallback_corner;

            for candidate in patch_vertices.iter().copied() {
                let candidate_pos = mesh.position(candidate);
                let delta = candidate_pos - centroid;
                let distance = delta.norm();
                if distance <= EPS {
                    continue;
                }

                let candidate_dir = delta / distance;
                let angle = preferred_dir.angle(&candidate_dir);
                let weighted_distance = distance * (2.0 - angle / std::f64::consts::PI);

                if weighted_distance > best_score {
                    best_score = weighted_distance;
                    best_corner = candidate;
                }
            }

            best_corner
        });

        // TODO later: Iteratively improve the 8 points together into more of a cube

        let mut adjacent_corner_pairs = cube_corner_pairs();
        adjacent_corner_pairs.sort_by(|&(a1, b1), &(a2, b2)| {
            let d1 = mesh
                .position(corner_vertices[a1])
                .metric_distance(&mesh.position(corner_vertices[b1]));
            let d2 = mesh
                .position(corner_vertices[a2])
                .metric_distance(&mesh.position(corner_vertices[b2]));
            d1.partial_cmp(&d2).unwrap_or(Ordering::Equal)
        });

        let mut occupied_patch_edges = HashSet::new();
        let mut region_paths = Vec::new();
        for (start_idx, end_idx) in adjacent_corner_pairs {
            let start = corner_vertices[start_idx];
            let end = corner_vertices[end_idx];

            if start == end {
                region_paths.push(vec![start]);
                continue;
            }

            if let Some(path) = restricted_patch_dijkstra(
                mesh,
                start,
                end,
                &patch_vertices,
                &occupied_patch_edges,
            ) {
                for segment in path.windows(2) {
                    occupied_patch_edges.insert(canonical_patch_edge(segment[0], segment[1]));
                }
                region_paths.push(path);
            }
        }

        partial_layout
            .corner_vertices
            .insert(node.index(), corner_vertices);
        partial_layout.corner_paths.insert(node.index(), region_paths);
    }


    // For each edge, find the 4 paths between cube corners. The result will be a tube.
    // TODO later: ...
    

    // Result now has a layout of the corner vertices, and paths.
    debug!(
        "Prepared skeleton partial layout: {} regions, {} region-internal mesh-edge paths.",
        partial_layout.corner_vertices.len(),
        partial_layout
            .corner_paths
            .values()
            .map(std::vec::Vec::len)
            .sum::<usize>()
    );

    let partial_layout = (!partial_layout.corner_vertices.is_empty()).then_some(partial_layout);

    (None, partial_layout)
}
