use crate::{
    layout::Layout,
    prelude::{INPUT, PrincipalDirection, VertID},
    skeleton::SkeletonData,
};
use itertools::Itertools;
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

fn in_patch_degree(mesh: &Mesh<INPUT>, vertex: VertID, patch_vertices: &HashSet<VertID>) -> usize {
    mesh.neighbors(vertex)
        .filter(|neighbor| patch_vertices.contains(neighbor))
        .count()
}

fn high_degree_patch_vertices(
    mesh: &Mesh<INPUT>,
    patch_vertices: &HashSet<VertID>,
    min_degree: usize,
) -> HashSet<VertID> {
    patch_vertices
        .iter()
        .copied()
        .filter(|&v| in_patch_degree(mesh, v, patch_vertices) >= min_degree)
        .collect()
}

fn patch_centroid(mesh: &Mesh<INPUT>, patch_vertices: &HashSet<VertID>) -> Vector3D {
    if patch_vertices.is_empty() {
        return Vector3D::new(0.0, 0.0, 0.0);
    }

    patch_vertices
        .iter()
        .fold(Vector3D::new(0.0, 0.0, 0.0), |acc, &v| {
            acc + mesh.position(v)
        })
        / patch_vertices.len() as f64
}

fn patch_subsurface_area(mesh: &Mesh<INPUT>, patch_vertices: &HashSet<VertID>) -> f64 {
    let mut patch_faces = HashSet::new();

    for &v in patch_vertices {
        for face in mesh.faces(v) {
            if patch_faces.contains(&face) {
                continue;
            }

            let fully_in_patch = mesh.vertices(face).all(|fv| patch_vertices.contains(&fv));
            if fully_in_patch {
                patch_faces.insert(face);
            }
        }
    }

    patch_faces.into_iter().map(|f| mesh.size(f)).sum()
}

fn pca_basis_from_patch(
    mesh: &Mesh<INPUT>,
    patch_vertices: &HashSet<VertID>,
    fallback_basis: [Vector3D; 3],
) -> [Vector3D; 3] {
    if patch_vertices.len() < 3 {
        return fallback_basis;
    }

    let centroid = patch_centroid(mesh, patch_vertices);
    let mut covariance = nalgebra::Matrix3::<f64>::zeros();

    for &v in patch_vertices {
        let d = mesh.position(v) - centroid;
        covariance += d * d.transpose();
    }

    let eigen = covariance.symmetric_eigen();
    let mut order = [0usize, 1usize, 2usize];
    order.sort_by(|a, b| {
        eigen.eigenvalues[*b]
            .partial_cmp(&eigen.eigenvalues[*a])
            .unwrap_or(Ordering::Equal)
    });

    let basis = [
        normalize_or(
            eigen.eigenvectors.column(order[0]).into_owned(),
            fallback_basis[0],
        ),
        normalize_or(
            eigen.eigenvectors.column(order[1]).into_owned(),
            fallback_basis[1],
        ),
        normalize_or(
            eigen.eigenvectors.column(order[2]).into_owned(),
            fallback_basis[2],
        ),
    ];

    orthonormalize_basis(basis)
}

fn canonical_box_vertices() -> [Vector3D; 8] {
    std::array::from_fn(|i| {
        Vector3D::new(
            if (i & 0b001) != 0 { 0.5 } else { -0.5 },
            if (i & 0b010) != 0 { 0.5 } else { -0.5 },
            if (i & 0b100) != 0 { 0.5 } else { -0.5 },
        )
    })
}

fn transform_box_points(
    canonical_box: &[Vector3D; 8],
    axis_lengths: Vector3D,
    rotation: &nalgebra::Matrix3<f64>,
    translation: Vector3D,
) -> [Vector3D; 8] {
    std::array::from_fn(|i| {
        let local = canonical_box[i].component_mul(&axis_lengths);
        (*rotation * local) + translation
    })
}

fn fit_rotation_kabsch(
    source: &[Vector3D; 8],
    target: &[Vector3D; 8],
) -> (nalgebra::Matrix3<f64>, Vector3D, Vector3D) {
    let mean_source = source
        .iter()
        .fold(Vector3D::new(0.0, 0.0, 0.0), |acc, &p| acc + p)
        / source.len() as f64;
    let mean_target = target
        .iter()
        .fold(Vector3D::new(0.0, 0.0, 0.0), |acc, &p| acc + p)
        / target.len() as f64;

    let mut covariance = nalgebra::Matrix3::<f64>::zeros();
    for i in 0..source.len() {
        let x = source[i] - mean_source;
        let y = target[i] - mean_target;
        covariance += x * y.transpose();
    }

    let svd = covariance.svd(true, true);
    let (Some(u), Some(v_t)) = (svd.u, svd.v_t) else {
        return (nalgebra::Matrix3::identity(), mean_source, mean_target);
    };

    let mut rotation = v_t.transpose() * u.transpose();
    if rotation.determinant() < 0.0 {
        let mut sign_fix = nalgebra::Matrix3::<f64>::identity();
        sign_fix[(2, 2)] = -1.0;
        rotation = v_t.transpose() * sign_fix * u.transpose();
    }

    (rotation, mean_source, mean_target)
}

fn fit_box_transform(
    source: &[Vector3D; 8],
    target: &[Vector3D; 8],
    initial_rotation: nalgebra::Matrix3<f64>,
    initial_axis_lengths: Vector3D,
) -> (Vector3D, nalgebra::Matrix3<f64>, Vector3D) {
    let target_center = target
        .iter()
        .fold(Vector3D::new(0.0, 0.0, 0.0), |acc, &p| acc + p)
        / target.len() as f64;
    let target_centered = std::array::from_fn(|i| target[i] - target_center);

    let mut rotation = initial_rotation;
    let mut axis_lengths = Vector3D::new(
        initial_axis_lengths.x.max(1e-6),
        initial_axis_lengths.y.max(1e-6),
        initial_axis_lengths.z.max(1e-6),
    );

    for _ in 0..5 {
        let r0 = rotation.column(0).into_owned();
        let r1 = rotation.column(1).into_owned();
        let r2 = rotation.column(2).into_owned();

        let mut numerators = [0.0_f64; 3];
        let mut denominators = [0.0_f64; 3];
        for i in 0..source.len() {
            let x = source[i];
            let y = target_centered[i];

            numerators[0] += x.x * r0.dot(&y);
            numerators[1] += x.y * r1.dot(&y);
            numerators[2] += x.z * r2.dot(&y);

            denominators[0] += x.x * x.x;
            denominators[1] += x.y * x.y;
            denominators[2] += x.z * x.z;
        }

        axis_lengths = Vector3D::new(
            (numerators[0] / denominators[0]).abs().max(1e-6),
            (numerators[1] / denominators[1]).abs().max(1e-6),
            (numerators[2] / denominators[2]).abs().max(1e-6),
        );

        let scaled_source: [Vector3D; 8] =
            std::array::from_fn(|i| source[i].component_mul(&axis_lengths));
        let (new_rotation, _, _) = fit_rotation_kabsch(&scaled_source, &target_centered);
        rotation = new_rotation;
    }

    let scaled_source: [Vector3D; 8] =
        std::array::from_fn(|i| source[i].component_mul(&axis_lengths));
    let mean_scaled_source = scaled_source
        .iter()
        .fold(Vector3D::new(0.0, 0.0, 0.0), |acc, &p| acc + p)
        / scaled_source.len() as f64;
    let translation = target_center - rotation * mean_scaled_source;

    (axis_lengths, rotation, translation)
}

fn snap_points_to_patch_vertices(
    mesh: &Mesh<INPUT>,
    primary_candidates: &HashSet<VertID>,
    fallback_candidates: &HashSet<VertID>,
    points: &[Vector3D; 8],
) -> [VertID; 8] {
    let primary_list = primary_candidates.iter().copied().collect_vec();
    let mut fallback_list = fallback_candidates.iter().copied().collect_vec();
    if fallback_list.is_empty() {
        fallback_list = primary_list.clone();
    }

    let fallback = fallback_list
        .first()
        .copied()
        .or_else(|| primary_list.first().copied())
        .expect("Expected non-empty candidate list when snapping box corners to patch vertices");
    let mut used = HashSet::new();

    std::array::from_fn(|i| {
        let mut primary_ranked = primary_list
            .iter()
            .copied()
            .map(|v| (v, mesh.position(v).metric_distance(&points[i])))
            .collect_vec();
        primary_ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        if let Some((v, _)) = primary_ranked.iter().find(|(v, _)| !used.contains(v)) {
            used.insert(*v);
            return *v;
        }

        let mut fallback_ranked = fallback_list
            .iter()
            .copied()
            .map(|v| (v, mesh.position(v).metric_distance(&points[i])))
            .collect_vec();
        fallback_ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        if let Some((v, _)) = fallback_ranked.iter().find(|(v, _)| !used.contains(v)) {
            used.insert(*v);
            *v
        } else {
            fallback
        }
    })
}

fn refine_corners_to_box_icp(
    mesh: &Mesh<INPUT>,
    patch_vertices: &HashSet<VertID>,
    preferred_corner_candidates: &HashSet<VertID>,
    centroid: Vector3D,
    pca_basis: [Vector3D; 3],
    initial_corners: [VertID; 8],
) -> [VertID; 8] {
    if patch_vertices.is_empty() {
        return initial_corners;
    }

    let canonical_box = canonical_box_vertices();
    let patch_area = patch_subsurface_area(mesh, patch_vertices);
    let avg_radius = patch_vertices
        .iter()
        .map(|&v| mesh.position(v).metric_distance(&centroid))
        .sum::<f64>()
        / patch_vertices.len() as f64;

    let fallback_length = if patch_area > EPS {
        (patch_area / 6.0).sqrt()
    } else {
        (2.0 * avg_radius / 3_f64.sqrt()).max(1e-3)
    };

    let mut axis_min = [f64::INFINITY; 3];
    let mut axis_max = [f64::NEG_INFINITY; 3];
    for &v in patch_vertices {
        let d = mesh.position(v) - centroid;
        for k in 0..3 {
            let proj = pca_basis[k].dot(&d);
            axis_min[k] = axis_min[k].min(proj);
            axis_max[k] = axis_max[k].max(proj);
        }
    }

    let mut axis_lengths = Vector3D::new(
        (axis_max[0] - axis_min[0]).abs(),
        (axis_max[1] - axis_min[1]).abs(),
        (axis_max[2] - axis_min[2]).abs(),
    );
    for k in 0..3 {
        if !axis_lengths[k].is_finite() || axis_lengths[k] <= EPS {
            axis_lengths[k] = fallback_length.max(1e-3);
        }
    }

    let mut rotation = nalgebra::Matrix3::from_columns(&pca_basis);
    let mut translation = centroid;

    let max_iterations = 20;
    let movement_tolerance = 1e-4;
    for _ in 0..max_iterations {
        let current_template =
            transform_box_points(&canonical_box, axis_lengths, &rotation, translation);

        let correspondences = std::array::from_fn(|i| {
            let target = current_template[i];
            patch_vertices
                .iter()
                .copied()
                .min_by_key(|&v| OrderedFloat(mesh.position(v).metric_distance(&target)))
                .map(|v| mesh.position(v))
                .unwrap_or(target)
        });

        let (new_axis_lengths, new_rotation, new_translation) = fit_box_transform(
            &canonical_box,
            &correspondences,
            rotation,
            axis_lengths,
        );
        let new_template = transform_box_points(
            &canonical_box,
            new_axis_lengths,
            &new_rotation,
            new_translation,
        );

        let max_movement = (0..8)
            .map(|i| (new_template[i] - current_template[i]).norm())
            .fold(0.0_f64, f64::max);

        axis_lengths = new_axis_lengths;
        rotation = new_rotation;
        translation = new_translation;

        if max_movement < movement_tolerance {
            break;
        }
    }

    let refined_template =
        transform_box_points(&canonical_box, axis_lengths, &rotation, translation);
    let refined_vertices = snap_points_to_patch_vertices(
        mesh,
        preferred_corner_candidates,
        patch_vertices,
        &refined_template,
    );

    let unique_count = refined_vertices.iter().copied().collect::<HashSet<_>>().len();
    if unique_count < 4 {
        initial_corners
    } else {
        refined_vertices
    }
}

fn restricted_patch_dijkstra(
    mesh: &Mesh<INPUT>,
    start: VertID,
    end: VertID,
    patch_vertices: &HashSet<VertID>,
    blocked_edges: &HashSet<(VertID, VertID)>,
    blocked_vertices: &HashSet<VertID>,
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
    let mut unvisited = patch_vertices
        .iter()
        .copied()
        .filter(|&v| !blocked_vertices.contains(&v) || v == start || v == end)
        .collect::<HashSet<_>>();

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
            if blocked_vertices.contains(&neighbor) && neighbor != end {
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

        let high_degree_candidates = high_degree_patch_vertices(mesh, &patch_vertices, 3);
        let corner_selection_candidates = if high_degree_candidates.is_empty() {
            &patch_vertices
        } else {
            &high_degree_candidates
        };

        let fallback_corner = corner_selection_candidates
            .iter()
            .copied()
            .min_by_key(|&v| OrderedFloat(mesh.position(v).metric_distance(&centroid)))
            .or_else(|| {
                patch_vertices
                    .iter()
                    .copied()
                    .min_by_key(|&v| OrderedFloat(mesh.position(v).metric_distance(&centroid)))
            })
            .unwrap();

        let selected_corner_vertices = std::array::from_fn(|octant| {
            let preferred_dir = octant_direction(octant, &basis);
            let mut best_score = f64::NEG_INFINITY;
            let mut best_corner = fallback_corner;

            for candidate in corner_selection_candidates.iter().copied() {
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

        let pca_basis = pca_basis_from_patch(mesh, &patch_vertices, basis);
        let corner_vertices = refine_corners_to_box_icp(
            mesh,
            &patch_vertices,
            &high_degree_candidates,
            centroid,
            pca_basis,
            selected_corner_vertices,
        );

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
        let mut occupied_patch_vertices = HashSet::new();
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
                &occupied_patch_vertices,
            ) {
                for segment in path.windows(2) {
                    occupied_patch_edges.insert(canonical_patch_edge(segment[0], segment[1]));
                }

                for &interior in path
                    .iter()
                    .skip(1)
                    .take(path.len().saturating_sub(2))
                {
                    occupied_patch_vertices.insert(interior);
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
