use super::*;

/// Sphere topology with isolated disk patches on a subdivided box surface.
fn star(leaves: usize) -> (Mesh<INPUT>, Skeleton) {
    star_with_spacing(leaves, 3)
}

fn star_with_spacing(leaves: usize, spacing: usize) -> (Mesh<INPUT>, Skeleton) {
    let nx = (spacing * leaves + 2) as i32;
    let ny = 8;
    let cells: HashSet<_> = (0..nx)
        .flat_map(|x| (0..ny).map(move |y| [x, y, 0]))
        .collect();
    let sides = [
        ([1, 0, 0], [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]]),
        ([-1, 0, 0], [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]]),
        ([0, 1, 0], [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]]),
        ([0, -1, 0], [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]]),
        ([0, 0, 1], [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]),
        ([0, 0, -1], [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]),
    ];
    let mut points = Vec::new();
    let mut ids = HashMap::new();
    let mut faces = Vec::new();
    for x in 0..nx {
        for y in 0..ny {
            let c = [x, y, 0];
            for (normal, offsets) in sides {
                if cells.contains(&[x + normal[0], y + normal[1], normal[2]]) {
                    continue;
                }
                let q = offsets.map(|offset| {
                    let p = [c[0] + offset[0], c[1] + offset[1], offset[2]];
                    *ids.entry(p).or_insert_with(|| {
                        let id = points.len();
                        points.push(Vector3D::new(p[0] as f64, p[1] as f64, p[2] as f64));
                        id
                    })
                });
                faces.push(vec![q[0], q[1], q[2]]);
                faces.push(vec![q[0], q[2], q[3]]);
            }
        }
    }
    let (mesh, mapping, _) = Mesh::from(&faces, &points).unwrap();
    let mut skeleton = Skeleton::default();
    let leaf_vertices: Vec<_> = (0..leaves)
        .map(|i| *mapping.key(ids[&[(spacing * i) as i32 + 2, 4, 1]]).unwrap())
        .collect();
    let center = skeleton.nodes.insert(SkeletonNode {
        position: Vector3D::new(nx as f64 / 2., 4., 0.5),
        patch_vertices: mesh
            .vert_ids()
            .into_iter()
            .filter(|v| !leaf_vertices.contains(v))
            .collect(),
    });
    for v in leaf_vertices {
        let leaf = skeleton.nodes.insert(SkeletonNode {
            position: mesh.position(v),
            patch_vertices: vec![v],
        });
        skeleton.edges.insert(SkeletonEdge {
            nodes: [center, leaf],
            boundary: PatchBoundary {
                edge_midpoints: Vec::new(),
            },
        });
    }
    crate::validate::assign_boundaries(&mut skeleton, &mesh).unwrap();
    skeleton.validate(&mesh).unwrap();
    (mesh, skeleton)
}

#[test]
fn oversized_junctions_are_split_into_valid_lower_degree_patches() {
    for leaves in [7, 11] {
        let (mesh, skeleton) = star(leaves);
        let result = attempt(&mesh, &skeleton).unwrap();
        assert!(result.failure.is_none(), "{:?}", result.failure);
        assert!(!result.junction_splits.is_empty());
        if leaves == 11 {
            assert!(result.junction_splits.len() >= 2);
        }
        result.completed().unwrap().validate(&mesh).unwrap();
        let mut degree = SecondaryMap::new();
        for n in result.skeleton.nodes.keys() {
            degree.insert(n, 0);
        }
        for edge in result.skeleton.edges.values() {
            for n in edge.nodes {
                degree[n] += 1;
            }
        }
        assert!(degree.values().all(|&d| d <= 6));
        for split in &result.junction_splits {
            assert!(degree[split.original_node] >= 3);
            assert!(degree[split.added_node] >= 3);
        }
        assert_eq!(
            result.skeleton.nodes.len(),
            skeleton.nodes.len() + result.junction_splits.len()
        );
        // Existing boundary cycles still separate the same external leaf patches.
        for (e, edge) in &skeleton.edges {
            let old: HashSet<_> = edge
                .boundary
                .edge_midpoints
                .iter()
                .map(|&e| if e < mesh.twin(e) { e } else { mesh.twin(e) })
                .collect();
            let new: HashSet<_> = result.skeleton.edges[e]
                .boundary
                .edge_midpoints
                .iter()
                .map(|&e| if e < mesh.twin(e) { e } else { mesh.twin(e) })
                .collect();
            assert_eq!(old, new);
        }
        skeleton.validate(&mesh).unwrap();
    }
}

#[test]
fn degree_six_already_fits_and_is_not_split() {
    let (mesh, skeleton) = star(6);
    let result = attempt(&mesh, &skeleton).unwrap();
    assert!(result.junction_splits.is_empty());
    assert_eq!(result.skeleton.nodes.len(), skeleton.nodes.len());
    result.completed().unwrap().validate(&mesh).unwrap();
}

#[test]
fn local_score_reserves_opposite_ports_on_the_connecting_edge() {
    let costs = [[0., 1., 10., 10., 10., 10.]; 2];
    assert_eq!(star_cost(&costs, 0), 11.);
    assert_eq!(star_cost(&costs, 2), 1.);
}

#[test]
fn inseparable_boundaries_report_failure_without_breaking_surface_invariants() {
    let (mesh, skeleton) = star_with_spacing(7, 2);
    let result = attempt(&mesh, &skeleton).unwrap();
    assert!(matches!(
        result.failure,
        Some(Failure::JunctionSplitUnavailable { degree: 7, .. })
    ));
    assert!(result.completed().is_none());
    assert!(result.junction_splits.is_empty());
    assert_eq!(result.skeleton.nodes.len(), skeleton.nodes.len());
    result.skeleton.validate(&mesh).unwrap();
}
