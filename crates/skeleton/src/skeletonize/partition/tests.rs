use super::*;

fn tetrahedron() -> Mesh<INPUT> {
    Mesh::from(
        &[vec![0, 2, 1], vec![0, 1, 3], vec![0, 3, 2], vec![1, 2, 3]],
        &[
            Vector3D::zeros(),
            Vector3D::x(),
            Vector3D::y(),
            Vector3D::z(),
        ],
    )
    .unwrap()
    .0
}

fn two_patches(mesh: &Mesh<INPUT>) -> Partition {
    let vertices = mesh.vert_ids();
    let mut partition = Partition::default();
    let a = partition.nodes.insert(Patch {
        position: Vector3D::x(),
        vertices: vec![vertices[0]],
    });
    let b = partition.nodes.insert(Patch {
        position: Vector3D::y(),
        vertices: vertices[1..].to_vec(),
    });
    partition.edges.insert(PartitionEdge {
        nodes: [a, b],
        boundary: Boundary::default(),
    });
    assign_boundaries(&mut partition, mesh).unwrap();
    partition
}

#[test]
fn conversion_orients_loops_and_keeps_graph_positions() {
    let mesh = tetrahedron();
    let mut partition = two_patches(&mesh);
    for reverse in [false, true] {
        if reverse {
            for edge in partition.edges.values_mut() {
                edge.boundary.crossed_edges.reverse();
            }
        }
        let skeleton = partition.to_skeleton(&mesh).unwrap();
        for ((_, before), (_, after)) in partition.nodes.iter().zip(&skeleton.nodes) {
            assert_eq!(before.position, after.position);
        }
        let edge = skeleton.edges.values().next().unwrap();
        let source = partition.edges.values().next().unwrap();
        let left = &partition.nodes[source.nodes[0]].vertices;
        let right = &partition.nodes[source.nodes[1]].vertices;
        assert_eq!(
            edge.boundary.crossings.len(),
            source.boundary.crossed_edges.len()
        );
        let mut seen = HashSet::new();
        for (i, &id) in edge.boundary.crossings.iter().enumerate() {
            assert!(seen.insert(id));
            let exit = skeleton.surface.crossings[id].exit;
            let previous = edge.boundary.crossings
                [(i + edge.boundary.crossings.len() - 1) % edge.boundary.crossings.len()];
            let entry = mesh.twin(skeleton.surface.crossings[previous].exit);
            assert_eq!(mesh.face(entry), mesh.face(exit));
            assert!(left.contains(&mesh.toor(exit)));
            assert!(right.contains(&mesh.root(exit)));
            assert_eq!(
                skeleton.surface.edge_order[&canonical(&mesh, exit)],
                vec![id]
            );
            // Independently check the left/right convention against oriented surface geometry.
            let midpoint = |e| (mesh.position(mesh.root(e)) + mesh.position(mesh.toor(e))) * 0.5;
            let tangent = midpoint(exit) - midpoint(entry);
            let toward_left = mesh.position(mesh.toor(exit)) - midpoint(exit);
            assert!(
                mesh.normal(mesh.face(exit))
                    .cross(&tangent)
                    .dot(&toward_left)
                    > 0.
            );
        }
        assert_eq!(seen.len(), skeleton.surface.crossings.len());
    }
}

#[test]
fn single_region_conversion_has_no_boundary_or_vertex_ownership() {
    let mesh = tetrahedron();
    let mut partition = Partition::default();
    partition.nodes.insert(Patch {
        position: Vector3D::z(),
        vertices: mesh.vert_ids(),
    });
    let skeleton = partition.to_skeleton(&mesh).unwrap();
    assert_eq!(skeleton.nodes.len(), 1);
    assert!(skeleton.edges.is_empty());
    assert!(skeleton.surface.crossings.is_empty());
    assert!(skeleton.surface.edge_order.is_empty());
}

#[test]
fn invalid_private_provenance_cannot_escape_as_a_skeleton() {
    let mesh = tetrahedron();
    let mut partition = two_patches(&mesh);
    partition
        .nodes
        .values_mut()
        .next()
        .unwrap()
        .vertices
        .clear();
    assert!(partition.to_skeleton(&mesh).is_err());
    let mut partition = two_patches(&mesh);
    partition
        .edges
        .values_mut()
        .next()
        .unwrap()
        .boundary
        .crossed_edges
        .pop();
    assert!(partition.to_skeleton(&mesh).is_err());
    let mut partition = Partition::default();
    for v in mesh.vert_ids() {
        partition.nodes.insert(Patch {
            position: mesh.position(v),
            vertices: vec![v],
        });
    }
    assert!(
        partition
            .to_skeleton(&mesh)
            .unwrap_err()
            .to_string()
            .contains("Triple junction")
    );
}

#[test]
fn diagnostics_keep_only_graph_geometry() {
    let mesh = tetrahedron();
    let partition = two_patches(&mesh);
    let graph = partition.diagnostic();
    assert_eq!(graph.positions.len(), partition.nodes.len());
    assert_eq!(graph.edges.len(), partition.edges.len());
    for (id, node) in &partition.nodes {
        assert_eq!(graph.positions[id], node.position);
    }
}

#[test]
fn conversion_keeps_parallel_boundaries_and_rejects_a_hidden_handle() {
    let (n, m) = (6, 8);
    let mut positions = Vec::new();
    let mut faces = Vec::new();
    for i in 0..n {
        for j in 0..m {
            let u = std::f64::consts::TAU * i as f64 / n as f64;
            let v = std::f64::consts::TAU * j as f64 / m as f64;
            positions.push(Vector3D::new(
                (3. + v.cos()) * u.cos(),
                (3. + v.cos()) * u.sin(),
                v.sin(),
            ));
            let [a, b, c, d] = [
                i * m + j,
                ((i + 1) % n) * m + j,
                ((i + 1) % n) * m + (j + 1) % m,
                i * m + (j + 1) % m,
            ];
            faces.extend([vec![a, b, c], vec![a, c, d]]);
        }
    }
    let (mesh, mapping, _) = Mesh::<INPUT>::from(&faces, &positions).unwrap();
    let mut partition = Partition::default();
    let a = partition.nodes.insert(Patch {
        position: Vector3D::x(),
        vertices: (0..n * m / 2).map(|i| *mapping.key(i).unwrap()).collect(),
    });
    let b = partition.nodes.insert(Patch {
        position: -Vector3D::x(),
        vertices: (n * m / 2..n * m)
            .map(|i| *mapping.key(i).unwrap())
            .collect(),
    });
    for _ in 0..2 {
        partition.edges.insert(PartitionEdge {
            nodes: [a, b],
            boundary: Boundary::default(),
        });
    }
    assign_boundaries(&mut partition, &mesh).unwrap();
    let skeleton = partition.to_skeleton(&mesh).unwrap();
    assert_eq!(skeleton.nodes.len(), 2);
    assert_eq!(skeleton.edges.len(), 2);
    let loops: Vec<_> = skeleton.edges.values().collect();
    assert_eq!(loops[0].nodes, loops[1].nodes);
    assert!(
        loops[0]
            .boundary
            .crossings
            .iter()
            .all(|id| !loops[1].boundary.crossings.contains(id))
    );
    assert_eq!(
        skeleton.surface.crossings.len(),
        loops
            .iter()
            .map(|e| e.boundary.crossings.len())
            .sum::<usize>()
    );

    let mut hidden_handle = Partition::default();
    hidden_handle.nodes.insert(Patch {
        position: Vector3D::zeros(),
        vertices: mesh.vert_ids(),
    });
    assert!(
        hidden_handle
            .to_skeleton(&mesh)
            .unwrap_err()
            .to_string()
            .contains("genus zero")
    );
}
