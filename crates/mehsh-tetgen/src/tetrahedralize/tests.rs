use super::*;
use crate::TetNode;
define_tag!(TEST);

fn fixture() -> (Mesh<TEST>, Vec<VertKey<TEST>>, TetMeshData) {
    let mesh = Mesh::from(
        &[vec![0, 2, 1], vec![0, 1, 3], vec![0, 3, 2], vec![1, 2, 3]],
        &[
            Vector3D::new(0., 0., 0.),
            Vector3D::new(1., 0., 0.),
            Vector3D::new(0., 1., 0.),
            Vector3D::new(0., 0., 1.),
        ],
    )
    .unwrap()
    .0;
    let input = mesh.vert_ids();
    let nodes = [2, 0, 3, 1]
        .into_iter()
        .map(|i| TetNode {
            position: mesh.position(input[i]),
            marker: -(i as i32) - 2,
        })
        .collect();
    (
        mesh,
        input,
        TetMeshData {
            nodes,
            tets: vec![[0, 1, 2, 3]],
        },
    )
}

#[test]
fn maps_reordered_output_to_original_vertex_ids() {
    let (mesh, input, data) = fixture();
    let result = map_output(&mesh, input.clone(), data).unwrap();
    for (i, &v) in input.iter().enumerate() {
        assert_eq!(mesh.position(v), result.vertices[i]);
    }
    assert_eq!(result.tets, vec![[2, 0, 3, 1]]);
}

#[test]
fn rejects_missing_duplicate_or_moved_input_vertices() {
    let (mesh, input, mut data) = fixture();
    data.nodes[0].marker = 0;
    assert!(
        map_output(&mesh, input, data)
            .unwrap_err()
            .to_string()
            .contains("omitted")
    );
    let (mesh, input, mut data) = fixture();
    data.nodes[0].marker = data.nodes[1].marker;
    assert!(map_output(&mesh, input, data).is_err());
    let (mesh, input, mut data) = fixture();
    data.nodes[0].position.x += 0.1;
    assert!(
        map_output(&mesh, input, data)
            .unwrap_err()
            .to_string()
            .contains("moved")
    );
}

#[test]
fn rejects_a_changed_volume_boundary() {
    let (mesh, input, mut data) = fixture();
    data.nodes.push(TetNode {
        position: Vector3D::new(0.2, 0.2, 0.2),
        marker: 0,
    });
    data.tets[0][0] = 4;
    assert!(
        map_output(&mesh, input, data)
            .unwrap_err()
            .to_string()
            .contains("triangulation")
    );
}

#[test]
fn missing_executable_reports_an_error() {
    let (mesh, _, _) = fixture();
    let options = TetgenOptions {
        executable: std::env::temp_dir().join("dualcube-nonexistent-tetgen-executable"),
        use_wsl: false,
    };
    let error = tetrahedralize_with_options(&mesh, &options).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::NotFound);
    assert!(error.to_string().contains("Could not launch TetGen"));
}

/// TetGen output for the fixture's tetrahedron with Steiner points. Tets index the corners
/// (origin, x, y, z) and then the Steiner points, and are oriented positively here.
fn with_steiner(steiner: &[Vector3D], tets: &[[usize; 4]]) -> (Mesh<TEST>, TetrahedralMesh<TEST>) {
    let (mesh, input, _) = fixture();
    let corners = [Vector3D::zeros(), Vector3D::x(), Vector3D::y(), Vector3D::z()];
    let node = |i: usize| {
        corners
            .get(i)
            .map_or(i, |&p| input.iter().position(|&v| mesh.position(v) == p).unwrap())
    };
    let mut nodes: Vec<_> = input
        .iter()
        .enumerate()
        .map(|(i, &v)| TetNode {
            position: mesh.position(v),
            marker: -(i as i32) - 2,
        })
        .collect();
    nodes.extend(steiner.iter().map(|&position| TetNode {
        position,
        marker: 0,
    }));
    let tets = tets
        .iter()
        .map(|tet| {
            let [a, b, c, d] = tet.map(node);
            let [p, q, r, s] = [a, b, c, d].map(|i| nodes[i].position);
            if (q - p).cross(&(r - p)).dot(&(s - p)) < 0. {
                [a, b, d, c]
            } else {
                [a, b, c, d]
            }
        })
        .collect();
    let result = map_output(&mesh, input, TetMeshData { nodes, tets }).unwrap();
    (mesh, result)
}

/// The result is a valid tetrahedralization of the fixture with its input boundary.
fn assert_valid(mesh: &Mesh<TEST>, result: &TetrahedralMesh<TEST>) {
    let index: HashMap<_, _> = result
        .input_vertices
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();
    let expected: HashSet<[usize; 3]> = mesh
        .face_ids()
        .into_iter()
        .map(|f| {
            let mut face: Vec<_> = mesh.vertices(f).map(|v| index[&v]).collect();
            face.sort_unstable();
            [face[0], face[1], face[2]]
        })
        .collect();
    let mut count = HashMap::<[usize; 3], usize>::new();
    let mut total = 0.;
    for &t in &result.tets {
        let [a, b, c, d] = t.map(|i| result.vertices[i]);
        let volume = (b - a).cross(&(c - a)).dot(&(d - a));
        assert!(volume > 1e-9, "inverted or flat tet {t:?}: {volume}");
        total += volume;
        for [i, j, k] in [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]] {
            let mut face = [t[i], t[j], t[k]];
            face.sort_unstable();
            *count.entry(face).or_default() += 1;
        }
    }
    let boundary: HashSet<_> = count
        .into_iter()
        .filter_map(|(face, count)| (count == 1).then_some(face))
        .collect();
    assert_eq!(boundary, expected);
    // Positive tets with the input boundary filling the input volume do not overlap.
    assert!((total - 1.).abs() < 1e-9, "volume {total}");
}

#[test]
fn a_steiner_point_on_an_input_edge_is_moved_inside() {
    let (mesh, result) = with_steiner(
        &[Vector3D::new(0.5, 0., 0.)],
        &[[4, 0, 2, 3], [4, 1, 2, 3]],
    );
    assert_valid(&mesh, &result);
    assert_eq!(result.tets.len(), 4);
}

#[test]
fn several_steiner_points_on_one_edge_are_moved_inside() {
    let (mesh, result) = with_steiner(
        &[Vector3D::new(0.5, 0., 0.), Vector3D::new(0.25, 0., 0.), Vector3D::new(0.75, 0., 0.)],
        &[[5, 0, 2, 3], [5, 4, 2, 3], [6, 4, 2, 3], [6, 1, 2, 3]],
    );
    assert_valid(&mesh, &result);
}

#[test]
fn a_steiner_point_inside_an_input_face_is_moved_inside() {
    let (mesh, result) = with_steiner(
        &[Vector3D::new(0.25, 0.25, 0.)],
        &[[4, 0, 1, 3], [4, 1, 2, 3], [4, 2, 0, 3]],
    );
    assert_valid(&mesh, &result);
}

#[test]
fn steiner_points_on_edges_of_one_face_are_moved_inside() {
    // On the edges 0-1 and 0-2 of the face 0-1-2.
    let (mesh, result) = with_steiner(
        &[Vector3D::new(0.5, 0., 0.), Vector3D::new(0., 0.5, 0.)],
        &[[4, 1, 2, 3], [4, 5, 2, 3], [4, 0, 5, 3]],
    );
    assert_valid(&mesh, &result);
}
