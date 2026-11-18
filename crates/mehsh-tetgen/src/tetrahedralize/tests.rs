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
            .contains("refined")
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

/// The fixture's surface and volume with a Steiner point at the middle of the edge from the
/// origin to x, splitting the tetrahedron in two.
fn split_on_edge() -> Option<Surface> {
    let (mesh, input, _) = fixture();
    let at = |p: Vector3D| input.iter().position(|&v| mesh.position(v) == p).unwrap();
    let [o, x, y, z] = [Vector3D::zeros(), Vector3D::x(), Vector3D::y(), Vector3D::z()].map(at);
    let mut vertices: Vec<_> = input.iter().map(|&v| mesh.position(v)).collect();
    vertices.push(Vector3D::new(0.5, 0., 0.));
    let tets = vec![[4, o, y, z], [4, x, z, y]];
    let mut incidence = HashMap::new();
    for t in &tets {
        for [i, j, k] in [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]] {
            let mut face = [t[i], t[j], t[k]];
            face.sort();
            *incidence.entry(face).or_default() += 1;
        }
    }
    boundary_surface(&mesh, &input, &vertices, &tets, &incidence).unwrap()
}

#[test]
fn a_boundary_steiner_point_refines_the_surface() {
    let surface = split_on_edge().expect("refined surface");
    // The two faces beside the split edge become four.
    assert_eq!(surface.positions.len(), 5);
    assert_eq!(surface.faces.len(), 6);
    assert_eq!(surface.positions[4], Vector3D::new(0.5, 0., 0.));
    Mesh::<TEST>::from(
        &surface.faces.iter().map(|f| f.to_vec()).collect::<Vec<_>>(),
        &surface.positions,
    )
    .expect("closed manifold surface");
    // The two input faces away from the split edge are kept, with their orientation.
    let (mesh, input, _) = fixture();
    let index: HashMap<_, _> = input.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let kept: Vec<_> = mesh
        .face_ids()
        .into_iter()
        .map(|f| rotated(mesh.vertices(f).map(|v| index[&v]).collect_array::<3>().unwrap()))
        .filter(|f| surface.faces.iter().any(|&g| rotated(g) == *f))
        .collect();
    assert_eq!(kept.len(), 2);
}

/// A tetrahedron surface with the given corners, vertices listed in the order `perm`.
fn tetrahedron(corners: [Vector3D; 4], perm: [usize; 4]) -> Mesh<TEST> {
    let at = |i: usize| perm.iter().position(|&p| p == i).unwrap();
    let faces = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
        .map(|f| f.map(at).to_vec())
        .to_vec();
    Mesh::from(&faces, &perm.map(|i| corners[i])).unwrap().0
}

#[test]
fn a_stored_volume_is_found_for_its_surface_in_any_vertex_order() {
    let corners = [
        Vector3D::new(10., 0., 0.),
        Vector3D::new(11., 0., 0.),
        Vector3D::new(10., 1., 0.),
        Vector3D::new(10., 0., 1.),
    ];
    let mut vertices = corners.to_vec();
    vertices.push(Vector3D::new(10.1, 0.1, 0.1));
    let tets = vec![[0, 1, 2, 4], [0, 1, 4, 3], [0, 4, 2, 3], [4, 1, 2, 3]];
    cache::store(vertices, tets);
    let mesh = tetrahedron(corners, [2, 0, 3, 1]);
    let found = cache::lookup(&mesh).expect("stored volume");
    assert_eq!(found.input_vertices, mesh.vert_ids());
    assert_eq!(found.tets.len(), 4);
    for (i, &v) in mesh.vert_ids().iter().enumerate() {
        assert_eq!(found.vertices[i], mesh.position(v));
    }
}

#[test]
fn a_stored_volume_is_not_found_for_another_surface() {
    let corners = [
        Vector3D::new(20., 0., 0.),
        Vector3D::new(21., 0., 0.),
        Vector3D::new(20., 1., 0.),
        Vector3D::new(20., 0., 1.),
    ];
    cache::store(corners.to_vec(), vec![[0, 1, 2, 3]]);
    let mut moved = corners;
    moved[3].z = 2.;
    assert!(cache::lookup(&tetrahedron(moved, [0, 1, 2, 3])).is_none());
}

#[test]
fn conform_with_a_missing_executable_reports_an_error() {
    let corners = [
        Vector3D::new(30., 0., 0.),
        Vector3D::new(31., 0., 0.),
        Vector3D::new(30., 1., 0.),
        Vector3D::new(30., 0., 1.),
    ];
    let options = TetgenOptions {
        executable: "/nonexistent/tetgen".into(),
        use_wsl: false,
    };
    assert!(conform(&tetrahedron(corners, [0, 1, 2, 3]), &options).is_err());
}

#[test]
fn rotation_starts_at_the_smallest_vertex_and_keeps_orientation() {
    assert_eq!(rotated([5, 2, 7]), [2, 7, 5]);
    assert_ne!(rotated([5, 2, 7]), rotated([5, 7, 2]));
}
