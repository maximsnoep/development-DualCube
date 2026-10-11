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
