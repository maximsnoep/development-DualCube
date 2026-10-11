use dualcube_types::prelude::*;
use skeleton::{
    Skeleton, SkeletonNode,
    skeletonize::{self, Options, deflate, refinement},
};

fn tetrahedron() -> Mesh<INPUT> {
    Mesh::from(
        &[vec![0, 2, 1], vec![0, 1, 3], vec![0, 3, 2], vec![1, 2, 3]],
        &[
            Vector3D::new(0., 0., 0.),
            Vector3D::new(1., 0., 0.),
            Vector3D::new(0., 1., 0.),
            Vector3D::new(0., 0., 1.),
        ],
    )
    .unwrap()
    .0
}

fn torus() -> Mesh<INPUT> {
    let (n, m) = (16, 8);
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
        }
    }
    for i in 0..n {
        for j in 0..m {
            let (a, b, c, d) = (
                i * m + j,
                ((i + 1) % n) * m + j,
                ((i + 1) % n) * m + (j + 1) % m,
                i * m + (j + 1) % m,
            );
            faces.push(vec![a, b, c]);
            faces.push(vec![a, c, d]);
        }
    }
    Mesh::from(&faces, &positions).unwrap().0
}

#[test]
fn extracts_sphere_topology_and_keeps_input_unchanged() {
    let mesh = tetrahedron();
    let original: Vec<_> = mesh.vert_ids().iter().map(|&v| mesh.position(v)).collect();
    let result = skeletonize::extract(&mesh, Options::default()).unwrap();
    assert_eq!(result.genus, 0);
    assert!(!result.skeleton.nodes.is_empty());
    assert_eq!(result.skeleton.edges.len() + 1, result.skeleton.nodes.len());
    result.raw.validate(&mesh).unwrap();
    result.skeleton.validate(&mesh).unwrap();
    assert_eq!(
        original,
        mesh.vert_ids()
            .iter()
            .map(|&v| mesh.position(v))
            .collect::<Vec<_>>()
    );
    assert_eq!(result.contracted_mesh.nr_faces(), mesh.nr_faces());
    assert!(
        result
            .contracted_mesh
            .vert_ids()
            .iter()
            .any(|&v| result.contracted_mesh.position(v) != mesh.position(VertKey::new(v.raw())))
    );
}

#[test]
fn refinement_can_be_disabled_or_called_independently() {
    let mesh = tetrahedron();
    let result = skeletonize::extract(
        &mesh,
        Options {
            max_iterations: 3,
            refine_embedding: false,
            ..Options::default()
        },
    )
    .unwrap();
    for (id, node) in &result.raw.nodes {
        assert_eq!(node.position, result.skeleton.nodes[id].position);
    }
    let mut refined = result.raw.clone();
    refinement::run(&mesh, &mut refined).unwrap();
    refined.validate(&mesh).unwrap();
}

#[test]
fn single_node_sphere_is_valid_and_refinable() {
    let mesh = tetrahedron();
    let mut skeleton = Skeleton::default();
    let node = skeleton.nodes.insert(SkeletonNode {
        position: Vector3D::zeros(),
        patch_vertices: mesh.vert_ids(),
    });
    skeleton.validate(&mesh).unwrap();
    refinement::run(&mesh, &mut skeleton).unwrap();
    assert!(skeleton.nodes[node].position.iter().all(|x| x.is_finite()));
    assert_ne!(skeleton.nodes[node].position, Vector3D::zeros());
}

#[test]
fn rejects_missing_ownership_and_hidden_patch_handle() {
    let mesh = tetrahedron();
    let mut skeleton = Skeleton::default();
    let node = skeleton.nodes.insert(SkeletonNode {
        position: Vector3D::zeros(),
        patch_vertices: mesh.vert_ids(),
    });
    skeleton.nodes[node].patch_vertices.pop();
    assert!(skeleton.validate(&mesh).is_err());
    let mesh = torus();
    skeleton.nodes[node].patch_vertices = mesh.vert_ids();
    assert!(
        skeleton
            .validate(&mesh)
            .unwrap_err()
            .to_string()
            .contains("genus zero")
    );
}

#[test]
fn rejects_invalid_input_before_contraction() {
    assert!(skeletonize::run(&Mesh::default()).is_err());
    let mut mesh = tetrahedron();
    let v = mesh.vert_ids()[0];
    mesh.set_position(v, Vector3D::new(f64::NAN, 0., 0.));
    assert!(deflate::run(&mesh, 1).is_err());
}

#[test]
#[ignore = "requires TetGen; set TETGEN_EXECUTABLE and run with --include-ignored"]
fn torus_extraction_preserves_one_cycle() {
    let mesh = torus();
    let result = skeletonize::extract(&mesh, Options::default()).unwrap();
    assert_eq!(result.genus, 1);
    assert_eq!(result.skeleton.edges.len(), result.skeleton.nodes.len());
    result.skeleton.validate(&mesh).unwrap();
}

#[test]
fn boundary_validation_rejects_missing_crossings_and_accepts_reverse_winding() {
    let mesh = tetrahedron();
    let mut skeleton = skeletonize::run(&mesh).unwrap();
    let edge = skeleton.edges.keys().next().unwrap();
    skeleton.edges[edge].boundary.edge_midpoints.reverse();
    skeleton.validate(&mesh).unwrap();
    skeleton.edges[edge].boundary.edge_midpoints.pop();
    assert!(skeleton.validate(&mesh).is_err());
}

#[test]
fn validation_rejects_triple_junctions() {
    let mesh = tetrahedron();
    let mut skeleton = Skeleton::default();
    for v in mesh.vert_ids() {
        skeleton.nodes.insert(SkeletonNode {
            position: mesh.position(v),
            patch_vertices: vec![v],
        });
    }
    assert!(
        skeleton
            .validate(&mesh)
            .unwrap_err()
            .to_string()
            .contains("Triple junction")
    );
}

#[test]
fn deflation_preserves_vertex_ids_and_surgery_rejects_other_topologies() {
    let mesh = tetrahedron();
    let mut contracted = deflate::run(&mesh, 2).unwrap();
    assert_eq!(
        mesh.vert_ids(),
        contracted
            .vert_ids()
            .iter()
            .map(|v| VertKey::new(v.raw()))
            .collect::<Vec<_>>()
    );
    let e = contracted.edge_ids()[0];
    contracted.edge_root.insert(e, contracted.toor(e));
    assert!(
        skeletonize::surgery::run(&contracted, &mesh)
            .unwrap_err()
            .to_string()
            .contains("topology")
    );
}

#[test]
#[ignore = "requires TetGen; set TETGEN_EXECUTABLE and run with --include-ignored"]
fn two_handle_surface_preserves_two_cycles() {
    // A voxel slab with two through-holes; all exposed quads are triangulated.
    let cells: HashSet<[i32; 3]> = (0..7)
        .flat_map(|x| {
            (0..5).filter_map(move |y| (!(y == 2 && (x == 2 || x == 4))).then_some([x, y, 0]))
        })
        .collect();
    let sides = [
        ([1, 0, 0], [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]]),
        ([-1, 0, 0], [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]]),
        ([0, 1, 0], [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]]),
        ([0, -1, 0], [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]]),
        ([0, 0, 1], [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]),
        ([0, 0, -1], [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]),
    ];
    let mut ids = HashMap::new();
    let mut positions = Vec::new();
    let mut faces = Vec::new();
    let mut ordered: Vec<_> = cells.iter().copied().collect();
    ordered.sort();
    for c in ordered {
        for (normal, offsets) in sides {
            if cells.contains(&[c[0] + normal[0], c[1] + normal[1], c[2] + normal[2]]) {
                continue;
            }
            let quad = offsets.map(|d| {
                let p = [c[0] + d[0], c[1] + d[1], c[2] + d[2]];
                *ids.entry(p).or_insert_with(|| {
                    let i = positions.len();
                    positions.push(Vector3D::new(p[0] as f64, p[1] as f64, p[2] as f64));
                    i
                })
            });
            faces.push(vec![quad[0], quad[1], quad[2]]);
            faces.push(vec![quad[0], quad[2], quad[3]]);
        }
    }
    let mesh = Mesh::from(&faces, &positions).unwrap().0;
    let result = skeletonize::extract(&mesh, Options::default()).unwrap();
    assert_eq!(result.genus, 2);
    assert_eq!(
        result.skeleton.edges.len() + 1,
        result.skeleton.nodes.len() + 2
    );
    result.skeleton.validate(&mesh).unwrap();
}

#[test]
fn rejects_invalid_surgery_weights() {
    for weight in [-1., f64::NAN, f64::INFINITY] {
        let mut options = Options::default();
        options.surgery.shape_weight = weight;
        let error = skeletonize::extract(&tetrahedron(), options).unwrap_err();
        assert!(error.to_string().contains("weights"));
    }
}

#[test]
fn extraction_can_resume_from_a_contracted_mesh() {
    let mesh = tetrahedron();
    let options = Options {
        max_iterations: 3,
        ..Options::default()
    };
    let contracted = deflate::run(&mesh, options.max_iterations).unwrap();
    let resumed = skeletonize::extract_from_contracted(&mesh, contracted.clone(), options).unwrap();
    let fresh = skeletonize::extract(&mesh, options).unwrap();
    assert_eq!(resumed.skeleton.nodes.len(), fresh.skeleton.nodes.len());
    assert_eq!(resumed.skeleton.edges.len(), fresh.skeleton.edges.len());
    for (id, node) in &fresh.skeleton.nodes {
        assert_eq!(node.position, resumed.skeleton.nodes[id].position);
        assert_eq!(
            node.patch_vertices,
            resumed.skeleton.nodes[id].patch_vertices
        );
    }
    let mut invalid = contracted;
    let e = invalid.edge_ids()[0];
    invalid.edge_root.insert(e, invalid.toor(e));
    assert!(skeletonize::extract_from_contracted(&mesh, invalid, options).is_err());
}
