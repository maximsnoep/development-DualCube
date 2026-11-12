use super::*;
use crate::surface::tests::tetrahedron;

fn torus() -> Mesh<INPUT> {
    let (n, m) = (12, 6);
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

fn index(mesh: &Mesh<INPUT>) -> HashMap<VertID, usize> {
    mesh.vert_ids()
        .into_iter()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect()
}

fn class(mesh: &Mesh<INPUT>, cocycles: &HashMap<[usize; 2], Class>, walk: &[VertID]) -> Class {
    let index = index(mesh);
    (0..walk.len())
        .map(|i| {
            let [a, b] = [index[&walk[i]], index[&walk[(i + 1) % walk.len()]]];
            cocycles
                .get(&[a.min(b), a.max(b)])
                .cloned()
                .unwrap_or_default()
        })
        .fold(Class::default(), |sum, c| &sum ^ &c)
}

#[test]
fn torus_cocycles_are_closed_and_detect_both_generators() {
    let mesh = torus();
    let index = index(&mesh);
    let cocycles = homology::cocycles(&mesh, &index);
    let used = cocycles.values().fold(Class::default(), |mut all, c| {
        all |= c;
        all
    });
    assert_eq!(used.count_ones(), 2);
    assert_eq!(used, &Class::bit(0) ^ &Class::bit(1));
    for f in mesh.face_ids() {
        let corners: Vec<_> = mesh.edges(f).map(|e| mesh.root(e)).collect();
        assert!(class(&mesh, &cocycles, &corners).is_zero());
    }
    // Grid vertex (i, j) by position, as `Mesh::from` may reorder vertices.
    let grid = |i: usize, j: usize| {
        let u = std::f64::consts::TAU * i as f64 / 12.;
        let v = std::f64::consts::TAU * j as f64 / 6.;
        let p = Vector3D::new((3. + v.cos()) * u.cos(), (3. + v.cos()) * u.sin(), v.sin());
        mesh.vert_ids()
            .into_iter()
            .find(|&w| (mesh.position(w) - p).norm() < 1e-9)
            .unwrap()
    };
    let meridian: Vec<_> = (0..6).map(|j| grid(0, j)).collect();
    let longitude: Vec<_> = (0..12).map(|i| grid(i, 0)).collect();
    let (a, b) = (
        class(&mesh, &cocycles, &meridian),
        class(&mesh, &cocycles, &longitude),
    );
    assert!(!a.is_zero() && !b.is_zero() && a != b);
}

/// Every arc is a chain of consecutive triangles and ends where its junctions say.
fn assert_consistent(surgery: &Surgery<'_>) {
    let mesh = surgery.mesh;
    let mut ends = HashMap::<FaceID, Vec<ArcID>>::new();
    for (id, arc) in &surgery.arcs {
        for w in arc.exits.iter().collect::<Vec<_>>().windows(2) {
            assert_eq!(mesh.face(mesh.twin(*w[0])), mesh.face(*w[1]));
        }
        match surgery.junction_ends(id) {
            Some(faces) => faces.map(|f| ends.entry(f).or_default().push(id)),
            None => {
                let (first, last) = (arc.exits[0], *arc.exits.back().unwrap());
                assert_eq!(mesh.face(mesh.twin(last)), mesh.face(first));
                [(); 2]
            }
        };
        for patch in surgery.sides(id) {
            assert!(surgery.patch(patch).arcs.contains(&id));
        }
    }
    assert_eq!(ends.len(), surgery.junctions.len());
    for (face, mut arcs) in ends {
        let mut listed = surgery.junctions[&face].clone();
        arcs.sort();
        listed.sort();
        assert_eq!(arcs, listed);
        assert!(listed.len() >= 3);
    }
}

#[test]
fn seeds_the_dual_graph() {
    let mesh = tetrahedron();
    let contracted = Mesh::<CONTRACTION>::convert(&mesh);
    let surgery = Surgery::new(&contracted, &mesh, 0, Options::default()).unwrap();
    assert_eq!(surgery.arcs.len(), 6);
    assert_eq!(surgery.junctions.len(), 4);
    assert_eq!(surgery.patches.iter().flatten().count(), 4);
    assert_consistent(&surgery);
}

#[test]
fn surgery_ends_with_closed_loops_between_genus_zero_patches() {
    let mesh = tetrahedron();
    let contracted = Mesh::<CONTRACTION>::convert(&mesh);
    let mut surgery = Surgery::new(&contracted, &mesh, 0, Options::default()).unwrap();
    surgery.run().unwrap();
    assert!(surgery.junctions.is_empty());
    assert!(surgery.arcs.values().all(|a| a.closed));
    assert_consistent(&surgery);
    let skeleton = run(&contracted, &mesh).unwrap();
    assert_eq!(skeleton.edges.len() + 1, skeleton.nodes.len());
}

/// Merge an edge's endpoints and drop its boundary.
fn contract(mesh: &Mesh<INPUT>, skeleton: &mut Skeleton, edge: crate::SkeletonEdgeID) {
    let removed = skeleton.edges.remove(edge).unwrap();
    let [x, y] = removed.nodes;
    for e in skeleton.edges.values_mut() {
        e.nodes = e.nodes.map(|n| if n == y { x } else { n });
    }
    skeleton.nodes.remove(y);
    for &id in &removed.boundary.crossings {
        let exit = skeleton.surface.crossings.remove(id).unwrap().exit;
        skeleton
            .surface
            .edge_order
            .remove(&crate::surface::canonical(mesh, exit));
    }
}

#[test]
fn self_loops_and_parallel_edges_are_subdivided() {
    let (mesh, mut skeleton) =
        crate::surface::tests::bands(&[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]);
    let edges: Vec<_> = skeleton.edges.keys().collect();
    contract(&mesh, &mut skeleton, edges[0]);
    let mut parallel = skeleton.clone();
    output::simplify(&mesh, &mut parallel);
    parallel.validate(&mesh).unwrap();
    assert_eq!((parallel.nodes.len(), parallel.edges.len()), (3, 3));
    contract(&mesh, &mut skeleton, edges[1]);
    let edge = skeleton.edges.values().next().unwrap();
    assert_eq!(edge.nodes[0], edge.nodes[1]);
    output::simplify(&mesh, &mut skeleton);
    skeleton.validate(&mesh).unwrap();
    assert_eq!((skeleton.nodes.len(), skeleton.edges.len()), (3, 3));
}

#[test]
#[ignore = "requires TetGen; set TETGEN_EXECUTABLE and run with --include-ignored"]
fn torus_surgery_keeps_one_loop_beyond_a_tree() {
    let mesh = torus();
    let contracted = super::super::deflate::run(&mesh, 50).unwrap();
    let mut surgery = Surgery::new(&contracted, &mesh, 1, Options::default()).unwrap();
    surgery.run().unwrap();
    assert_consistent(&surgery);
    let patches = surgery.patches.iter().flatten().count();
    assert_eq!(surgery.arcs.len(), patches);
    let skeleton = run(&contracted, &mesh).unwrap();
    assert_eq!(skeleton.edges.len(), skeleton.nodes.len());
}
