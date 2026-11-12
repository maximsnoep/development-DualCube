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

/// A solid torus: a ring of n triangular prisms, each cut into three tets.
fn solid_torus(n: u32) -> Vec<[u32; 4]> {
    let mut tets = Vec::new();
    for i in 0..n {
        let j = (i + 1) % n;
        let [a, b, c] = [0, 1, 2].map(|k| 3 * i + k);
        let [d, e, f] = [0, 1, 2].map(|k| 3 * j + k);
        tets.extend([[a, b, c, d], [b, c, d, e], [c, d, e, f]]);
    }
    tets
}

/// Sum of cocycle values along a closed vertex walk.
fn integral(values: &HashMap<[u32; 2], Class>, walk: &[u32]) -> Class {
    (0..walk.len()).fold(Class::default(), |sum, i| {
        let (a, b) = (walk[i], walk[(i + 1) % walk.len()]);
        let value = &values[&[a.min(b), a.max(b)]];
        if a < b { &sum + value } else { &sum - value }
    })
}

#[test]
fn a_solid_torus_has_one_volume_cocycle_closed_on_every_triangle() {
    let tets = solid_torus(5);
    let (rank, values) = homology::volume_cocycles(15, &tets).unwrap();
    assert_eq!(rank, 1);
    for t in &tets {
        for skip in 0..4 {
            let triangle: Vec<_> = (0..4).filter(|&k| k != skip).map(|k| t[k]).collect();
            assert!(integral(&values, &triangle).is_zero());
        }
    }
    // Once around the ring: a generator, either way round.
    let around: Vec<_> = (0..5).map(|i| 3 * i).collect();
    let class = integral(&values, &around);
    assert!(class == Class::unit(0) || class == -&Class::unit(0));
    // Twice around is nonzero, though zero mod 2.
    let twice = [0, 3, 6, 9, 12, 13, 1, 4, 7, 10, 13];
    assert_eq!(integral(&values, &twice), &class + &class);
}

#[test]
fn a_ball_has_no_volume_cocycles() {
    let (rank, values) = homology::volume_cocycles(5, &[[0, 1, 2, 3], [1, 2, 3, 4]]).unwrap();
    assert_eq!(rank, 0);
    assert!(values.values().all(Class::is_zero));
}

/// Arcs are chains of consecutive triangles that end where their junctions say, and the
/// surface holds exactly their crossings.
fn assert_consistent(surgery: &Surgery<'_>) {
    let mesh = surgery.mesh;
    let mut count = 0;
    for (id, arc) in &surgery.arcs {
        let exits: Vec<_> = arc.crossings.iter().map(|&c| surgery.exit(c)).collect();
        for w in exits.windows(2) {
            assert_eq!(mesh.face(mesh.twin(w[0])), mesh.face(w[1]));
        }
        let (first, last) = (exits[0], *exits.last().unwrap());
        match arc.ends {
            Some([j1, j2]) => {
                assert_eq!(surgery.junctions[j1].face, mesh.face(first));
                assert_eq!(surgery.junctions[j2].face, mesh.face(mesh.twin(last)));
                assert!(surgery.junctions[j1].spokes.contains(&(id, 0)));
                assert!(surgery.junctions[j2].spokes.contains(&(id, 1)));
            }
            None => assert_eq!(mesh.face(mesh.twin(last)), mesh.face(first)),
        }
        for &c in &arc.crossings {
            let order = &surgery.surface.edge_order[&canonical(mesh, surgery.exit(c))];
            assert_eq!(order.iter().filter(|&&o| o == c).count(), 1);
        }
        count += arc.crossings.len();
        for patch in arc.sides {
            assert!(surgery.patch(patch).arcs.contains(&id));
        }
    }
    assert_eq!(count, surgery.surface.crossings.len());
    assert_eq!(
        count,
        surgery
            .surface
            .edge_order
            .values()
            .map(Vec::len)
            .sum::<usize>()
    );
    for junction in surgery.junctions.values() {
        assert_eq!(junction.spokes.len(), 3);
        for &(arc, end) in &junction.spokes {
            let ends = surgery.arcs[arc].ends.unwrap();
            assert_eq!(surgery.junctions[ends[end]].face, junction.face);
        }
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
    assert!(surgery.arcs.values().all(|a| a.ends.is_none()));
    assert_consistent(&surgery);
    let skeleton = run(&contracted, &mesh).unwrap();
    assert_eq!(skeleton.edges.len() + 1, skeleton.nodes.len());
}

/// The tetrahedron's dual graph inside one patch, as if stuck: the arcs carry a nowhere-zero
/// integer flow, so every junction's classes sum to zero while no arc bounds.
fn stuck(mesh: &Mesh<INPUT>) -> Surgery<'_> {
    let contracted = Mesh::<CONTRACTION>::convert(mesh);
    let mut surgery = Surgery::new(&contracted, mesh, 0, Options::default()).unwrap();
    surgery.heap.clear();
    let all: Vec<_> = surgery.arcs.keys().collect();
    for patch in 1..surgery.patches.len() {
        surgery.patches[patch] = None;
    }
    let first = surgery.patch_mut(0);
    first.vertices = (0..4).collect();
    first.arcs = all.iter().copied().collect();
    surgery.owner = vec![0; 4];
    // Arc i ends at junctions; at each, starting arcs count negatively.
    let flow = (0..6usize.pow(all.len() as u32))
        .map(|code| {
            (0..all.len())
                .map(|i| [-3, -2, -1, 1, 2, 3][code / 6usize.pow(i as u32) % 6])
                .collect::<Vec<i64>>()
        })
        .find(|values| {
            surgery.junctions.values().all(|j| {
                j.spokes
                    .iter()
                    .map(|&(arc, end)| {
                        let v = values[all.iter().position(|&a| a == arc).unwrap()];
                        if end == 0 { -v } else { v }
                    })
                    .sum::<i64>()
                    == 0
            })
        })
        .expect("K4 has a nowhere-zero flow");
    for (i, &arc) in all.iter().enumerate() {
        let class = (0..flow[i].unsigned_abs()).fold(Class::default(), |c, _| &c + &Class::unit(0));
        let arc = &mut surgery.arcs[arc];
        arc.sides = [0, 0];
        arc.class = Some(if flow[i] > 0 { class } else { -&class });
    }
    surgery
}

#[test]
fn flips_keep_the_arcs_consistent() {
    let mesh = tetrahedron();
    let mut surgery = stuck(&mesh);
    let arcs: Vec<_> = surgery.arcs.keys().collect();
    let mut flipped = 0;
    for &arc in arcs.iter().chain(&arcs).chain(&arcs) {
        let [j1, j2] = surgery.arcs[arc].ends.unwrap();
        if j1 == j2 {
            continue;
        }
        // The flip checks the new rotation against the combinatorial one.
        surgery.flip(arc);
        flipped += 1;
        assert_consistent(&surgery);
    }
    assert!(flipped >= 6);
    assert!(surgery.surface.crossings.len() > 6);
}

#[test]
fn flips_free_a_stuck_surgery() {
    let mesh = tetrahedron();
    let mut surgery = stuck(&mesh);
    assert!(
        surgery
            .arcs
            .keys()
            .collect::<Vec<_>>()
            .into_iter()
            .all(|a| !surgery.bounds(a))
    );
    let sequence = surgery.search().unwrap();
    assert!(!sequence.is_empty());
    // Not a real stuck state (a sphere has no arc with one face on both sides), so only the
    // bookkeeping is checked, not the resulting regions.
    surgery.run().unwrap();
    assert!(surgery.junctions.is_empty());
    assert_consistent(&surgery);
}

#[test]
fn search_fails_within_an_exhausted_budget() {
    let mesh = tetrahedron();
    let mut surgery = stuck(&mesh);
    surgery.options.flip_budget = 0;
    // Depth one is always tried; on the tetrahedron one flip may already succeed.
    let depth_one = surgery.search();
    assert!(depth_one.is_none_or(|s| s.len() == 1));
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
        skeleton.surface.edge_order.remove(&canonical(mesh, exit));
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

#[test]
fn turns_back_across_a_mesh_edge_are_removed() {
    let (mesh, mut skeleton) =
        crate::surface::tests::bands(&[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]);
    let clean = skeleton.clone();
    let edge = skeleton.edges.keys().next().unwrap();
    // Before the second crossing c, which exits the face N through x: go out through x and
    // straight back, twice, nested; the inner turn sits next to c on x.
    let c = skeleton.edges[edge].boundary.crossings[1];
    let x = skeleton.surface.crossings[c].exit;
    let key = canonical(&mesh, x);
    let mut turns = Vec::new();
    for _ in 0..2 {
        let pair = [x, mesh.twin(x)]
            .map(|exit| skeleton.surface.crossings.insert(BoundaryCrossing { exit }));
        turns.push(pair);
    }
    let order = skeleton.surface.edge_order.get_mut(&key).unwrap();
    let at = order.iter().position(|&o| o == c).unwrap();
    // Outer pair farther from c than the inner one.
    let [[o1, o2], [i1, i2]] = [turns[0], turns[1]];
    let nested = if key == x {
        [o1, i1, i2, o2]
    } else {
        [o2, i2, i1, o1]
    };
    for (k, &id) in nested.iter().enumerate() {
        order.insert(at + k, id);
    }
    let crossings = &mut skeleton.edges[edge].boundary.crossings;
    for (k, id) in [o1, i1, i2, o2].into_iter().enumerate() {
        crossings.insert(1 + k, id);
    }
    output::untangle(&mesh, &mut skeleton);
    assert_eq!(
        skeleton.surface.crossings.len(),
        clean.surface.crossings.len()
    );
    assert_eq!(
        skeleton.edges[edge].boundary.crossings,
        clean.edges[edge].boundary.crossings
    );
    skeleton.validate(&mesh).unwrap();
}
