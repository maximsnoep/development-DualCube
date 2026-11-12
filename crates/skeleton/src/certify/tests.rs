use super::*;

fn torus() -> (Mesh<INPUT>, Skeleton) {
    crate::surface::tests::bands(&[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
}

/// A stand-in DiskScissors that always or never reports a disk.
#[cfg(unix)]
fn fake(disk: bool) -> Options {
    let directory = std::env::temp_dir().join(format!("dualcube-fake-diskscissors-{disk}"));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("diskscissors");
    let output = if disk { "Snip" } else { "No disk" };
    std::fs::write(&path, format!("#!/bin/sh\necho {output}\n")).unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    Options {
        executable: path,
        ..Options::default()
    }
}

#[test]
fn trees_span_the_graph_and_contain_forced_edges() {
    let (mesh, skeleton) = torus();
    let edges: Vec<_> = skeleton.edges.keys().collect();
    let tree = spanning_tree(&mesh, &skeleton, &[]).unwrap();
    assert_eq!(tree.len(), skeleton.nodes.len() - 1);
    for &forced in &edges {
        let tree = spanning_tree(&mesh, &skeleton, &[forced]).unwrap();
        assert!(tree.contains(&forced));
        assert_eq!(tree.len(), skeleton.nodes.len() - 1);
    }
    // All edges of the cycle cannot be in one tree.
    assert!(spanning_tree(&mesh, &skeleton, &edges).is_none());
}

/// The torus with a parallel copy beside each band boundary, so mesh edges are crossed twice.
fn doubled() -> (Mesh<INPUT>, Skeleton) {
    let (mesh, mut skeleton) = torus();
    let edges: Vec<_> = skeleton.edges.keys().collect();
    for edge in edges {
        let boundary = skeleton.edges[edge].boundary.clone();
        let copy = crate::surface::edit::parallel(&mesh, &mut skeleton, &boundary, 0);
        let nodes = skeleton.edges[edge].nodes;
        skeleton.edges.insert(crate::SkeletonEdge {
            nodes,
            boundary: copy,
        });
    }
    (mesh, skeleton)
}

/// Refine and check: same volume, consistently oriented tets, conforming faces, the same
/// surface area, and every boundary a simple cycle of edges of surface faces.
fn check_refinement(
    mesh: &Mesh<INPUT>,
    skeleton: &Skeleton,
    vertices: &[Vector3D],
    tets: &[[usize; 4]],
) {
    let index: HashMap<_, _> = mesh
        .vert_ids()
        .into_iter()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    let loops: Vec<_> = skeleton.edges.values().map(|e| &e.boundary).collect();
    let refined = refine::refine(mesh, &skeleton.surface, &loops, &index, vertices, tets);
    let volume = |v: &[Vector3D], [a, b, c, d]: [usize; 4]| {
        (v[b] - v[a]).cross(&(v[c] - v[a])).dot(&(v[d] - v[a]))
    };
    let total = |v: &[Vector3D], t: &[[usize; 4]]| t.iter().map(|&t| volume(v, t)).sum::<f64>();
    let before = total(vertices, tets);
    assert!((total(&refined.vertices, &refined.tets) - before).abs() < 1e-9 * before.abs());
    let sign = before.signum();
    assert!(
        refined
            .tets
            .iter()
            .all(|&t| volume(&refined.vertices, t) * sign > 0.)
    );
    let mut faces: HashMap<[usize; 3], usize> = HashMap::new();
    for t in &refined.tets {
        for skip in 0..4 {
            let mut f: Vec<_> = (0..4).filter(|&k| k != skip).map(|k| t[k]).collect();
            f.sort_unstable();
            *faces.entry([f[0], f[1], f[2]]).or_default() += 1;
        }
    }
    assert!(faces.values().all(|&n| n <= 2));
    let area =
        |v: &[Vector3D], [a, b, c]: [usize; 3]| (v[b] - v[a]).cross(&(v[c] - v[a])).norm() / 2.;
    let surface: Vec<_> = faces.iter().filter(|f| *f.1 == 1).map(|f| *f.0).collect();
    let refined_area: f64 = surface.iter().map(|&f| area(&refined.vertices, f)).sum();
    let mesh_area: f64 = mesh
        .face_ids()
        .into_iter()
        .map(|f| {
            let c: Vec<_> = mesh.edges(f).map(|e| index[&mesh.root(e)]).collect();
            area(vertices, [c[0], c[1], c[2]])
        })
        .sum();
    assert!((refined_area - mesh_area).abs() < 1e-9 * mesh_area);
    let surface_edges: HashSet<[usize; 2]> = surface
        .iter()
        .flat_map(|&[a, b, c]| [[a, b], [b, c], [a, c]])
        .collect();
    for edge in skeleton.edges.values() {
        let cycle: Vec<_> = edge
            .boundary
            .crossings
            .iter()
            .map(|c| refined.crossings[c])
            .collect();
        assert_eq!(cycle.iter().collect::<HashSet<_>>().len(), cycle.len());
        for i in 0..cycle.len() {
            let (a, b) = (cycle[i], cycle[(i + 1) % cycle.len()]);
            assert!(surface_edges.contains(&[a.min(b), a.max(b)]));
        }
    }
}

#[test]
fn refining_a_single_tet_makes_boundaries_simple_cycles() {
    let mesh = crate::surface::tests::tetrahedron();
    let skeleton = crate::skeletonize::run(&mesh).unwrap();
    assert!(!skeleton.surface.crossings.is_empty());
    let vertices: Vec<_> = mesh
        .vert_ids()
        .into_iter()
        .map(|v| mesh.position(v))
        .collect();
    check_refinement(&mesh, &skeleton, &vertices, &[[0, 1, 2, 3]]);
}

#[test]
#[ignore = "requires TetGen"]
fn refining_a_torus_volume_makes_boundaries_simple_cycles() {
    for (mesh, skeleton) in [torus(), doubled()] {
        let tets = mehsh_tetgen::tetrahedralize(&mesh).unwrap();
        assert_eq!(tets.input_vertices, mesh.vert_ids());
        check_refinement(&mesh, &skeleton, &tets.vertices, &tets.tets);
    }
}

#[test]
fn convex_regions_are_triangulated_without_flat_triangles() {
    // Sides 0, 1, 2 as bits; a corner lies on two.
    let flat =
        |t: &[usize], masks: &HashMap<usize, u8>| t.iter().fold(0b111, |m, v| m & masks[v]) != 0;
    for polygon in [
        // Three points on side 0 and the opposite corner.
        vec![(0, 0b001), (1, 0b001), (2, 0b001), (3, 0b110)],
        // A corner with points on both its sides.
        vec![(0, 0b101), (1, 0b001), (2, 0b001), (3, 0b100), (4, 0b100)],
    ] {
        let masks: HashMap<_, _> = polygon.iter().copied().collect();
        let mut faces = Vec::new();
        refine::triangulate(polygon.clone(), &mut faces);
        let faces: Vec<Vec<usize>> = faces.into_iter().map(Vec::from).collect();
        assert_eq!(faces.len(), polygon.len() - 2);
        assert!(faces.iter().all(|t| !flat(t, &masks)));
    }
}

#[test]
fn chords_cut_a_polygon_into_regions() {
    // Hexagon with chords 1-4 and 2-3.
    let mut partner = vec![None; 6];
    for (a, b) in [(1, 4), (2, 3)] {
        partner[a] = Some(b);
        partner[b] = Some(a);
    }
    let mut regions = refine::regions(&partner);
    regions.sort();
    assert_eq!(regions, [vec![0, 1, 4, 5], vec![1, 2, 3, 4], vec![2, 3]]);
}

#[test]
fn genus_zero_needs_no_disks() {
    let mesh = crate::surface::tests::tetrahedron();
    let skeleton = crate::skeletonize::run(&mesh).unwrap();
    let options = Options {
        executable: "/nonexistent/diskscissors".into(),
        ..Options::default()
    };
    let certificate = run_with(&mesh, &skeleton, &options).unwrap();
    assert!(certificate.disks.is_empty() && certificate.misses.is_empty());
}

#[test]
#[cfg(unix)]
#[ignore = "requires TetGen"]
fn a_disk_for_the_one_cotree_loop_certifies_a_torus() {
    let (mesh, skeleton) = torus();
    let options = fake(true);
    let certificate = run_with(&mesh, &skeleton, &options).unwrap();
    assert_eq!(certificate.disks.len(), 1);
    assert!(certificate.misses.is_empty());
    assert!(!certificate.tree.contains(&certificate.disks[0]));
}

#[test]
#[cfg(unix)]
#[ignore = "requires TetGen"]
fn misses_move_into_the_tree_until_they_close_a_cycle() {
    let (mesh, skeleton) = torus();
    let options = fake(false);
    let Err(Failure::NoTree { misses }) = run_with(&mesh, &skeleton, &options) else {
        panic!("expected NoTree");
    };
    // Every loop of the cycle was tried once before no tree remained.
    assert_eq!(misses.len(), skeleton.edges.len());
    assert!(misses.iter().all(|m| m.1 == Miss::NoDisk));
}

#[test]
#[ignore = "requires TetGen and DiskScissors"]
fn diskscissors_certifies_torus_meridians() {
    let (mesh, skeleton) = torus();
    let certificate = run(&mesh, &skeleton).unwrap();
    assert_eq!(certificate.disks.len(), 1);
    assert!(certificate.misses.is_empty());
}

#[test]
#[ignore = "requires TetGen and DiskScissors"]
fn diskscissors_finds_no_disk_for_a_longitude() {
    let (mesh, skeleton) = torus();
    // The outer equator (z = 0, radius 4) runs once around the hole.
    let mut ring: Vec<_> = mesh
        .vert_ids()
        .into_iter()
        .filter(|&v| {
            let p = mesh.position(v);
            p.z.abs() < 1e-9 && (p.x.hypot(p.y) - 4.).abs() < 1e-9
        })
        .collect();
    ring.sort_by(|&a, &b| {
        let angle = |v| mesh.position(v).y.atan2(mesh.position(v).x);
        angle(a).total_cmp(&angle(b))
    });
    assert!(ring.len() >= 6);
    let volume = Volume::new(&mesh).unwrap();
    let options = Options::default();
    let timeout = std::time::Duration::from_secs(60);
    let ring: Vec<_> = ring.iter().map(|v| volume.index[v]).collect();
    assert_eq!(
        volume
            .disk(&volume.mesh, &options, timeout, &ring)
            .unwrap(),
        Err(Miss::NoDisk)
    );
    let edge = skeleton.edges.keys().next().unwrap();
    assert_eq!(
        volume
            .certify(&mesh, &skeleton, edge, &options, timeout)
            .unwrap(),
        Ok(())
    );
}

#[test]
fn walks_beside_band_boundaries_are_simple_and_adjacent() {
    let (mesh, skeleton) = torus();
    for edge in skeleton.edges.keys() {
        let walk = walk(&mesh, &skeleton, edge).expect("a band boundary has a simple side");
        for i in 0..walk.len() {
            let (a, b) = (walk[i], walk[(i + 1) % walk.len()]);
            assert!(mesh.neighbors(a).any(|v| v == b));
        }
    }
}

#[test]
fn reduction_removes_repeats_and_backtracks_across_the_seam() {
    let mesh = crate::surface::tests::tetrahedron();
    let v = mesh.vert_ids();
    let (a, b, c, d) = (v[0], v[1], v[2], v[3]);
    assert_eq!(reduce(vec![a, b, b, c]), [a, b, c]);
    assert_eq!(reduce(vec![a, b, d, b, c]), [a, b, c]);
    assert_eq!(reduce(vec![a, b, c, a, d]), [a, b, c]);
    assert_eq!(reduce(vec![a, d, b, c, d]), [d, b, c]);
    assert_eq!(reduce(vec![a, b, a]), [a]);
}
