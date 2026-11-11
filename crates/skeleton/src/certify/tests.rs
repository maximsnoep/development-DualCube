use super::*;

fn torus() -> (Mesh<INPUT>, Skeleton) {
    crate::surface::tests::bands(&[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
}

/// A stand-in DiskScissors that always or never reports a disk.
fn fake(disk: bool) -> Options {
    let directory = scratch_directory().unwrap();
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

#[test]
fn walks_are_simple_closed_vertex_paths() {
    let (mesh, skeleton) = torus();
    for edge in skeleton.edges.keys() {
        let walks = walks(&mesh, &skeleton, edge);
        assert_eq!(walks.len(), 2, "both sides of a band boundary are simple");
        for walk in walks {
            for i in 0..walk.len() {
                let (a, b) = (walk[i], walk[(i + 1) % walk.len()]);
                assert!(
                    mesh.neighbors(a).any(|v| v == b),
                    "{a:?} and {b:?} not adjacent"
                );
            }
        }
    }
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
    assert_eq!(volume.disk(&options, &ring).unwrap(), Err(Miss::NoDisk));
    let edge = skeleton.edges.keys().next().unwrap();
    let meridian = &walks(&mesh, &skeleton, edge)[0];
    assert_eq!(volume.disk(&options, meridian).unwrap(), Ok(()));
}

#[test]
fn reduction_removes_repeats_and_backtracks_across_the_seam() {
    let mesh = crate::surface::tests::tetrahedron();
    let v = mesh.vert_ids();
    let (a, b, c, d) = (v[0], v[1], v[2], v[3]);
    assert_eq!(reduce(vec![a, b, b, c]), [a, b, c]);
    assert_eq!(reduce(vec![a, b, d, b, c]), [a, b, c]);
    // Spur at the last vertex: ..., a, d | a, b, c.
    assert_eq!(reduce(vec![a, b, c, a, d]), [a, b, c]);
    // Spur at the first vertex: d | a, d, b, c.
    assert_eq!(reduce(vec![a, d, b, c, d]), [d, b, c]);
    assert_eq!(reduce(vec![a, b, a]), [a]);
}
