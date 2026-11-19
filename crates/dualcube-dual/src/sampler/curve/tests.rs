use super::*;
use crate::LoopState;
use crate::partial::PartialTopology;

const CORNERS: [Point2; 3] = [[0., 0.], [1., 0.], [0., 1.]];

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

// A legacy loop around the root of `first`, crossing each edge at `offset` from that root.
fn around_vertex(mesh: &Mesh<INPUT>, first: EdgeID, offset: f64, axis: Direction) -> Loop {
    let mut edge = first;
    let mut curve = Loop::new(vec![], axis);
    loop {
        let exit = mesh.next(mesh.next(edge));
        curve.edges.extend([edge, exit]);
        curve.offsets.extend([offset, 1. - offset]);
        edge = mesh.twin(exit);
        if edge == first {
            break;
        }
    }
    curve
}

fn as_curve(mesh: &Mesh<INPUT>, l: &Loop) -> Loop {
    Loop::from_surface(l.direction, l.surface_arcs(mesh).unwrap())
}

// The gaps of a legacy loop among the given loops (see `construct_valid_loop`).
fn gaps_among(mesh: &Mesh<INPUT>, loops: &SlotMap<LoopID, Loop>, l: &Loop) -> Vec<(usize, usize)> {
    let others: Vec<_> = loops
        .values()
        .flat_map(|o| o.exits(mesh).unwrap())
        .collect();
    (0..l.edges.len())
        .filter(|&i| mesh.twin(l.edges[i]) == l.edges[(i + 1) % l.edges.len()])
        .map(|i| {
            let (e, t) = (l.edges[i], l.offsets[i]);
            let before = others
                .iter()
                .filter(|&&(o, s)| (o == e && s < t) || (o == mesh.twin(e) && 1. - s < t))
                .count();
            (i, before)
        })
        .collect()
}

#[test]
fn among_straight_chords_a_new_curve_is_the_legacy_insertion() {
    let mesh = tetrahedron();
    let starts: Vec<EdgeID> = mesh
        .edge_ids()
        .into_iter()
        .unique_by(|&e| mesh.root(e))
        .take(3)
        .collect();
    let axes = [Direction::X, Direction::Y, Direction::Z];
    // The old path (legacy loops placed by `add_loop_in_gaps`) next to the new one (curves).
    let mut legacy = LoopState::from_loops(&mesh, SlotMap::with_key());
    let first = legacy
        .as_mut(&mesh)
        .add_loop(around_vertex(&mesh, starts[0], 0.6, axes[0]));
    let mut loops: SlotMap<LoopID, Loop> = SlotMap::with_key();
    loops.insert(as_curve(&mesh, &legacy.loops[first]));
    // Each new loop crosses the earlier ones (radius over half the edges).
    for k in 1..3 {
        let route = around_vertex(&mesh, starts[k], 0.6, axes[k]);
        let gaps = gaps_among(&mesh, &loops, &route);
        let occupied = ids::SecMap::new();
        let sampler = LoopSampler::new(&mesh, &loops, &occupied, None);
        let curve = sampler
            .curve_in_gaps(&route.edges, axes[k], gaps.clone())
            .unwrap();
        assert_eq!(
            curve.surface.len(),
            gaps.len(),
            "one straight segment per triangle"
        );
        let id = legacy
            .as_mut(&mesh)
            .add_loop_in_gaps(Loop::new(route.edges, axes[k]), gaps);
        // The same crossings (the curve may start elsewhere along the loop).
        let old = legacy.loops[id].exits(&mesh).unwrap();
        let new = curve.exits(&mesh).unwrap();
        assert_eq!(old.len(), new.len());
        let shift = new.iter().position(|&(e, _)| e == old[0].0).unwrap();
        for (i, &(e, t)) in old.iter().enumerate() {
            let (f, s) = new[(i + shift) % new.len()];
            assert!(e == f && (t - s).abs() < 1e-9, "{e:?} {t} vs {f:?} {s}");
        }
        loops.insert(curve);
        let drawn = PartialTopology::new(&mesh, &loops).unwrap();
        assert!(drawn.equivalent(&PartialTopology::new(&mesh, &legacy.loops).unwrap()));
    }
}

#[test]
fn bends_around_a_curve_that_the_straight_chord_crosses_twice() {
    let mesh = tetrahedron();
    let face = mesh.face_ids()[0];
    let first = mesh.edges(face).next().unwrap();
    let corners: Vec<VertID> = mesh.vertices(face).collect();
    let c = corners.iter().position(|&v| v == mesh.root(first)).unwrap();
    let (d, e) = ((c + 1) % 3, (c + 2) % 3);
    // A loop around corner `c` whose visit of `face` bulges deep toward corner `d`.
    let mut curve = as_curve(&mesh, &around_vertex(&mesh, first, 0.3, Direction::X));
    let i = curve.surface.iter().position(|a| a.face == face).unwrap();
    let arc = curve.surface[i].clone();
    let weight = |w: [f64; 3]| {
        let mut p = [0., 0.];
        for (k, &x) in w.iter().enumerate() {
            p[0] += x * CORNERS[k][0];
            p[1] += x * CORNERS[k][1];
        }
        p
    };
    let mut w = [0.; 3];
    (w[c], w[d], w[e]) = (0.05, 0.9, 0.05);
    let m = weight(w);
    curve.surface.splice(
        i..=i,
        [
            SurfaceArc {
                face,
                from: arc.from,
                to: m,
            },
            SurfaceArc {
                face,
                from: m,
                to: arc.to,
            },
        ],
    );
    let mut loops = SlotMap::with_key();
    loops.insert(curve);
    let cells = general::build(&mesh, &loops).unwrap().general.unwrap();
    let visit = [arc.from, m, arc.to];
    // A chord near corner `d`, on the sides at `d`: not separated by the visit, but crossing it twice.
    let near = |other: usize| {
        let mut w = [0.; 3];
        (w[d], w[other]) = (0.8, 0.2);
        weight(w)
    };
    let (p, q) = (near(c), near(e));
    assert_eq!(crossings(p, q, &visit), Some(2));
    let points = bend(&cells, face, p, q, 0, |_| false).unwrap();
    assert!(points.len() > 2);
    for w in points.windows(2) {
        assert_eq!(crossings(w[0], w[1], &visit), Some(0));
    }
    assert!(
        points
            .iter()
            .all(|p| p[0] > -1e-9 && p[1] > -1e-9 && p[0] + p[1] < 1. + 1e-9)
    );
}
