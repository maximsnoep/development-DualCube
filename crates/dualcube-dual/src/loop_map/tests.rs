use super::*;
use crate::SurfaceArc;
use crate::partial::PartialTopology;

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

fn triangle(mesh: &Mesh<INPUT>, axis: Direction, points: &[[f64; 2]]) -> Loop {
    let face = mesh.face_ids()[0];
    Loop::from_surface(
        axis,
        (0..points.len())
            .map(|i| SurfaceArc {
                face,
                from: points[i],
                to: points[(i + 1) % points.len()],
            })
            .collect(),
    )
}

// A legacy loop around the root of `first`, crossing each edge at `offset`.
fn around_vertex(mesh: &Mesh<INPUT>, first: EdgeID, offset: f64) -> Loop {
    let mut edge = first;
    let mut curve = Loop::new(vec![], Direction::X);
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

fn same(mesh: &Mesh<INPUT>, loops: &SlotMap<LoopID, Loop>) {
    let map = LoopMap::from_loops(mesh, loops).unwrap();
    let exact = PartialTopology::from_map(&map);
    let geometric = PartialTopology::new(mesh, loops).unwrap();
    assert!(exact.equivalent(&geometric));
    assert!(geometric.equivalent(&exact));
    drawn_alike(mesh, &map, loops);
}

// Realized curves have the map's topology.
fn drawn_alike(mesh: &Mesh<INPUT>, map: &LoopMap, loops: &SlotMap<LoopID, Loop>) {
    let mut drawn = loops.clone();
    for (id, arcs) in map.realize(mesh).unwrap() {
        drawn[id] = Loop::from_surface(loops[id].direction, arcs);
    }
    let geometric = PartialTopology::new(mesh, &drawn).unwrap();
    assert!(geometric.equivalent(&PartialTopology::from_map(map)));
}

#[test]
fn map_of_crossing_interior_curves_has_their_topology() {
    let mesh = tetrahedron();
    let mut loops = SlotMap::with_key();
    for (axis, points) in [
        (Direction::X, vec![[0.15, 0.15], [0.65, 0.15], [0.15, 0.65]]),
        (Direction::Y, vec![[0.10, 0.30], [0.48, 0.30], [0.30, 0.10]]),
        (Direction::Z, vec![[0.25, 0.35], [0.27, 0.35], [0.25, 0.37]]),
    ] {
        loops.insert(triangle(&mesh, axis, &points));
        same(&mesh, &loops);
    }
}

#[test]
fn map_of_legacy_chords_has_their_topology() {
    let mesh = tetrahedron();
    let mut loops = SlotMap::with_key();
    loops.insert(around_vertex(&mesh, mesh.edge_ids()[0], 0.3));
    same(&mesh, &loops);
    let other = mesh.edge_ids()[5];
    loops.insert(around_vertex(&mesh, other, 0.4));
    same(&mesh, &loops);
}

#[test]
fn map_keeps_returning_arcs() {
    let mesh = tetrahedron();
    let mut loops = SlotMap::with_key();
    loops.insert(triangle(
        &mesh,
        Direction::X,
        &[[0.2, 0.], [0.25, 0.3], [0.6, 0.], [0.4, 0.1]],
    ));
    same(&mesh, &loops);
}

#[test]
fn disjoint_crossings_give_the_same_map_topology() {
    let mesh = tetrahedron();
    let mut loops = SlotMap::with_key();
    let curve = around_vertex(&mesh, mesh.edge_ids()[0], 0.3);
    let id = loops.insert(curve.clone());
    let exits: Vec<_> = curve.edges.iter().skip(1).step_by(2).copied().collect();
    let mut order = HashMap::new();
    for (o, &e) in exits.iter().enumerate() {
        let key = if e < mesh.twin(e) { e } else { mesh.twin(e) };
        order.insert(key, vec![o]);
    }
    let mut map = LoopMap::new(&mesh);
    map.add_disjoint(&mesh, &[(id, (0..exits.len()).collect())], &exits, &order)
        .unwrap();
    let exact = PartialTopology::from_map(&map);
    let geometric = PartialTopology::new(&mesh, &loops).unwrap();
    assert!(exact.equivalent(&geometric));
    assert_eq!(exact.regions, 2);
    drawn_alike(&mesh, &map, &loops);
}
