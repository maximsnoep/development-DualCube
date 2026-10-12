use super::*;
use crate::{BoundaryCrossing, SkeletonEdge, SkeletonNode};

pub(crate) fn tetrahedron() -> Mesh<INPUT> {
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

/// Disjoint disks crossing the same input edge twice, with no original vertices inside.
pub(crate) fn lenses(count: usize) -> (Mesh<INPUT>, Skeleton) {
    let mesh = tetrahedron();
    let edge = canonical(&mesh, mesh.edge_ids()[0]);
    let mut skeleton = Skeleton::default();
    let hub = skeleton.nodes.insert(SkeletonNode {
        position: Vector3D::repeat(0.25),
    });
    let mut order = Vec::new();
    for i in 0..count {
        let t = (i + 1) as f64 / (count + 1) as f64;
        let leaf = skeleton.nodes.insert(SkeletonNode {
            position: mesh.position(mesh.root(edge)) * (1. - t)
                + mesh.position(mesh.toor(edge)) * t,
        });
        let a = skeleton
            .surface
            .crossings
            .insert(BoundaryCrossing { exit: edge });
        let b = skeleton.surface.crossings.insert(BoundaryCrossing {
            exit: mesh.twin(edge),
        });
        order.extend([a, b]);
        skeleton.edges.insert(SkeletonEdge {
            nodes: [leaf, hub],
            boundary: PatchBoundary {
                crossings: vec![a, b],
            },
        });
    }
    if count > 0 {
        skeleton.surface.edge_order.insert(edge, order);
    }
    (mesh, skeleton)
}

#[test]
fn zero_vertex_regions_and_returning_arcs_are_valid() {
    let (mesh, skeleton) = lenses(7);
    let regions = Regions::build(&mesh, &skeleton).unwrap();
    let hub = skeleton.nodes.keys().next().unwrap();
    assert!(
        regions
            .cells
            .iter()
            .enumerate()
            .filter(|(i, _)| regions.owners[*i] != hub)
            .all(|(_, c)| c.vertices.iter().all(|v| matches!(v, Vertex::Crossing(_))))
    );
    assert_eq!(
        regions.owners.iter().copied().collect::<HashSet<_>>().len(),
        8
    );
    assert!(regions.cells.iter().any(|c| c.vertices.len() == 2));
    let (mesh, single) = lenses(0);
    single.validate(&mesh).unwrap();
}

#[test]
fn rejects_crossing_arcs_wrong_incidence_and_orphan_occurrences() {
    let (mesh, skeleton) = lenses(2);
    let mut crossing = skeleton.clone();
    crossing
        .surface
        .edge_order
        .values_mut()
        .next()
        .unwrap()
        .swap(1, 2);
    assert!(
        crossing
            .validate(&mesh)
            .unwrap_err()
            .to_string()
            .contains("cross")
    );
    let mut swapped = skeleton.clone();
    swapped.edges.values_mut().next().unwrap().nodes.swap(0, 1);
    assert!(swapped.validate(&mesh).is_err());
    let mut orphan = skeleton.clone();
    let c = *orphan.surface.crossings.values().next().unwrap();
    orphan.surface.crossings.insert(c);
    assert!(orphan.validate(&mesh).is_err());
    let mut missing = skeleton.clone();
    let id = missing.surface.crossings.keys().next().unwrap();
    missing.surface.crossings.remove(id);
    assert!(missing.validate(&mesh).is_err());
    let mut repeated = skeleton.clone();
    let order = repeated.surface.edge_order.values_mut().next().unwrap();
    order.push(order[0]);
    assert!(repeated.validate(&mesh).is_err());
    let mut duplicate_loop = skeleton.clone();
    let edge = duplicate_loop.edges.values().next().unwrap().clone();
    duplicate_loop.edges.insert(edge);
    assert!(duplicate_loop.validate(&mesh).is_err());
}

#[test]
fn reversing_winding_requires_swapping_incident_regions() {
    let (mesh, mut skeleton) = lenses(3);
    let edge = skeleton.edges.keys().next().unwrap();
    let mut boundary = skeleton.edges[edge].boundary.clone();
    edit::reverse(&mesh, &mut skeleton, &mut boundary);
    skeleton.edges[edge].boundary = boundary;
    skeleton.edges[edge].nodes.swap(0, 1);
    skeleton.validate(&mesh).unwrap();
}

#[test]
fn detects_disconnected_regions_even_when_graph_counts_match() {
    let (mesh, mut skeleton) = lenses(2);
    let edges: Vec<_> = skeleton.edges.keys().collect();
    let keep = skeleton.edges[edges[0]].nodes[0];
    let remove = skeleton.edges[edges[1]].nodes[0];
    skeleton.edges[edges[1]].nodes[0] = keep;
    skeleton.nodes.remove(remove);
    assert!(skeleton.validate(&mesh).is_err());
}

#[test]
fn separator_reuses_faces_and_edges_without_crossing() {
    let (mesh, skeleton) = lenses(4);
    let regions = Regions::build(&mesh, &skeleton).unwrap();
    let owner = skeleton.nodes.keys().next().unwrap();
    let ids: Vec<_> = skeleton.edges.keys().collect();
    let route = edit::connector(&regions, &skeleton, owner, [ids[0], ids[2]], None).unwrap();
    let (split, added, edge) =
        edit::band(&mesh, &skeleton, &regions, owner, [ids[0], ids[2]], &route).unwrap();
    split.validate(&mesh).unwrap();
    assert_eq!(
        split
            .edges
            .values()
            .filter(|e| e.nodes.contains(&added))
            .count(),
        3
    );
    let path = &split.edges[edge].boundary.crossings;
    let faces: HashSet<_> = path
        .iter()
        .map(|&c| mesh.face(split.surface.crossings[c].exit))
        .collect();
    assert!(path.len() > faces.len());
    assert_eq!(mesh.nr_verts(), 4);
    skeleton.validate(&mesh).unwrap();
}

#[test]
fn separator_routes_across_triangles_and_handles_reversed_boundaries() {
    let (mesh, mut skeleton) = lenses(4);
    let ids: Vec<_> = skeleton.edges.keys().collect();
    let old = *skeleton.surface.edge_order.keys().next().unwrap();
    let opposite = mesh
        .edge_ids()
        .into_iter()
        .find(|&e| {
            let ends = [mesh.root(old), mesh.toor(old)];
            !ends.contains(&mesh.root(e)) && !ends.contains(&mesh.toor(e))
        })
        .unwrap();
    let opposite = canonical(&mesh, opposite);
    let crossings = skeleton.edges[ids[3]].boundary.crossings.clone();
    skeleton
        .surface
        .edge_order
        .get_mut(&old)
        .unwrap()
        .retain(|id| !crossings.contains(id));
    skeleton.surface.crossings[crossings[0]].exit = opposite;
    skeleton.surface.crossings[crossings[1]].exit = mesh.twin(opposite);
    skeleton.surface.edge_order.insert(opposite, crossings);
    let mut boundary = skeleton.edges[ids[0]].boundary.clone();
    edit::reverse(&mesh, &mut skeleton, &mut boundary);
    skeleton.edges[ids[0]].boundary = boundary;
    skeleton.edges[ids[0]].nodes.swap(0, 1);
    let regions = Regions::build(&mesh, &skeleton).unwrap();
    let owner = skeleton.nodes.keys().next().unwrap();
    let route = edit::connector(&regions, &skeleton, owner, [ids[0], ids[3]], None).unwrap();
    assert!(!route.steps.is_empty());
    let (split, _, _) =
        edit::band(&mesh, &skeleton, &regions, owner, [ids[0], ids[3]], &route).unwrap();
    split.validate(&mesh).unwrap();
    skeleton.validate(&mesh).unwrap();
}

/// A cycle of annuli on a torus; no tetrahedralization or vertex ownership is needed.
pub(crate) fn bands(positions: &[[f64; 3]]) -> (Mesh<INPUT>, Skeleton) {
    let (n, m) = (positions.len() * 2, 8);
    let mut points = Vec::new();
    let mut faces = Vec::new();
    for i in 0..n {
        for j in 0..m {
            let u = std::f64::consts::TAU * i as f64 / n as f64;
            let v = std::f64::consts::TAU * j as f64 / m as f64;
            points.push(Vector3D::new(
                (3. + v.cos()) * u.cos(),
                (3. + v.cos()) * u.sin(),
                v.sin(),
            ));
            let [a, b, c, d] = [
                i * m + j,
                ((i + 1) % n) * m + j,
                ((i + 1) % n) * m + (j + 1) % m,
                i * m + (j + 1) % m,
            ];
            faces.extend([vec![a, b, c], vec![a, c, d]]);
        }
    }
    let (mesh, mapping, _) = Mesh::<INPUT>::from(&faces, &points).unwrap();
    let rings: HashMap<_, _> = (0..n * m)
        .map(|i| (*mapping.key(i).unwrap(), i / m))
        .collect();
    let mut skeleton = Skeleton::default();
    let nodes: Vec<_> = positions
        .iter()
        .map(|p| {
            skeleton.nodes.insert(SkeletonNode {
                position: Vector3D::from(*p),
            })
        })
        .collect();
    for i in 0..nodes.len() {
        let exits: Vec<_> = mesh
            .edge_ids()
            .into_iter()
            .filter(|&e| {
                rings[&mesh.toor(e)] == 2 * i && rings[&mesh.root(e)] == (2 * i + n - 1) % n
            })
            .collect();
        let mut current = exits[0];
        let mut crossings = Vec::new();
        loop {
            let id = skeleton
                .surface
                .crossings
                .insert(BoundaryCrossing { exit: current });
            skeleton
                .surface
                .edge_order
                .insert(canonical(&mesh, current), vec![id]);
            crossings.push(id);
            current = *exits
                .iter()
                .find(|&&e| mesh.face(e) == mesh.face(mesh.twin(current)))
                .unwrap();
            if current == exits[0] {
                break;
            }
            assert!(crossings.len() < exits.len());
        }
        skeleton.edges.insert(SkeletonEdge {
            nodes: [nodes[i], nodes[(i + nodes.len() - 1) % nodes.len()]],
            boundary: PatchBoundary { crossings },
        });
    }
    skeleton.validate(&mesh).unwrap();
    (mesh, skeleton)
}

#[test]
fn region_validator_rejects_hidden_genus_and_unrepresented_nodes() {
    let (mesh, _) = bands(&[[0.; 3]; 2]);
    let mut skeleton = Skeleton::default();
    skeleton.nodes.insert(SkeletonNode {
        position: Vector3D::zeros(),
    });
    assert!(
        skeleton
            .validate(&mesh)
            .unwrap_err()
            .to_string()
            .contains("genus zero")
    );
    let (mesh, mut skeleton) = lenses(1);
    skeleton.nodes.insert(SkeletonNode {
        position: Vector3D::zeros(),
    });
    assert!(skeleton.validate(&mesh).is_err());
}

#[test]
fn every_boundary_pair_can_be_connected_across_mixed_orientations() {
    let (mesh, mut skeleton) = lenses(8);
    let carriers: Vec<_> = mesh
        .edge_ids()
        .into_iter()
        .filter(|&e| e == canonical(&mesh, e))
        .collect();
    let edges: Vec<_> = skeleton.edges.keys().collect();
    skeleton.surface.edge_order.clear();
    for (i, &edge) in edges.iter().enumerate() {
        let carrier = carriers[i % carriers.len()];
        let mut boundary = skeleton.edges[edge].boundary.clone();
        let ids = &boundary.crossings;
        skeleton.surface.crossings[ids[0]].exit = carrier;
        skeleton.surface.crossings[ids[1]].exit = mesh.twin(carrier);
        skeleton
            .surface
            .edge_order
            .entry(carrier)
            .or_default()
            .extend(ids);
        if i % 2 == 0 {
            edit::reverse(&mesh, &mut skeleton, &mut boundary);
            skeleton.edges[edge].boundary = boundary;
            skeleton.edges[edge].nodes.swap(0, 1);
        }
    }
    let regions = Regions::build(&mesh, &skeleton).unwrap();
    let owner = skeleton.nodes.keys().next().unwrap();
    for a in 0..edges.len() {
        for b in a + 1..edges.len() {
            let pair = [edges[a], edges[b]];
            let first = edit::connector(&regions, &skeleton, owner, pair, None).unwrap();
            let mut routes = vec![first.clone()];
            if let Some((_, gap)) = first.steps.first() {
                if let Some(other) = edit::connector(&regions, &skeleton, owner, pair, Some(*gap)) {
                    routes.push(other);
                }
            }
            for route in routes {
                let (split, _, _) =
                    edit::band(&mesh, &skeleton, &regions, owner, pair, &route).unwrap();
                split.validate(&mesh).unwrap();
            }
        }
    }
    crate::label::run(&mesh, &skeleton)
        .unwrap()
        .validate(&mesh)
        .unwrap();
}
