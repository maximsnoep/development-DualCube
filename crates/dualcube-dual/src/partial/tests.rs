use super::*;
use crate::SurfaceArc;

#[test]
fn reversing_topology_matches_geometric_rebuild() {
    let mesh = Mesh::<INPUT>::from(
        &[vec![0, 2, 1], vec![0, 1, 3], vec![0, 3, 2], vec![1, 2, 3]],
        &[
            Vector3D::zeros(),
            Vector3D::x(),
            Vector3D::y(),
            Vector3D::z(),
        ],
    )
    .unwrap()
    .0;
    let face = mesh.face_ids()[0];
    let mut loops = SlotMap::with_key();
    for (axis, points) in [
        (Direction::X, vec![[0.15, 0.15], [0.65, 0.15], [0.15, 0.65]]),
        (Direction::Y, vec![[0.10, 0.30], [0.48, 0.30], [0.30, 0.10]]),
        (Direction::Z, vec![[0.25, 0.35], [0.27, 0.35], [0.25, 0.37]]),
    ] {
        loops.insert(Loop::from_surface(
            axis,
            (0..points.len())
                .map(|i| SurfaceArc {
                    face,
                    from: points[i],
                    to: points[(i + 1) % points.len()],
                })
                .collect(),
        ));
        let original = PartialTopology::new(&mesh, &loops).unwrap();
        for id in loops.keys() {
            let mut reversed = original.clone();
            reversed.reverse_loop(id);
            let mut curves = loops.clone();
            curves[id].reverse();
            let rebuilt = PartialTopology::new(&mesh, &curves).unwrap();
            assert!(reversed.equivalent(&rebuilt));
            reversed.reverse_loop(id);
            assert_eq!(reversed.along, original.along);
            assert_eq!(reversed.sides, original.sides);
            assert_eq!(reversed.positive, original.positive);
        }
    }
}
