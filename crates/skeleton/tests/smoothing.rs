use dualcube_types::prelude::*;
use skeleton::Skeleton;
use skeleton::skeletonize::{self, Options, smoothing};

fn mesh() -> Mesh<INPUT> {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../mehsh/assets/blub001k.obj");
    Mesh::<INPUT>::from_obj(&path).unwrap().0
}

fn length(mesh: &Mesh<INPUT>, skeleton: &Skeleton) -> f64 {
    let positions = skeleton.surface.positions(mesh);
    let mut total = 0.;
    for edge in skeleton.edges.values() {
        let c = &edge.boundary.crossings;
        for k in 0..c.len() {
            total += (positions[&c[(k + 1) % c.len()]] - positions[&c[k]]).norm();
        }
    }
    total
}

#[test]
fn smoothing_shortens_boundaries_and_keeps_topology() {
    let mesh = mesh();
    let options = Options {
        max_iterations: 10,
        ..Options::default()
    };
    let plain = skeletonize::extract(&mesh, options).unwrap();
    assert!(plain.skeleton.surface.offsets.is_empty());
    let smooth = skeletonize::extract_from_contracted(
        &mesh,
        plain.contracted_mesh.clone(),
        Options {
            smooth_boundaries: true,
            ..options
        },
    )
    .unwrap();
    let (before, after) = (&plain.skeleton, &smooth.skeleton);
    assert_eq!(before.surface.edge_order, after.surface.edge_order);
    assert!(!after.surface.offsets.is_empty());
    after.validate(&mesh).unwrap();
    skeleton::geometry::SurfaceGeometry::build(&mesh, after).unwrap();
    // Offsets increase strictly along every edge, away from its ends.
    for order in after.surface.edge_order.values() {
        let mut last = 0.;
        for id in order {
            let t = after.surface.offsets[id];
            assert!(last < t && t < 1.);
            last = t;
        }
    }
    assert!(length(&mesh, after) < 0.9 * length(&mesh, before));
    // Smoothing again changes little.
    let mut again = after.clone();
    smoothing::run(&mesh, &mut again);
    assert!((length(&mesh, &again) - length(&mesh, after)).abs() < 1e-3 * length(&mesh, after));
}
