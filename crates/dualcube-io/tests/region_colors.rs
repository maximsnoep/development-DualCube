use dualcube::skeleton::skeletonize::{self, Options};
use io::figure::{PASTEL, region_colors};
use std::path::Path;

#[test]
fn neighboring_regions_differ_and_colors_are_balanced() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../mehsh/assets/blub001k.obj");
    let mesh = io::import_mesh(&path).unwrap().0;
    let skeleton = skeletonize::extract(&mesh, Options::default())
        .unwrap()
        .skeleton;
    let colors = region_colors(&skeleton);
    assert_eq!(colors.len(), skeleton.nodes.len());
    for edge in skeleton.edges.values() {
        let [a, b] = edge.nodes;
        assert!(a == b || colors[&a] != colors[&b]);
    }
    let counts = PASTEL.map(|p| colors.values().filter(|&&c| c == p).count());
    assert_eq!(counts.iter().sum::<usize>(), skeleton.nodes.len());
    let (min, max) = (counts.iter().min().unwrap(), counts.iter().max().unwrap());
    assert!(max - min <= 2, "{counts:?}");
}
