use dualcube::skeleton::Skeleton;
use dualcube::skeleton::pipeline::{Progress, Stage};
use dualcube::skeleton::skeletonize::{self, Options};
use io::figure::{PASTEL, region_colors};
use std::path::Path;

#[test]
fn neighbors_differ_in_every_stage_and_colors_are_balanced() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../mehsh/assets/blub001k.obj");
    let mesh = io::import_mesh(&path).unwrap().0;
    let extraction = skeletonize::extract(&mesh, Options::default()).unwrap();
    let mut progress = Progress::from_skeleton(extraction.skeleton.clone());
    progress.run_through(&mesh, Stage::Optimization).unwrap();
    let current = &progress.labeled.as_ref().unwrap().skeleton;
    let optimized = &progress.optimized.as_ref().unwrap().labeled.skeleton;
    assert!(optimized.nodes.len() < current.nodes.len());

    let stages = [optimized, current, &extraction.raw];
    let colors = region_colors(&stages);
    for stage in stages {
        assert!(stage.nodes.keys().all(|id| colors.contains_key(&id)));
        for edge in stage.edges.values() {
            let [a, b] = edge.nodes;
            assert!(a == b || colors[&a] != colors[&b]);
        }
    }
    // The optimized skeleton is colored first, so it alone is balanced.
    let balanced = |stage: &Skeleton| {
        let counts = PASTEL.map(|p| stage.nodes.keys().filter(|id| colors[id] == p).count());
        counts.iter().max().unwrap() - counts.iter().min().unwrap() <= 2
    };
    assert!(balanced(optimized));
}
