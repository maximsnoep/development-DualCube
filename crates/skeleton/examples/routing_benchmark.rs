//! Run with a mesh path and optional repetition count; excludes extraction from routing time.
use dualcube_types::prelude::*;
use std::{path::PathBuf, time::Instant};

fn main() {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_max_level(tracing::Level::INFO)
        .init();
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().expect("mesh path"));
    let repetitions: usize = args.next().map_or(1, |s| s.parse().unwrap());
    let mesh = if path
        .extension()
        .is_some_and(|s| s.eq_ignore_ascii_case("stl"))
    {
        Mesh::<INPUT>::from_stl(&path).unwrap().0
    } else {
        Mesh::<INPUT>::from_obj(&path).unwrap().0
    };
    let skeleton = skeleton::skeletonize::run(&mesh).unwrap();
    let mut progress = skeleton::pipeline::Progress::from_skeleton(skeleton);
    progress.optimize(&mesh).unwrap();
    for run in 0..repetitions {
        let start = Instant::now();
        let attempt =
            skeleton::draw_loops::attempt(&mesh, progress.optimized.as_ref().unwrap()).unwrap();
        println!(
            "run={} elapsed={:.3}s loops={} searches={} failure={:?}",
            run + 1,
            start.elapsed().as_secs_f64(),
            attempt.loops.len(),
            attempt.searches,
            attempt.failure
        );
        assert!(attempt.failure.is_none());
    }
}
