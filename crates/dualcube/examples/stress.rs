//! Stress test: initialize, anneal, and reconstruct solutions for a list of meshes; report failures and panics.
//!
//! `cargo run -p dualcube --example stress --release -- <mesh>...`

use dualcube::prelude::*;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::time::Instant;

fn run(path: &std::path::Path) -> String {
    let timer = Instant::now();
    let (mesh, _, _) = Mesh::<INPUT>::from_file(path).unwrap();
    let faces = mesh.nr_faces();
    let mut solution = Solution::new(Arc::new(mesh));
    solution.initialize();
    let init_time = timer.elapsed();
    if solution.loops.len() < 3 {
        return format!("faces={faces} INIT FAILED ({init_time:.1?})");
    }
    let init_quality = solution.get_quality();

    let timer = Instant::now();
    let annealed = solution.anneal(
        &AnnealParams {
            steps: 8,
            offspring: 12,
            ..Default::default()
        },
        &OptimizationMonitor::default(),
    );
    let anneal_time = timer.elapsed();
    let Ok(mut annealed) = annealed else {
        return format!(
            "faces={faces} init q={init_quality:.3?} ({init_time:.1?}) ANNEAL FAILED: {:?}",
            annealed.err()
        );
    };

    // Exercise adding and removing loops on the annealed solution.
    let timer = Instant::now();
    let mut added = 0;
    for direction in DIRECTIONS {
        if let Some(edges) = annealed
            .sample_loops(1, direction, OrderedFloat, |(_, s)| s)
            .into_iter()
            .next()
        {
            let mut candidate = annealed.clone();
            candidate.add_loop(Loop::new(edges, direction));
            if candidate.reconstruct_solution(false).is_ok() && candidate.layout.is_some() {
                annealed = candidate;
                added += 1;
            }
        }
    }
    if let Some(loop_id) = annealed.loops.keys().next() {
        let mut candidate = annealed.clone();
        candidate.del_loop(loop_id);
        let _ = candidate.reconstruct_solution(false);
    }
    let edit_time = timer.elapsed();

    format!(
        "faces={faces} init q={:.3} ({init_time:.1?}) -> annealed loops={} q={:.3} ({anneal_time:.1?}) -> +{added} loops q={:.3} ({edit_time:.1?})",
        init_quality.unwrap_or(0.),
        annealed.loops.len(),
        annealed.get_quality().unwrap_or(0.),
        annealed.get_quality().unwrap_or(0.),
    )
}

fn main() {
    for arg in std::env::args().skip(1) {
        let path = std::path::PathBuf::from(&arg);
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let result = catch_unwind(AssertUnwindSafe(|| run(&path)));
        match result {
            Ok(report) => println!("{name}: {report}"),
            Err(panic) => {
                let message = panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                    .unwrap_or_default();
                println!("{name}: PANIC {message}");
            }
        }
    }
}
