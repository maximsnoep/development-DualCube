//! Times path smoothing and quad-mesh construction on a model: `cargo run --release -p dualcube --example quad_bench -- <mesh.obj|stl> [omega] [generations]`
//! (with generations, the solution is evolved first, for a realistic number of loops).
use dualcube::prelude::*;
use std::{path::PathBuf, sync::Arc, time::Instant};

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let args: Vec<String> = std::env::args().collect();
    let path = PathBuf::from(&args[1]);
    let omega = args.get(2).map_or(5, |s| s.parse().unwrap());
    let generations: usize = args.get(3).map_or(0, |s| s.parse().unwrap());
    let mesh = Mesh::from_file(&path).unwrap().0;
    let mut solution = Solution::new(Arc::new(mesh));
    solution.initialize();
    if generations > 0 {
        solution = solution
            .evolve(&EvolutionParams {
                max_generations: generations,
                population: 10,
                offspring: 30,
                ..EvolutionParams::default()
            })
            .unwrap();
    }
    solution.reconstruct_solution(true).unwrap();
    println!("loops={}", solution.loops.len());
    assert!(solution.layout.is_some(), "no layout");
    let t = Instant::now();
    let mut smoothed = solution.clone();
    let cloned = t.elapsed().as_secs_f64();
    smoothed.smooth_layout().unwrap();
    println!(
        "clone: {cloned:.3}s, smooth_layout: {:.3}s (faces {} -> {})",
        t.elapsed().as_secs_f64() - cloned,
        solution.layout.as_ref().unwrap().granulated_mesh.nr_faces(),
        smoothed.layout.as_ref().unwrap().granulated_mesh.nr_faces()
    );
    let t = Instant::now();
    solution.construct_quad(QuadDensity::Fixed(omega)).unwrap();
    let quad = solution.quad.as_ref().expect("quad construction failed");
    println!(
        "construct_quad: {:.3}s, quad faces={}",
        t.elapsed().as_secs_f64(),
        quad.quad_mesh.nr_faces()
    );
}
