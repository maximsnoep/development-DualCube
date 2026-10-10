//! Temporary fuzzing harness: the whole pipeline on one model, every step guarded, with invariant checks.
//! `fuzz_pipeline <mesh> <tmpdir>`; prints one line per step: `STEP <name> OK|FAIL|PANIC <ms> <detail>`.
use dualcube::prelude::*;
use std::collections::{HashMap, HashSet};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

static LAST_PANIC: Mutex<String> = Mutex::new(String::new());

fn step<T>(name: &str, f: impl FnOnce() -> Result<T, String>) -> Option<T> {
    let timer = Instant::now();
    let result = catch_unwind(AssertUnwindSafe(f));
    let ms = timer.elapsed().as_millis();
    match result {
        Ok(Ok(value)) => {
            println!("STEP {name} OK {ms}");
            Some(value)
        }
        Ok(Err(detail)) => {
            println!("STEP {name} FAIL {ms} {}", detail.replace('\n', " | "));
            None
        }
        Err(_) => {
            println!("STEP {name} PANIC {ms} {}", LAST_PANIC.lock().unwrap().replace('\n', " | "));
            None
        }
    }
}

fn check(solution: &Solution, what: &str) -> Result<f64, String> {
    if !solution.dual_is_ok() {
        return Err(format!("{what}: invalid dual"));
    }
    let layout = solution.layout.as_ref().ok_or(format!("{what}: no layout"))?;
    if !layout.is_complete() {
        return Err(format!("{what}: incomplete layout"));
    }
    let quality = solution.get_quality().ok_or(format!("{what}: no quality"))?;
    if !quality.is_finite() || !(0.0..=1.0).contains(&quality) {
        return Err(format!("{what}: quality {quality}"));
    }
    let report = solution.quality_report().ok_or(format!("{what}: no report"))?;
    let terms = [report.fidelity, report.path_backtrack, report.path_stretch, report.coherence, report.rectangularity, report.corner_regularity, report.non_degeneracy, report.correspondence];
    if terms.iter().flatten().any(|t| !t.is_finite() || *t < 0.) {
        return Err(format!("{what}: bad term in {report:?}"));
    }
    Ok(quality)
}

fn main() {
    std::panic::set_hook(Box::new(|info| {
        let location = info.location().map(|l| format!("{}:{}", l.file(), l.line())).unwrap_or_default();
        let message = info.payload().downcast_ref::<&str>().map(|s| s.to_string())
            .or_else(|| info.payload().downcast_ref::<String>().cloned()).unwrap_or_default();
        *LAST_PANIC.lock().unwrap() = format!("{message} at {location}");
    }));
    let args: Vec<String> = std::env::args().collect();
    let path = Path::new(&args[1]);
    let tmp = Path::new(&args[2]);
    let Some(mesh) = step("load", || Mesh::<INPUT>::from_file(path).map(|(m, ..)| Arc::new(m)).map_err(|e| format!("{e:?}"))) else { return };
    let (v, e, f) = (mesh.nr_verts() as i64, mesh.nr_edges() as i64 / 2, mesh.nr_faces() as i64);
    let genus = (2 - (v - e + f)) / 2;
    println!("INFO verts {v} faces {f} genus {genus}");

    let Some(mut solution) = step("initialize", || {
        let mut s = Solution::new(mesh.clone());
        s.initialize();
        check(&s, "initialize").map(|q| { println!("INFO initial quality {q:.4} loops {}", s.loops.len()); s })
    }) else { return };

    for (phase, generations) in [(LoopPhase::Initialization, 2), (LoopPhase::Growth, 3), (LoopPhase::Optimization, 2), (LoopPhase::Pruning, 2)] {
        let name = format!("evolve-{}", phase.name());
        if let Some(next) = step(&name, || {
            let params = EvolutionParams { population: 6, offspring: 12, max_generations: generations, phase, ..EvolutionParams::default() };
            let evolved = solution.evolve(&params).map_err(|e| format!("{e}"))?;
            check(&evolved, &name).map(|q| { println!("INFO {name} quality {q:.4} loops {}", evolved.loops.len()); evolved })
        }) { solution = next; }
    }

    if let Some(next) = step("optimize_layout", || {
        let params = LayoutEvolutionParams { max_generations: 4, population: 4, offspring: 8, ..LayoutEvolutionParams::default() };
        let optimized = solution.optimize_layout(&params, &EvolutionMonitor::default()).map_err(|e| format!("{e}"))?;
        check(&optimized, "optimize_layout").map(|_| optimized)
    }) { solution = next; }

    step("paths_consistent", || {
        let layout = solution.layout.as_ref().ok_or("no layout")?;
        let structure = &layout.polycube_ref.structure;
        let mut bad = 0;
        for edge in structure.edge_ids() {
            let (a, b) = (layout.edge_to_path.get(&edge), layout.edge_to_path.get(&structure.twin(edge)));
            match (a, b) {
                (Some(a), Some(b)) if a.iter().rev().eq(b.iter()) => {}
                _ => bad += 1,
            }
        }
        // Patches: every face of the refined mesh in exactly one patch.
        let mut count = HashMap::new();
        for patch in layout.face_to_patch.values() { for f in &patch.faces { *count.entry(*f).or_insert(0) += 1; } }
        let multi = count.values().filter(|&&c| c > 1).count();
        let missing = layout.granulated_mesh.face_ids().into_iter().filter(|f| !count.contains_key(f)).count();
        // Patch boundaries: every edge between two patches lies on a path.
        let mut path_edges = HashSet::new();
        for path in layout.edge_to_path.values() { for w in path.windows(2) { path_edges.insert((w[0], w[1])); path_edges.insert((w[1], w[0])); } }
        let mesh = &layout.granulated_mesh;
        let mut patch_of = HashMap::new();
        for (p, patch) in &layout.face_to_patch { for f in &patch.faces { patch_of.insert(*f, *p); } }
        let stray = mesh.edge_ids().into_iter().filter(|&e| patch_of.get(&mesh.face(e)) != patch_of.get(&mesh.face(mesh.twin(e))) && !path_edges.contains(&(mesh.root(e), mesh.toor(e)))).count() / 2;
        if stray > 0 { return Err(format!("{stray} edges between patches that are not on a path")); }
        if bad + multi + missing > 0 { Err(format!("{bad} paths not reversed twins, {multi} faces in several patches, {missing} faces in no patch")) } else { Ok(()) }
    });
    let medial = step("medial_loops", || {
        let before = solution.get_quality();
        match solution.medial_loops() {
            None => Err("none".to_owned()),
            Some(m) => {
                let q = check(&m, "medial")?;
                if let Some(b) = before && (q - b).abs() > 1e-9 { return Err(format!("quality changed {b} -> {q}")); }
                Ok(m)
            }
        }
    });
    if let Some(m) = medial { solution = m; }

    step("quad", || {
        let mut s = solution.clone();
        s.construct_quad(QuadDensity::Fixed(3)).map_err(|e| format!("{e}"))?;
        let quad = s.quad.as_ref().ok_or("no quad mesh")?;
        let mesh = &quad.quad_mesh;
        let euler = mesh.nr_verts() as i64 - mesh.nr_edges() as i64 / 2 + mesh.nr_faces() as i64;
        if euler != 2 - 2 * genus { return Err(format!("euler {euler}, expected {}", 2 - 2 * genus)); }
        if let Some(face) = mesh.face_ids().into_iter().find(|&f| mesh.vertices(f).count() != 4) { return Err(format!("face {face:?} has {} corners", mesh.vertices(face).count())); }
        if mesh.vert_ids().into_iter().any(|v| mesh.position(v).iter().any(|c| !c.is_finite())) { return Err("non-finite position".into()); }
        println!("INFO quad faces {}", mesh.nr_faces());
        Ok(())
    });

    step("smooth", || {
        let mut s = solution.clone();
        s.smooth_layout().map_err(|e| format!("{e}"))?;
        check(&s, "smooth").map(|_| ())
    });

    for ext in ["loops", "dc"] {
        step(&format!("roundtrip-{ext}"), || {
            let file = tmp.join(format!("fuzz.{ext}"));
            io::export_solution(&solution, &file).map_err(|e| format!("export {e}"))?;
            let back = io::import_solution(&file).map_err(|e| format!("import {e}"))?;
            if back.loops.len() != solution.loops.len() { return Err(format!("{} loops back, {} before", back.loops.len(), solution.loops.len())); }
            let mut back = back;
            if back.layout.is_none() { back.reconstruct_solution(false).map_err(|e| format!("reconstruct {e}"))?; }
            check(&back, "roundtrip").map(|_| ())
        });
    }

    step("remesh", || {
        let (remeshed, report) = solution.mesh_ref.remesh(&RemeshParams::default())?;
        let m = &remeshed;
        let euler = m.nr_verts() as i64 - m.nr_edges() as i64 / 2 + m.nr_faces() as i64;
        if euler != 2 - 2 * genus { return Err(format!("remeshed euler {euler} ({report:?})")); }
        println!("INFO remeshed faces {}", m.nr_faces());
        Ok(())
    });
}
