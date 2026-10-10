//! Temporary fuzzing harness: mutated OBJ and STL files must load or fail cleanly (never panic or hang).
use dualcube::prelude::*;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Mutex;

static LAST_PANIC: Mutex<String> = Mutex::new(String::new());

fn main() {
    std::panic::set_hook(Box::new(|info| {
        let location = info.location().map(|l| format!("{}:{}", l.file(), l.line())).unwrap_or_default();
        let message = info.payload().downcast_ref::<&str>().map(|s| s.to_string())
            .or_else(|| info.payload().downcast_ref::<String>().cloned()).unwrap_or_default();
        *LAST_PANIC.lock().unwrap() = format!("{message} at {location}");
    }));
    let args: Vec<String> = std::env::args().collect();
    let tmp = std::path::Path::new(&args[1]);
    let sources: Vec<&String> = args[2..].iter().collect();
    let mut state = 0x853C49E6748FEA9Bu64;
    let mut next = move || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; state };
    let mut panics: std::collections::BTreeMap<String, (usize, String)> = Default::default();
    let (mut loaded, mut errors, mut total) = (0, 0, 0);
    let mut odd: std::collections::BTreeMap<&str, usize> = Default::default();
    for source in &sources {
        let bytes = std::fs::read(source).unwrap();
        let ext = if source.ends_with(".stl") { "stl" } else { "obj" };
        let text = String::from_utf8_lossy(&bytes).to_string();
        let lines: Vec<&str> = text.lines().collect();
        for k in 0..300 {
            let mutated: Vec<u8> = match (ext, k % 6) {
                (_, 0) => { let mut b = bytes.clone(); for _ in 0..1 + next() % 20 { if b.is_empty() { break } let i = (next() as usize) % b.len(); b[i] = next() as u8; } b }
                (_, 1) => bytes[..(next() as usize) % (bytes.len() + 1)].to_vec(),
                ("obj", 2) => { let mut l = lines.clone(); for _ in 0..1 + next() % 5 { if l.is_empty() { break } let i = (next() as usize) % l.len(); l.remove(i); } l.join("\n").into_bytes() }
                ("obj", 3) => { let mut l = lines.clone(); for _ in 0..1 + next() % 5 { if l.is_empty() { break } let i = (next() as usize) % l.len(); let j = (next() as usize) % l.len(); let line = l[i]; l.insert(j, line); } l.join("\n").into_bytes() }
                ("obj", 4) => { lines.iter().map(|line| if line.starts_with("f ") && next() % 50 == 0 { let n = next() % 4; match n { 0 => "f 1 1 1".to_string(), 1 => "f 0 2 3".to_string(), 2 => format!("f 1 2 {}", 1 + next() % 10_000_000), _ => "f 1 2".to_string() } } else { line.to_string() }).collect::<Vec<_>>().join("\n").into_bytes() }
                ("obj", 5) => { lines.iter().map(|line| if line.starts_with("v ") && next() % 50 == 0 { ["v nan 0 0", "v inf 1 2", "v 1e400 0 0", "v 1 2", "v a b c"][(next() % 5) as usize].to_string() } else { line.to_string() }).collect::<Vec<_>>().join("\n").into_bytes() }
                _ => { let mut b = bytes.clone(); let i = (next() as usize) % (b.len().max(1)); b.insert(i.min(b.len()), next() as u8); b }
            };
            let file = tmp.join(format!("fuzz_load.{ext}"));
            std::fs::write(&file, &mutated).unwrap();
            total += 1;
            match catch_unwind(AssertUnwindSafe(|| Mesh::<INPUT>::from_file(&file).map(|(m, ..)| {
                let nonfinite = m.vert_ids().into_iter().any(|v| m.position(v).iter().any(|c| !c.is_finite()));
                let repeated = m.face_ids().into_iter().any(|f| { let vs: Vec<_> = m.vertices(f).collect(); (1..vs.len()).any(|i| vs[..i].contains(&vs[i])) });
                let euler = m.nr_verts() as i64 - m.nr_edges() as i64 / 2 + m.nr_faces() as i64;
                (nonfinite, repeated, euler)
            }))) {
                Ok(Ok((nonfinite, repeated, euler))) => {
                    loaded += 1;
                    if nonfinite { *odd.entry("non-finite positions").or_default() += 1; }
                    if repeated { *odd.entry("faces with repeated corners").or_default() += 1; }
                    if euler % 2 != 0 || euler > 2 { *odd.entry("impossible Euler characteristic (not closed/orientable)").or_default() += 1; }
                }
                Ok(Err(_)) => errors += 1,
                Err(_) => {
                    let msg = LAST_PANIC.lock().unwrap().clone();
                    let entry = panics.entry(msg.split(" at ").last().unwrap_or("").to_string()).or_insert((0, msg.clone()));
                    entry.0 += 1;
                    if entry.0 == 1 { std::fs::copy(&file, tmp.join(format!("panic_{}.{ext}", panics.len()))).unwrap(); }
                }
            }
        }
    }
    println!("{total} mutated files: {loaded} loaded, {errors} errors, {} panics", panics.values().map(|p| p.0).sum::<usize>());
    for (what, count) in &odd { println!("  loaded with {what}: {count}"); }
    for (location, (count, msg)) in panics { println!("  {count}x {msg} [{location}]"); }
}
