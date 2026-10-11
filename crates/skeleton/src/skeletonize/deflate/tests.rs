use super::*;

#[test]
#[ignore = "manual timing; run with --release -- --ignored --nocapture"]
fn contraction_benchmark() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../mehsh/assets/blub001k.obj");
    let mesh = Mesh::<CONTRACTION>::from_obj(&path).unwrap().0;
    run(&mesh, 50).unwrap();
    let mut elapsed = Vec::new();
    for _ in 0..7 {
        let start = std::time::Instant::now();
        run(&mesh, 50).unwrap();
        elapsed.push(start.elapsed());
    }
    elapsed.sort();
    println!("blub001k contraction median: {:?}", elapsed[3]);
}
