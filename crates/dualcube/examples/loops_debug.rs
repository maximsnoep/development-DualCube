//! Debug loop construction: sample loops per axis, report statistics, and render them to a PNG.
//!
//! `cargo run -p dualcube --example loops_debug --release -- <mesh.obj|stl> <out.png> [initialize]`

use dualcube::prelude::*;
use std::io::Write;
use std::sync::Arc;

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let path = std::path::Path::new(&args[1]);
    let out = std::path::Path::new(&args[2]);
    let initialize = args.get(3).is_some_and(|a| a == "initialize");

    let (mesh, _, _) = Mesh::<INPUT>::from_file(path).unwrap();
    let mut solution = Solution::new(Arc::new(mesh));
    let timer = std::time::Instant::now();
    solution.prepare_flow();
    println!("flow fields and graphs: {:?}", timer.elapsed());
    let timer = std::time::Instant::now();

    let mut drawn: Vec<(Vec<Vector3D>, [u8; 3])> = vec![];
    if initialize {
        solution.initialize();
        println!(
            "initialized: loops={} quality={:?} in {:?}",
            solution.loops.len(),
            solution.get_quality(),
            timer.elapsed()
        );
        for (loop_id, lewp) in &solution.loops {
            let positions = solution.get_coordinates_of_loop(loop_id);
            report(&format!("{:?}", lewp.direction), &positions);
            drawn.push((positions, color(lewp.direction)));
        }
    } else {
        for direction in DIRECTIONS {
            let loops = solution.sample_loops(3, direction, OrderedFloat, |(_, s)| s);
            for edges in loops {
                if edges.is_empty() {
                    continue;
                }
                let midpoints = edges
                    .iter()
                    .map(|&e| solution.mesh_ref.position(e))
                    .collect::<Vec<_>>();
                let positions = solution.loop_positions(&edges, direction);
                report(&format!("{direction:?} midpoints"), &midpoints);
                report(&format!("{direction:?} taut     "), &positions);
                drawn.push((positions, color(direction)));
            }
        }
    }

    if !initialize {
        // Why do combinations of sampled loops fail?
        let samples = DIRECTIONS
            .map(|d| solution.sample_loops(3, d, OrderedFloat, |(p, _)| -(p.len() as f64)));
        for x in &samples[0] {
            for y in &samples[1] {
                for z in &samples[2] {
                    let mut candidate = solution.clone();
                    candidate.add_loop(Loop::new(x.clone(), Direction::X));
                    candidate.add_loop(Loop::new(y.clone(), Direction::Y));
                    candidate.add_loop(Loop::new(z.clone(), Direction::Z));
                    let dual = Dual::from(candidate.mesh_ref.clone(), &candidate.loops);
                    let full = candidate.reconstruct_solution_with(false, LayoutParams::fast());
                    println!(
                        "  combination: dual={:?} reconstruct={:?}",
                        dual.err(),
                        full.err()
                    );
                }
            }
        }
    }
    render(&solution.mesh_ref, &drawn, out, 1000);
    println!("wrote {}", out.display());
}

fn color(direction: Direction) -> [u8; 3] {
    match direction {
        Direction::X => [220, 40, 40],
        Direction::Y => [40, 170, 40],
        Direction::Z => [40, 80, 230],
    }
}

// Length and total turning angle (in degrees) of a closed polyline.
fn report(name: &str, positions: &[Vector3D]) {
    // A loop crosses an edge as two consecutive half-edges at the same position.
    let positions = positions
        .iter()
        .copied()
        .dedup_by(|a, b| (a - b).norm() < 1e-12)
        .collect::<Vec<_>>();
    let n = positions.len();
    let length: f64 = (0..n)
        .map(|i| (positions[(i + 1) % n] - positions[i]).norm())
        .sum();
    let turning: f64 = (0..n)
        .filter_map(|i| {
            let a = positions[(i + 1) % n] - positions[i];
            let b = positions[(i + 2) % n] - positions[(i + 1) % n];
            (a.norm() > 1e-12 && b.norm() > 1e-12).then(|| a.angle(&b))
        })
        .sum();
    println!(
        "  {name}: points={n} length={length:.4} turning={:.0}deg",
        turning.to_degrees()
    );
}

// Minimal z-buffer rasterizer (front view along -z, plus a second view rotated by 90 degrees) to a PNG.
fn render(
    mesh: &Mesh<INPUT>,
    polylines: &[(Vec<Vector3D>, [u8; 3])],
    out: &std::path::Path,
    size: usize,
) {
    let views = [0.0f64, std::f64::consts::FRAC_PI_2, std::f64::consts::PI];
    let width = size * views.len();
    let mut image = vec![[255u8; 3]; width * size];
    let center = mesh.center();
    let scale = 0.9 * size as f64 / (2. * mesh.max_dim());
    for (v, &yaw) in views.iter().enumerate() {
        let (sin, cos) = (yaw.sin(), yaw.cos());
        let tilt = 0.4f64;
        let project = |p: Vector3D| {
            let p = p - center;
            let (x, z) = (cos * p.x + sin * p.z, -sin * p.x + cos * p.z);
            let (y, z) = (
                tilt.cos() * p.y - tilt.sin() * z,
                tilt.sin() * p.y + tilt.cos() * z,
            );
            (
                x * scale + size as f64 / 2. + (v * size) as f64,
                -y * scale + size as f64 / 2.,
                z,
            )
        };
        let mut depth = vec![f64::NEG_INFINITY; width * size];
        let light = Vector3D::new(0.3, 0.5, 1.0).normalize();
        for face in mesh.face_ids() {
            let [a, b, c] = mesh
                .vertices(face)
                .map(|v| project(mesh.position(v)))
                .collect_array::<3>()
                .unwrap();
            let n = mesh.normal(face);
            let n = Vector3D::new(cos * n.x + sin * n.z, n.y, -sin * n.x + cos * n.z);
            let n = Vector3D::new(
                n.x,
                tilt.cos() * n.y - tilt.sin() * n.z,
                tilt.sin() * n.y + tilt.cos() * n.z,
            );
            let shade = (0.35 + 0.65 * n.dot(&light).abs()) * 235.;
            let (min_x, max_x) = (
                a.0.min(b.0).min(c.0).floor().max(0.) as usize,
                a.0.max(b.0).max(c.0).ceil().min(width as f64 - 1.) as usize,
            );
            let (min_y, max_y) = (
                a.1.min(b.1).min(c.1).floor().max(0.) as usize,
                a.1.max(b.1).max(c.1).ceil().min(size as f64 - 1.) as usize,
            );
            let area = (b.0 - a.0) * (c.1 - a.1) - (c.0 - a.0) * (b.1 - a.1);
            if area.abs() < 1e-12 {
                continue;
            }
            for py in min_y..=max_y {
                for px in min_x..=max_x {
                    let (x, y) = (px as f64 + 0.5, py as f64 + 0.5);
                    let w0 = ((b.0 - x) * (c.1 - y) - (c.0 - x) * (b.1 - y)) / area;
                    let w1 = ((c.0 - x) * (a.1 - y) - (a.0 - x) * (c.1 - y)) / area;
                    let w2 = 1. - w0 - w1;
                    if w0 < 0. || w1 < 0. || w2 < 0. {
                        continue;
                    }
                    let z = w0 * a.2 + w1 * b.2 + w2 * c.2;
                    let i = py * width + px;
                    if z > depth[i] {
                        depth[i] = z;
                        image[i] = [shade as u8; 3];
                    }
                }
            }
        }
        let bias = 0.01 * mesh.max_dim() * scale / scale;
        for (polyline, rgb) in polylines {
            for k in 0..polyline.len() {
                let (a, b) = (
                    project(polyline[k]),
                    project(polyline[(k + 1) % polyline.len()]),
                );
                let steps = ((b.0 - a.0).abs().max((b.1 - a.1).abs()) * 2.)
                    .ceil()
                    .max(1.) as usize;
                for s in 0..=steps {
                    let t = s as f64 / steps as f64;
                    let (x, y, z) = (
                        a.0 + t * (b.0 - a.0),
                        a.1 + t * (b.1 - a.1),
                        a.2 + t * (b.2 - a.2),
                    );
                    for (dx, dy) in [(0i64, 0i64), (1, 0), (0, 1), (1, 1)] {
                        let (px, py) = (x as i64 + dx, y as i64 + dy);
                        if px < 0 || py < 0 || px >= width as i64 || py >= size as i64 {
                            continue;
                        }
                        let i = py as usize * width + px as usize;
                        if z + bias >= depth[i] {
                            image[i] = *rgb;
                        }
                    }
                }
            }
        }
    }
    write_png(out, width, size, &image);
}

// Uncompressed PNG writer (zlib stored blocks).
fn write_png(path: &std::path::Path, width: usize, height: usize, pixels: &[[u8; 3]]) {
    fn crc32(data: &[u8]) -> u32 {
        let mut crc = 0xffff_ffffu32;
        for &byte in data {
            crc ^= u32::from(byte);
            for _ in 0..8 {
                crc = if crc & 1 != 0 {
                    0xedb8_8320 ^ (crc >> 1)
                } else {
                    crc >> 1
                };
            }
        }
        !crc
    }
    fn chunk(out: &mut Vec<u8>, kind: &[u8], data: &[u8]) {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let mut body = kind.to_vec();
        body.extend_from_slice(data);
        out.extend_from_slice(&body);
        out.extend_from_slice(&crc32(&body).to_be_bytes());
    }
    let mut raw = Vec::with_capacity(height * (width * 3 + 1));
    for y in 0..height {
        raw.push(0);
        for x in 0..width {
            raw.extend_from_slice(&pixels[y * width + x]);
        }
    }
    let mut zlib = vec![0x78, 0x01];
    for (i, block) in raw.chunks(65535).enumerate() {
        let last = (i + 1) * 65535 >= raw.len();
        zlib.push(u8::from(last));
        zlib.extend_from_slice(&(block.len() as u16).to_le_bytes());
        zlib.extend_from_slice(&(!(block.len() as u16)).to_le_bytes());
        zlib.extend_from_slice(block);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in &raw {
        a = (a + u32::from(byte)) % 65521;
        b = (b + a) % 65521;
    }
    zlib.extend_from_slice(&((b << 16) | a).to_be_bytes());

    let mut png = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    let mut header = vec![];
    header.extend_from_slice(&(width as u32).to_be_bytes());
    header.extend_from_slice(&(height as u32).to_be_bytes());
    header.extend_from_slice(&[8, 2, 0, 0, 0]);
    chunk(&mut png, b"IHDR", &header);
    chunk(&mut png, b"IDAT", &zlib);
    chunk(&mut png, b"IEND", &[]);
    std::fs::File::create(path)
        .unwrap()
        .write_all(&png)
        .unwrap();
}
