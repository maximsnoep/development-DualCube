use super::uncross;
use std::collections::HashMap;

fn interleave(a: usize, b: usize, c: usize, d: usize) -> bool {
    let inside = |x| (a < x && x < b) || (b < a && (x > a || x < b));
    inside(c) != inside(d)
}

fn crossings(
    points: &[usize],
    faces: &[usize],
    position: &HashMap<(usize, usize), usize>,
) -> usize {
    let mut count = 0;
    for j in 0..faces.len() {
        for i in 0..j {
            if faces[i] == faces[j] {
                let [a, b, c, d] = [i, i + 1, j, j + 1].map(|s| position[&(points[s], faces[i])]);
                count += interleave(a, b, c, d) as usize;
            }
        }
    }
    count
}

#[test]
fn second_pass_reconnects_without_crossing() {
    // Face 0 is passed twice, along chords 0-4 and 2-6 of its boundary.
    let position = HashMap::from([
        ((0, 0), 0),
        ((1, 0), 4),
        ((1, 1), 0),
        ((2, 1), 1),
        ((2, 0), 2),
        ((3, 0), 6),
    ]);
    let mut points = vec![0, 1, 2, 3];
    let mut faces = vec![0, 1, 0];
    uncross(&mut points, &mut faces, |p, f| position[&(p, f)]);
    assert_eq!(points, [0, 2, 1, 3]);
    assert_eq!(faces, [0, 1, 0]);
}

#[test]
fn random_curves_become_simple() {
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut random = |n: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % n as u64) as usize
    };
    for _ in 0..500 {
        let chords = 2 + random(30);
        let count = 2 + random(4);
        let mut faces = vec![random(count)];
        while faces.len() < chords {
            let f = (faces.last().unwrap() + 1 + random(count - 1)) % count;
            faces.push(f);
        }
        // Distinct positions of every point around each face it lies on.
        let mut position = HashMap::new();
        let mut used = HashMap::<usize, Vec<usize>>::new();
        for (p, f) in (0..=chords).flat_map(|p| {
            [
                p.checked_sub(1).map(|j| (p, faces[j])),
                faces.get(p).map(|&f| (p, f)),
            ]
            .into_iter()
            .flatten()
        }) {
            let taken = used.entry(f).or_default();
            let mut x = random(1000);
            while taken.contains(&x) {
                x = random(1000);
            }
            taken.push(x);
            position.insert((p, f), x);
        }
        let mut points: Vec<_> = (0..=chords).collect();
        let before = faces.clone();
        uncross(&mut points, &mut faces, |p, f| position[&(p, f)]);
        assert_eq!(crossings(&points, &faces, &position), 0);
        assert_eq!((points[0], points[chords]), (0, chords));
        let mut sorted = points.clone();
        sorted.sort();
        assert!(sorted.iter().copied().eq(0..=chords));
        let mut faces_sorted = faces.clone();
        faces_sorted.sort();
        let mut before_sorted = before;
        before_sorted.sort();
        assert_eq!(faces_sorted, before_sorted);
    }
}
