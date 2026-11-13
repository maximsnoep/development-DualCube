//! Move boundary Steiner points, which TetGen adds on input edges or faces when it cannot recover
//! the surface otherwise, into the volume so the boundary is the input triangulation again.
//!
//! A boundary Steiner point `s` refines input faces: its boundary triangles tile either one face
//! (`s` inside it) or two faces sharing an edge (`s` on it, between the two link vertices on that
//! edge). Its star is replaced by a cone from an interior point over the star's other faces and
//! the merged faces; points blocking a merge (several on one edge or face) go first. Every new tet
//! must keep the orientation of the tet it comes from, which makes the cone a valid
//! tetrahedralization of the same region.

use mehsh::prelude::*;

/// Returns how many boundary Steiner points were moved inside, or `None` if one could not be.
/// `input_faces` are the input triangles (sorted).
pub(super) fn move_inside(
    n_input: usize,
    vertices: &mut [Vector3D],
    tets: &mut Vec<[usize; 4]>,
    input_faces: &HashSet<[usize; 3]>,
) -> Option<usize> {
    let mut moved = 0;
    let mut on = None;
    loop {
        let boundary = boundary_faces(tets);
        let on = on.get_or_insert_with(|| locate(n_input, vertices, &boundary, input_faces));
        let mut steiner: Vec<usize> = boundary
            .iter()
            .flatten()
            .copied()
            .filter(|&v| v >= n_input)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        if steiner.is_empty() {
            return Some(moved);
        }
        steiner.sort_unstable();
        let mut star = HashMap::<usize, Vec<usize>>::new();
        for (t, tet) in tets.iter().enumerate() {
            for &v in tet {
                star.entry(v).or_default().push(t);
            }
        }
        // Stars of points moved in one pass do not share tets.
        let mut touched = HashSet::new();
        let mut removed = HashSet::new();
        let mut added = vec![];
        for s in steiner {
            let around = &star[&s];
            if around.iter().any(|t| touched.contains(t)) {
                continue;
            }
            let Some(cone) = cone(s, vertices, tets, around, &boundary, on) else {
                continue;
            };
            touched.extend(
                around
                    .iter()
                    .flat_map(|&t| tets[t])
                    .flat_map(|v| star[&v].iter().copied()),
            );
            removed.extend(around.iter().copied());
            vertices[s] = cone.0;
            added.extend(cone.1);
            moved += 1;
        }
        if added.is_empty() {
            return None;
        }
        let mut index = 0;
        tets.retain(|_| {
            index += 1;
            !removed.contains(&(index - 1))
        });
        tets.extend(added);
    }
}

/// Boundary faces, sorted.
fn boundary_faces(tets: &[[usize; 4]]) -> HashSet<[usize; 3]> {
    let mut count = HashMap::<[usize; 3], usize>::new();
    for &tet in tets {
        for opposite in 0..4 {
            let mut face = without(tet, opposite);
            face.sort_unstable();
            *count.entry(face).or_default() += 1;
        }
    }
    count
        .into_iter()
        .filter_map(|(face, count)| (count == 1).then_some(face))
        .collect()
}

fn without(tet: [usize; 4], i: usize) -> [usize; 3] {
    let mut face = [0; 3];
    let mut k = 0;
    for (j, &v) in tet.iter().enumerate() {
        if j != i {
            face[k] = v;
            k += 1;
        }
    }
    face
}

fn volume(vertices: &[Vector3D], tet: [usize; 4]) -> f64 {
    let [a, b, c, d] = tet.map(|i| vertices[i]);
    (b - a).cross(&(c - a)).dot(&(d - a))
}

/// The tet's points with vertex `apex` at `p`.
fn points(vertices: &[Vector3D], tet: [usize; 4], apex: usize, p: Vector3D) -> [Vector3D; 4] {
    let mut points = tet.map(|i| vertices[i]);
    points[apex] = p;
    points
}

/// The new position of `s` and the tets replacing its star.
fn cone(
    s: usize,
    vertices: &[Vector3D],
    tets: &[[usize; 4]],
    around: &[usize],
    boundary: &HashSet<[usize; 3]>,
    on: &HashMap<usize, Option<[usize; 2]>>,
) -> Option<(Vector3D, Vec<[usize; 4]>)> {
    // (tet, index of the vertex the new point replaces, sign of the tet it comes from). The star's
    // tets keep their faces opposite `s`.
    let mut templates = vec![];
    for &t in around {
        let tet = tets[t];
        templates.push((tet, tet.iter().position(|&v| v == s)?, volume(vertices, tet).signum()));
    }
    // Boundary triangles around `s`, as (tet, index of the tet's vertex off the triangle).
    let mut triangles = vec![];
    for &t in around {
        for opposite in 0..4 {
            let mut face = without(tets[t], opposite);
            face.sort_unstable();
            if face.contains(&s) && boundary.contains(&face) {
                triangles.push((tets[t], opposite));
            }
        }
    }
    let link = boundary_link(s, vertices, &triangles)?;
    let merged = merge(s, &link, on)?;
    // A boundary triangle has the orientation of the merged face containing it, so its tet with
    // `s` replaced by that face's third vertex and its apex by the new point keeps its sign.
    for &(tet, opposite) in &triangles {
        let others: Vec<_> = without(tet, opposite)
            .into_iter()
            .filter(|&v| v != s)
            .collect();
        let target = merged
            .iter()
            .find(|m| others.iter().all(|v| m.contains(v)))?;
        let third = *target.iter().find(|v| !others.contains(v))?;
        let mut new = tet;
        new[tet.iter().position(|&v| v == s)?] = third;
        templates.push((new, opposite, volume(vertices, tet).signum()));
    }
    // Triangles of one merged face give the same tet.
    let mut seen = HashSet::new();
    let mut unique = vec![];
    for (tet, apex, sign) in templates {
        let mut key = tet;
        key[apex] = usize::MAX;
        key.sort_unstable();
        if seen.insert(key) {
            unique.push((tet, apex, sign));
        }
    }
    if unique.iter().any(|&(_, _, sign)| sign == 0.) {
        return None;
    }
    let scale = link
        .iter()
        .map(|&v| (vertices[v] - vertices[s]).norm())
        .fold(0., f64::max);
    let origin = vertices[s];
    let signed = |p: Vector3D, (tet, apex, sign): ([usize; 4], usize, f64)| {
        let [a, b, c, d] = points(vertices, tet, apex, p);
        (b - a).cross(&(c - a)).dot(&(d - a)) * sign
    };
    // Each volume is affine in the point: value at `s` and gradient.
    let planes: Vec<_> = unique
        .iter()
        .map(|&template| {
            let at = signed(origin, template);
            let gradient = Vector3D::from_fn(|k, _| {
                (signed(origin + Vector3D::ith(k, scale), template) - at) / scale
            });
            (at.max(0.), gradient)
        })
        .collect();
    let position = origin + deepest(&planes, scale)?;
    // Each volume well above its rounding error (bounded by the product of the edge lengths).
    let valid = unique.iter().all(|&template| {
        let [a, b, c, d] = points(vertices, template.0, template.1, position);
        let bound = (b - a).norm() * (c - a).norm() * (d - a).norm();
        signed(position, template) > 1e-13 * bound
    });
    if !valid {
        return None;
    }
    let cone = unique
        .into_iter()
        .map(|(mut tet, apex, _)| {
            tet[apex] = s;
            tet
        })
        .collect();
    Some((position, cone))
}

/// The cycle of vertices around `s` on the boundary.
fn boundary_link(
    s: usize,
    vertices: &[Vector3D],
    triangles: &[([usize; 4], usize)],
) -> Option<Vec<usize>> {
    let mut next = HashMap::new();
    for &(tet, opposite) in triangles {
        let mut face = without(tet, opposite);
        // The same orientation (relative to the volume) for every triangle.
        if (opposite % 2 == 1) != (volume(vertices, tet) < 0.) {
            face.swap(0, 1);
        }
        let i = face.iter().position(|&v| v == s)?;
        let (a, b) = (face[(i + 1) % 3], face[(i + 2) % 3]);
        if next.insert(a, b).is_some() {
            return None;
        }
    }
    let start = *next.keys().min()?;
    let mut cycle = vec![start];
    let mut v = next[&start];
    while v != start {
        if cycle.len() > next.len() {
            return None;
        }
        cycle.push(v);
        v = *next.get(&v)?;
    }
    (cycle.len() == next.len()).then_some(cycle)
}

/// The input edge each boundary Steiner point lies on, or `None` inside an input face. Candidate
/// edges join input vertices next to the point or to Steiner points connected to it.
fn locate(
    n_input: usize,
    vertices: &[Vector3D],
    boundary: &HashSet<[usize; 3]>,
    input_faces: &HashSet<[usize; 3]>,
) -> HashMap<usize, Option<[usize; 2]>> {
    let mut neighbors = HashMap::<usize, HashSet<usize>>::new();
    for face in boundary {
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    neighbors.entry(face[i]).or_default().insert(face[j]);
                }
            }
        }
    }
    let edges: HashSet<[usize; 2]> = input_faces
        .iter()
        .flat_map(|&[a, b, c]| [[a, b], [a, c], [b, c]])
        .collect();
    let mut result = HashMap::new();
    for &s in neighbors.keys().filter(|&&v| v >= n_input) {
        let mut cluster = vec![s];
        let mut seen = HashSet::from([s]);
        let mut corners = HashSet::new();
        while let Some(v) = cluster.pop() {
            for &w in &neighbors[&v] {
                if w < n_input {
                    corners.insert(w);
                } else if seen.insert(w) {
                    cluster.push(w);
                }
            }
        }
        let p = vertices[s];
        let mut best: Option<(f64, [usize; 2])> = None;
        for &a in &corners {
            for &b in &corners {
                if a < b && edges.contains(&[a, b]) {
                    let (x, y) = (vertices[b] - vertices[a], p - vertices[a]);
                    let t = x.dot(&y) / x.norm_squared();
                    let distance = x.cross(&y).norm() / x.norm_squared();
                    if t > 0. && t < 1. && best.is_none_or(|(d, _)| distance < d) {
                        best = Some((distance, [a, b]));
                    }
                }
            }
        }
        // TetGen puts points on segments up to rounding.
        result.insert(s, best.filter(|&(d, _)| d < 1e-9).map(|(_, edge)| edge));
    }
    result
}

/// The faces the boundary triangles around `s` merge into: one face for a point inside a face,
/// two for a point on an edge, split by the two link vertices on that edge. `None` while another
/// point on the same edge or face is in the way.
fn merge(
    s: usize,
    link: &[usize],
    on: &HashMap<usize, Option<[usize; 2]>>,
) -> Option<Vec<[usize; 3]>> {
    match (on.get(&s)?, link) {
        (None, &[a, b, c]) => Some(vec![[a, b, c]]),
        (&Some(edge), &[a, b, c, d]) => {
            let along = |v: usize| edge.contains(&v) || on.get(&v) == Some(&Some(edge));
            match [a, b, c, d].map(along) {
                [true, false, true, false] => Some(vec![[a, c, b], [a, c, d]]),
                [false, true, false, true] => Some(vec![[b, d, a], [b, d, c]]),
                _ => None,
            }
        }
        _ => None,
    }
}

/// The offset `x` (within `radius` per coordinate) maximizing the smallest distance
/// `(b + g·x) / |g|` over the given `(b, g)` with `b >= 0`, by the simplex method.
fn deepest(planes: &[(f64, Vector3D)], radius: f64) -> Option<Vector3D> {
    // Variables: x⁺ (3), x⁻ (3), depth; rows `-g·x + |g| depth <= b` and `±x_k <= radius`.
    let n = 7;
    let mut rows: Vec<(Vec<f64>, f64)> = vec![];
    for &(b, g) in planes {
        let norm = g.norm();
        if norm == 0. {
            return None;
        }
        let mut row = vec![0.; n];
        for k in 0..3 {
            row[k] = -g[k] / norm;
            row[3 + k] = g[k] / norm;
        }
        row[6] = 1.;
        rows.push((row, b / norm));
    }
    for k in 0..6 {
        let mut row = vec![0.; n];
        row[k] = 1.;
        rows.push((row, radius));
    }
    let m = rows.len();
    // Tableau with slack basis; the origin is feasible since every `b >= 0`.
    let mut table: Vec<Vec<f64>> = rows
        .iter()
        .enumerate()
        .map(|(i, (row, b))| {
            let mut line = row.clone();
            line.extend((0..m).map(|j| f64::from(u8::from(i == j))));
            line.push(*b);
            line
        })
        .collect();
    let mut objective = vec![0.; n + m + 1];
    objective[6] = -1.;
    let mut basis: Vec<usize> = (n..n + m).collect();
    for _ in 0..1000 {
        // Bland's rule: the lowest improving column, then the lowest basis index among ties.
        let Some(column) = (0..n + m).find(|&j| objective[j] < -1e-12) else {
            let mut x = [0.; 7];
            for (i, &var) in basis.iter().enumerate() {
                if var < n {
                    x[var] = table[i][n + m];
                }
            }
            return Some(Vector3D::new(x[0] - x[3], x[1] - x[4], x[2] - x[5]));
        };
        let mut pivot: Option<(f64, usize)> = None;
        for i in 0..m {
            if table[i][column] > 1e-12 {
                let ratio = table[i][n + m] / table[i][column];
                if pivot.is_none_or(|(best, row)| {
                    ratio < best - 1e-15 || (ratio <= best + 1e-15 && basis[i] < basis[row])
                }) {
                    pivot = Some((ratio, i));
                }
            }
        }
        let (_, row) = pivot?;
        let value = table[row][column];
        for x in &mut table[row] {
            *x /= value;
        }
        let pivot_row = table[row].clone();
        for (i, line) in table.iter_mut().enumerate() {
            let factor = line[column];
            if i != row && factor != 0. {
                for (x, p) in line.iter_mut().zip(&pivot_row) {
                    *x -= factor * p;
                }
            }
        }
        let factor = objective[column];
        for (x, p) in objective.iter_mut().zip(&pivot_row) {
            *x -= factor * p;
        }
        basis[row] = column;
    }
    None
}
