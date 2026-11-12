//! The tetrahedralized volume refined so that a boundary loop on its surface is a simple cycle
//! of vertices, for loops whose vertex walks repeat vertices.
//!
//! Every crossing becomes a vertex on its mesh edge, spaced evenly in edge order. Surface
//! triangles are cut along the loop segments inside them: boundaries are disjoint closed
//! curves, so the segments do not cross and cut a triangle into convex regions. Other faces
//! with points on their edges are triangulated as they are. Each tet with points on an edge is
//! replaced by the cone from its centroid over its subdivided faces, which is valid for any
//! convex polyhedron. A second TetGen run on the cut surface would fail on the slivers that
//! cutting thin triangles creates.

use crate::{BoundaryCrossingID, BoundarySurface, PatchBoundary};
use dualcube_types::prelude::*;

pub(super) struct Refined {
    pub vertices: Vec<Vector3D>,
    pub tets: Vec<[usize; 4]>,
    /// The vertex of each crossing.
    pub crossings: HashMap<BoundaryCrossingID, usize>,
}

/// Refine along the given loops; `index` gives the volume vertex of each surface vertex.
pub(super) fn refine(
    mesh: &Mesh<INPUT>,
    surface: &BoundarySurface,
    loops: &[&PatchBoundary],
    index: &HashMap<VertID, usize>,
    vertices: &[Vector3D],
    tets: &[[usize; 4]],
) -> Refined {
    let mut vertices = vertices.to_vec();
    // Points on each surface edge, from its smaller to its larger vertex.
    let mut along: HashMap<[usize; 2], Vec<usize>> = HashMap::new();
    let mut point = HashMap::new();
    let used: HashSet<_> = loops
        .iter()
        .flat_map(|l| l.crossings.iter().copied())
        .collect();
    for (&key, order) in &surface.edge_order {
        let order: Vec<_> = order.iter().copied().filter(|c| used.contains(c)).collect();
        if order.is_empty() {
            continue;
        }
        let (a, b) = (index[&mesh.root(key)], index[&mesh.toor(key)]);
        let (p, q) = (vertices[a], vertices[b]);
        let parts = order.len() as f64 + 1.;
        let mut points = Vec::new();
        for (rank, crossing) in order.into_iter().enumerate() {
            point.insert(crossing, vertices.len());
            points.push(vertices.len());
            vertices.push(p + (q - p) * ((rank + 1) as f64 / parts));
        }
        if a > b {
            points.reverse();
        }
        along.insert(sorted2(a, b), points);
    }
    let mut segments: HashMap<[usize; 3], Vec<[usize; 2]>> = HashMap::new();
    for boundary in loops {
        let crossings = &boundary.crossings;
        for (i, &to) in crossings.iter().enumerate() {
            let from = crossings[(i + crossings.len() - 1) % crossings.len()];
            let face = mesh.face(surface.crossings[to].exit);
            let corners: Vec<_> = mesh.edges(face).map(|e| index[&mesh.root(e)]).collect();
            segments
                .entry(sorted3([corners[0], corners[1], corners[2]]))
                .or_default()
                .push([point[&from], point[&to]]);
        }
    }
    let mut split: HashMap<[usize; 3], Vec<[usize; 3]>> = HashMap::new();
    let mut refined = Vec::with_capacity(tets.len());
    for &tet in tets {
        let touched =
            (0..4).any(|i| (i + 1..4).any(|j| along.contains_key(&sorted2(tet[i], tet[j]))));
        if !touched {
            refined.push(tet);
            continue;
        }
        let sign = volume(&vertices, tet).signum();
        let center = vertices.len();
        vertices.push(tet.iter().map(|&v| vertices[v]).sum::<Vector3D>() / 4.);
        for skip in 0..4 {
            let face = sorted3(
                [0, 1, 2, 3]
                    .map(|k| tet[k])
                    .into_iter()
                    .filter(|&v| v != tet[skip])
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
            );
            let triangles = split
                .entry(face)
                .or_insert_with(|| split_face(face, &along, segments.get(&face)));
            for &[a, b, c] in triangles.iter() {
                let mut new = [a, b, c, center];
                if volume(&vertices, new).signum() != sign {
                    new.swap(0, 1);
                }
                refined.push(new);
            }
        }
    }
    Refined {
        vertices,
        tets: refined,
        crossings: point,
    }
}

fn sorted2(a: usize, b: usize) -> [usize; 2] {
    if a < b { [a, b] } else { [b, a] }
}

fn sorted3(mut face: [usize; 3]) -> [usize; 3] {
    face.sort_unstable();
    face
}

fn volume(vertices: &[Vector3D], [a, b, c, d]: [usize; 4]) -> f64 {
    let o = vertices[a];
    (vertices[b] - o)
        .cross(&(vertices[c] - o))
        .dot(&(vertices[d] - o))
}

/// A face with points on its edges, cut along the given segments and triangulated.
fn split_face(
    [a, b, c]: [usize; 3],
    along: &HashMap<[usize; 2], Vec<usize>>,
    segments: Option<&Vec<[usize; 2]>>,
) -> Vec<[usize; 3]> {
    // Corners and points in the order a, b, c, with the sides each lies on (ab, bc, ca).
    let mut boundary: Vec<(usize, u8)> = Vec::new();
    for (side, (u, v), corner) in [(0, (a, b), 0b101), (1, (b, c), 0b011), (2, (c, a), 0b110)] {
        boundary.push((u, corner));
        let mut points = along.get(&sorted2(u, v)).cloned().unwrap_or_default();
        if u > v {
            points.reverse();
        }
        boundary.extend(points.into_iter().map(|p| (p, 1 << side)));
    }
    let at: HashMap<_, _> = boundary.iter().enumerate().map(|(i, b)| (b.0, i)).collect();
    let mut partner = vec![None; boundary.len()];
    for &[p, q] in segments.into_iter().flatten() {
        let (i, j) = (at[&p], at[&q]);
        assert_ne!(
            boundary[i].1 & boundary[j].1,
            boundary[i].1,
            "a boundary turns back on one mesh edge"
        );
        partner[i] = Some(j);
        partner[j] = Some(i);
    }
    let mut triangles = Vec::new();
    for region in regions(&partner) {
        triangulate(
            region.into_iter().map(|i| boundary[i]).collect(),
            &mut triangles,
        );
    }
    triangles
}

/// Regions of a polygon cut by disjoint chords, as positions along it in its order. A region
/// leaves a chord endpoint along the polygon exactly when it entered by the chord.
pub(super) fn regions(partner: &[Option<usize>]) -> Vec<Vec<usize>> {
    let n = partner.len();
    let mut visited = vec![false; n];
    let mut regions = Vec::new();
    for start in 0..n {
        if visited[start] {
            continue;
        }
        let mut region = Vec::new();
        let mut i = start;
        loop {
            visited[i] = true;
            region.push(i);
            let mut j = (i + 1) % n;
            if let Some(q) = partner[j] {
                region.push(j);
                j = q;
            }
            if j == start {
                break;
            }
            i = j;
        }
        regions.push(region);
    }
    regions
}

/// Clip ears of a convex polygon; points sharing a side bit are collinear, so neither an ear
/// nor what is left may lie on one side.
pub(super) fn triangulate(mut polygon: Vec<(usize, u8)>, triangles: &mut Vec<[usize; 3]>) {
    let flat = |points: &mut dyn Iterator<Item = &(usize, u8)>| {
        points.fold(0b111, |sides, p| sides & p.1) != 0
    };
    while polygon.len() > 3 {
        let n = polygon.len();
        let k = (0..n)
            .find(|&k| {
                let ear = [(k + n - 1) % n, k, (k + 1) % n].map(|i| polygon[i]);
                !flat(&mut ear.iter())
                    && !flat(&mut polygon.iter().enumerate().filter(|e| e.0 != k).map(|e| e.1))
            })
            .expect("a convex region has an ear");
        triangles.push([(k + n - 1) % n, k, (k + 1) % n].map(|i| polygon[i].0));
        polygon.remove(k);
    }
    assert!(!flat(&mut polygon.iter()), "a region lies on one side");
    triangles.push([polygon[0].0, polygon[1].0, polygon[2].0]);
}
