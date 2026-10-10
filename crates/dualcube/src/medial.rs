//! Medial loops: every loop through the middle of the patches of the layout (see `Solution::medial_loops`).
//!
//! Every patch of a layout is dual to an intersection of two loops, and has four sides (paths): two are crossed by
//! the one loop, two by the other. Every patch is mapped onto the unit square (a Tutte embedding of its triangles of the
//! refined mesh: its corners onto the corners of the square, its sides by arc length, every inner vertex at the mean of
//! its neighbors), and its two loops become the lines x = 1/2 and y = 1/2 of the square. So every loop crosses every
//! side at its middle (by arc length, the same from both of its patches), runs through the middle of every patch, and
//! the two loops of a patch cross exactly once, in it (a Tutte embedding onto a convex polygon is bijective). The
//! loops thus form the loop structure that the layout belongs to, and the layout is kept as it is.
//!
//! The loops are traced on the refined mesh, and become loops on the input mesh by where they cross its edges (with
//! straight chords in between, inside its triangles): two curves inside a triangle cross an odd number of times
//! exactly if their chords cross, so the loop structure is kept (checked).

use crate::prelude::*;
use slotmap::SlotMap;

// The values of the level sets are kept this far from those of the vertices (such that no level set passes through
// a vertex).
const LEVEL_EPSILON: f64 = 1e-9;
// The Tutte embedding of a patch: the (relative) accuracy, and the maximum number of iterations of its solver.
const TUTTE_TOLERANCE: f64 = 1e-10;
const TUTTE_ITERATIONS: usize = 5000;

// A loop crossing an edge of the refined mesh: from the face `from` to the face `to`, at the given point.
#[derive(Clone, Copy, Debug)]
struct Crossing {
    from: FaceID,
    to: FaceID,
    point: Vector3D,
}

impl Solution {
    /// This solution with every loop moved to the middle of the patches of its layout (see the module documentation),
    /// with the layout as it is (it belongs to the same loop structure). `None` if there is no layout (with region
    /// tracking), or if the loops cannot be traced or form another structure (e.g., on a mesh too coarse for its
    /// patches).
    #[must_use]
    pub fn medial_loops(&self) -> Option<Self> {
        let timer = std::time::Instant::now();
        let dual = self.dual.as_ref().ok()?;
        let layout = self.layout.as_ref()?;
        let regions = layout.regions.as_ref()?;
        let polycube = &layout.polycube_ref;
        let structure = &polycube.structure;
        let loop_structure = &dual.loop_structure;
        let granulated = &layout.granulated_mesh;

        // The side (a half-edge of the polycube) that every segment leaving an intersection crosses, and the patch of
        // every intersection. The polycube is the dual of the loop structure: every segment crosses the polycube edge
        // between the corners of the regions on both of its sides, and the half-edge of that edge in the patch of the
        // intersection is the directed one (corners may repeat around a patch, e.g., (u, v, w, v), but every directed
        // half-edge is unique). The direction follows from how the polycube is built: whichever of the two
        // conventions gives every intersection one patch (that of all its segments).
        let corner_of =
            |region: LoopRegionID| polycube.region_to_vertex.get_by_left(&region).copied();
        let directed = |segment: LoopSegmentID, forward: bool| {
            let a = corner_of(loop_structure.face(segment))?;
            let b = corner_of(loop_structure.face(loop_structure.twin(segment)))?;
            let (u, v) = if forward { (a, b) } else { (b, a) };
            structure.edge_between_verts(u, v).map(|(edge, _)| edge)
        };
        let (patch_of, side_of) = [true, false].into_iter().find_map(|forward| {
            let mut patch_of: HashMap<LoopIntersectionID, FaceKey<POLYCUBE>> = HashMap::new();
            let mut side_of: HashMap<LoopSegmentID, EdgeKey<POLYCUBE>> = HashMap::new();
            for intersection in loop_structure.vert_ids() {
                let mut patch = None;
                for segment in loop_structure.edges(intersection) {
                    let side = directed(segment, forward)?;
                    if *patch.get_or_insert(structure.face(side)) != structure.face(side) {
                        return None;
                    }
                    side_of.insert(segment, side);
                }
                patch_of.insert(intersection, patch?);
            }
            Some((patch_of, side_of))
        })?;

        // Every patch onto the unit square.
        let squares = structure
            .face_ids()
            .into_par()
            .map(|patch| tutte_square(layout, patch).map(|square| (patch, square)))
            .collect::<Vec<_>>()
            .into_iter()
            .collect::<Option<HashMap<_, _>>>()?;

        // Every loop: through its patches in its order, on the refined mesh, then on the input mesh.
        let mut loops: SlotMap<LoopID, Loop> = self.loops.clone();
        for loop_id in self.loops.keys() {
            let chain = loop_chain(dual, loop_id);
            if chain.len() < 2 {
                return None;
            }
            let mut crossings = vec![];
            for k in 0..chain.len() {
                // (The segment that arrives at the intersection, as one that leaves it: its twin.)
                let (intersection, exit) = chain[k];
                let (_, arrival) = chain[(k + chain.len() - 1) % chain.len()];
                let patch = patch_of[&intersection];
                let entry = *side_of.get(&loop_structure.twin(arrival))?;
                let exit = *side_of.get(&exit)?;
                crossings.extend(trace_in_patch(
                    layout,
                    patch,
                    &squares[&patch],
                    entry,
                    exit,
                )?);
            }
            let old = &self.loops[loop_id];
            loops[loop_id] = input_loop(&self.mesh_ref, regions, &crossings, old.direction)?;
        }

        // The same loop structure: every loop crosses the same loops in the same (cyclic) order.
        let LoopState { loops, occupied } = LoopState::from_loops(&self.mesh_ref, loops);
        let new_dual = Dual::from(self.mesh_ref.clone(), &loops).ok()?;
        for loop_id in loops.keys() {
            if !same_cycle(
                &crossing_order(dual, loop_id),
                &crossing_order(&new_dual, loop_id),
            ) {
                debug!("medial_loops: loop {loop_id:?} crosses differently");
                return None;
            }
        }

        // The corners of the new regions: every new intersection lies in its patch, and a region has the corner
        // that the patches of its intersections share.
        let mut pieces: HashMap<FaceID, Vec<FaceID>> = HashMap::new();
        for (&piece, &input) in &regions.face_input {
            pieces.entry(input).or_default().push(piece);
        }
        let mut patch_of_face: HashMap<FaceID, FaceKey<POLYCUBE>> = HashMap::new();
        for (&patch, faces) in &layout.face_to_patch {
            for &face in &faces.faces {
                patch_of_face.insert(face, patch);
            }
        }
        let new_patch_of = |intersection: LoopIntersectionID| {
            let LoopIntersection { face, position, .. } = *new_dual.intersection(intersection);
            pieces
                .get(&face)?
                .iter()
                .min_by_key(|&&piece| {
                    OrderedFloat(distance_to_triangle(granulated, piece, position))
                })
                .and_then(|piece| patch_of_face.get(piece).copied())
        };
        let new_structure = &new_dual.loop_structure;
        // (A map of the type of `Polycube::region_to_vertex`.)
        let mut region_to_vertex = polycube.region_to_vertex.clone();
        region_to_vertex.clear();
        for region in new_structure.face_ids() {
            let mut common: Option<HashSet<VertKey<POLYCUBE>>> = None;
            for intersection in new_structure.vertices(region) {
                let corners: HashSet<_> = structure.vertices(new_patch_of(intersection)?).collect();
                common = Some(match common {
                    Some(common) => common.intersection(&corners).copied().collect(),
                    None => corners,
                });
            }
            let common = common?;
            if common.len() != 1 {
                debug!("medial_loops: a region has {} corners", common.len());
                return None;
            }
            region_to_vertex
                .insert_no_overwrite(region, *common.iter().next()?)
                .ok()?;
        }
        if region_to_vertex.len() != structure.vert_ids().len() {
            return None;
        }

        let mut result = self.clone();
        result.loops = loops;
        result.occupied = occupied;
        if let Some(polycube) = result.polycube.as_mut() {
            polycube.region_to_vertex = region_to_vertex.clone();
        }
        if let Some(layout) = result.layout.as_mut() {
            // (Its region tracking still describes the loops it was embedded with; mutations do not need it, see
            // `Layout::begin_mutation`.)
            layout.polycube_ref.region_to_vertex = region_to_vertex;
            layout.dual_ref = new_dual.clone();
        }
        result.dual = Ok(new_dual);
        result.quad = None;
        result.corner_hint = None;
        result.targets = None;
        result.loop_penalties = None;
        info!(
            "medial_loops: {} loops in {:?}; quality {:?}",
            result.loops.len(),
            timer.elapsed(),
            result.get_quality()
        );
        Some(result)
    }
}

// The intersections of a loop in its order, each with the segment that leaves it (along the direction of the loop).
fn loop_chain(dual: &Dual, loop_id: LoopID) -> Vec<(LoopIntersectionID, LoopSegmentID)> {
    let structure = &dual.loop_structure;
    let next: HashMap<_, _> = structure
        .edge_ids()
        .into_iter()
        .filter(|&segment| {
            dual.segment_to_loop(segment) == loop_id
                && dual.segment_to_orientation(segment) == Sign::Positive
        })
        .map(|segment| (structure.root(segment), segment))
        .collect();
    let Some(&start) = next.keys().next() else {
        return vec![];
    };
    let mut chain = vec![];
    let mut at = start;
    while let Some(&segment) = next.get(&at) {
        chain.push((at, segment));
        at = structure.toor(segment);
        if at == start || chain.len() > next.len() {
            break;
        }
    }
    chain
}

// The crossings of a loop in the order along it (the other loops), starting anywhere.
fn crossing_order(dual: &Dual, loop_id: LoopID) -> Vec<LoopID> {
    loop_chain(dual, loop_id)
        .into_iter()
        .map(|(intersection, _)| {
            let [a, b] = dual.intersection(intersection).loops;
            if a == loop_id { b } else { a }
        })
        .collect()
}

// Whether two sequences are the same up to a rotation.
fn same_cycle(a: &[LoopID], b: &[LoopID]) -> bool {
    a.len() == b.len()
        && (a.is_empty()
            || (0..b.len()).any(|shift| (0..a.len()).all(|i| a[i] == b[(i + shift) % b.len()])))
}

// A patch onto the unit square (see the module documentation): the position of every vertex of its triangles. The
// i-th side of the patch (in the order of `structure.edges(patch)`) goes from the i-th to the next corner of the square
// (0, 0), (1, 0), (1, 1), (0, 1).
fn tutte_square(layout: &Layout, patch: FaceKey<POLYCUBE>) -> Option<HashMap<VertID, [f64; 2]>> {
    const CORNERS: [[f64; 2]; 4] = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]];
    let mesh = &layout.granulated_mesh;
    let structure = &layout.polycube_ref.structure;
    let sides = structure.edges(patch).collect_vec();
    if sides.len() != 4 {
        return None;
    }
    let mut square: HashMap<VertID, [f64; 2]> = HashMap::new();
    for (i, &side) in sides.iter().enumerate() {
        let (a, b) = (CORNERS[i], CORNERS[(i + 1) % 4]);
        for (vert, t) in side_positions(layout, side)? {
            square.insert(vert, [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
        }
    }
    let faces = &layout.face_to_patch.get(&patch)?.faces;
    let inner = faces
        .iter()
        .flat_map(|&face| mesh.vertices(face).collect_vec())
        .filter(|vert| !square.contains_key(vert))
        .unique()
        .collect_vec();
    if inner.is_empty() {
        return Some(square);
    }
    // Every inner vertex at the mean of its neighbors: L x = b, with L the graph Laplacian on the inner vertices
    // (symmetric positive definite), solved with conjugate gradients (Jacobi preconditioned), for both coordinates.
    let index: HashMap<VertID, usize> = inner.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let mut neighbors = vec![vec![]; inner.len()];
    let mut degree = vec![0.; inner.len()];
    let mut rhs = vec![[0.; 2]; inner.len()];
    for (i, &vert) in inner.iter().enumerate() {
        for neighbor in mesh.neighbors(vert) {
            degree[i] += 1.;
            if let Some(&j) = index.get(&neighbor) {
                neighbors[i].push(j);
            } else if let Some(p) = square.get(&neighbor) {
                rhs[i][0] += p[0];
                rhs[i][1] += p[1];
            } else {
                // A neighbor outside the patch: the patch is not bounded by its sides.
                return None;
            }
        }
    }
    let apply = |x: &[f64], out: &mut [f64]| {
        for i in 0..x.len() {
            out[i] = degree[i] * x[i] - neighbors[i].iter().map(|&j| x[j]).sum::<f64>();
        }
    };
    for axis in 0..2 {
        let b = rhs.iter().map(|r| r[axis]).collect_vec();
        let mut x = vec![0.5; inner.len()];
        let mut ax = vec![0.; inner.len()];
        apply(&x, &mut ax);
        let mut r = b.iter().zip(&ax).map(|(b, a)| b - a).collect_vec();
        let mut z = r.iter().zip(&degree).map(|(r, d)| r / d).collect_vec();
        let mut p = z.clone();
        let mut rz: f64 = r.iter().zip(&z).map(|(r, z)| r * z).sum();
        let norm_b = b.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-300);
        for _ in 0..TUTTE_ITERATIONS {
            if r.iter().map(|v| v * v).sum::<f64>().sqrt() <= TUTTE_TOLERANCE * norm_b {
                break;
            }
            apply(&p, &mut ax);
            let alpha = rz / p.iter().zip(&ax).map(|(p, a)| p * a).sum::<f64>();
            for i in 0..x.len() {
                x[i] += alpha * p[i];
                r[i] -= alpha * ax[i];
            }
            for i in 0..z.len() {
                z[i] = r[i] / degree[i];
            }
            let rz_next: f64 = r.iter().zip(&z).map(|(r, z)| r * z).sum();
            let beta = rz_next / rz;
            rz = rz_next;
            for i in 0..p.len() {
                p[i] = z[i] + beta * p[i];
            }
        }
        for (i, &vert) in inner.iter().enumerate() {
            square.entry(vert).or_insert([0.; 2])[axis] = x[i];
        }
    }
    Some(square)
}

// The relative arc length of every vertex of the path of a side (a polycube half-edge), from its start (0) to its end
// (1). Computed once per polycube edge (along the half-edge with the smaller key), and mirrored (1 - t) for its twin,
// such that both patches of a side agree exactly on where its middle is; a vertex at the middle is moved slightly
// (along that half-edge), such that no middle line runs through it.
fn side_positions(layout: &Layout, side: EdgeKey<POLYCUBE>) -> Option<Vec<(VertID, f64)>> {
    let mesh = &layout.granulated_mesh;
    let twin = layout.polycube_ref.structure.twin(side);
    let canonical = if side.raw() < twin.raw() { side } else { twin };
    let path = layout.edge_to_path.get(&canonical)?;
    let lengths = path
        .windows(2)
        .scan(0., |sum, w| {
            *sum += (mesh.position(w[1]) - mesh.position(w[0])).norm();
            Some(*sum)
        })
        .collect_vec();
    let total = lengths.last().copied().unwrap_or(0.).max(1e-300);
    Some(
        path.iter()
            .enumerate()
            .map(|(k, &vert)| {
                let mut t = if k == 0 { 0. } else { lengths[k - 1] / total };
                if (t - 0.5).abs() < 2. * LEVEL_EPSILON {
                    t = 0.5 + 2. * LEVEL_EPSILON;
                }
                (vert, if canonical == side { t } else { 1. - t })
            })
            .collect(),
    )
}

// The middle line of a patch for the loop that enters it through the side `entry` and leaves through the side `exit`
// (opposite sides, half-edges of the patch), on the refined mesh: the crossings of its edges, from the entry side up to
// (not including) the exit side.
fn trace_in_patch(
    layout: &Layout,
    patch: FaceKey<POLYCUBE>,
    square: &HashMap<VertID, [f64; 2]>,
    entry: EdgeKey<POLYCUBE>,
    exit: EdgeKey<POLYCUBE>,
) -> Option<Vec<Crossing>> {
    let mesh = &layout.granulated_mesh;
    let structure = &layout.polycube_ref.structure;
    let sides = structure.edges(patch).collect_vec();
    let (i, j) = (
        sides.iter().position(|&s| s == entry)?,
        sides.iter().position(|&s| s == exit)?,
    );
    if (i + 2) % 4 != j {
        return None;
    }
    // Sides 0 and 2 run along x (the loop is the line x = 1/2), sides 1 and 3 along y.
    let axis = i % 2;
    let level = |vert: VertID| {
        let value = square.get(&vert).map_or(0.5, |p| p[axis]);
        if (value - 0.5).abs() < LEVEL_EPSILON {
            0.5 + LEVEL_EPSILON
        } else {
            value
        }
    };
    let faces = &layout.face_to_patch.get(&patch)?.faces;
    let crosses = |edge: EdgeID| (level(mesh.root(edge)) < 0.5) != (level(mesh.toor(edge)) < 0.5);
    let point = |edge: EdgeID| {
        let (a, b) = (mesh.root(edge), mesh.toor(edge));
        let (la, lb) = (level(a), level(b));
        let t = ((0.5 - la) / (lb - la)).clamp(0., 1.);
        mesh.position(a) + (mesh.position(b) - mesh.position(a)) * t
    };
    let entry_path = layout.edge_to_path.get(&entry)?;
    let exit_path = layout.edge_to_path.get(&exit)?;
    let on_exit: HashSet<VertID> = exit_path.iter().copied().collect();
    // The start: the edge of the entry side that the line crosses (as a half-edge of a face of the patch).
    let start = entry_path.windows(2).find_map(|w| {
        let (edge, twin) = mesh.edge_between_verts(w[0], w[1])?;
        let inside = if faces.contains(&mesh.face(edge)) {
            edge
        } else {
            twin
        };
        (crosses(inside) && faces.contains(&mesh.face(inside))).then_some(inside)
    })?;
    let mut crossings = vec![Crossing {
        from: mesh.face(mesh.twin(start)),
        to: mesh.face(start),
        point: point(start),
    }];
    let mut edge = start;
    for _ in 0..=faces.len() {
        let face = mesh.face(edge);
        let next = mesh.edges(face).find(|&e| e != edge && crosses(e))?;
        // The exit side (an edge of its path, with the next face outside the patch): the start of the next patch.
        let to = mesh.face(mesh.twin(next));
        if on_exit.contains(&mesh.root(next))
            && on_exit.contains(&mesh.toor(next))
            && !faces.contains(&to)
        {
            return Some(crossings);
        }
        crossings.push(Crossing {
            from: face,
            to,
            point: point(next),
        });
        edge = mesh.twin(next);
    }
    None
}

// A loop on the input mesh from its crossings of the edges of the refined mesh (in order, closed): where it moves from
// one input face to another, it crosses their shared edge. Back-and-forth crossings of the same edge are left out.
fn input_loop(
    mesh: &Mesh<INPUT>,
    regions: &RegionTracking,
    crossings: &[Crossing],
    direction: Direction,
) -> Option<Loop> {
    // The input crossings: the half-edge of the face left, and the position along it.
    let mut list: Vec<(EdgeID, f64)> = vec![];
    for crossing in crossings {
        let (Some(&from), Some(&to)) = (
            regions.face_input.get(&crossing.from),
            regions.face_input.get(&crossing.to),
        ) else {
            return None;
        };
        if from == to {
            continue;
        }
        let (edge, _) = mesh.edge_between_faces(from, to)?;
        let (a, b) = (
            mesh.position(mesh.root(edge)),
            mesh.position(mesh.toor(edge)),
        );
        let t =
            ((crossing.point - a).dot(&(b - a)) / (b - a).norm_squared()).clamp(1e-6, 1. - 1e-6);
        // Leaving a face through the edge it just entered by: both crossings are left out.
        if let Some(&(last, _)) = list.last()
            && mesh.twin(last) == edge
        {
            list.pop();
            continue;
        }
        list.push((edge, t));
    }
    // (Also around the start of the loop.)
    while list.len() >= 2 && mesh.twin(list[list.len() - 1].0) == list[0].0 {
        list.pop();
        list.remove(0);
    }
    // A loop passes through every face (and so crosses every edge) at most once: in long triangles, it may leave one
    // and come back into it later. The shorter part between the two visits (an excursion out of the face) is left out,
    // such that the loop runs straight through the face.
    while let Some((i, j)) = repeated_visit(mesh, &list) {
        // Visit `i` of a face is between crossings `i - 1` (in) and `i` (out); without crossings `i..j`, the loop
        // enters by the first and leaves by the last.
        let inner = j - i;
        if inner <= list.len() - inner {
            list.drain(i..j);
        } else {
            list.truncate(j);
            list.drain(..i);
        }
    }
    if list.len() < 2 {
        return None;
    }
    let edges = list
        .iter()
        .flat_map(|&(edge, _)| [edge, mesh.twin(edge)])
        .collect_vec();
    let offsets = list.iter().flat_map(|&(_, t)| [t, 1. - t]).collect_vec();
    // A loop crosses every edge at most once.
    if edges.iter().duplicates().next().is_some() {
        return None;
    }
    Some(Loop {
        edges,
        direction,
        offsets,
    })
}

// The first two visits of the same face in a (closed) list of crossings: their indices, with visit `k` the stay in the
// face between crossing `k - 1` (in) and crossing `k` (out).
fn repeated_visit(mesh: &Mesh<INPUT>, list: &[(EdgeID, f64)]) -> Option<(usize, usize)> {
    let mut seen: HashMap<FaceID, usize> = HashMap::new();
    for (k, &(edge, _)) in list.iter().enumerate() {
        // (The face that crossing `k` leaves.)
        if let Some(&first) = seen.get(&mesh.face(edge)) {
            return Some((first, k));
        }
        seen.insert(mesh.face(edge), k);
    }
    None
}

// The distance from a point to a triangle of the mesh (to its plane if the point projects inside it, else to its
// nearest vertex).
fn distance_to_triangle(mesh: &Mesh<INPUT>, face: FaceID, p: Vector3D) -> f64 {
    let [a, b, c] = mesh
        .vertices(face)
        .map(|v| mesh.position(v))
        .collect_array::<3>()
        .unwrap_or([p; 3]);
    let n = (b - a).cross(&(c - a));
    if n.norm_squared() > 0. {
        let q = p - n * ((p - a).dot(&n) / n.norm_squared());
        let inside = [(a, b), (b, c), (c, a)]
            .iter()
            .all(|&(u, v)| (v - u).cross(&(q - u)).dot(&n) >= 0.);
        if inside {
            return (p - q).norm();
        }
    }
    [a, b, c]
        .iter()
        .map(|v| (p - v).norm())
        .fold(f64::INFINITY, f64::min)
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use std::sync::Arc;

    // The medial loops of an evolved solution (on blub, or on the mesh in `DUALCUBE_TEST_MESH`) form the same loop
    // structure, and keep the layout (and so its quality).
    #[test]
    fn medial_loops_keep_the_structure() {
        let path = std::env::var("DUALCUBE_TEST_MESH").map_or_else(
            |_| {
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../mehsh/assets/blub001k.obj")
            },
            std::path::PathBuf::from,
        );
        let mesh = Arc::new(Mesh::<INPUT>::from_obj(&path).unwrap().0);
        // (On a coarse mesh such as blub, a medial loop may not fit its triangles: then another solution is tried.)
        let (evolved, medial) = (0..3)
            .find_map(|_| {
                let mut solution = Solution::new(mesh.clone());
                solution.initialize();
                let evolved = solution
                    .evolve(&EvolutionParams {
                        max_generations: 10,
                        ..EvolutionParams::default()
                    })
                    .ok()?
                    .optimize_layout(
                        &LayoutEvolutionParams {
                            max_generations: 20,
                            ..LayoutEvolutionParams::default()
                        },
                        &EvolutionMonitor::default(),
                    )
                    .ok()?;
                let medial = evolved.medial_loops()?;
                Some((evolved, medial))
            })
            .expect("medial loops");
        let timer = std::time::Instant::now();
        let _ = evolved.medial_loops();
        println!(
            "medial: {} loops, quality {:?} -> {:?} in {:?}",
            evolved.loops.len(),
            evolved.get_quality(),
            medial.get_quality(),
            timer.elapsed()
        );
        assert_eq!(medial.loops.len(), evolved.loops.len());
        // (Up to the order of summation.)
        let same = |a: Option<f64>, b: Option<f64>| (a.unwrap() - b.unwrap()).abs() < 1e-12;
        assert!(same(medial.get_quality(), evolved.get_quality()));
        let again = medial.medial_loops().expect("medial loops again");
        assert!(same(again.get_quality(), evolved.get_quality()));
    }
}
