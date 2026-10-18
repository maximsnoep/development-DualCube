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
//! As post-processing (see `Solution::smooth_loops_in_patches`, not used by the optimization), the loops can be smoothed
//! within the patches: where a loop crosses a side, it may slide along the side (towards the middle of its neighboring
//! crossings on the surface, see `smooth_crossings`), and in every patch it runs straight (in the square) from where it
//! enters to where it leaves. Two such segments between opposite sides of the square still cross exactly once, so the
//! loop structure stays the same.
//!
//! The loops are traced on the refined mesh, and become loops on the input mesh by where they cross its edges (with
//! straight chords in between, inside its triangles): two curves inside a triangle cross an odd number of times
//! exactly if their chords cross, so the loop structure is kept (checked).

use crate::prelude::*;
use slotmap::SlotMap;

// The values of the level sets are kept this far from those of the vertices (such that no level set passes through
// a vertex).
const LEVEL_EPSILON: f64 = 1e-9;
// The corners of the square of a patch (see `tutte_square`).
const SQUARE: [[f64; 2]; 4] = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]];
// The smoothing of the crossings of the sides (see `smooth_crossings`): its rounds, how far a crossing moves to its
// target per round, and how far (relative) the crossings stay from the ends of their sides.
const SMOOTHING_ROUNDS: usize = 30;
const SMOOTHING_STEP: f64 = 0.5;
const SIDE_MARGIN: f64 = 0.1;
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
        self.traced_loops(false)
    }

    /// See `medial_loops`, with the loops smoothed within the patches (see the module documentation): post-processing,
    /// for the solutions that are shown or returned (the layout, and so the quality, stay the same).
    #[must_use]
    pub fn smooth_loops_in_patches(&self) -> Option<Self> {
        self.traced_loops(true)
    }

    // The loops traced through the patches: through the middles of their sides, or (`smooth`) through the smoothed
    // crossings of the sides (see `smooth_crossings`).
    fn traced_loops(&self, smooth: bool) -> Option<Self> {
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
            let mut patch_of: HashMap<LoopIntersectionID, FaceKey<POLYCUBE>> = HashMap::default();
            let mut side_of: HashMap<LoopSegmentID, EdgeKey<POLYCUBE>> = HashMap::default();
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
            .with_pool(pool())
            .map(|patch| {
                let square = tutte_square(layout, patch)?;
                // (Smoothed: straight lines in the square are smoother on the surface with mean value weights.)
                Some((patch, if smooth { mean_value_square(layout, square) } else { square }))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .collect::<Option<HashMap<_, _>>>()?;

        // Every loop: its patches in its order, each with the sides through which it enters and leaves.
        let mut passes = vec![];
        for loop_id in self.loops.keys() {
            let chain = loop_chain(dual, loop_id);
            if chain.len() < 2 {
                return None;
            }
            let mut pass = vec![];
            for k in 0..chain.len() {
                // (The segment that arrives at the intersection, as one that leaves it: its twin.)
                let (intersection, exit) = chain[k];
                let (_, arrival) = chain[(k + chain.len() - 1) % chain.len()];
                let entry = *side_of.get(&loop_structure.twin(arrival))?;
                pass.push((patch_of[&intersection], entry, *side_of.get(&exit)?));
            }
            passes.push((loop_id, pass));
        }
        // Where the loops cross the sides (along their canonical half-edges): the middles, or smoothed.
        let positions = if smooth {
            smooth_crossings(layout, &passes)?
        } else {
            HashMap::default()
        };
        let at = |side: EdgeKey<POLYCUBE>| {
            let twin = structure.twin(side);
            if side.raw() < twin.raw() {
                positions.get(&side).copied().unwrap_or(0.5)
            } else {
                1. - positions.get(&twin).copied().unwrap_or(0.5)
            }
        };

        // Every loop: through its patches, on the refined mesh, then on the input mesh.
        let mut loops: SlotMap<LoopID, Loop> = self.loops.clone();
        for (loop_id, pass) in &passes {
            let mut crossings = vec![];
            for &(patch, entry, exit) in pass {
                crossings.extend(trace_in_patch(
                    layout,
                    patch,
                    &squares[&patch],
                    (entry, at(entry)),
                    (exit, at(exit)),
                )?);
            }
            let old = &self.loops[*loop_id];
            loops[*loop_id] = input_loop(&self.mesh_ref, regions, &crossings, old.direction)?;
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
        let mut pieces: HashMap<FaceID, Vec<FaceID>> = HashMap::default();
        for (&piece, &input) in &regions.face_input {
            pieces.entry(input).or_default().push(piece);
        }
        let mut patch_of_face: HashMap<FaceID, FaceKey<POLYCUBE>> = HashMap::default();
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

// Where the loops cross the sides of the patches (the relative position along every polycube edge, in the direction of
// its half-edge with the smaller key, see `side_positions`): their middles, smoothed. Every crossing moves, every round,
// towards the position on its side closest to the middle of the crossings before and after it along its loop
// (Laplacian smoothing on the surface, constrained to the sides), and stays away from the ends of its side.
#[allow(clippy::type_complexity)]
fn smooth_crossings(
    layout: &Layout,
    passes: &[(LoopID, Vec<(FaceKey<POLYCUBE>, EdgeKey<POLYCUBE>, EdgeKey<POLYCUBE>)>)],
) -> Option<HashMap<EdgeKey<POLYCUBE>, f64>> {
    let mesh = &layout.granulated_mesh;
    let structure = &layout.polycube_ref.structure;
    let canonical = |side: EdgeKey<POLYCUBE>| {
        let twin = structure.twin(side);
        if side.raw() < twin.raw() { side } else { twin }
    };
    // Every side (canonical): its path as points with their relative positions, increasing.
    let mut sides: HashMap<EdgeKey<POLYCUBE>, Vec<(f64, Vector3D)>> = HashMap::default();
    for (_, pass) in passes {
        for &(_, _, exit) in pass {
            let side = canonical(exit);
            if !sides.contains_key(&side) {
                let mut points = side_positions(layout, side)?
                    .into_iter()
                    .map(|(vert, t)| (t, mesh.position(vert)))
                    .collect_vec();
                points.sort_by(|a, b| a.0.total_cmp(&b.0));
                sides.insert(side, points);
            }
        }
    }
    let point = |side: EdgeKey<POLYCUBE>, at: f64| {
        let points = &sides[&side];
        let k = points.partition_point(|&(t, _)| t < at).clamp(1, points.len() - 1);
        let ((t0, p0), (t1, p1)) = (points[k - 1], points[k]);
        p0 + (p1 - p0) * ((at - t0) / (t1 - t0).max(1e-300)).clamp(0., 1.)
    };
    let closest = |side: EdgeKey<POLYCUBE>, target: Vector3D| {
        sides[&side]
            .windows(2)
            .map(|w| {
                let ((t0, p0), (t1, p1)) = (w[0], w[1]);
                let d = p1 - p0;
                let s = ((target - p0).dot(&d) / d.norm_squared().max(1e-300)).clamp(0., 1.);
                ((p0 + d * s - target).norm(), t0 + s * (t1 - t0))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map_or(0.5, |(_, at)| at)
    };
    let mut positions: HashMap<EdgeKey<POLYCUBE>, f64> = sides.keys().map(|&side| (side, 0.5)).collect();
    // The crossings of every loop, in order: the sides through which it leaves its patches.
    let orders = passes
        .iter()
        .map(|(_, pass)| pass.iter().map(|&(_, _, exit)| canonical(exit)).collect_vec())
        .collect_vec();
    for _ in 0..SMOOTHING_ROUNDS {
        for order in &orders {
            let n = order.len();
            if n < 3 {
                continue;
            }
            for k in 0..n {
                let (before, after) = (order[(k + n - 1) % n], order[(k + 1) % n]);
                let target = (point(before, positions[&before]) + point(after, positions[&after])) / 2.;
                let side = order[k];
                let at = positions[&side];
                let moved = at + SMOOTHING_STEP * (closest(side, target) - at);
                positions.insert(side, moved.clamp(SIDE_MARGIN, 1. - SIDE_MARGIN));
            }
        }
    }
    Some(positions)
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
    let mesh = &layout.granulated_mesh;
    let structure = &layout.polycube_ref.structure;
    let sides = structure.edges(patch).collect_vec();
    if sides.len() != 4 {
        return None;
    }
    let mut square: HashMap<VertID, [f64; 2]> = HashMap::default();
    for (i, &side) in sides.iter().enumerate() {
        let (a, b) = (SQUARE[i], SQUARE[(i + 1) % 4]);
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

// A patch onto the unit square with mean value weights (Floater): every inner vertex at the weighted mean of its
// neighbors, with weights (tan(a/2) + tan(b/2)) / length for the angles a and b at the vertex next to the edge to the
// neighbor. Positive weights, so still bijective; closer to the shape of the patch than the Tutte embedding (uniform
// weights), which is where it starts (successive over-relaxation, until it hardly changes).
fn mean_value_square(layout: &Layout, mut square: HashMap<VertID, [f64; 2]>) -> HashMap<VertID, [f64; 2]> {
    const SWEEPS: usize = 2000;
    const RELAXATION: f64 = 1.;
    const CONVERGED: f64 = 1e-9;
    let mesh = &layout.granulated_mesh;
    // The inner vertices (those with all their faces in the patch): not on a side.
    let sides: HashSet<VertID> = layout
        .edge_to_path
        .values()
        .flatten()
        .copied()
        .collect();
    let inner = square.keys().copied().filter(|v| !sides.contains(v)).collect_vec();
    let angle = |at: VertID, a: VertID, b: VertID| {
        (mesh.position(a) - mesh.position(at)).angle(&(mesh.position(b) - mesh.position(at)))
    };
    // The weights of the neighbors of every inner vertex.
    let weights = inner
        .iter()
        .map(|&v| {
            mesh.neighbors(v)
                .filter_map(|n| {
                    let (edge, twin) = mesh.edge_between_verts(v, n)?;
                    let other = |e: EdgeID| mesh.vertices(mesh.face(e)).find(|&w| w != v && w != n);
                    let (a, b) = (other(edge)?, other(twin)?);
                    let length = (mesh.position(n) - mesh.position(v)).norm().max(1e-300);
                    let w = ((angle(v, n, a) / 2.).tan() + (angle(v, n, b) / 2.).tan()) / length;
                    (w.is_finite() && w > 0.).then_some((n, w))
                })
                .collect_vec()
        })
        .collect_vec();
    // The sweeps on dense arrays (no hashing): the positions of all vertices of the square (the inner ones first), and
    // per inner vertex its neighbors (indices into the positions; those outside the square at the middle).
    let mut index: HashMap<VertID, usize> = inner.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let mut positions = inner.iter().map(|v| square[v]).collect_vec();
    for (&v, &p) in &square {
        if let std::collections::hash_map::Entry::Vacant(entry) = index.entry(v) {
            entry.insert(positions.len());
            positions.push(p);
        }
    }
    let outside = positions.len();
    positions.push([0.5; 2]);
    let neighbors = weights
        .iter()
        .map(|neighbors| {
            let total: f64 = neighbors.iter().map(|(_, w)| w).sum();
            let indexed = neighbors
                .iter()
                .map(|&(n, w)| (index.get(&n).copied().unwrap_or(outside), w))
                .collect_vec();
            (indexed, total)
        })
        .collect_vec();
    for _ in 0..SWEEPS {
        let mut change: f64 = 0.;
        for (i, (neighbors, total)) in neighbors.iter().enumerate() {
            let total = *total;
            if total <= 0. {
                continue;
            }
            let mut mean = [0.; 2];
            for &(n, w) in neighbors {
                let p = positions[n];
                mean[0] += w * p[0] / total;
                mean[1] += w * p[1] / total;
            }
            let p = &mut positions[i];
            for axis in 0..2 {
                let next = p[axis] + RELAXATION * (mean[axis] - p[axis]);
                change = change.max((next - p[axis]).abs());
                p[axis] = next;
            }
        }
        if change < CONVERGED {
            break;
        }
    }
    for (v, p) in inner.iter().zip(positions) {
        square.insert(*v, p);
    }
    square
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

// The line of a patch for the loop that enters it through the side `entry` and leaves through the side `exit` (opposite
// sides, half-edges of the patch, each with the relative position along it, see `side_positions`; at their middles, the
// middle line): the straight segment between those points in the square, on the refined mesh: the crossings of its
// edges, from the entry side up to (not including) the exit side.
fn trace_in_patch(
    layout: &Layout,
    patch: FaceKey<POLYCUBE>,
    square: &HashMap<VertID, [f64; 2]>,
    (entry, entry_at): (EdgeKey<POLYCUBE>, f64),
    (exit, exit_at): (EdgeKey<POLYCUBE>, f64),
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
    // The segment from the entry to the exit point in the square; `level` is on which side of it a vertex lies (for the
    // middle lines, exactly 1/2 minus its x or y, with the same choice for a vertex on the line as before).
    let on_side = |side: usize, at: f64| {
        let (a, b) = (SQUARE[side], SQUARE[(side + 1) % 4]);
        [a[0] + at * (b[0] - a[0]), a[1] + at * (b[1] - a[1])]
    };
    let (from, to) = (on_side(i, entry_at), on_side(j, exit_at));
    let level = |vert: VertID| {
        let q = square.get(&vert).copied().unwrap_or(from);
        let value = (to[0] - from[0]) * (q[1] - from[1]) - (to[1] - from[1]) * (q[0] - from[0]);
        if value.abs() < LEVEL_EPSILON {
            -LEVEL_EPSILON
        } else {
            value
        }
    };
    let faces = &layout.face_to_patch.get(&patch)?.faces;
    let crosses = |edge: EdgeID| (level(mesh.root(edge)) < 0.) != (level(mesh.toor(edge)) < 0.);
    let point = |edge: EdgeID| {
        let (a, b) = (mesh.root(edge), mesh.toor(edge));
        let (la, lb) = (level(a), level(b));
        let t = (la / (la - lb)).clamp(0., 1.);
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
    let mut seen: HashMap<FaceID, usize> = HashMap::default();
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

    // The medial loops of an annealed solution (on blub, or on the mesh in `DUALCUBE_TEST_MESH`) form the same loop
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
        let (annealed, medial) = (0..3)
            .find_map(|_| {
                let mut solution = Solution::new(mesh.clone());
                solution.initialize();
                let annealed = solution
                    .anneal(
                        &AnnealParams {
                            steps: 10,
                            ..AnnealParams::default()
                        },
                        &OptimizationMonitor::default(),
                    )
                    .ok()?
                    .optimize_layout(
                        &LayoutEvolutionParams {
                            max_generations: 20,
                            ..LayoutEvolutionParams::default()
                        },
                        &OptimizationMonitor::default(),
                    )
                    .ok()?;
                let medial = annealed.medial_loops()?;
                Some((annealed, medial))
            })
            .expect("medial loops");
        let timer = std::time::Instant::now();
        let _ = annealed.medial_loops();
        println!(
            "medial: {} loops, quality {:?} -> {:?} in {:?}",
            annealed.loops.len(),
            annealed.get_quality(),
            medial.get_quality(),
            timer.elapsed()
        );
        assert_eq!(medial.loops.len(), annealed.loops.len());
        // (Up to the order of summation.)
        let same = |a: Option<f64>, b: Option<f64>| (a.unwrap() - b.unwrap()).abs() < 1e-12;
        assert!(same(medial.get_quality(), annealed.get_quality()));
        let again = medial.medial_loops().expect("medial loops again");
        assert!(same(again.get_quality(), annealed.get_quality()));
        // The same layout, so the same quad mesh (its patches on the same faces of the polycube).
        let quad = |solution: &Solution| {
            let mut solution = solution.clone();
            solution.construct_quad(QuadDensity::Fixed(3)).unwrap();
            let mesh = solution.quad.unwrap().quad_mesh;
            let sum = mesh
                .vert_ids()
                .iter()
                .map(|&v| mesh.position(v))
                .sum::<Vector3D>();
            (mesh.nr_faces(), sum)
        };
        let ((faces, sum), (medial_faces, medial_sum)) = (quad(&annealed), quad(&medial));
        assert_eq!(faces, medial_faces);
        assert!((sum - medial_sum).norm() < 1e-6 * sum.norm().max(1.));
    }
}
