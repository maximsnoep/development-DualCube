use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;

use faer::{
    prelude::Solve,
    sparse::{
        linalg::solvers::{Llt, SymbolicLlt},
        SparseColMat, Triplet,
    },
    Mat, Side,
};
use log::error;
use mehsh::prelude::{HasNeighbors, HasPosition, Mesh, Vector2D};

use crate::prelude::{VertID, INPUT};
use super::disk_boundary::{CutPath, CutSide};



/// Returns corner positions for a regular 2D polygon with `n_sides` sides,
/// inscribed in the unit circle.
pub(super) fn polygon_corners(n_sides: usize) -> Vec<Vector2D> {
    (0..n_sides)
        .map(|i| {
            let angle = 2.0 * PI * (i as f64) / (n_sides as f64);
            Vector2D::new(angle.cos(), angle.sin())
        })
        .collect()
}

/// Maps disk boundary segments to polygon sides.
///
/// Returns:
/// - `boundary_positions`: single-valued position for every boundary-loop and cut-endpoint vertex.
/// - `cut_dual_values`: for each cut-path interior vertex, its *two* polygon positions
///   (from the forward and backward traversals). Stored as `(VertID, Vector2D, Vector2D)`.
pub(super) fn parameterize_disk_to_polygon(
    segments: &[Vec<VertID>],
    degree: usize,
    mesh: &Mesh<INPUT>,
) -> (HashMap<VertID, Vector2D>, Vec<(VertID, Vector2D, Vector2D)>) {
    let n_sides = 4 * (degree - 1);
    let corners = polygon_corners(n_sides);

    let mut boundary_positions: HashMap<VertID, Vector2D> = HashMap::new();
    // Map from vertex to (forward_segment_idx, backward_segment_idx, forward_pos, backward_pos).
    // We'll figure out which is which after processing all segments.
    let mut vertex_segment_positions: HashMap<VertID, Vec<(usize, Vector2D)>> = HashMap::new();

    for (seg_idx, seg) in segments.iter().enumerate() {
        if seg.len() < 2 {
            // Degenerate segment (single vertex = a polygon corner).
            if let Some(&v) = seg.first() {
                let corner = corners[seg_idx % n_sides];
                boundary_positions.insert(v, corner);
            }
            continue;
        }

        let p0 = corners[seg_idx % n_sides];
        let p1 = corners[(seg_idx + 1) % n_sides];

        // Compute cumulative arc-lengths within this segment.
        let mut cumulative: Vec<f64> = vec![0.0; seg.len()];
        for i in 1..seg.len() {
            cumulative[i] = cumulative[i - 1] + (mesh.position(seg[i]) - mesh.position(seg[i - 1])).norm();
        }
        let seg_total = cumulative[seg.len() - 1];

        for (i, &v) in seg.iter().enumerate() {
            let t = if seg_total > 0.0 {
                (cumulative[i] / seg_total).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let pos = Vector2D::new(p0.x * (1.0 - t) + p1.x * t, p0.y * (1.0 - t) + p1.y * t);

            vertex_segment_positions
                .entry(v)
                .or_default()
                .push((seg_idx, pos));
        }
    }

    // Classify: vertices appearing once are single-valued boundary/endpoint vertices.
    // Vertices appearing twice are cut-interior vertices with two values.
    let mut cut_dual_values: Vec<(VertID, Vector2D, Vector2D)> = Vec::new();

    for (v, entries) in &vertex_segment_positions {
        if entries.len() == 1 {
            boundary_positions.insert(*v, entries[0].1);
        } else if entries.len() == 2 {
            // Two appearances: one forward traversal, one backward.
            // The one with the lower segment index is "forward".
            let (pos_a, pos_b) = if entries[0].0 < entries[1].0 {
                (entries[0].1, entries[1].1)
            } else {
                (entries[1].1, entries[0].1)
            };
            cut_dual_values.push((*v, pos_a, pos_b));
            // Also insert a default into boundary_positions (for any non-cut-aware code paths).
            boundary_positions.insert(*v, pos_a);
        } else if entries.len() > 2 {
            // Vertex appears in 3+ segments: it's a junction vertex (polygon corner).
            // Use the first occurrence.
            boundary_positions.insert(*v, entries[0].1);
        }
    }

    (boundary_positions, cut_dual_values)
}


/// Determines which side of a cut path a neighbor vertex is on.
///
/// At `cut_vertex` with predecessor `prev` and successor `next` along the cut,
/// we walk the rotational neighbor order from `prev` to `next`. If `query` is
/// encountered in that sweep, it's on the Forward side; otherwise Backward.
fn classify_cut_side(
    query: VertID,
    cut_vertex: VertID,
    prev_on_cut: VertID,
    next_on_cut: VertID,
    mesh: &Mesh<INPUT>,
) -> CutSide {
    let neighbors: Vec<VertID> = mesh.neighbors(cut_vertex).collect();
    let n = neighbors.len();
    let Some(prev_pos) = neighbors.iter().position(|&v| v == prev_on_cut) else {
        return CutSide::Forward; // fallback
    };
    let Some(next_pos) = neighbors.iter().position(|&v| v == next_on_cut) else {
        return CutSide::Forward;
    };
    let Some(query_pos) = neighbors.iter().position(|&v| v == query) else {
        return CutSide::Forward;
    };

    // Walk from prev to next (exclusive of both endpoints).
    let mut i = (prev_pos + 1) % n;
    while i != next_pos {
        if i == query_pos {
            return CutSide::Forward;
        }
        i = (i + 1) % n;
    }
    CutSide::Backward
}

/// Solves the 2D Dirichlet problem for a region, handling dual-valued cut vertices.
///
/// For boundary-loop vertices and cut endpoints: single fixed position from `boundary_positions`.
/// For cut-interior vertices: two positions; the correct one is chosen based on which
/// side of the cut each free neighbor is on.
///
/// `cross_boundary_positions` supplies UV for vertices from *adjacent* patches that
/// are connected to our patch through boundary-loop half-edges. Including them ensures
/// the Laplacian accounts for every mesh edge incident on free vertices.
pub(super) fn solve_harmonic_2d_with_cuts(
    all_vertices: &[VertID],
    boundary_positions: &HashMap<VertID, Vector2D>,
    cut_dual_values: &[(VertID, Vector2D, Vector2D)],
    cut_paths: &[CutPath],
    cross_boundary_positions: &HashMap<VertID, Vector2D>,
    mesh: &Mesh<INPUT>,
) -> HashMap<VertID, Vector2D> {
    // Combine our-side boundary and cross-boundary into a single set.
    let all_boundary: HashSet<VertID> = boundary_positions
        .keys()
        .chain(cross_boundary_positions.keys())
        .copied()
        .collect();

    let mut free_mapping: HashMap<VertID, usize> = HashMap::new();
    let mut free_vertices: Vec<VertID> = Vec::new();
    for &v in all_vertices {
        if !all_boundary.contains(&v) {
            free_mapping.insert(v, free_vertices.len());
            free_vertices.push(v);
        }
    }

    let n_free = free_vertices.len();
    if n_free == 0 {
        let mut result = boundary_positions.clone();
        result.extend(cross_boundary_positions);
        return result;
    }

    // Pre-compute side-aware boundary values for cut-interior vertices.
    // Key: (free_vertex, boundary_neighbor) -> correct Vector2D to use.
    let dual_set: HashMap<VertID, (Vector2D, Vector2D)> = cut_dual_values
        .iter()
        .map(|&(v, fwd, bwd)| (v, (fwd, bwd)))
        .collect();

    // For side classification, we need prev/next on cut for each interior vertex.
    let mut cut_context: HashMap<VertID, (VertID, VertID)> = HashMap::new();
    for cut in cut_paths {
        for i in 1..cut.path.len().saturating_sub(1) {
            cut_context.insert(cut.path[i], (cut.path[i - 1], cut.path[i + 1]));
        }
    }

    let mut override_positions: HashMap<(VertID, VertID), Vector2D> = HashMap::new();
    for (&v, &(fwd, bwd)) in &dual_set {
        if let Some(&(prev, next)) = cut_context.get(&v) {
            for nbr in mesh.neighbors(v) {
                if free_mapping.contains_key(&nbr) {
                    let side = classify_cut_side(nbr, v, prev, next, mesh);
                    let pos = match side {
                        CutSide::Forward => fwd,
                        CutSide::Backward => bwd,
                    };
                    override_positions.insert((nbr, v), pos);
                }
            }
        }
    }

    // Build Laplacian and RHS.
    let mut triplets = Vec::new();
    let mut rhs_u = Mat::<f64>::zeros(n_free, 1);
    let mut rhs_v = Mat::<f64>::zeros(n_free, 1);

    for (row_idx, &mesh_key) in free_vertices.iter().enumerate() {
        let neighbors: Vec<VertID> = mesh.neighbors(mesh_key).collect();

        // Only count neighbours that are part of our system (free or boundary).
        // Cross-patch neighbours NOT in all_boundary would inflate the diagonal.
        let mut degree: f64 = 0.0;

        for &nbr in &neighbors {
            if let Some(&col_idx) = free_mapping.get(&nbr) {
                degree += 1.0;
                triplets.push(Triplet::new(row_idx, col_idx, -1.0));
            } else if all_boundary.contains(&nbr) {
                degree += 1.0;
                let pos = override_positions
                    .get(&(mesh_key, nbr))
                    .or_else(|| boundary_positions.get(&nbr))
                    .or_else(|| cross_boundary_positions.get(&nbr));
                if let Some(pos) = pos {
                    rhs_u[(row_idx, 0)] += pos.x;
                    rhs_v[(row_idx, 0)] += pos.y;
                }
            }
            // Neighbours outside both sets are ignored (and not counted in degree).
        }

        triplets.push(Triplet::new(row_idx, row_idx, degree));
    }

    let lhs = match SparseColMat::<usize, f64>::try_new_from_triplets(n_free, n_free, &triplets) {
        Ok(m) => m,
        Err(e) => {
            error!("Failed to build sparse matrix for harmonic 2D: {:?}", e);
            return boundary_positions.clone();
        }
    };
    let symbolic = match SymbolicLlt::try_new(lhs.symbolic(), Side::Lower) {
        Ok(s) => s,
        Err(e) => {
            error!("Symbolic LLT failed: {:?}", e);
            return boundary_positions.clone();
        }
    };
    let llt = match Llt::try_new_with_symbolic(symbolic, lhs.as_ref(), Side::Lower) {
        Ok(f) => f,
        Err(e) => {
            error!("Numeric LLT failed (not SPD?): {:?}", e);
            return boundary_positions.clone();
        }
    };

    let x_u = llt.solve(rhs_u.as_ref());
    let x_v = llt.solve(rhs_v.as_ref());

    let mut result: HashMap<VertID, Vector2D> = boundary_positions.clone();
    result.extend(cross_boundary_positions);
    for (i, &v) in free_vertices.iter().enumerate() {
        result.insert(v, Vector2D::new(x_u[(i, 0)], x_v[(i, 0)]));
    }
    result
}

/// Simple 2D Dirichlet solve (no cuts, no cross-boundary). Used for degree-1 regions.
pub(super) fn solve_harmonic_2d(
    all_vertices: &[VertID],
    boundary_positions: &HashMap<VertID, Vector2D>,
    cross_boundary_positions: &HashMap<VertID, Vector2D>,
    mesh: &Mesh<INPUT>,
) -> HashMap<VertID, Vector2D> {
    solve_harmonic_2d_with_cuts(all_vertices, boundary_positions, &[], &[], cross_boundary_positions, mesh)
}
