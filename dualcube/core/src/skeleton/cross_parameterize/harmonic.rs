use std::collections::HashMap;

use faer::{
    prelude::Solve,
    sparse::{
        linalg::solvers::{Llt, SymbolicLlt},
        SparseColMat, Triplet,
    },
    Mat, Side,
};
use log::error;
use mehsh::prelude::Vector2D;
use petgraph::{graph::NodeIndex, visit::EdgeRef};

use super::virtual_mesh::VirtualFlatGeometry;


/// Solves the 2D Dirichlet (Tutte embedding) problem on the VFG graph.
///
/// Boundary nodes have fixed 2D positions. Interior nodes are solved for by
/// minimizing the Laplacian energy with inverse-distance weights (w_ij = 1 / len_ij).
///
/// Returns a map from every VFG node to its 2D position (boundary nodes are included
/// unchanged, interior nodes are filled in by the solve).
pub(super) fn solve_dirichlet(
    vfg: &VirtualFlatGeometry,
    boundary_positions: &HashMap<NodeIndex, Vector2D>,
) -> HashMap<NodeIndex, Vector2D> {
    // Separate free (interior) and boundary nodes.
    let all_nodes: Vec<NodeIndex> = vfg.graph.node_indices().collect();

    let mut free_indices: HashMap<NodeIndex, usize> = HashMap::new();
    let mut free_nodes: Vec<NodeIndex> = Vec::new();
    for &n in &all_nodes {
        if !boundary_positions.contains_key(&n) {
            free_indices.insert(n, free_nodes.len());
            free_nodes.push(n);
        }
    }

    let n_free = free_nodes.len();
    if n_free == 0 {
        return boundary_positions.clone();
    }

    // Build the sparse Laplacian (free×free) and RHS from boundary contributions.
    let mut triplets: Vec<Triplet<usize, usize, f64>> = Vec::new();
    let mut rhs_u = Mat::<f64>::zeros(n_free, 1);
    let mut rhs_v = Mat::<f64>::zeros(n_free, 1);

    for (row, &node) in free_nodes.iter().enumerate() {
        let mut diag = 0.0;

        for edge_ref in vfg.graph.edges(node) {
            let nbr = if edge_ref.source() == node {
                edge_ref.target()
            } else {
                edge_ref.source()
            };
            let len = edge_ref.weight().length;
            let w = if len > 1e-15 { 1.0 / len } else { 1e15 };

            if let Some(&col) = free_indices.get(&nbr) {
                triplets.push(Triplet::new(row, col, -w));
                diag += w;
            } else if let Some(&pos) = boundary_positions.get(&nbr) {
                diag += w;
                rhs_u[(row, 0)] += w * pos.x;
                rhs_v[(row, 0)] += w * pos.y;
            }
            // Neighbors not in the system (shouldn't happen for a well-formed VFG) are ignored.
        }

        triplets.push(Triplet::new(row, row, diag));
    }

    // Solve with sparse Cholesky.
    let lhs = match SparseColMat::<usize, f64>::try_new_from_triplets(n_free, n_free, &triplets) {
        Ok(m) => m,
        Err(e) => {
            error!("Failed to build sparse Laplacian for Dirichlet solve: {:?}", e);
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
            error!("Numeric LLT failed (matrix not SPD?): {:?}", e);
            return boundary_positions.clone();
        }
    };

    let x_u = llt.solve(rhs_u.as_ref());
    let x_v = llt.solve(rhs_v.as_ref());

    // Assemble results.
    let mut result = boundary_positions.clone();
    for (i, &n) in free_nodes.iter().enumerate() {
        result.insert(n, Vector2D::new(x_u[(i, 0)], x_v[(i, 0)]));
    }
    result
}
