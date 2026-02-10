use mehsh::prelude::Mesh;
use std::collections::HashSet;
use crate::prelude::INPUT;
use crate::skeleton::curve_skeleton::{CurveSkeleton, CurveSkeletonManipulation};

/// Ensures necessary conditions for orthogonal embeddability on the curve skeleton:
/// - No cycles of length 3
/// - All vertex degrees <= 6
/// 
/// Note that these conditions are necessary but not sufficient for orthogonal embeddability.
pub fn make_embedding_possible(skeleton: &mut CurveSkeleton, mesh: &Mesh<INPUT>) {
    // Remove all cycles of 3 nodes by subdividing an edge.
    let mut found_triangle = true;
    while found_triangle {
        found_triangle = false;
        
        // Collect a snapshot of edges to iterate safely while mutating the graph.
        let edges: Vec<_> = skeleton.edge_indices().collect();

        for e in edges {
            // If edge was removed during previous operations, skip.
            let (a, b) = match skeleton.edge_endpoints(e) {
                Some(ep) => ep,
                None => continue,
            };

            // Gather common neighbors of a and b -> triangles (a,b,common)
            let neighbors_a: HashSet<_> = skeleton.neighbors(a).collect();
            let neighbors_b: Vec<_> = skeleton.neighbors(b).collect();

            for common in neighbors_b {
                if common == a || common == b {
                    continue;
                }
                if !neighbors_a.contains(&common) {
                    continue;
                }

                // We found a cycle of length 3.
                found_triangle = true;

                // Compute patch sizes for each node in the cycle.
                let size_a = skeleton.node_weight(a).map(|w| w.patch_vertices.len()).unwrap_or(0);
                let size_b = skeleton.node_weight(b).map(|w| w.patch_vertices.len()).unwrap_or(0);
                let size_c = skeleton.node_weight(common).map(|w| w.patch_vertices.len()).unwrap_or(0);

                // For each edge in the cycle, consider the combined patch size over edge endpoints.
                let sum_ab = size_a + size_b;
                let sum_ac = size_a + size_c;
                let sum_bc = size_b + size_c;

                // Try subdividing the edge with the largest combined patch size first.
                let mut pairs = vec![((a, b), sum_ab), ((a, common), sum_ac), ((b, common), sum_bc)];
                pairs.sort_by(|x, y| y.1.cmp(&x.1)); // descending by sum

                let mut handled = false;
                for ((u, v), _) in pairs {
                    if let Some(edge_idx) = skeleton.find_edge(u, v) {
                        if skeleton.subdivide_edge(edge_idx, mesh) {
                            handled = true;
                            break;
                        }
                    }
                }

                if !handled {
                    // TODO: implement a better fallback case, though all three edges not being subdividable 
                    // should not happen in practice.
                    panic!("Failed to subdivide edge in triangle, this should not happen");
                }

                // Restart outer loop scanning from the beginning after we mutated the graph.
                break;
            }

            if found_triangle {
                break;
            }
        }
    }

    // Split all vertices whose degree exceeds 6. Repeat until no splits occur.
    let mut any_split = true;
    while any_split {
        any_split = false;
        let nodes: Vec<_> = skeleton.node_indices().collect();
        for node in nodes {
            let degree = skeleton.neighbors(node).count();

            if degree > 6 {
                if skeleton.split_high_degree_node(node, mesh) {
                    any_split = true;
                }
            }
        }
    }
}