use std::collections::HashSet;

use log::info;
use mehsh::prelude::{HasPosition, HasVertices, Mesh, Vector3D, EPS};
use petgraph::graph::NodeIndex;

use crate::{
    prelude::{CurveSkeleton, INPUT},
    skeleton::curve_skeleton::{CurveSkeletonManipulation, CurveSkeletonSpatial, MergeBehavior},
};

// TODO: Maybe instead of simplifying everything possible, it might be better to simplify only to make regions closer to cubes

// TODO: Instead of just degree 2 nodes, merge nodes when possible and improves some cost

/// Simplifies a skeleton by removing nodes which are not crucial for structure:
/// when a node can be removed and the resulting edge is still within the mesh, it can be removed
/// We do this iteratively until no more nodes can be removed.
pub fn simplify_skeleton(skeleton: &mut CurveSkeleton, original_mesh: &Mesh<INPUT>) {
    // Cache intersection tests that came up negative. Valid as nodes never move.
    let mut intersection_cache: HashSet<(NodeIndex, NodeIndex)> = HashSet::new();

    // Look for a node of degree 2 that can be dissolved
    // Higher degree nodes are needed for structure and lower degree nodes are endpoints.
    let mut changed = true;
    while changed {
        changed = false;

        // Get all possible candidates (degree 2 nodes)
        let candidates: Vec<NodeIndex> = skeleton
            .node_indices()
            .filter(|&i| skeleton.neighbors(i).count() == 2)
            .collect();

        // Remove the first node that can be dissolved
        // TODO: investigate ordering this better. Not sure if/how much it matters.
        for node_index in candidates {
            let mut neighbors: Vec<NodeIndex> = skeleton.neighbors(node_index).collect();
            // Sort neighbors to ensure consistent ordering in cache
            if neighbors[0] > neighbors[1] {
                neighbors.swap(0, 1);
            }

            let pos_a = skeleton.node_weight(neighbors[0]).unwrap().position;
            let pos_b = skeleton.node_weight(neighbors[1]).unwrap().position;

            // Make sure a and b are not directly connected
            if skeleton.find_edge(neighbors[0], neighbors[1]).is_some() {
                continue;
            }

            // Check Intersection, first checking cache
            if intersection_cache.contains(&(neighbors[0], neighbors[1])) {
                continue;
            }

            // If the segment A->B hits the mesh, we cannot remove 'node' as it preserves geometry.
            if segment_intersects_mesh(pos_a, pos_b, original_mesh) {
                intersection_cache.insert((neighbors[0], neighbors[1]));
                continue;
            }

            // If we reach here, the node can be dissolved
            skeleton.dissolve_subdivision(node_index, original_mesh);

            // Change one node at a time.
            changed = true;
            break;
        }
    }

    // TODO: something like embedding refinement but that makes sure edges stay within the mesh.
    // Maybe something like moving along locked axes?
    // Maybe something that iteratively tries moving towards centroid until intersections happen?
    // Though ideally, intersections would result in close to 90 degree angles, this likely needs to be accounted for...
}

/// Tries to make all patches have a convexity score above the target by merging adjacent patches, and by splitting patches.
pub fn convexify(
    skeleton: &mut CurveSkeleton,
    original_mesh: &Mesh<INPUT>,
    target_convexity: f64,
    merge_threshold: f64,
) {
    // First we simply look to make regions more convex by merging.
    // At the ends of tubes/cubes we often get corner patches, these can safely be merged into their parent to achieve same/better convexity with less patches.
    // TODO: do this subtree at a time, instead of leaf at a time. One leaf might decrease convexity, but the subtree as a whole might be convex.
    // TODO: this can be made more sophisticated in general, not just leaves..

    let mut changed = true;
    while changed {
        changed = false;

        // Get all possible leaves to merge
        let leaves: Vec<NodeIndex> = skeleton
            .node_indices()
            .filter(|&i| skeleton.neighbors(i).count() == 1)
            .collect();

        // Get volumes for all leaves
        let mut leaf_data: Vec<(NodeIndex, f64)> = leaves
            .iter()
            .map(|&leaf| {
                let volume = skeleton.patch_volume(leaf, original_mesh);
                (leaf, volume)
            })
            .collect();

        // Sort by volume
        leaf_data.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Try merging all leaves with their parent, starting with the smallest.
        for (leaf, _) in leaf_data {
            let score = skeleton.patch_convexity_score(leaf, original_mesh);
            let parent = skeleton.neighbors(leaf).next().unwrap();

            // Check if merging would improve convexity enough, or if the new region by itself would already be convex enough.
            let merged_score = skeleton.patches_convexity_score(&[leaf, parent], original_mesh);
            if merged_score >= score * merge_threshold || merged_score >= target_convexity {
                skeleton.merge_nodes(leaf, parent, MergeBehavior::SourceIntoTarget);
                if merged_score >= target_convexity{}
                info!(
                    "Merged {:?} into {:?}, changed convexity from {:.3} to {:.3}",
                    leaf, parent, score, merged_score
                );

                // Change one node at a time.
                changed = true;
                break;
            }
        }

        if !changed {
            break;
        }
    }

    // TODO: Redraw boundaries to achieve better convexity... This likely needs some engineering as high-degree nodes can have many changes at once?
}

/// Checks if a line segment from `p0` to `p1` intersects any triangle in the mesh.
///
/// Note that is a naive O(n) implementation that checks each triangle one at a time.
/// This could be optimized using a spatial data structure like an octree.
fn segment_intersects_mesh(p0: Vector3D, p1: Vector3D, mesh: &Mesh<INPUT>) -> bool {
    let direction = p1 - p0;
    let segment_length_sq = direction.norm_squared();

    // Avoid division by zero for degenerate segments
    if segment_length_sq < EPS {
        return false;
    }

    for face_id in mesh.face_ids_iter() {
        // Get vertices of this face
        let verts: Vec<_> = mesh.vertices(face_id).map(|v| mesh.position(v)).collect();

        // For faces with more than 3 vertices, use fan triangulation
        for i in 1..verts.len() - 1 {
            let v0 = verts[0];
            let v1 = verts[i];
            let v2 = verts[i + 1];

            if segment_intersects_triangle(p0, p1, v0, v1, v2) {
                return true;
            }
        }
    }

    false
}

/// Moller–Trumbore algorithm to check if a segment intersects a triangle.
/// Returns true if the segment from `p` to `q` intersects the triangle (t0, t1, t2).
fn segment_intersects_triangle(
    p: Vector3D,
    q: Vector3D,
    t0: Vector3D,
    t1: Vector3D,
    t2: Vector3D,
) -> bool {
    let direction = q - p;
    let edge1 = t1 - t0;
    let edge2 = t2 - t0;

    let h = direction.cross(&edge2);
    let a = edge1.dot(&h);

    // Ray is parallel to the triangle
    if a.abs() < EPS {
        return false;
    }

    let f = 1.0 / a;
    let s = p - t0;
    let u = f * s.dot(&h);

    // Intersection point is outside the triangle (u coordinate)
    if !(0.0..=1.0).contains(&u) {
        return false;
    }

    let q = s.cross(&edge1);
    let v = f * direction.dot(&q);

    // Intersection point is outside the triangle (v coordinate)
    if v < 0.0 || u + v > 1.0 {
        return false;
    }

    // Calculate t to find intersection point along the ray
    let t = f * edge2.dot(&q);

    // Check if intersection is within the segment [0, 1]
    // We use a small epsilon buffer to avoid self-intersections at endpoints
    t > EPS && t < 1.0 - EPS
}
