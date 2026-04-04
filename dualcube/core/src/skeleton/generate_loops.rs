use std::collections::HashMap;

use log::warn;
use mehsh::prelude::Mesh;
use petgraph::{graph::EdgeIndex, visit::{EdgeRef, IntoEdgeReferences, IntoNodeReferences}};
use slotmap::SlotMap;

use crate::{
    prelude::{EdgeID, PrincipalDirection, INPUT},
    skeleton::{boundary_loop::BoundaryLoop, orthogonalize::LabeledCurveSkeleton, SkeletonData},
    solutions::{Loop, LoopID},
};

// custom error
pub enum LoopGenerationError {
    MissingLabeledSkeleton,
    // todo other error variants
}

/// Generates surface-embedded loops from a polycube and polycube map.
pub fn generate_loops(
    skeleton_data: &SkeletonData,
    mesh: &Mesh<INPUT>,
) -> Result<SlotMap<LoopID, Loop>, LoopGenerationError> {
    let mut map: SlotMap<LoopID, Loop> = SlotMap::with_key();

    // Use ortho-skeleton, for each patch boundary assign 4 points that will host loop (paths).
    // throw warn and return if not there
    let skeleton: &LabeledCurveSkeleton = skeleton_data
        .labeled_skeleton
        .as_ref()
        .ok_or_else(|| LoopGenerationError::MissingLabeledSkeleton)?;
    let (boundary_map, crossings) = get_boundaries_and_crossing_points(skeleton, &mut map);

    // For each patch, for each side that does not correspond to a boundary,
    // find a point on the surface that represents the center of that face.
    // TODO: from the skeleton node, we will have 6 vectors that are ideally all equally spaced angle-wise. For each direction that does not have a boundary, we can find an ideal direction, then find a point far in that direction on the surface.

    // Trace paths between boundaries and points to create the loops
    // TODO: restricted Dijkstra's or something. Can be somewhat smart about ordering and having the second loop of each pair be as far as possible from the first to nicely divide the surface.

    Ok(map)
}

/// Calculates for each patch-patch boundary the appropriate loop and crossing points for the other two loop types.
fn get_boundaries_and_crossing_points(
    skeleton: &LabeledCurveSkeleton,
    map: &mut SlotMap<LoopID, Loop>,
) -> (HashMap<EdgeIndex, LoopID>, HashMap<LoopID, ()>) {
    let crossings = HashMap::new();
    let mut boundary_map = HashMap::new();

    // Precompute using skeleton directionality per node. Determining what is up and down is easy for high-degree nodes,
    // so we start from those, then propogate that information to the rest of the skeleton. This will not be perfect, but hopefully good enough.
    // We can see this as establishing planes defining the space around the node.
    // For high-degree nodes, these planes will bend as back-to-back edges likely are not exactly aligned.
    // For a direction, if we only have 1 edge, we just extrapolate the other side as if the edge is straight.
    // We can then propogate these directions to neighbors adding 'fallback' constraints for if they do not have a neighbor in that direction.
    // These constraints can even be propogated around 'corners' (going from X to Y for example).
    // As a fallback, just use global directions to have _something_.
    let mut nodes_by_degree: Vec<_> = skeleton.node_references().collect();
    nodes_by_degree.sort_by_key(|n| skeleton.neighbors(n.0).count());
    // TODO: direction idea

    // Find each patch-patch boundary (which corresponds to skeleton edge)
    for edge in skeleton.edge_references() {
        let weight = edge.weight();
        let direction = weight.direction;
        let boundary = weight.boundary_loop.clone();

        // Create loop and save its ID
        let loop_id = map.insert(get_loop(boundary, direction));
        boundary_map.insert(edge.id(), loop_id);

        // For each direction vector, find point on boundary maximal in that direction, save it as crossing.
        // Note that these will be edge (midpoints) as loops cross mesh edges transversely. 
        // TODO, will be one per direction not filled for the node, always 4, as its 2 per other loop type (so for X loop, there are 2 Y crossings and 2 Z crossings).
    }

    (boundary_map, crossings)
}

fn get_loop(boundary: BoundaryLoop, direction: PrincipalDirection) -> Loop {
    Loop {
        edges: boundary.edge_midpoints,
        direction,
    }
}
