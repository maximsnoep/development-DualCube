use mehsh::prelude::Mesh;
use slotmap::SlotMap;

use crate::{
    prelude::INPUT,
    skeleton::SkeletonData,
    solutions::{Loop, LoopID},
};

/// Generates surface-embedded loops from a polycube and polycube map.
pub fn generate_loops(skeleton: &SkeletonData, mesh: &Mesh<INPUT>) -> SlotMap<LoopID, Loop> {
    let slots: SlotMap<LoopID, Loop> = SlotMap::with_key();

    // Use ortho-skeleton, for each patch boundary assign 4 points that will host loop (paths).
    // TODO

    // For each patch, for each side that does not correspond to a boundary, 
    // find a point on the surface that represents the center of that face.
    // TODO

    // Trace paths between boundaries and points to create the loops
    // TODO
    
    slots
}
