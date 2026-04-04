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

    // TODO

    slots
}
