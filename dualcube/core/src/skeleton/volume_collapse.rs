use mehsh::prelude::Mesh;
use petgraph::graph::NodeIndex;
use serde::{Deserialize, Serialize};

use crate::{
    prelude::{CurveSkeleton, INPUT},
    skeleton::curve_skeleton::{CurveSkeletonManipulation, CurveSkeletonSpatial, MergeBehavior},
};

/// Collapses the skeleton one
pub fn volume_based_collapse(
    original_skeleton: &CurveSkeleton,
    mesh: &Mesh<INPUT>,
) -> VolumeCollapseHistory {
    // This could be a lot smarter, current approach does not at all deal well with higher-genus.

    let mut skeleton = original_skeleton.clone();

    let mut collapses = Vec::new();

    // For as long as possible, merge smallest-volume leaf into its parent
    let mut changed: bool;
    loop {
        changed = false;

        // Find all leaves          // TODO: could mostly be cached but doesn't really matter here
        let leaves: Vec<NodeIndex> = skeleton
            .node_indices()
            .filter(|&i| skeleton.neighbors(i).count() == 1)
            .collect();

        // Compute volume for each patch
        let leaf_volumes: Vec<(NodeIndex, f64)> = leaves
            .iter()
            .map(|&leaf| (leaf, skeleton.patch_volume(leaf, mesh)))
            .collect();

        // Find the leaf with the smallest volume, and merge it into its parent
        if let Some((smallest_leaf, _)) = leaf_volumes
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        {
            let parent = skeleton.neighbors(*smallest_leaf).next().unwrap();
            collapses.push(VolumeCollapse {
                source_node: *smallest_leaf,
                target_node: parent,
            });
            skeleton.merge_nodes(*smallest_leaf, parent, MergeBehavior::SourceIntoTarget);
            changed = true;
        }

        if !changed {
            break;
        }
    }

    VolumeCollapseHistory { history: collapses }
}

pub fn construct_skeleton_from_history(
    original_skeleton: &CurveSkeleton,
    history: &VolumeCollapseHistory,
    position: usize,
) -> CurveSkeleton {
    // Since the only boundary changed at each step is the absorbed one, we can just keep the old ones
    // Start with the original skeleton and apply until at the right position
    let end = position.min(history.history.len());
    let mut skeleton = original_skeleton.clone();

    for collapse in &history.history[0..end] {
        // Merge the patch vertices of source into target
        skeleton.merge_nodes(collapse.source_node, collapse.target_node, MergeBehavior::SourceIntoTarget);
    }

    skeleton
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeCollapseHistory {
    /// The sequence of volume collapses that were performed, in order.
    /// For genus 0 meshes, 1 vertex will remain. For higher genus, this will be more.
    pub history: Vec<VolumeCollapse>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeCollapse {
    /// The node that gets collapsed into its neighbor. This node will be removed from the skeleton.
    source_node: NodeIndex,

    /// The node that remains after the collapse. The patch will include now the absorbed patch vertices.
    target_node: NodeIndex,
}
