use std::sync::Arc;

use log::{info, warn};
use mehsh::prelude::Mesh;
use serde::{Deserialize, Serialize};

use crate::{
    prelude::INPUT,
    skeleton::{
        connectivity_surgery::extract_skeleton,
        contraction::{CONTRACTION, contract_mesh},
        curve_skeleton::{CurveSkeleton, CurveSkeletonManipulation},
        embeddability::make_embedding_possible,
        orthogonalize::{LabeledCurveSkeleton, greedy_orthogonalization},
        simplify::{convexify, simplify_skeleton},
        volume_collapse::{
            VolumeCollapseHistory, construct_skeleton_from_history, volume_based_collapse
        },
    },
};

pub mod curve_skeleton;
pub mod orthogonalize;

mod boundary_loop;
mod connectivity_surgery;
mod contraction;
mod embeddability;
mod manipulation;
mod patch;
mod simplify;
mod volume_collapse;

/// Holds all relevant information for skeleton-based polycube initialization.
///
/// Fields will be gradually filled as computation proceeds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkeletonData {
    /// A contracted version of the input mesh.
    contraction_mesh: Arc<Mesh<CONTRACTION>>,

    /// The extracted curve skeleton for the input mesh, with induced surface patches,
    /// directly from connectivity surgery.
    raw_curve_skeleton: Option<CurveSkeleton>,

    /// Simplified version of the raw curve skeleton.
    cleaned_skeleton: Option<CurveSkeleton>,

    /// The history of doing volume-based collapses.
    collapse_history: Option<VolumeCollapseHistory>,

    /// The orthogonalized and labeled curve skeleton:
    ///  - Each node has an unique integer location,
    ///  - Each edge has a direction and length.
    labeled_skeleton: Option<LabeledCurveSkeleton>,
}

impl SkeletonData {
    /// Returns a reference to the contracted mesh.
    pub fn contraction_mesh(&self) -> &Mesh<CONTRACTION> {
        &self.contraction_mesh
    }

    /// Returns a reference to the curve skeleton if it has been computed.
    pub fn curve_skeleton(&self) -> Option<&CurveSkeleton> {
        self.raw_curve_skeleton.as_ref()
    }

    /// Returns a reference to the cleaned skeleton if it has been computed.
    pub fn cleaned_skeleton(&self) -> Option<&CurveSkeleton> {
        self.cleaned_skeleton.as_ref()
    }

    /// Returns a reference to the labeled skeleton if it has been computed.
    pub fn labeled_skeleton(&self) -> Option<&LabeledCurveSkeleton> {
        self.labeled_skeleton.as_ref()
    }

    /// Reconstructs what a skeleton looked like at a certain point in the volume collapse history, if the history is available.
    pub fn reconstruct_skeleton_from_collapse_history(
        &self,
        position: usize,
    ) -> Option<CurveSkeleton> {
        if let (Some(history), Some(cleaned_skeleton)) =
            (&self.collapse_history, &self.cleaned_skeleton)
        {
            Some(construct_skeleton_from_history(
                cleaned_skeleton,
                history,
                position,
            ))
        } else {
            None
        }
    }

    pub fn history_size(&self) -> Option<usize> {
        self.collapse_history.as_ref().map(|h| h.history.len())
    }
}

/// Generates a polycube and a homeomorphism between the input mesh and the polycube,
/// using skeletonization.
pub fn get_skeleton_based_mapping(mesh: Arc<Mesh<INPUT>>) -> SkeletonData {
    // Start by doing contraction
    let contracted_mesh = contract_mesh(&mesh, 50);

    // Turn the contracted mesh into a 1D curve skeleton
    let curve_skeleton = extract_skeleton(&contracted_mesh, &mesh);

    // Simplify skeleton to get more coherent features
    let mut cleaned_skeleton = curve_skeleton.clone();
    simplify_skeleton(&mut cleaned_skeleton, &mesh);

    // Smooth region boundaries
    cleaned_skeleton.smooth_boundaries(&mesh);

    // Convexify skeleton to make patch volume close to convex shapes, which map nicely to cubes.
    const CONVEXITY_THRESHOLD: f64 = 0.8; // TODO: make configurable in UI
    const CONVEXITY_MERGE_THRESHOLD: f64 = 0.95; // When merging two patches, the new convexity should be at least threshold*best_before 
    convexify(&mut cleaned_skeleton, &mesh, CONVEXITY_THRESHOLD, CONVEXITY_MERGE_THRESHOLD);

    // Fix necessary conditions for orthogonal embeddability, most of the times this changes nothing.
    make_embedding_possible(&mut cleaned_skeleton, &mesh);

    // Orthogonalize the curve skeleton
    let labeled = greedy_orthogonalization(&cleaned_skeleton);
    match &labeled {
        Some(_) => {
            info!("Orthogonalization successful.");
        }
        None => {
            warn!("Orthogonalization failed.");
        }
    }

    // (MAYBE TEMP) Do volume based collapse, and save the history.
    let history = volume_based_collapse(&cleaned_skeleton, &mesh);

    // Generate polycube based on labeled skeleton
    // TODO: polycube generation

    // Create the mapping between input mesh and polycube
    // TODO: mapping

    SkeletonData {
        contraction_mesh: Arc::new(contracted_mesh),
        raw_curve_skeleton: Some(curve_skeleton),
        cleaned_skeleton: Some(cleaned_skeleton),
        collapse_history: Some(history),
        labeled_skeleton: labeled,
    }
}

/// Generates surface-embedded loops from a polycube and polycube map.
pub fn generate_loops() {
    // todo!()
}
