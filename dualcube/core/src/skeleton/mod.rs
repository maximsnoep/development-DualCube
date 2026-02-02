use std::sync::Arc;

use mehsh::prelude::Mesh;
use serde::{Deserialize, Serialize};

use crate::{
    skeleton::{
        curve_skeleton::CurveSkeleton, 
        orthogonalize::LabeledCurveSkeleton
    },
};

/// Tag type for contraction-view of skeleton.
#[derive(Debug, Clone, Default, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CONTRACTION;

pub mod contraction;
pub mod curve_skeleton;
pub mod orthogonalize;

/// Holds all relevant information for skeleton-based polycube initialization.
///
/// Fields will be gradually filled as computation proceeds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkeletonData {
    /// A contracted version of the input mesh.
    contraction_mesh: Arc<Mesh<CONTRACTION>>,

    /// The extracted curve skeleton for the input mesh, with induced surface patches.
    curve_skeleton: Option<CurveSkeleton>,

    /// The orthogonalized and labeled curve skeleton:
    ///  - Each node has an unique integer location,
    ///  - Each edge has a direction and length.
    labeled_skeleton: Option<LabeledCurveSkeleton>,
}
