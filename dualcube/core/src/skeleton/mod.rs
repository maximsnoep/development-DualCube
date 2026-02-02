use std::sync::Arc;

use mehsh::prelude::{Mesh, Vector3D};
use serde::{Deserialize, Serialize};

use crate::{
    prelude::INPUT,
    skeleton::{
        curve_skeleton::CurveSkeleton, 
        orthogonalize::LabeledCurveSkeleton
    },
};

pub mod contraction;
pub mod curve_skeleton;
pub mod orthogonalize;

/// Holds all relevant information for skeleton-based polycube initialization.
///
/// Fields will be gradually filled as computation proceeds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkeletonData {
    /// A reference to the original mesh.
    original_mesh: Arc<Mesh<INPUT>>,

    /// Positions of the contracted mesh vertices.
    contracted_positions: Option<Vec<Vector3D>>,

    /// The extracted curve skeleton for the input mesh, with induced surface patches.
    curve_skeleton: Option<CurveSkeleton>,

    /// The orthogonalized and labeled curve skeleton:
    ///  - Each node has an unique integer location,
    ///  - Each edge has a direction and length.
    labeled_skeleton: Option<LabeledCurveSkeleton>,
}
