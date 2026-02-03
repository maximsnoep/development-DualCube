use std::sync::Arc;

use mehsh::prelude::Mesh;
use serde::{Deserialize, Serialize};

use crate::{
    prelude::INPUT,
    skeleton::{
        connectivity_surgery::extract_skeleton,
        contraction::{contract_mesh, CONTRACTION},
        curve_skeleton::CurveSkeleton,
        orthogonalize::LabeledCurveSkeleton,
    },
};

pub mod connectivity_surgery;
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

impl SkeletonData {
    /// Returns a reference to the contracted mesh.
    pub fn contraction_mesh(&self) -> &Mesh<CONTRACTION> {
        &self.contraction_mesh
    }
}

/// Generates a polycube and a homeomorphism between the input mesh and the polycube,
/// using skeletonization.
pub fn get_skeleton_based_mapping(mesh: Arc<Mesh<INPUT>>) -> SkeletonData {
    // Start by doing contraction
    let contracted_mesh = contract_mesh(&mesh, 50);

    // Turn the contracted mesh into a 1D curve skeleton
    let curve_skeleton = extract_skeleton(&contracted_mesh, &mesh);

    // Orthogonalize and label the curve skeleton
    // TODO: orthogonalization

    // Generate polycube based on labeled skeleton
    // TODO: polycube generation

    // Create the mapping between input mesh and polycube
    // TODO: mapping

    SkeletonData {
        contraction_mesh: Arc::new(contracted_mesh),
        curve_skeleton: Some(curve_skeleton),
        labeled_skeleton: None,
    }
}

/// Generates surface-embedded loops from a polycube and polycube map.
pub fn generate_loops() {
    // todo!()
}
