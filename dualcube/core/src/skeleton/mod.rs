use std::sync::Arc;

use log::{info, warn};
use mehsh::prelude::Mesh;
use serde::{Deserialize, Serialize};

use crate::{
    prelude::{Polycube, INPUT},
    quad::Quad,
    skeleton::{
        connectivity_surgery::extract_skeleton,
        contraction::{contract_mesh, CONTRACTION},
        curve_skeleton::{CurveSkeleton, CurveSkeletonManipulation, CurveSkeletonSpatial},
        embeddability::make_embedding_possible,
        orthogonalize::{greedy_orthogonalization, LabeledCurveSkeleton},
        simplify::{convexify, simplify_skeleton},
        volume_collapse::{
            construct_skeleton_from_history, volume_based_collapse, VolumeCollapseHistory,
        },
        voxelize::generate_polycube,
    },
};

pub mod curve_skeleton;
pub mod orthogonalize;

mod boundary_loop;
mod connectivity_surgery;
mod contraction;
mod cross_parameterize;
mod embeddability;
mod manipulation;
mod patch;
mod simplify;
mod volume_collapse;
mod voxelize;

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

    /// A skeleton isomorphic to `labeled_skeleton`, but with node positions and patch vertices updated to match the polycube structure.
    polycube_skeleton: Option<LabeledCurveSkeleton>,
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

    pub fn update_convexity(
        &mut self,
        mesh: Arc<Mesh<INPUT>>,
        convexity_threshold: f64,
        convexity_merge_threshold: f64,
        omega: usize,
    ) -> (Option<Polycube>, Option<Quad>) {
        // Reuse contraction
        let (curve_skeleton, mut cleaned_skeleton) =
            surgery_and_simplification(&mesh, &self.contraction_mesh);

        // Reuse pipeline post simplifcation
        let (labeled, history, polycube_and_skeleton) = post_simplification_stage(
            mesh,
            convexity_threshold,
            convexity_merge_threshold,
            &mut cleaned_skeleton,
            omega,
        );

        let (polycube, polycube_skeleton, quad) = match polycube_and_skeleton {
            Some((p, s, q)) => (Some(p), Some(s), Some(q)),
            None => (None, None, None),
        };

        self.raw_curve_skeleton = Some(curve_skeleton); // Not updated now, but we calculate it anyways so might as well save it
        self.cleaned_skeleton = Some(cleaned_skeleton);
        self.collapse_history = Some(history);
        self.labeled_skeleton = labeled;
        self.polycube_skeleton = polycube_skeleton;

        (polycube, quad)
    }
}

/// Generates a polycube and a homeomorphism between the input mesh and the polycube,
/// using skeletonization.
pub fn get_skeleton_based_mapping(
    mesh: Arc<Mesh<INPUT>>,
    convexity_threshold: f64,
    convexity_merge_threshold: f64,
    omega: usize,
) -> (SkeletonData, Option<Polycube>, Option<Quad>) {
    // Start by doing contraction
    let contracted_mesh = contract_mesh(&mesh, 50);

    let (raw_curve_skeleton, mut cleaned_skeleton) =
        surgery_and_simplification(&mesh, &contracted_mesh);

    let (labeled, history, polycube_and_skeleton) = post_simplification_stage(
        mesh,
        convexity_threshold,
        convexity_merge_threshold,
        &mut cleaned_skeleton,
        omega,
    );

    let (polycube, polycube_skeleton, quad) = match polycube_and_skeleton {
        Some((p, s, q)) => (Some(p), Some(s), Some(q)),
        None => (None, None, None),
    };

    (
        SkeletonData {
            contraction_mesh: Arc::new(contracted_mesh),
            raw_curve_skeleton: Some(raw_curve_skeleton),
            cleaned_skeleton: Some(cleaned_skeleton),
            collapse_history: Some(history),
            labeled_skeleton: labeled,
            polycube_skeleton,
        },
        polycube,
        quad,
    )
}

fn surgery_and_simplification(
    mesh: &Arc<Mesh<INPUT>>,
    contracted_mesh: &Mesh<CONTRACTION>,
) -> (CurveSkeleton, CurveSkeleton) {
    // Turn the contracted mesh into a 1D curve skeleton
    let curve_skeleton = extract_skeleton(contracted_mesh, mesh);

    // Simplify skeleton to get more coherent features
    let mut cleaned_skeleton = curve_skeleton.clone();
    simplify_skeleton(&mut cleaned_skeleton, mesh);

    // Smooth region boundaries
    cleaned_skeleton.smooth_boundaries(mesh);
    (curve_skeleton, cleaned_skeleton)
}

/// The decomposed part of the skeletonization process that happens after simplification.
fn post_simplification_stage(
    mesh: Arc<Mesh<INPUT>>,
    convexity_threshold: f64,
    convexity_merge_threshold: f64,
    cleaned_skeleton: &mut CurveSkeleton,
    omega: usize,
) -> (
    Option<LabeledCurveSkeleton>,
    VolumeCollapseHistory,
    Option<(Polycube, LabeledCurveSkeleton, Quad)>,
) {
    // Convexify skeleton to make patch volume close to convex shapes, which map nicely to cubes.
    convexify(
        cleaned_skeleton,
        &mesh,
        convexity_threshold,
        convexity_merge_threshold,
    );

    // Fix necessary conditions for orthogonal embeddability, most of the times this changes nothing.
    make_embedding_possible(cleaned_skeleton, &mesh);

    // Before labeling (which uses geometric node position), refine position again.
    // Positions likely are not accurate anymore after merges and splits and such.
    cleaned_skeleton.refine_embeddings(&mesh);

    // Orthogonalize the curve skeleton
    let labeled = greedy_orthogonalization(&*cleaned_skeleton);
    match &labeled {
        Some(_) => {
            info!("Orthogonalization successful.");
        }
        None => {
            warn!("Orthogonalization failed.");
        }
    }

    // (MAYBE TEMP) Do volume based collapse, and save the history.
    let history = volume_based_collapse(&*cleaned_skeleton, &mesh);

    // Generate polycube based on labeled skeleton
    let polycube: Option<(Polycube, LabeledCurveSkeleton, Quad)> = match &labeled {
        Some(labeled) => Some(generate_polycube(labeled, omega)),
        None => None,
    };

    // Create the mapping between input mesh and polycube
    // TODO: mapping
    (labeled, history, polycube)
}

/// Generates surface-embedded loops from a polycube and polycube map.
pub fn generate_loops() {
    // todo!()
}
