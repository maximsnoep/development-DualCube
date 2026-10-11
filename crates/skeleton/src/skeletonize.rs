//! Deflation, connectivity surgery, and embedding refinement.
//!
//! `run` returns the final skeleton; `extract` also retains the deflated mesh
//! and the skeleton before refinement. Each submodule can be called separately.

pub mod deflate;
pub mod refinement;
pub mod surgery;

use crate::{Skeleton, SkeletonResult};
use dualcube_types::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub max_iterations: usize,
    pub refine_embedding: bool,
    pub surgery: surgery::Options,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            refine_embedding: true,
            surgery: surgery::Options::default(),
        }
    }
}

/// Extraction states, retained for inspection without running later stages.
#[derive(Clone, Debug)]
pub struct Extraction {
    pub contracted_mesh: Mesh<deflate::CONTRACTION>,
    pub raw: Skeleton,
    pub skeleton: Skeleton,
    pub genus: usize,
}

/// Extract a skeleton with the default contraction and refinement settings.
pub fn run(mesh: &Mesh<INPUT>) -> SkeletonResult<Skeleton> {
    Ok(extract(mesh, Options::default())?.skeleton)
}

pub fn extract(mesh: &Mesh<INPUT>, options: Options) -> SkeletonResult<Extraction> {
    options.surgery.validate()?;
    let contracted_mesh = deflate::run(mesh, options.max_iterations)?;
    extract_from_contracted(mesh, contracted_mesh, options)
}

/// Continue from a contracted mesh with matching topology. Only surgery and refinement options apply.
pub fn extract_from_contracted(
    mesh: &Mesh<INPUT>,
    contracted_mesh: Mesh<deflate::CONTRACTION>,
    options: Options,
) -> SkeletonResult<Extraction> {
    options.surgery.validate()?;
    let genus = crate::validate::input_genus(mesh)?;
    let raw = surgery::run_with_options(&contracted_mesh, mesh, options.surgery)?;
    let mut skeleton = raw.clone();
    if options.refine_embedding {
        refinement::run(mesh, &mut skeleton)?;
    }
    skeleton.validate(mesh)?;
    Ok(Extraction {
        contracted_mesh,
        raw,
        skeleton,
        genus,
    })
}
