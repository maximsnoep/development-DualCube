//! Skeleton-based initialization of a dual loop structure.

pub mod draw_loops;
pub mod geometry;
pub mod label;
pub mod model;
pub mod optimize;
pub mod pipeline;
pub mod place_cubes;
pub mod skeletonize;
pub(crate) mod surface;
pub mod validate;

pub use model::*;
pub use pipeline::{SkeletonRun, run};

/// Result of an initialization step.
pub type SkeletonResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[cfg(test)]
mod reference_drawing;
