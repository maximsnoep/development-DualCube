//! Skeleton-based initialization of a dual loop structure.

pub mod certify;
pub mod cut_system;
pub mod draw_loops;
pub mod edit;
pub mod geometry;
pub mod label;
pub mod model;
pub mod optimize;
pub mod pipeline;
pub mod place_cubes;
pub mod quality;
pub mod skeletonize;
pub(crate) mod surface;
pub mod validate;

pub use model::*;

/// Result of an initialization step.
pub type SkeletonResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;
