//! Skeleton-based initialization of a dual loop structure.
//!
//! Skeleton extraction → labeling → cube placement → optimization → loop drawing.
//! Each step can be called independently; `run` combines them.

pub mod draw_loops;
pub mod label;
pub mod model;
pub mod optimize;
pub mod pipeline;
pub mod place_cubes;
pub mod skeletonize;

pub use model::*;
pub use pipeline::{SkeletonRun, run};

/// Result of an initialization step.
pub type SkeletonResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;
