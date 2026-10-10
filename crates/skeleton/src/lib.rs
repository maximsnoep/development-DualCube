//! TODO

pub mod draw_loops;
pub mod skeletonize;
pub mod label;
pub mod model;
pub mod optimize;

pub use model::*;

/// Stages can propagate their algorithm's errors with `?`, without a shared
/// registry of every method's error variants. Errors may carry diagnostics.
pub type StageResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;
