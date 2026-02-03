use mehsh::prelude::{Vector3D, VertKey};
use petgraph::graph::UnGraph;

use crate::prelude::INPUT;

/// Nodes store their 3D position and a list of original mesh vertex keys
/// that represent the induced surface patch.
pub type SkeletonNode = (Vector3D, Vec<VertKey<INPUT>>);

/// The extracted 1D skeleton, embedded in 3D space.
pub type CurveSkeleton = UnGraph<SkeletonNode, ()>;