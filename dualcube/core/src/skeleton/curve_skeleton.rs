use mehsh::prelude::Vector3D;
use petgraph::graph::UnGraph;

/// Nodes store their 3D position and a list of original mesh vertex indices that represent the induce surface patch.
pub type SkeletonNode = (Vector3D, Vec<usize>);

/// The extracted 1D skeleton, embedded in 3D space.
pub type CurveSkeleton = UnGraph<SkeletonNode, ()>;