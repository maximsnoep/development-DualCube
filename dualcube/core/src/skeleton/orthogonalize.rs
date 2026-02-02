use petgraph::graph::UnGraph;

use crate::{prelude::PrincipalDirection, skeleton::curve_skeleton::SkeletonNode};


/// A 3-dimensional integer vector.
pub type IVector3D = nalgebra::SVector<i32, 3>;

/// Data payload for an orthogonalized skeleton node.
pub type OrthogonalSkeletonNode = (SkeletonNode, IVector3D);

/// Edge payload for orthogonalized skeleton edges, along with their length.
pub type OrthogonalSkeletonEdge = (PrincipalDirection, u32);

/// A curve skeleton graph labelled such that it represents an orthogonal embedding.
pub type LabeledCurveSkeleton = UnGraph<OrthogonalSkeletonNode, OrthogonalSkeletonEdge>;
