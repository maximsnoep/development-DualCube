pub mod mesh {
    pub mod builder;
    pub mod connectivity;
    pub mod verification;
    pub mod elem {
        pub mod edge;
        pub mod face;
        pub mod vert;
    }
    pub mod algo {
        pub mod projection;
        pub mod refinement;
        pub mod triangulate;
        pub mod location {
            pub mod face;
            pub mod vert;
        }
    }
}

pub mod integrations {
    #[cfg(feature = "bevy")]
    pub mod bevy;
    #[cfg(feature = "obj")]
    pub mod obj;
    #[cfg(feature = "petgraph")]
    pub mod petgraph;
    #[cfg(feature = "stl")]
    pub mod stl;
}

pub mod utils {
    pub mod geom;
    pub mod ids;
    pub mod math;
    pub mod primitives;
}

pub mod prelude {
    pub use crate::define_tag;
    pub use crate::mesh::algo::location::{face::FaceLocation, vert::VertLocation};
    pub use crate::mesh::connectivity::{
        EDGE, EdgeKey, FACE, FaceKey, HasEdges, HasFaces, HasNeighbors, HasNormal, HasPosition, HasRing, HasSize, HasVertices, Mesh, MeshError, SetPosition,
        Tag, VERT, VertKey,
    };
    pub use crate::utils::geom;
    pub use crate::utils::ids;
    pub use crate::utils::math;
    pub use crate::utils::primitives::*;
}

#[cfg(test)]
pub mod tests;
