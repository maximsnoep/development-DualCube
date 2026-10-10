pub mod coupled;
pub mod evolution;
pub mod initialize;
pub mod layout_evolution;
pub mod loop_adapters;
pub mod loop_band;
pub mod medial;
pub mod phases;
pub mod quality;
pub mod solution;

pub mod prelude {
    pub use dualcube_dual::prelude::*;
    pub use dualcube_flow::prelude::*;
    pub use dualcube_primal::prelude::*;
    pub use dualcube_quad::{QuadDensity, DEFAULT_QUAD_FACTOR, build_quad_from_layout};
    pub use dualcube_types::prelude::*;

    pub use crate::coupled::*;
    pub use crate::evolution::*;
    pub use crate::layout_evolution::*;
    pub use crate::loop_band::*;
    pub use crate::phases::*;
    pub use crate::quality::*;
    pub use crate::solution::*;

    pub use orx_parallel::*;
}

pub use dualcube_dual as dual;
pub use dualcube_flow as flow;
pub use dualcube_primal as primal;
pub use dualcube_quad as quad;
pub use dualcube_types as types;

pub use prelude::*;
