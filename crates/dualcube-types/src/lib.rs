use mehsh::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::{HashMap, HashSet};
use std::fmt::Display;

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct INPUT;
pub type VertID = VertKey<INPUT>;
pub type EdgeID = EdgeKey<INPUT>;
pub type FaceID = FaceKey<INPUT>;

#[derive(Copy, Clone, Default, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
pub enum Direction {
    #[default]
    X,
    Y,
    Z,
}

impl Display for Direction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::X => write!(f, "x-direction"),
            Self::Y => write!(f, "y-direction"),
            Self::Z => write!(f, "z-direction"),
        }
    }
}

pub const DIRECTIONS: [Direction; 3] = [Direction::X, Direction::Y, Direction::Z];

#[derive(Default, Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Sign {
    #[default]
    Positive,
    Negative,
}

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize)]
pub enum Perspective {
    Primal,
    #[default]
    Dual,
}

impl From<Direction> for Vector3D {
    fn from(dir: Direction) -> Self {
        match dir {
            Direction::X => Self::new(1., 0., 0.),
            Direction::Y => Self::new(0., 1., 0.),
            Direction::Z => Self::new(0., 0., 1.),
        }
    }
}

pub fn to_principal_direction(v: Vector3D) -> (Direction, Sign) {
    let x_is_max = v.x.abs() >= v.y.abs() && v.x.abs() >= v.z.abs();
    let y_is_max = v.y.abs() > v.x.abs() && v.y.abs() >= v.z.abs();
    let z_is_max = v.z.abs() > v.x.abs() && v.z.abs() > v.y.abs();
    if !(x_is_max || y_is_max || z_is_max) {
        // Only reachable for NaN components (e.g. a normalized zero-length vector).
        warn!("to_principal_direction: degenerate vector {v:?}, defaulting to +X");
        return (Direction::X, Sign::Positive);
    }

    if x_is_max {
        if v.x > 0. {
            (Direction::X, Sign::Positive)
        } else {
            (Direction::X, Sign::Negative)
        }
    } else if y_is_max {
        if v.y > 0. {
            (Direction::Y, Sign::Positive)
        } else {
            (Direction::Y, Sign::Negative)
        }
    } else if z_is_max {
        if v.z > 0. {
            (Direction::Z, Sign::Positive)
        } else {
            (Direction::Z, Sign::Negative)
        }
    } else {
        unreachable!()
    }
}

pub fn to_vector(dir: Direction, sign: Sign) -> Vector3D {
    let v = Vector3D::from(dir);
    match sign {
        Sign::Positive => v,
        Sign::Negative => -v,
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct POLYCUBE;

mehsh::prelude::define_tag!(QUAD);

#[derive(Clone, Debug, Default)]
pub struct Quad {
    pub triangle_mesh_polycube: Mesh<INPUT>,
    pub quad_mesh_polycube: Mesh<QUAD>,
    pub quad_mesh: Mesh<QUAD>,
    pub face_to_verts: HashMap<FaceKey<POLYCUBE>, Vec<Vec<VertKey<QUAD>>>>,
    pub edge_to_verts: HashMap<EdgeKey<POLYCUBE>, Vec<VertKey<QUAD>>>,
    pub frozen: HashSet<VertKey<QUAD>>,
}

impl Serialize for Quad {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_unit()
    }
}

impl<'de> Deserialize<'de> for Quad {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Consume the unit written by `serialize` so the stream stays aligned.
        <()>::deserialize(deserializer)?;
        Ok(Quad::default())
    }
}

pub mod prelude {
    pub use crate::{
        DIRECTIONS, Direction, EdgeID, FaceID, INPUT, POLYCUBE, Perspective, QUAD, Quad, Sign,
        VertID, to_principal_direction, to_vector,
    };
    pub use grapff;
    pub use itertools::Itertools;
    pub use mehsh::prelude::*;
    pub use ordered_float::OrderedFloat;
    pub use std::collections::{HashMap, HashSet, VecDeque};
    pub use tracing::{debug, info, warn};
}
