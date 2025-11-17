pub mod dual;
pub mod feature;
pub mod layout;
pub mod polycube;
pub mod quad;
pub mod solutions;

pub mod prelude {
    use mehsh::prelude::*;
    use std::fmt::Display;

    pub use crate::polycube::Polycube;
    pub use crate::solutions::Solution;

    mehsh::prelude::define_tag!(INPUT);
    pub type VertID = VertKey<INPUT>;
    pub type EdgeID = EdgeKey<INPUT>;
    pub type FaceID = FaceKey<INPUT>;

    // Principal directions, used to characterize a polycube (each edge and face is associated with a principal direction)
    #[derive(Copy, Clone, Default, PartialEq, Eq, Debug, Hash)]
    pub enum PrincipalDirection {
        #[default]
        X,
        Y,
        Z,
    }
    impl Display for PrincipalDirection {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::X => write!(f, "X-axis"),
                Self::Y => write!(f, "Y-axis"),
                Self::Z => write!(f, "Z-axis"),
            }
        }
    }

    #[derive(Default, Copy, Clone, Debug, PartialEq, Eq, Hash)]
    pub enum Orientation {
        #[default]
        Forwards,
        Backwards,
    }

    #[derive(Copy, Clone, Default, Debug)]
    pub enum Perspective {
        Primal,
        #[default]
        Dual,
    }

    impl From<PrincipalDirection> for Vector3D {
        fn from(dir: PrincipalDirection) -> Self {
            match dir {
                PrincipalDirection::X => Self::new(1., 0., 0.),
                PrincipalDirection::Y => Self::new(0., 1., 0.),
                PrincipalDirection::Z => Self::new(0., 0., 1.),
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum Side {
        Upper,
        Lower,
    }

    pub fn to_principal_direction(v: Vector3D) -> (PrincipalDirection, Orientation) {
        let x_is_max = v.x.abs() >= v.y.abs() && v.x.abs() >= v.z.abs();
        let y_is_max = v.y.abs() > v.x.abs() && v.y.abs() >= v.z.abs();
        let z_is_max = v.z.abs() > v.x.abs() && v.z.abs() > v.y.abs();
        assert!(x_is_max ^ y_is_max ^ z_is_max, "{v:?}");

        if x_is_max {
            if v.x > 0. {
                (PrincipalDirection::X, Orientation::Forwards)
            } else {
                (PrincipalDirection::X, Orientation::Backwards)
            }
        } else if y_is_max {
            if v.y > 0. {
                (PrincipalDirection::Y, Orientation::Forwards)
            } else {
                (PrincipalDirection::Y, Orientation::Backwards)
            }
        } else if z_is_max {
            if v.z > 0. {
                (PrincipalDirection::Z, Orientation::Forwards)
            } else {
                (PrincipalDirection::Z, Orientation::Backwards)
            }
        } else {
            unreachable!()
        }
    }

    pub fn to_vector(dir: PrincipalDirection, orientation: Orientation) -> Vector3D {
        let v = Vector3D::from(dir);
        match orientation {
            Orientation::Forwards => v,
            Orientation::Backwards => -v,
        }
    }
}
