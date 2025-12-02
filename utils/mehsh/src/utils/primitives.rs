pub type Float = f64;
pub type Vector2D = nalgebra::SVector<Float, 2>;
pub type Vector3D = nalgebra::SVector<Float, 3>;
pub type Matrix22 = nalgebra::SMatrix<Float, 2, 2>;
pub type Matrix32 = nalgebra::SMatrix<Float, 3, 2>;
pub type Color = [f32; 3];

pub const PI: f64 = std::f64::consts::PI;
pub const EPS: f64 = f64::EPSILON;
pub const INF: f64 = f64::INFINITY;
pub const NEG_INF: f64 = f64::NEG_INFINITY;
