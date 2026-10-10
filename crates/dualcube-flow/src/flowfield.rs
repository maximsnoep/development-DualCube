//! Flow fields.
//!
//! For every axis, a tangent vector field along which loops of that axis run: at every vertex, the cross product of
//! the normal and the tangential part of the axis. Its direction runs around the axis, and its magnitude (the sine of
//! the angle between the normal and the axis) is the confidence of the flow: one on the sides around the axis, zero on
//! the caps that face the axis (where the direction is arbitrary). There is deliberately no smoothness term:
//! smoothing spreads directions across features, and loops that follow a smoothed field are worse loops for
//! polycubes.

use mehsh::prelude::*;
use slotmap::{SlotMap, new_key_type};

new_key_type! {
    pub struct GlobalVectorKey;
}

const EPS: f64 = 1e-12;

const X_AXIS: Vector3D = Vector3D::new(1.0, 0.0, 0.0);
const Y_AXIS: Vector3D = Vector3D::new(0.0, 1.0, 0.0);
const Z_AXIS: Vector3D = Vector3D::new(0.0, 0.0, 1.0);
const AXES: [Vector3D; 3] = [X_AXIS, Y_AXIS, Z_AXIS];

#[derive(Debug, Clone)]
pub struct Field<T: Tag> {
    pub map: HashMap<ids::Key<VERT, T>, GlobalVectorKey>,
    pub vectors: SlotMap<GlobalVectorKey, Vector3D>,
}

#[derive(Debug, Clone)]
pub struct Fields<T: Tag> {
    pub field_x: Field<T>,
    pub field_y: Field<T>,
    pub field_z: Field<T>,
}

impl<T: Tag> Field<T> {
    pub fn vector_at(&self, id: ids::Key<VERT, T>) -> Option<Vector3D> {
        self.map.get(&id).map(|key| self.vectors[*key])
    }
}

impl<T: Tag> Fields<T> {
    /// The fields of the three axes (see the module documentation).
    pub fn new(mesh: &Mesh<T>) -> Self {
        let ids = mesh.vert_ids();
        let [field_x, field_y, field_z] = AXES.map(|axis| {
            let mut field = Field {
                map: HashMap::with_capacity(ids.len()),
                vectors: SlotMap::with_key(),
            };
            for &id in &ids {
                let normal = mesh.normal(id).normalize();
                let (around, confidence) = around_axis_target(normal, axis)
                    .unwrap_or_else(|| (tangent_fallback(normal), 0.));
                field
                    .map
                    .insert(id, field.vectors.insert(around * confidence));
            }
            field
        });
        Self {
            field_x,
            field_y,
            field_z,
        }
    }
}

fn around_axis_target(normal: Vector3D, axis: Vector3D) -> Option<(Vector3D, f64)> {
    let tangent_axis = project_to_tangent(axis, normal);
    let confidence = tangent_axis.norm();
    (confidence >= EPS).then(|| (normal.cross(&tangent_axis).normalize(), confidence))
}

fn project_to_tangent(v: Vector3D, n: Vector3D) -> Vector3D {
    v - n * v.dot(&n)
}

fn tangent_fallback(n: Vector3D) -> Vector3D {
    let axis = if n.x.abs() < 0.9 { X_AXIS } else { Y_AXIS };
    project_to_tangent(axis, n).normalize()
}
