use mehsh::prelude::*;
use serde::{Deserialize, Serialize};
use slotmap::{new_key_type, SecondaryMap, SlotMap};
use std::collections::HashMap;
use std::collections::HashSet;

new_key_type! { pub struct VectorKey; }

// A 1-vector field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Field<T: Tag> {
    pub map: HashMap<ids::Key<VERT, T>, VectorKey>,
    pub vectors: SlotMap<VectorKey, Vector3D>,
    pub connectivity: SecondaryMap<VectorKey, HashSet<VectorKey>>,
}

// My special directional field with
// Three 1-vector fields
// they are coupled s.t. together they are close to orthogonal to eachother (on each point)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fields<T: Tag> {
    pub field_x: Field<T>,
    pub field_y: Field<T>,
    pub field_z: Field<T>,
}

impl<T: Tag> Field<T> {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            vectors: SlotMap::with_key(),
            connectivity: SecondaryMap::new(),
        }
    }

    pub fn from_mesh(mesh: &Mesh<T>) -> Self {
        let mut field = Field::new();

        println!(
            "Initializing field from mesh with {} faces",
            mesh.face_ids().len()
        );

        // Populate the vectors
        for id in mesh.vert_ids() {
            // random unit vector
            let vec = Vector3D::new(
                rand::random::<f64>() - 0.5,
                rand::random::<f64>() - 0.5,
                rand::random::<f64>() - 0.5,
            )
            .normalize();
            let key = field.vectors.insert(vec);
            field.map.insert(id.to_owned(), key);
        }
        // Populate connectivity
        for id in mesh.vert_ids() {
            let key = field.map.get(&id).unwrap().to_owned();
            let neighbors = mesh
                .neighbors(id)
                .map(|id| field.map.get(&id).unwrap().to_owned())
                .collect();

            field.connectivity.insert(key, neighbors);
        }
        field
    }

    pub fn align_with_normals(&mut self, mesh: &Mesh<T>, axis: Vector3D) {
        for (id, vector_key) in &self.map {
            let normal = mesh.normal(*id);
            let direction = normal.cross(&axis);
            self.vectors[*vector_key] = direction;
        }
    }
}

impl<T: Tag> Fields<T> {
    pub fn new() -> Self {
        Self {
            field_x: Field::new(),
            field_y: Field::new(),
            field_z: Field::new(),
        }
    }
}
