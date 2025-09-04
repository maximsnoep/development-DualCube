use crate::prelude::*;
use itertools::Itertools;
use std::collections::HashMap;

impl<M: Tag> Mesh<M> {
    #[must_use]
    fn empty() -> Self {
        Self::default()
    }

    // This is a struct that defines an embedded mesh with vertices (with position), edges, and faces (with clockwise ordering).
    pub fn from(faces: &[Vec<usize>], positions: &[Vector3D]) -> Result<(Self, ids::IdMap<VERT, M>, ids::IdMap<FACE, M>), MeshError<M>> {
        let mut mesh = Self::empty();

        // 1. Create the vertices.
        //      trivial; get all unique input vertices (from the faces), and create a vertex for each of them
        //
        // 2. Create the faces with its (half)edges.
        //      each face has edges defined by a sequence of vertices, example:
        //          face = [v0, v1, v2]
        //          then we create three edges = [(v0, v1), (v1, v2), (v2, v0)]
        //                v0
        //                *
        //               ^ \
        //              /   \ e0
        //          e2 /     \
        //            /       v
        //        v2 * < - - - * v1
        //                e1
        //
        //      Also assign representatives to vertices and faces whenever you make them.
        //
        // 3. Assign twins.
        //      trivial; just assign THE edge that has the same endpoints, but swapped (just requires some bookkeeping)
        //      return error if no such edge exists
        //

        // 1. Create the vertices.
        // Need mapping between original indices, and new pointers
        let mut vertex_pointers = ids::IdMap::new();
        let mut face_pointers = ids::IdMap::new();

        for &inp_vert_id in faces.iter().flatten().unique() {
            vertex_pointers.insert(inp_vert_id, mesh.add_vertex(positions[inp_vert_id]));
        }

        // 2. Create the faces with its (half)edges.
        // Need mapping between endpoints and edges for later use (assigning twins).
        let mut endpoints_to_edges = HashMap::<(VertKey<M>, VertKey<M>), EdgeKey<M>>::new();
        for (inp_face_id, inp_face_verts) in faces.iter().enumerate() {
            let face_id = mesh.add_face();
            face_pointers.insert(inp_face_id, face_id);

            let mut edge_ids = vec![];
            for i in 0..inp_face_verts.len() {
                let inp_start_vertex = inp_face_verts[i];
                let inp_end_vertex = inp_face_verts[(i + 1) % inp_face_verts.len()];
                let (&start_vertex, &end_vertex) = (vertex_pointers.key(inp_start_vertex).unwrap(), vertex_pointers.key(inp_end_vertex).unwrap());
                let edge_id = mesh.add_edge();
                if endpoints_to_edges.insert((start_vertex, end_vertex), edge_id).is_some() {
                    return Err(MeshError::DuplicateEdge(start_vertex, end_vertex));
                }
                edge_ids.push(edge_id);
                mesh.face_repr.insert(face_id, edge_id);
                mesh.vert_repr.insert(start_vertex, edge_id);
                mesh.edge_root.insert(edge_id, start_vertex);
                mesh.edge_face.insert(edge_id, face_id);
            }

            // Linking each edge to its next edge in the face
            for edge_index in 0..edge_ids.len() {
                mesh.edge_next.insert(edge_ids[edge_index], edge_ids[(edge_index + 1) % edge_ids.len()]);
            }
        }

        // 3. Assign twins.
        for (&(vert_a, vert_b), &edge_id) in &endpoints_to_edges {
            // Retrieve the twin edge
            if let Some(&twin_id) = endpoints_to_edges.get(&(vert_b, vert_a)) {
                // Assign twins
                mesh.edge_twin.insert(edge_id, twin_id);
                mesh.edge_twin.insert(twin_id, edge_id);
            } else {
                return Err(MeshError::NoTwin(vert_a, vert_b));
            }
        }

        // Assert that all elements have their required properties set.
        mesh.assert_properties();
        mesh.assert_references();
        mesh.assert_invariants();

        // mesh.is_connected();
        if mesh.is_polygonal().is_err() {
            return Err(MeshError::Unknown("Mesh is not polygonal".to_string()));
        }

        Ok((mesh, vertex_pointers, face_pointers))
    }
}
