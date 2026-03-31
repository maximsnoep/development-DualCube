use itertools::Itertools;

use crate::prelude::*;
use std::collections::HashMap;

// Given an arbitrary polygonal mesh, triangulate all faces to obtain a triangular mesh.
// This is useful for rendering or further processing.
impl<M: Tag> Mesh<M> {
    pub fn triangulate(&self) -> Result<(Self, HashMap<FaceKey<M>, FaceKey<M>>), MeshError<M>> {
        let mut new_mesh = self.clone();
        let mut new_faces = HashMap::new();

        for face in self.face_ids() {
            if self.vertices(face).count() > 3 {
                for new_face in new_mesh.triangulate_face(face)? {
                    // Insert the new face into the new mesh
                    new_faces.insert(new_face, face);
                }
            }
        }

        log::info!(
            "Triangulated {} faces into {} new faces",
            self.faces.len(),
            new_faces.len()
        );

        Ok((new_mesh, new_faces))
    }

    pub fn triangulate_face(&mut self, face: FaceKey<M>) -> Result<Vec<FaceKey<M>>, MeshError<M>> {
        let edges = self.edges(face).collect_vec();
        let original_edges = self
            .edges(face)
            .map(|e| self.vertices(e).collect_vec())
            .collect_vec();
        let original_vertices = self.vertices(face).collect_vec();

        let triangles = [[1, 0, 3], [3, 2, 1]]; // For testing purposes, replace with actual triangulation for higher degree faces.

        // Remove the old face
        self.faces.remove(face);

        let mut new_faces = vec![];
        let mut endpoints_to_edges = HashMap::<(VertKey<M>, VertKey<M>), EdgeKey<M>>::new();
        for triangle in triangles {
            let face_id = self.add_face();
            new_faces.push(face_id);

            let mut edge_ids = vec![];

            let triangle = [triangle[0], triangle[2], triangle[1]];

            for i in 0..3 {
                let v1 = triangle[i];
                let v2 = triangle[(i + 1) % triangle.len()];
                let (start_vertex, end_vertex) = (original_vertices[v1], original_vertices[v2]);

                // Check if edge already exists
                if original_edges.contains(&vec![start_vertex, end_vertex]) {
                    // If it exists, use the existing edge

                    let index = original_edges
                        .iter()
                        .position(|e| e == &vec![start_vertex, end_vertex])
                        .unwrap();
                    let e1 = edges[index];

                    edge_ids.push(e1);
                } else {
                    // If it doesn't exist, create a new edge
                    let edge_id = self.add_edge();
                    // ONLY DO IF THE EDGE IS NEW..
                    if endpoints_to_edges
                        .insert((start_vertex, end_vertex), edge_id)
                        .is_some()
                    {
                        return Err(MeshError::DuplicateEdge(start_vertex, end_vertex));
                    }
                    edge_ids.push(edge_id);
                }

                let edge_id = edge_ids.last().unwrap().to_owned();

                self.face_repr.insert(face_id, edge_id);
                self.vert_repr.insert(start_vertex, edge_id);
                self.edge_root.insert(edge_id, start_vertex);
                self.edge_face.insert(edge_id, face_id);
            }

            // Linking each edge to its next edge in the face
            for edge_index in 0..edge_ids.len() {
                self.edge_next.insert(
                    edge_ids[edge_index],
                    edge_ids[(edge_index + 1) % edge_ids.len()],
                );
            }
        }

        for (&(vert_a, vert_b), &edge_id) in &endpoints_to_edges {
            // Retrieve the twin edge
            if let Some(&twin_id) = endpoints_to_edges.get(&(vert_b, vert_a)) {
                // Assign twins
                self.edge_twin.insert(edge_id, twin_id);
                self.edge_twin.insert(twin_id, edge_id);
            } else {
                return Err(MeshError::NoTwin(vert_a, vert_b));
            }
        }

        Ok(new_faces)
    }
}
