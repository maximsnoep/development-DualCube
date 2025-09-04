use std::sync::Arc;

use crate::Import;
use serde::{Deserialize, Serialize};
use slotmap::Key;
use slotmap::SecondaryMap;
use slotmap::SlotMap;

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct Douconel<VertID: Key, V, EdgeID: Key, E, FaceID: Key, F> {
    pub verts: SlotMap<VertID, V>,
    pub edges: SlotMap<EdgeID, E>,
    pub faces: SlotMap<FaceID, F>,

    edge_root: SecondaryMap<EdgeID, VertID>,
    edge_face: SecondaryMap<EdgeID, FaceID>,
    edge_next: SecondaryMap<EdgeID, EdgeID>,
    edge_twin: SecondaryMap<EdgeID, EdgeID>,

    vert_rep: SecondaryMap<VertID, EdgeID>,
    face_rep: SecondaryMap<FaceID, EdgeID>,
}

type Float = f64;
type Vector3D = nalgebra::SVector<Float, 3>;

pub trait HasPosition {
    fn position(&self) -> Vector3D;
    fn set_position(&mut self, position: Vector3D);
}

// Embedded vertices (have a position)
#[derive(Default, Copy, Clone, Debug, Serialize, Deserialize)]
pub struct EmbeddedVertex {
    position: Vector3D,
}

impl HasPosition for EmbeddedVertex {
    fn position(&self) -> Vector3D {
        self.position
    }
    fn set_position(&mut self, position: Vector3D) {
        self.position = position;
    }
}

pub type Empty = u8;
slotmap::new_key_type! {
    pub struct VertID;
    pub struct EdgeID;
    pub struct FaceID;
}

pub type EmbeddedMesh = Douconel<VertID, EmbeddedVertex, EdgeID, Empty, FaceID, Empty>;

// Principal directions, used to characterize a polycube (each edge and face is associated with a principal direction)
#[derive(Copy, Clone, Default, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
pub enum PrincipalDirection {
    #[default]
    X,
    Y,
    Z,
}

// A loop forms the basis of the dual structure.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Loop {
    // A loop is defined by a sequence of half-edges.
    pub edges: Vec<EdgeID>,
    // the direction or labeling associated with the loop
    pub direction: PrincipalDirection,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SaveStateObject {
    mesh: EmbeddedMesh,
    loops: Vec<Loop>,
}

impl<VertID: slotmap::Key, V: Default, EdgeID: Key, E: Default, FaceID: Key, F: Default> Douconel<VertID, V, EdgeID, E, FaceID, F> {
    // Returns the "representative" edge of the given vertex.
    // Panics if the vertex has no representative edge defined.
    #[must_use]
    pub fn vrep(&self, id: VertID) -> EdgeID {
        self.vert_rep.get(id).copied().unwrap_or_else(|| panic!("{id:?} has no vrep"))
    }

    // Returns the "representative" edge of the given face.
    // Panics if the face has no representative edge defined.
    #[must_use]
    pub fn frep(&self, id: FaceID) -> EdgeID {
        self.face_rep.get(id).copied().unwrap_or_else(|| panic!("{id:?} has no frep"))
    }

    // Returns the root vertex of the given edge.
    // Panics if the edge has no root defined or if the root does not exist.
    #[must_use]
    pub fn root(&self, id: EdgeID) -> VertID {
        self.edge_root.get(id).copied().unwrap_or_else(|| panic!("{id:?} has no root"))
    }

    // Returns the root of the twin edge of the given edge. (also named toor, reverse of root)
    // Panics if the edge has no twin defined or if the twin does not exist.
    #[must_use]
    pub fn toor(&self, id: EdgeID) -> VertID {
        self.root(self.twin(id))
    }

    // Returns the twin edge of the given edge.
    // Panics if the edge has no twin defined or if the twin does not exist.
    #[must_use]
    pub fn twin(&self, id: EdgeID) -> EdgeID {
        self.edge_twin.get(id).copied().unwrap_or_else(|| panic!("{id:?} has no twin"))
    }

    // Returns the next edge of the given edge.
    // Panics if the edge has no next defined or if the next does not exist.
    #[inline]
    #[must_use]
    pub fn next(&self, id: EdgeID) -> EdgeID {
        self.edge_next.get(id).copied().unwrap_or_else(|| panic!("{id:?} has no next"))
    }

    #[inline]
    #[must_use]
    pub fn nexts(&self, id: EdgeID) -> Vec<EdgeID> {
        let mut nexts = vec![];
        let mut cur = id;
        loop {
            cur = self.next(cur);
            if cur == id {
                return nexts;
            }
            nexts.push(cur);
        }
    }

    // Returns the four edges around a given edge.
    #[inline]
    #[must_use]
    pub fn quad(&self, id: EdgeID) -> [EdgeID; 4] {
        let edge0 = self.next(id);
        let edge1 = self.next(edge0);
        let twin = self.twin(id);
        let edge2 = self.next(twin);
        let edge3 = self.next(edge2);
        [edge0, edge1, edge2, edge3]
    }

    // Returns the face of the given edge.
    // Panics if the edge has no face defined or if the face does not exist.
    #[inline]
    #[must_use]
    pub fn face(&self, id: EdgeID) -> FaceID {
        self.edge_face.get(id).copied().unwrap_or_else(|| panic!("{id:?} has no face"))
    }

    // Returns the start and end vertex IDs of the given edge.
    // Panics if any of the roots are not defined or do not exist.
    #[inline]
    #[must_use]
    pub fn endpoints(&self, id: EdgeID) -> (VertID, VertID) {
        (self.root(id), self.root(self.twin(id)))
    }

    // Returns the corner vertices of a given face.
    #[inline]
    #[must_use]
    pub fn corners(&self, id: FaceID) -> Vec<VertID> {
        self.edges(id).into_iter().map(|edge_id| self.root(edge_id)).collect()
    }

    // Returns the outgoing edges of a given vertex. (clockwise order)
    #[inline]
    #[must_use]
    pub fn outgoing(&self, id: VertID) -> Vec<EdgeID> {
        let mut edges = vec![self.vrep(id)];
        loop {
            let next_of_twin = self.next(self.twin(edges.last().copied().unwrap()));
            if edges.contains(&next_of_twin) {
                return edges;
            }
            edges.push(next_of_twin);
        }
    }

    // Returns the edges of a given face. (anticlockwise order)
    #[inline]
    #[must_use]
    pub fn edges(&self, id: FaceID) -> Vec<EdgeID> {
        [vec![self.frep(id)], self.nexts(self.frep(id))].concat()
    }

    // Returns the faces around a given vertex. (clockwise order)
    #[inline]
    #[must_use]
    pub fn star(&self, id: VertID) -> Vec<FaceID> {
        self.outgoing(id).iter().map(|&edge_id| self.face(edge_id)).collect()
    }

    // Returns the faces around a given edge.
    #[inline]
    #[must_use]
    pub fn faces(&self, id: EdgeID) -> [FaceID; 2] {
        [self.face(id), self.face(self.twin(id))]
    }

    #[inline]
    #[must_use]
    pub fn common_endpoint(&self, edge_a: EdgeID, edge_b: EdgeID) -> Option<VertID> {
        let (a0, a1) = self.endpoints(edge_a);
        let (b0, b1) = self.endpoints(edge_b);
        if a0 == b0 || a0 == b1 {
            Some(a0)
        } else if a1 == b0 || a1 == b1 {
            Some(a1)
        } else {
            None
        }
    }

    // Returns the edge between the two vertices. Returns None if the vertices are not connected.
    #[inline]
    #[must_use]
    pub fn edge_between_verts(&self, id_a: VertID, id_b: VertID) -> Option<(EdgeID, EdgeID)> {
        for &edge_a_id in &self.outgoing(id_a) {
            for &edge_b_id in &self.outgoing(id_b) {
                if self.twin(edge_a_id) == edge_b_id {
                    return Some((edge_a_id, edge_b_id));
                }
            }
        }
        None
    }

    // Returns the edge between the two faces. Returns None if the faces do not share an edge.
    #[must_use]
    pub fn edge_between_faces(&self, id_a: FaceID, id_b: FaceID) -> Option<(EdgeID, EdgeID)> {
        let edges_a = self.edges(id_a);
        let edges_b = self.edges(id_b);
        for &edge_a_id in &edges_a {
            for &edge_b_id in &edges_b {
                if self.twin(edge_a_id) == edge_b_id {
                    return Some((edge_a_id, edge_b_id));
                }
            }
        }
        None
    }

    // Returns the neighbors of a given vertex.
    #[must_use]
    pub fn vneighbors(&self, id: VertID) -> Vec<VertID> {
        self.outgoing(id).iter().map(|&edge_id| self.root(self.twin(edge_id))).collect()
    }

    // Returns the (edge-wise) neighbors of a given face.
    #[must_use]
    pub fn fneighbors(&self, id: FaceID) -> Vec<FaceID> {
        self.edges(id).into_iter().map(|edge_id| self.face(self.twin(edge_id))).collect()
    }

    // Returns the number of vertices in the mesh.
    #[must_use]
    pub fn nr_verts(&self) -> usize {
        self.verts.len()
    }

    // Returns the number of (half)edges in the mesh.
    #[must_use]
    pub fn nr_edges(&self) -> usize {
        self.edges.len()
    }

    // Returns the number of faces in the mesh.
    #[must_use]
    pub fn nr_faces(&self) -> usize {
        self.faces.len()
    }

    #[must_use]
    pub fn vert_ids(&self) -> Vec<VertID> {
        self.verts.keys().collect()
    }

    #[must_use]
    pub fn edge_ids(&self) -> Vec<EdgeID> {
        self.edges.keys().collect()
    }

    #[must_use]
    pub fn face_ids(&self) -> Vec<FaceID> {
        self.faces.keys().collect()
    }
}

pub struct BackwardsCompatibility;

impl Import for BackwardsCompatibility {
    fn import(path: &std::path::Path) -> Result<dualcube::prelude::Solution, Box<dyn std::error::Error>> {
        let res = serde_json::from_reader::<_, SaveStateObject>(std::fs::File::open(path)?);
        if let Ok(loaded_state) = res {
            // Create the mesh
            let original_vert_map = SecondaryMap::<VertID, usize>::from_iter(loaded_state.mesh.verts.keys().enumerate().map(|(i, vert_id)| (vert_id, i)));

            let fs = loaded_state
                .mesh
                .face_ids()
                .into_iter()
                .map(|face_id| {
                    loaded_state
                        .mesh
                        .corners(face_id)
                        .into_iter()
                        .map(|vert_id| original_vert_map.get(vert_id).unwrap().to_owned())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let vs = loaded_state.mesh.verts.values().map(|v| v.position()).collect::<Vec<_>>();

            let mesh_result = mehsh::prelude::Mesh::<dualcube::prelude::INPUT>::from(&fs, &vs);
            let Ok((mesh, vert_map, _)) = mesh_result else {
                return Err(format!("Reading invalid mesh: {:?}", mesh_result).into());
            };

            let mut solution = dualcube::prelude::Solution::new(Arc::new(mesh.clone()));

            // Reconstruct the loops
            for lewp in loaded_state.loops {
                let edges = lewp
                    .edges
                    .into_iter()
                    .map(|edge_id| {
                        let (start_vert, end_vert) = loaded_state.mesh.endpoints(edge_id);
                        // Use the vert_map to get the correct VertID for the edge
                        let start_usize = original_vert_map.get(start_vert).unwrap().to_owned();
                        let start_key = vert_map.key(start_usize).unwrap().to_owned();
                        let end_usize = original_vert_map.get(end_vert).unwrap().to_owned();
                        let end_key = vert_map.key(end_usize).unwrap().to_owned();
                        mesh.edge_between_verts(start_key, end_key).unwrap().0
                    })
                    .collect::<Vec<_>>();
                let direction = match lewp.direction {
                    PrincipalDirection::X => dualcube::prelude::PrincipalDirection::X,
                    PrincipalDirection::Y => dualcube::prelude::PrincipalDirection::Y,
                    PrincipalDirection::Z => dualcube::prelude::PrincipalDirection::Z,
                };

                solution.add_loop(dualcube::solutions::Loop { edges, direction });
            }

            return Ok(solution);
        }
        Err(Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "Failed to load save state")))
    }
}
