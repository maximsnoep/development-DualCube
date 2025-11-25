use crate::prelude::*;
use rand::seq::IteratorRandom;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use thiserror::Error;

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VERT;
pub type VertKey<M> = ids::Key<VERT, M>;

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FACE;
pub type FaceKey<M> = ids::Key<FACE, M>;

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EDGE;
pub type EdgeKey<M> = ids::Key<EDGE, M>;

#[derive(Error, Debug, Clone)]
pub enum MeshError<M> {
    #[error("({0}, {1}) does not have a twin (mesh is not a closed 2-manifold)")]
    NoTwin(VertKey<M>, VertKey<M>),
    #[error("({0}, {1}) exists multiple times (mesh is not a closed 2-manifold)")]
    DuplicateEdge(VertKey<M>, VertKey<M>),
    #[error("Mesh is not orientable")]
    NotOrientable,
    #[error("Mesh is not connected")]
    NotConnected,
    #[error("{0} is not a polygon (less than 3 vertices)")]
    FaceNotPolygon(FaceKey<M>),
    #[error("{0} is not planar (vertices are not coplanar)")]
    FaceNotPlanar(FaceKey<M>),
    #[error("{0} is not simple (edges intersect)")]
    FaceNotSimple(FaceKey<M>),
    #[error("Unknown error ({0})")]
    Unknown(String),
}

// Define a new trait that combines all required supertraits.
pub trait Tag: Default + Debug + Clone + Copy + PartialEq + Eq + std::hash::Hash {}
impl<T> Tag for T where T: Default + Debug + Clone + Copy + PartialEq + Eq + std::hash::Hash {}

#[macro_export]
macro_rules! define_tag {
    ($name:ident) => {
        #[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name;
    };
}

// This is a struct that defines a mesh with vertices, edges, and faces.
// This mesh is:
// 1) closed 2-manifold: Each edge corresponds to exactly two faces.
// 2) connected: There exists a path between any two vertices.
// 3) orientable: There exists a consistent normal for each face.
// These requirements will be true per construction.
// We use a doubly connected edge list (DCEL) data structure, also known as the half-edge data structure (HEDS).
#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct Mesh<M: Tag> {
    pub verts: ids::IdxMap<VERT, M, Vector3D>,
    pub edges: ids::IdxMap<EDGE, M, u8>,
    pub faces: ids::IdxMap<FACE, M, u8>,
    pub edge_root: ids::AssMap<EDGE, VERT, M>,
    pub edge_face: ids::AssMap<EDGE, FACE, M>,
    pub edge_next: ids::AssMap<EDGE, EDGE, M>,
    pub edge_twin: ids::AssMap<EDGE, EDGE, M>,
    pub vert_repr: ids::AssMap<VERT, EDGE, M>,
    pub face_repr: ids::AssMap<FACE, EDGE, M>,
}

impl<M: Tag> Mesh<M> {
    // Create mesh from other mesh
    pub fn convert<T: Tag>(other: &Mesh<T>) -> Self {
        Self {
            verts: ids::IdxMap::convert(&other.verts),
            edges: ids::IdxMap::convert(&other.edges),
            faces: ids::IdxMap::convert(&other.faces),
            edge_root: ids::AssMap::convert(&other.edge_root),
            edge_face: ids::AssMap::convert(&other.edge_face),
            edge_next: ids::AssMap::convert(&other.edge_next),
            edge_twin: ids::AssMap::convert(&other.edge_twin),
            vert_repr: ids::AssMap::convert(&other.vert_repr),
            face_repr: ids::AssMap::convert(&other.face_repr),
        }
    }

    // Adds a vertex to the mesh and returns its ID.
    pub fn add_vertex(&mut self, pos: Vector3D) -> VertKey<M> {
        self.verts.insert(pos)
    }

    // Adds an edge to the mesh and returns its ID.
    pub fn add_edge(&mut self) -> EdgeKey<M> {
        self.edges.insert(0)
    }

    // Adds a face to the mesh and returns its ID.
    pub fn add_face(&mut self) -> FaceKey<M> {
        self.faces.insert(0)
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
    pub fn vert_ids(&self) -> Vec<VertKey<M>> {
        self.verts.ids().collect()
    }

    #[must_use]
    pub fn edge_ids(&self) -> Vec<EdgeKey<M>> {
        self.edges.ids().collect()
    }

    #[must_use]
    pub fn face_ids(&self) -> Vec<FaceKey<M>> {
        self.faces.ids().collect()
    }

    // Return `n` random vertices.
    #[must_use]
    pub fn random_verts(&self, n: usize) -> Vec<VertKey<M>> {
        self.verts.ids().choose_multiple(&mut rand::rng(), n)
    }

    // Return `n` random edges.
    #[must_use]
    pub fn random_edges(&self, n: usize) -> Vec<EdgeKey<M>> {
        self.edges.ids().choose_multiple(&mut rand::rng(), n)
    }

    // Return `n` random faces.
    #[must_use]
    pub fn random_faces(&self, n: usize) -> Vec<FaceKey<M>> {
        self.faces.ids().choose_multiple(&mut rand::rng(), n)
    }

    // TODO: make this more ergonamic
    pub fn neighbor_function_primal(&self) -> impl Fn(VertKey<M>) -> Vec<VertKey<M>> + '_ {
        |v_id| self.neighbors(v_id)
    }

    // TODO: make this more ergonamic
    pub fn neighbor_function_edgegraph(&self) -> impl Fn(EdgeKey<M>) -> Vec<EdgeKey<M>> + '_ {
        |e_id| vec![self.next(e_id), self.next(self.next(e_id)), self.twin(e_id)]
    }

    // TODO: make this more ergonamic
    pub fn neighbor_function_edgepairgraph(&self) -> impl Fn([EdgeKey<M>; 2]) -> Vec<[EdgeKey<M>; 2]> + '_ {
        |[_, to]| {
            let next = self.twin(to);
            vec![[self.next(next), self.next(self.next(next))]]
        }
    }

    // List of all edges in the mesh (positions of endpoints)
    #[must_use]
    pub fn edges_positions(&self) -> Vec<(Vector3D, Vector3D)> {
        self.edge_ids()
            .iter()
            .map(|&edge_id| {
                let vertices = self.vertices(edge_id);
                let (u, v) = (vertices[0], vertices[1]);
                (self.position(u), self.position(v))
            })
            .collect()
    }

    #[must_use]
    pub fn get_aabb(&self) -> (Vector3D, Vector3D) {
        // An axis-aligned bounding box, defined by:
        // a center,
        // the distances from the center to each faces along the axis, the faces are orthogonal to the axis.

        let (min, max) = self
            .verts
            .vals()
            .fold((Vector3D::repeat(f64::INFINITY), Vector3D::repeat(f64::NEG_INFINITY)), |(min, max), v| {
                (min.zip_map(v, f64::min), max.zip_map(v, f64::max))
            });

        let center = (min + max) / 2.0;
        let half_extents = (max - min) / 2.0;
        (center, half_extents)
    }

    #[must_use]
    pub fn center(&self) -> Vector3D {
        let (center, _) = self.get_aabb();
        center
    }

    #[must_use]
    pub fn max_dim(&self) -> f64 {
        let (_, half_extents) = self.get_aabb();
        half_extents.max()
    }
}

pub trait SetPosition<K, M> {
    fn set_position(&mut self, id: ids::Key<K, M>, pos: Vector3D);
}

pub trait HasPosition<K, M> {
    #[must_use]
    fn position(&self, id: ids::Key<K, M>) -> Vector3D;
}

pub trait HasNormal<K, M> {
    #[must_use]
    fn normal(&self, id: ids::Key<K, M>) -> Vector3D;
}

pub trait HasSize<K, M> {
    #[must_use]
    fn size(&self, id: ids::Key<K, M>) -> Float;
}

pub trait HasVertices<K, M> {
    #[must_use]
    fn vertices(&self, id: ids::Key<K, M>) -> Vec<ids::Key<VERT, M>>;
}

pub trait HasEdges<K, M> {
    #[must_use]
    fn edges(&self, id: ids::Key<K, M>) -> Vec<ids::Key<EDGE, M>>;
}

pub trait HasFaces<K, M> {
    #[must_use]
    fn faces(&self, id: ids::Key<K, M>) -> Vec<ids::Key<FACE, M>>;
}

pub trait HasNeighbors<K, M> {
    #[must_use]
    fn neighbors(&self, id: ids::Key<K, M>) -> Vec<ids::Key<K, M>>;
    fn neighbors_k(&self, id: ids::Key<K, M>, k: usize) -> Vec<ids::Key<K, M>>;
}

pub trait HasRing<K, M> {
    #[must_use]
    fn ring(&self, id: ids::Key<K, M>, k: usize) -> Vec<Vec<ids::Key<K, M>>>;
}
