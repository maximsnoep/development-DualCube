use crate::dual::Dual;
use crate::polycube::{Polycube, POLYCUBE};
use crate::prelude::*;
use bimap::BiHashMap;
use grapff::Grapff;
use itertools::Itertools;
use mehsh::prelude::*;
use ordered_float::OrderedFloat;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Eq, Hash, Copy)]
pub enum NodeType {
    Vertex(VertID),
    Face(FaceID),
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Patch {
    // A patch is defined by a set of faces
    pub faces: HashSet<FaceID>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Layout {
    // TODO: make this an actual (arc) reference
    pub polycube_ref: Polycube,
    // TODO: make this an actual (arc) reference
    pub dual_ref: Dual,

    // Mapping:
    pub granulated_mesh: Mesh<INPUT>,
    pub vert_to_corner: BiHashMap<VertKey<POLYCUBE>, VertID>,
    pub edge_to_path: HashMap<EdgeKey<POLYCUBE>, Vec<VertID>>,
    pub face_to_patch: HashMap<FaceKey<POLYCUBE>, Patch>,

    // Quality:
    pub alignment_per_triangle: ids::SecMap<FACE, INPUT, f64>,
    pub alignment: Option<f64>,
    pub orthogonality_per_vert: ids::SecMap<VERT, POLYCUBE, f64>,
    pub orthogonality: Option<f64>,
}

#[derive(Debug, Error, Clone, Serialize, Deserialize)]
pub enum LayoutError {
    #[error("Unknown error")]
    UnknownError,
    #[error("Computed path is invalid, or no valid path could be found.")]
    InvalidPath,
    #[error("Computed patch is invalid, or no valid patch could be found.")]
    InvalidPatches,
}

impl Layout {
    pub fn new(dual_ref: &Dual, polycube_ref: &Polycube) -> Self {
        Self {
            polycube_ref: polycube_ref.clone(),
            dual_ref: dual_ref.clone(),
            granulated_mesh: (*dual_ref.mesh_ref).clone(),
            vert_to_corner: BiHashMap::new(),
            face_to_patch: HashMap::new(),
            edge_to_path: HashMap::new(),
            alignment_per_triangle: ids::SecMap::new(),
            alignment: None,
            orthogonality_per_vert: ids::SecMap::new(),
            orthogonality: None,
        }
    }

    /// Takes a dual representation, and a primal representation (polycube) and embeds it onto the input mesh.
    pub fn embed(dual_ref: &Dual, polycube_ref: &Polycube) -> Result<Self, LayoutError> {
        let mut layout = Self {
            polycube_ref: polycube_ref.clone(),
            dual_ref: dual_ref.clone(),
            granulated_mesh: (*dual_ref.mesh_ref).clone(),
            vert_to_corner: BiHashMap::new(),
            face_to_patch: HashMap::new(),
            edge_to_path: HashMap::new(),
            alignment_per_triangle: ids::SecMap::new(),
            alignment: None,
            orthogonality_per_vert: ids::SecMap::new(),
            orthogonality: None,
        };
        layout.place_all_corners();
        layout.place_all_paths()?;
        layout.assign_all_patches()?;
        Ok(layout)
    }

    pub fn place_all_corners(&mut self) {
        // Clear the mapping
        self.vert_to_corner.clear();
        self.edge_to_path.clear();
        self.face_to_patch.clear();

        // Find a candidate location for each region
        // We know for each loop region what are going to be the aligned directions of the patches
        // For each vertex in the region, we count the number of (relevant) directions adjacent to it
        // Then the candidates for this loop region are vertices with the highest count
        // If the loop region is a flat corner, we dont care, and take all vertices

        let mut region_to_labels = HashMap::new();
        let mut region_to_candidates = HashMap::new();
        for region_id in self.dual_ref.loop_structure.face_ids() {
            // Get the relevant directions for this region
            let polycube_vert = self.polycube_ref.region_to_vertex.get_by_left(&region_id).unwrap().to_owned();
            let polycube_faces = self.polycube_ref.structure.faces(polycube_vert);

            // Super strict vertex placement:

            let face_labels = polycube_faces
                .iter()
                .map(|&f| to_principal_direction(self.polycube_ref.structure.normal(f)))
                .collect::<HashSet<_>>()
                .into_iter()
                .collect_vec();
            region_to_labels.insert(region_id, face_labels.clone());

            // Get all vertices in the region
            let verts = self.dual_ref.region_to_verts(region_id);

            if face_labels.len() <= 1 {
                region_to_candidates.insert(region_id, verts.clone().into_iter().collect_vec());
            } else {
                // Count the number of relevant directions adjacent to each vertex
                let mut vertex_to_count = HashMap::new();

                for &vert in &verts {
                    let mut scores = vec![0.; face_labels.len()];
                    let normals = self
                        .dual_ref
                        .mesh_ref
                        .faces(vert)
                        .into_iter()
                        .map(|face| self.dual_ref.mesh_ref.normal(face))
                        .collect_vec();

                    for normal in normals {
                        for i in 0..face_labels.len() {
                            let angle = to_vector(face_labels[i].0, face_labels[i].1).angle(&normal);
                            if angle < 1. {
                                let score = (2. - angle).powi(2);
                                if score > scores[i] {
                                    scores[i] = score;
                                }
                            }
                        }
                    }

                    vertex_to_count.insert(vert, scores.iter().product::<f64>());
                }

                // Get the highest score
                let max_score = *vertex_to_count.values().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

                // Get all vertices with the highest count
                let candidates = vertex_to_count
                    .iter()
                    .filter(|&(_, &count)| count >= max_score * 0.9)
                    .map(|(&v, _)| v)
                    .collect_vec();

                region_to_candidates.insert(region_id, candidates);
            }
        }

        // Less strict:

        // let face_labels = polycube_faces
        //         .iter()
        //         .map(|&f| to_principal_direction(self.polycube_ref.structure.normal(f)))
        //         .collect::<HashSet<_>>();

        //     // Get all vertices in the region
        //     let verts = self.dual_ref.region_to_verts(region_id);

        //     if face_labels.len() == 1 {
        //         region_to_candidates.insert(region_id, verts.clone().into_iter().collect_vec());
        //     } else {
        //         // Count the number of relevant directions adjacent to each vertex
        //         let mut vertex_to_count = HashMap::new();
        //         for &vert in &verts {
        //             let labels = self
        //                 .dual_ref
        //                 .mesh_ref
        //                 .faces(vert)
        //                 .into_iter()
        //                 .map(|face| {
        //                     let normal = self.dual_ref.mesh_ref.normal(face);
        //                     to_principal_direction(normal)
        //                 })
        //                 .collect::<HashSet<_>>();

        //             // Count the number of relevant directions adjacent to this vertex
        //             let positive_count = labels.clone().into_iter().filter(|&d| face_labels.contains(&d)).count() as i32;
        //             let negative_count = labels.clone().into_iter().filter(|&d| !face_labels.contains(&d)).count() as i32;
        //             vertex_to_count.insert(vert, positive_count - negative_count);
        //         }

        //         // Get the highest count
        //         let max_count = *vertex_to_count.values().max().unwrap();

        //         // Get all vertices with the highest count
        //         let candidates = vertex_to_count.iter().filter(|&(_, &count)| count == max_count).map(|(&v, _)| v).collect_vec();

        //         region_to_candidates.insert(region_id, candidates);
        //     }
        // }

        // For each zone, find a candidate slice (value), that minimizes the Hausdorf distance to the candidate locations of the regions in the zone
        // We simply take the coordinate that minimizes the Hausdorf distance to the candidate locations of the regions in the zone
        let mut zone_to_candidate = HashMap::new();
        for (zone_id, zone_obj) in &self.dual_ref.level_graphs.zones {
            let zone_type = zone_obj.direction;

            let irregular_corners_exist = zone_obj
                .regions
                .iter()
                .filter(|&&region_id| {
                    let face_labels = region_to_labels[&region_id].clone();
                    face_labels.len() > 2
                })
                .count()
                > 0;

            // Get all coordinates of the regions in the zone
            let zone_regions_with_candidates = zone_obj
                .regions
                .iter()
                .filter_map(|&region_id| {
                    let face_labels = region_to_labels[&region_id].clone();
                    if face_labels.len() <= 2 && irregular_corners_exist {
                        None
                    } else {
                        Some(
                            region_to_candidates[&region_id]
                                .iter()
                                .map(|&v| self.dual_ref.mesh_ref.position(v)[zone_type as usize])
                                .collect_vec(),
                        )
                    }
                })
                .collect_vec();

            // Find the coordinate that minimizes the worst distance to all regions (defined by candidates), do this in N steps
            let n = 100;
            let min = zone_regions_with_candidates
                .iter()
                .flatten()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
                .to_owned();
            let max = zone_regions_with_candidates
                .iter()
                .flatten()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
                .to_owned();
            let steps = (0..n).map(|i| (max - min).mul_add(f64::from(i) / f64::from(n), min)).collect_vec();
            let mut best_step = min;
            let mut best_worst_distance = f64::INFINITY;
            for step in steps {
                let mut worst_distance_for_step = 0.;
                for region_with_candidates in &zone_regions_with_candidates {
                    let best_distance_to_region = region_with_candidates
                        .iter()
                        .map(|&candidate| (step - candidate).abs())
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap();
                    if best_distance_to_region > worst_distance_for_step {
                        worst_distance_for_step = best_distance_to_region;
                    }
                }
                if worst_distance_for_step < best_worst_distance {
                    best_worst_distance = worst_distance_for_step;
                    best_step = step;
                }
            }

            zone_to_candidate.insert(zone_id, best_step);
        }

        // Find the actual vertex in the subsurface that is closest to the candidate location (by combining the three candidate coordinates of corresponding zones)
        for region_id in self.dual_ref.loop_structure.face_ids() {
            let target = Vector3D::from(
                [PrincipalDirection::X, PrincipalDirection::Y, PrincipalDirection::Z]
                    .map(|direction| zone_to_candidate[&self.dual_ref.region_to_zone(region_id, direction)]),
            );

            let vertices = &region_to_candidates[&region_id];
            // let vertices = region_obj.verts.clone();

            let best_vertex = vertices
                .iter()
                .map(|&v| (v, self.dual_ref.mesh_ref.position(v)))
                .map(|(v, pos)| (v, pos.metric_distance(&target)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap()
                .0;

            self.vert_to_corner
                .insert(self.polycube_ref.region_to_vertex.get_by_left(&region_id).unwrap().to_owned(), best_vertex);
        }
    }

    pub fn compute_path(
        &self,
        edge_id: EdgeKey<POLYCUBE>,
        occupied_vertices: &HashSet<VertID>,
        occupied_edges: &HashSet<(VertID, VertID)>,
        occupied_faces: &HashSet<FaceID>,
    ) -> Result<(Vec<VertID>, Mesh<INPUT>), LayoutError> {
        let polycube = &self.polycube_ref.structure;
        let mut granulated_mesh = self.granulated_mesh.clone();

        let endpoints = polycube.vertices(edge_id);
        let (Some(&u), Some(&v)) = (self.vert_to_corner.get_by_left(&endpoints[0]), self.vert_to_corner.get_by_left(&endpoints[1])) else {
            return Err(LayoutError::UnknownError);
        };

        // Neighborhood function
        let n_function = |node: NodeType| match node {
            NodeType::Face(f_id) => {
                let f_neighbors: Vec<NodeType> = {
                    // Disallow occupied faces
                    if occupied_faces.contains(&f_id) {
                        return vec![];
                    }
                    // Only allowed if the edge between the two faces is not occupied.
                    let blocked = |f1: FaceID, f2: FaceID| {
                        let (edge_id, _) = granulated_mesh.edge_between_faces(f1, f2).unwrap();
                        let endpoints = granulated_mesh.vertices(edge_id);
                        occupied_edges.contains(&(endpoints[0], endpoints[1]))
                    };
                    granulated_mesh
                        .neighbors(f_id)
                        .into_iter()
                        .filter(|&n_id| !blocked(f_id, n_id))
                        .map(NodeType::Face)
                        .collect_vec()
                };
                let v_neighbors = granulated_mesh.vertices(f_id).into_iter().map(NodeType::Vertex).collect_vec();
                [v_neighbors, f_neighbors].concat()
            }
            NodeType::Vertex(v_id) => {
                // Only allowed if the vertex is not occupied
                if occupied_vertices.contains(&v_id) && v_id != u && v_id != v {
                    return vec![];
                }
                let v_neighbors = granulated_mesh.neighbors(v_id).into_iter().map(NodeType::Vertex).collect_vec();
                let f_neighbors = granulated_mesh.faces(v_id).into_iter().map(NodeType::Face).collect_vec();
                [v_neighbors, f_neighbors].concat()
            }
        };

        // Weight function
        let nodetype_to_pos = |node: NodeType| match node {
            NodeType::Face(f_id) => granulated_mesh.position(f_id),
            NodeType::Vertex(v_id) => granulated_mesh.position(v_id),
        };
        let normal_on_left = polycube.normal(polycube.face(edge_id));
        let normal_on_right = polycube.normal(polycube.face(polycube.twin(edge_id)));
        let angle_between_normals = normal_on_left.angle(&normal_on_right);
        let ridge_function = |a: NodeType, b: NodeType| {
            let (normal1, normal2) = match (a, b) {
                (NodeType::Vertex(a), NodeType::Vertex(b)) => {
                    let (edge1, edge2) = granulated_mesh.edge_between_verts(a, b).unwrap();
                    let normal1 = granulated_mesh.normal(granulated_mesh.face(edge1));
                    let normal2 = granulated_mesh.normal(granulated_mesh.face(edge2));
                    (normal1, normal2)
                }
                (NodeType::Face(f), NodeType::Face(_)) => {
                    let normal = granulated_mesh.normal(f);
                    (normal, normal)
                }
                (NodeType::Face(f), NodeType::Vertex(_)) => {
                    let normal = granulated_mesh.normal(f);
                    (normal, normal)
                }
                (NodeType::Vertex(_), NodeType::Face(f)) => {
                    let normal = granulated_mesh.normal(f);
                    (normal, normal)
                }
            };

            let angle = normal1.angle(&normal2);
            let angle1 = normal1.angle(&normal_on_left);
            let angle2 = normal2.angle(&normal_on_right);
            let difference_between_angles = (angle - angle_between_normals).abs();

            if difference_between_angles < std::f64::consts::PI / 4.0 && angle1 + angle2 < std::f64::consts::PI / 4.0 {
                0.5
            } else {
                1.
            }
        };
        let w_function = |(a, b)| OrderedFloat(ridge_function(a, b) * nodetype_to_pos(a).metric_distance(&nodetype_to_pos(b)));

        let result = {
            let nn = grapff::fluid::FluidGraph::new(n_function);
            nn.shortest_path(NodeType::Vertex(u), NodeType::Vertex(v), w_function)
        };

        if result.is_none() {
            return Err(LayoutError::InvalidPath);
        }

        let path = result.unwrap().0;
        let mut granulated_path = vec![];

        let mut last_f_ids_maybe: Option<[FaceID; 3]> = None;
        for node in path {
            match node {
                NodeType::Vertex(v_id) => {
                    granulated_path.push((v_id, false));
                    last_f_ids_maybe = None;
                }
                NodeType::Face(f_id) => {
                    let new_v_pos = granulated_mesh.position(f_id);
                    let (new_v_id, new_f_ids) = granulated_mesh.split_face(f_id);
                    granulated_mesh.set_position(new_v_id, new_v_pos);
                    if let Some(last_f_ids) = last_f_ids_maybe {
                        for last_f_id in last_f_ids {
                            for new_f_id in new_f_ids {
                                if let Some((edge_id, _)) = granulated_mesh.edge_between_faces(last_f_id, new_f_id) {
                                    let midpoint_of_edge = granulated_mesh.position(edge_id);
                                    let (mid_v_id, _) = granulated_mesh.split_edge(edge_id);
                                    granulated_mesh.set_position(mid_v_id, midpoint_of_edge);
                                    granulated_path.push((mid_v_id, false));
                                }
                            }
                        }
                    }

                    last_f_ids_maybe = Some(new_f_ids);
                    granulated_path.push((new_v_id, true));
                }
            }
        }

        let granulated_path = granulated_path.into_iter().map(|(v_id, _)| v_id).collect_vec();

        if granulated_path.is_empty() {
            return Err(LayoutError::InvalidPath);
        }

        Ok((granulated_path, granulated_mesh))
    }

    pub fn place_path(&mut self, edge_id: EdgeKey<POLYCUBE>) -> Result<(), LayoutError> {
        let (occupied_vertices, occupied_edges) = self.compute_occupied();

        let (path, granulated_mesh) = self.compute_path(edge_id, &occupied_vertices, &occupied_edges, &HashSet::new())?;
        let path_reversed = path.clone().into_iter().rev().collect_vec();

        // Update the granulated mesh
        self.granulated_mesh = granulated_mesh;

        // Insert the calculated path
        self.edge_to_path.insert(edge_id, path);
        // Also insert for the twin, the calculated path
        self.edge_to_path.insert(self.polycube_ref.structure.twin(edge_id), path_reversed);

        Ok(())
    }

    // TODO: Make this robust
    pub fn place_all_paths(&mut self) -> Result<(), LayoutError> {
        let primal = &self.polycube_ref;

        self.edge_to_path.clear();

        for vert in primal.structure.vert_ids() {
            assert!(
                self.vert_to_corner.get_by_left(&vert).is_some(),
                "Missing corner vertex for primal vertex {:?}",
                vert
            );
        }

        let mut edge_queue = primal.structure.edge_ids();

        edge_queue.shuffle(&mut rand::rng());

        let mut edge_queue = VecDeque::from(edge_queue);

        let mut first_separating_edge = None;
        let mut is_maximal = false;

        let mut counter = 0;

        while let Some(edge_id) = edge_queue.pop_front() {
            //println!("Edge queue: {}", edge_queue.len());

            let (mut occupied_vertices, occupied_edges) = self.compute_occupied();

            // if already found (because of twin), skip
            if self.edge_to_path.contains_key(&edge_id) {
                continue;
            }

            if counter > 1000 {
                return Err(LayoutError::InvalidPath);
            }
            counter += 1;

            // check if edge is separating (in combination with the edges already done)
            let covered_edges = self.edge_to_path.keys().chain([&edge_id]).collect::<HashSet<_>>();

            let ccs = grapff::fluid::FluidGraph::new(|face_id| {
                primal
                    .structure
                    .neighbors(face_id)
                    .into_iter()
                    .filter(|&n_id| !covered_edges.contains(&primal.structure.edge_between_faces(face_id, n_id).unwrap().0))
                    .collect()
            })
            .connected_components(&primal.structure.face_ids());

            //println!("cc: {:?} == {:?}", cc.len(), primal.structure.faces.len());

            if !is_maximal && first_separating_edge == Some(edge_id) {
                is_maximal = true;
            }

            if ccs.len() != 1 && !is_maximal {
                // separating edge -> add to the end of the queue
                if first_separating_edge.is_none() {
                    first_separating_edge = Some(edge_id);
                }
                edge_queue.push_back(edge_id);
                continue;
            }

            let endpoints = primal.structure.vertices(edge_id);

            let (u, v) = (
                self.vert_to_corner.get_by_left(&endpoints[0]).unwrap().to_owned(),
                self.vert_to_corner.get_by_left(&endpoints[1]).unwrap().to_owned(),
            );

            // Find edge in `u_new`
            let edges_done_in_u_new = primal
                .structure
                .edges(endpoints[0])
                .into_iter()
                .filter(|&e| self.edge_to_path.contains_key(&e) || e == edge_id)
                .collect_vec();

            let mut occupied_faces = HashSet::new();
            // If this is 3 or larger, this means we must make sure the new edge is placed inbetween existing edges, in the correct order
            if edges_done_in_u_new.len() >= 3 {
                // Find the edge that is "above" the new edge
                let edge_id_position = edges_done_in_u_new.iter().position(|&e| e == edge_id).unwrap();
                let above = (edge_id_position + 1) % edges_done_in_u_new.len();
                let below = (edge_id_position + edges_done_in_u_new.len() - 1) % edges_done_in_u_new.len();
                // find above edge in the granulated mesh
                let above_edge_id = edges_done_in_u_new[above];
                let above_edge_obj = self.edge_to_path.get(&above_edge_id).unwrap();
                let above_edge_start = above_edge_obj[0];
                assert!(above_edge_start == u);
                let above_edge_start_plus_one = above_edge_obj[1];
                let above_edge_real_edge = self.granulated_mesh.edge_between_verts(above_edge_start, above_edge_start_plus_one).unwrap().0;
                // find below edge in the granulated mesh
                let below_edge_id = edges_done_in_u_new[below];
                let below_edge_obj = self.edge_to_path.get(&below_edge_id).unwrap();
                let below_edge_start = below_edge_obj[0];
                assert!(below_edge_start == u);
                let below_edge_start_plus_one = below_edge_obj[1];
                let below_edge_real_edge = self.granulated_mesh.edge_between_verts(below_edge_start, below_edge_start_plus_one).unwrap().0;
                // so starting from below edge, we insert all faces up until the above edge
                let all_edges = self
                    .granulated_mesh
                    .edges(u)
                    .into_iter()
                    .flat_map(|e| [e, self.granulated_mesh.twin(e)])
                    .collect_vec();
                let allowed_edges = all_edges
                    .into_iter()
                    .cycle()
                    .skip_while(|&e| e != below_edge_real_edge)
                    .skip(1)
                    .take_while(|&e| e != above_edge_real_edge)
                    .collect_vec();
                let allowed_faces = allowed_edges.into_iter().map(|e| self.granulated_mesh.face(e)).collect_vec();
                assert!(!allowed_faces.is_empty());
                for face_id in self.granulated_mesh.faces(u) {
                    if !allowed_faces.contains(&face_id) {
                        occupied_faces.insert(face_id);
                    }
                }
            }

            let twin_id = primal.structure.twin(edge_id);
            // Find edge in `v_new`
            let edges_done_in_v_new = primal
                .structure
                .edges(endpoints[1])
                .into_iter()
                .filter(|&e| self.edge_to_path.contains_key(&e) || e == twin_id)
                .collect_vec();

            // If this is 3 or larger, this means we must make sure the new edge is placed inbetween existing edges, in the correct order
            if edges_done_in_v_new.len() >= 3 {
                // Find the edge that is "above" the new edge
                let edge_id_position = edges_done_in_v_new.iter().position(|&e| e == twin_id).unwrap();
                let above = (edge_id_position + 1) % edges_done_in_v_new.len();
                let below = (edge_id_position + edges_done_in_v_new.len() - 1) % edges_done_in_v_new.len();
                // find above edge in the granulated mesh
                let above_edge_id = edges_done_in_v_new[above];
                let above_edge_obj = self.edge_to_path.get(&above_edge_id).unwrap();
                let above_edge_start = above_edge_obj[0];
                assert!(above_edge_start == v);
                let above_edge_start_plus_one = above_edge_obj[1];
                let above_edge_real_edge = self.granulated_mesh.edge_between_verts(above_edge_start, above_edge_start_plus_one).unwrap().0;
                // find below edge in the granulated mesh
                let below_edge_id = edges_done_in_v_new[below];
                let below_edge_obj = self.edge_to_path.get(&below_edge_id).unwrap();
                let below_edge_start = below_edge_obj[0];
                assert!(below_edge_start == v);
                let below_edge_start_plus_one = below_edge_obj[1];
                let below_edge_real_edge = self.granulated_mesh.edge_between_verts(below_edge_start, below_edge_start_plus_one).unwrap().0;
                // so starting from below edge, we insert all faces up until the above edge
                let all_edges = self
                    .granulated_mesh
                    .edges(v)
                    .into_iter()
                    .flat_map(|e| [e, self.granulated_mesh.twin(e)])
                    .collect_vec();
                let allowed_edges = all_edges
                    .into_iter()
                    .cycle()
                    .skip_while(|&e| e != below_edge_real_edge)
                    .skip(1)
                    .take_while(|&e| e != above_edge_real_edge)
                    .collect_vec();
                let allowed_faces = allowed_edges.into_iter().map(|e| self.granulated_mesh.face(e)).collect_vec();
                assert!(!allowed_faces.is_empty());
                for face_id in self.granulated_mesh.faces(v) {
                    if !allowed_faces.contains(&face_id) {
                        occupied_faces.insert(face_id);
                    }
                }
            }

            for &occupied_face in &occupied_faces {
                occupied_vertices.extend(self.granulated_mesh.vertices(occupied_face));
            }

            println!("computing for {:?}", edge_id);

            let (path, granulated_mesh) = self.compute_path(edge_id, &occupied_vertices, &occupied_edges, &occupied_faces)?;
            let path_reversed = path.clone().into_iter().rev().collect_vec();

            // Update the granulated mesh
            self.granulated_mesh = granulated_mesh;

            // Insert the calculated path
            self.edge_to_path.insert(edge_id, path);
            // Also insert for the twin, the calculated path
            self.edge_to_path.insert(self.polycube_ref.structure.twin(edge_id), path_reversed);

            counter = 0;
        }
        Ok(())
    }

    // pub fn random_improvement_corner(&mut self, polycube_vert: VertKey<POLYCUBE>) -> Result<(), PropertyViolationError> {
    //     // INVARIANT: ALL CORNERS AND PATHS MUST ALREADY BE PLACED
    //     let vertex = self.vert_to_corner.get_by_left(&polycube_vert).unwrap().to_owned();

    //     // grab all vertices in the k-neighborhood
    //     let candidate_vertices = self.dual_ref.mesh_ref.neighbors_k(vertex, 3);

    //     // grab random vertex from candidate_vertices
    //     let new_vertex = candidate_vertices.choose(&mut rand::rng()).unwrap().to_owned();

    //     self.move_corner(polycube_vert, new_vertex)?;
    //     Ok(())
    // }

    pub fn assign_all_patches(&mut self) -> Result<(), LayoutError> {
        // Verify the paths
        self.verify_paths()?;

        // Get all blocked edges (ALL PATHS)
        let blocked = self
            .edge_to_path
            .values()
            .flat_map(|path| path.windows(2))
            .map(|verts| self.granulated_mesh.edge_between_verts(verts[0], verts[1]).unwrap())
            .flat_map(|(a, b)| {
                vec![
                    (self.granulated_mesh.face(a), self.granulated_mesh.face(b)),
                    (self.granulated_mesh.face(b), self.granulated_mesh.face(a)),
                ]
            })
            .collect::<HashSet<_>>();

        // Get all face neighbors, but filter out neighbors blocked by the blocked edges
        let face_to_neighbors = self
            .granulated_mesh
            .face_ids()
            .into_iter()
            .map(|face_id| {
                (
                    face_id,
                    self.granulated_mesh
                        .neighbors(face_id)
                        .into_iter()
                        .filter(|&neighbor_id| !blocked.contains(&(face_id, neighbor_id)))
                        .collect_vec(),
                )
            })
            .collect::<HashMap<_, _>>();

        // Find all patches (should be equal to the number of faces in the polycube)
        let patches =
            grapff::fluid::FluidGraph::new(|face_id: FaceID| face_to_neighbors[&face_id].clone()).connected_components(&self.granulated_mesh.face_ids());

        if patches.len() != self.polycube_ref.structure.face_ids().len() {
            return Err(LayoutError::InvalidPatches);
        }

        // Every path should be part of exactly TWO patches (on both sides)
        let mut path_to_ccs: HashMap<EdgeKey<POLYCUBE>, [usize; 2]> = HashMap::new();
        for (path_id, path) in &self.edge_to_path {
            // Loop segment should simply have only two connected components (one for each side)
            // We do not check all its edges, but only the first one (since they should all be the same)
            let arbitrary_edge = self.granulated_mesh.edge_between_verts(path[0], path[1]).unwrap().0;
            // Edge has two faces
            let (face1, face2) = (
                self.granulated_mesh.face(arbitrary_edge),
                self.granulated_mesh.face(self.granulated_mesh.twin(arbitrary_edge)),
            );

            let cc1 = patches.iter().position(|cc| cc.contains(&face1)).unwrap();
            let cc2 = patches.iter().position(|cc| cc.contains(&face2)).unwrap();
            if cc1 == cc2 {
                return Err(LayoutError::InvalidPatches);
            }
            path_to_ccs.insert(*path_id, (cc1, cc2).into());
        }

        // For every patch, get the connected component that is shared among its paths
        for &face_id in &self.polycube_ref.structure.face_ids() {
            let paths = self.polycube_ref.structure.edges(face_id);

            // Select an arbitrary path
            let arbitrary_path = paths[0];
            let [cc1, cc2] = path_to_ccs[&arbitrary_path];

            // Check whether all paths share the same connected component
            let cc1_shared = paths.iter().all(|&path| path_to_ccs[&path].contains(&cc1));
            let cc2_shared = paths.iter().all(|&path| path_to_ccs[&path].contains(&cc2));
            if !(cc1_shared ^ cc2_shared) {
                return Err(LayoutError::InvalidPatches);
            }

            let faces = if cc1_shared { patches[cc1].clone() } else { patches[cc2].clone() };
            self.face_to_patch.insert(face_id, Patch { faces });
        }

        self.compute_quality();

        Ok(())
    }

    fn verify_paths(&self) -> Result<(), LayoutError> {
        for path in self.edge_to_path.values() {
            for (a, b) in path.windows(2).map(|verts| (verts[0], verts[1])) {
                // check if edge between them exists
                let edge = self.granulated_mesh.edge_between_verts(a, b);
                if edge.is_none() {
                    return Err(LayoutError::InvalidPath);
                }
                if self.granulated_mesh.size(edge.unwrap().0) == 0. {
                    return Err(LayoutError::InvalidPath);
                }
            }
        }
        Ok(())
    }

    fn compute_occupied(&self) -> (HashSet<VertID>, HashSet<(VertID, VertID)>) {
        let mut occupied_vertices = HashSet::new();
        let mut occupied_edges = HashSet::new();
        for path in self.edge_to_path.values() {
            for &v_id in path {
                occupied_vertices.insert(v_id);
            }

            for edgepair in path.windows(2) {
                let (u, v) = (edgepair[0], edgepair[1]);
                occupied_edges.insert((u, v));
                occupied_edges.insert((v, u));
            }
        }
        (occupied_vertices, occupied_edges)
    }

    pub fn laplacian_corner_shoot(&mut self, polycube_vert: VertKey<POLYCUBE>, vert_lookup: &mehsh::prelude::VertLocation<INPUT>) -> Result<(), LayoutError> {
        let mesh_vertex = self.vert_to_corner.get_by_left(&polycube_vert).unwrap().to_owned();
        let mesh_vertex_position = self.granulated_mesh.position(mesh_vertex);
        let mut x_targets = vec![mesh_vertex_position.x];
        let mut y_targets = vec![mesh_vertex_position.y];
        let mut z_targets = vec![mesh_vertex_position.z];

        for polycube_neighbor in self.polycube_ref.structure.neighbors(polycube_vert) {
            let mesh_neighbor = self.vert_to_corner.get_by_left(&polycube_neighbor).unwrap().to_owned();
            let mesh_neighbor_position = self.granulated_mesh.position(mesh_neighbor);

            let direction_of_edge = self.polycube_ref.get_direction_of_edge(polycube_vert, polycube_neighbor).0;
            match direction_of_edge {
                PrincipalDirection::X => {
                    y_targets.push(mesh_neighbor_position.y);
                    z_targets.push(mesh_neighbor_position.z);
                }
                PrincipalDirection::Y => {
                    x_targets.push(mesh_neighbor_position.x);
                    z_targets.push(mesh_neighbor_position.z);
                }
                PrincipalDirection::Z => {
                    x_targets.push(mesh_neighbor_position.x);
                    y_targets.push(mesh_neighbor_position.y);
                }
            }
        }

        let target = [
            x_targets.iter().sum::<f64>() / x_targets.len() as f64,
            y_targets.iter().sum::<f64>() / y_targets.len() as f64,
            z_targets.iter().sum::<f64>() / z_targets.len() as f64,
        ];

        self.move_corner(polycube_vert, vert_lookup.nearest(&target).1)?;

        Ok(())
    }

    pub fn move_corner(&mut self, vert: VertKey<POLYCUBE>, new_vert: VertID) -> Result<(), LayoutError> {
        let edges = self.polycube_ref.structure.edges(vert);

        // Remove adjacent paths
        for &edge in &edges {
            self.edge_to_path.remove(&edge);
            self.edge_to_path.remove(&self.polycube_ref.structure.twin(edge));
        }

        // Move the corner
        self.vert_to_corner.insert(vert, new_vert);

        // Re-compute adjacent paths
        for &edge in &edges {
            self.place_path(edge)?;
        }

        self.assign_all_patches()?;
        Ok(())
    }

    fn compute_quality(&mut self) {
        let polycube = &self.polycube_ref;

        self.alignment_per_triangle.clear();
        self.alignment = None;
        self.orthogonality_per_vert.clear();
        self.orthogonality = None;

        // Compute alignment
        let alignments = polycube
            .structure
            .face_ids()
            .iter()
            .flat_map(|&patch| {
                let target_normal = polycube.structure.normal(patch).normalize();
                let patch_faces = &self.face_to_patch.get(&patch).unwrap().faces;
                patch_faces
                    .iter()
                    .map(|&triangle_id| {
                        let alignment = self.granulated_mesh.normal(triangle_id).normalize().dot(&target_normal);
                        (triangle_id, alignment)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        // Assign the alignments
        let total_area: f64 = self.granulated_mesh.face_ids().into_iter().map(|f| self.granulated_mesh.size(f)).sum();
        self.alignment = Some(
            alignments
                .iter()
                .map(|&(triangle_id, a)| a * self.granulated_mesh.size(triangle_id) / total_area)
                .sum::<f64>(),
        );
        for (triangle_id, alignment) in alignments {
            self.alignment_per_triangle.insert(&triangle_id, alignment);
        }

        // Compute orthogonality
        let orthogonalities = polycube
            .structure
            .vert_ids()
            .iter()
            .flat_map(|&corner| {
                polycube
                    .structure
                    .faces(corner)
                    .iter()
                    .map(|&face_id| polycube.structure.edges_in_face_with_vert(face_id, corner).unwrap())
                    .collect::<Vec<_>>()
            })
            .map(|[edge1, edge2]| {
                let [u, v] = polycube.structure.vertices(edge1)[..2] else { panic!() };
                let [v2, w] = polycube.structure.vertices(edge2)[..2] else { panic!() };
                assert!(v == v2);
                let u_in_mesh = self.vert_to_corner.get_by_left(&u).unwrap().to_owned();
                let v_in_mesh = self.vert_to_corner.get_by_left(&v).unwrap().to_owned();
                let w_in_mesh = self.vert_to_corner.get_by_left(&w).unwrap().to_owned();

                let vector1 = self.granulated_mesh.position(u_in_mesh) - self.granulated_mesh.position(v_in_mesh);
                let vector2 = self.granulated_mesh.position(w_in_mesh) - self.granulated_mesh.position(v_in_mesh);
                let angle = vector1.angle(&vector2);
                let orthogonality = (90. - (angle.to_degrees() - 90.).abs()) / 90.;
                orthogonality
            })
            .collect::<Vec<_>>();

        // Assign the orthogonality
        self.orthogonality = Some(orthogonalities.iter().cloned().sum::<f64>() / orthogonalities.len() as f64);
    }
}
