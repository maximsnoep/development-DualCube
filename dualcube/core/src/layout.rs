use crate::dual::{Dual, PropertyViolationError, LOOPSTRUCTURE};
use crate::polycube::{Polycube, POLYCUBE};
use crate::prelude::*;
use bimap::BiHashMap;
use grapff::Grapff;
use itertools::Itertools;
use mehsh::prelude::*;
use ordered_float::OrderedFloat;
use rand::seq::SliceRandom;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Clone, Debug, PartialEq, Eq, Hash, Copy)]
pub enum NodeType {
    Vertex(VertID),
    Face(FaceID),
}

#[derive(Clone, Debug, Default)]
pub struct Patch {
    // A patch is defined by a set of faces
    pub faces: HashSet<FaceID>,
}

#[derive(Clone, Debug)]
pub struct Layout {
    // TODO: make this an actual (arc) reference
    pub polycube_ref: Polycube,
    // TODO: make this an actual (arc) reference
    pub dual_ref: Dual,

    // mapping:
    pub granulated_mesh: Mesh<INPUT>,
    pub vert_to_corner: BiHashMap<VertKey<POLYCUBE>, VertID>,
    pub edge_to_path: HashMap<EdgeKey<POLYCUBE>, Vec<VertID>>,
    pub face_to_patch: HashMap<FaceKey<POLYCUBE>, Patch>,
}

impl Layout {
    pub fn embed(dual_ref: &Dual, polycube_ref: &Polycube) -> Result<Self, PropertyViolationError> {
        let mut layout = Self {
            polycube_ref: polycube_ref.clone(),
            dual_ref: dual_ref.clone(),
            granulated_mesh: (*dual_ref.mesh_ref).clone(),
            vert_to_corner: BiHashMap::new(),
            face_to_patch: HashMap::new(),
            edge_to_path: HashMap::new(),
        };
        layout.place_vertices();
        layout.place_paths()?;

        layout.verify_paths();

        layout.assign_patches();
        Ok(layout)
    }

    fn place_vertices(&mut self) {
        // Clear the mapping
        self.vert_to_corner.clear();

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

    pub fn place_path(&mut self, edge_id: EdgeKey<POLYCUBE>) -> Result<(), PropertyViolationError> {
        let primal = &self.polycube_ref;
        let mut occupied_vertices = HashSet::new();
        let mut occupied_edges = HashSet::new();

        let normal_on_left = primal.structure.normal(primal.structure.face(edge_id));
        let normal_on_right = primal.structure.normal(primal.structure.face(primal.structure.twin(edge_id)));

        let primal_vertices = primal
            .structure
            .vert_ids()
            .iter()
            .map(|&x| self.vert_to_corner.get_by_left(&x).unwrap().to_owned())
            .collect_vec();

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

        let endpoints = primal.structure.vertices(edge_id);

        let (u, v) = (
            self.vert_to_corner.get_by_left(&endpoints[0]).unwrap().to_owned(),
            self.vert_to_corner.get_by_left(&endpoints[1]).unwrap().to_owned(),
        );

        // shortest path from u to v
        let n_function = |node: NodeType| match node {
            NodeType::Face(f_id) => {
                let f_neighbors: Vec<NodeType> = {
                    // Only allowed if the edge between the two faces is not occupied.
                    let blocked = |f1: FaceID, f2: FaceID| {
                        let (edge_id, _) = self.granulated_mesh.edge_between_faces(f1, f2).unwrap();
                        let endpoints = self.granulated_mesh.vertices(edge_id);
                        occupied_edges.contains(&(endpoints[0], endpoints[1]))
                    };
                    self.granulated_mesh
                        .neighbors(f_id)
                        .into_iter()
                        .filter(|&n_id| !blocked(f_id, n_id))
                        .map(NodeType::Face)
                        .collect_vec()
                };
                let v_neighbors = self.granulated_mesh.vertices(f_id).into_iter().map(NodeType::Vertex).collect_vec();
                [v_neighbors, f_neighbors].concat()
            }
            NodeType::Vertex(v_id) => {
                // Only allowed if the vertex is not occupied
                if (occupied_vertices.contains(&v_id) || primal_vertices.contains(&v_id)) && v_id != u && v_id != v {
                    return vec![];
                }
                let v_neighbors = self.granulated_mesh.neighbors(v_id).into_iter().map(NodeType::Vertex).collect_vec();
                let f_neighbors = self.granulated_mesh.faces(v_id).into_iter().map(NodeType::Face).collect_vec();
                [v_neighbors, f_neighbors].concat()
            }
        };
        // neighbors of u using n_function
        let nodetype_to_pos = |node: NodeType| match node {
            NodeType::Face(f_id) => self.granulated_mesh.position(f_id),
            NodeType::Vertex(v_id) => self.granulated_mesh.position(v_id),
        };

        let ridge_function = |a: NodeType, b: NodeType| {
            let (normal1, normal2) = match (a, b) {
                (NodeType::Vertex(a), NodeType::Vertex(b)) => {
                    let (edge1, edge2) = self.granulated_mesh.edge_between_verts(a, b).unwrap();
                    let normal1 = self.granulated_mesh.normal(self.granulated_mesh.face(edge1));
                    let normal2 = self.granulated_mesh.normal(self.granulated_mesh.face(edge2));
                    (normal1, normal2)
                }
                (NodeType::Face(f), NodeType::Face(_)) => {
                    let normal = self.granulated_mesh.normal(f);
                    (normal, normal)
                }
                (NodeType::Face(f), NodeType::Vertex(_)) => {
                    let normal = self.granulated_mesh.normal(f);
                    (normal, normal)
                }
                (NodeType::Vertex(_), NodeType::Face(f)) => {
                    let normal = self.granulated_mesh.normal(f);
                    (normal, normal)
                }
            };

            let angle1 = normal1.angle(&normal_on_left);
            let angle2 = normal2.angle(&normal_on_right);

            if angle1 + angle2 < std::f64::consts::PI / 4.0 {
                // If the angles are small, we can use a simpler function
                0.5
            } else {
                1.5
            }
        };

        let w_function = |(a, b)| OrderedFloat(ridge_function(a, b) * nodetype_to_pos(a).metric_distance(&nodetype_to_pos(b)));

        let mut granulated_path = vec![];

        let pp = {
            let nn = grapff::fluid::FluidGraph::new(n_function);
            nn.shortest_path(NodeType::Vertex(u), NodeType::Vertex(v), w_function)
        };

        if let Some((path, _)) = pp {
            let mut last_f_ids_maybe: Option<[FaceID; 3]> = None;
            for node in path {
                match node {
                    NodeType::Vertex(v_id) => {
                        granulated_path.push((v_id, false));
                        last_f_ids_maybe = None;
                    }
                    NodeType::Face(f_id) => {
                        let new_v_pos = self.granulated_mesh.position(f_id);
                        let (new_v_id, new_f_ids) = self.granulated_mesh.split_face(f_id);
                        self.granulated_mesh.set_position(new_v_id, new_v_pos);

                        if let Some(last_f_ids) = last_f_ids_maybe {
                            for last_f_id in last_f_ids {
                                for new_f_id in new_f_ids {
                                    if let Some((edge_id, _)) = self.granulated_mesh.edge_between_faces(last_f_id, new_f_id) {
                                        let endpoints = self.granulated_mesh.vertices(edge_id);
                                        let u = endpoints[0];
                                        let v = endpoints[1];
                                        let c1 = self.granulated_mesh.vertices(last_f_id).into_iter().find(|&c| c != u && c != v).unwrap();
                                        let c2 = self
                                            .granulated_mesh
                                            .vertices(new_f_id)
                                            .into_iter()
                                            .find(|&c| c != u && c != v && c != c1)
                                            .unwrap();

                                        let mut smallest_distance = f64::MAX;
                                        let mut smallest_pos = Vector3D::new(0., 0., 0.);
                                        let samples = 10;
                                        for s in 1..samples {
                                            let dir_vec = self.granulated_mesh.vector(edge_id);
                                            let pos = self.granulated_mesh.position(u) + dir_vec * (f64::from(s) / f64::from(samples));
                                            let distance_c1 = pos.metric_distance(&self.granulated_mesh.position(c1));
                                            let distance_c2 = pos.metric_distance(&self.granulated_mesh.position(c2));
                                            let distance = distance_c1 + distance_c2;
                                            if distance < smallest_distance {
                                                smallest_distance = distance;
                                                smallest_pos = pos;
                                            }
                                        }
                                        let (mid_v_id, _) = self.granulated_mesh.split_edge(edge_id);
                                        self.granulated_mesh.set_position(mid_v_id, smallest_pos);
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
        };

        let granulated_path = granulated_path.into_iter().map(|(v_id, _)| v_id).collect_vec();

        if granulated_path.is_empty() {
            return Err(PropertyViolationError::PathEmpty);
        }

        for &v_id in &granulated_path {
            occupied_vertices.insert(v_id);
        }

        for edgepair in granulated_path.windows(2) {
            let (u, v) = (edgepair[0], edgepair[1]);
            occupied_edges.insert((u, v));
            occupied_edges.insert((v, u));
        }
        self.edge_to_path.insert(edge_id, granulated_path.clone());

        // for the twin, we insert the reverse
        let mut rev_path = granulated_path;
        rev_path.reverse();
        self.edge_to_path.insert(primal.structure.twin(edge_id), rev_path);

        Ok(())
    }

    // TODO: Make this robust
    pub fn place_paths(&mut self) -> Result<(), PropertyViolationError> {
        let primal = &self.polycube_ref;

        self.edge_to_path.clear();

        for vert in primal.structure.vert_ids() {
            println!("vert: {vert:?}");
            assert!(self.vert_to_corner.get_by_left(&vert).is_some());
        }

        let primal_vertices = primal
            .structure
            .vert_ids()
            .iter()
            .map(|&x| self.vert_to_corner.get_by_left(&x).unwrap().to_owned())
            .collect_vec();

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

        let mut edge_queue = primal.structure.edge_ids();

        edge_queue.shuffle(&mut rand::rng());

        let mut edge_queue = VecDeque::from(edge_queue);

        let mut first_separating_edge = None;
        let mut is_maximal = false;

        let mut counter = 0;

        while let Some(edge_id) = edge_queue.pop_front() {
            //println!("Edge queue: {}", edge_queue.len());

            let normal_on_left = primal.structure.normal(primal.structure.face(edge_id));
            let normal_on_right = primal.structure.normal(primal.structure.face(primal.structure.twin(edge_id)));

            // if already found (because of twin), skip
            if self.edge_to_path.contains_key(&edge_id) {
                continue;
            }

            if counter > 1000 {
                return Err(PropertyViolationError::UnknownError);
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

            let mut blocked_faces = HashSet::new();
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
                        blocked_faces.insert(face_id);
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
                        blocked_faces.insert(face_id);
                    }
                }
            }

            let mut blocked_vertices = HashSet::new();
            for &blocked_face in &blocked_faces {
                blocked_vertices.extend(self.granulated_mesh.vertices(blocked_face));
            }

            // shortest path from u to v
            let n_function = |node: NodeType| match node {
                NodeType::Face(f_id) => {
                    let f_neighbors: Vec<NodeType> = {
                        // Disallow blocked faces
                        if blocked_faces.contains(&f_id) {
                            return vec![];
                        }
                        // Only allowed if the edge between the two faces is not occupied.
                        let blocked = |f1: FaceID, f2: FaceID| {
                            let (edge_id, _) = self.granulated_mesh.edge_between_faces(f1, f2).unwrap();
                            let endpoints = self.granulated_mesh.vertices(edge_id);
                            occupied_edges.contains(&(endpoints[0], endpoints[1]))
                        };
                        self.granulated_mesh
                            .neighbors(f_id)
                            .into_iter()
                            .filter(|&n_id| !blocked(f_id, n_id))
                            .map(NodeType::Face)
                            .collect_vec()
                    };
                    let v_neighbors = self.granulated_mesh.vertices(f_id).into_iter().map(NodeType::Vertex).collect_vec();
                    [v_neighbors, f_neighbors].concat()
                }
                NodeType::Vertex(v_id) => {
                    // Disallow vertices of blocked faces (unless it is the start or end vertex)
                    if blocked_vertices.contains(&v_id) && v_id != u && v_id != v {
                        return vec![];
                    }
                    // Only allowed if the vertex is not occupied
                    if (occupied_vertices.contains(&v_id) || primal_vertices.contains(&v_id)) && v_id != u && v_id != v {
                        return vec![];
                    }
                    let v_neighbors = self.granulated_mesh.neighbors(v_id).into_iter().map(NodeType::Vertex).collect_vec();
                    let f_neighbors = self.granulated_mesh.faces(v_id).into_iter().map(NodeType::Face).collect_vec();
                    [v_neighbors, f_neighbors].concat()
                }
            };
            // neighbors of u using n_function
            let nodetype_to_pos = |node: NodeType| match node {
                NodeType::Face(f_id) => self.granulated_mesh.position(f_id),
                NodeType::Vertex(v_id) => self.granulated_mesh.position(v_id),
            };

            let ridge_function = |a: NodeType, b: NodeType| {
                let (normal1, normal2) = match (a, b) {
                    (NodeType::Vertex(a), NodeType::Vertex(b)) => {
                        let (edge1, edge2) = self.granulated_mesh.edge_between_verts(a, b).unwrap();
                        let normal1 = self.granulated_mesh.normal(self.granulated_mesh.face(edge1));
                        let normal2 = self.granulated_mesh.normal(self.granulated_mesh.face(edge2));
                        (normal1, normal2)
                    }
                    (NodeType::Face(f), NodeType::Face(_)) => {
                        let normal = self.granulated_mesh.normal(f);
                        (normal, normal)
                    }
                    (NodeType::Face(f), NodeType::Vertex(_)) => {
                        let normal = self.granulated_mesh.normal(f);
                        (normal, normal)
                    }
                    (NodeType::Vertex(_), NodeType::Face(f)) => {
                        let normal = self.granulated_mesh.normal(f);
                        (normal, normal)
                    }
                };

                let angle1 = normal1.angle(&normal_on_left);
                let angle2 = normal2.angle(&normal_on_right);

                if angle1 + angle2 < std::f64::consts::PI / 4.0 {
                    // If the angles are small, we can use a simpler function
                    0.5
                } else {
                    1.5
                }
            };

            let w_function = |(a, b)| OrderedFloat(ridge_function(a, b) * nodetype_to_pos(a).metric_distance(&nodetype_to_pos(b)));

            let mut granulated_path = vec![];

            let pp = {
                let nn = grapff::fluid::FluidGraph::new(n_function);
                nn.shortest_path(NodeType::Vertex(u), NodeType::Vertex(v), w_function)
            };

            if let Some((path, _)) = pp {
                let mut last_f_ids_maybe: Option<[FaceID; 3]> = None;
                for node in path {
                    match node {
                        NodeType::Vertex(v_id) => {
                            granulated_path.push((v_id, false));
                            last_f_ids_maybe = None;
                        }
                        NodeType::Face(f_id) => {
                            let new_v_pos = self.granulated_mesh.position(f_id);
                            let (new_v_id, new_f_ids) = self.granulated_mesh.split_face(f_id);
                            self.granulated_mesh.set_position(new_v_id, new_v_pos);

                            if let Some(last_f_ids) = last_f_ids_maybe {
                                for last_f_id in last_f_ids {
                                    for new_f_id in new_f_ids {
                                        if let Some((edge_id, _)) = self.granulated_mesh.edge_between_faces(last_f_id, new_f_id) {
                                            let endpoints = self.granulated_mesh.vertices(edge_id);
                                            let u = endpoints[0];
                                            let v = endpoints[1];
                                            let c1 = self.granulated_mesh.vertices(last_f_id).into_iter().find(|&c| c != u && c != v).unwrap();
                                            let c2 = self
                                                .granulated_mesh
                                                .vertices(new_f_id)
                                                .into_iter()
                                                .find(|&c| c != u && c != v && c != c1)
                                                .unwrap();

                                            let mut smallest_distance = f64::MAX;
                                            let mut smallest_pos = Vector3D::new(0., 0., 0.);
                                            let samples = 10;
                                            for s in 1..samples {
                                                let dir_vec = self.granulated_mesh.vector(edge_id);
                                                let pos = self.granulated_mesh.position(u) + dir_vec * (f64::from(s) / f64::from(samples));
                                                let distance_c1 = pos.metric_distance(&self.granulated_mesh.position(c1));
                                                let distance_c2 = pos.metric_distance(&self.granulated_mesh.position(c2));
                                                let distance = distance_c1 + distance_c2;
                                                if distance < smallest_distance {
                                                    smallest_distance = distance;
                                                    smallest_pos = pos;
                                                }
                                            }
                                            let (mid_v_id, _) = self.granulated_mesh.split_edge(edge_id);
                                            self.granulated_mesh.set_position(mid_v_id, smallest_pos);
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
            };

            let granulated_path = granulated_path.into_iter().map(|(v_id, _)| v_id).collect_vec();

            if granulated_path.is_empty() {
                return Err(PropertyViolationError::PathEmpty);
            }

            for &v_id in &granulated_path {
                occupied_vertices.insert(v_id);
            }

            for edgepair in granulated_path.windows(2) {
                let (u, v) = (edgepair[0], edgepair[1]);
                occupied_edges.insert((u, v));
                occupied_edges.insert((v, u));
            }
            self.edge_to_path.insert(edge_id, granulated_path.clone());

            // for the twin, we insert the reverse
            let mut rev_path = granulated_path;
            rev_path.reverse();
            self.edge_to_path.insert(primal.structure.twin(edge_id), rev_path);

            counter = 0;
        }
        Ok(())
    }

    pub fn optimize_corner(&mut self, polycube_vert: VertKey<POLYCUBE>) {
        // INVARIANT: ALL CORNERS AND PATHS MUST ALREADY BE PLACED

        // LaPlacian optimization of specific corner:
        // 1. place corner in the average position of its neighbors (and constrained by region formed by existing paths)
        // 2. find new paths (constrained by existing paths)

        let polycube_neighbors = self.polycube_ref.structure.neighbors(polycube_vert);

        println!("Optimizing corner for polycube vertex: {polycube_vert:?}, neighbors: {polycube_neighbors:?}");

        // grab all vertices in the corresponding patches (they are candidates)
        let polycube_faces = self.polycube_ref.structure.faces(polycube_vert);
        let patches = polycube_faces.iter().map(|&face| self.face_to_patch.get(&face).unwrap()).collect_vec();
        let candidate_vertices = patches
            .into_iter()
            .flat_map(|patch| patch.faces.clone())
            .flat_map(|f| self.granulated_mesh.vertices(f))
            .filter(|&v| !self.vert_to_corner.contains_right(&v)) // filter out already occupied vertices
            .filter(|&v| !self.edge_to_path.values().any(|path| path.contains(&v))) // filter out vertices that are part of a path
            .collect_vec();

        // Map polycube to mesh
        let mesh_vert = self.vert_to_corner.get_by_left(&polycube_vert).unwrap().to_owned();
        let mesh_neighbors = polycube_neighbors
            .into_iter()
            .filter_map(|neighbor| self.vert_to_corner.get_by_left(&neighbor))
            .collect_vec();

        let sum = mesh_neighbors.iter().map(|&&v| self.granulated_mesh.position(v)).sum::<Vector3D>();
        let avg = sum / mesh_neighbors.len() as f64;

        // Find vertex on mesh with smallest distance to avg
        // let vert_lookup = self.granulated_mesh.kdtree();
        // let nearest_vert = vert_lookup.nearest(&[avg.x, avg.y, avg.z]).1;
        let nearest_vert = candidate_vertices
            .iter()
            .min_by_key(|&&v| {
                let pos = self.granulated_mesh.position(v);
                OrderedFloat(pos.metric_distance(&avg))
            })
            .unwrap()
            .to_owned();

        println!("Optimizing corner for polycube vertex: {polycube_vert:?}, nearest mesh vertex: {nearest_vert:?}");

        if self.vert_to_corner.contains_right(&nearest_vert) {
            println!("Vertex already occupied, skipping optimization");
            return;
        }

        self.vert_to_corner.insert(polycube_vert, nearest_vert);
    }

    pub fn assign_patches(&mut self) {
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

        assert!(patches.len() == self.polycube_ref.structure.face_ids().len());

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
            assert_ne!(cc1, cc2);
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
            assert!(cc1_shared ^ cc2_shared);

            let faces = if cc1_shared { patches[cc1].clone() } else { patches[cc2].clone() };
            self.face_to_patch.insert(face_id, Patch { faces });
        }
    }

    pub fn verify_paths(&self) {
        for path in self.edge_to_path.values() {
            for (a, b) in path.windows(2).map(|verts| (verts[0], verts[1])) {
                // check if edge between them exists
                let edge = self.granulated_mesh.edge_between_verts(a, b);
                assert!(edge.is_some());
                assert!(self.granulated_mesh.size(edge.unwrap().0) > 0.);
            }
        }
    }
}
