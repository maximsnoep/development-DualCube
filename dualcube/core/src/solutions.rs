use crate::layout::LayoutError;
use crate::polycube::POLYCUBE;
use crate::prelude::*;
use crate::skeleton::{
    SkeletonData,
    generate_loops,
    get_skeleton_based_mapping,
};
use crate::{
    dual::{Dual, PropertyViolationError},
    layout::Layout,
    polycube::Polycube,
    quad::Quad,
};
use grapff::Grapff;
use itertools::Itertools;
use mehsh::prelude::*;
use ordered_float::OrderedFloat;
use orx_parallel::*;
use rand::{rng, seq::IteratorRandom};
use serde::{Deserialize, Serialize};
use slotmap::SlotMap;
use std::{collections::HashSet, hash::Hash, sync::Arc};
use thiserror::Error;

slotmap::new_key_type! {
    pub struct LoopID;
}

#[derive(Default, Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct NodeCopy {
    pub id: [EdgeID; 2],
    pub t: usize,
}

// A loop forms the basis of the dual structure.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Loop {
    // A loop is defined by a sequence of half-edges.
    pub edges: Vec<EdgeID>,
    // the direction or labeling associated with the loop
    pub direction: PrincipalDirection,
}

#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum SolutionError {
    #[error("Something is wrong with the DUAL representation: {0}")]
    DualError(#[from] PropertyViolationError),
    #[error("Something is wrong with the PRIMAL representation: {0}")]
    PrimalError(#[from] LayoutError),
    #[error("The DUAL representation is not initialized and can therefore not be modified.")]
    NoDual,
    #[error("The PRIMAL representation is not initialized and can therefore not be modified.")]
    NoPrimal,
    #[error("The POLYCUBE representation is not initialized and can therefore not be modified.")]
    NoPolycube,
}

#[must_use]
pub fn wrap_pairs<T: Copy>(sequence: &[T]) -> Vec<(T, T)> {
    sequence
        .iter()
        .cycle()
        .copied()
        .take(sequence.len() + 1)
        .tuple_windows()
        .collect()
}

impl Loop {
    #[must_use]
    pub fn contains_pair(&self, needle: (EdgeID, EdgeID)) -> bool {
        wrap_pairs(&self.edges)
            .into_iter()
            .any(|(a, b)| a == needle.0 && b == needle.1)
    }

    #[must_use]
    fn find_edge(&self, needle: EdgeID) -> usize {
        self.edges.iter().position(|&e| e == needle).unwrap()
    }

    #[must_use]
    pub fn between(&self, start: EdgeID, end: EdgeID) -> Vec<EdgeID> {
        let start_pos = self.find_edge(start);
        let end_pos = self.find_edge(end);

        let mut seq = vec![];
        if start_pos < end_pos {
            // if start_pos < end_pos, we return [start...end]
            seq.extend(self.edges[start_pos..=end_pos].iter());
        } else {
            // if start_pos > end_pos, we return [start...MAX] + [0...end]
            seq.extend(self.edges[start_pos..].iter());
            seq.extend(self.edges[..=end_pos].iter());
        }
        seq
    }

    #[must_use]
    pub fn occupied(loops: &SlotMap<LoopID, Self>) -> ids::SecMap<EDGE, INPUT, Vec<LoopID>> {
        let mut occupied = ids::SecMap::new();
        for loop_id in loops.keys() {
            for &edge in &loops[loop_id].edges {
                if !occupied.contains_key(&edge) {
                    occupied.insert(&edge, vec![]);
                }
                occupied.get_mut(&edge).unwrap().push(loop_id);
            }
        }
        occupied
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Solution {
    pub mesh_ref: Arc<Mesh<INPUT>>,
    pub loops: SlotMap<LoopID, Loop>,
    occupied: ids::SecMap<EDGE, INPUT, Vec<LoopID>>,
    pub last_loop: Option<LoopID>,

    pub skeleton: Option<SkeletonData>,

    pub dual: Result<Dual, PropertyViolationError>,
    pub polycube: Option<Polycube>,
    pub layout: Option<Layout>,
    pub quad: Option<Quad>,

    pub fields: Option<crate::field::Fields<INPUT>>,

    pub external_flag: Option<ids::SecMap<FACE, INPUT, usize>>,
}

impl Solution {
    pub fn clear(&mut self) {
        self.dual = Err(PropertyViolationError::default());
        self.polycube = None;
        self.layout = None;
        // Note: skeleton is NOT cleared here as it's independent of the loop-based solution
    }

    // ***
    // STEPS OF A SOLUTION
    // ***

    // Create new solution
    pub fn new(mesh_ref: Arc<Mesh<INPUT>>) -> Self {
        Self {
            mesh_ref,
            loops: SlotMap::with_key(),
            occupied: ids::SecMap::new(),
            skeleton: None,
            dual: Err(PropertyViolationError::default()),
            polycube: None,
            layout: None,
            quad: None,
            external_flag: None,
            last_loop: None,
            fields: None,
        }
    }

    // Calculate skeleton
    pub fn calculate_skeleton(
        &mut self,
        convexity_threshold: f64,
        convexity_merge_threshold: f64,
    ) {
        self.skeleton = Some(get_skeleton_based_mapping(
            self.mesh_ref.clone(),
            convexity_threshold,
            convexity_merge_threshold,
        ));
        generate_loops();
    }

    // Initialize loop structure
    pub fn initialize(&mut self, flow_graphs: &[grapff::fixed::FixedGraph<EdgeID, f64>; 3]) {
        // Basic initialization: sample some loops in each direction
        let m = |b: f64| OrderedFloat(b.powi(10));
        let s = |(p, _): (&[EdgeID], f64)| -(p.len() as f64);

        let samples = 3;
        let x_loops = self.sample_loops(samples, PrincipalDirection::X, flow_graphs, m, s);
        let y_loops = self.sample_loops(samples, PrincipalDirection::Y, flow_graphs, m, s);
        let z_loops = self.sample_loops(samples, PrincipalDirection::Z, flow_graphs, m, s);

        // Compute all n^3 combinations
        let combinations = x_loops
            .into_iter()
            .cartesian_product(y_loops)
            .cartesian_product(z_loops)
            .map(|((x, y), z)| (x, y, z))
            .collect_vec();

        let candidate_solutions = combinations
            .into_par()
            .filter_map(|(x_loop, y_loop, z_loop)| {
                let mut solution = self.clone();
                solution.add_loop(Loop {
                    edges: x_loop,
                    direction: PrincipalDirection::X,
                });
                solution.add_loop(Loop {
                    edges: y_loop,
                    direction: PrincipalDirection::Y,
                });
                solution.add_loop(Loop {
                    edges: z_loop,
                    direction: PrincipalDirection::Z,
                });
                if solution.reconstruct_solution(false, 1).is_err() {
                    None
                } else {
                    Some(solution)
                }
            })
            .collect::<Vec<_>>();

        // Get the best solution based on quality
        if let Some(best_solution) = candidate_solutions
            .into_iter()
            .max_by_key(|solution| OrderedFloat(solution.get_quality().unwrap()))
        {
            *self = best_solution;
        }
    }

    // Evolve loop structure
    pub fn evolve(
        &self,
        iterations: usize,
        pool1_size: usize,
        pool2_size: usize,
        flowgraphs: &[grapff::fixed::FixedGraph<EdgeID, f64>; 3],
    ) -> Option<Self> {
        let flowgraphs_clone = flowgraphs.clone();
        let mut pool1 = vec![(self.clone(), self.get_quality().unwrap()); pool1_size];
        for _ in 0..iterations {
            let pool2 = (0..pool2_size)
                .into_par()
                .map(|_| {
                    // Grab a random solution from pool1
                    let index = rand::Rng::random_range(&mut rand::rng(), 0..pool1.len());
                    pool1[index].clone()
                })
                .filter_map(|(sol, _)| {
                    // Mutate the solution
                    sol.mutation(&flowgraphs_clone).map_or_else(
                        || None,
                        |mutation| {
                            let quality = mutation.get_quality().unwrap_or(0.0);
                            // Compute quality
                            Some((mutation, quality))
                        },
                    )
                })
                .collect::<Vec<_>>()
                .into_iter()
                .sorted_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .rev();

            for (_, quality) in pool2.clone() {
                println!("Generated solution with quality: {quality}");
            }

            // overwrite pool1 with top 5 solutions of pool1 and top 5 solutions of pool2

            pool1 = pool1
                .into_iter()
                .take(5)
                .chain(pool2.take(5))
                .sorted_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .rev()
                .collect();

            for (_, quality) in &pool1 {
                println!("PICKED solution with quality: {quality}");
            }
        }

        if pool1.is_empty() {
            println!("No valid solutions generated.");
            return None;
        }

        // Log all qualities of generated solutions
        for (_, quality) in &pool1 {
            println!("Generated solution with quality: {quality}");
        }

        // Grab the best solution
        let (sol, quality) = pool1
            .into_iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        println!("Picked solution with quality: {quality}");
        Some(sol)
    }

    // Construct dual and polycube
    pub fn construct_dual_and_polycube(&mut self) -> Result<(), PropertyViolationError> {
        self.layout = None;
        self.polycube = None;

        self.dual = Dual::from(self.mesh_ref.clone(), &self.loops);
        if let Err(e) = &self.dual {
            return Err(e.clone());
        }

        self.polycube = Some(Polycube::from_dual(self.dual.as_ref().unwrap()));

        // check all faces of the polycube have a normal
        for face in self.polycube.as_ref().unwrap().structure.face_ids() {
            let normal = self.polycube.as_ref().unwrap().structure.normal(face);
            if normal.x.is_nan() || normal.y.is_nan() || normal.z.is_nan() {
                println!("cube: {normal}");
                return Err(PropertyViolationError::UnknownError);
            }
        }

        Ok(())
    }

    // Place all corners
    pub fn place_corners(&mut self) -> Result<(), SolutionError> {
        if self.dual.is_err() {
            return Err(SolutionError::NoDual);
        }
        if self.polycube.is_none() {
            return Err(SolutionError::NoPolycube);
        }

        let mut layout = Layout::new(self.dual.as_ref().unwrap(), self.polycube.as_ref().unwrap());
        layout.place_all_corners();
        self.layout = Some(layout);
        Ok(())
    }

    // Move a corner
    pub fn move_corner_to(
        &mut self,
        corner: VertKey<POLYCUBE>,
        new_vertex: VertID,
    ) -> Result<(), SolutionError> {
        if self.layout.is_none() {
            return Err(SolutionError::NoPrimal);
        }

        let mut layout = self.layout.clone().unwrap();
        layout.move_corner(corner, new_vertex)?;
        self.layout = Some(layout);

        Ok(())
    }

    // Place all paths
    pub fn place_paths(&mut self) -> Result<(), SolutionError> {
        if self.layout.is_none() {
            return Err(SolutionError::NoPrimal);
        }
        let layout = self.layout.as_mut().unwrap();
        layout.place_all_paths()?;
        layout.assign_all_patches()?;
        Ok(())
    }

    // Optimize corners
    pub fn optimize_corners(&mut self) -> Result<(), SolutionError> {
        if self.dual.is_err() {
            return Err(SolutionError::NoDual);
        }
        let dual = self.layout.as_ref().unwrap().dual_ref.clone();
        let polycube = self.layout.as_ref().unwrap().polycube_ref.clone();

        let polycube_vertices = dual
            .loop_structure
            .face_ids()
            .iter()
            .map(|f| polycube.region_to_vertex.get_by_left(f).unwrap().to_owned())
            // .filter(|&v| layout.polycube_ref.structure.edges(v).len() == 4)
            .collect_vec();

        let mut solution_backup = self.clone();

        let mut solution_clone = self.clone();
        let mut current_quality = solution_clone.get_quality().unwrap();

        // Create priority queue with "quality" around a vertex as key;
        // let mut priority_queue = std::collections::BinaryHeap::new();
        // for &polycube_vertex in &polycube_vertices {
        //     let mesh_vertex = solution_clone
        //         .layout
        //         .as_mut()
        //         .unwrap()
        //         .vert_to_corner
        //         .get_by_left(&polycube_vertex)
        //         .unwrap()
        //         .to_owned();
        //     let faces_of_vert = solution_clone.layout.as_mut().unwrap().granulated_mesh.faces(mesh_vertex);

        //     let worst_quality = faces_of_vert
        //         .iter()
        //         .map(|&f| OrderedFloat(solution_clone.alignment_per_triangle.get(&f).unwrap().to_owned()))
        //         .min()
        //         .unwrap()
        //         .0;

        //     priority_queue.push((polycube_vertex, OrderedFloat(100. - worst_quality)));
        // }

        let vert_lookup = self.layout.as_ref().unwrap().granulated_mesh.kdtree();

        // for _ in 0..5 {
        //     for &polycube_vertex in &polycube_vertices {
        //         println!(
        //             "orthogonality of v {:?} is {:?}",
        //             polycube_vertex,
        //             solution_clone.orthogonality_per_vert.get(&polycube_vertex)
        //         );

        //         if solution_clone.orthogonality_per_vert.get(&polycube_vertex).is_some()
        //             && *solution_clone.orthogonality_per_vert.get(&polycube_vertex).unwrap() > 0.7
        //         {
        //             continue;
        //         }

        //         if solution_clone
        //             .layout
        //             .as_mut()
        //             .unwrap()
        //             .laplacian_corner_shoot(polycube_vertex, &vert_lookup)
        //             .is_err()
        //         {
        //             solution_clone = solution_backup.clone();
        //             continue;
        //         }

        //         solution_clone.compute_quality();
        //         let quality = solution_clone.get_quality().unwrap();
        //         if quality >= current_quality {
        //             solution_backup = solution_clone.clone();
        //             current_quality = quality;
        //             println!("New quality: {:?}", current_quality);
        //         } else {
        //             solution_clone = solution_backup.clone();
        //         }
        //     }
        // }

        for _ in 0..1 {
            for &polycube_vertex in &polycube_vertices {
                // if self.polycube.as_ref().unwrap().structure.edges(vert).len() == 4 {
                //     continue;
                // }
                // check the "quality" around this vert, if its already good, dont improve it
                let layout = solution_clone.layout.as_mut().unwrap();
                // let mesh_vertex = layout.vert_to_corner.get_by_left(&polycube_vertex).unwrap().to_owned();
                // let faces_of_vert = layout.granulated_mesh.faces(mesh_vertex);

                // let worst_quality = faces_of_vert
                //     .iter()
                //     .map(|&f| OrderedFloat(layout.alignment_per_triangle.get(&f).unwrap().to_owned()))
                //     .min()
                //     .unwrap();

                // if worst_quality > OrderedFloat(0.5) {
                //     continue;
                // }

                if layout
                    .laplacian_corner_shoot(polycube_vertex, &vert_lookup)
                    .is_err()
                {
                    solution_clone = solution_backup.clone();
                    continue;
                }

                // let new_worst_quality = faces_of_vert
                //     .iter()
                //     .map(|&f| OrderedFloat(layout.alignment_per_triangle.get(&f).unwrap().to_owned()))
                //     .min()
                //     .unwrap();

                // if new_worst_quality < worst_quality {
                //     solution_clone = solution_backup.clone();
                //     continue;
                // }

                // solution_clone.get_quality();
                let quality = solution_clone.get_quality().unwrap();
                // if quality >= current_quality {
                solution_backup = solution_clone.clone();
                current_quality = quality;
                println!("New quality: {:?}", current_quality);
                // } else {
                //     solution_clone = solution_backup.clone();
                // }
            }
        }

        // for _ in 0..2 {
        //     for &vert in &verts {
        //         if solution_clone.layout.as_mut().unwrap().laplacian_corner(vert).is_err() {
        //             solution_clone = solution_backup.clone();
        //             continue;
        //         }
        //         solution_clone.compute_quality();
        //         let quality = solution_clone.get_quality().unwrap();
        //         if quality >= current_quality {
        //             solution_backup = solution_clone.clone();
        //             current_quality = quality;
        //             println!("New quality: {:?}", current_quality);
        //         } else {
        //             solution_clone = solution_backup.clone();
        //         }
        //     }
        // }

        *self = solution_clone;

        Ok(())
    }

    // Construct quad
    pub fn construct_quad(&mut self, omega: usize) -> Result<(), PropertyViolationError> {
        self.quad = Quad::from_layout(self.layout.as_ref().unwrap(), omega);
        Ok(())
    }

    /// REST OF THE FUNCS
    ///
    ///
    ///
    ///
    ///
    ///
    ///
    ///
    ///
    ///
    ///
    ///
    ///

    pub fn recompute_occupied(&mut self) {
        self.occupied.clear();
        for (loop_id, loop_) in &self.loops {
            for &e in &loop_.edges {
                if !self.occupied.contains_key(&e) {
                    self.occupied.insert(&e, vec![]);
                }
                self.occupied.get_mut(&e).unwrap().push(loop_id);
            }
        }
    }

    pub fn del_loop(&mut self, loop_id: LoopID) {
        for &e in &self.loops[loop_id].edges.clone() {
            if let Some(v) = self.occupied.get_mut(&e) {
                v.retain(|&l| l != loop_id);
                if v.is_empty() {
                    self.occupied.remove(&e);
                }
            }
        }

        self.loops.remove(loop_id);
    }

    pub fn add_loop(&mut self, l: Loop) -> LoopID {
        let loop_id = self.loops.insert(l);

        for e in self.loops[loop_id].edges.clone() {
            if !self.occupied.contains_key(&e) {
                self.occupied.insert(&e, vec![]);
            }
            self.occupied.get_mut(&e).unwrap().push(loop_id);
        }

        self.last_loop = Some(loop_id);

        loop_id
    }

    pub fn get_coordinates_of_loop_in_edge(&self, l: LoopID, e: EdgeID) -> Vector3D {
        let loops_in_edge = self.loops_on_edge(e);
        // sort based on the global order of the loops
        // ... todo
        // get the index of the edge in the loop
        let edge_index = self.loops[l].edges.iter().position(|&e2| e2 == e).unwrap();
        let incoming_or_outgoing = edge_index % 2 == 0;
        // find the index of the loop in the sorted list
        let i = {
            if incoming_or_outgoing {
                loops_in_edge.iter().position(|&l2| l2 == l).unwrap() as f64
            } else {
                loops_in_edge.iter().rev().position(|&l2| l2 == l).unwrap() as f64
            }
        };
        // compute the coordinates, based on the index
        let n = loops_in_edge.len() as f64;
        let offset = (i + 1.) / (n + 1.);
        // define the loop segment, starting point is p, ending point is q

        let Some([v1, v2]) = self.mesh_ref.vertices(e).collect_array::<2>() else {
            panic!()
        };
        let p = self.mesh_ref.position(v1);
        let q = self.mesh_ref.position(v2);
        // compute the coordinates
        p + offset * (q - p)
    }

    pub fn get_loops_in_direction(&self, direction: PrincipalDirection) -> Vec<LoopID> {
        self.loops
            .iter()
            .filter_map(|(id, l)| {
                if l.direction == direction {
                    Some(id)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn count_loops_in_direction(&self, direction: PrincipalDirection) -> usize {
        self.get_loops_in_direction(direction).len()
    }

    pub fn loop_to_direction(&self, loop_id: LoopID) -> PrincipalDirection {
        self.loops[loop_id].direction
    }

    pub fn get_pairs_of_loop(&self, loop_id: LoopID) -> Vec<[EdgeID; 2]> {
        self.get_pairs_of_sequence(&self.loops[loop_id].edges)
    }

    pub fn get_pairs_of_sequence(&self, sequence: &[EdgeID]) -> Vec<[EdgeID; 2]> {
        sequence
            .windows(2)
            .filter_map(|w| {
                if self.mesh_ref.twin(w[0]) == w[1] {
                    None
                } else {
                    Some([w[0], w[1]])
                }
            })
            .collect()
    }

    pub fn cycled_windows(sequence: &[EdgeID]) -> Vec<[EdgeID; 2]> {
        (0..sequence.len())
            .map(|i| {
                let a = sequence[i];
                let b = sequence[(i + 1) % sequence.len()];
                [a, b]
            })
            .collect_vec()
    }

    pub fn is_occupied(&self, [e1, e2]: [EdgeID; 2]) -> Option<LoopID> {
        if self.mesh_ref.twin(e1) == e2 {
            return None;
        }
        if let Some(loops_e1) = self.occupied.get(&e1) {
            if let Some(loops_e2) = self.occupied.get(&e2) {
                for &loop_e1 in loops_e1 {
                    for &loop_e2 in loops_e2 {
                        if loop_e1 == loop_e2
                            && (self.loops[loop_e1].contains_pair((e1, e2))
                                || self.loops[loop_e1].contains_pair((e2, e1)))
                        {
                            return Some(loop_e1);
                        }
                    }
                }
            }
        }
        None
    }

    pub fn loops_on_edge(&self, edge: EdgeID) -> Vec<LoopID> {
        self.occupied.get(&edge).cloned().unwrap_or_default()
    }

    pub fn occupied_edgepairs(&self) -> HashSet<(EdgeID, EdgeID)> {
        self.occupied
            .iter()
            .flat_map(|(edge_id, loops_on_edge)| {
                self.mesh_ref
                    .edges(self.mesh_ref.face(edge_id))
                    .flat_map(|neighbor_id| {
                        if loops_on_edge.iter().any(|loop_on_edge| {
                            self.loops_on_edge(neighbor_id).contains(loop_on_edge)
                        }) {
                            vec![(edge_id, neighbor_id), (neighbor_id, edge_id)]
                        } else {
                            vec![]
                        }
                    })
                    .collect_vec()
            })
            .collect()
    }

    pub fn check_loop(&self, lewp: &[EdgeID]) -> Result<(), PropertyViolationError> {
        let edges = lewp;

        if edges.is_empty() {
            return Err(PropertyViolationError::UnknownError);
        }

        // Check if none of the edges are already occupied
        for edge_pair in Self::cycled_windows(edges) {
            if self.is_occupied(edge_pair).is_some() {
                println!("{:?}", edge_pair);
                println!("error 1");
                return Err(PropertyViolationError::UnknownError);
            }
        }

        // Check if the loop contains any duplicates
        let mut seen = HashSet::new();
        for &edge in edges {
            if !seen.insert(edge) {
                println!("{:?}", edge);
                println!("error 4");
                return Err(PropertyViolationError::UnknownError);
            }
        }

        // Check if the loop is valid
        // Loop should alternate between edges that are twins, and edges that are sharing a face.
        // If alternate is true, then the next edge should be a twin of the current edge
        let mut alternate = self.mesh_ref.twin(edges[0]) == edges[1];
        for edge_pair in Self::cycled_windows(edges) {
            if alternate {
                if self.mesh_ref.twin(edge_pair[0]) != edge_pair[1] {
                    println!("error 2");
                    return Err(PropertyViolationError::UnknownError);
                }
                assert!(self.mesh_ref.twin(edge_pair[0]) == edge_pair[1]);
                alternate = false;
            } else {
                if self.mesh_ref.face(edge_pair[0]) != self.mesh_ref.face(edge_pair[1]) {
                    println!("error 3");
                    return Err(PropertyViolationError::UnknownError);
                }
                assert!(self.mesh_ref.face(edge_pair[0]) == self.mesh_ref.face(edge_pair[1]));
                alternate = true;
            }
        }

        Ok(())
    }

    pub fn construct_part_of_loop(
        &self,
        [e1, e2]: [EdgeID; 2],
        domain: &grapff::fixed::FixedGraph<EdgeID, f64>,
        measure: &impl Fn(f64) -> OrderedFloat<f64>,
    ) -> Option<(Vec<EdgeID>, f64)> {
        let (solution, cost) = domain.shortest_path(e1, e2, measure).unwrap_or_default();

        // let flatten = solution;

        // // // If three edges share the same face, remove the middle edge
        // // let mut short = vec![];
        // // for i in 0..flatten.len() {
        // //     if self.mesh_ref.face(flatten[i]) == self.mesh_ref.face(flatten[(i + flatten.len() - 1) % flatten.len()])
        // //         && self.mesh_ref.face(flatten[i]) == self.mesh_ref.face(flatten[(i + 1) % flatten.len()])
        // //     {
        // //         continue;
        // //     }

        // //     short.push(flatten[i]);
        // // }

        Some((solution, *cost))
    }

    pub fn construct_loop(
        &self,
        [e1, e2]: [EdgeID; 2],
        domain: &grapff::fixed::FixedGraph<EdgeID, f64>,
        measure: &impl Fn(f64) -> OrderedFloat<f64>,
    ) -> Option<(Vec<EdgeID>, f64)> {
        if !domain.node_exists(e1)
            || !domain.node_exists(e2)
            || !domain.edge_exists(e1, e2)
            || !domain.edge_exists(e2, e1)
        {
            return None;
        }

        if let (Some(n1), Some(n2)) = (domain.node_to_index(&e1), domain.node_to_index(&e2)) {
            // Get the better direction
            let (n1, n2) = if measure(domain.get_weight(n1, n2).to_owned())
                < measure(domain.get_weight(n2, n1).to_owned())
            {
                (e1, e2)
            } else {
                (e2, e1)
            };

            let (cost, solution) = domain
                .shortest_cycle_edge((n1, n2), measure)
                .unwrap_or_default();

            let flatten = solution;

            // If three edges share the same face, remove the middle edge
            let mut short = vec![];
            for i in 0..flatten.len() {
                if self.mesh_ref.face(flatten[i])
                    == self
                        .mesh_ref
                        .face(flatten[(i + flatten.len() - 1) % flatten.len()])
                    && self.mesh_ref.face(flatten[i])
                        == self.mesh_ref.face(flatten[(i + 1) % flatten.len()])
                {
                    continue;
                }

                short.push(flatten[i]);
            }

            if self.check_loop(&short).is_err() {
                return None;
            }

            Some((short, *cost))
        } else {
            None
        }
    }

    pub fn construct_loop_with_anchors(
        &self,
        anchors: &[[EdgeID; 2]],
        direction: PrincipalDirection,
        flow_graph: &grapff::fixed::FixedGraph<EdgeID, f64>,
        measure: impl Fn(f64) -> OrderedFloat<f64>,
    ) -> Option<(Vec<EdgeID>, f64)> {
        // Filter the original flow graph
        let occupied = self.occupied_edgepairs();
        let filter_edges = |edge: (&EdgeID, &EdgeID)| !occupied.contains(&(*edge.0, *edge.1));
        let filter_nodes = |&node: &EdgeID| {
            !self
                .loops_on_edge(node)
                .iter()
                .any(|&loop_id| self.loops[loop_id].direction == direction)
        };
        let g = flow_graph.filter_edges(filter_edges);
        let g = g.filter_nodes(filter_nodes);

        let mut two_loops = vec![];
        for reverse in [true, false] {
            let mut flipped_anchors = vec![];
            for &[e1, e2] in anchors {
                if let (Some(n1), Some(n2)) =
                    (flow_graph.node_to_index(&e1), flow_graph.node_to_index(&e2))
                {
                    // Get the better direction
                    println!("{:?} {:?}", e1, e2);
                    let (new_e1, new_e2) = if measure(flow_graph.get_weight(n1, n2).to_owned())
                        < measure(flow_graph.get_weight(n2, n1).to_owned())
                    {
                        (e1, e2)
                    } else {
                        (e2, e1)
                    };
                    flipped_anchors.push([new_e1, new_e2]);
                }
            }

            let mut anchors_flattened = flipped_anchors.iter().flatten().copied().collect_vec();
            anchors_flattened.push(anchors_flattened.first().cloned().unwrap());

            let mut new_loop = vec![];

            if reverse {
                anchors_flattened.reverse();
            }

            for (&start, &end) in anchors_flattened.iter().tuple_windows() {
                println!("{:?} {:?}", start, end);
                if let Some((mut part, _)) = self.construct_part_of_loop([start, end], &g, &measure)
                {
                    if part.is_empty() {
                        return None;
                    }
                    part.pop().unwrap();
                    new_loop.extend(part);
                } else {
                    return None;
                }
            }

            // If three edges share the same face, remove the middle edge
            let mut short = vec![];
            for i in 0..new_loop.len() {
                if self.mesh_ref.face(new_loop[i])
                    == self
                        .mesh_ref
                        .face(new_loop[(i + new_loop.len() - 1) % new_loop.len()])
                    && self.mesh_ref.face(new_loop[i])
                        == self.mesh_ref.face(new_loop[(i + 1) % new_loop.len()])
                {
                    continue;
                }

                short.push(new_loop[i]);
            }

            two_loops.push(short);
        }

        let shortest_loop = if two_loops[0].len() < two_loops[1].len() {
            two_loops[0].clone()
        } else {
            two_loops[1].clone()
        };

        println!("{:?}", shortest_loop);
        let res = self.check_loop(&shortest_loop);
        println!("{:?}", res);

        Some((shortest_loop, 0.0))
    }

    pub fn construct_unbounded_loop(
        &self,
        [e1, e2]: [EdgeID; 2],
        direction: PrincipalDirection,
        flow_graph: &grapff::fixed::FixedGraph<EdgeID, f64>,
        measure: impl Fn(f64) -> OrderedFloat<f64>,
    ) -> Option<(Vec<EdgeID>, f64)> {
        // Filter the original flow graph
        let occupied = self.occupied_edgepairs();
        let filter_edges = |edge: (&EdgeID, &EdgeID)| !occupied.contains(&(*edge.0, *edge.1));
        let filter_nodes = |&node: &EdgeID| {
            !self
                .loops_on_edge(node)
                .iter()
                .any(|&loop_id| self.loops[loop_id].direction == direction)
        };
        let g = flow_graph.filter_edges(filter_edges);
        let g = g.filter_nodes(filter_nodes);

        if let Some((option, cost)) = self.construct_loop([e1, e2], &g, &measure) {
            let mut cleaned_option = vec![];
            for edge_id in &option {
                if cleaned_option.contains(&edge_id) {
                    cleaned_option = cleaned_option
                        .into_iter()
                        .take_while(|&x| x != edge_id)
                        .collect_vec();
                }
                cleaned_option.push(edge_id);
            }
            Some((option, cost))
        } else {
            None
        }
    }

    pub fn sample_loops(
        &self,
        n: usize,
        axis: PrincipalDirection,
        flow_graphs: &[grapff::fixed::FixedGraph<EdgeID, f64>; 3],
        measure: impl Fn(f64) -> OrderedFloat<f64> + std::marker::Sync + std::marker::Send,
        score: impl Fn((&[EdgeID], f64)) -> f64 + std::marker::Sync + std::marker::Send,
    ) -> Vec<Vec<EdgeID>> {
        (0..n.pow(4))
            .map(|_| {
                let e1 = self
                    .mesh_ref
                    .edge_ids()
                    .into_iter()
                    .choose(&mut rng())
                    .unwrap();
                let e2 = self.mesh_ref.next(e1);
                [e1, e2]
            })
            .sorted_by_key(|&[e1, e2]| {
                let n1 = flow_graphs[axis as usize].node_to_index(&e1).unwrap();
                let n2 = flow_graphs[axis as usize].node_to_index(&e2).unwrap();
                OrderedFloat(measure(
                    flow_graphs[axis as usize].get_weight(n1, n2).to_owned(),
                ))
            })
            .take(n.pow(2))
            .iter_into_par()
            .filter_map(|es| {
                self.construct_unbounded_loop(es, axis, &flow_graphs[axis as usize], &measure)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .sorted_by_key(|(path, s)| OrderedFloat(score((path, *s))))
            .take(n)
            .map(|(x, _)| x)
            .collect::<Vec<_>>()
    }

    pub fn dual_is_ok(&self) -> bool {
        Dual::from(self.mesh_ref.clone(), &self.loops).is_ok()
    }

    pub fn reconstruct_solution(&mut self, unit: bool, omega: usize) -> Result<(), SolutionError> {
        self.clear();

        if self.loops.len() < 3 {
            return Ok(());
        }

        log::info!(
            "Reconstructing solution ({} loops) with unit: {}, omega: {}.",
            self.loops.len(),
            unit,
            omega
        );

        let dual = Dual::from(self.mesh_ref.clone(), &self.loops)?;
        let polycube = Polycube::from_dual(&dual);

        // check all faces of the polycube have a normal
        for face in polycube.structure.face_ids() {
            let normal = polycube.structure.normal(face);
            if normal.x.is_nan() || normal.y.is_nan() || normal.z.is_nan() {
                return Err(SolutionError::NoPolycube);
            }
        }

        'outer: for _ in 0..10 {
            let layout = Layout::embed(&dual, &polycube);
            if let Ok(ok_layout) = layout {
                self.layout = Some(ok_layout);
                break 'outer;
            }
        }

        if self.layout.is_none() {
            return Err(SolutionError::NoPrimal);
        }
        self.polycube = Some(polycube);
        self.dual = Ok(dual);

        self.resize_polycube(unit);

        log::info!(
            "The constructed solution has quality: {:?}",
            self.get_quality()
        );

        self.quad = Quad::from_layout(self.layout.as_ref().unwrap(), omega);

        Ok(())
    }

    pub fn resize_polycube(&mut self, unit: bool) -> Result<(), SolutionError> {
        if self.dual.is_err() {
            return Err(SolutionError::NoDual);
        }
        let dual = self.dual.as_ref().unwrap();
        if self.polycube.is_none() {
            return Err(SolutionError::NoPolycube);
        }
        let polycube = self.polycube.as_mut().unwrap();

        if unit {
            polycube.resize(dual, None);
            return Ok(());
        } else {
            if self.layout.is_none() {
                return Err(SolutionError::NoPrimal);
            }
            let layout = self.layout.as_ref().unwrap();
            polycube.resize(dual, Some(layout));
            return Ok(());
        }
    }

    pub fn get_quality(&self) -> Option<f64> {
        if self.layout.is_none() {
            return None;
        }
        let layout = self.layout.as_ref().unwrap();
        let beta = 0.001;
        if let (Some(alignment), Some(orthogonality)) = (layout.alignment, layout.orthogonality) {
            Some(alignment + orthogonality - beta * self.loops.len() as f64)
        } else {
            None
        }
    }

    pub fn mutation(
        &self,
        flow_graphs: &[grapff::fixed::FixedGraph<EdgeID, f64>; 3],
    ) -> Option<Self> {
        // Three types of mutation:
        // 1. Add loop(s)
        // 2. Remove loop(s)
        // 3. Replace loop(s)

        let m = |b: f64| OrderedFloat(b.powi(10));
        let s = |(_, s): (&[EdgeID], f64)| s;

        let mut mutated = false;
        let mut mutated_solution = self.clone();
        let mut case = (rand::random::<u8>() % 2) + 1;

        if self.loops.is_empty() {
            return None;
        }

        if self.loops.len() < 5 {
            case = 1;
        }

        match case {
            1 => {
                // Add loop(s)
                let x = rand::random::<u8>() % 3;
                let y = rand::random::<u8>() % 3;
                let z = rand::random::<u8>() % 3;
                if x + y + z == 0 {
                    return None;
                }

                let x_loops = self
                    .sample_loops(x as usize, PrincipalDirection::X, flow_graphs, m, s)
                    .into_iter()
                    .map(|x| (x, PrincipalDirection::X))
                    .collect_vec();
                let y_loops = self
                    .sample_loops(y as usize, PrincipalDirection::Y, flow_graphs, m, s)
                    .into_iter()
                    .map(|y| (y, PrincipalDirection::Y))
                    .collect_vec();
                let z_loops = self
                    .sample_loops(z as usize, PrincipalDirection::Z, flow_graphs, m, s)
                    .into_iter()
                    .map(|z| (z, PrincipalDirection::Z))
                    .collect_vec();

                // Iteratively add the loops, save result if result if valid.
                for (lewp, axis) in x_loops.into_iter().chain(y_loops).chain(z_loops) {
                    let mut candidate_solution = mutated_solution.clone();
                    candidate_solution.add_loop(Loop {
                        edges: lewp,
                        direction: axis,
                    });
                    // Check solution
                    if candidate_solution.dual_is_ok() {
                        mutated_solution = candidate_solution;
                        mutated = true;
                    }
                }
            }
            2 => {
                // Remove loop(s)
                let loop_id = self.loops.keys().choose(&mut rand::rng()).unwrap();
                let mut candidate_solution = mutated_solution.clone();
                candidate_solution.del_loop(loop_id);

                // Check solution
                if candidate_solution.dual_is_ok() {
                    mutated_solution = candidate_solution;
                    mutated = true;
                }
            }
            _ => unreachable!(),
        };

        if !mutated {
            return None;
        }

        if mutated_solution.reconstruct_solution(true, 1).is_err() {
            return None;
        }

        Some(mutated_solution)
    }

    // pub fn mutate_add_loop(&self, nr_loops: usize, flow_graphs: &[Graaf<EdgeID, f64>; 3]) -> Option<Self> {
    //     let edges = self.mesh_ref.edges.keys().collect_vec();

    //     // map edges to alignment of their two neighboring triangles in a vec of tuples
    //     let alignments = edges
    //         .iter()
    //         .map(|&edge| {
    //             let twin = self.mesh_ref.twin(edge);
    //             let alignment = (self.alignment_per_triangle.get(self.mesh_ref.face(edge)).unwrap()
    //                 + self.alignment_per_triangle.get(self.mesh_ref.face(twin)).unwrap())
    //                 / 2.;
    //             (1. - alignment)
    //         })
    //         .collect_vec();

    //     let dist = WeightedIndex::new(&alignments).unwrap();

    //     (0..nr_loops)
    //         .into_par_iter()
    //         .flat_map(move |_| {
    //             let mut rng = rng();
    //             let mut new_solution = self.clone();

    //             // random int between 1 and 3
    //             let random = rand::distributions::Uniform::new_inclusive(1, 3).sample(&mut rng);
    //             for _ in 0..random {
    //                 let direction = [PrincipalDirection::X, PrincipalDirection::Y, PrincipalDirection::Z]
    //                     .choose(&mut rng)
    //                     .unwrap()
    //                     .to_owned();

    //                 // select randomly an edge with skew to lower alignment
    //                 let e1 = edges[dist.sample(&mut rng)];
    //                 let e2 = self.mesh_ref.next(e1);

    //                 let alpha = rand::random::<f64>();
    //                 let measure = |b: f64| b.powi(10);

    //                 // Add guaranteed loop
    //                 if let Some((new_loop, _)) = self.construct_unbounded_loop([e1, e2], direction, &flow_graphs[direction as usize], measure) {
    //                     new_solution.add_loop(Loop { edges: new_loop, direction });
    //                     if new_solution.reconstruct_solution(false).is_err() {
    //                         return None;
    //                     }
    //                 } else {
    //                     error!("Failed to construct loop in direction {:?}", direction);
    //                     return None;
    //                 }
    //             }

    //             Some(new_solution)
    //         })
    //         .map(|solution| (solution.clone(), solution.get_quality()))
    //         .filter(|(_, quality)| quality.is_some())
    //         .max_by_key(|(_, quality)| OrderedFloat(quality.unwrap()))
    //         .map(|(solution, _)| solution)
    // }

    // pub fn mutate_del_loop(&self, nr_loops: usize) -> Option<Self> {
    //     self.loops
    //         .keys()
    //         .choose_multiple(&mut rand::rng(), nr_loops)
    //         .into_par_iter()
    //         .flat_map(|loop_id| {
    //             let mut real_solution = self.clone();
    //             let cur_quality = real_solution.get_quality();
    //             real_solution.del_loop(loop_id);
    //             if real_solution.reconstruct_solution(false).is_ok() {
    //                 let new_quality = real_solution.get_quality();
    //                 if new_quality > cur_quality {
    //                     return Some(real_solution);
    //                 }
    //             }
    //             None
    //         })
    //         .map(|solution| (solution.clone(), solution.get_quality()))
    //         .filter(|(_, quality)| quality.is_some())
    //         .max_by_key(|(_, quality)| OrderedFloat(quality.unwrap()))
    //         .map(|(solution, _)| solution)
    // }

    // pub fn construct_valid_loop_graph(&self, direction: PrincipalDirection) -> Option<Graaf<LoopSegmentID, f64>> {
    //     if let Ok(dual) = &self.dual {
    //         let nodes = dual.loop_structure.edge_ids().into_iter().collect_vec();
    //         let edges = dual
    //             .loop_structure
    //             .edge_ids()
    //             .into_iter()
    //             .flat_map(|edge| {
    //                 let label = dual.segment_to_direction(edge);
    //                 if label == direction {
    //                     return vec![];
    //                 }

    //                 let twin = dual.loop_structure.twin(edge);
    //                 let mut valid_neighbors = vec![twin];

    //                 let neighbors = dual.loop_structure.nexts(edge);
    //                 let face = [vec![edge], neighbors].into_iter().flatten().collect_vec();

    //                 // For each neighbor, figure out their label and orientation.
    //                 let face_with_labels = face
    //                     .into_iter()
    //                     .map(|segment| {
    //                         let label = dual.segment_to_direction(segment);
    //                         let orientation = dual.segment_to_orientation(segment);
    //                         (segment, label, orientation)
    //                     })
    //                     .collect_vec();

    //                 // We are splitting the loop region in two, with a directed edge.
    //                 // Left = Lower
    //                 // Right = Upper

    //                 for i in 1..face_with_labels.len() {
    //                     // Check for (3) Within each loop region boundary, no two loop segments have the same axis label and side label.
    //                     // in other words: if I would cut the loop region like this, would it invalidate (3)?

    //                     // The left side is everything (including current edge) up until and including the neighbor (that will be split)
    //                     let left_side = face_with_labels[..=i].to_vec();
    //                     // Check if the left side is valid
    //                     let mut px = left_side
    //                         .iter()
    //                         .filter(|&&(_, label, orientation)| label == PrincipalDirection::X && orientation == Orientation::Forwards)
    //                         .count();

    //                     let mut py = left_side
    //                         .iter()
    //                         .filter(|&&(_, label, orientation)| label == PrincipalDirection::Y && orientation == Orientation::Forwards)
    //                         .count();

    //                     let mut pz = left_side
    //                         .iter()
    //                         .filter(|&&(_, label, orientation)| label == PrincipalDirection::Z && orientation == Orientation::Forwards)
    //                         .count();

    //                     let mut mx = left_side
    //                         .iter()
    //                         .filter(|&&(_, label, orientation)| label == PrincipalDirection::X && orientation == Orientation::Backwards)
    //                         .count();

    //                     let mut my = left_side
    //                         .iter()
    //                         .filter(|&&(_, label, orientation)| label == PrincipalDirection::Y && orientation == Orientation::Backwards)
    //                         .count();

    //                     let mut mz = left_side
    //                         .iter()
    //                         .filter(|&&(_, label, orientation)| label == PrincipalDirection::Z && orientation == Orientation::Backwards)
    //                         .count();

    //                     match direction {
    //                         PrincipalDirection::X => mx += 1,
    //                         PrincipalDirection::Y => my += 1,
    //                         PrincipalDirection::Z => mz += 1,
    //                     }

    //                     let left_valid = px <= 1 && py <= 1 && pz <= 1 && mx <= 1 && my <= 1 && mz <= 1;
    //                     if left_valid {
    //                         assert!(px + py + pz + mx + my + mz <= 6);
    //                     }

    //                     // The right side is the neighbor (that will be split) and everything after the neighbor (that will be split), and the current edge
    //                     let right_side = face_with_labels[i..].iter().copied().chain(face_with_labels[..1].iter().copied()).collect_vec();
    //                     // Check if the right side is valid
    //                     let mut px = right_side
    //                         .iter()
    //                         .filter(|&&(_, label, orientation)| label == PrincipalDirection::X && orientation == Orientation::Forwards)
    //                         .count();

    //                     let mut py = right_side
    //                         .iter()
    //                         .filter(|&&(_, label, orientation)| label == PrincipalDirection::Y && orientation == Orientation::Forwards)
    //                         .count();

    //                     let mut pz = right_side
    //                         .iter()
    //                         .filter(|&&(_, label, orientation)| label == PrincipalDirection::Z && orientation == Orientation::Forwards)
    //                         .count();

    //                     let mut mx = right_side
    //                         .iter()
    //                         .filter(|&&(_, label, orientation)| label == PrincipalDirection::X && orientation == Orientation::Backwards)
    //                         .count();

    //                     let mut my = right_side
    //                         .iter()
    //                         .filter(|&&(_, label, orientation)| label == PrincipalDirection::Y && orientation == Orientation::Backwards)
    //                         .count();

    //                     let mut mz = right_side
    //                         .iter()
    //                         .filter(|&&(_, label, orientation)| label == PrincipalDirection::Z && orientation == Orientation::Backwards)
    //                         .count();

    //                     match direction {
    //                         PrincipalDirection::X => px += 1,
    //                         PrincipalDirection::Y => py += 1,
    //                         PrincipalDirection::Z => pz += 1,
    //                     }

    //                     let right_valid = px <= 1 && py <= 1 && pz <= 1 && mx <= 1 && my <= 1 && mz <= 1;

    //                     // If both sides are valid, we can add the edge.
    //                     if left_valid && right_valid {
    //                         // Add the edge
    //                         valid_neighbors.push(face_with_labels[i].0);
    //                     }
    //                 }

    //                 valid_neighbors.into_iter().map(move |neighbor| (edge, neighbor, 1.)).collect_vec()
    //             })
    //             .collect_vec();

    //         let graph = Graaf::from(nodes.clone(), edges.clone());

    //         return Some(graph);
    //     }

    //     None
    // }

    // pub fn find_some_valid_loops_through_region(&self, region_id: LoopRegionID, direction: PrincipalDirection) -> Option<Vec<Vec<LoopSegmentID>>> {
    //     if let Ok(dual) = &self.dual {
    //         let region_segments = dual.loop_structure.edges(region_id);
    //         let full_graph = self.construct_valid_loop_graph(direction)?;
    //         let ccs = full_graph
    //             .cc()
    //             .into_iter()
    //             .filter(|cc| cc.len() > 1)
    //             .filter(|cc| cc.iter().any(|&edge| dual.loop_structure.face(edge) == region_id))
    //             .collect_vec();
    //         assert!(ccs.len() == 1);
    //         let reachable_nodes = ccs[0].clone();
    //         let filtered_edges = full_graph
    //             .edges()
    //             .into_iter()
    //             .filter(|(from, to, _)| reachable_nodes.contains(from) && reachable_nodes.contains(to))
    //             .collect_vec();
    //         let filtered_graph = Graaf::from(reachable_nodes.clone(), filtered_edges);

    //         let mut constraint_map = HashMap::new();

    //         for (index, region) in dual.loop_structure.face_ids().into_iter().enumerate() {
    //             for segment in dual
    //                 .loop_structure
    //                 .edges(region)
    //                 .into_iter()
    //                 .filter(|segment| reachable_nodes.contains(segment))
    //             {
    //                 constraint_map.insert(filtered_graph.node_to_index(&segment).unwrap(), index);
    //             }
    //         }

    //         // Find all "shortest" cycles starting from the starting segment to every other reachable segment.
    //         // e.g. starting segment u, any reachable segment v
    //         // find the shortest path u->v, and shortest path v->u, and combine them to form a cycle
    //         // Filter out cycles that invalidate any of the constraints
    //         let mut cycles = vec![];
    //         for segment in region_segments.iter().filter(|&segment| reachable_nodes.contains(segment)) {
    //             let u = filtered_graph.node_to_index(segment).unwrap();
    //             for other_segment in &reachable_nodes {
    //                 let v = filtered_graph.node_to_index(other_segment).unwrap();
    //                 let shortest_path = filtered_graph.shortest_path(u, v, &|x| x).unwrap().1;
    //                 // remove last element of shortest path, as it is the same as the first element of shortest_path_back
    //                 let shortest_path = shortest_path.iter().copied().take(shortest_path.len() - 1).collect_vec();
    //                 let shortest_path_back = filtered_graph.shortest_path(v, u, &|x| x).unwrap().1;
    //                 // remove last element of shortest path, as it is the same as the first element of shortest_path_back
    //                 let shortest_path_back = shortest_path_back.iter().copied().take(shortest_path_back.len() - 1).collect_vec();
    //                 let cycle = [shortest_path, shortest_path_back].concat();
    //                 cycles.push(cycle);
    //             }
    //         }

    //         // Filter out duplicates
    //         let cycles = cycles
    //             .into_iter()
    //             .unique_by(|cycle| {
    //                 let mut sorted = cycle.clone();
    //                 sorted.sort();
    //                 sorted
    //             })
    //             .collect_vec();

    //         // convert NodeIndex to SegmentID
    //         let cycles = cycles
    //             .iter()
    //             .map(|cycle| cycle.iter().map(|&edge| filtered_graph.index_to_node(edge).unwrap().to_owned()).collect_vec())
    //             .collect_vec();

    //         // Filter such that all cycles are of length larger than 2
    //         let cycles = cycles.into_iter().filter(|cycle| cycle.len() > 2).collect_vec();

    //         // Filter such that all cycles do not traverse the same loop region twice
    //         let cycles = cycles
    //             .into_iter()
    //             .filter(|cycle| {
    //                 let regions = cycle.iter().map(|&edge| dual.loop_structure.face(edge)).collect_vec();
    //                 let unique_regions = regions.iter().unique().count();
    //                 unique_regions * 2 == regions.len()
    //             })
    //             .collect_vec();

    //         return Some(cycles);
    //     }
    //     None
    // }

    // pub fn construct_guaranteed_loop(
    //     &self,
    //     [e1, e2]: [EdgeID; 2],
    //     direction: PrincipalDirection,
    //     flow_graph: &Graaf<[EdgeID; 2], (f64, f64, f64)>,
    //     measure: impl Fn((f64, f64, f64)) -> f64,
    // ) -> Option<Vec<EdgeID>> {
    //     if let Ok(dual) = &self.dual {
    //         let region_id = dual
    //             .loop_structure
    //             .face_ids()
    //             .iter()
    //             .find(|&&region_id| {
    //                 let verts = &dual.loop_structure.faces[region_id].verts;
    //                 verts.contains(&self.mesh_ref.endpoints(e1).0)
    //                     || verts.contains(&self.mesh_ref.endpoints(e1).1)
    //                     || verts.contains(&self.mesh_ref.endpoints(e2).0)
    //                     || verts.contains(&self.mesh_ref.endpoints(e2).1)
    //             })
    //             .unwrap()
    //             .to_owned();

    //         if let Some(cycle_to_edges) = self.find_some_valid_loops_through_region(region_id, direction) {
    //             let chosen_cycle = cycle_to_edges.iter().choose(&mut rand::rng()).unwrap();

    //             let mut augmented_cycle = chosen_cycle.clone();

    //             augmented_cycle = augmented_cycle.into_iter().rev().collect_vec();

    //             // Find a path that goes exactly through the selected segments of the cycle.

    //             let all_segments = chosen_cycle
    //                 .iter()
    //                 .map(|&edge| dual.loop_structure.nexts(edge))
    //                 .flatten()
    //                 .collect::<HashSet<_>>();
    //             let blocked_segments = all_segments
    //                 .iter()
    //                 .copied()
    //                 .filter(|&segment| !augmented_cycle.contains(&segment))
    //                 .collect::<HashSet<_>>();
    //             let blocked_edges = blocked_segments
    //                 .into_iter()
    //                 .flat_map(|segment| dual.segment_to_edges(segment))
    //                 .collect::<HashSet<_>>();

    //             let mut blocked_edges2 = HashSet::new();
    //             // make sure every two segments in chosen_cycle are pairs, it could be the case that the 1st and nth segment are a pair
    //             if augmented_cycle[0] == dual.loop_structure.twin(chosen_cycle[augmented_cycle.len() - 1]) {
    //                 augmented_cycle.push(augmented_cycle[0]);
    //                 augmented_cycle.remove(0);
    //             }

    //             // Find all directed edges passing through the selected segments (chosen_cycle)
    //             for selected_segment_pair in augmented_cycle.windows(2) {
    //                 let (segment1, segment2) = (selected_segment_pair[0], selected_segment_pair[1]);
    //                 // we consider this pair only if segment1 is the first segment, and twin of segment2
    //                 if segment1 != dual.loop_structure.twin(segment2) {
    //                     continue;
    //                 }

    //                 let edges = dual.segment_to_edges(segment1);
    //                 // their edges are the same, but we only consider segment1, as its the first segment

    //                 // for every two edges, find the third edge of the passed triangle
    //                 for edge_pair in edges.windows(2) {
    //                     let (e1, e2) = (edge_pair[0], edge_pair[1]);

    //                     if e1 == self.mesh_ref.twin(e2) {
    //                         continue;
    //                     }

    //                     let triangle = self.mesh_ref.face(e1);
    //                     assert!(triangle == self.mesh_ref.face(e2));

    //                     let third_edge = self.mesh_ref.edges(triangle).into_iter().find(|&e| e != e1 && e != e2).unwrap();

    //                     // figure out if the third edge is adjacent to this segment (or to its twin)
    //                     let this_region = dual.loop_structure.face(segment1);
    //                     let this_region_verts = &dual.loop_structure.faces[this_region].verts;
    //                     let (vert1, vert2) = self.mesh_ref.endpoints(third_edge);
    //                     if this_region_verts.contains(&vert1) && this_region_verts.contains(&vert2) {
    //                         // this is edge is inside the region adjacent to the segment
    //                         // this means that we allow traversal FROM this edge, into the segment/loop
    //                         // this means we do NOT allow traversal FROM the segment/loop, into this edge

    //                         // as such, we block the edges from the segment/loop to this edge
    //                         blocked_edges2.insert([third_edge, e1]);
    //                         blocked_edges2.insert([third_edge, e2]);
    //                     } else {
    //                         // this edge is outside the region adjacent to the segment
    //                         // this means that we allow traversal FROM the segment/loop, into this edge
    //                         // this means we do NOT allow traversal FROM this edge, into the segment/loop

    //                         // as such, we block the edges from this edge to the segment/loop

    //                         blocked_edges2.insert([e1, third_edge]);
    //                         blocked_edges2.insert([e2, third_edge]);
    //                     }
    //                 }
    //             }

    //             let filter = |(a, b): (&[EdgeID; 2], &[EdgeID; 2])| {
    //                 !blocked_edges.contains(&a[0])
    //                     && !blocked_edges.contains(&a[1])
    //                     && !blocked_edges.contains(&b[0])
    //                     && !blocked_edges.contains(&b[1])
    //                     && !blocked_edges2.contains(a)
    //                     && !blocked_edges2.contains(b)
    //             };

    //             let g_original = flow_graph;
    //             let g = g_original.filter(filter);

    //             let (option_a, _) = Self::construct_loop(&[e1, e2], &g, &measure);
    //             let (option_b, _) = Self::construct_loop(&[e2, e1], &g, &measure);

    //             // The path may contain self intersections. We can remove these.
    //             // If duplicated vertices are present, remove everything between them.
    //             let mut cleaned_option_a = vec![];
    //             for edge_id in option_a {
    //                 if cleaned_option_a.contains(&edge_id) {
    //                     cleaned_option_a = cleaned_option_a.into_iter().take_while(|&x| x != edge_id).collect_vec();
    //                 }
    //                 cleaned_option_a.push(edge_id);
    //             }

    //             let mut cleaned_option_b = vec![];
    //             for edge_id in option_b {
    //                 if cleaned_option_b.contains(&edge_id) {
    //                     cleaned_option_b = cleaned_option_b.into_iter().take_while(|&x| x != edge_id).collect_vec();
    //                 }
    //                 cleaned_option_b.push(edge_id);
    //             }

    //             let best_option = if cleaned_option_a.len() > cleaned_option_b.len() {
    //                 cleaned_option_a
    //             } else {
    //                 cleaned_option_b
    //             };

    //             if best_option.len() < 5 {
    //                 return None;
    //             }

    //             return Some(best_option);
    //         }
    //     }

    //     None
    // }

    // pub fn construct_guaranteed_loop2(&self, region_id: RegionID, selected_edges: [EdgeID; 2], direction: PrincipalDirection) -> Vec<Vec<EdgeID>> {
    //     println!("Constructing guaranteed loop");

    //     let mut candidate_paths = vec![];

    //     if let Ok(dual) = &self.dual {
    //         if let Some(cycle_to_edges) = self.find_some_valid_loops_through_region(region_id, direction) {
    //             candidate_paths = cycle_to_edges
    //                 .into_iter()
    //                 .flat_map(|chosen_cycle| {
    //                     let mut augmented_cycle = chosen_cycle.clone();
    //                     // make sure every two segments in chosen_cycle are pairs, it could be the case that the 1st and nth segment are a pair
    //                     if augmented_cycle[0] == dual.loop_structure.twin(chosen_cycle[augmented_cycle.len() - 1]) {
    //                         augmented_cycle.push(augmented_cycle[0]);
    //                         augmented_cycle.remove(0);
    //                     }

    //                     let mut all_nodes = vec![];
    //                     let mut all_edges = vec![];

    //                     // Construct the domain of the path: stitch the consecutive loop regions together.
    //                     for selected_segment_pair in augmented_cycle.windows(2) {
    //                         if selected_segment_pair[0] != dual.loop_structure.twin(selected_segment_pair[1]) {
    //                             continue;
    //                         }
    //                         // Look at the lower segment of the pair
    //                         let segment = selected_segment_pair[0];
    //                         // Get its loop region
    //                         let region = dual.loop_structure.face(segment);
    //                         // Figure out the domain
    //                         let verts = dual.loop_structure.faces[region].verts.clone();
    //                         let faces: Vec<FaceID> = verts
    //                             .iter()
    //                             .flat_map(|&v| self.mesh_ref.star(v))
    //                             .filter(|&face| self.mesh_ref.corners(face).iter().all(|c| verts.contains(c)))
    //                             .collect_vec();
    //                         let nodes: Vec<NodeCopy> = faces
    //                             .into_iter()
    //                             .flat_map(|face_id| {
    //                                 let edges = self.mesh_ref.edges(face_id);
    //                                 assert!(edges.len() == 3);
    //                                 vec![
    //                                     NodeCopy {
    //                                         id: [edges[0], edges[1]],
    //                                         t: 0,
    //                                     },
    //                                     NodeCopy {
    //                                         id: [edges[1], edges[0]],
    //                                         t: 0,
    //                                     },
    //                                     NodeCopy {
    //                                         id: [edges[0], edges[2]],
    //                                         t: 0,
    //                                     },
    //                                     NodeCopy {
    //                                         id: [edges[2], edges[0]],
    //                                         t: 0,
    //                                     },
    //                                     NodeCopy {
    //                                         id: [edges[1], edges[2]],
    //                                         t: 0,
    //                                     },
    //                                     NodeCopy {
    //                                         id: [edges[2], edges[1]],
    //                                         t: 0,
    //                                     },
    //                                 ]
    //                             })
    //                             .collect_vec();

    //                         let edges: Vec<(NodeCopy, NodeCopy)> = nodes
    //                             .clone()
    //                             .into_iter()
    //                             .flat_map(|node| {
    //                                 self.mesh_ref.neighbor_function_edgepairgraph()(node.id)
    //                                     .into_iter()
    //                                     .map(move |neighbor| {
    //                                         assert!(self.mesh_ref.twin(node.id[1]) == neighbor[0]);
    //                                         (node, NodeCopy { id: neighbor, t: 0 })
    //                                     })
    //                                     .filter(|(a, b)| nodes.contains(a) && nodes.contains(b))
    //                             })
    //                             .collect_vec();

    //                         // Get all nodes that are ON the segment
    //                         let faces_on_segment = dual
    //                             .segment_to_edges_excl(segment)
    //                             .into_iter()
    //                             .map(|edge| self.mesh_ref.face(edge))
    //                             .collect::<HashSet<_>>();
    //                         let nodes_on_segment = faces_on_segment
    //                             .into_iter()
    //                             .flat_map(|face_id| {
    //                                 let edges = self.mesh_ref.edges(face_id);
    //                                 assert!(edges.len() == 3);
    //                                 vec![
    //                                     NodeCopy {
    //                                         id: [edges[0], edges[1]],
    //                                         t: 0,
    //                                     },
    //                                     NodeCopy {
    //                                         id: [edges[1], edges[0]],
    //                                         t: 0,
    //                                     },
    //                                     NodeCopy {
    //                                         id: [edges[0], edges[2]],
    //                                         t: 0,
    //                                     },
    //                                     NodeCopy {
    //                                         id: [edges[2], edges[0]],
    //                                         t: 0,
    //                                     },
    //                                     NodeCopy {
    //                                         id: [edges[1], edges[2]],
    //                                         t: 0,
    //                                     },
    //                                     NodeCopy {
    //                                         id: [edges[2], edges[1]],
    //                                         t: 0,
    //                                     },
    //                                 ]
    //                             })
    //                             .collect_vec();

    //                         let edges_through_segment: Vec<(NodeCopy, NodeCopy)> = nodes_on_segment
    //                             .clone()
    //                             .into_iter()
    //                             .flat_map(|node| {
    //                                 self.mesh_ref.neighbor_function_edgepairgraph()(node.id)
    //                                     .into_iter()
    //                                     .map(move |neighbor| {
    //                                         assert!(self.mesh_ref.twin(node.id[1]) == neighbor[0]);
    //                                         (node, NodeCopy { id: neighbor, t: 0 })
    //                                     })
    //                                     .filter(|(a, b)| nodes_on_segment.contains(a) && nodes_on_segment.contains(b))
    //                                 // .filter(|(a, b)| {
    //                                 //     let (a1, a2) = self.mesh_ref.endpoints(a.id[0]);
    //                                 //     let (b1, b2) = self.mesh_ref.endpoints(b.id[1]);
    //                                 //     verts.contains(&a1) && verts.contains(&a2) && !verts.contains(&b1) && !verts.contains(&b2)
    //                                 // })
    //                             })
    //                             .collect_vec();

    //                         let edges_before_segment: Vec<(NodeCopy, NodeCopy)> = nodes_on_segment
    //                             .clone()
    //                             .into_iter()
    //                             .flat_map(|node| {
    //                                 self.mesh_ref.neighbor_function_edgepairgraph()(node.id)
    //                                     .into_iter()
    //                                     .map(move |neighbor| {
    //                                         assert!(self.mesh_ref.twin(node.id[1]) == neighbor[0]);
    //                                         (node, NodeCopy { id: neighbor, t: 0 })
    //                                     })
    //                                     .filter(|(a, b)| nodes_on_segment.contains(a) && nodes_on_segment.contains(b))
    //                                 // .filter(|(a, b)| {
    //                                 //     let (a1, a2) = self.mesh_ref.endpoints(a.id[0]);
    //                                 //     let (b1, b2) = self.mesh_ref.endpoints(b.id[1]);
    //                                 //     verts.contains(&a1) && verts.contains(&a2) && verts.contains(&b1) && verts.contains(&b2)
    //                                 // })
    //                             })
    //                             .flat_map(|(a, b)| [(a, b), (b, a)])
    //                             .collect_vec();

    //                         all_nodes.extend(nodes);
    //                         all_nodes.extend(nodes_on_segment);
    //                         all_edges.extend(edges);
    //                         all_edges.extend(edges_through_segment);
    //                         all_edges.extend(edges_before_segment);
    //                     }

    //                     //     // Get the domain of this loop region
    //                     //     let domain = self
    //                     //         .mesh_ref
    //                     //         .verts_to_edges(&dual.loop_structure.faces[region].verts.clone().into_iter().collect_vec());
    //                     //     let domain_copy = domain.clone().into_iter().map(|v| NodeCopy { id: v, t: 0 }).collect_vec();

    //                     //     // For any vertex, check its adjacent vertices, if this adjacent vertex is also in the domain, we have the edge
    //                     //     let edges = domain
    //                     //         .iter()
    //                     //         .flat_map(|&vertex| {
    //                     //             let adjacent = self.mesh_ref.neighbor_function_edgepairgraph()(vertex);
    //                     //             adjacent
    //                     //                 .iter()
    //                     //                 .filter_map(|&adjacent_vertex| {
    //                     //                     if domain.contains(&adjacent_vertex) {
    //                     //                         Some((NodeCopy { id: vertex, t: 0 }, NodeCopy { id: adjacent_vertex, t: 0 }, 1.))
    //                     //                     } else {
    //                     //                         None
    //                     //                     }
    //                     //                 })
    //                     //                 .collect_vec()
    //                     //         })
    //                     //         .collect_vec();

    //                     //     let mut crossing_edges = vec![];
    //                     //     // for every two edges in this segment, find the third edge
    //                     //     for edge_pair in dual.segment_to_edges_excl(segment).windows(2) {
    //                     //         let (e1, e2) = (edge_pair[0], edge_pair[1]);
    //                     //         let common_endpoint = self.mesh_ref.common_endpoint(e1, e2).unwrap();
    //                     //         // Only look at the consecutive edges that are not twins.
    //                     //         if e1 == self.mesh_ref.twin(e2) {
    //                     //             continue;
    //                     //         }

    //                     //         // Find the triangle of these two edges
    //                     //         let triangle = self.mesh_ref.face(e1);
    //                     //         assert!(triangle == self.mesh_ref.face(e2));
    //                     //         // Get the third edge of this triangle
    //                     //         let third_edge = self.mesh_ref.edges(triangle).into_iter().find(|&e| e != e1 && e != e2).unwrap();

    //                     //         // Figure out if the third edge is adjacent to this segment (or to its twin)
    //                     //         let (vert1, vert2) = self.mesh_ref.endpoints(third_edge);
    //                     //         let third_edge_is_in_domain = domain.contains(&vert1) && domain.contains(&vert2);
    //                     //         if third_edge_is_in_domain {
    //                     //             // This means we traverse from this loop region INTO the next loop region (corresponding to the upper segment)
    //                     //             crossing_edges.push((NodeCopy { id: vert1, t: 0 }, NodeCopy { id: common_endpoint, t: 0 }, 1.));
    //                     //             crossing_edges.push((NodeCopy { id: vert2, t: 0 }, NodeCopy { id: common_endpoint, t: 0 }, 1.));
    //                     //         } else {
    //                     //             // We do not allow to traverse from the upper segment INTO the lower segment
    //                     //         }
    //                     //     }

    //                     //     // Add the edges to the domain
    //                     //     all_nodes.extend(domain_copy);
    //                     //     all_edges.extend(edges);
    //                     //     all_edges.extend(crossing_edges);
    //                     // }

    //                     // // Add the starting loop region again (but as a duplicate, so that we can find a shortest path to itself..)
    //                     // let last_segment = augmented_cycle[augmented_cycle.len() - 1];
    //                     // let last_region = dual.loop_structure.face(last_segment);
    //                     // let last_domain = dual.loop_structure.faces[last_region].verts.clone();
    //                     // let last_domain_copy = last_domain.clone().into_iter().map(|v| NodeCopy { id: v, t: 1 }).collect_vec();
    //                     // let last_edges = last_domain
    //                     //     .iter()
    //                     //     .flat_map(|&vertex| {
    //                     //         let adjacent = self.mesh_ref.vneighbors(vertex);
    //                     //         adjacent
    //                     //             .iter()
    //                     //             .filter_map(|&adjacent_vertex| {
    //                     //                 if last_domain.contains(&adjacent_vertex) {
    //                     //                     Some((NodeCopy { id: vertex, t: 1 }, NodeCopy { id: adjacent_vertex, t: 1 }, 1.))
    //                     //                 } else {
    //                     //                     None
    //                     //                 }
    //                     //             })
    //                     //             .collect_vec()
    //                     //     })
    //                     //     .collect_vec();

    //                     // let mut crossing_edges = vec![];
    //                     // // for every two edges in this segment, find the third edge
    //                     // for edge_pair in dual.segment_to_edges_excl(last_segment).windows(2) {
    //                     //     let (e1, e2) = (edge_pair[0], edge_pair[1]);
    //                     //     let common_endpoint = self.mesh_ref.common_endpoint(e1, e2).unwrap();
    //                     //     // Only look at the consecutive edges that are not twins.
    //                     //     if e1 == self.mesh_ref.twin(e2) {
    //                     //         continue;
    //                     //     }

    //                     //     // Find the triangle of these two edges
    //                     //     let triangle = self.mesh_ref.face(e1);
    //                     //     assert!(triangle == self.mesh_ref.face(e2));
    //                     //     // Get the third edge of this triangle
    //                     //     let third_edge = self.mesh_ref.edges(triangle).into_iter().find(|&e| e != e1 && e != e2).unwrap();

    //                     //     // Figure out if the third edge is adjacent to this segment (or to its twin)
    //                     //     let (vert1, vert2) = self.mesh_ref.endpoints(third_edge);
    //                     //     let third_edge_is_in_domain = last_domain.contains(&vert1) && last_domain.contains(&vert2);
    //                     //     if third_edge_is_in_domain {
    //                     //         // This means we traverse from this loop region INTO the next loop region (corresponding to the upper segment)
    //                     //         crossing_edges.push((NodeCopy { id: vert1, t: 0 }, NodeCopy { id: common_endpoint, t: 1 }, 1.));
    //                     //         crossing_edges.push((NodeCopy { id: vert2, t: 0 }, NodeCopy { id: common_endpoint, t: 1 }, 1.));
    //                     //     } else {
    //                     //         // We do not allow to traverse from the upper segment INTO the lower segment
    //                     //     }
    //                     // }

    //                     // all_nodes.extend(last_domain_copy);
    //                     // all_edges.extend(last_edges);
    //                     // all_edges.extend(crossing_edges);

    //                     let all_edges_weighted = all_edges
    //                         .into_iter()
    //                         .map(|(a, b)| {
    //                             let node = a.id;
    //                             let neighbor = b.id;

    //                             let middle_edge = node[1];
    //                             let (m1, m2) = self.mesh_ref.endpoints(middle_edge);

    //                             let start_edge = node[0];
    //                             let end_edge = neighbor[1];
    //                             // Vector from middle_edge to start_edge
    //                             let vector_a = self.mesh_ref.midpoint(start_edge) - self.mesh_ref.midpoint(middle_edge);
    //                             // Vector from middle_edge to end_edge
    //                             let vector_b = self.mesh_ref.midpoint(end_edge) - self.mesh_ref.midpoint(middle_edge);

    //                             // Vector from middle_edge to m1
    //                             let vector_m1 = self.mesh_ref.position(m1) - self.mesh_ref.midpoint(middle_edge);
    //                             // Vector from middle_edge to m2
    //                             let vector_m2 = self.mesh_ref.position(m2) - self.mesh_ref.midpoint(middle_edge);

    //                             // Angle around m1
    //                             let angle_around_m1 = self.mesh_ref.vec_angle(vector_a, vector_m1) + self.mesh_ref.vec_angle(vector_b, vector_m1);

    //                             // Angle around m2
    //                             let angle_around_m2 = self.mesh_ref.vec_angle(vector_a, vector_m2) + self.mesh_ref.vec_angle(vector_b, vector_m2);

    //                             // Whichever angle is shorter is the "real" angle
    //                             let angle = if angle_around_m1 < angle_around_m2 {
    //                                 angle_around_m1
    //                             } else {
    //                                 angle_around_m2
    //                             };

    //                             // Weight is based on how far the angle is from 180 degrees
    //                             let temp = PI - angle;
    //                             let angular_weight = if temp < 0. { 0. } else { temp };

    //                             // Alignment per edge
    //                             // Vector_a
    //                             // Find the face that is bounded by the two edges
    //                             let face_a = self.mesh_ref.face(start_edge);
    //                             let alignment_vector_a = (-vector_a).cross(&self.mesh_ref.normal(face_a)).angle(&direction.into());

    //                             // Vector_b
    //                             // Find the face that is bounded by the two edges
    //                             let face_b = self.mesh_ref.face(end_edge);
    //                             let alignment_vector_b = vector_b.cross(&self.mesh_ref.normal(face_b)).angle(&direction.into());

    //                             let weight = (angular_weight, (alignment_vector_a + alignment_vector_b) / 2., 0.);

    //                             (a, b, weight)
    //                         })
    //                         .collect_vec();

    //                     let g = Graaf::from(all_nodes, all_edges_weighted);

    //                     // between 5 and 15
    //                     let alpha = rand::random::<f64>();

    //                     let measure = |(a, b, c): (f64, f64, f64)| alpha * a.powi(10) + (1. - alpha) * b.powi(10);

    //                     let [e1, e2] = selected_edges;

    //                     let a = g.node_to_index(&NodeCopy { id: [e1, e2], t: 0 }).unwrap();
    //                     let b = g.node_to_index(&NodeCopy { id: [e2, e1], t: 0 }).unwrap();
    //                     let mut option_a = g
    //                         .shortest_cycle(a, &measure)
    //                         .unwrap_or_default()
    //                         .into_iter()
    //                         .map(|node_index| g.index_to_node(node_index).unwrap().to_owned())
    //                         .flat_map(|node_copy| node_copy.id)
    //                         .collect_vec();
    //                     let mut option_b = g
    //                         .shortest_cycle(b, &measure)
    //                         .unwrap_or_default()
    //                         .into_iter()
    //                         .map(|node_index| g.index_to_node(node_index).unwrap().to_owned())
    //                         .flat_map(|node_copy| node_copy.id)
    //                         .collect_vec();

    //                     assert!(option_a.len() % 2 == 0);
    //                     assert!(option_b.len() % 2 == 0);

    //                     let mut option_a_valid = true;
    //                     let mut option_b_valid = true;

    //                     // If path has duplicates, the option is invalid
    //                     option_a_valid = option_a.iter().unique().count() == option_a.len();
    //                     option_b_valid = option_b.iter().unique().count() == option_b.len();

    //                     // If path has less than 5 edges, the option is invalid
    //                     option_a_valid = option_a.len() >= 5;
    //                     option_b_valid = option_b.len() >= 5;

    //                     if !option_a_valid && !option_b_valid {
    //                         return None;
    //                     }

    //                     if option_a_valid && !option_b_valid {
    //                         return Some(option_a);
    //                     }

    //                     if !option_a_valid && option_b_valid {
    //                         return Some(option_b);
    //                     }

    //                     Some(option_a)
    //                 })
    //                 .collect_vec();
    //         }
    //     }

    //     println!("paths: {:?}", candidate_paths);

    //     println!("Found {} candidate paths", candidate_paths.len());

    //     candidate_paths
    // }
}
