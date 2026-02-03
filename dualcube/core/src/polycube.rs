use crate::dual::{Dual, LoopRegionID};
use crate::layout::Layout;
use crate::prelude::*;
use bimap::BiHashMap;
use itertools::Itertools;
use mehsh::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct POLYCUBE;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Polycube {
    pub structure: Mesh<POLYCUBE>,

    // Mapping from dual to primal
    pub region_to_vertex: BiHashMap<LoopRegionID, VertKey<POLYCUBE>>,
}

impl Polycube {
    pub fn from_dual(dual: &Dual) -> Self {
        let primal_vertices = dual.loop_structure.face_ids();
        let primal_faces = dual.loop_structure.vert_ids();

        let mut region_to_vertex = BiHashMap::new();

        // Each face to an int
        let vert_to_int: HashMap<LoopRegionID, usize> = primal_vertices.clone().into_iter().enumerate().map(|(i, f)| (f, i)).collect();

        // Create the dual (primal)
        // By creating the primal faces
        let faces = primal_faces
            .iter()
            .map(|&dual_vert_id| dual.loop_structure.faces(dual_vert_id).collect_vec().into_iter().rev().collect_vec())
            .collect_vec();
        let int_faces = faces.iter().map(|face| face.iter().map(|vert| vert_to_int[vert]).collect_vec()).collect_vec();

        let (primal, vert_map, _) = Mesh::<POLYCUBE>::from(&int_faces, &vec![Vector3D::new(0., 0., 0.); primal_vertices.len()]).unwrap();

        for vert_id in &primal.vert_ids() {
            let region_id = primal_vertices[vert_map.id(vert_id).unwrap().to_owned()];
            region_to_vertex.insert(region_id, vert_id.to_owned());
        }

        let mut polycube = Self {
            structure: primal,
            region_to_vertex,
        };

        polycube.resize(dual, None);

        log::info!("Constructed a polycube with {} faces.", polycube.structure.nr_faces());

        polycube
    }

    pub fn resize(&mut self, dual: &Dual, layout: Option<&Layout>) {
        let mut vert_to_coord = HashMap::new();
        for vert_id in self.structure.vert_ids() {
            vert_to_coord.insert(vert_id, [0., 0., 0.]);
        }

        let mut levels = [Vec::new(), Vec::new(), Vec::new()];

        // Fix the positions of the vertices that are in the same level
        for direction in [PrincipalDirection::X, PrincipalDirection::Y, PrincipalDirection::Z] {
            for (level, zones) in dual.level_graphs.levels[direction as usize].iter().enumerate() {
                let verts_in_level = zones
                    .iter()
                    .flat_map(|&zone_id| {
                        dual.level_graphs.zones[zone_id]
                            .regions
                            .iter()
                            .map(|&region_id| self.region_to_vertex.get_by_left(&region_id).unwrap().to_owned())
                    })
                    .collect_vec();

                let value = layout.map_or(level as f64, |lay| {
                    let verts_in_mesh = verts_in_level
                        .iter()
                        .map(|vert| lay.granulated_mesh.position(lay.vert_to_corner.get_by_left(vert).unwrap().to_owned())[direction as usize])
                        .collect_vec();
                    mehsh::utils::math::calculate_average_f64(verts_in_mesh.into_iter())
                });

                levels[direction as usize].push((value, verts_in_level.clone()));
            }
        }

        // scale the coordinates s.t. smallest edge is 1, and all other edges are multiples of 1 (integer lengths)
        let mut min_distance = f64::MAX;
        for direction in [PrincipalDirection::X, PrincipalDirection::Y, PrincipalDirection::Z] {
            let direction_levels = &levels[direction as usize];
            for (level1, level2) in direction_levels.iter().tuple_windows() {
                let distance = (level2.0 - level1.0).abs();
                if distance < min_distance {
                    min_distance = distance;
                }
            }
        }
        assert!(!(min_distance == 0.), "The distance between two levels is 0. This should not happen.");
        let scale = 1. / min_distance;
        for direction in [PrincipalDirection::X, PrincipalDirection::Y, PrincipalDirection::Z] {
            for (level, verts_in_level) in levels[direction as usize].iter() {
                // println!("Level {:?}", level);
                let value = level * scale;
                // round to nearest integer
                let value = value.round();
                // println!("New level {:?}", value);
                for vert in verts_in_level {
                    vert_to_coord.get_mut(vert).unwrap()[direction as usize] = value;
                }
            }
        }

        // Assign the positions to the vertices
        for vert_id in self.structure.vert_ids() { 
            let [x, y, z] = vert_to_coord[&vert_id];
            self.structure.set_position(vert_id, Vector3D::new(x, y, z));
        }
    }

    // Get the signed direction of an edge in the polycube (+X, -X, +Y, -Y, +Z, or -Z)
    pub fn get_direction_of_edge(&self, a: VertKey<POLYCUBE>, b: VertKey<POLYCUBE>) -> (PrincipalDirection, Orientation) {
        to_principal_direction(self.structure.vector(self.structure.edge_between_verts(a, b).unwrap().0))
    }

    pub fn to_dotgraph(dual: &Dual, layout: &Layout, path: &PathBuf) -> Result<(), std::io::Error> {
        let mut file = std::fs::File::create(path)?;

        let mut polycube = Self::from_dual(dual);
        polycube.resize(dual, Some(layout));

        let mut vert_ids = ids::IdMap::<VERT, POLYCUBE>::new();
        for (i, vert_id) in polycube.structure.vert_ids().into_iter().enumerate() {
            vert_ids.insert(i, vert_id);
        }

        writeln!(
            file,
            "/ comments are lines starting with a slash (/), they should be ignored when parsing the file"
        )?;
        writeln!(file, "/ ")?;
        writeln!(file, "/ number of faces, number of edges, and number of vertices:")?;
        writeln!(
            file,
            "{} {} {}",
            polycube.structure.face_ids().len(),
            polycube.structure.edge_ids().len() / 2,
            polycube.structure.vert_ids().len(),
        )?;

        writeln!(file, "/ faces in format of:")?;
        writeln!(file, "/ <VERT_ID> <VERT_ID> <VERT_ID> <VERT_ID>")?;
        for face_id in polycube.structure.face_ids() {
            let vertices = polycube.structure.vertices(face_id);
            writeln!(
                file,
                "{}",
                vertices
                    .map(|vert_id| format!("{}", vert_ids.id(&vert_id).unwrap()))
                    .collect_vec()
                    .into_iter()
                    .rev()
                    .join(" ")
            )?;
        }

        writeln!(file, "/ edges in format of:")?;
        writeln!(file, "/ <VERT_ID> <VERT_ID> <AXIS_LABEL> <TARGET_LENGTH>")?;
        let mut edge_strings = vec![];
        let mut edge_lengths = vec![];
        for edge_id in polycube.structure.edge_ids() {
            let direction_vector = polycube.structure.vector(edge_id).normalize();
            let (direction, orientation) = to_principal_direction(direction_vector);
            if orientation == Orientation::Backwards {
                continue;
            }
            let label = match direction {
                PrincipalDirection::X => "X",
                PrincipalDirection::Y => "Y",
                PrincipalDirection::Z => "Z",
            };
            let Some([v1, v2]) = polycube.structure.vertices(edge_id).collect_array::<2>() else {
                panic!()
            };
            edge_strings.push(format!("{} {} {label}", vert_ids.id(&v1).unwrap(), vert_ids.id(&v2).unwrap()));

            let path = layout.edge_to_path.get(&edge_id).unwrap();
            let length_of_path = path.windows(2).map(|w| layout.granulated_mesh.distance(w[0], w[1])).sum::<f64>();
            edge_lengths.push(length_of_path);
        }

        let min_edge_length = edge_lengths.iter().cloned().fold(f64::MAX, f64::min);
        let edge_lengths_int = edge_lengths
            .into_iter()
            .map(|length: f64| (length / min_edge_length).ceil() as u32)
            .collect::<Vec<_>>();

        for i in 0..edge_strings.len() {
            writeln!(file, "{} {}", edge_strings[i], edge_lengths_int[i])?;
        }

        Ok(())
    }
}
