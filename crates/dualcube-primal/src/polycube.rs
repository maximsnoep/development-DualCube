use crate::layout::Layout;
use bimap::BiHashMap;
use dualcube_dual::prelude::*;
use dualcube_types::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
        let vert_to_int: HashMap<LoopRegionID, usize> = primal_vertices
            .clone()
            .into_iter()
            .enumerate()
            .map(|(i, f)| (f, i))
            .collect();

        // Create the dual (primal)
        // By creating the primal faces
        let faces = primal_faces
            .iter()
            .map(|&dual_vert_id| {
                dual.loop_structure
                    .faces(dual_vert_id)
                    .collect_vec()
                    .into_iter()
                    .rev()
                    .collect_vec()
            })
            .collect_vec();
        let int_faces = faces
            .iter()
            .map(|face| face.iter().map(|vert| vert_to_int[vert]).collect_vec())
            .collect_vec();

        let (primal, vert_map, _) = Mesh::<POLYCUBE>::from(
            &int_faces,
            &vec![Vector3D::new(0., 0., 0.); primal_vertices.len()],
        )
        .unwrap();

        for vert_id in &primal.vert_ids() {
            let region_id = primal_vertices[vert_map.id(vert_id).unwrap().to_owned()];
            region_to_vertex.insert(region_id, vert_id.to_owned());
        }

        let mut polycube = Self {
            structure: primal,
            region_to_vertex,
        };

        polycube.resize(dual, None);

        polycube
    }

    pub fn resize(&mut self, dual: &Dual, layout: Option<&Layout>) {
        let mut vert_to_coord = HashMap::new();
        for vert_id in self.structure.vert_ids() {
            vert_to_coord.insert(vert_id, [0., 0., 0.]);
        }

        let mut levels = [Vec::new(), Vec::new(), Vec::new()];

        // Fix the positions of the vertices that are in the same level
        for direction in DIRECTIONS {
            for (level, zones) in dual.level_graphs.levels[direction as usize]
                .iter()
                .enumerate()
            {
                let verts_in_level = zones
                    .iter()
                    .flat_map(|&zone_id| {
                        dual.level_graphs.zones[zone_id]
                            .regions
                            .iter()
                            .map(|&region_id| {
                                self.region_to_vertex
                                    .get_by_left(&region_id)
                                    .unwrap()
                                    .to_owned()
                            })
                    })
                    .collect_vec();

                let value = layout.map_or(level as f64, |lay| {
                    let verts_in_mesh = verts_in_level
                        .iter()
                        .map(|vert| {
                            lay.granulated_mesh
                                .position(lay.vert_to_corner.get_by_left(vert).unwrap().to_owned())
                                [direction as usize]
                        })
                        .collect_vec();
                    math::calculate_average_f64(verts_in_mesh.into_iter())
                });

                levels[direction as usize].push((value, verts_in_level.clone()));
            }
        }

        // The levels of every direction are ordered (their order is that of the level graph), so their coordinates must
        // increase strictly; the measured positions (the mean position of their corners) may not. Make them increasing,
        // with a minimum gap, as close as possible to the measured positions (see `increasing_with_gap`).
        let gaps = levels.each_ref().map(|direction_levels| {
            let values = direction_levels
                .iter()
                .map(|(value, _)| *value)
                .collect_vec();
            let increasing = increasing_with_gap(&values);
            increasing
                .iter()
                .tuple_windows()
                .map(|(a, b)| b - a)
                .collect_vec()
        });

        // Integer coordinates: the smallest gap (over all directions) is 1, and every gap is a positive integer (so
        // every edge has a positive length).
        let min_gap = gaps.iter().flatten().copied().fold(f64::INFINITY, f64::min);
        let scale = if min_gap.is_finite() && min_gap > 0. {
            1. / min_gap
        } else {
            1.
        };
        for direction in DIRECTIONS {
            let mut coordinate = 0.;
            for (index, (_, verts_in_level)) in levels[direction as usize].iter().enumerate() {
                if index > 0 {
                    coordinate += (gaps[direction as usize][index - 1] * scale)
                        .round()
                        .max(1.);
                }
                for vert in verts_in_level {
                    vert_to_coord.get_mut(vert).unwrap()[direction as usize] = coordinate;
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
    pub fn get_direction_of_edge(
        &self,
        a: VertKey<POLYCUBE>,
        b: VertKey<POLYCUBE>,
    ) -> (Direction, Sign) {
        to_principal_direction(
            self.structure
                .vector(self.structure.edge_between_verts(a, b).unwrap().0),
        )
    }
}

/// The increasing sequence closest (in the least-squares sense) to `values` in which consecutive values differ by at
/// least a minimum gap (5% of the mean spacing of the values): an isotonic regression (pool adjacent violators) of
/// `values[i] - i * gap`, plus `i * gap`.
fn increasing_with_gap(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    if n < 2 {
        return values.to_vec();
    }
    let span = values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - values.iter().copied().fold(f64::INFINITY, f64::min);
    let gap = if span > 0. {
        0.05 * span / (n - 1) as f64
    } else {
        1.
    };
    // Pool adjacent violators: blocks of (mean, count), merged while the means decrease.
    let mut blocks: Vec<(f64, usize)> = Vec::with_capacity(n);
    for (i, &value) in values.iter().enumerate() {
        blocks.push((value - i as f64 * gap, 1));
        while blocks.len() >= 2 && blocks[blocks.len() - 2].0 > blocks[blocks.len() - 1].0 {
            let (mean2, count2) = blocks.pop().unwrap();
            let (mean1, count1) = blocks.pop().unwrap();
            let count = count1 + count2;
            blocks.push((
                (mean1 * count1 as f64 + mean2 * count2 as f64) / count as f64,
                count,
            ));
        }
    }
    blocks
        .into_iter()
        .flat_map(|(mean, count)| std::iter::repeat_n(mean, count))
        .enumerate()
        .map(|(i, value)| value + i as f64 * gap)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::increasing_with_gap;

    #[test]
    fn levels_become_strictly_increasing() {
        for values in [
            vec![0., 1., 2., 3.],
            vec![0., 2., 1., 3.],
            vec![3., 2., 1., 0.],
            vec![0., 0., 0.],
            vec![0., 5., 4.9, 5.1, 10.],
        ] {
            let result = increasing_with_gap(&values);
            assert_eq!(result.len(), values.len());
            for pair in result.windows(2) {
                assert!(pair[1] > pair[0], "{values:?} -> {result:?}");
            }
        }
        // Already increasing (with enough spacing): unchanged.
        let values = [0., 1., 2., 3.];
        let result = increasing_with_gap(&values);
        for (a, b) in values.iter().zip(&result) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}
