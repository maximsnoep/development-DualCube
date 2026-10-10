use crate::arrangement::Arrangement;
use crate::loops::*;
use dualcube_types::prelude::*;
use grapff::Grapff;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use slotmap::SlotMap;
use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::Arc,
};
use thiserror::Error;

// A collection of loops forms a loop structure; a graph, where
// the vertices correspond to loop intersections,
// the edges correspond to loop segments,
// and the faces correspond to loop regions.
slotmap::new_key_type! {
    pub struct ZoneID;
}

pub type LoopIntersectionID = VertKey<LOOPSTRUCTURE>;
pub type LoopSegmentID = EdgeKey<LOOPSTRUCTURE>;
pub type LoopRegionID = FaceKey<LOOPSTRUCTURE>;

#[derive(Default, Copy, Clone, Debug, Serialize, Deserialize)]
pub struct LoopSegment {
    // A loop segment has a corresponding loop (id) and an orientation (either following the direction of the loop, or opposite direction of the loop)
    pub loop_id: LoopID,
    // TODO: make a function that reads the orientation by checking the corresponding loop..
    pub orientation: Sign,
}

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct LoopRegion {
    // A loop region has a corresponding surface, in this implementation, the surface is defined by a set of mesh vertices
    pub verts: HashSet<VertID>,
    // Since multiple loops can pass through the same faces, a loop region does not necessarily contain mesh vertices.
    // Therefore, we also store a point for every part of a face (that is crossed by loops) that belongs to the loop region.
    #[serde(default)]
    pub points: Vec<(FaceID, Vector3D)>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoopIntersection {
    // The two loops that intersect
    pub loops: [LoopID; 2],
    // The face of the mesh in which the loops intersect
    pub face: FaceID,
    // The position of the intersection
    pub position: Vector3D,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Zone {
    // A zone is defined by a direction
    pub direction: Direction,
    // All regions that are part of the zone
    pub regions: HashSet<LoopRegionID>,
}

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct LevelGraphs {
    //
    pub zones: SlotMap<ZoneID, Zone>,
    //
    pub graphs: [grapff::fixed::FixedGraph<ZoneID, LoopID>; 3],
    //
    pub region_to_zones: [HashMap<LoopRegionID, ZoneID>; 3],
    //
    pub levels: [Vec<HashSet<ZoneID>>; 3],
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LOOPSTRUCTURE;

pub type LoopStructure = Mesh<LOOPSTRUCTURE>;

// Dual structure (of a polycube)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dual {
    pub mesh_ref: Arc<Mesh<INPUT>>,
    // TODO: make this an actual (arc) reference somehow?
    pub loops_ref: SlotMap<LoopID, Loop>,

    pub loop_structure: LoopStructure,

    pub level_graphs: LevelGraphs,

    #[serde(default)]
    intersections: HashMap<LoopIntersectionID, LoopIntersection>,

    // Locates the loop region of points on the mesh (not serialized; rebuilt with the dual structure).
    #[serde(skip)]
    locator: Option<Arc<RegionLocator>>,
    loop_segments: HashMap<LoopSegmentID, LoopSegment>,
    loop_regions: HashMap<LoopRegionID, LoopRegion>,
}

#[derive(Error, Default, Debug, Clone, Serialize, Deserialize)]
pub enum PropertyViolationError {
    #[default]
    #[error("Unknown error")]
    UnknownError,
    #[error("Face has degree less than three")]
    FaceWithDegreeLessThanThree,
    #[error("Face has degree more than six")]
    FaceWithDegreeMoreThanSix,
    #[error("Invalid face boundary")]
    InvalidFaceBoundary,
    #[error("Cyclic dependency detected")]
    CyclicDependency,
    #[error("Path is empty")]
    PathEmpty,
    #[error("Loop has too few intersections")]
    LoopHasTooFewIntersections,
    #[error("Loop region is not a disk")]
    RegionNotDisk,
}

/// Euler characteristic (V - E + F) of a closed mesh in half-edge representation.
fn euler_characteristic<T: Tag>(mesh: &Mesh<T>) -> i64 {
    mesh.nr_verts() as i64 - (mesh.nr_edges() / 2) as i64 + mesh.nr_faces() as i64
}

/// The input mesh refined along the loops (every face lies in a single loop region; loops run along edges), see
/// `Dual::refined_mesh`.
#[derive(Clone, Debug)]
pub struct RefinedMesh {
    pub mesh: Mesh<INPUT>,
    /// The loop region of every face.
    pub face_regions: HashMap<FaceID, LoopRegionID>,
    /// The vertices that lie on loops.
    pub on_loop: HashSet<VertID>,
    /// The vertex of the refined mesh of every vertex of the input mesh.
    pub vertex_map: HashMap<VertID, VertID>,
    /// The faces of the refined mesh that every face of the input mesh is split into.
    pub face_pieces: HashMap<FaceID, Vec<FaceID>>,
}

// The arrangement of the loops on the mesh, with the loop region of every cell.
#[derive(Debug)]
struct RegionLocator {
    arrangement: Arrangement,
    cell_regions: Vec<Option<LoopRegionID>>,
    // The refined mesh (see `Dual::refined_mesh`), computed once (shared by all clones of the dual structure).
    refined: std::sync::OnceLock<Option<RefinedMesh>>,
}

impl Dual {
    pub fn from(
        mesh_ref: Arc<Mesh<INPUT>>,
        loops_ref: &SlotMap<LoopID, Loop>,
    ) -> Result<Self, PropertyViolationError> {
        let mut dual = Self {
            mesh_ref,
            loops_ref: loops_ref.clone(),
            loop_structure: Mesh::default(),
            level_graphs: LevelGraphs::default(),
            intersections: HashMap::new(),
            locator: None,
            loop_segments: HashMap::new(),
            loop_regions: HashMap::new(),
        };

        // Compute the arrangement of the loops on the mesh (multiple loops may pass through the same faces)
        let arrangement = Arrangement::new(&dual.mesh_ref, &dual.loops_ref)?;

        // Find all intersections and loop regions induced by the loops, and compute the loop structure
        let crossing_to_intersection = dual.assign_loop_structure(&arrangement)?;

        // For each loop region, find its actual subsurface (on the mesh)
        let cell_regions = dual.assign_subsurfaces(&arrangement, &crossing_to_intersection)?;
        dual.locator = Some(Arc::new(RegionLocator {
            arrangement,
            cell_regions,
            refined: std::sync::OnceLock::new(),
        }));

        // Find the zones and construct the level graphs
        dual.assign_level_graphs();

        // Verify properties
        dual.verify_properties()?;

        // Assign levels to the loop structure
        dual.assign_levels();

        Ok(dual)
    }

    #[must_use]
    pub fn intersection(&self, intersection: LoopIntersectionID) -> &LoopIntersection {
        &self.intersections[&intersection]
    }

    #[must_use]
    pub fn segment_to_loop(&self, segment: LoopSegmentID) -> LoopID {
        self.loop_segments.get(&segment).unwrap().loop_id
    }

    #[must_use]
    pub fn segment_to_orientation(&self, segment: LoopSegmentID) -> Sign {
        self.loop_segments.get(&segment).unwrap().orientation
    }

    #[must_use]
    pub fn segment_to_endpoints(
        &self,
        segment: LoopSegmentID,
    ) -> (LoopIntersectionID, LoopIntersectionID) {
        let Some([start, end]) = self.loop_structure.vertices(segment).collect_array::<2>() else {
            panic!("Expecting segment {segment:?} to have exactly two endpoints");
        };
        (start, end)
    }

    #[must_use]
    pub fn region_to_verts(&self, region: LoopRegionID) -> HashSet<VertID> {
        self.loop_regions[&region].verts.clone()
    }

    #[must_use]
    pub fn region_to_points(&self, region: LoopRegionID) -> Vec<(FaceID, Vector3D)> {
        self.loop_regions[&region].points.clone()
    }

    #[must_use]
    pub fn region_to_zone(&self, region: LoopRegionID, direction: Direction) -> ZoneID {
        self.level_graphs.region_to_zones[direction as usize][&region]
    }

    #[must_use]
    pub fn segment_to_direction(&self, segment: LoopSegmentID) -> Direction {
        let loop_id = self.segment_to_loop(segment);
        self.loops_ref[loop_id].direction
    }

    // Returns error if:
    //    1. A loop has less than 4 intersections.
    //    2. A face has more than 6 edges (we know the face degree is at most 6, so we can early stop, and we also want to prevent infinite loops / malformed faces)
    //    3. Invalid intersection.
    // Returns the loop structure vertex (intersection) of every crossing in the arrangement.
    fn assign_loop_structure(
        &mut self,
        arrangement: &Arrangement,
    ) -> Result<Vec<LoopIntersectionID>, PropertyViolationError> {
        // Intersections are crossings of two loops inside a face. Multiple loops may pass through the same face,
        // and loops are ordered along the edges they share. Thus, no three loops intersect at a single point.
        let crossings = &arrangement.crossings;

        // If any loop has too few intersections (less than 4), we return an error
        if arrangement.loop_crossings.values().any(|x| x.len() < 4) {
            return Err(PropertyViolationError::LoopHasTooFewIntersections);
        }

        // For each intersection we find its adjacent intersections (should be 4, by following its associated (two) loops in all (two) directions).
        let mut intersections = Vec::with_capacity(crossings.len());
        for (crossing_id, crossing) in crossings.iter().enumerate() {
            let face = &arrangement.faces[&crossing.face];
            let [a, b] = crossing.chords.map(|c| &face.chords[c]);

            let neighbors = |loop_id: LoopID| {
                let list = &arrangement.loop_crossings[&loop_id];
                let pos = list.iter().position(|&c| c == crossing_id).unwrap();
                (
                    list[(pos + 1) % list.len()],
                    list[(pos + list.len() - 1) % list.len()],
                )
            };
            let (a_next, a_prev) = neighbors(a.loop_id);
            let (b_next, b_prev) = neighbors(b.loop_id);

            // We can order the intersections (counter-clockwise) based on the directions of the loops in the face
            let da = [a.to[0] - a.from[0], a.to[1] - a.from[1]];
            let db = [b.to[0] - b.from[0], b.to[1] - b.from[1]];
            let ordered_adjacent_intersections = if da[0] * db[1] - da[1] * db[0] > 0. {
                [
                    (a.loop_id, a_next, Sign::Positive),
                    (b.loop_id, b_next, Sign::Positive),
                    (a.loop_id, a_prev, Sign::Negative),
                    (b.loop_id, b_prev, Sign::Negative),
                ]
            } else {
                [
                    (a.loop_id, a_next, Sign::Positive),
                    (b.loop_id, b_prev, Sign::Negative),
                    (a.loop_id, a_prev, Sign::Negative),
                    (b.loop_id, b_next, Sign::Positive),
                ]
            };

            if ordered_adjacent_intersections
                .iter()
                .map(|x| x.1)
                .collect::<HashSet<_>>()
                .len()
                != 4
            {
                debug!(
                    "Invalid intersection: adjacent intersections are not unique. {:?}",
                    ordered_adjacent_intersections
                );
                return Err(PropertyViolationError::UnknownError);
            }

            intersections.push(ordered_adjacent_intersections);
        }

        // Create DCEL based on the intersections and loop regions
        // Construct all faces
        let mut edges = intersections
            .iter()
            .enumerate()
            .flat_map(|(this, nexts)| nexts.iter().map(move |next| (this, next.1)))
            .collect_vec();
        let mut remaining = edges.iter().copied().collect::<HashSet<_>>();

        let mut faces = vec![];
        while let Some(start) = edges.pop() {
            if !remaining.remove(&start) {
                continue;
            }
            let mut counter = 0;
            let mut face = vec![start.0, start.1];
            loop {
                let u = face[face.len() - 2];
                let v = face[face.len() - 1];
                // get all intersections that are adjacent to v
                let adj = intersections[v];
                let u_index = adj.iter().position(|&(_, x, _)| x == u).unwrap();
                let w = adj[(u_index + 4 - 1) % 4].1;
                remaining.remove(&(v, w));
                if w == face[0] {
                    break;
                }
                counter += 1;
                if counter > 6 {
                    return Err(PropertyViolationError::FaceWithDegreeMoreThanSix);
                }
                face.push(w);
            }
            faces.push(face);
        }

        let Ok((douconel, vmap, _)) = LoopStructure::from(
            &faces,
            &vec![Vector3D::new(0., 0., 0.); intersections.len()],
        ) else {
            warn!("Failed to create loop structure from faces.");
            return Err(PropertyViolationError::UnknownError);
        };
        assert!(4 * douconel.vert_ids().len() == douconel.edge_ids().len());

        let mut crossing_to_intersection = vec![LoopIntersectionID::default(); crossings.len()];
        for vertex_id in douconel.vert_ids() {
            let crossing_id = vmap.id(&vertex_id).unwrap().to_owned();
            crossing_to_intersection[crossing_id] = vertex_id;
            let crossing = &crossings[crossing_id];
            let face = &arrangement.faces[&crossing.face];
            self.intersections.insert(
                vertex_id,
                LoopIntersection {
                    loops: crossing.chords.map(|c| face.chords[c].loop_id),
                    face: crossing.face,
                    position: Arrangement::to_3d(&self.mesh_ref, &face.sides, crossing.point),
                },
            );
        }
        for edge_id in douconel.edge_ids() {
            let Some([this, next]) = douconel.vertices(edge_id).collect_array::<2>() else {
                panic!("Expecting edge {edge_id:?} to have exactly two vertices");
            };
            let this = vmap.id(&this).unwrap().to_owned();
            let next = vmap.id(&next).unwrap().to_owned();

            let (loop_id, _, orientation) = intersections[this]
                .iter()
                .find(|&(_, x, _)| *x == next)
                .unwrap()
                .to_owned();

            self.loop_segments.insert(
                edge_id,
                LoopSegment {
                    loop_id,
                    orientation,
                },
            );
        }

        self.loop_structure = douconel;

        Ok(crossing_to_intersection)
    }

    // Returns the loop region of every cell of the arrangement.
    fn assign_subsurfaces(
        &mut self,
        arrangement: &Arrangement,
        crossing_to_intersection: &[LoopIntersectionID],
    ) -> Result<Vec<Option<LoopRegionID>>, PropertyViolationError> {
        // The loops cut the faces of the mesh into cells. All connected components of cells are loop regions.
        let (cell_components, component_count, vert_components) =
            arrangement.components(&self.mesh_ref)?;

        // This number should be equal to the number of faces in the loop structure
        if component_count != self.loop_structure.face_ids().len() {
            return Err(PropertyViolationError::UnknownError);
        }

        let intersection_to_crossing: HashMap<LoopIntersectionID, usize> = crossing_to_intersection
            .iter()
            .enumerate()
            .map(|(crossing, &intersection)| (intersection, crossing))
            .collect();

        // Every loop segment should be part of exactly TWO connected components (on both sides)
        let mut segment_to_components: HashMap<LoopSegmentID, [usize; 2]> = HashMap::new();
        for &segment_id in &self.loop_structure.edge_ids() {
            // Loop segment should simply have only two connected components (one for each side)
            // We do not check all its parts, but only the first one (since they should all be the same)
            let (start, end) = self.segment_to_endpoints(segment_id);
            let first = match self.segment_to_orientation(segment_id) {
                Sign::Positive => start,
                Sign::Negative => end,
            };
            let [cell1, cell2] = arrangement
                .cells_after_crossing(
                    self.segment_to_loop(segment_id),
                    intersection_to_crossing[&first],
                )
                .ok_or(PropertyViolationError::UnknownError)?;
            segment_to_components
                .insert(segment_id, [cell_components[cell1], cell_components[cell2]]);
        }

        let mut component_to_verts: HashMap<usize, HashSet<VertID>> = HashMap::new();
        for (vert, component) in vert_components {
            component_to_verts
                .entry(component)
                .or_default()
                .insert(vert);
        }
        let mut component_to_points: HashMap<usize, Vec<(FaceID, Vector3D)>> = HashMap::new();
        for (cell, face, point) in arrangement.cell_centroids(&self.mesh_ref) {
            component_to_points
                .entry(cell_components[cell])
                .or_default()
                .push((face, point));
        }

        // For every loop region, get the connected component that is shared among its loop segments
        let mut component_to_region = HashMap::new();
        for &face_id in &self.loop_structure.face_ids() {
            let mut loop_segments = self.loop_structure.edges(face_id);
            // Select an arbitrary loop segment
            let [component1, component2] =
                segment_to_components[&self.loop_structure.edges(face_id).next().unwrap()];
            // Check whether all loop segments share the same connected component
            let component1_is_shared =
                loop_segments.all(|segment| segment_to_components[&segment].contains(&component1));
            let component = if component1_is_shared {
                component1
            } else {
                component2
            };

            component_to_region.insert(component, face_id);
            self.loop_regions.insert(
                face_id,
                LoopRegion {
                    verts: component_to_verts.remove(&component).unwrap_or_default(),
                    points: component_to_points.remove(&component).unwrap_or_default(),
                },
            );
        }

        Ok(cell_components
            .iter()
            .map(|component| component_to_region.get(component).copied())
            .collect())
    }

    fn assign_level_graphs(&mut self) {
        // A zone is a collection of loop regions that are connected, and are bounded by only one type of loop segment (either X, Y, or Z).
        self.level_graphs.zones = SlotMap::with_key();

        for direction in DIRECTIONS {
            // All loop segments with the given direction are blocked
            let blocked = self
                .loop_structure
                .edge_ids()
                .into_iter()
                .filter(|&segment| self.segment_to_direction(segment) == direction)
                .collect::<HashSet<_>>();

            // Then all connected components of the loop structure that are not blocked are zones
            let zones = grapff::fluid::FluidGraph::new(|loop_region_id: LoopRegionID| {
                let mut neighbors = self.loop_structure.neighbors(loop_region_id).collect_vec();
                neighbors.retain(|&neighbor_id| {
                    !blocked.contains(
                        &self
                            .loop_structure
                            .edge_between_faces(loop_region_id, neighbor_id)
                            .unwrap()
                            .0,
                    )
                });
                neighbors
            })
            .connected_components(&self.loop_structure.face_ids());

            for zone in zones {
                self.level_graphs.zones.insert(Zone {
                    direction,
                    regions: zone,
                });
            }
        }

        for direction in DIRECTIONS {
            // Add all the zones as nodes to the level graph
            let mut edges = vec![];

            let zone_ids = self
                .level_graphs
                .zones
                .iter()
                .filter(|(_, zone)| zone.direction == direction)
                .map(|(zone_id, _)| zone_id)
                .collect_vec();

            // Create a mapping from loop regions to zones
            let region_to_zone = zone_ids
                .iter()
                .flat_map(|&zone_id| {
                    self.level_graphs.zones[zone_id]
                        .regions
                        .iter()
                        .map(move |&region_id| (region_id, zone_id))
                })
                .collect::<HashMap<_, _>>();
            self.level_graphs.region_to_zones[direction as usize] = region_to_zone;

            for &zone_id in &zone_ids {
                // The loop segments (in direction) of this zone
                let segments = self.level_graphs.zones[zone_id]
                    .regions
                    .iter()
                    .flat_map(|&region_id| self.loop_structure.edges(region_id))
                    .filter(|&segment_id| {
                        self.segment_to_direction(segment_id) == direction
                            && self.segment_to_orientation(segment_id) == Sign::Positive
                    });
                // The adjacent loop regions of this zone
                let adjacent_regions = segments.map(|segment_id| {
                    let corresponding_loop = self.segment_to_loop(segment_id);
                    (
                        self.loop_structure
                            .face(self.loop_structure.twin(segment_id)),
                        corresponding_loop,
                    )
                });
                // The adjacent zones of this zone
                let adjacent_zones = adjacent_regions.map(|(region_id, corresponding_loop)| {
                    (
                        self.region_to_zone(region_id, direction),
                        corresponding_loop,
                    )
                });

                for (adjacent_id, loop_id) in adjacent_zones {
                    edges.push((zone_id, adjacent_id, loop_id));
                }
            }

            // Build the level graph once, after collecting every zone's edges
            // (previously this was rebuilt on every iteration of the loop above).
            self.level_graphs.graphs[direction as usize] =
                grapff::fixed::FixedGraph::from(zone_ids, edges);
        }
    }

    pub fn assign_levels(&mut self) {
        for direction in DIRECTIONS {
            let graph = &self.level_graphs.graphs[direction as usize];
            let mut topo_sort = graph.topological_sort().unwrap();

            debug!(
                "assign_levels: {direction} topological order over {} zones: {topo_sort:?}",
                topo_sort.len()
            );
            let mut levels = HashMap::new();
            levels.insert(topo_sort.first().unwrap().to_owned(), 100_000usize);

            for node in topo_sort.clone() {
                debug!("assign_levels: forward pass visiting zone {node:?}");
                if let Some(node_level) = levels.get(&node).cloned() {
                    for neighbor in graph.neighbors_undirected(node) {
                        match (
                            graph.directed_edge_exists(node, neighbor),
                            graph.directed_edge_exists(neighbor, node),
                        ) {
                            (true, false) => {
                                if levels.contains_key(&neighbor) {
                                    let cur = levels[&neighbor];
                                    levels.insert(neighbor, cur.max(node_level + 1));
                                } else {
                                    levels.insert(neighbor, node_level + 1);
                                }
                            }
                            (false, true) => {
                                if levels.contains_key(&neighbor) {
                                    let cur = levels[&neighbor];
                                    levels.insert(neighbor, cur.min(node_level - 1));
                                } else {
                                    levels.insert(neighbor, node_level - 1);
                                }
                            }
                            _ => {
                                panic!();
                            }
                        }
                    }
                }
            }

            topo_sort.reverse();

            for node in topo_sort {
                debug!("assign_levels: backward pass visiting zone {node:?}");
                if let Some(node_level) = levels.get(&node).cloned() {
                    for neighbor in graph.neighbors_undirected(node) {
                        match (
                            graph.directed_edge_exists(node, neighbor),
                            graph.directed_edge_exists(neighbor, node),
                        ) {
                            (true, false) => {
                                if levels.contains_key(&neighbor) {
                                    let cur = levels[&neighbor];
                                    levels.insert(neighbor, cur.max(node_level + 1));
                                } else {
                                    levels.insert(neighbor, node_level + 1);
                                }
                            }
                            (false, true) => {
                                if levels.contains_key(&neighbor) {
                                    let cur = levels[&neighbor];
                                    levels.insert(neighbor, cur.min(node_level - 1));
                                } else {
                                    levels.insert(neighbor, node_level - 1);
                                }
                            }
                            _ => {
                                panic!();
                            }
                        }
                    }
                }
            }

            let mut level_to_zone = HashMap::new();
            let minimum = *levels.values().min().unwrap();
            for (zone_id, level) in levels {
                level_to_zone.insert(zone_id, level - minimum);
            }
            self.level_graphs.levels[direction as usize] =
                vec![HashSet::new(); *level_to_zone.values().max().unwrap() + 1];
            for (zone_id, level) in level_to_zone {
                self.level_graphs.levels[direction as usize][level].insert(zone_id);
            }
        }
    }

    /// Check whether removing the given loop results in a valid polycube loop structure, without rebuilding the
    /// dual structure. Removing a loop merges the regions on both sides of each of its segments, and fuses the two
    /// segments of every other loop that meet at one of its intersections. Following the paper, only the merged
    /// regions have to be checked for conditions 1-4, and only the level graph of the axis of the loop for
    /// condition 5. Runs in time linear in the size of the regions along the loop (plus a search in one level graph).
    pub fn check_removal(&self, loop_id: LoopID) -> Result<(), PropertyViolationError> {
        let structure = &self.loop_structure;
        let on_loop = |segment: LoopSegmentID| self.segment_to_loop(segment) == loop_id;
        let segments = structure
            .edge_ids_iter()
            .filter(|&segment| on_loop(segment))
            .collect_vec();
        if segments.is_empty() {
            return Err(PropertyViolationError::UnknownError);
        }

        // Every other loop must keep enough intersections (as required when constructing the dual structure).
        let mut intersection_count: HashMap<LoopID, usize> = HashMap::new();
        let mut lost_count: HashMap<LoopID, usize> = HashMap::new();
        for intersection in self.intersections.values() {
            for other in intersection.loops {
                *intersection_count.entry(other).or_default() += 1;
            }
            if intersection.loops.contains(&loop_id) {
                for other in intersection.loops {
                    if other != loop_id {
                        *lost_count.entry(other).or_default() += 1;
                    }
                }
            }
        }
        for (other, lost) in lost_count {
            if intersection_count[&other] - lost < 4 {
                return Err(PropertyViolationError::LoopHasTooFewIntersections);
            }
        }

        // Condition 4. The k segments of the loop each merge two regions. The intersections on the loop disappear
        // (V' = V - k), its segments disappear and the two segments of the other loop at each of its intersections
        // are fused (E' = E - 2k). So V' - E' + F' = chi(M) holds if and only if F' = F - k, i.e., if no segment merges
        // two regions that were already merged by other segments (otherwise the merged region is not a disk).
        let faces = structure.face_ids();
        let index: HashMap<LoopRegionID, usize> =
            faces.iter().enumerate().map(|(i, &f)| (f, i)).collect();
        let mut parent = (0..faces.len()).collect_vec();
        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }
        for &segment in &segments {
            let twin = structure.twin(segment);
            if segment.raw() > twin.raw() {
                continue;
            }
            let a = find(&mut parent, index[&structure.face(segment)]);
            let b = find(&mut parent, index[&structure.face(twin)]);
            if a == b {
                return Err(PropertyViolationError::RegionNotDisk);
            }
            parent[a] = b;
        }

        // Conditions 2 and 3, for every merged region: walk along its boundary, skipping the segments of the loop
        // (continuing in the region on the other side), and fuse consecutive segments of the same loop.
        let mut visited = HashSet::new();
        for &segment in &segments {
            let face = structure.face(segment);
            let Some(start) = structure.edges(face).find(|&s| !on_loop(s)) else {
                return Err(PropertyViolationError::FaceWithDegreeLessThanThree);
            };
            if visited.contains(&start) {
                continue;
            }
            let mut boundary = vec![];
            let mut current = start;
            loop {
                visited.insert(current);
                boundary.push(current);
                let mut next = structure.next(current);
                while on_loop(next) {
                    next = structure.next(structure.twin(next));
                }
                if next == start {
                    break;
                }
                if boundary.len() > structure.nr_edges() {
                    return Err(PropertyViolationError::UnknownError);
                }
                current = next;
            }

            // Fuse consecutive segments of the same loop (they meet at an intersection with the removed loop).
            let mut labels = boundary
                .iter()
                .map(|&s| {
                    (
                        self.segment_to_loop(s),
                        self.segment_to_direction(s),
                        self.segment_to_orientation(s),
                    )
                })
                .collect_vec();
            labels.dedup_by_key(|label| label.0);
            if labels.len() > 1 && labels.first().map(|l| l.0) == labels.last().map(|l| l.0) {
                labels.pop();
            }
            if labels.len() < 3 {
                return Err(PropertyViolationError::FaceWithDegreeLessThanThree);
            }
            let mut seen = HashSet::new();
            if labels
                .iter()
                .any(|&(_, direction, orientation)| !seen.insert((direction, orientation)))
            {
                return Err(PropertyViolationError::InvalidFaceBoundary);
            }
        }

        // Condition 5. Only the zones of the axis of the loop change: the zones on both sides of the loop merge, i.e.,
        // the edge(s) of the loop in the level graph are contracted. This creates a cycle if and only if there is
        // another path between the two zones (the graph is acyclic, so only from the negative to the positive side).
        let direction = self.loops_ref[loop_id].direction;
        let graph = &self.level_graphs.graphs[direction as usize];
        let Some(&(from, to, _)) = graph.edges_ref().iter().find(|&&(_, _, l)| l == loop_id) else {
            return Err(PropertyViolationError::UnknownError);
        };
        let mut stack = vec![from];
        let mut reached = HashSet::from([from]);
        while let Some(zone) = stack.pop() {
            for (next, l) in graph.outgoing(zone) {
                if l == loop_id {
                    continue;
                }
                if next == to {
                    return Err(PropertyViolationError::CyclicDependency);
                }
                if reached.insert(next) {
                    stack.push(next);
                }
            }
        }

        Ok(())
    }

    /// The loop region containing the point at parameter `t` (from root to tip) along the half-edge, on the side of
    /// its face. The point must not lie on a loop.
    #[must_use]
    pub fn region_on_edge(&self, edge: EdgeID, t: f64) -> Option<LoopRegionID> {
        let locator = self.locator.as_ref()?;
        let cell = locator
            .arrangement
            .locate_on_edge(&self.mesh_ref, edge, t)?;
        locator.cell_regions.get(cell).copied().flatten()
    }

    /// The input mesh refined along the loops (see `RefinedMesh`), or `None` if not available (no locator, e.g., after
    /// deserialization) or in degenerate configurations.
    #[must_use]
    pub fn refined_mesh(&self) -> Option<RefinedMesh> {
        let locator = self.locator.as_ref()?;
        locator
            .refined
            .get_or_init(|| self.build_refined_mesh())
            .clone()
    }

    /// Drop the cached refined mesh (to save memory, e.g., for the many solutions of an evolution).
    pub fn release_refined_mesh(&mut self) {
        if let Some(locator) = &self.locator
            && locator.refined.get().is_some()
        {
            self.locator = Some(Arc::new(RegionLocator {
                arrangement: locator.arrangement.clone(),
                cell_regions: locator.cell_regions.clone(),
                refined: std::sync::OnceLock::new(),
            }));
        }
    }

    fn build_refined_mesh(&self) -> Option<RefinedMesh> {
        let locator = self.locator.as_ref()?;
        let Some(refinement) = locator.arrangement.refine(&self.mesh_ref) else {
            warn!("refined_mesh: the faces cannot be split into their cells");
            return None;
        };
        let (mesh, vmap, fmap) = Mesh::<INPUT>::from(&refinement.faces, &refinement.positions)
            .inspect_err(|e| warn!("refined_mesh: the refined mesh is invalid: {e:?}"))
            .ok()?;
        let vertex = |i: usize| vmap.key(i).copied();
        let mut face_regions = HashMap::new();
        let mut face_pieces: HashMap<FaceID, Vec<FaceID>> = HashMap::new();
        for (i, (&cell, &input)) in refinement
            .face_cells
            .iter()
            .zip(&refinement.face_input)
            .enumerate()
        {
            let face = *fmap.key(i)?;
            let Some(region) = locator.cell_regions.get(cell).copied().flatten() else {
                warn!("refined_mesh: cell {cell} is not in a loop region");
                return None;
            };
            face_regions.insert(face, region);
            face_pieces.entry(input).or_default().push(face);
        }
        let on_loop = (0..refinement.positions.len())
            .filter(|&i| refinement.on_loop[i])
            .filter_map(vertex)
            .collect();
        let vertex_map = (0..refinement.positions.len())
            .filter_map(|i| Some((refinement.vertex_input[i]?, vertex(i)?)))
            .collect();
        Some(RefinedMesh {
            mesh,
            face_regions,
            on_loop,
            vertex_map,
            face_pieces,
        })
    }

    /// Whether the dual structure can locate points in loop regions (not available after deserialization).
    #[must_use]
    pub fn has_locator(&self) -> bool {
        self.locator.is_some()
    }

    /// Whether a new loop of the given axis may enter the loop region of `entry` through (the segment of) `entry`
    /// and leave it through `exit` (both loop-segment half-edges of the region), such that the two parts of the
    /// region satisfy conditions 2 and 3 (the paper's filtered graph G^V). Loops of the same axis cannot be crossed.
    /// The region lies to the left of its half-edges (counter-clockwise). The loop splits it into a part to the right
    /// of the loop (the boundary from `entry` to `exit`), and a part to the left of the loop (from `exit` to `entry`).
    /// A part to the left of a loop lies on its negative side (it gets label (axis, positive), as for the existing
    /// loops), the part to the right on its positive side.
    #[must_use]
    pub fn valid_exit(&self, entry: LoopSegmentID, exit: LoopSegmentID, axis: Direction) -> bool {
        let structure = &self.loop_structure;
        if entry == exit
            || structure.face(entry) != structure.face(exit)
            || self.segment_to_direction(entry) == axis
            || self.segment_to_direction(exit) == axis
        {
            return false;
        }
        let label = |segment| {
            (
                self.segment_to_direction(segment),
                self.segment_to_orientation(segment),
            )
        };
        // Walk from `from` to `to` (inclusive) along the boundary, and check that no segment has the given label.
        let arc_avoids = |from: LoopSegmentID, to: LoopSegmentID, forbidden: (Direction, Sign)| {
            let mut current = from;
            for _ in 0..=structure.nr_edges() {
                if label(current) == forbidden {
                    return false;
                }
                if current == to {
                    return true;
                }
                current = structure.next(current);
            }
            false
        };
        arc_avoids(entry, exit, (axis, Sign::Negative))
            && arc_avoids(exit, entry, (axis, Sign::Positive))
    }

    // The nodes reachable by a new loop of the given axis after entering a region through `entry`: the twins of the
    // valid exits.
    fn valid_successors(&self, entry: LoopSegmentID, axis: Direction) -> Vec<LoopSegmentID> {
        let structure = &self.loop_structure;
        structure
            .edges(structure.face(entry))
            .filter(|&exit| self.valid_exit(entry, exit, axis))
            .map(|exit| structure.twin(exit))
            .collect()
    }

    // Breadth-first search over the valid transitions from `start` (excluding nodes in `forbidden` regions), returns
    // the predecessor of every reached node.
    fn valid_bfs(
        &self,
        start: LoopSegmentID,
        axis: Direction,
        forbidden: &HashSet<LoopRegionID>,
        stop: impl Fn(LoopSegmentID) -> bool,
        cache: &mut HashMap<LoopSegmentID, Vec<LoopSegmentID>>,
    ) -> (HashMap<LoopSegmentID, LoopSegmentID>, Option<LoopSegmentID>) {
        let structure = &self.loop_structure;
        let mut previous = HashMap::new();
        let mut queue = VecDeque::from([start]);
        let mut seen = HashSet::from([start]);
        while let Some(node) = queue.pop_front() {
            let successors = cache
                .entry(node)
                .or_insert_with(|| self.valid_successors(node, axis))
                .clone();
            for next in successors {
                if stop(next) {
                    previous.insert(next, node);
                    return (previous, Some(next));
                }
                if forbidden.contains(&structure.face(next)) {
                    continue;
                }
                if seen.insert(next) {
                    previous.insert(next, node);
                    queue.push_back(next);
                }
            }
        }
        (previous, None)
    }

    /// Topological structures of valid new loops of the given axis that start in the given region (the paper's
    /// strategy, Section 4.1): for other regions R, the cycle through the start region and R that visits as few
    /// regions as possible. A cycle is given by the loop-segment half-edges through which it enters its regions (in
    /// order; it starts by leaving the start region, and ends by entering it again). Every region is visited at most
    /// once, and every cycle crosses at least four segments. At most `limit` cycles are returned, for regions R chosen
    /// uniformly at random (not the shortest cycles: loops around long features need long cycles).
    #[must_use]
    pub fn valid_cycles(
        &self,
        start: LoopRegionID,
        axis: Direction,
        limit: usize,
    ) -> Vec<Vec<LoopSegmentID>> {
        let structure = &self.loop_structure;
        let mut cycles: Vec<Vec<LoopSegmentID>> = vec![];
        let mut seen = HashSet::new();
        let path_to = |previous: &HashMap<LoopSegmentID, LoopSegmentID>,
                       from: LoopSegmentID,
                       to: LoopSegmentID| {
            let mut path = vec![to];
            let mut current = to;
            while current != from {
                current = previous[&current];
                path.push(current);
            }
            path.reverse();
            path
        };
        // Shortest paths from (having entered the start region through) every entry to all regions.
        let mut cache = HashMap::new();
        let mut candidates = vec![];
        let mut searches = vec![];
        for entry in structure.edges(start).collect_vec() {
            if self.segment_to_direction(entry) == axis {
                continue;
            }
            let (previous, _) =
                self.valid_bfs(entry, axis, &HashSet::from([start]), |_| false, &mut cache);
            let mut targets: HashMap<LoopRegionID, LoopSegmentID> = HashMap::new();
            // BFS order is not kept by the map; pick the node with the shortest path for every region.
            let mut lengths: HashMap<LoopSegmentID, usize> = HashMap::from([(entry, 0)]);
            fn depth(
                node: LoopSegmentID,
                previous: &HashMap<LoopSegmentID, LoopSegmentID>,
                lengths: &mut HashMap<LoopSegmentID, usize>,
            ) -> usize {
                if let Some(&d) = lengths.get(&node) {
                    return d;
                }
                let d = depth(previous[&node], previous, lengths) + 1;
                lengths.insert(node, d);
                d
            }
            for &node in previous.keys() {
                let length = depth(node, &previous, &mut lengths);
                let region = structure.face(node);
                if targets
                    .get(&region)
                    .is_none_or(|&best| lengths[&best] > length)
                {
                    targets.insert(region, node);
                }
            }
            for (&region, &node) in &targets {
                if region != start {
                    candidates.push((searches.len(), node));
                }
            }
            searches.push((entry, previous));
        }
        candidates.shuffle(&mut rand::rng());
        for (search, node) in candidates {
            if cycles.len() >= limit {
                break;
            }
            let (entry, previous) = &searches[search];
            let (entry, region) = (*entry, structure.face(node));
            {
                let there = path_to(previous, entry, node);
                // Back to the start region (entering it through `entry`), avoiding the regions visited so far.
                let forbidden = there
                    .iter()
                    .map(|&n| structure.face(n))
                    .filter(|&r| r != start)
                    .collect::<HashSet<_>>();
                let mut forbidden_back = forbidden.clone();
                forbidden_back.remove(&region);
                forbidden_back.insert(start);
                let (back_previous, found) =
                    self.valid_bfs(node, axis, &forbidden_back, |n| n == entry, &mut cache);
                if found.is_none() {
                    continue;
                }
                let back = path_to(&back_previous, node, entry);
                // The cycle: entry, ..., node, ..., (entry again, omitted).
                let mut cycle = there;
                cycle.extend_from_slice(&back[1..back.len() - 1]);
                let regions = cycle.iter().map(|&n| structure.face(n)).collect_vec();
                if cycle.len() < 4 || regions.iter().unique().count() != regions.len() {
                    continue;
                }
                if seen.insert(cycle.clone()) {
                    cycles.push(cycle);
                }
            }
        }
        cycles
    }

    /// All loops that can be removed (see `check_removal`).
    /// The intersections of the given loop: the mesh face of every intersection, and the loop it crosses there.
    #[must_use]
    pub fn intersections_of(&self, loop_id: LoopID) -> Vec<(FaceID, LoopID)> {
        self.intersections
            .values()
            .filter_map(|intersection| match intersection.loops {
                [a, b] if a == loop_id => Some((intersection.face, b)),
                [a, b] if b == loop_id => Some((intersection.face, a)),
                _ => None,
            })
            .collect()
    }

    #[must_use]
    pub fn removable_loops(&self) -> Vec<LoopID> {
        self.loops_ref
            .keys()
            .filter(|&loop_id| self.check_removal(loop_id).is_ok())
            .collect()
    }

    fn verify_properties(&self) -> Result<(), PropertyViolationError> {
        // Definition 3.2. An oriented loop structure L is a polycube loop structure if:
        // 1. No three loops intersect at a single point.
        // 2. Each loop region is bounded by at least three loop segments.
        // 3. Within each loop region boundary, no two loop segments have the same axis label and side label.
        // 4. Each loop region has the topology of a disk.
        // 5. The level graphs are acyclic.

        // 1. is verified by construction, simply by the way we construct the loop structure.

        for face_id in self.loop_structure.face_ids() {
            // Verify 2.
            if self.loop_structure.edges(face_id).count() < 3 {
                return Err(PropertyViolationError::FaceWithDegreeLessThanThree);
            }

            // Verify 3.
            let mut label_count = [0; 6];
            for edge in self.loop_structure.edges(face_id) {
                let loop_id = self.segment_to_loop(edge);
                let direction = self.loops_ref[loop_id].direction;
                let orientation = self.segment_to_orientation(edge);
                match (direction, orientation) {
                    (Direction::X, Sign::Positive) => label_count[0] += 1,
                    (Direction::X, Sign::Negative) => label_count[1] += 1,
                    (Direction::Y, Sign::Positive) => label_count[2] += 1,
                    (Direction::Y, Sign::Negative) => label_count[3] += 1,
                    (Direction::Z, Sign::Positive) => label_count[4] += 1,
                    (Direction::Z, Sign::Negative) => label_count[5] += 1,
                }
            }
            if label_count.iter().any(|&x| x > 1) {
                return Err(PropertyViolationError::InvalidFaceBoundary);
            }
        }

        // Verify 4.
        // The loops form a graph embedded on the surface (intersections and segments) that cuts the surface into the
        // loop regions. By additivity of the Euler characteristic, chi(M) = V - E + sum of chi(R) over all regions R.
        // Every region (a connected surface with at least one boundary curve) has chi(R) <= 1, with equality if and
        // only if it is a disk. So all regions are disks if and only if V - E + F = chi(M). Every loop is part of the
        // graph, as loops with too few intersections were already rejected.
        if euler_characteristic(&self.loop_structure) != euler_characteristic(&self.mesh_ref) {
            return Err(PropertyViolationError::RegionNotDisk);
        }

        // Verify 5.
        for graph in &self.level_graphs.graphs {
            let topological_sort = grapff::fluid::FluidGraph::new(|z| graph.neighbors(z))
                .topological_sort(&graph.nodes());

            if topological_sort.is_none() {
                return Err(PropertyViolationError::CyclicDependency);
            }
        }

        Ok(())
    }
}
