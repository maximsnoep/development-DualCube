use crate::polycube::*;
use bimap::BiHashMap;
use dualcube_dual::prelude::*;
use dualcube_types::prelude::*;
use grapff::Grapff;
use orx_parallel::{IntoParIter, ParIter};
use rand::seq::{IteratorRandom, SliceRandom};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;

// Number of times the granulated mesh is refined (around the corners) when paths or patches cannot be constructed.
const REFINEMENT_ROUNDS: usize = 2;

#[derive(Clone, Debug, PartialEq, Eq, Hash, Copy)]
pub enum NodeType {
    Vertex(VertID),
    Face(FaceID),
}

// The smallest barycentric coordinate of `p` (projected onto the plane of the triangle) with respect to the triangle
// `a`, `b`, `c` (non-negative iff the projection lies inside it). Scale invariant (unlike
// `geom::calculate_barycentric_coordinates`, whose absolute degeneracy threshold misfires on small triangles).
fn min_barycentric(p: Vector3D, a: Vector3D, b: Vector3D, c: Vector3D) -> f64 {
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d00, d01, d11) = (ab.dot(&ab), ab.dot(&ac), ac.dot(&ac));
    let (d20, d21) = (ap.dot(&ab), ap.dot(&ac));
    let denom = d00 * d11 - d01 * d01;
    if denom <= f64::EPSILON * d00 * d11 {
        return f64::NEG_INFINITY;
    }
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    (1. - v - w).min(v).min(w)
}

// A candidate location for a polycube corner.
#[derive(Clone, Copy, Debug)]
enum CornerCandidate {
    Vertex(VertID),
    // A point inside a face of the input mesh. Used for loop regions that contain no mesh vertices
    // (multiple loops can pass through the same faces).
    Point(FaceID, Vector3D),
}

/// The loop regions of the faces of the granulated mesh, when it is built from the input mesh refined along the loops
/// (see `Dual::refined_mesh`). Then, every path is restricted to the two loop regions of the corners it connects, and
/// crosses only the loop segment between them (as in the paper), which also guarantees enough resolution for paths
/// inside narrow regions. Kept up to date when faces and edges are split.
#[derive(Clone, Debug, Default)]
pub struct RegionTracking {
    pub face_region: HashMap<FaceID, LoopRegionID>,
    pub on_loop: HashSet<VertID>,
    /// Vertex of the granulated mesh of every vertex of the input mesh.
    pub vertex_map: HashMap<VertID, VertID>,
    /// Faces of the granulated mesh that every face of the input mesh is split into.
    pub face_pieces: HashMap<FaceID, Vec<FaceID>>,
    /// The face of the input mesh that every face of the granulated mesh lies in.
    pub face_input: HashMap<FaceID, FaceID>,
    /// While mutating (see `Layout::begin_mutation`): the patch of every face of the granulated mesh.
    pub face_patch: HashMap<FaceID, FaceKey<POLYCUBE>>,
}

impl RegionTracking {
    fn split_face(&mut self, face: FaceID, new_faces: &[FaceID]) {
        if let Some(&region) = self.face_region.get(&face) {
            for &f in new_faces {
                self.face_region.insert(f, region);
            }
        }
        if let Some(&patch) = self.face_patch.get(&face) {
            for &f in new_faces {
                self.face_patch.insert(f, patch);
            }
        }
        if let Some(&input) = self.face_input.get(&face) {
            for &f in new_faces {
                self.face_input.insert(f, input);
            }
        }
    }

    // `faces` as returned by `Mesh::split_edge`: the faces of the edge (kept) and their new halves.
    fn split_edge(&mut self, vert: VertID, faces: [FaceID; 4]) {
        let (a, b) = (
            self.face_region.get(&faces[0]).copied(),
            self.face_region.get(&faces[1]).copied(),
        );
        if let Some(a) = a {
            self.face_region.insert(faces[2], a);
        }
        if let Some(b) = b {
            self.face_region.insert(faces[3], b);
        }
        for (old, new) in [(faces[0], faces[2]), (faces[1], faces[3])] {
            if let Some(&input) = self.face_input.get(&old) {
                self.face_input.insert(new, input);
            }
            if let Some(&patch) = self.face_patch.get(&old) {
                self.face_patch.insert(new, patch);
            }
        }
        if a != b {
            self.on_loop.insert(vert);
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Patch {
    // A patch is defined by a set of faces
    pub faces: HashSet<FaceID>,
}

/// Mesh edges whose faces' normals differ by more than this angle are sharp features (creases, not soft bends of a
/// smooth surface).
pub const SHARP_ANGLE: f64 = std::f64::consts::PI / 3.;

/// How the path of a polycube edge is computed: the cost per unit of length of the shortest path between its corners.
/// Which style is best differs per edge; the layout evolution tries them (see `Layout::set_path_style`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PathStyle {
    /// Cheaper along ridges of the mesh whose dihedral angle matches that of the polycube edge (the original method).
    #[default]
    Ridge,
    /// The shortest path.
    Shortest,
    /// Cheaper in the direction of the polycube edge (its axis).
    Axis,
    /// Cheaper along sharp features of the mesh (edges with a large dihedral angle, whatever the polycube edge).
    Feature,
}

impl PathStyle {
    pub const ALL: [Self; 4] = [Self::Ridge, Self::Shortest, Self::Axis, Self::Feature];

    // The smallest cost per unit of length (a lower bound for the A* heuristic).
    const fn min_factor(self) -> f64 {
        match self {
            Self::Ridge => 0.5,
            Self::Shortest => 1.,
            Self::Axis => 0.5,
            Self::Feature => 0.3,
        }
    }
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

    #[serde(skip)]
    pub regions: Option<RegionTracking>,
    /// The style of every path (both orientations of an edge); `PathStyle::default()` if missing.
    #[serde(skip)]
    pub path_styles: HashMap<EdgeKey<POLYCUBE>, PathStyle>,
    /// Per patch: the area and the alignment (normal . label) of each of its triangles (see `compute_quality`), such
    /// that quality terms over the triangles need no geometry, and only changed patches are computed again.
    #[serde(skip)]
    pub patch_triangles: HashMap<FaceKey<POLYCUBE>, Arc<Vec<(f64, f64)>>>,
    /// Per patch: the sum of the area-weighted normals of its triangles, and its area (see `compute_quality`).
    #[serde(skip)]
    pub patch_normals: HashMap<FaceKey<POLYCUBE>, (Vector3D, f64)>,
    /// While mutating (see `begin_mutation`): paths are bounded by the patches of the layout instead of the loop
    /// regions; per path (both orientations), the patches that it may run through if not its two adjacent ones.
    #[serde(skip)]
    pub patch_bounds: Option<HashMap<EdgeKey<POLYCUBE>, Vec<FaceKey<POLYCUBE>>>>,
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
            regions: None,
            path_styles: HashMap::new(),
            patch_triangles: HashMap::new(),
            patch_normals: HashMap::new(),
            patch_bounds: None,
        }
    }

    /// Takes a dual representation, and a primal representation (polycube) and embeds it onto the input mesh.
    pub fn embed(dual_ref: &Dual, polycube_ref: &Polycube) -> Result<Self, LayoutError> {
        let timer = Instant::now();
        let mut layout = Self {
            polycube_ref: polycube_ref.clone(),
            dual_ref: dual_ref.clone(),
            granulated_mesh: (*dual_ref.mesh_ref).clone(),
            vert_to_corner: BiHashMap::new(),
            face_to_patch: HashMap::new(),
            edge_to_path: HashMap::new(),
            alignment_per_triangle: ids::SecMap::new(),
            alignment: None,
            regions: None,
            path_styles: HashMap::new(),
            patch_triangles: HashMap::new(),
            patch_normals: HashMap::new(),
            patch_bounds: None,
        };
        let corners_timer = Instant::now();
        layout.place_all_corners();
        let corners_ms = corners_timer.elapsed();

        let paths_timer = Instant::now();
        layout.place_paths_and_patches()?;
        let paths_ms = paths_timer.elapsed();

        debug!(
            "Layout::embed: loops={} polycube_faces={} corners={:?} paths_patches_quality={:?} total={:?}",
            dual_ref.loops_ref.len(),
            polycube_ref.structure.nr_faces(),
            corners_ms,
            paths_ms,
            timer.elapsed()
        );
        Ok(layout)
    }

    /// Embed the layout with several path insertion orders (in parallel), and keep the embedding of shortest total
    /// path length. Following Born et al. (Layout Embedding via Combinatorial Optimization, 2021), shorter embeddings
    /// avoid paths that swirl around other corners. Instead of their (exhaustive) branch-and-bound search over all
    /// insertion orders, we only try a shortest-first order and a few random orders.
    pub fn embed_best(
        dual_ref: &Dual,
        polycube_ref: &Polycube,
        candidates: usize,
    ) -> Result<Self, LayoutError> {
        Self::embed_best_attempt(dual_ref, polycube_ref, candidates, 0)
    }

    /// See `embed_best`. Only the first attempt (`attempt == 0`) includes the deterministic shortest-first order;
    /// retries use random orders only (otherwise a failing shortest-first order would be repeated).
    pub fn embed_best_attempt(
        dual_ref: &Dual,
        polycube_ref: &Polycube,
        candidates: usize,
        attempt: usize,
    ) -> Result<Self, LayoutError> {
        Self::embed_best_attempt_near(dual_ref, polycube_ref, candidates, attempt, None)
    }

    /// See `embed_best_attempt`; the corners of the regions with a target are placed as close as possible to it (e.g.,
    /// the positions of the corners of a similar layout, see `place_all_corners_near`), and the paths get the given
    /// styles.
    pub fn embed_best_attempt_near(
        dual_ref: &Dual,
        polycube_ref: &Polycube,
        candidates: usize,
        attempt: usize,
        targets: Option<(
            &HashMap<LoopRegionID, Vector3D>,
            &HashMap<EdgeKey<POLYCUBE>, PathStyle>,
        )>,
    ) -> Result<Self, LayoutError> {
        let mut layout = Self::new(dual_ref, polycube_ref);
        if let Some((_, styles)) = targets {
            layout.path_styles = styles.clone();
        }
        layout.place_all_corners_near(None, targets.map(|(corners, _)| corners));
        layout.place_paths_best_attempt(candidates, attempt)?;
        Ok(layout)
    }

    /// Place all paths (for the current corners) with several insertion orders (in parallel), and keep the
    /// embedding of shortest total path length. See `embed_best`.
    pub fn place_paths_best(&mut self, candidates: usize) -> Result<(), LayoutError> {
        self.place_paths_best_attempt(candidates, 0)
    }

    /// See `place_paths_best`; for `attempt > 0`, all candidates use random insertion orders.
    pub fn place_paths_best_attempt(
        &mut self,
        candidates: usize,
        attempt: usize,
    ) -> Result<(), LayoutError> {
        let timer = Instant::now();
        let layout = &*self;

        // Shortest-first: by the distance between the corners of the polycube edge.
        let polycube = &layout.polycube_ref.structure;
        let shortest_first = polycube
            .edge_ids()
            .into_iter()
            .sorted_by_key(|&edge| {
                let Some([u, v]) = polycube
                    .vertices(edge)
                    .map(|corner| layout.vert_to_corner.get_by_left(&corner).copied())
                    .collect_array::<2>()
                else {
                    return OrderedFloat(f64::INFINITY);
                };
                let (Some(u), Some(v)) = (u, v) else {
                    return OrderedFloat(f64::INFINITY);
                };
                OrderedFloat(
                    (layout.granulated_mesh.position(u) - layout.granulated_mesh.position(v))
                        .norm(),
                )
            })
            .collect_vec();

        let results = (0..candidates.max(1))
            .collect_vec()
            .into_par()
            .map(|candidate_index| {
                let mut candidate = layout.clone();
                let order = if attempt == 0 && candidate_index == 0 {
                    shortest_first.clone()
                } else {
                    let mut order = shortest_first.clone();
                    order.shuffle(&mut rand::rng());
                    order
                };
                candidate
                    .place_paths_and_patches_in_order(&order)
                    .map(|()| candidate)
            })
            .collect::<Vec<_>>();

        let lengths = results
            .iter()
            .map(|result| result.as_ref().ok().map(Self::total_path_length))
            .collect_vec();
        let best = results
            .into_iter()
            .filter_map(Result::ok)
            .min_by_key(|candidate| OrderedFloat(candidate.total_path_length()))
            .ok_or(LayoutError::InvalidPath)?;

        debug!(
            "Layout::place_paths_best: candidates={candidates} lengths={lengths:?} best={} elapsed={:?}",
            best.total_path_length(),
            timer.elapsed()
        );
        *self = best;
        Ok(())
    }

    /// Whether all paths and patches are placed.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        let structure = &self.polycube_ref.structure;
        structure
            .edge_ids()
            .iter()
            .all(|edge| self.edge_to_path.contains_key(edge))
            && structure
                .face_ids()
                .iter()
                .all(|face| self.face_to_patch.contains_key(face))
    }

    /// Total length of all paths.
    #[must_use]
    pub fn total_path_length(&self) -> f64 {
        self.polycube_ref
            .structure
            .edge_ids()
            .into_iter()
            .filter(|&edge| edge.raw() < self.polycube_ref.structure.twin(edge).raw())
            .filter_map(|edge| self.edge_to_path.get(&edge))
            .flat_map(|path| path.windows(2))
            .map(|w| {
                (self.granulated_mesh.position(w[0]) - self.granulated_mesh.position(w[1])).norm()
            })
            .sum()
    }

    /// Place all paths and assign all patches. Many loops can pass through the same faces, so corners can be
    /// packed closely together (relative to the resolution of the mesh). If the paths or patches cannot be
    /// constructed, the granulated mesh is refined around the corners, and we try again.
    pub fn place_paths_and_patches(&mut self) -> Result<(), LayoutError> {
        let mut order = self.polycube_ref.structure.edge_ids();
        order.shuffle(&mut rand::rng());
        self.place_paths_and_patches_in_order(&order)
    }

    /// See `place_paths_and_patches`, with a given path insertion order.
    pub fn place_paths_and_patches_in_order(
        &mut self,
        order: &[EdgeKey<POLYCUBE>],
    ) -> Result<(), LayoutError> {
        // Placing paths splits faces of the granulated mesh; failed attempts start over from this mesh.
        let mut base = self.granulated_mesh.clone();
        let mut result = self
            .place_all_paths_in_order(order)
            .and_then(|()| self.assign_all_patches());
        for round in 0..REFINEMENT_ROUNDS {
            if result.is_ok() {
                break;
            }
            debug!(
                "Layout::place_paths_and_patches: refining around corners (round {round}) after {result:?}"
            );
            self.granulated_mesh = base;
            self.refine_around_corners();
            base = self.granulated_mesh.clone();
            result = self
                .place_all_paths_in_order(order)
                .and_then(|()| self.assign_all_patches());
        }
        result
    }

    // Split all edges of the faces around the corners (at their midpoints).
    fn refine_around_corners(&mut self) {
        self.edge_to_path.clear();
        self.face_to_patch.clear();

        let mesh = &self.granulated_mesh;
        let edges = self
            .vert_to_corner
            .right_values()
            .flat_map(|&vert| mesh.faces(vert).collect_vec())
            .flat_map(|face| mesh.edges(face).collect_vec())
            .map(|edge| {
                let twin = mesh.twin(edge);
                if edge.raw() < twin.raw() { edge } else { twin }
            })
            .collect::<HashSet<_>>();

        for edge in edges {
            let midpoint = self.granulated_mesh.position(edge);
            let (vert, faces) = self.granulated_mesh.split_edge(edge);
            self.granulated_mesh.set_position(vert, midpoint);
            if let Some(regions) = &mut self.regions {
                regions.split_edge(vert, faces);
            }
        }
    }

    pub fn place_all_corners(&mut self) {
        self.place_all_corners_with_levels(None);
    }

    /// Place all corners (see `place_all_corners`). If `levels` are given, they are used as the target coordinates
    /// of the zones (e.g., from `fit_zone_levels`), instead of the coordinate that is closest to the candidate
    /// locations of the regions in the zone.
    pub fn place_all_corners_with_levels(&mut self, levels: Option<&HashMap<ZoneID, f64>>) {
        self.place_all_corners_near(levels, None);
    }

    /// See `place_all_corners_with_levels`; the corner of a region with a target is placed at the vertex (or point) of
    /// the region closest to it, any vertex of the region (not only the best candidates).
    pub fn place_all_corners_near(
        &mut self,
        levels: Option<&HashMap<ZoneID, f64>>,
        targets: Option<&HashMap<LoopRegionID, Vector3D>>,
    ) {
        // Clear the mapping
        self.vert_to_corner.clear();
        self.edge_to_path.clear();
        self.face_to_patch.clear();
        // Corners may be inserted as new vertices, so start from the input mesh: refined along the loops if possible
        // (see `RegionTracking`), the input mesh itself otherwise.
        match self.dual_ref.refined_mesh() {
            Some(refined) => {
                self.granulated_mesh = refined.mesh;
                let face_input = refined
                    .face_pieces
                    .iter()
                    .flat_map(|(&input, pieces)| pieces.iter().map(move |&piece| (piece, input)))
                    .collect();
                self.regions = Some(RegionTracking {
                    face_region: refined.face_regions,
                    on_loop: refined.on_loop,
                    vertex_map: refined.vertex_map,
                    face_pieces: refined.face_pieces,
                    face_input,
                    face_patch: HashMap::new(),
                });
            }
            None => {
                warn!(
                    "place_all_corners: the mesh refined along the loops is not available (locator: {}); paths are not restricted to loop regions",
                    self.dual_ref.has_locator()
                );
                self.granulated_mesh = (*self.dual_ref.mesh_ref).clone();
                self.regions = None;
            }
        }

        // Find a candidate location for each region
        // We know for each loop region what are going to be the aligned directions of the patches
        // For each vertex in the region, we count the number of (relevant) directions adjacent to it
        // Then the candidates for this loop region are vertices with the highest count
        // If the loop region is a flat corner, we dont care, and take all vertices
        // If the loop region contains no vertices, the candidates are points inside the faces (parts) of the region

        let mesh = &self.dual_ref.mesh_ref;
        let candidate_position = |candidate: CornerCandidate| match candidate {
            CornerCandidate::Vertex(v) => mesh.position(v),
            CornerCandidate::Point(_, p) => p,
        };

        let mut region_to_labels = HashMap::new();
        let mut region_to_candidates = HashMap::new();
        for region_id in self.dual_ref.loop_structure.face_ids() {
            // Get the relevant directions for this region
            let polycube_vert = self
                .polycube_ref
                .region_to_vertex
                .get_by_left(&region_id)
                .unwrap()
                .to_owned();
            let polycube_faces = self.polycube_ref.structure.faces(polycube_vert);

            // Super strict vertex placement:

            let face_labels = polycube_faces
                .map(|f| to_principal_direction(self.polycube_ref.structure.normal(f)))
                .collect::<HashSet<_>>()
                .into_iter()
                .collect_vec();
            region_to_labels.insert(region_id, face_labels.clone());

            // Get all vertices in the region (or points, if there are no vertices)
            let verts = self.dual_ref.region_to_verts(region_id);
            let candidates = if verts.is_empty() {
                self.dual_ref
                    .region_to_points(region_id)
                    .into_iter()
                    .map(|(face, point)| CornerCandidate::Point(face, point))
                    .collect_vec()
            } else {
                verts.into_iter().map(CornerCandidate::Vertex).collect_vec()
            };

            if face_labels.len() <= 1 {
                region_to_candidates.insert(region_id, candidates);
            } else {
                // Count the number of relevant directions adjacent to each vertex
                let mut candidate_to_count = vec![];

                for &candidate in &candidates {
                    let mut scores = vec![0.; face_labels.len()];
                    let normals = match candidate {
                        CornerCandidate::Vertex(vert) => {
                            mesh.faces(vert).map(|face| mesh.normal(face)).collect_vec()
                        }
                        CornerCandidate::Point(face, _) => vec![mesh.normal(face)],
                    };

                    for normal in normals {
                        let unit = normal.normalize();
                        for i in 0..face_labels.len() {
                            let label = to_vector(face_labels[i].0, face_labels[i].1);
                            // Only angles below 1 radian count: skip the arc cosine of the others.
                            if unit.dot(&label) <= 1f64.cos() {
                                continue;
                            }
                            let angle = label.angle(&normal);
                            if angle < 1. {
                                let score = (2. - angle).powi(2);
                                if score > scores[i] {
                                    scores[i] = score;
                                }
                            }
                        }
                    }

                    candidate_to_count.push((candidate, scores.iter().product::<f64>()));
                }

                // Get the highest score
                let max_score = candidate_to_count
                    .iter()
                    .map(|&(_, count)| count)
                    .max_by(|a, b| a.total_cmp(b))
                    .unwrap();

                // Get all vertices with the highest count
                let candidates = candidate_to_count
                    .into_iter()
                    .filter(|&(_, count)| count >= max_score * 0.9)
                    .map(|(v, _)| v)
                    .collect_vec();

                region_to_candidates.insert(region_id, candidates);
            }
        }

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
                                .map(|&c| candidate_position(c)[zone_type as usize])
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
                .min_by(|a, b| a.total_cmp(b))
                .unwrap()
                .to_owned();
            let max = zone_regions_with_candidates
                .iter()
                .flatten()
                .max_by(|a, b| a.total_cmp(b))
                .unwrap()
                .to_owned();
            let steps = (0..n)
                .map(|i| (max - min).mul_add(f64::from(i) / f64::from(n), min))
                .collect_vec();
            let mut best_step = min;
            let mut best_worst_distance = f64::INFINITY;
            for step in steps {
                let mut worst_distance_for_step = 0.;
                for region_with_candidates in &zone_regions_with_candidates {
                    let best_distance_to_region = region_with_candidates
                        .iter()
                        .map(|&candidate| (step - candidate).abs())
                        .min_by(|a, b| a.total_cmp(b))
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
        if let Some(levels) = levels {
            for (zone_id, &level) in levels {
                if zone_to_candidate.contains_key(zone_id) {
                    zone_to_candidate.insert(*zone_id, level);
                }
            }
        }

        // Find the actual vertex in the subsurface that is closest to the candidate location (by combining the three candidate coordinates of corresponding zones)
        let mut best_candidates = vec![];
        for region_id in self.dual_ref.loop_structure.face_ids() {
            let (target, candidates) = match targets.and_then(|targets| targets.get(&region_id)) {
                Some(&target) => {
                    let verts = self.dual_ref.region_to_verts(region_id);
                    let all = if verts.is_empty() {
                        self.dual_ref
                            .region_to_points(region_id)
                            .into_iter()
                            .map(|(face, point)| CornerCandidate::Point(face, point))
                            .collect_vec()
                    } else {
                        verts.into_iter().map(CornerCandidate::Vertex).collect_vec()
                    };
                    (target, all)
                }
                None => (
                    Vector3D::from(DIRECTIONS.map(|direction| {
                        zone_to_candidate[&self.dual_ref.region_to_zone(region_id, direction)]
                    })),
                    region_to_candidates[&region_id].clone(),
                ),
            };

            let best_candidate = candidates
                .iter()
                .map(|&c| (c, candidate_position(c).metric_distance(&target)))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap()
                .0;

            best_candidates.push((region_id, best_candidate));
        }

        // Points inside faces are inserted into the granulated mesh as new vertices
        let mut face_to_pieces = self
            .regions
            .as_ref()
            .map(|regions| regions.face_pieces.clone())
            .unwrap_or_default();
        for (region_id, best_candidate) in best_candidates {
            let best_vertex = match best_candidate {
                CornerCandidate::Vertex(vert) => self
                    .regions
                    .as_ref()
                    .and_then(|regions| regions.vertex_map.get(&vert).copied())
                    .unwrap_or(vert),
                CornerCandidate::Point(face, point) => {
                    self.insert_point(face, point, &mut face_to_pieces)
                }
            };

            self.vert_to_corner.insert(
                self.polycube_ref
                    .region_to_vertex
                    .get_by_left(&region_id)
                    .unwrap()
                    .to_owned(),
                best_vertex,
            );
        }
    }

    /// Fit the coordinate (level) of every zone to the patches: a polycube face with a label along axis `a` lies in
    /// a single zone of axis `a` (its corners are connected without crossing loops of axis `a`), and the level of the
    /// zone is the area-weighted mean `a`-coordinate of the triangles of all such patches in the zone, i.e., the
    /// least-squares fit of the polycube faces (planes) to the patches. Zones without such patches are omitted.
    /// Requires a complete layout.
    #[must_use]
    pub fn fit_zone_levels(&self) -> HashMap<ZoneID, f64> {
        let polycube = &self.polycube_ref;
        let mesh = &self.granulated_mesh;
        let mut sums: HashMap<ZoneID, (f64, f64)> = HashMap::new();
        for (&face, patch) in &self.face_to_patch {
            let (direction, _) = to_principal_direction(polycube.structure.normal(face));
            let Some(corner) = polycube.structure.vertices(face).next() else {
                continue;
            };
            let Some(&region) = polycube.region_to_vertex.get_by_right(&corner) else {
                continue;
            };
            let zone = self.dual_ref.region_to_zone(region, direction);
            let entry = sums.entry(zone).or_default();
            for &triangle in &patch.faces {
                let area = mesh.size(triangle);
                entry.0 += area * mesh.position(triangle)[direction as usize];
                entry.1 += area;
            }
        }
        sums.into_iter()
            .filter(|(_, (_, area))| *area > 0.)
            .map(|(zone, (weighted, area))| (zone, weighted / area))
            .collect()
    }

    // Insert a point inside a face (of the input mesh) as a new vertex of the granulated mesh.
    // `face_to_pieces` tracks the faces of the granulated mesh that a face of the input mesh has been split into.
    fn insert_point(
        &mut self,
        face: FaceID,
        point: Vector3D,
        face_to_pieces: &mut HashMap<FaceID, Vec<FaceID>>,
    ) -> VertID {
        let pieces = face_to_pieces.entry(face).or_insert_with(|| vec![face]);
        // The piece that contains the point (the one with the largest minimal barycentric coordinate)
        let piece = pieces
            .iter()
            .copied()
            .max_by_key(|&piece| {
                let Some([a, b, c]) = self
                    .granulated_mesh
                    .vertices(piece)
                    .map(|v| self.granulated_mesh.position(v))
                    .collect_array::<3>()
                else {
                    return OrderedFloat(f64::NEG_INFINITY);
                };
                OrderedFloat(min_barycentric(point, a, b, c))
            })
            .unwrap();
        let (vert, new_faces) = self.granulated_mesh.split_face(piece);
        self.granulated_mesh.set_position(vert, point);
        if let Some(regions) = &mut self.regions {
            regions.split_face(piece, &new_faces);
        }
        pieces.extend(new_faces.into_iter().filter(|&f| f != piece));
        vert
    }

    /// Compute the path of a polycube edge between its corners. It cannot pass through `occupied_vertices` or
    /// `extra_occupied` (except its corners), cross the edges between `occupied_edges` (pairs of vertices, both
    /// orientations), or use `occupied_faces`.
    pub fn compute_path(
        &mut self,
        edge_id: EdgeKey<POLYCUBE>,
        occupied_vertices: &HashSet<VertID>,
        extra_occupied: &HashSet<VertID>,
        occupied_edges: &HashSet<(VertID, VertID)>,
        occupied_faces: &HashSet<FaceID>,
    ) -> Result<Vec<VertID>, LayoutError> {
        let polycube = &self.polycube_ref.structure;
        let granulated_mesh = &mut self.granulated_mesh;
        let regions = &mut self.regions;

        let Some([pu, pv]) = polycube.vertices(edge_id).collect_array::<2>() else {
            return Err(LayoutError::UnknownError);
        };
        let (Some(&u), Some(&v)) = (
            self.vert_to_corner.get_by_left(&pu),
            self.vert_to_corner.get_by_left(&pv),
        ) else {
            return Err(LayoutError::UnknownError);
        };
        // With region tracking: the path stays inside the loop regions of its two corners (and thus crosses only the
        // loop segment between them), and never touches a loop otherwise.
        let allowed_regions = regions.as_ref().and_then(|_| {
            Some([
                *self.polycube_ref.region_to_vertex.get_by_right(&pu)?,
                *self.polycube_ref.region_to_vertex.get_by_right(&pv)?,
            ])
        });
        // While mutating: the path stays inside its two patches (or the patches given for it, e.g., all patches around a
        // corner that moves), and may cross loops anywhere (the loops follow the layout afterwards).
        let bounds = self.patch_bounds.as_ref().map(|overrides| {
            overrides.get(&edge_id).cloned().unwrap_or_else(|| {
                vec![
                    polycube.face(edge_id),
                    polycube.face(polycube.twin(edge_id)),
                ]
            })
        });
        let face_allowed = |face: FaceID| match (&bounds, regions.as_ref(), allowed_regions) {
            (Some(patches), Some(tracking), _) => tracking
                .face_patch
                .get(&face)
                .is_some_and(|patch| patches.contains(patch)),
            (None, Some(tracking), Some(allowed)) => tracking
                .face_region
                .get(&face)
                .is_some_and(|r| allowed.contains(r)),
            _ => true,
        };
        let vertex_allowed = |vert: VertID| {
            vert == u
                || vert == v
                || if bounds.is_some() {
                    // A vertex between patches lies on a path (occupied), so one of its faces tells its patch.
                    face_allowed(granulated_mesh.face(granulated_mesh.vrep(vert)))
                } else {
                    regions
                        .as_ref()
                        .is_none_or(|tracking| !tracking.on_loop.contains(&vert))
                }
        };

        // Neighborhood function
        let n_function = |node: NodeType| match node {
            NodeType::Face(f_id) => {
                // Disallow occupied faces
                if occupied_faces.contains(&f_id) {
                    return Vec::new();
                }

                let mut neighbors = Vec::with_capacity(6);
                neighbors.extend(
                    granulated_mesh
                        .vertices(f_id)
                        .filter(|&w| vertex_allowed(w))
                        .map(NodeType::Vertex),
                );

                for n_id in granulated_mesh.neighbors(f_id) {
                    // Moving to a neighboring face crosses their shared edge, which must not be on a path.
                    let crosses_path =
                        granulated_mesh
                            .edge_between_faces(f_id, n_id)
                            .is_some_and(|(edge, _)| {
                                occupied_edges.contains(&(
                                    granulated_mesh.root(edge),
                                    granulated_mesh.toor(edge),
                                ))
                            });
                    if !crosses_path && face_allowed(n_id) {
                        neighbors.push(NodeType::Face(n_id));
                    }
                }

                neighbors
            }
            NodeType::Vertex(v_id) => {
                // Only allowed if the vertex is not occupied
                if (occupied_vertices.contains(&v_id) || extra_occupied.contains(&v_id))
                    && v_id != u
                    && v_id != v
                {
                    return Vec::new();
                }

                let mut neighbors = Vec::with_capacity(12);
                neighbors.extend(
                    granulated_mesh
                        .neighbors(v_id)
                        .filter(|&w| vertex_allowed(w))
                        .map(NodeType::Vertex),
                );
                neighbors.extend(
                    granulated_mesh
                        .faces(v_id)
                        .filter(|&f| face_allowed(f))
                        .map(NodeType::Face),
                );
                neighbors
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
        let style = self.path_styles.get(&edge_id).copied().unwrap_or_default();

        // The direction of the polycube edge (an axis).
        let axis = (polycube.position(pv) - polycube.position(pu)).normalize();
        // The dihedral angle at the mesh edge between two vertices.
        let dihedral = |a: VertID, b: VertID| {
            granulated_mesh
                .edge_between_verts(a, b)
                .map_or(0., |(edge1, edge2)| {
                    granulated_mesh
                        .normal(granulated_mesh.face(edge1))
                        .angle(&granulated_mesh.normal(granulated_mesh.face(edge2)))
                })
        };
        let (unit_left, unit_right) = (normal_on_left.normalize(), normal_on_right.normalize());
        let ridge_function = |a: NodeType, b: NodeType| {
            if normal_on_left == normal_on_right {
                return 1.0;
            }
            // Shortcuts that give the same result as the full test below, without its arc cosines: a step over a face
            // has one normal, so the angle between its normals (0) differs by the angle between the polycube faces from
            // that angle; a step along an edge needs each normal within 45 degrees of its polycube face.
            let cos_45 = std::f64::consts::FRAC_1_SQRT_2;
            match (a, b) {
                (NodeType::Vertex(a), NodeType::Vertex(b)) => {
                    let (edge1, edge2) = granulated_mesh.edge_between_verts(a, b).unwrap();
                    let n1 = granulated_mesh
                        .normal(granulated_mesh.face(edge1))
                        .normalize();
                    let n2 = granulated_mesh
                        .normal(granulated_mesh.face(edge2))
                        .normalize();
                    if n1.dot(&unit_left) <= cos_45 || n2.dot(&unit_right) <= cos_45 {
                        return 1.;
                    }
                }
                _ if angle_between_normals >= std::f64::consts::FRAC_PI_4 => return 1.,
                _ => {}
            }
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

            if difference_between_angles < std::f64::consts::PI / 4.0
                && angle1 + angle2 < std::f64::consts::PI / 4.0
            {
                0.5
            } else {
                1.
            }
        };
        let factor = |a: NodeType, b: NodeType| {
            match style {
                PathStyle::Ridge => ridge_function(a, b),
                PathStyle::Shortest => 1.,
                PathStyle::Axis => {
                    // From 0.5 (along the axis) to 1.5 (orthogonal to it).
                    let direction = nodetype_to_pos(b) - nodetype_to_pos(a);
                    let norm = direction.norm();
                    if norm > 0. {
                        1.5 - direction.dot(&axis).abs() / norm
                    } else {
                        1.
                    }
                }
                PathStyle::Feature => match (a, b) {
                    (NodeType::Vertex(a), NodeType::Vertex(b)) if dihedral(a, b) > SHARP_ANGLE => {
                        0.3
                    }
                    _ => 1.,
                },
            }
        };
        let w_function = |(a, b)| {
            OrderedFloat(factor(a, b) * nodetype_to_pos(a).metric_distance(&nodetype_to_pos(b)))
        };
        let heuristic_scale = if style == PathStyle::Ridge && normal_on_left == normal_on_right {
            1.0
        } else {
            style.min_factor()
        };
        let h_function = |(a, b)| {
            // Edge weights are at least `heuristic_scale * Euclidean distance` (see `PathStyle::min_factor`).
            OrderedFloat(heuristic_scale * nodetype_to_pos(a).metric_distance(&nodetype_to_pos(b)))
        };

        let result = {
            let nn = grapff::fluid::FluidGraph::new(n_function);
            nn.shortest_path_heuristic(
                NodeType::Vertex(u),
                NodeType::Vertex(v),
                w_function,
                h_function,
            )
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
                    if let Some(tracking) = regions.as_mut() {
                        tracking.split_face(f_id, &new_f_ids);
                    }
                    if let Some(last_f_ids) = last_f_ids_maybe {
                        for last_f_id in last_f_ids {
                            for new_f_id in new_f_ids {
                                if let Some((edge_id, _)) =
                                    granulated_mesh.edge_between_faces(last_f_id, new_f_id)
                                {
                                    let midpoint_of_edge = granulated_mesh.position(edge_id);
                                    let (mid_v_id, split_faces) =
                                        granulated_mesh.split_edge(edge_id);
                                    granulated_mesh.set_position(mid_v_id, midpoint_of_edge);
                                    if let Some(tracking) = regions.as_mut() {
                                        tracking.split_edge(mid_v_id, split_faces);
                                    }
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

        let granulated_path = granulated_path
            .into_iter()
            .map(|(v_id, _)| v_id)
            .collect_vec();

        if granulated_path.is_empty() {
            return Err(LayoutError::InvalidPath);
        }

        Ok(granulated_path)
    }

    pub fn place_path(&mut self, edge_id: EdgeKey<POLYCUBE>) -> Result<(), LayoutError> {
        let (occupied_vertices, occupied_edges) = self.compute_occupied();

        // Compute the path
        let path = self.compute_path(
            edge_id,
            &occupied_vertices,
            &HashSet::new(),
            &occupied_edges,
            &HashSet::new(),
        )?;
        let path_reversed = path.iter().cloned().rev().collect_vec();
        // Insert the calculated path
        self.edge_to_path.insert(edge_id, path);
        self.edge_to_path
            .insert(self.polycube_ref.structure.twin(edge_id), path_reversed);

        Ok(())
    }

    /// Place all paths, in the given insertion order (separating paths are postponed).
    pub fn place_all_paths_in_order(
        &mut self,
        order: &[EdgeKey<POLYCUBE>],
    ) -> Result<(), LayoutError> {
        let timer = Instant::now();
        self.edge_to_path.clear();

        let primal = self.polycube_ref.structure.clone();

        let mut occupied_vertices = HashSet::new();
        let mut occupied_edges: HashSet<(VertID, VertID)> = HashSet::new();

        let mut edge_queue = VecDeque::from(order.to_vec());

        // Separating edges are postponed, until a full round through the queue placed no path (then they are placed
        // anyway). Previously, this was detected by meeting the first postponed edge again, which never happened if its
        // twin was placed in the meantime (the edge was then skipped), so the placement got stuck.
        let mut postponed_in_a_row = 0usize;
        // Time spent per step (for profiling).
        let mut time_separating = std::time::Duration::ZERO;
        let mut time_prepare = std::time::Duration::ZERO;
        let mut time_search = std::time::Duration::ZERO;
        let mut pops = 0usize;

        while let Some(edge_id) = edge_queue.pop_front() {
            // If already found, skip
            if self.edge_to_path.contains_key(&edge_id) {
                continue;
            }

            pops += 1;
            let step_timer = Instant::now();
            // With region tracking, every path stays inside the two regions of its corners, so paths cannot enclose
            // corners they should not, and postponing separating paths is not needed (it is also expensive).
            let separating = self.regions.is_none() && {
                // check if edge is separating (in combination with the edges already done)
                let ccs = grapff::fluid::FluidGraph::new(
                    |face_id: FaceKey<POLYCUBE>| -> Vec<FaceKey<POLYCUBE>> {
                        primal
                            .neighbors(face_id)
                            .filter(|&n_id| {
                                let edge_between =
                                    primal.edge_between_faces(face_id, n_id).unwrap().0;
                                edge_between != edge_id
                                    && !self.edge_to_path.contains_key(&edge_between)
                            })
                            .collect::<Vec<FaceKey<POLYCUBE>>>()
                    },
                )
                .connected_components(&primal.face_ids());
                ccs.len() != 1
            };
            time_separating += step_timer.elapsed();
            if separating && postponed_in_a_row <= edge_queue.len() {
                // separating edge -> add to the end of the queue
                postponed_in_a_row += 1;
                edge_queue.push_back(edge_id);
                continue;
            }

            let Some([pu, pv]) = primal.vertices(edge_id).collect_array::<2>() else {
                panic!("Expected edge {edge_id:?} to have exactly two vertices");
            };
            let (Some(&u), Some(&v)) = (
                self.vert_to_corner.get_by_left(&pu),
                self.vert_to_corner.get_by_left(&pv),
            ) else {
                panic!(
                    "Expected corner vertices for primal vertices {:?} and {:?}",
                    pu, pv
                );
            };

            // Find edge in `u_new`
            let edges_done_in_u_new = primal
                .edges(pu)
                .filter(|&e| self.edge_to_path.contains_key(&e) || e == edge_id)
                .collect_vec();

            let mut occupied_faces = HashSet::new();
            // If this is 3 or larger, this means we must make sure the new edge is placed inbetween existing edges, in the correct order
            if edges_done_in_u_new.len() >= 3 {
                // Find the edge that is "above" the new edge
                let edge_id_position = edges_done_in_u_new
                    .iter()
                    .position(|&e| e == edge_id)
                    .unwrap();
                let above = (edge_id_position + 1) % edges_done_in_u_new.len();
                let below =
                    (edge_id_position + edges_done_in_u_new.len() - 1) % edges_done_in_u_new.len();
                // find above edge in the granulated mesh
                let above_edge_id = edges_done_in_u_new[above];
                let above_edge_obj = self
                    .edge_to_path
                    .get(&above_edge_id)
                    .ok_or(LayoutError::InvalidPath)?;
                let above_edge_start = above_edge_obj[0];
                if above_edge_start != u {
                    debug!(
                        "place_all_paths: neighboring path (above) does not start at the corner"
                    );
                    return Err(LayoutError::InvalidPath);
                }
                let above_edge_start_plus_one = above_edge_obj[1];
                let above_edge_real_edge = self
                    .granulated_mesh
                    .edge_between_verts(above_edge_start, above_edge_start_plus_one)
                    .ok_or(LayoutError::InvalidPath)?
                    .0;
                // find below edge in the granulated mesh
                let below_edge_id = edges_done_in_u_new[below];
                let below_edge_obj = self
                    .edge_to_path
                    .get(&below_edge_id)
                    .ok_or(LayoutError::InvalidPath)?;
                let below_edge_start = below_edge_obj[0];
                if below_edge_start != u {
                    debug!(
                        "place_all_paths: neighboring path (below) does not start at the corner"
                    );
                    return Err(LayoutError::InvalidPath);
                }
                let below_edge_start_plus_one = below_edge_obj[1];
                let below_edge_real_edge = self
                    .granulated_mesh
                    .edge_between_verts(below_edge_start, below_edge_start_plus_one)
                    .ok_or(LayoutError::InvalidPath)?
                    .0;
                // so starting from below edge, we insert all faces up until the above edge
                let allowed_edges = self
                    .granulated_mesh
                    .edges(u)
                    .flat_map(|e| [e, self.granulated_mesh.twin(e)])
                    .collect_vec()
                    .into_iter()
                    .cycle()
                    .skip_while(|&e| e != below_edge_real_edge)
                    .skip(1)
                    .take_while(|&e| e != above_edge_real_edge);
                let allowed_faces = allowed_edges
                    .map(|e| self.granulated_mesh.face(e))
                    .collect_vec();
                if allowed_faces.is_empty() {
                    debug!("place_all_paths: no faces between the neighboring paths at a corner");
                    return Err(LayoutError::InvalidPath);
                }
                for face_id in self.granulated_mesh.faces(u) {
                    if !allowed_faces.contains(&face_id) {
                        occupied_faces.insert(face_id);
                    }
                }
            }

            let twin_id = primal.twin(edge_id);
            // Find edge in `v_new`
            let edges_done_in_v_new = primal
                .edges(pv)
                .filter(|&e| self.edge_to_path.contains_key(&e) || e == twin_id)
                .collect_vec();

            // If this is 3 or larger, this means we must make sure the new edge is placed inbetween existing edges, in the correct order
            if edges_done_in_v_new.len() >= 3 {
                // Find the edge that is "above" the new edge
                let edge_id_position = edges_done_in_v_new
                    .iter()
                    .position(|&e| e == twin_id)
                    .unwrap();
                let above = (edge_id_position + 1) % edges_done_in_v_new.len();
                let below =
                    (edge_id_position + edges_done_in_v_new.len() - 1) % edges_done_in_v_new.len();
                // find above edge in the granulated mesh
                let above_edge_id = edges_done_in_v_new[above];
                let above_edge_obj = self
                    .edge_to_path
                    .get(&above_edge_id)
                    .ok_or(LayoutError::InvalidPath)?;
                let above_edge_start = above_edge_obj[0];
                if above_edge_start != v {
                    debug!(
                        "place_all_paths: neighboring path (above, v) does not start at the corner"
                    );
                    return Err(LayoutError::InvalidPath);
                }
                let above_edge_start_plus_one = above_edge_obj[1];
                let above_edge_real_edge = self
                    .granulated_mesh
                    .edge_between_verts(above_edge_start, above_edge_start_plus_one)
                    .ok_or(LayoutError::InvalidPath)?
                    .0;
                // find below edge in the granulated mesh
                let below_edge_id = edges_done_in_v_new[below];
                let below_edge_obj = self
                    .edge_to_path
                    .get(&below_edge_id)
                    .ok_or(LayoutError::InvalidPath)?;
                let below_edge_start = below_edge_obj[0];
                if below_edge_start != v {
                    debug!(
                        "place_all_paths: neighboring path (below, v) does not start at the corner"
                    );
                    return Err(LayoutError::InvalidPath);
                }
                let below_edge_start_plus_one = below_edge_obj[1];
                let below_edge_real_edge = self
                    .granulated_mesh
                    .edge_between_verts(below_edge_start, below_edge_start_plus_one)
                    .ok_or(LayoutError::InvalidPath)?
                    .0;
                // so starting from below edge, we insert all faces up until the above edge
                let allowed_edges = self
                    .granulated_mesh
                    .edges(v)
                    .flat_map(|e| [e, self.granulated_mesh.twin(e)])
                    .collect_vec()
                    .into_iter()
                    .cycle()
                    .skip_while(|&e| e != below_edge_real_edge)
                    .skip(1)
                    .take_while(|&e| e != above_edge_real_edge);
                let allowed_faces = allowed_edges
                    .into_iter()
                    .map(|e| self.granulated_mesh.face(e))
                    .collect_vec();
                if allowed_faces.is_empty() {
                    debug!("place_all_paths: no faces between the neighboring paths at a corner");
                    return Err(LayoutError::InvalidPath);
                }
                for face_id in self.granulated_mesh.faces(v) {
                    if !allowed_faces.contains(&face_id) {
                        occupied_faces.insert(face_id);
                    }
                }
            }

            let step_timer = Instant::now();
            let mut extra_occupied = HashSet::new();
            for &occupied_face in &occupied_faces {
                extra_occupied.extend(self.granulated_mesh.vertices(occupied_face));
            }

            time_prepare += step_timer.elapsed();
            let step_timer = Instant::now();
            let path = self
                .compute_path(
                    edge_id,
                    &occupied_vertices,
                    &extra_occupied,
                    &occupied_edges,
                    &occupied_faces,
                )
                .inspect_err(|_| {
                    debug!(
                        "place_all_paths: no path for a polycube edge (placed {} of {}; constrained sectors: {}; regions: {})",
                        self.edge_to_path.len() / 2,
                        primal.nr_edges() / 2,
                        !occupied_faces.is_empty(),
                        self.regions.is_some()
                    );
                })?;
            time_search += step_timer.elapsed();
            let path_reversed = path.clone().into_iter().rev().collect_vec();

            // Update occupied vertices and edges
            for window in path.windows(2) {
                let (a, b) = (window[0], window[1]);
                occupied_edges.insert((a, b));
                occupied_edges.insert((b, a));
                occupied_vertices.insert(a);
                occupied_vertices.insert(b);
            }

            // Insert the calculated path
            self.edge_to_path.insert(edge_id, path);
            // Also insert for the twin, the calculated path
            self.edge_to_path
                .insert(primal.twin(edge_id), path_reversed);

            postponed_in_a_row = 0;
        }
        debug!(
            "Layout::place_all_paths: polycube_edges={} stored_paths={} pops={pops} separating={time_separating:?} prepare={time_prepare:?} search={time_search:?} elapsed={:?}",
            primal.nr_edges(),
            self.edge_to_path.len(),
            timer.elapsed()
        );
        Ok(())
    }

    pub fn assign_all_patches(&mut self) -> Result<(), LayoutError> {
        let timer = Instant::now();
        // Verify the paths
        let verify_timer = Instant::now();
        self.verify_paths()?;
        let verify_ms = verify_timer.elapsed();

        // The half-edges on the paths (both orientations): the patches are separated there.
        let blocked_timer = Instant::now();
        let mesh = &self.granulated_mesh;
        let mut blocked: HashSet<EdgeID> = HashSet::new();
        for path in self.edge_to_path.values() {
            for verts in path.windows(2) {
                let (a, b) = mesh.edge_between_verts(verts[0], verts[1]).unwrap();
                blocked.insert(a);
                blocked.insert(b);
            }
        }
        let blocked_ms = blocked_timer.elapsed();

        // The patches: the connected components of the faces without crossing the paths (flood fill), and the
        // component of every face.
        let components_timer = Instant::now();
        let mut component: ids::SecMap<FACE, INPUT, usize> = ids::SecMap::new();
        let mut patches: Vec<HashSet<FaceID>> = vec![];
        for start in mesh.face_ids() {
            if component.contains_key(&start) {
                continue;
            }
            let id = patches.len();
            let mut faces = HashSet::from([start]);
            component.insert(&start, id);
            let mut stack = vec![start];
            while let Some(face) = stack.pop() {
                for edge in mesh.edges(face) {
                    if blocked.contains(&edge) {
                        continue;
                    }
                    let neighbor = mesh.face(mesh.twin(edge));
                    if !component.contains_key(&neighbor) {
                        component.insert(&neighbor, id);
                        faces.insert(neighbor);
                        stack.push(neighbor);
                    }
                }
            }
            patches.push(faces);
        }
        let components_ms = components_timer.elapsed();

        if patches.len() != self.polycube_ref.structure.face_ids().len() {
            return Err(LayoutError::InvalidPatches);
        }

        let assign_timer = Instant::now();
        // Every path should be part of exactly TWO patches (on both sides)
        let mut path_to_ccs: HashMap<EdgeKey<POLYCUBE>, [usize; 2]> = HashMap::new();
        for (path_id, path) in &self.edge_to_path {
            // Loop segment should simply have only two connected components (one for each side)
            // We do not check all its edges, but only the first one (since they should all be the same)
            let arbitrary_edge = self
                .granulated_mesh
                .edge_between_verts(path[0], path[1])
                .unwrap()
                .0;
            // Edge has two faces
            let (face1, face2) = (
                self.granulated_mesh.face(arbitrary_edge),
                self.granulated_mesh
                    .face(self.granulated_mesh.twin(arbitrary_edge)),
            );

            let (Some(&cc1), Some(&cc2)) = (component.get(&face1), component.get(&face2)) else {
                return Err(LayoutError::InvalidPatches);
            };
            if cc1 == cc2 {
                return Err(LayoutError::InvalidPatches);
            }
            path_to_ccs.insert(*path_id, (cc1, cc2).into());
        }

        // For every patch, get the connected component that is shared among its paths
        for &face_id in &self.polycube_ref.structure.face_ids() {
            // Select an arbitrary path
            let arbitrary_path = self.polycube_ref.structure.edges(face_id).next().unwrap();
            let [cc1, cc2] = path_to_ccs[&arbitrary_path];

            // Check whether all paths share the same connected component
            let cc1_shared = self
                .polycube_ref
                .structure
                .edges(face_id)
                .all(|path| path_to_ccs[&path].contains(&cc1));
            let cc2_shared = self
                .polycube_ref
                .structure
                .edges(face_id)
                .all(|path| path_to_ccs[&path].contains(&cc2));
            if !(cc1_shared ^ cc2_shared) {
                return Err(LayoutError::InvalidPatches);
            }

            let faces = if cc1_shared {
                patches[cc1].clone()
            } else {
                patches[cc2].clone()
            };
            self.face_to_patch.insert(face_id, Patch { faces });
        }

        let assign_ms = assign_timer.elapsed();

        let quality_timer = Instant::now();
        self.compute_quality();
        let quality_ms = quality_timer.elapsed();

        debug!(
            "Layout::assign_all_patches: paths={} blocked_edges={} patches={} verify={:?} blocked={:?} components={:?} assign={:?} quality={:?} total={:?}",
            self.edge_to_path.len(),
            blocked.len(),
            patches.len(),
            verify_ms,
            blocked_ms,
            components_ms,
            assign_ms,
            quality_ms,
            timer.elapsed()
        );

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

    // The vertices and edges (both orientations) of all paths. Sequential: it runs inside the parallel mutations.
    fn compute_occupied(&self) -> (HashSet<VertID>, HashSet<(VertID, VertID)>) {
        let (mut vertices, mut edges) = (HashSet::new(), HashSet::new());
        for path in self.edge_to_path.values() {
            vertices.extend(path.iter().copied());
            for pair in path.windows(2) {
                edges.insert((pair[0], pair[1]));
                edges.insert((pair[1], pair[0]));
            }
        }
        (vertices, edges)
    }

    fn occupied_vertices(&self) -> HashSet<VertID> {
        self.edge_to_path.values().flatten().copied().collect()
    }

    /// Target positions for a corner (for local corner optimization): the intersection of the fitted planes of its
    /// three zones (see `fit_zone_levels`), and the position aligned with its neighbors along the shared axes (the
    /// average of its coordinate and those of its neighbors that share it).
    #[must_use]
    pub fn corner_targets(
        &self,
        corner: VertKey<POLYCUBE>,
        levels: &HashMap<ZoneID, f64>,
    ) -> Vec<Vector3D> {
        let mut targets = vec![];
        let Some(&vert) = self.vert_to_corner.get_by_left(&corner) else {
            return targets;
        };
        let position = self.granulated_mesh.position(vert);
        if let Some(&region) = self.polycube_ref.region_to_vertex.get_by_right(&corner) {
            let plane = DIRECTIONS.map(|direction| {
                levels
                    .get(&self.dual_ref.region_to_zone(region, direction))
                    .copied()
                    .unwrap_or(position[direction as usize])
            });
            targets.push(Vector3D::from(plane));
        }
        let mut sums = [position.x, position.y, position.z];
        let mut counts = [1.; 3];
        for neighbor in self.polycube_ref.structure.neighbors(corner) {
            let Some(&other) = self.vert_to_corner.get_by_left(&neighbor) else {
                continue;
            };
            let other = self.granulated_mesh.position(other);
            let (direction, _) = self.polycube_ref.get_direction_of_edge(corner, neighbor);
            for axis in DIRECTIONS {
                if axis != direction {
                    sums[axis as usize] += other[axis as usize];
                    counts[axis as usize] += 1.;
                }
            }
        }
        targets.push(Vector3D::new(
            sums[0] / counts[0],
            sums[1] / counts[1],
            sums[2] / counts[2],
        ));
        targets
    }

    /// Candidate positions for a corner (for local corner optimization): free vertices of the granulated mesh inside
    /// the loop region of the corner (not on loops, not on paths), the `per_target` nearest to every target, and
    /// `random` more within the same distance. Requires region tracking.
    #[must_use]
    pub fn corner_candidates(
        &self,
        corner: VertKey<POLYCUBE>,
        targets: &[Vector3D],
        per_target: usize,
        random: usize,
    ) -> Vec<VertID> {
        let Some(tracking) = &self.regions else {
            return vec![];
        };
        let Some(&region) = self.polycube_ref.region_to_vertex.get_by_right(&corner) else {
            return vec![];
        };
        let Some(&current) = self.vert_to_corner.get_by_left(&corner) else {
            return vec![];
        };
        let mesh = &self.granulated_mesh;
        let free = if self.patch_bounds.is_some() {
            // While mutating: anywhere inside the patches around the corner (not on the other paths; its own paths
            // are placed again).
            let structure = &self.polycube_ref.structure;
            let own: HashSet<_> = structure
                .edges(corner)
                .flat_map(|edge| [edge, structure.twin(edge)])
                .collect();
            let occupied: HashSet<VertID> = self
                .edge_to_path
                .iter()
                .filter(|(edge, _)| !own.contains(edge))
                .flat_map(|(_, path)| path.iter().copied())
                .collect();
            structure
                .faces(corner)
                .filter_map(|patch| self.face_to_patch.get(&patch))
                .flat_map(|patch| patch.faces.iter().copied())
                .flat_map(|face| mesh.vertices(face).collect_vec())
                .filter(|v| *v != current && !occupied.contains(v))
                .collect::<HashSet<_>>()
                .into_iter()
                .collect_vec()
        } else {
            let occupied = self.occupied_vertices();
            // The faces of the region of the corner: a flood fill from the corner within the region.
            let in_region = |face: &FaceID| tracking.face_region.get(face) == Some(&region);
            let mut faces: HashSet<FaceID> = mesh.faces(current).filter(in_region).collect();
            let mut stack = faces.iter().copied().collect_vec();
            while let Some(face) = stack.pop() {
                for edge in mesh.edges(face) {
                    let neighbor = mesh.face(mesh.twin(edge));
                    if in_region(&neighbor) && faces.insert(neighbor) {
                        stack.push(neighbor);
                    }
                }
            }
            faces
                .iter()
                .flat_map(|&face| mesh.vertices(face).collect_vec())
                .filter(|v| *v != current && !tracking.on_loop.contains(v) && !occupied.contains(v))
                .collect::<HashSet<_>>()
                .into_iter()
                .collect_vec()
        };
        let mut candidates = HashSet::new();
        let mut radius: f64 = 0.;
        for target in targets {
            let nearest = free
                .iter()
                .copied()
                .sorted_by_key(|&v| OrderedFloat((mesh.position(v) - target).norm()))
                .take(per_target)
                .collect_vec();
            for &v in &nearest {
                radius = radius.max((mesh.position(v) - mesh.position(current)).norm());
                candidates.insert(v);
            }
        }
        let nearby = free
            .iter()
            .copied()
            .filter(|&v| (mesh.position(v) - mesh.position(current)).norm() <= radius.max(1e-12))
            .collect_vec();
        candidates.extend(nearby.into_iter().sample(&mut rand::rng(), random));
        candidates.into_iter().collect()
    }

    pub fn move_corner(
        &mut self,
        vert: VertKey<POLYCUBE>,
        new_vert: VertID,
    ) -> Result<(), LayoutError> {
        let structure = &self.polycube_ref.structure;
        let edges = structure.edges(vert).collect_vec();
        let star = structure.faces(vert).collect_vec();

        // Remove adjacent paths
        for &edge in &edges {
            self.edge_to_path.remove(&edge);
            self.edge_to_path
                .remove(&self.polycube_ref.structure.twin(edge));
        }

        // Move the corner
        self.vert_to_corner.insert(vert, new_vert);

        // Re-compute adjacent paths (while mutating: anywhere inside the patches around the corner)
        if let Some(bounds) = self.patch_bounds.as_mut() {
            for &edge in &edges {
                bounds.insert(edge, star.clone());
                bounds.insert(self.polycube_ref.structure.twin(edge), star.clone());
            }
        }
        let placed = edges.iter().try_for_each(|&edge| self.place_path(edge));
        if let Some(bounds) = self.patch_bounds.as_mut() {
            for &edge in &edges {
                bounds.remove(&edge);
                bounds.remove(&self.polycube_ref.structure.twin(edge));
            }
        }
        placed?;

        self.assign_patches_around(&edges)?;
        self.relabel_patches(&star);
        Ok(())
    }

    /// Start mutating this (complete) layout: from now on, corners and paths are bounded by the patches (a corner may
    /// move anywhere inside the patches around it, a path anywhere inside its two patches), not by the loop regions,
    /// such that the layout stays valid but may leave its loops (which follow it afterwards, see the medial loops).
    /// `false` (and nothing changes) without region tracking or with an incomplete layout.
    pub fn begin_mutation(&mut self) -> bool {
        if !self.is_complete() || self.regions.is_none() {
            return false;
        }
        self.patch_bounds = Some(HashMap::new());
        let all = self.polycube_ref.structure.face_ids();
        self.relabel_patches(&all);
        true
    }

    /// See `begin_mutation`: back to paths bounded by the loop regions.
    pub fn end_mutation(&mut self) {
        self.patch_bounds = None;
        if let Some(tracking) = self.regions.as_mut() {
            tracking.face_patch.clear();
        }
    }

    // While mutating: the labels of the faces of the given patches (after they changed).
    fn relabel_patches(&mut self, patches: &[FaceKey<POLYCUBE>]) {
        if self.patch_bounds.is_none() {
            return;
        }
        let Some(tracking) = self.regions.as_mut() else {
            return;
        };
        for patch in patches {
            if let Some(faces) = self.face_to_patch.get(patch) {
                for &face in &faces.faces {
                    tracking.face_patch.insert(face, *patch);
                }
            }
        }
    }

    /// Remove the path of a polycube edge and compute it again (with all other paths in place, which may have changed
    /// since it was placed), and re-assign the patches.
    pub fn reroute_path(&mut self, edge: EdgeKey<POLYCUBE>) -> Result<(), LayoutError> {
        self.reroute_path_with(edge, None)
    }

    /// Set the style of the path of a polycube edge (both orientations); it is used whenever the path is computed.
    pub fn set_path_style(&mut self, edge: EdgeKey<POLYCUBE>, style: PathStyle) {
        let twin = self.polycube_ref.structure.twin(edge);
        self.path_styles.insert(edge, style);
        self.path_styles.insert(twin, style);
    }

    /// See `reroute_path`; with a new style for the path (if given).
    pub fn reroute_path_with(
        &mut self,
        edge: EdgeKey<POLYCUBE>,
        style: Option<PathStyle>,
    ) -> Result<(), LayoutError> {
        if let Some(style) = style {
            self.set_path_style(edge, style);
        }
        self.edge_to_path.remove(&edge);
        self.edge_to_path
            .remove(&self.polycube_ref.structure.twin(edge));
        self.place_path(edge)?;
        self.assign_patches_around(&[edge])?;
        let structure = &self.polycube_ref.structure;
        let sides = [structure.face(edge), structure.face(structure.twin(edge))];
        self.relabel_patches(&sides);
        Ok(())
    }

    /// Assign the patches again after the paths of the given polycube edges changed (see `assign_all_patches`): only
    /// the patches on both sides of these edges change. They cover the same area as before (bounded by the other
    /// paths), plus the faces created since (by splits for the new paths). Falls back to `assign_all_patches`.
    pub fn assign_patches_around(
        &mut self,
        changed: &[EdgeKey<POLYCUBE>],
    ) -> Result<(), LayoutError> {
        if self.face_to_patch.len() != self.polycube_ref.structure.face_ids().len()
            || self.patch_triangles.is_empty()
        {
            return self.assign_all_patches();
        }
        match self.assign_patches_around_inner(changed) {
            Some(()) => Ok(()),
            None => self.assign_all_patches(),
        }
    }

    fn assign_patches_around_inner(&mut self, changed: &[EdgeKey<POLYCUBE>]) -> Option<()> {
        let structure = &self.polycube_ref.structure;
        let affected = changed
            .iter()
            .flat_map(|&edge| [structure.face(edge), structure.face(structure.twin(edge))])
            .unique()
            .collect_vec();
        let mesh = &self.granulated_mesh;
        // The area of the affected patches: their faces, and the faces in no patch (created since).
        let mut area: HashSet<FaceID> = HashSet::new();
        for patch in &affected {
            area.extend(self.face_to_patch.get(patch)?.faces.iter().copied());
        }
        for face in mesh.face_ids() {
            if !self.alignment_per_triangle.contains_key(&face) {
                area.insert(face);
            }
        }
        // The path edges (both orientations) separate the patches.
        let mut blocked: HashSet<EdgeID> = HashSet::new();
        for path in self.edge_to_path.values() {
            for pair in path.windows(2) {
                let (a, b) = mesh.edge_between_verts(pair[0], pair[1])?;
                if mesh.size(a) == 0. {
                    return None;
                }
                blocked.insert(a);
                blocked.insert(b);
            }
        }
        // The components of the area (flood fill without crossing paths).
        let mut component: HashMap<FaceID, usize> = HashMap::new();
        let mut components: Vec<HashSet<FaceID>> = vec![];
        for &start in &area {
            if component.contains_key(&start) {
                continue;
            }
            let id = components.len();
            let mut faces = HashSet::from([start]);
            component.insert(start, id);
            let mut stack = vec![start];
            while let Some(face) = stack.pop() {
                for edge in mesh.edges(face) {
                    if blocked.contains(&edge) {
                        continue;
                    }
                    let neighbor = mesh.face(mesh.twin(edge));
                    if area.contains(&neighbor) && !component.contains_key(&neighbor) {
                        component.insert(neighbor, id);
                        faces.insert(neighbor);
                        stack.push(neighbor);
                    }
                }
            }
            components.push(faces);
        }
        if components.len() != affected.len() {
            return None;
        }
        // Every affected patch is the component on its side of all its paths.
        let mut assigned = HashSet::new();
        let mut patches = vec![];
        for &patch in &affected {
            let mut candidates: Option<HashSet<usize>> = None;
            for edge in structure.edges(patch) {
                let path = self.edge_to_path.get(&edge)?;
                let (e, twin) = mesh.edge_between_verts(path[0], path[1])?;
                // The side of the patch: the face of the half-edge that runs along the path (as the polycube edge
                // runs along its face), or of its twin; both are allowed (the orientation is checked by uniqueness).
                let sides = [mesh.face(e), mesh.face(twin)]
                    .into_iter()
                    .filter_map(|face| component.get(&face).copied())
                    .collect::<HashSet<_>>();
                candidates = Some(match candidates {
                    None => sides,
                    Some(previous) => previous.intersection(&sides).copied().collect(),
                });
            }
            let candidates = candidates?;
            if candidates.len() != 1 {
                return None;
            }
            let id = candidates.into_iter().next()?;
            if !assigned.insert(id) {
                return None;
            }
            patches.push((patch, id));
        }
        for (patch, id) in patches {
            self.face_to_patch.insert(
                patch,
                Patch {
                    faces: std::mem::take(&mut components[id]),
                },
            );
        }
        self.compute_quality_of(&affected);
        Some(())
    }

    fn compute_quality(&mut self) {
        self.alignment_per_triangle.clear();
        self.patch_triangles.clear();
        self.patch_normals.clear();
        let patches = self.polycube_ref.structure.face_ids();
        self.compute_quality_of(&patches);
    }

    // The alignment of every triangle of the given patches (its normal and area from one cross product), and the
    // area-weighted mean alignment of all patches.
    fn compute_quality_of(&mut self, patches: &[FaceKey<POLYCUBE>]) {
        let polycube = &self.polycube_ref;
        let mesh = &self.granulated_mesh;
        for &patch in patches {
            let target_normal = polycube.structure.normal(patch).normalize();
            let mut triangles = Vec::with_capacity(self.face_to_patch[&patch].faces.len());
            let (mut normal_sum, mut patch_area) = (Vector3D::zeros(), 0.);
            for &triangle in &self.face_to_patch[&patch].faces {
                let Some([a, b, c]) = mesh
                    .vertices(triangle)
                    .map(|v| mesh.position(v))
                    .collect_array::<3>()
                else {
                    continue;
                };
                let cross = (b - a).cross(&(c - a));
                let length = cross.norm();
                let alignment = if length > 0. {
                    cross.dot(&target_normal) / length
                } else {
                    0.
                };
                triangles.push((0.5 * length, alignment));
                normal_sum += 0.5 * cross;
                patch_area += 0.5 * length;
                self.alignment_per_triangle.insert(&triangle, alignment);
            }
            self.patch_triangles.insert(patch, Arc::new(triangles));
            self.patch_normals.insert(patch, (normal_sum, patch_area));
        }
        let (mut weighted, mut total_area) = (0., 0.);
        for triangles in self.patch_triangles.values() {
            for &(area, alignment) in triangles.iter() {
                weighted += alignment * area;
                total_area += area;
            }
        }
        self.alignment = (total_area > 0.).then(|| weighted / total_area);
    }
}
