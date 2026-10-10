use crate::prelude::*;
use itertools::Itertools;
use std::collections::{HashMap, HashSet, VecDeque};

const CAP_POSITION_STITCH_EPS: f64 = 1.0e-8;

/// Result of cutting a mesh and capping both sides with an indexed cap disk.
#[derive(Clone, Debug)]
pub struct CutAndCapOutput<M: Tag> {
    pub mesh_a: Mesh<M>,
    pub mesh_b: Mesh<M>,

    /// Maps input cap-position indices to the corresponding vertex in `mesh_a`.
    /// Boundary cap vertices map to the original cut-loop `VertKey`s; interior cap
    /// vertices map to newly created vertices.
    pub cap_vertices_a: ids::IdMap<VERT, M>,

    /// Maps input cap-position indices to the corresponding vertex in `mesh_b`.
    /// Boundary cap vertices map to the original cut-loop `VertKey`s; interior cap
    /// vertices map to newly created vertices.
    pub cap_vertices_b: ids::IdMap<VERT, M>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ResolvedCapVertex<M: Tag> {
    Original(VertKey<M>),
    Cap(usize),
}

struct ResolvedCap<M: Tag> {
    triangles: Vec<[ResolvedCapVertex<M>; 3]>,
    vertices: HashMap<usize, ResolvedCapVertex<M>>,
}

impl<M: Tag> Mesh<M> {
    #[must_use]
    fn empty() -> Self {
        Self::default()
    }

    // This is a struct that defines an embedded mesh with vertices (with position), edges, and faces (with clockwise ordering).
    pub fn from(
        faces: &[Vec<usize>],
        positions: &[Vector3D],
    ) -> Result<(Self, ids::IdMap<VERT, M>, ids::IdMap<FACE, M>), MeshError<M>> {
        let mut mesh = Self::empty();

        // 1. Create the vertices.
        //      trivial; get all unique input vertices (from the faces), and create a vertex for each of them
        //
        // 2. Create the faces with its (half)edges.
        //      each face has edges defined by a sequence of vertices, example:
        //          face = [v0, v1, v2]
        //          then we create three edges = [(v0, v1), (v1, v2), (v2, v0)]
        //                v0
        //                *
        //               ^ \
        //              /   \ e0
        //          e2 /     \
        //            /       v
        //        v2 * < - - - * v1
        //                e1
        //
        //      Also assign representatives to vertices and faces whenever you make them.
        //
        // 3. Assign twins.
        //      trivial; just assign THE edge that has the same endpoints, but swapped (just requires some bookkeeping)
        //      return error if no such edge exists
        //

        // 0. Validate the input, so malformed files produce an error instead of a panic.
        for (inp_face_id, inp_face_verts) in faces.iter().enumerate() {
            if inp_face_verts.len() < 3 {
                return Err(MeshError::Unknown(format!(
                    "face {inp_face_id} has {} vertices (need at least 3)",
                    inp_face_verts.len()
                )));
            }
            if let Some(&bad) = inp_face_verts.iter().find(|&&v| v >= positions.len()) {
                return Err(MeshError::Unknown(format!(
                    "face {inp_face_id} references vertex {bad}, but only {} vertices exist",
                    positions.len()
                )));
            }
        }

        // 1. Create the vertices.
        // Need mapping between original indices, and new pointers
        let mut vertex_pointers = ids::IdMap::new();
        let mut face_pointers = ids::IdMap::new();

        // Dense lookup table for the hot loop below (the IdMap is a SipHash bimap).
        let mut vert_lookup: Vec<Option<VertKey<M>>> = vec![None; positions.len()];
        for &inp_vert_id in faces.iter().flatten() {
            if vert_lookup[inp_vert_id].is_none() {
                let vert_id = mesh.add_vertex(positions[inp_vert_id]);
                vert_lookup[inp_vert_id] = Some(vert_id);
                vertex_pointers.insert(inp_vert_id, vert_id);
            }
        }

        // 2. Create the faces with its (half)edges.
        // Need mapping between endpoints and edges for later use (assigning twins).
        let mut endpoints_to_edges =
            rustc_hash::FxHashMap::<(VertKey<M>, VertKey<M>), EdgeKey<M>>::with_capacity_and_hasher(
                faces.iter().map(Vec::len).sum(),
                rustc_hash::FxBuildHasher,
            );
        for (inp_face_id, inp_face_verts) in faces.iter().enumerate() {
            let face_id = mesh.add_face();
            face_pointers.insert(inp_face_id, face_id);

            let mut edge_ids = vec![];
            for i in 0..inp_face_verts.len() {
                let inp_start_vertex = inp_face_verts[i];
                let inp_end_vertex = inp_face_verts[(i + 1) % inp_face_verts.len()];
                let (start_vertex, end_vertex) = (
                    vert_lookup[inp_start_vertex].unwrap(),
                    vert_lookup[inp_end_vertex].unwrap(),
                );
                let edge_id = mesh.add_edge();
                if endpoints_to_edges
                    .insert((start_vertex, end_vertex), edge_id)
                    .is_some()
                {
                    return Err(MeshError::DuplicateEdge(start_vertex, end_vertex));
                }
                edge_ids.push(edge_id);
                mesh.face_repr.insert(face_id, edge_id);
                mesh.vert_repr.insert(start_vertex, edge_id);
                mesh.edge_root.insert(edge_id, start_vertex);
                mesh.edge_face.insert(edge_id, face_id);
            }

            // Linking each edge to its next edge in the face
            for edge_index in 0..edge_ids.len() {
                mesh.edge_next.insert(
                    edge_ids[edge_index],
                    edge_ids[(edge_index + 1) % edge_ids.len()],
                );
            }
        }

        // 3. Assign twins.
        for (&(vert_a, vert_b), &edge_id) in &endpoints_to_edges {
            // Retrieve the twin edge
            if let Some(&twin_id) = endpoints_to_edges.get(&(vert_b, vert_a)) {
                // Assign twins
                mesh.edge_twin.insert(edge_id, twin_id);
                mesh.edge_twin.insert(twin_id, edge_id);
            } else {
                return Err(MeshError::NoTwin(vert_a, vert_b));
            }
        }

        // Assert that all elements have their required properties set. These are internal consistency
        // checks of the construction above (O(E) each, and `from` is called in hot paths such as mesh
        // refinement), so they only run in debug builds.
        #[cfg(debug_assertions)]
        {
            mesh.assert_properties();
            mesh.assert_references();
            mesh.assert_invariants();
        }

        // mesh.is_connected();
        if mesh.is_polygonal().is_err() {
            return Err(MeshError::Unknown("Mesh is not polygonal".to_string()));
        }

        Ok((mesh, vertex_pointers, face_pointers))
    }

    /// Cut `self` along `cut_loop` and cap both resulting sides with `cap_triangles`.
    ///
    /// `cut_loop` is a cyclic sequence of consecutive mesh vertices. It may either omit
    /// the repeated final vertex (`[a, b, c]`) or include it (`[a, b, c, a]`).
    ///
    /// `cap_triangles` are oriented triangles expressed with vertex keys from `self`.
    /// Their boundary must be exactly the cut loop; interior cap edges must be paired by
    /// opposite triangle orientations. The cap orientation may match either side of the
    /// cut: this builder flips the cap per side as needed.
    ///
    /// The returned meshes are freshly rebuilt, but every original vertex that appears
    /// in an output mesh keeps the same `VertKey` it had in `self`. Edge and face keys
    /// are new.
    pub fn cut_and_cap(
        &self,
        cut_loop: &[VertKey<M>],
        cap_triangles: &[[VertKey<M>; 3]],
    ) -> Result<(Self, Self), MeshError<M>> {
        let cut_loop = self.normalize_cut_loop(cut_loop)?;
        let cut_halfedges = self.validate_cut_loop(&cut_loop)?;
        self.validate_cap_vertices(cap_triangles)?;

        let cap_boundary = Self::cap_boundary_edges(cap_triangles)?;
        Self::validate_cap_boundary(&cut_loop, &cut_halfedges, &cap_boundary)?;

        let components = self.face_components_without_edges(&cut_halfedges);
        let [component_a, component_b] = components.as_slice() else {
            return Err(MeshError::Unknown(format!(
                "Cut loop must split the mesh into exactly two face components, got {}",
                components.len()
            )));
        };

        let mesh_a = self.build_capped_component(
            component_a,
            cap_triangles,
            &cut_halfedges,
            &cap_boundary,
            cut_loop.len(),
        )?;
        let mesh_b = self.build_capped_component(
            component_b,
            cap_triangles,
            &cut_halfedges,
            &cap_boundary,
            cut_loop.len(),
        )?;

        Ok((mesh_a, mesh_b))
    }

    /// Cut `self` along `cut_loop`, then cap both resulting sides with indexed
    /// cap triangles whose vertices are given by positions.
    ///
    /// This is the most convenient form for DiskScissors/VTK output: pass the VTK
    /// `POINTS` array as `cap_positions`, and the disk triangle index list as
    /// `cap_triangles`. Only position indices referenced by `cap_triangles` are used.
    ///
    /// Boundary cap positions are stitched to original cut-loop vertices when they are
    /// within `1e-8` of a cut-loop vertex position. Non-boundary cap positions become
    /// new vertices in both output meshes. Original vertices keep their original
    /// `VertKey`s; new cap vertices are reported through the returned cap maps.
    pub fn cut_and_cap_from_positions(
        &self,
        cut_loop: &[VertKey<M>],
        cap_positions: &[Vector3D],
        cap_triangles: &[[usize; 3]],
    ) -> Result<CutAndCapOutput<M>, MeshError<M>> {
        self.cut_and_cap_from_positions_with_epsilon(
            cut_loop,
            cap_positions,
            cap_triangles,
            CAP_POSITION_STITCH_EPS,
        )
    }

    /// Same as [`Self::cut_and_cap_from_positions`], with an explicit stitching
    /// distance for matching cap boundary positions to cut-loop vertices.
    pub fn cut_and_cap_from_positions_with_epsilon(
        &self,
        cut_loop: &[VertKey<M>],
        cap_positions: &[Vector3D],
        cap_triangles: &[[usize; 3]],
        stitch_epsilon: f64,
    ) -> Result<CutAndCapOutput<M>, MeshError<M>> {
        if stitch_epsilon < 0.0 {
            return Err(MeshError::Unknown(
                "Stitch epsilon must be non-negative".to_string(),
            ));
        }

        let cut_loop = self.normalize_cut_loop(cut_loop)?;
        let cut_halfedges = self.validate_cut_loop(&cut_loop)?;
        let resolved_cap = self.resolve_cap_triangles_from_positions(
            &cut_loop,
            cap_positions,
            cap_triangles,
            stitch_epsilon,
        )?;

        let cap_boundary = Self::resolved_cap_boundary_edges(&resolved_cap.triangles)?;
        Self::validate_cap_boundary(&cut_loop, &cut_halfedges, &cap_boundary)?;

        let components = self.face_components_without_edges(&cut_halfedges);
        let [component_a, component_b] = components.as_slice() else {
            return Err(MeshError::Unknown(format!(
                "Cut loop must split the mesh into exactly two face components, got {}",
                components.len()
            )));
        };

        let (mesh_a, new_cap_vertices_a) = self.build_capped_component_from_resolved(
            component_a,
            &resolved_cap.triangles,
            cap_positions,
            &cut_halfedges,
            &cap_boundary,
            cut_loop.len(),
        )?;
        let (mesh_b, new_cap_vertices_b) = self.build_capped_component_from_resolved(
            component_b,
            &resolved_cap.triangles,
            cap_positions,
            &cut_halfedges,
            &cap_boundary,
            cut_loop.len(),
        )?;

        Ok(CutAndCapOutput {
            mesh_a,
            mesh_b,
            cap_vertices_a: Self::cap_output_map(&resolved_cap.vertices, &new_cap_vertices_a)?,
            cap_vertices_b: Self::cap_output_map(&resolved_cap.vertices, &new_cap_vertices_b)?,
        })
    }

    fn normalize_cut_loop(&self, cut_loop: &[VertKey<M>]) -> Result<Vec<VertKey<M>>, MeshError<M>> {
        let mut cut_loop = cut_loop.to_vec();
        if cut_loop.len() > 1 && cut_loop.first() == cut_loop.last() {
            cut_loop.pop();
        }

        if cut_loop.len() < 3 {
            return Err(MeshError::Unknown(
                "Cut loop must contain at least three distinct vertices".to_string(),
            ));
        }

        let unique = cut_loop.iter().copied().collect::<HashSet<_>>();
        if unique.len() != cut_loop.len() {
            return Err(MeshError::Unknown(
                "Cut loop must be simple; only the final closing vertex may repeat".to_string(),
            ));
        }

        Ok(cut_loop)
    }

    fn validate_cut_loop(
        &self,
        cut_loop: &[VertKey<M>],
    ) -> Result<HashSet<(VertKey<M>, VertKey<M>)>, MeshError<M>> {
        let mut cut_halfedges = HashSet::new();

        for i in 0..cut_loop.len() {
            let a = cut_loop[i];
            let b = cut_loop[(i + 1) % cut_loop.len()];

            if !self.verts.contains(a) || !self.verts.contains(b) {
                return Err(MeshError::Unknown(format!(
                    "Cut loop edge ({a:?}, {b:?}) references a non-existing vertex"
                )));
            }

            if self.edge_between_verts(a, b).is_none() {
                return Err(MeshError::Unknown(format!(
                    "Cut loop vertices {a:?} and {b:?} are not consecutive"
                )));
            }

            cut_halfedges.insert((a, b));
            cut_halfedges.insert((b, a));
        }

        Ok(cut_halfedges)
    }

    fn validate_cap_vertices(&self, cap_triangles: &[[VertKey<M>; 3]]) -> Result<(), MeshError<M>> {
        if cap_triangles.is_empty() {
            return Err(MeshError::Unknown(
                "Cap must contain at least one triangle".to_string(),
            ));
        }

        for &[a, b, c] in cap_triangles {
            if a == b || b == c || c == a {
                return Err(MeshError::Unknown(format!(
                    "Cap triangle [{a:?}, {b:?}, {c:?}] is degenerate"
                )));
            }

            for vertex in [a, b, c] {
                if !self.verts.contains(vertex) {
                    return Err(MeshError::Unknown(format!(
                        "Cap triangle references non-existing vertex {vertex:?}"
                    )));
                }
            }
        }

        Ok(())
    }

    fn resolve_cap_triangles_from_positions(
        &self,
        cut_loop: &[VertKey<M>],
        cap_positions: &[Vector3D],
        cap_triangles: &[[usize; 3]],
        stitch_epsilon: f64,
    ) -> Result<ResolvedCap<M>, MeshError<M>> {
        if cap_triangles.is_empty() {
            return Err(MeshError::Unknown(
                "Cap must contain at least one triangle".to_string(),
            ));
        }

        let mut vertices = HashMap::<usize, ResolvedCapVertex<M>>::new();
        let mut stitched_cut_vertices = HashMap::<VertKey<M>, usize>::new();

        for cap_index in cap_triangles.iter().flatten().copied().unique() {
            let position = cap_positions.get(cap_index).copied().ok_or_else(|| {
                MeshError::Unknown(format!(
                    "Cap triangle references position index {cap_index}, but only {} positions were provided",
                    cap_positions.len()
                ))
            })?;

            let matching_cut_vertices = cut_loop
                .iter()
                .copied()
                .filter(|&vertex| (self.position(vertex) - position).norm() <= stitch_epsilon)
                .collect_vec();

            let resolved = match matching_cut_vertices.as_slice() {
                [] => ResolvedCapVertex::Cap(cap_index),
                [vertex] => {
                    if let Some(previous_cap_index) =
                        stitched_cut_vertices.insert(*vertex, cap_index)
                    {
                        return Err(MeshError::Unknown(format!(
                            "Cap position indices {previous_cap_index} and {cap_index} both stitch to cut vertex {vertex:?}"
                        )));
                    }
                    ResolvedCapVertex::Original(*vertex)
                }
                _ => {
                    return Err(MeshError::Unknown(format!(
                        "Cap position index {cap_index} is within stitching distance of multiple cut vertices"
                    )));
                }
            };

            vertices.insert(cap_index, resolved);
        }

        let mut triangles = Vec::with_capacity(cap_triangles.len());
        for &[a, b, c] in cap_triangles {
            if a == b || b == c || c == a {
                return Err(MeshError::Unknown(format!(
                    "Cap triangle [{a}, {b}, {c}] is degenerate"
                )));
            }

            let resolved = [vertices[&a], vertices[&b], vertices[&c]];
            if resolved[0] == resolved[1]
                || resolved[1] == resolved[2]
                || resolved[2] == resolved[0]
            {
                return Err(MeshError::Unknown(format!(
                    "Cap triangle [{a}, {b}, {c}] collapses after stitching"
                )));
            }

            triangles.push(resolved);
        }

        Ok(ResolvedCap {
            triangles,
            vertices,
        })
    }

    fn cap_boundary_edges(
        cap_triangles: &[[VertKey<M>; 3]],
    ) -> Result<HashSet<(VertKey<M>, VertKey<M>)>, MeshError<M>> {
        let mut directed_counts = HashMap::<(VertKey<M>, VertKey<M>), usize>::new();

        for &[a, b, c] in cap_triangles {
            for edge in [(a, b), (b, c), (c, a)] {
                *directed_counts.entry(edge).or_insert(0) += 1;
            }
        }

        for (&(a, b), &count) in &directed_counts {
            if count > 1 {
                return Err(MeshError::DuplicateEdge(a, b));
            }
        }

        Ok(directed_counts
            .keys()
            .filter(|&&(a, b)| !directed_counts.contains_key(&(b, a)))
            .copied()
            .collect())
    }

    fn resolved_cap_boundary_edges(
        cap_triangles: &[[ResolvedCapVertex<M>; 3]],
    ) -> Result<HashSet<(VertKey<M>, VertKey<M>)>, MeshError<M>> {
        let mut directed_counts =
            HashMap::<(ResolvedCapVertex<M>, ResolvedCapVertex<M>), usize>::new();

        for &[a, b, c] in cap_triangles {
            for edge in [(a, b), (b, c), (c, a)] {
                *directed_counts.entry(edge).or_insert(0) += 1;
            }
        }

        for (&(a, b), &count) in &directed_counts {
            if count > 1 {
                return match (a, b) {
                    (ResolvedCapVertex::Original(a), ResolvedCapVertex::Original(b)) => {
                        Err(MeshError::DuplicateEdge(a, b))
                    }
                    _ => Err(MeshError::Unknown(format!(
                        "Cap triangle soup contains duplicate directed edge ({a:?}, {b:?})"
                    ))),
                };
            }
        }

        let mut boundary = HashSet::new();
        for &(a, b) in directed_counts.keys() {
            if directed_counts.contains_key(&(b, a)) {
                continue;
            }

            match (a, b) {
                (ResolvedCapVertex::Original(a), ResolvedCapVertex::Original(b)) => {
                    boundary.insert((a, b));
                }
                _ => {
                    return Err(MeshError::Unknown(format!(
                        "Cap boundary edge ({a:?}, {b:?}) does not stitch to the cut loop"
                    )));
                }
            }
        }

        Ok(boundary)
    }

    fn validate_cap_boundary(
        cut_loop: &[VertKey<M>],
        cut_halfedges: &HashSet<(VertKey<M>, VertKey<M>)>,
        cap_boundary: &HashSet<(VertKey<M>, VertKey<M>)>,
    ) -> Result<(), MeshError<M>> {
        if cap_boundary.len() != cut_loop.len() {
            return Err(MeshError::Unknown(format!(
                "Cap boundary must have exactly {} edges, got {}",
                cut_loop.len(),
                cap_boundary.len()
            )));
        }

        for &(a, b) in cap_boundary {
            if !cut_halfedges.contains(&(a, b)) {
                return Err(MeshError::Unknown(format!(
                    "Cap boundary edge ({a:?}, {b:?}) is not part of the cut loop"
                )));
            }
        }

        for i in 0..cut_loop.len() {
            let a = cut_loop[i];
            let b = cut_loop[(i + 1) % cut_loop.len()];
            let forward = cap_boundary.contains(&(a, b));
            let backward = cap_boundary.contains(&(b, a));

            if forward == backward {
                return Err(MeshError::Unknown(format!(
                    "Cap boundary must contain exactly one orientation of cut edge ({a:?}, {b:?})"
                )));
            }
        }

        Ok(())
    }

    fn face_components_without_edges(
        &self,
        blocked_halfedges: &HashSet<(VertKey<M>, VertKey<M>)>,
    ) -> Vec<Vec<FaceKey<M>>> {
        let mut visited = HashSet::<FaceKey<M>>::new();
        let mut components = vec![];

        for start in self.face_ids() {
            if visited.contains(&start) {
                continue;
            }

            let mut queue = VecDeque::from([start]);
            let mut component = vec![];
            visited.insert(start);

            while let Some(face) = queue.pop_front() {
                component.push(face);

                for edge in self.edges(face) {
                    if blocked_halfedges.contains(&(self.root(edge), self.toor(edge))) {
                        continue;
                    }

                    let neighbor = self.face(self.twin(edge));
                    if visited.insert(neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }

            components.push(component);
        }

        components
    }

    fn build_capped_component(
        &self,
        component: &[FaceKey<M>],
        cap_triangles: &[[VertKey<M>; 3]],
        cut_halfedges: &HashSet<(VertKey<M>, VertKey<M>)>,
        cap_boundary: &HashSet<(VertKey<M>, VertKey<M>)>,
        cut_edge_count: usize,
    ) -> Result<Self, MeshError<M>> {
        let side_boundary =
            self.component_cut_boundary(component, cut_halfedges, cut_edge_count)?;
        let reverse_cap = Self::should_reverse_cap(&side_boundary, cap_boundary)?;

        let mut faces = component
            .iter()
            .map(|&face| self.vertices(face).collect_vec())
            .collect_vec();

        for &[a, b, c] in cap_triangles {
            faces.push(if reverse_cap {
                vec![a, c, b]
            } else {
                vec![a, b, c]
            });
        }

        self.build_from_vertex_faces(&faces)
    }

    fn build_capped_component_from_resolved(
        &self,
        component: &[FaceKey<M>],
        cap_triangles: &[[ResolvedCapVertex<M>; 3]],
        cap_positions: &[Vector3D],
        cut_halfedges: &HashSet<(VertKey<M>, VertKey<M>)>,
        cap_boundary: &HashSet<(VertKey<M>, VertKey<M>)>,
        cut_edge_count: usize,
    ) -> Result<(Self, ids::IdMap<VERT, M>), MeshError<M>> {
        let side_boundary =
            self.component_cut_boundary(component, cut_halfedges, cut_edge_count)?;
        let reverse_cap = Self::should_reverse_cap(&side_boundary, cap_boundary)?;

        let mut faces = component
            .iter()
            .map(|&face| {
                self.vertices(face)
                    .map(ResolvedCapVertex::Original)
                    .collect_vec()
            })
            .collect_vec();

        for &[a, b, c] in cap_triangles {
            faces.push(if reverse_cap {
                vec![a, c, b]
            } else {
                vec![a, b, c]
            });
        }

        self.build_from_resolved_vertex_faces(&faces, cap_positions)
    }

    fn component_cut_boundary(
        &self,
        component: &[FaceKey<M>],
        cut_halfedges: &HashSet<(VertKey<M>, VertKey<M>)>,
        cut_edge_count: usize,
    ) -> Result<HashSet<(VertKey<M>, VertKey<M>)>, MeshError<M>> {
        let mut boundary = HashSet::new();

        for &face in component {
            for edge in self.edges(face) {
                let oriented_edge = (self.root(edge), self.toor(edge));
                if cut_halfedges.contains(&oriented_edge) {
                    boundary.insert(oriented_edge);
                }
            }
        }

        if boundary.len() != cut_edge_count {
            return Err(MeshError::Unknown(format!(
                "Each cut side must contain exactly {cut_edge_count} boundary halfedges, got {}",
                boundary.len()
            )));
        }

        Ok(boundary)
    }

    fn should_reverse_cap(
        side_boundary: &HashSet<(VertKey<M>, VertKey<M>)>,
        cap_boundary: &HashSet<(VertKey<M>, VertKey<M>)>,
    ) -> Result<bool, MeshError<M>> {
        let cap_as_is_fits = cap_boundary
            .iter()
            .all(|&(a, b)| side_boundary.contains(&(b, a)));
        let reversed_cap_fits = cap_boundary
            .iter()
            .all(|&(a, b)| side_boundary.contains(&(a, b)));

        match (cap_as_is_fits, reversed_cap_fits) {
            (true, _) => Ok(false),
            (false, true) => Ok(true),
            (false, false) => Err(MeshError::Unknown(
                "Cap boundary orientation does not match this cut side".to_string(),
            )),
        }
    }

    fn build_from_vertex_faces(&self, faces: &[Vec<VertKey<M>>]) -> Result<Self, MeshError<M>> {
        let used_vertices = faces.iter().flatten().copied().collect::<HashSet<_>>();

        let mut mesh = Self::empty();
        mesh.verts = self.verts.clone();
        for vertex in self.vert_ids() {
            if !used_vertices.contains(&vertex) {
                mesh.verts.remove(vertex);
            }
        }

        Self::populate_from_vertex_faces(mesh, faces)
    }

    fn build_from_resolved_vertex_faces(
        &self,
        faces: &[Vec<ResolvedCapVertex<M>>],
        cap_positions: &[Vector3D],
    ) -> Result<(Self, ids::IdMap<VERT, M>), MeshError<M>> {
        let used_original_vertices = faces
            .iter()
            .flatten()
            .filter_map(|&vertex| match vertex {
                ResolvedCapVertex::Original(vertex) => Some(vertex),
                ResolvedCapVertex::Cap(_) => None,
            })
            .collect::<HashSet<_>>();
        let cap_indices = faces
            .iter()
            .flatten()
            .filter_map(|&vertex| match vertex {
                ResolvedCapVertex::Original(_) => None,
                ResolvedCapVertex::Cap(index) => Some(index),
            })
            .unique()
            .sorted()
            .collect_vec();

        let mut mesh = Self::empty();
        mesh.verts = self.verts.clone();

        let mut cap_vertex_map = ids::IdMap::<VERT, M>::new();
        for cap_index in cap_indices {
            let position = cap_positions.get(cap_index).copied().ok_or_else(|| {
                MeshError::Unknown(format!(
                    "Cap vertex index {cap_index} has no position; only {} positions were provided",
                    cap_positions.len()
                ))
            })?;
            let vertex = mesh.add_vertex(position);
            cap_vertex_map.insert(cap_index, vertex);
        }

        // Remove original vertices only after adding cap vertices. This prevents SlotMap
        // from reusing an old original `VertKey` for a new cap vertex.
        for vertex in self.vert_ids() {
            if !used_original_vertices.contains(&vertex) {
                mesh.verts.remove(vertex);
            }
        }

        let key_faces = faces
            .iter()
            .map(|face| {
                face.iter()
                    .map(|&vertex| match vertex {
                        ResolvedCapVertex::Original(vertex) => vertex,
                        ResolvedCapVertex::Cap(index) => *cap_vertex_map.key(index).unwrap(),
                    })
                    .collect_vec()
            })
            .collect_vec();

        Ok((
            Self::populate_from_vertex_faces(mesh, &key_faces)?,
            cap_vertex_map,
        ))
    }

    fn cap_output_map(
        resolved_vertices: &HashMap<usize, ResolvedCapVertex<M>>,
        new_cap_vertices: &ids::IdMap<VERT, M>,
    ) -> Result<ids::IdMap<VERT, M>, MeshError<M>> {
        let mut output = ids::IdMap::<VERT, M>::new();

        for (&cap_index, &resolved_vertex) in resolved_vertices {
            let vertex = match resolved_vertex {
                ResolvedCapVertex::Original(vertex) => vertex,
                ResolvedCapVertex::Cap(index) => *new_cap_vertices.key(index).ok_or_else(|| {
                    MeshError::Unknown(format!("New cap vertex {index} is missing from output map"))
                })?,
            };

            output.insert(cap_index, vertex);
        }

        Ok(output)
    }

    fn populate_from_vertex_faces(
        mut mesh: Self,
        faces: &[Vec<VertKey<M>>],
    ) -> Result<Self, MeshError<M>> {
        let mut endpoints_to_edges = HashMap::<(VertKey<M>, VertKey<M>), EdgeKey<M>>::new();
        for face_verts in faces {
            let face_id = mesh.add_face();
            if face_verts.len() < 3 {
                return Err(MeshError::FaceNotPolygon(face_id));
            }

            let mut edge_ids = vec![];
            for i in 0..face_verts.len() {
                let start_vertex = face_verts[i];
                let end_vertex = face_verts[(i + 1) % face_verts.len()];

                if !mesh.verts.contains(start_vertex) || !mesh.verts.contains(end_vertex) {
                    return Err(MeshError::Unknown(format!(
                        "Face references a non-existing vertex ({start_vertex:?}, {end_vertex:?})"
                    )));
                }

                let edge_id = mesh.add_edge();
                if endpoints_to_edges
                    .insert((start_vertex, end_vertex), edge_id)
                    .is_some()
                {
                    return Err(MeshError::DuplicateEdge(start_vertex, end_vertex));
                }

                edge_ids.push(edge_id);
                mesh.face_repr.insert(face_id, edge_id);
                mesh.vert_repr.insert(start_vertex, edge_id);
                mesh.edge_root.insert(edge_id, start_vertex);
                mesh.edge_face.insert(edge_id, face_id);
            }

            for edge_index in 0..edge_ids.len() {
                mesh.edge_next.insert(
                    edge_ids[edge_index],
                    edge_ids[(edge_index + 1) % edge_ids.len()],
                );
            }
        }

        for (&(vert_a, vert_b), &edge_id) in &endpoints_to_edges {
            if let Some(&twin_id) = endpoints_to_edges.get(&(vert_b, vert_a)) {
                mesh.edge_twin.insert(edge_id, twin_id);
                mesh.edge_twin.insert(twin_id, edge_id);
            } else {
                return Err(MeshError::NoTwin(vert_a, vert_b));
            }
        }

        mesh.assert_properties();
        mesh.assert_references();
        mesh.assert_invariants();

        if mesh.is_polygonal().is_err() {
            return Err(MeshError::Unknown("Mesh is not polygonal".to_string()));
        }

        Ok(mesh)
    }
}
