use std::collections::{HashMap, HashSet};

use mehsh::prelude::{HasEdges, HasNeighbors, HasPosition, HasVertices, Mesh, Vector3D};
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::stable_graph::StableUnGraph;
use petgraph::visit::EdgeRef;

use crate::prelude::{EdgeID, FaceID, VertID, INPUT};
use crate::skeleton::boundary_loop::BoundaryLoop;
use crate::skeleton::orthogonalize::LabeledCurveSkeleton;

use super::{CutPath, CuttingPlan, SurfacePoint};

/// Tracks where a virtual node came from, so that we can map results back to the
/// original mesh and relate duplicated cut nodes to each other.
#[derive(Debug, Clone)]
pub enum VirtualNodeOrigin {
    /// A regular mesh vertex that is not on any cut and not a boundary midpoint.
    MeshVertex(VertID),

    /// A boundary-loop midpoint introduced as a real vertex.
    /// Stores the boundary edge whose midpoint this is and the skeleton edge it
    /// belongs to.
    BoundaryMidpoint { edge: EdgeID, boundary: EdgeIndex },

    /// A vertex on a cut that has been duplicated. Each side of the cut gets its
    /// own copy. `peer` points to the node index of the other copy in the same
    /// `VirtualFlatGeometry` graph.
    ///
    /// `original` is the underlying surface point (could be a mesh vertex or an
    /// on-edge point from the geodesic straightened cut path).
    CutDuplicate {
        original: SurfacePoint,
        /// The index of the other copy (the "peer" on the opposite side of the cut).
        /// Set to `None` during construction and filled in once both copies exist.
        peer: Option<petgraph::graph::NodeIndex>,
        /// Which cut this came from (index into `CuttingPlan::cuts`).
        cut_index: usize,
        /// Which side of the cut: 0 or 1.
        side: u8,
    },
}

/// Weight on virtual-mesh edges. Currently stores the Euclidean length so that
/// Laplacian weights are readily available.
#[derive(Debug, Clone)]
pub struct VirtualEdgeWeight {
    pub length: f64,
}

/// A graph-based mesh representation of a single region after cutting.
///
/// The surface is "opened up" along every cut so that each side of a cut has its
/// own copy of the cut vertices. Boundary midpoints from the boundary loops are
/// introduced as explicit vertices. The region is re-triangulated around the cuts 
/// so that the result is a proper manifold mesh with a single boundary loop.
pub struct VirtualFlatGeometry {
    /// The mesh-like adjacency graph. Each node carries a 3D position plus its
    /// origin information; each edge carries a length.
    pub graph: StableUnGraph<VirtualNode, VirtualEdgeWeight>,

    /// Quick lookup: *original* mesh vertex → virtual node(s).
    /// Interior vertices map to exactly one node; cut vertices map to two.
    pub vert_to_nodes: HashMap<VertID, Vec<petgraph::graph::NodeIndex>>,

    /// The single boundary loop of this virtual mesh, as an ordered sequence of
    /// node indices. After all cuts are applied the topology is a disk, so the
    /// boundary is one simple cycle.
    pub boundary_loop: Vec<petgraph::graph::NodeIndex>,
}

/// Per-node payload in the virtual graph.
#[derive(Debug, Clone)]
pub struct VirtualNode {
    pub position: Vector3D,
    pub origin: VirtualNodeOrigin,
}

impl VirtualFlatGeometry {
    /// Builds the virtual flat geometry for one side of a region, given the
    /// skeleton, the mesh, and the cutting plan that was already computed.
    pub fn build(
        node_idx: NodeIndex,
        skeleton: &LabeledCurveSkeleton,
        mesh: &Mesh<INPUT>,
        cutting_plan: &CuttingPlan,
    ) -> Self {
        let patch_verts = &skeleton[node_idx].skeleton_node.patch_vertices;
        let patch_set: HashSet<VertID> = patch_verts.iter().copied().collect();

        // Collect faces whose *all* vertices are in the patch.
        let patch_faces: Vec<FaceID> = mesh
            .face_ids()
            .into_iter()
            .filter(|&f| mesh.vertices(f).all(|v| patch_set.contains(&v)))
            .collect();

        let mut builder = Builder::new(mesh, &patch_set, &patch_faces);

        // Introduce boundary midpoints.
        for edge_ref in skeleton.edges(node_idx) {
            let boundary = &edge_ref.weight().boundary_loop;
            builder.add_boundary_midpoints(boundary, edge_ref.id());
        }

        // Collect cut surface points and classify edges crossed by cuts.
        builder.register_cuts(&cutting_plan.cuts);

        // Populate the graph: create interior + boundary-midpoint + duplicated
        //    cut nodes, then wire up edges respecting the split.
        builder.build_graph();

        // Trace the disk boundary.
        let boundary_loop = builder.trace_boundary();

        VirtualFlatGeometry {
            graph: builder.graph,
            vert_to_nodes: builder.vert_to_nodes,
            boundary_loop,
        }
    }
}

/// Intermediate state used during construction.
struct Builder<'a> {
    mesh: &'a Mesh<INPUT>,
    patch_set: &'a HashSet<VertID>,
    patch_faces: &'a [FaceID],

    graph: StableUnGraph<VirtualNode, VirtualEdgeWeight>,

    // Maps from mesh VertID → virtual node(s).
    vert_to_nodes: HashMap<VertID, Vec<petgraph::graph::NodeIndex>>,

    // Boundary midpoint nodes: boundary EdgeID → virtual node index.
    midpoint_nodes: HashMap<EdgeID, petgraph::graph::NodeIndex>,

    // Cut information: for each cut, an ordered list of virtual surface points
    // along the cut (endpoints are boundary midpoints, interior points are
    // on-edge or on-vertex).
    cuts: Vec<CutInfo>,

    // Which mesh vertices are on *any* cut. Maps VertID → list of cut indices.
    verts_on_cuts: HashMap<VertID, Vec<usize>>,

    // For surface points that are on mesh edges (SurfacePoint::OnEdge), the
    // mesh edge → cut index mapping, so we know which face-edges are crossed.
    edges_on_cuts: HashMap<EdgeID, usize>,

    // After duplication: for each (VertID that's on a cut, side), the virtual node.
    // Side 0 and side 1 are the two sides of the cut.
    cut_node_sides: HashMap<(VertID, u8), petgraph::graph::NodeIndex>,

    // Same but for on-edge surface points on cuts (keyed by EdgeID, side).
    cut_edge_point_sides: HashMap<(EdgeID, u8), petgraph::graph::NodeIndex>,
}

/// Internal record for a single cut.
struct CutInfo {
    /// Ordered surface points along the cut, from start boundary to end boundary.
    points: Vec<SurfacePoint>,
    /// Index of the start boundary midpoint edge.
    start_midpoint_edge: EdgeID,
    /// Index of the end boundary midpoint edge.
    end_midpoint_edge: EdgeID,
}

type NIdx = petgraph::graph::NodeIndex;

impl<'a> Builder<'a> {
    fn new(
        mesh: &'a Mesh<INPUT>,
        patch_set: &'a HashSet<VertID>,
        patch_faces: &'a [FaceID],
    ) -> Self {
        Builder {
            mesh,
            patch_set,
            patch_faces,
            graph: StableUnGraph::default(),
            vert_to_nodes: HashMap::new(),
            midpoint_nodes: HashMap::new(),
            cuts: Vec::new(),
            verts_on_cuts: HashMap::new(),
            edges_on_cuts: HashMap::new(),
            cut_node_sides: HashMap::new(),
            cut_edge_point_sides: HashMap::new(),
        }
    }

    fn add_boundary_midpoints(&mut self, boundary: &BoundaryLoop, edge_idx: EdgeIndex) {
        for &edge in &boundary.edge_midpoints {
            let pos = self.mesh.position(self.mesh.root(edge)) + self.mesh.vector(edge) * 0.5;

            let node = self.graph.add_node(VirtualNode {
                position: pos,
                origin: VirtualNodeOrigin::BoundaryMidpoint {
                    edge,
                    boundary: edge_idx,
                },
            });
            self.midpoint_nodes.insert(edge, node);
        }
    }

    fn register_cuts(&mut self, cuts: &[CutPath]) {
        for (cut_index, cut) in cuts.iter().enumerate() {
            let points = cut.path.points.clone();

            // Identify the boundary midpoint edges at the endpoints.
            let start_edge = match points.first() {
                Some(SurfacePoint::OnEdge { edge, .. }) => *edge,
                _ => panic!("Cut path must start at a boundary midpoint (OnEdge)"),
            };
            let end_edge = match points.last() {
                Some(SurfacePoint::OnEdge { edge, .. }) => *edge,
                _ => panic!("Cut path must end at a boundary midpoint (OnEdge)"),
            };

            // Register mesh vertices and mesh edges that lie on this cut.
            for pt in &points[1..points.len() - 1] {
                match *pt {
                    SurfacePoint::OnVertex { vertex } => {
                        self.verts_on_cuts
                            .entry(vertex)
                            .or_default()
                            .push(cut_index);
                    }
                    SurfacePoint::OnEdge { edge, .. } => {
                        self.edges_on_cuts.insert(edge, cut_index);
                        // Also register the twin so we can detect cut crossings
                        // from either half-edge direction.
                        self.edges_on_cuts.insert(self.mesh.twin(edge), cut_index);
                    }
                }
            }

            self.cuts.push(CutInfo {
                points,
                start_midpoint_edge: start_edge,
                end_midpoint_edge: end_edge,
            });
        }
    }

    fn build_graph(&mut self) {
        // Create virtual nodes for every patch vertex.
        for &v in self.patch_set.iter() {
            if self.verts_on_cuts.contains_key(&v) {
                // Duplicate: create two copies.
                let pos = self.mesh.position(v);
                let cuts_for_v = self.verts_on_cuts[&v].clone();
                let cut_index = cuts_for_v[0]; // primary cut (vertex can only be on one cut path in the simple-cycle case)

                let n0 = self.graph.add_node(VirtualNode {
                    position: pos,
                    origin: VirtualNodeOrigin::CutDuplicate {
                        original: SurfacePoint::OnVertex { vertex: v },
                        peer: None,
                        cut_index,
                        side: 0,
                    },
                });
                let n1 = self.graph.add_node(VirtualNode {
                    position: pos,
                    origin: VirtualNodeOrigin::CutDuplicate {
                        original: SurfacePoint::OnVertex { vertex: v },
                        peer: None,
                        cut_index,
                        side: 1,
                    },
                });
                // Wire up peers.
                if let VirtualNodeOrigin::CutDuplicate { ref mut peer, .. } = self.graph[n0].origin
                {
                    *peer = Some(n1);
                }
                if let VirtualNodeOrigin::CutDuplicate { ref mut peer, .. } = self.graph[n1].origin
                {
                    *peer = Some(n0);
                }

                self.vert_to_nodes.insert(v, vec![n0, n1]);
                self.cut_node_sides.insert((v, 0), n0);
                self.cut_node_sides.insert((v, 1), n1);
            } else {
                // Regular interior or boundary vertex — single node.
                let pos = self.mesh.position(v);
                let node = self.graph.add_node(VirtualNode {
                    position: pos,
                    origin: VirtualNodeOrigin::MeshVertex(v),
                });
                self.vert_to_nodes.insert(v, vec![node]);
            }
        }

        // Create virtual nodes for on-edge cut interior points (excluding
        //     boundary midpoints at the cut endpoints).
        for (cut_index, cut) in self.cuts.iter().enumerate() {
            for pt in &cut.points[1..cut.points.len() - 1] {
                if let SurfacePoint::OnEdge { edge, t } = *pt {
                    let pos = self.mesh.position(self.mesh.root(edge)) * (1.0 - t)
                        + self.mesh.position(self.mesh.toor(edge)) * t;

                    let n0 = self.graph.add_node(VirtualNode {
                        position: pos,
                        origin: VirtualNodeOrigin::CutDuplicate {
                            original: SurfacePoint::OnEdge { edge, t },
                            peer: None,
                            cut_index,
                            side: 0,
                        },
                    });
                    let n1 = self.graph.add_node(VirtualNode {
                        position: pos,
                        origin: VirtualNodeOrigin::CutDuplicate {
                            original: SurfacePoint::OnEdge { edge, t },
                            peer: None,
                            cut_index,
                            side: 1,
                        },
                    });
                    if let VirtualNodeOrigin::CutDuplicate { ref mut peer, .. } =
                        self.graph[n0].origin
                    {
                        *peer = Some(n1);
                    }
                    if let VirtualNodeOrigin::CutDuplicate { ref mut peer, .. } =
                        self.graph[n1].origin
                    {
                        *peer = Some(n0);
                    }

                    self.cut_edge_point_sides.insert((edge, 0), n0);
                    self.cut_edge_point_sides.insert((edge, 1), n1);
                    // Also store the twin mapping.
                    let twin = self.mesh.twin(edge);
                    self.cut_edge_point_sides.insert((twin, 0), n0);
                    self.cut_edge_point_sides.insert((twin, 1), n1);
                }
            }
        }

        // Wire edges.
        // For every pair of mesh-adjacent patch vertices, add a virtual edge
        // unless the mesh edge crosses a cut (in which case the edge is replaced
        // by connections to the cut-point duplicates on each side).
        self.wire_interior_edges();

        // Wire boundary midpoints to their adjacent mesh vertices.
        self.wire_boundary_midpoints();

        // Wire consecutive cut points along each cut (both sides).
        self.wire_cut_chains();
    }

    /// Adds virtual edges between adjacent patch vertices, respecting cuts.
    fn wire_interior_edges(&mut self) {
        // We iterate over patch faces and add edges for each face's three vertex
        // pairs. We track which pairs we've already added to avoid duplicates.
        let mut added: HashSet<(NIdx, NIdx)> = HashSet::new();

        for &face in self.patch_faces {
            let verts: Vec<VertID> = self.mesh.vertices(face).collect();
            if verts.len() != 3 {
                continue;
            }

            // Determine which side of the cut this face is on (if any).
            let face_side = self.classify_face_side(face);

            for i in 0..3 {
                let va = verts[i];
                let vb = verts[(i + 1) % 3];

                // Check if this mesh edge is crossed by a cut.
                let (edge_ab, _) = match self.mesh.edge_between_verts(va, vb) {
                    Some(pair) => pair,
                    None => continue,
                };

                if self.edges_on_cuts.contains_key(&edge_ab) {
                    // This edge is cut — don't connect va and vb directly.
                    // Instead, connect each to the appropriate side of the cut
                    // point that lies on this edge.
                    let side = face_side.unwrap_or(0);

                    if let Some(&cut_node) = self.cut_edge_point_sides.get(&(edge_ab, side)) {
                        let na = self.resolve_node(va, Some(side));
                        let nb = self.resolve_node(vb, Some(side));

                        self.add_edge_once(na, cut_node, &mut added);
                        self.add_edge_once(nb, cut_node, &mut added);
                    }
                } else {
                    // Normal edge — connect the two vertices (using the correct
                    // side if either is a cut vertex).
                    let side = face_side;
                    let na = self.resolve_node(va, side);
                    let nb = self.resolve_node(vb, side);
                    self.add_edge_once(na, nb, &mut added);
                }
            }
        }
    }

    /// Adds virtual edges from boundary midpoints to the patch-side vertices of
    /// their boundary edges.
    fn wire_boundary_midpoints(&mut self) {
        for (&edge, &mid_node) in &self.midpoint_nodes {
            let root = self.mesh.root(edge);
            let toor = self.mesh.toor(edge);

            // Connect to whichever endpoint is in the patch.
            if self.patch_set.contains(&root) {
                let n = self.resolve_node_boundary(root);
                let len = (self.graph[mid_node].position - self.graph[n].position).norm();
                self.graph
                    .add_edge(mid_node, n, VirtualEdgeWeight { length: len });
            }
            if self.patch_set.contains(&toor) {
                let n = self.resolve_node_boundary(toor);
                let len = (self.graph[mid_node].position - self.graph[n].position).norm();
                self.graph
                    .add_edge(mid_node, n, VirtualEdgeWeight { length: len });
            }
        }
    }

    /// Wires consecutive cut points along each cut, on both sides.
    fn wire_cut_chains(&mut self) {
        let mut added: HashSet<(NIdx, NIdx)> = HashSet::new();

        // Pre-collect all node chains to avoid borrowing self.cuts while mutating self.graph.
        let mut chains: Vec<Vec<NIdx>> = Vec::new();
        for cut in &self.cuts {
            for side in 0..2u8 {
                let nodes: Vec<NIdx> = cut
                    .points
                    .iter()
                    .map(|pt| self.resolve_cut_point(pt, side, cut))
                    .collect();
                chains.push(nodes);
            }
        }

        for nodes in &chains {
            for pair in nodes.windows(2) {
                self.add_edge_once(pair[0], pair[1], &mut added);
            }
        }
    }

    /// Traces the single boundary loop of the virtual mesh.
    ///
    /// After all cuts the topology is a disk. The boundary consists of:
    /// - Arcs of boundary midpoints (and their neighboring mesh vertices on the
    ///   boundary) between cut endpoints.
    /// - Both sides of each cut path.
    ///
    /// We walk the boundary by starting at any boundary midpoint and repeatedly
    /// moving to the next boundary node (a node with fewer graph neighbors than
    /// expected, or more practically, a node whose degree in the virtual graph is
    /// less than what a fully interior node would have).
    fn trace_boundary(&self) -> Vec<NIdx> {
        // A node is on the boundary if it is a boundary midpoint, a cut
        // duplicate, or a mesh vertex that has at least one neighbor outside the
        // patch (i.e., it is on the original mesh boundary).
        let mut boundary_set: HashSet<NIdx> = HashSet::new();

        // All boundary midpoints are on the boundary.
        for &n in self.midpoint_nodes.values() {
            boundary_set.insert(n);
        }

        // All cut duplicates are on the boundary.
        for &n in self.cut_node_sides.values() {
            boundary_set.insert(n);
        }
        for &n in self.cut_edge_point_sides.values() {
            boundary_set.insert(n);
        }

        // Mesh boundary vertices (those adjacent to a vertex outside the patch).
        for (&v, nodes) in &self.vert_to_nodes {
            let on_mesh_boundary = self
                .mesh
                .neighbors(v)
                .any(|nbr| !self.patch_set.contains(&nbr));
            if on_mesh_boundary {
                for &n in nodes {
                    boundary_set.insert(n);
                }
            }
        }

        if boundary_set.is_empty() {
            return Vec::new();
        }

        // Walk the boundary: start at an arbitrary boundary-midpoint node (which
        // is guaranteed to be on the boundary), and follow boundary-neighbor links.
        let start = *self.midpoint_nodes.values().next().unwrap();
        let mut loop_nodes = vec![start];
        let mut visited: HashSet<NIdx> = HashSet::new();
        visited.insert(start);

        let mut current = start;
        loop {
            // Among the graph neighbors of `current`, find the next boundary node
            // we haven't visited yet.
            let next = self
                .graph
                .edges(current)
                .map(|e| {
                    let other = if e.source() == current {
                        e.target()
                    } else {
                        e.source()
                    };
                    other
                })
                .find(|&n| boundary_set.contains(&n) && !visited.contains(&n));

            match next {
                Some(n) => {
                    loop_nodes.push(n);
                    visited.insert(n);
                    current = n;
                }
                None => {
                    // Check if we can close the loop back to start.
                    let can_close = self.graph.edges(current).any(|e| {
                        let other = if e.source() == current {
                            e.target()
                        } else {
                            e.source()
                        };
                        other == start
                    });
                    if can_close && loop_nodes.len() > 2 {
                        // Closed loop — done.
                        break;
                    }
                    // Dead end — boundary tracing is imperfect. Return what we have.
                    break;
                }
            }
        }

        loop_nodes
    }

    /// Resolves a mesh vertex to its virtual node, picking the correct side if
    /// the vertex is on a cut.
    fn resolve_node(&self, v: VertID, side: Option<u8>) -> NIdx {
        if let Some(s) = side {
            if let Some(&n) = self.cut_node_sides.get(&(v, s)) {
                return n;
            }
        }
        // Not a cut vertex (or side is None) — use the single node.
        self.vert_to_nodes[&v][0]
    }

    /// Like `resolve_node` but for boundary-midpoint wiring: always resolves to
    /// the non-cut node or the side-0 node if on a cut. (Boundary midpoint
    /// vertices sit *on* the boundary, not on a cut interior.)
    fn resolve_node_boundary(&self, v: VertID) -> NIdx {
        self.vert_to_nodes[&v][0]
    }

    /// Resolves a cut surface point to the corresponding virtual node on a given
    /// side. Cut endpoints (boundary midpoints) resolve to the midpoint node.
    fn resolve_cut_point(&self, pt: &SurfacePoint, side: u8, cut: &CutInfo) -> NIdx {
        match *pt {
            SurfacePoint::OnVertex { vertex } => self
                .cut_node_sides
                .get(&(vertex, side))
                .copied()
                .unwrap_or_else(|| self.vert_to_nodes[&vertex][0]),
            SurfacePoint::OnEdge { edge, .. } => {
                // Check if this is a boundary midpoint (cut endpoint).
                if edge == cut.start_midpoint_edge {
                    return self.midpoint_nodes[&edge];
                }
                if edge == cut.end_midpoint_edge {
                    return self.midpoint_nodes[&edge];
                }
                // Interior cut edge point.
                self.cut_edge_point_sides
                    .get(&(edge, side))
                    .copied()
                    .expect("Cut edge point not found in cut_edge_point_sides")
            }
        }
    }

    /// Determines which "side" of the cut a face is on.
    ///
    /// For a face not touching any cut, returns `None`.
    /// For a face on one side of a cut, returns `Some(0)` or `Some(1)`.
    ///
    /// The convention is: if the face shares a vertex with a cut vertex, or if
    /// one of its edges is crossed by a cut, we determine the side by looking at
    /// the face's position relative to the cut direction (using the cross product
    /// of the cut tangent with the face normal).
    fn classify_face_side(&self, face: FaceID) -> Option<u8> {
        let face_verts: Vec<VertID> = self.mesh.vertices(face).collect();

        // Check if any face vertex is on a cut.
        for &v in &face_verts {
            if let Some(cut_indices) = self.verts_on_cuts.get(&v) {
                let cut_index = cut_indices[0];
                return Some(self.side_of_face_relative_to_cut(face, cut_index));
            }
        }

        // Check if any face edge is crossed by a cut.
        for edge in self.mesh.edges(face) {
            if let Some(&cut_index) = self.edges_on_cuts.get(&edge) {
                return Some(self.side_of_face_relative_to_cut(face, cut_index));
            }
        }

        None
    }

    /// Determines which side (0 or 1) a face is on relative to a specific cut.
    ///
    /// Uses the signed volume / cross-product test: the cut has a direction
    /// (from start boundary to end boundary). A face is on side 0 if it is on the
    /// "left" of the cut direction (in a surface-normal sense), and side 1 if on
    /// the "right".
    fn side_of_face_relative_to_cut(&self, face: FaceID, cut_index: usize) -> u8 {
        let cut = &self.cuts[cut_index];
        let face_verts: Vec<VertID> = self.mesh.vertices(face).collect();
        let face_center = face_verts
            .iter()
            .map(|&v| self.mesh.position(v))
            .sum::<Vector3D>()
            / face_verts.len() as f64;

        // Find the closest cut point to the face center and use the cut tangent there.
        let (cut_pos, cut_tangent) = self.cut_tangent_near(cut, &face_center);

        // Surface normal at the cut point (average face normal).
        let surface_normal = face_verts
            .iter()
            .map(|&v| self.mesh.position(v))
            .collect::<Vec<_>>();
        let face_normal = if surface_normal.len() >= 3 {
            let e1 = surface_normal[1] - surface_normal[0];
            let e2 = surface_normal[2] - surface_normal[0];
            e1.cross(&e2)
        } else {
            Vector3D::new(0.0, 0.0, 1.0)
        };

        // The side is determined by the sign of dot(cross(tangent, normal), face_center - cut_pos).
        let to_face = face_center - cut_pos;
        let side_vec = cut_tangent.cross(&face_normal);
        if side_vec.dot(&to_face) >= 0.0 {
            0
        } else {
            1
        }
    }

    /// Returns the position and tangent of the cut at the point closest to `query`.
    fn cut_tangent_near(&self, cut: &CutInfo, query: &Vector3D) -> (Vector3D, Vector3D) {
        let positions: Vec<Vector3D> = cut.points.iter().map(|pt| pt.position(self.mesh)).collect();

        if positions.len() < 2 {
            return (
                positions.first().copied().unwrap_or(Vector3D::zeros()),
                Vector3D::new(1.0, 0.0, 0.0),
            );
        }

        // Find the segment closest to the query point.
        let mut best_dist = f64::INFINITY;
        let mut best_pos = positions[0];
        let mut best_tangent = positions[1] - positions[0];
        for i in 0..positions.len() - 1 {
            let a = positions[i];
            let b = positions[i + 1];
            let ab = b - a;
            let ab_len = ab.norm();
            if ab_len < 1e-15 {
                continue;
            }
            let t = (((*query) - a).dot(&ab) / (ab_len * ab_len)).clamp(0.0, 1.0);
            let closest = a + ab * t;
            let dist = (closest - (*query)).norm();
            if dist < best_dist {
                best_dist = dist;
                best_pos = closest;
                best_tangent = ab;
            }
        }

        (best_pos, best_tangent)
    }

    /// Adds an edge between two virtual nodes if it hasn't been added yet.
    fn add_edge_once(&mut self, a: NIdx, b: NIdx, added: &mut HashSet<(NIdx, NIdx)>) {
        if a == b {
            return;
        }
        let key = if a < b { (a, b) } else { (b, a) };
        if added.insert(key) {
            let len = (self.graph[a].position - self.graph[b].position).norm();
            self.graph.add_edge(a, b, VirtualEdgeWeight { length: len });
        }
    }
}
