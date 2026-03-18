use std::collections::{HashMap, HashSet};

use log::error;
use mehsh::prelude::{HasPosition, HasVertices, Mesh, Vector3D};
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::stable_graph::StableUnGraph;
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};

use crate::prelude::{EdgeID, FaceID, VertID, INPUT};
use crate::skeleton::boundary_loop::BoundaryLoop;
use crate::skeleton::orthogonalize::LabeledCurveSkeleton;

use super::{CutPath, CuttingPlan, SurfacePoint};

/// Tracks where a virtual node came from, so that we can map results back to the
/// original mesh and relate duplicated cut nodes to each other.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// `original` is the underlying surface point (always a mesh vertex for
    /// edge-following cuts).
    CutDuplicate {
        original: SurfacePoint,
        /// The index of the other copy (the "peer" on the opposite side of the cut).
        /// Set to `None` during construction and filled in once both copies exist.
        peer: Option<NodeIndex>,
        /// Which cut this came from (index into `CuttingPlan::cuts`).
        cut_index: usize,
        /// Which side of the cut: 0 or 1.
        side: u8,
    },
}

/// Weight on virtual-mesh edges. Currently stores the Euclidean length so that
/// Laplacian weights are easy to compute.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualEdgeWeight {
    pub length: f64,
}

/// A graph-based mesh representation of a single region after cutting.
///
/// The surface is "opened up" along every cut so that each side of a cut has its
/// own copy of the cut vertices. Boundary midpoints from the boundary loops are
/// introduced as explicit vertices. The result is a topological disk with a
/// single boundary loop.
///
/// Cuts follow mesh edges exclusively — no face-interior crossing points exist.
/// The only face-traversing segments are at cut endpoints (boundary midpoint →
/// first/last mesh vertex of the cut).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualFlatGeometry {
    /// The mesh-like adjacency graph. Each node carries a 3D position plus its
    /// origin information; each edge carries a length.
    pub graph: StableUnGraph<VirtualNode, VirtualEdgeWeight>,

    /// original mesh vertex -> virtual node(s).
    /// Interior vertices map to exactly one node; cut vertices map to two.
    pub vert_to_nodes: HashMap<VertID, Vec<NodeIndex>>,

    /// The single boundary loop of this virtual mesh, as an ordered sequence of
    /// node indices. After all cuts are applied the topology is a disk, so the
    /// boundary is one simple cycle. Every node appears at most once.
    pub boundary_loop: Vec<NodeIndex>,
}

/// Per-node payload in the virtual graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualNode {
    pub position: Vector3D,
    pub origin: VirtualNodeOrigin,
}

impl Default for VirtualFlatGeometry {
    fn default() -> Self {
        Self::empty()
    }
}

impl VirtualFlatGeometry {
    /// Returns an empty `VirtualFlatGeometry` with no nodes, edges, or boundary.
    /// Used as a placeholder for degree-0 regions that are TODO for now.
    pub fn empty() -> Self {
        VirtualFlatGeometry {
            graph: StableUnGraph::default(),
            vert_to_nodes: HashMap::new(),
            boundary_loop: Vec::new(),
        }
    }

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

        // Register cut vertices and identify which mesh vertices are on cuts.
        builder.register_cuts(&cutting_plan.cuts);

        // Duplicate boundary midpoints that are cut endpoints so the boundary
        // loop becomes a proper simple cycle.
        builder.duplicate_cut_endpoint_midpoints();

        // Populate the graph: create interior + boundary-midpoint + duplicated
        // cut nodes, then wire up edges.
        builder.build_graph();

        // Trace the disk boundary.
        let boundary_loop = builder.trace_boundary();

        // Verify all midpoint nodes are in the boundary loop.
        {
            let bset: HashSet<NodeIndex> = boundary_loop.iter().copied().collect();
            for (&edge, &node) in &builder.midpoint_nodes {
                if !bset.contains(&node) {
                    let skel_edge = builder.skeleton_edge_for_midpoint.get(&edge);
                    log::error!(
                        "Primary midpoint {:?} (edge {:?}, skel {:?}) NOT in boundary loop \
                         (loop len = {}, n_cuts = {})",
                        node, edge, skel_edge, boundary_loop.len(), builder.cuts.len()
                    );
                }
            }
            for (&edge, &node) in &builder.midpoint_node_peers {
                if !bset.contains(&node) {
                    log::error!(
                        "Peer midpoint {:?} (edge {:?}) NOT in boundary loop",
                        node, edge
                    );
                }
            }
        }

        // Wire edges between consecutive boundary-loop nodes so every boundary
        // node has at least two boundary neighbours.
        wire_boundary_loop_edges(&mut builder.graph, &boundary_loop);

        let vfg = VirtualFlatGeometry {
            graph: builder.graph,
            vert_to_nodes: builder.vert_to_nodes,
            boundary_loop,
        };

        // Invariant checks.
        check_invariants(&vfg);

        vfg
    }
}

/// Checks structural invariants on the completed VFG.
fn check_invariants(vfg: &VirtualFlatGeometry) {
    let boundary_set: HashSet<NodeIndex> = vfg.boundary_loop.iter().copied().collect();

    for node in vfg.graph.node_indices() {
        let degree = vfg.graph.edges(node).count();
        let is_boundary = boundary_set.contains(&node);

        if is_boundary {
            // Boundary nodes need at least 1 neighbour. In degenerate cases
            // (e.g. all boundary edges share one patch vertex on a small
            // polycube) a midpoint may only connect to that single vertex.
            assert!(
                degree >= 1,
                "VFG invariant violated: boundary node {:?} ({:?}) has 0 neighbours",
                node,
                vfg.graph[node].origin,
            );
        } else {
            // Interior nodes need at least 3 neighbours (triangle mesh).
            assert!(
                degree >= 3,
                "VFG invariant violated: interior node {:?} ({:?}) has {} neighbours, expected >= 3",
                node,
                vfg.graph[node].origin,
                degree
            );
        }
    }

    // Boundary loop is a simple cycle with at least 3 nodes.
    assert!(
        vfg.boundary_loop.len() >= 3,
        "VFG invariant violated: boundary loop has {} nodes, expected >= 3",
        vfg.boundary_loop.len()
    );
    assert_eq!(
        boundary_set.len(),
        vfg.boundary_loop.len(),
        "VFG invariant violated: boundary loop contains duplicate nodes ({} unique out of {})",
        boundary_set.len(),
        vfg.boundary_loop.len()
    );
}

/// Adds edges between consecutive nodes in `boundary_loop` (cyclic).
fn wire_boundary_loop_edges(
    graph: &mut StableUnGraph<VirtualNode, VirtualEdgeWeight>,
    boundary_loop: &[NodeIndex],
) {
    let n = boundary_loop.len();
    for i in 0..n {
        let a = boundary_loop[i];
        let b = boundary_loop[(i + 1) % n];
        if a == b {
            continue;
        }
        // Check if the edge already exists.
        let exists = graph
            .edges(a)
            .any(|e| (e.source() == a && e.target() == b) || (e.source() == b && e.target() == a));
        if !exists {
            let len = (graph[a].position - graph[b].position).norm();
            graph.add_edge(a, b, VirtualEdgeWeight { length: len });
        }
    }
}

// ===========================================================================
//  Builder
// ===========================================================================

/// Intermediate state used during construction.
struct Builder<'a> {
    mesh: &'a Mesh<INPUT>,
    patch_set: &'a HashSet<VertID>,
    patch_faces: &'a [FaceID],

    graph: StableUnGraph<VirtualNode, VirtualEdgeWeight>,

    // Maps from mesh VertID -> virtual node(s).
    vert_to_nodes: HashMap<VertID, Vec<NodeIndex>>,

    // Boundary midpoint nodes: boundary EdgeID -> virtual node index (primary copy).
    midpoint_nodes: HashMap<EdgeID, NodeIndex>,

    // For cut-endpoint midpoints: boundary EdgeID -> peer virtual node index (side-1 copy).
    midpoint_node_peers: HashMap<EdgeID, NodeIndex>,

    // Cut information.
    cuts: Vec<CutInfo>,

    // Which mesh vertices are on any cut. Maps VertID -> list of cut indices.
    verts_on_cuts: HashMap<VertID, Vec<usize>>,

    // After duplication: for each (VertID that's on a cut, side), the virtual node.
    cut_node_sides: HashMap<(VertID, u8), NodeIndex>,

    // Ordered boundary edge IDs per skeleton edge.
    boundary_edge_order: HashMap<EdgeIndex, Vec<EdgeID>>,

    // Reverse map: boundary EdgeID -> skeleton edge index.
    skeleton_edge_for_midpoint: HashMap<EdgeID, EdgeIndex>,

    // Cut side chains: per cut index, [side 0 chain, side 1 chain].
    // Each chain is [start_midpoint_node, ..interior_vertices.., end_midpoint_node].
    cut_side_chains: Vec<[Vec<NodeIndex>; 2]>,
}

/// Internal record for a single cut.
struct CutInfo {
    /// Ordered surface points along the cut, from start boundary to end boundary.
    /// First/last are OnEdge (boundary midpoints), interior are OnVertex.
    points: Vec<SurfacePoint>,
    /// Boundary edge at the start of the cut.
    start_midpoint_edge: EdgeID,
    /// Boundary edge at the end of the cut.
    end_midpoint_edge: EdgeID,
}

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
            midpoint_node_peers: HashMap::new(),
            cuts: Vec::new(),
            verts_on_cuts: HashMap::new(),
            cut_node_sides: HashMap::new(),
            boundary_edge_order: HashMap::new(),
            skeleton_edge_for_midpoint: HashMap::new(),
            cut_side_chains: Vec::new(),
        }
    }

    fn add_boundary_midpoints(&mut self, boundary: &BoundaryLoop, edge_idx: EdgeIndex) {
        let mut order = Vec::new();
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
            self.skeleton_edge_for_midpoint.insert(edge, edge_idx);
            order.push(edge);
        }
        self.boundary_edge_order.insert(edge_idx, order);
    }

    /// Registers cut paths. Interior points must all be OnVertex (edge-following cuts).
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

            // Register mesh vertices that lie on this cut (interior points only).
            for pt in &points[1..points.len() - 1] {
                match *pt {
                    SurfacePoint::OnVertex { vertex } => {
                        self.verts_on_cuts
                            .entry(vertex)
                            .or_default()
                            .push(cut_index);
                    }
                    SurfacePoint::OnEdge { .. } => {
                        panic!(
                            "Interior cut points must be mesh vertices (OnVertex), \
                             not on-edge points. Geodesic straightening should have been removed."
                        );
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

    /// Creates a peer (side-1) copy for each boundary midpoint that is a cut endpoint.
    fn duplicate_cut_endpoint_midpoints(&mut self) {
        let mut to_duplicate: Vec<EdgeID> = Vec::new();
        for cut in &self.cuts {
            to_duplicate.push(cut.start_midpoint_edge);
            to_duplicate.push(cut.end_midpoint_edge);
        }
        to_duplicate.sort();
        to_duplicate.dedup();

        for edge in to_duplicate {
            let primary = self.midpoint_nodes[&edge];
            let pos = self.graph[primary].position;
            let boundary = match &self.graph[primary].origin {
                VirtualNodeOrigin::BoundaryMidpoint { boundary, .. } => *boundary,
                _ => unreachable!("Expected BoundaryMidpoint origin"),
            };

            let peer = self.graph.add_node(VirtualNode {
                position: pos,
                origin: VirtualNodeOrigin::BoundaryMidpoint { edge, boundary },
            });
            self.midpoint_node_peers.insert(edge, peer);
        }
    }

    fn build_graph(&mut self) {
        // Create virtual nodes for every patch vertex.
        for &v in self.patch_set.iter() {
            if self.verts_on_cuts.contains_key(&v) {
                // Duplicate: create two copies (one per side of the cut).
                let pos = self.mesh.position(v);
                let cuts_for_v = &self.verts_on_cuts[&v];
                assert_eq!(
                    cuts_for_v.len(),
                    1,
                    "Vertex {v:?} is on {} cuts; expected exactly one (disjoint cuts invariant)",
                    cuts_for_v.len()
                );
                let cut_index = cuts_for_v[0];

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

        // Wire edges.
        self.wire_interior_edges();
        self.wire_boundary_midpoints();
        self.wire_boundary_arcs();
        self.wire_cut_chains();
    }

    /// Adds virtual edges between adjacent patch vertices, using per-vertex
    /// resolution to pick the correct side of any cut.
    fn wire_interior_edges(&mut self) {
        let mut added: HashSet<(NodeIndex, NodeIndex)> = HashSet::new();

        for &face in self.patch_faces {
            let verts: Vec<VertID> = self.mesh.vertices(face).collect();
            if verts.len() != 3 {
                continue;
            }

            // Resolve each vertex to its virtual node for this face.
            let nodes: Vec<NodeIndex> = verts
                .iter()
                .map(|&v| self.resolve_node_for_face(v, face))
                .collect();

            for i in 0..3 {
                self.add_edge_once(nodes[i], nodes[(i + 1) % 3], &mut added);
            }
        }
    }

    /// Adds virtual edges from boundary midpoints to the patch-side vertices of
    /// their boundary edges. Connects both primary and peer copies.
    fn wire_boundary_midpoints(&mut self) {
        // Collect all (edge, node) pairs first to avoid borrow conflicts.
        let mut to_wire: Vec<(EdgeID, NodeIndex)> = self
            .midpoint_nodes
            .iter()
            .map(|(&e, &n)| (e, n))
            .collect();
        // Also collect peer copies.
        for (&e, &n) in &self.midpoint_node_peers {
            to_wire.push((e, n));
        }

        for (edge, mid_node) in to_wire {
            self.connect_midpoint_to_patch_verts(edge, mid_node);
        }
    }

    fn connect_midpoint_to_patch_verts(&mut self, edge: EdgeID, mid_node: NodeIndex) {
        let root = self.mesh.root(edge);
        let toor = self.mesh.toor(edge);

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

    /// Wires consecutive cut points along each cut, on both sides.
    /// Also stores the cut side chains for explicit boundary construction.
    fn wire_cut_chains(&mut self) {
        let mut added: HashSet<(NodeIndex, NodeIndex)> = HashSet::new();

        let mut all_side_chains: Vec<[Vec<NodeIndex>; 2]> = Vec::new();
        for cut in &self.cuts {
            let mut sides: [Vec<NodeIndex>; 2] = [Vec::new(), Vec::new()];
            for side in 0..2u8 {
                let nodes: Vec<NodeIndex> = cut
                    .points
                    .iter()
                    .map(|pt| self.resolve_cut_point(pt, side, cut))
                    .collect();
                sides[side as usize] = nodes;
            }
            all_side_chains.push(sides);
        }

        for sides in &all_side_chains {
            for side_nodes in sides {
                for pair in side_nodes.windows(2) {
                    self.add_edge_once(pair[0], pair[1], &mut added);
                }
            }
        }

        self.cut_side_chains = all_side_chains;
    }

    /// Adds edges between consecutive boundary-side patch vertices.
    fn wire_boundary_arcs(&mut self) {
        let mut added: HashSet<(NodeIndex, NodeIndex)> = HashSet::new();
        let mut pairs: Vec<(VertID, VertID)> = Vec::new();
        for edges in self.boundary_edge_order.values() {
            let n = edges.len();
            for i in 0..n {
                let e_from = edges[i];
                let e_to = edges[(i + 1) % n];
                let from_v = self.patch_vertex_of_boundary_edge(e_from);
                let to_v = self.patch_vertex_of_boundary_edge(e_to);
                if from_v != to_v {
                    pairs.push((from_v, to_v));
                }
            }
        }
        for (from_v, to_v) in pairs {
            let na = self.resolve_node_boundary(from_v);
            let nb = self.resolve_node_boundary(to_v);
            self.add_edge_once(na, nb, &mut added);
        }
    }

    // -----------------------------------------------------------------------
    //  Boundary tracing
    // -----------------------------------------------------------------------

    fn trace_boundary(&self) -> Vec<NodeIndex> {
        let raw = if self.cuts.is_empty() {
            self.trace_boundary_no_cuts()
        } else {
            self.trace_boundary_with_cuts()
        };

        // Deduplicate: shared corner vertices (at the junction of multiple
        // boundary loops) may appear in more than one boundary walk. Keep the
        // first occurrence so the loop stays a simple cycle.
        let mut seen = HashSet::new();
        raw.into_iter().filter(|node| seen.insert(*node)).collect()
    }

    /// Boundary trace for regions with no cuts (degree ≤ 1).
    fn trace_boundary_no_cuts(&self) -> Vec<NodeIndex> {
        if self.boundary_edge_order.is_empty() {
            return Vec::new();
        }
        let edges = self.boundary_edge_order.values().next().unwrap();
        let n = edges.len();
        if n == 0 {
            return Vec::new();
        }
        let mut result = Vec::new();
        for i in 0..n {
            let edge = edges[i];
            let next_edge = edges[(i + 1) % n];
            result.push(self.midpoint_nodes[&edge]);
            self.emit_connecting_vertices(edge, next_edge, &mut result);
        }
        result
    }

    /// Constructs the boundary loop for regions with cuts.
    ///
    /// The boundary of the cut-open disk is assembled by DFS over the
    /// boundary-loop tree. Cut-endpoint midpoints have been duplicated: the
    /// primary copy is used when entering a cut, the peer when returning.
    fn trace_boundary_with_cuts(&self) -> Vec<NodeIndex> {
        // Build the boundary-loop spanning tree.
        let mut tree_adj: HashMap<EdgeIndex, Vec<(EdgeIndex, usize)>> = HashMap::new();
        for (cut_idx, cut) in self.cuts.iter().enumerate() {
            let start_skel = self.skeleton_edge_for_midpoint[&cut.start_midpoint_edge];
            let end_skel = self.skeleton_edge_for_midpoint[&cut.end_midpoint_edge];
            tree_adj
                .entry(start_skel)
                .or_default()
                .push((end_skel, cut_idx));
            tree_adj
                .entry(end_skel)
                .or_default()
                .push((start_skel, cut_idx));
        }

        let root_skel = *self.boundary_edge_order.keys().next().unwrap();
        let mut result = Vec::new();
        let mut visited_boundaries: HashSet<EdgeIndex> = HashSet::new();

        self.dfs_boundary(
            root_skel,
            None,
            &tree_adj,
            &mut visited_boundaries,
            &mut result,
        );

        result
    }

    /// DFS step: emits boundary nodes for one boundary loop.
    fn dfs_boundary(
        &self,
        boundary_idx: EdgeIndex,
        entry_midpoint_edge: Option<EdgeID>,
        tree_adj: &HashMap<EdgeIndex, Vec<(EdgeIndex, usize)>>,
        visited: &mut HashSet<EdgeIndex>,
        result: &mut Vec<NodeIndex>,
    ) {
        visited.insert(boundary_idx);

        let edges = match self.boundary_edge_order.get(&boundary_idx) {
            Some(e) => e,
            None => return,
        };
        let n = edges.len();
        if n == 0 {
            return;
        }

        // Identify cut endpoints on this boundary that lead to unvisited children.
        let mut cuts_here: HashMap<EdgeID, (usize, EdgeIndex)> = HashMap::new();
        if let Some(adjs) = tree_adj.get(&boundary_idx) {
            for &(target, cut_idx) in adjs {
                if visited.contains(&target) {
                    continue;
                }
                let cut = &self.cuts[cut_idx];
                let mid_edge =
                    if self.skeleton_edge_for_midpoint.get(&cut.start_midpoint_edge)
                        == Some(&boundary_idx)
                    {
                        cut.start_midpoint_edge
                    } else {
                        cut.end_midpoint_edge
                    };
                cuts_here.insert(mid_edge, (cut_idx, target));
            }
        }

        // Find starting position in the boundary loop.
        let start_idx = if let Some(entry) = entry_midpoint_edge {
            edges.iter().position(|&e| e == entry).unwrap_or(0)
        } else {
            // Root: start at the first cut endpoint if any, otherwise 0.
            edges
                .iter()
                .position(|&e| cuts_here.contains_key(&e))
                .unwrap_or(0)
        };

        // Walk the full boundary loop.
        for i in 0..n {
            let idx = (start_idx + i) % n;
            let edge = edges[idx];
            let next_edge = edges[(idx + 1) % n];

            // Emit midpoint of current boundary edge.
            // If this midpoint has a peer (cut endpoint), emit the primary (side 0).
            result.push(self.midpoint_nodes[&edge]);

            // If this midpoint is a cut endpoint leading to an unvisited
            // boundary, perform the cut detour.
            if let Some(&(cut_idx, target)) = cuts_here.get(&edge) {
                let cut = &self.cuts[cut_idx];
                let chains = &self.cut_side_chains[cut_idx];
                let starts_here = cut.start_midpoint_edge == edge;
                let other_mid_edge = if starts_here {
                    cut.end_midpoint_edge
                } else {
                    cut.start_midpoint_edge
                };

                // Emit interior nodes of cut side 0 (excluding endpoint midpoints).
                let interior_0 = &chains[0][1..chains[0].len().saturating_sub(1)];
                if starts_here {
                    for &node in interior_0 {
                        result.push(node);
                    }
                } else {
                    for &node in interior_0.iter().rev() {
                        result.push(node);
                    }
                }

                // Recurse into the connected boundary loop.
                // The child will push the primary copy of other_mid_edge as its
                // first midpoint.
                self.dfs_boundary(target, Some(other_mid_edge), tree_adj, visited, result);

                // Emit the returning midpoint: peer (side-1) copy of other_mid_edge.
                result.push(
                    self.midpoint_node_peers
                        .get(&other_mid_edge)
                        .copied()
                        .unwrap_or_else(|| self.midpoint_nodes[&other_mid_edge]),
                );

                // Emit interior nodes of cut side 1 (reverse direction).
                let interior_1 = &chains[1][1..chains[1].len().saturating_sub(1)];
                if starts_here {
                    for &node in interior_1.iter().rev() {
                        result.push(node);
                    }
                } else {
                    for &node in interior_1 {
                        result.push(node);
                    }
                }

                // Emit the peer (side-1) copy of THIS midpoint (returning from cut).
                result.push(
                    self.midpoint_node_peers
                        .get(&edge)
                        .copied()
                        .unwrap_or_else(|| self.midpoint_nodes[&edge]),
                );
            }

            // Emit connecting mesh vertices to the next midpoint.
            self.emit_connecting_vertices(edge, next_edge, result);
        }
    }

    // -----------------------------------------------------------------------
    //  Resolution helpers
    // -----------------------------------------------------------------------

    /// Resolves a mesh vertex to its virtual node for a specific face.
    /// If the vertex is on a cut, determines which side of the cut the face is
    /// on and returns the corresponding copy.
    fn resolve_node_for_face(&self, v: VertID, face: FaceID) -> NodeIndex {
        if let Some(cut_indices) = self.verts_on_cuts.get(&v) {
            let cut_index = cut_indices[0];
            let side = self.side_of_face_relative_to_cut(face, cut_index);
            self.cut_node_sides[&(v, side)]
        } else {
            self.vert_to_nodes[&v][0]
        }
    }

    /// Like `resolve_node_for_face` but for boundary wiring: always resolves to
    /// the first node (side-0 if on a cut).
    fn resolve_node_boundary(&self, v: VertID) -> NodeIndex {
        self.vert_to_nodes[&v][0]
    }

    /// Resolves a cut surface point to the corresponding virtual node on a given
    /// side. Cut endpoints (boundary midpoints) resolve to the primary or peer
    /// midpoint node depending on `side`.
    fn resolve_cut_point(&self, pt: &SurfacePoint, side: u8, cut: &CutInfo) -> NodeIndex {
        match *pt {
            SurfacePoint::OnVertex { vertex } => self
                .cut_node_sides
                .get(&(vertex, side))
                .copied()
                .unwrap_or_else(|| self.vert_to_nodes[&vertex][0]),
            SurfacePoint::OnEdge { edge, .. } => {
                assert!(
                    edge == cut.start_midpoint_edge || edge == cut.end_midpoint_edge,
                    "OnEdge surface point in cut must be a boundary midpoint"
                );
                if side == 0 {
                    self.midpoint_nodes[&edge]
                } else {
                    self.midpoint_node_peers
                        .get(&edge)
                        .copied()
                        .unwrap_or_else(|| self.midpoint_nodes[&edge])
                }
            }
        }
    }

    /// Determines which side (0 or 1) a face is on relative to a specific cut.
    ///
    /// Uses the signed cross-product test: the cut has a direction (from start
    /// boundary to end boundary). A face is on side 0 if it is on the "left" of
    /// the cut direction (in a surface-normal sense), and side 1 if on the "right".
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

        // Face normal.
        let face_positions = face_verts
            .iter()
            .map(|&v| self.mesh.position(v))
            .collect::<Vec<_>>();
        let face_normal = if face_positions.len() >= 3 {
            let e1 = face_positions[1] - face_positions[0];
            let e2 = face_positions[2] - face_positions[0];
            e1.cross(&e2)
        } else {
            Vector3D::new(0.0, 0.0, 1.0)
        };

        let to_face = face_center - cut_pos;
        let side_vec = cut_tangent.cross(&face_normal);
        if side_vec.dot(&side_vec) < 1e-20 {
            error!("Degenerate cut tangent and face normal; defaulting to side 0");
            return 0;
        }
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

    // -----------------------------------------------------------------------
    //  Small helpers
    // -----------------------------------------------------------------------

    /// Returns the vertex of a boundary edge that belongs to the current patch.
    fn patch_vertex_of_boundary_edge(&self, edge: EdgeID) -> VertID {
        let root = self.mesh.root(edge);
        if self.patch_set.contains(&root) {
            root
        } else {
            self.mesh.toor(edge)
        }
    }

    /// Emits the mesh boundary vertices connecting two consecutive midpoints.
    fn emit_connecting_vertices(
        &self,
        from_edge: EdgeID,
        to_edge: EdgeID,
        result: &mut Vec<NodeIndex>,
    ) {
        let from_vert = self.patch_vertex_of_boundary_edge(from_edge);
        let to_vert = self.patch_vertex_of_boundary_edge(to_edge);
        if from_vert == to_vert {
            result.push(self.resolve_node_boundary(from_vert));
        } else {
            result.push(self.resolve_node_boundary(from_vert));
            result.push(self.resolve_node_boundary(to_vert));
        }
    }

    /// Adds an edge between two virtual nodes if it hasn't been added yet.
    fn add_edge_once(
        &mut self,
        a: NodeIndex,
        b: NodeIndex,
        added: &mut HashSet<(NodeIndex, NodeIndex)>,
    ) {
        if a == b {
            return;
        }
        let key = if a < b { (a, b) } else { (b, a) };
        if added.insert(key) {
            let len = (self.graph[a].position - self.graph[b].position).norm();
            self.graph
                .add_edge(a, b, VirtualEdgeWeight { length: len });
        }
    }
}
