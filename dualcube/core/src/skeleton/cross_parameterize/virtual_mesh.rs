use std::collections::{HashMap, HashSet};

use log;
use mehsh::prelude::{HasEdges, HasNeighbors, HasPosition, HasVertices, Mesh, Vector3D};
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::stable_graph::StableUnGraph;
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};

use crate::prelude::{EdgeID, VertID, INPUT};
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
        /// Which side of the cut: `false` = left (the side containing the face
        /// to the left of the cut direction start→end), `true` = right.
        side: bool,
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
    ///
    /// The algorithm:
    /// 1. Add VFG nodes for all patch vertices (cut vertices get two copies).
    /// 2. Add VFG nodes for all boundary midpoints.
    /// 3. Wire edges: mesh edges between patch vertices become VFG edges,
    ///    with cut edges duplicated (one per side). Boundary midpoints connect
    ///    to their adjacent patch vertex (or cut duplicate).
    /// 4. Trace the single boundary loop via DFS on the cut spanning tree.
    pub fn build(
        node_idx: NodeIndex,
        skeleton: &LabeledCurveSkeleton,
        mesh: &Mesh<INPUT>,
        cutting_plan: &CuttingPlan,
    ) -> Self {
        let patch_verts = &skeleton[node_idx].skeleton_node.patch_vertices;
        let patch_set: HashSet<VertID> = patch_verts.iter().copied().collect();

        // ── Phase 1: Collect context ──────────────────────────────────────

        // Identify cut vertices and which cut they belong to.
        let mut cut_vertex_set: HashSet<VertID> = HashSet::new();
        let mut vert_to_cut_index: HashMap<VertID, usize> = HashMap::new();
        for (ci, cut) in cutting_plan.cuts.iter().enumerate() {
            for pt in &cut.path.points {
                if let SurfacePoint::OnVertex { vertex } = pt {
                    cut_vertex_set.insert(*vertex);
                    vert_to_cut_index.insert(*vertex, ci);
                }
            }
        }

        // Build the ordered list of mesh vertices along each cut path
        // (excluding the boundary midpoint endpoints).
        let cut_vertex_paths: Vec<Vec<VertID>> = cutting_plan
            .cuts
            .iter()
            .map(|cut| {
                cut.path
                    .points
                    .iter()
                    .filter_map(|pt| match pt {
                        SurfacePoint::OnVertex { vertex } => Some(*vertex),
                        _ => None,
                    })
                    .collect()
            })
            .collect();

        // Set of unordered cut edges (pairs of consecutive cut vertices).
        let mut cut_edge_set: HashSet<(VertID, VertID)> = HashSet::new();
        for vpath in &cut_vertex_paths {
            for w in vpath.windows(2) {
                let key = if w[0] < w[1] {
                    (w[0], w[1])
                } else {
                    (w[1], w[0])
                };
                cut_edge_set.insert(key);
            }
        }

        // Collect all boundary edges and which skeleton edge they belong to.
        let mut boundary_edge_to_skeleton: HashMap<EdgeID, EdgeIndex> = HashMap::new();
        for edge_ref in skeleton.edges(node_idx) {
            let skel_edge = edge_ref.id();
            for &be in &edge_ref.weight().boundary_loop.edge_midpoints {
                boundary_edge_to_skeleton.insert(be, skel_edge);
            }
        }

        // Which boundary midpoints are cut endpoints?
        // Maps boundary EdgeID -> (cut_index, is_start).
        let mut cut_endpoint_midpoints: HashMap<EdgeID, (usize, bool)> = HashMap::new();
        for (ci, cut) in cutting_plan.cuts.iter().enumerate() {
            if let SurfacePoint::OnEdge { edge, .. } = cut.path.points[0] {
                cut_endpoint_midpoints.insert(edge, (ci, true));
            }
            if let SurfacePoint::OnEdge { edge, .. } = *cut.path.points.last().unwrap() {
                cut_endpoint_midpoints.insert(edge, (ci, false));
            }
        }

        // ── Phase 2: Compute side assignments for cut vertices ────────────
        // For each cut vertex, determine which mesh neighbors go to left vs right.
        // Convention: "left" (side=false) = face to the left of the cut direction.

        let cut_side_assignments: HashMap<VertID, HashMap<VertID, bool>> =
            compute_all_cut_side_assignments(
                &cutting_plan.cuts,
                &cut_vertex_paths,
                &patch_set,
                mesh,
            );

        // ── Phase 3: Add nodes ────────────────────────────────────────────

        let mut graph: StableUnGraph<VirtualNode, VirtualEdgeWeight> = StableUnGraph::default();
        let mut vert_to_nodes: HashMap<VertID, Vec<NodeIndex>> = HashMap::new();

        // 3a. Non-cut patch vertices: one node each.
        for &v in patch_verts {
            if cut_vertex_set.contains(&v) {
                continue;
            }
            let ni = graph.add_node(VirtualNode {
                position: mesh.position(v),
                origin: VirtualNodeOrigin::MeshVertex(v),
            });
            vert_to_nodes.insert(v, vec![ni]);
        }

        // 3b. Cut vertices: two nodes each (left and right).
        let mut cut_duplicates: HashMap<VertID, (NodeIndex, NodeIndex)> = HashMap::new();
        for &v in &cut_vertex_set {
            let cut_index = vert_to_cut_index[&v];
            let pos = mesh.position(v);

            let left = graph.add_node(VirtualNode {
                position: pos,
                origin: VirtualNodeOrigin::CutDuplicate {
                    original: SurfacePoint::OnVertex { vertex: v },
                    peer: None,
                    cut_index,
                    side: false,
                },
            });
            let right = graph.add_node(VirtualNode {
                position: pos,
                origin: VirtualNodeOrigin::CutDuplicate {
                    original: SurfacePoint::OnVertex { vertex: v },
                    peer: None,
                    cut_index,
                    side: true,
                },
            });

            // Set peer pointers.
            if let VirtualNodeOrigin::CutDuplicate { ref mut peer, .. } = graph[left].origin {
                *peer = Some(right);
            }
            if let VirtualNodeOrigin::CutDuplicate { ref mut peer, .. } = graph[right].origin {
                *peer = Some(left);
            }

            cut_duplicates.insert(v, (left, right));
            vert_to_nodes.insert(v, vec![left, right]);
        }

        // 3c. Boundary midpoint nodes.
        let mut midpoint_nodes: HashMap<EdgeID, NodeIndex> = HashMap::new();
        for edge_ref in skeleton.edges(node_idx) {
            let skel_edge = edge_ref.id();
            for &be in &edge_ref.weight().boundary_loop.edge_midpoints {
                let p0 = mesh.position(mesh.root(be));
                let p1 = mesh.position(mesh.toor(be));
                let pos = (p0 + p1) * 0.5;
                let ni = graph.add_node(VirtualNode {
                    position: pos,
                    origin: VirtualNodeOrigin::BoundaryMidpoint {
                        edge: be,
                        boundary: skel_edge,
                    },
                });
                midpoint_nodes.insert(be, ni);
            }
        }

        // ── Phase 4: Wire edges ───────────────────────────────────────────

        // Helper to add a weighted edge.
        let add_edge = |graph: &mut StableUnGraph<VirtualNode, VirtualEdgeWeight>,
                        a: NodeIndex,
                        b: NodeIndex| {
            let len = (graph[a].position - graph[b].position).norm();
            graph.add_edge(a, b, VirtualEdgeWeight { length: len });
        };

        // 4a. Mesh edges between patch vertices.
        // Process each patch vertex's outgoing half-edges, deduplicate by ordering.
        for &v in patch_verts {
            for nbr in mesh.neighbors(v) {
                if !patch_set.contains(&nbr) {
                    continue;
                }
                // Deduplicate: only process when v < nbr.
                if v >= nbr {
                    continue;
                }

                let v_is_cut = cut_vertex_set.contains(&v);
                let nbr_is_cut = cut_vertex_set.contains(&nbr);
                let is_cut_edge = {
                    let key = if v < nbr { (v, nbr) } else { (nbr, v) };
                    cut_edge_set.contains(&key)
                };

                if is_cut_edge {
                    // Cut edge: duplicate — one edge for left side, one for right.
                    let (v_left, v_right) = cut_duplicates[&v];
                    let (nbr_left, nbr_right) = cut_duplicates[&nbr];
                    add_edge(&mut graph, v_left, nbr_left);
                    add_edge(&mut graph, v_right, nbr_right);
                } else if v_is_cut && nbr_is_cut {
                    // Both are cut vertices but NOT on the same cut edge.
                    // This means they are on different cuts (cuts are vertex-disjoint).
                    // Each has a side assignment for the other.
                    let v_side = cut_side_assignments[&v][&nbr];
                    let nbr_side = cut_side_assignments[&nbr][&v];
                    let v_node = if v_side {
                        cut_duplicates[&v].1
                    } else {
                        cut_duplicates[&v].0
                    };
                    let nbr_node = if nbr_side {
                        cut_duplicates[&nbr].1
                    } else {
                        cut_duplicates[&nbr].0
                    };
                    add_edge(&mut graph, v_node, nbr_node);
                } else if v_is_cut {
                    let side = cut_side_assignments[&v][&nbr];
                    let v_node = if side {
                        cut_duplicates[&v].1
                    } else {
                        cut_duplicates[&v].0
                    };
                    add_edge(&mut graph, v_node, vert_to_nodes[&nbr][0]);
                } else if nbr_is_cut {
                    let side = cut_side_assignments[&nbr][&v];
                    let nbr_node = if side {
                        cut_duplicates[&nbr].1
                    } else {
                        cut_duplicates[&nbr].0
                    };
                    add_edge(&mut graph, vert_to_nodes[&v][0], nbr_node);
                } else {
                    // Neither is a cut vertex — simple edge.
                    add_edge(&mut graph, vert_to_nodes[&v][0], vert_to_nodes[&nbr][0]);
                }
            }
        }

        // 4b. Boundary midpoint edges.
        // Each boundary midpoint connects to the patch vertex(es) of its edge.
        for edge_ref in skeleton.edges(node_idx) {
            let boundary = &edge_ref.weight().boundary_loop;
            for &be in &boundary.edge_midpoints {
                let mid_node = midpoint_nodes[&be];
                let root = mesh.root(be);
                let toor = mesh.toor(be);

                // Connect to whichever endpoint(s) are in the patch.
                for &v in &[root, toor] {
                    if !patch_set.contains(&v) {
                        continue;
                    }
                    if cut_vertex_set.contains(&v) {
                        // Is this midpoint the cut's own start/end midpoint?
                        if let Some(&(ci, is_start)) = cut_endpoint_midpoints.get(&be) {
                            // This midpoint IS where the cut attaches to the boundary.
                            // It connects to BOTH copies of the cut endpoint vertex.
                            let (left, right) = cut_duplicates[&v];
                            add_edge(&mut graph, mid_node, left);
                            add_edge(&mut graph, mid_node, right);
                            let _ = (ci, is_start); // used only to identify
                        } else {
                            // Regular boundary midpoint adjacent to a cut vertex.
                            // Connect to the appropriate side based on fan assignment.
                            // The half-edge from v toward the other endpoint of be
                            // determines which side.
                            let other = if root == v { toor } else { root };
                            let side =
                                side_of_non_patch_neighbor(v, other, &cut_side_assignments, mesh);
                            let node = if side {
                                cut_duplicates[&v].1
                            } else {
                                cut_duplicates[&v].0
                            };
                            add_edge(&mut graph, mid_node, node);
                        }
                    } else {
                        add_edge(&mut graph, mid_node, vert_to_nodes[&v][0]);
                    }
                }
            }
        }

        // 4c. Direct edges between consecutive boundary midpoints.
        // These ensure the boundary loop can always chain through midpoints
        // without requiring a shared patch vertex between every pair.
        for edge_ref in skeleton.edges(node_idx) {
            let boundary = &edge_ref.weight().boundary_loop;
            let edges = &boundary.edge_midpoints;
            let n = edges.len();
            for i in 0..n {
                let a = midpoint_nodes[&edges[i]];
                let b = midpoint_nodes[&edges[(i + 1) % n]];
                add_edge(&mut graph, a, b);
            }
        }

        // ── Phase 5: Trace boundary loop ──────────────────────────────────

        let boundary_loop = trace_boundary_loop(
            node_idx,
            skeleton,
            cutting_plan,
            &cut_vertex_paths,
            &cut_duplicates,
            &midpoint_nodes,
            &cut_endpoint_midpoints,
        );

        let vfg = VirtualFlatGeometry {
            graph,
            vert_to_nodes,
            boundary_loop,
        };

        check_invariants(&vfg);

        vfg
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Cut side assignment
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// For each cut vertex, determines which of its mesh neighbors are on the
/// left (`false`) vs right (`true`) side of the cut.
///
/// Convention: "left" is the side containing the face to the left of the
/// directed cut path (start_boundary → end_boundary).
fn compute_all_cut_side_assignments(
    cuts: &[CutPath],
    cut_vertex_paths: &[Vec<VertID>],
    patch_set: &HashSet<VertID>,
    mesh: &Mesh<INPUT>,
) -> HashMap<VertID, HashMap<VertID, bool>> {
    let mut result: HashMap<VertID, HashMap<VertID, bool>> = HashMap::new();

    for (ci, vpath) in cut_vertex_paths.iter().enumerate() {
        if vpath.is_empty() {
            continue;
        }

        let cut = &cuts[ci];

        for (k, &v) in vpath.iter().enumerate() {
            // Find the two separator half-edges from v in the mesh edge fan.
            // The "prev" direction (toward start) and "next" direction (toward end).
            let he_prev = prev_separator(k, vpath, cut, mesh);
            let he_next = next_separator(k, vpath, cut, mesh);

            let sides = assign_fan_sides(v, he_prev, he_next, patch_set, mesh);
            result.insert(v, sides);
        }
    }

    result
}

/// Returns the half-edge from `vpath[k]` toward the predecessor on the cut.
/// For the first vertex (k=0), the predecessor is toward the boundary midpoint's
/// edge, so we return the half-edge from v0 toward the non-patch endpoint of
/// the start boundary edge.
fn prev_separator(k: usize, vpath: &[VertID], cut: &CutPath, mesh: &Mesh<INPUT>) -> EdgeID {
    let v = vpath[k];
    if k > 0 {
        // Half-edge from v toward the previous cut vertex.
        let prev_v = vpath[k - 1];
        let (e_v_to_prev, _) = mesh
            .edge_between_verts(v, prev_v)
            .unwrap_or_else(|| panic!("No edge between cut vertices {:?} and {:?}", v, prev_v));
        e_v_to_prev
    } else {
        // First vertex: separator toward the boundary midpoint.
        // The start boundary edge has v as one endpoint.
        let start_edge = match cut.path.points[0] {
            SurfacePoint::OnEdge { edge, .. } => edge,
            _ => panic!("Cut path first point is not OnEdge"),
        };
        // Find half-edge from v toward the other endpoint of the boundary edge.
        let other = if mesh.root(start_edge) == v {
            mesh.toor(start_edge)
        } else {
            assert_eq!(mesh.toor(start_edge), v, "v0 not on start boundary edge");
            mesh.root(start_edge)
        };
        let (he, _) = mesh
            .edge_between_verts(v, other)
            .unwrap_or_else(|| panic!("No edge from {:?} to boundary endpoint {:?}", v, other));
        he
    }
}

/// Returns the half-edge from `vpath[k]` toward the successor on the cut.
/// For the last vertex, the successor is toward the end boundary midpoint.
fn next_separator(k: usize, vpath: &[VertID], cut: &CutPath, mesh: &Mesh<INPUT>) -> EdgeID {
    let v = vpath[k];
    if k + 1 < vpath.len() {
        let next_v = vpath[k + 1];
        let (e_v_to_next, _) = mesh
            .edge_between_verts(v, next_v)
            .unwrap_or_else(|| panic!("No edge between cut vertices {:?} and {:?}", v, next_v));
        e_v_to_next
    } else {
        // Last vertex: separator toward the end boundary midpoint.
        let end_edge = match *cut.path.points.last().unwrap() {
            SurfacePoint::OnEdge { edge, .. } => edge,
            _ => panic!("Cut path last point is not OnEdge"),
        };
        let other = if mesh.root(end_edge) == v {
            mesh.toor(end_edge)
        } else {
            assert_eq!(mesh.toor(end_edge), v, "vN not on end boundary edge");
            mesh.root(end_edge)
        };
        let (he, _) = mesh
            .edge_between_verts(v, other)
            .unwrap_or_else(|| panic!("No edge from {:?} to boundary endpoint {:?}", v, other));
        he
    }
}

/// Splits the half-edge fan around `v` into left (false) and right (true)
/// using two separator half-edges. The "left" side is the arc starting at
/// `he_next` and going forward (in `mesh.edges(v)` cyclic order) until
/// `he_prev`. The "right" side is the remaining arc.
///
/// Returns a map: neighbor VertID -> side (bool). Only includes neighbors
/// that are in `patch_set` and are NOT the direct targets of the separators.
fn assign_fan_sides(
    v: VertID,
    he_prev: EdgeID,
    he_next: EdgeID,
    patch_set: &HashSet<VertID>,
    mesh: &Mesh<INPUT>,
) -> HashMap<VertID, bool> {
    let fan: Vec<EdgeID> = mesh.edges(v).collect();
    let n = fan.len();

    let i_prev = fan
        .iter()
        .position(|&e| e == he_prev)
        .unwrap_or_else(|| panic!("prev separator not in fan of {:?}", v));
    let i_next = fan
        .iter()
        .position(|&e| e == he_next)
        .unwrap_or_else(|| panic!("next separator not in fan of {:?}", v));

    let mut sides = HashMap::new();

    // Walk from i_next+1 to i_prev-1 (exclusive on separators): left side.
    let mut i = (i_next + 1) % n;
    while i != i_prev {
        let nbr = mesh.toor(fan[i]);
        if patch_set.contains(&nbr) {
            sides.insert(nbr, false); // left
        }
        i = (i + 1) % n;
    }

    // Walk from i_prev+1 to i_next-1 (exclusive on separators): right side.
    let mut i = (i_prev + 1) % n;
    while i != i_next {
        let nbr = mesh.toor(fan[i]);
        if patch_set.contains(&nbr) {
            sides.insert(nbr, true); // right
        }
        i = (i + 1) % n;
    }

    sides
}

/// Determines which side of a cut vertex a non-patch neighbor falls on.
/// Used for boundary midpoints whose other endpoint is outside the patch.
///
/// Finds the half-edge from `cut_vertex` toward `non_patch_nbr` in the fan,
/// and returns whichever side that position falls on.
fn side_of_non_patch_neighbor(
    cut_vertex: VertID,
    non_patch_nbr: VertID,
    cut_side_assignments: &HashMap<VertID, HashMap<VertID, bool>>,
    mesh: &Mesh<INPUT>,
) -> bool {
    // The cut_side_assignments only has patch neighbors. For a non-patch neighbor,
    // we find the fan position and check which arc it falls in.
    // We can determine this by looking at the face: the half-edge from cut_vertex
    // to non_patch_nbr shares a face with some patch neighbor. Return that
    // neighbor's side.
    let (he_to_other, _) = mesh
        .edge_between_verts(cut_vertex, non_patch_nbr)
        .unwrap_or_else(|| {
            panic!(
                "No edge between cut vertex {:?} and non-patch neighbor {:?}",
                cut_vertex, non_patch_nbr
            )
        });

    // The face of he_to_other has other vertices. Find a patch neighbor in that face
    // that has a side assignment.
    let sides = &cut_side_assignments[&cut_vertex];

    // Check both faces incident to the edge between cut_vertex and non_patch_nbr.
    // The half-edge he_to_other belongs to one face, its twin to the other.
    for &he in &[he_to_other, mesh.twin(he_to_other)] {
        let face_verts: Vec<VertID> = mesh.vertices(mesh.face(he)).collect();
        for &fv in &face_verts {
            if fv != cut_vertex && fv != non_patch_nbr {
                if let Some(&side) = sides.get(&fv) {
                    return side;
                }
            }
        }
    }

    // Fallback: walk the fan from he_to_other in both directions to find
    // the nearest patch neighbor with a known side.
    let fan: Vec<EdgeID> = mesh.edges(cut_vertex).collect();
    let n = fan.len();
    let start = fan
        .iter()
        .position(|&e| e == he_to_other)
        .unwrap_or_else(|| panic!("half-edge not in fan"));

    // Search forward.
    for step in 1..n {
        let idx = (start + step) % n;
        let nbr = mesh.toor(fan[idx]);
        if let Some(&side) = sides.get(&nbr) {
            return side;
        }
    }

    panic!(
        "Could not determine side for non-patch neighbor {:?} of cut vertex {:?}",
        non_patch_nbr, cut_vertex
    );
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Boundary loop tracing
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Traces the single boundary loop of the VFG by traversing boundary chains
/// and splicing in cut paths via DFS on the cut spanning tree.
///
/// The boundary of the disk is formed by walking each original boundary loop,
/// and at cut endpoints, detouring along the cut to the adjacent boundary,
/// traversing it, and returning along the other side of the cut.
#[allow(clippy::too_many_arguments)]
fn trace_boundary_loop(
    node_idx: NodeIndex,
    skeleton: &LabeledCurveSkeleton,
    cutting_plan: &CuttingPlan,
    cut_vertex_paths: &[Vec<VertID>],
    cut_duplicates: &HashMap<VertID, (NodeIndex, NodeIndex)>,
    midpoint_nodes: &HashMap<EdgeID, NodeIndex>,
    cut_endpoint_midpoints: &HashMap<EdgeID, (usize, bool)>,
) -> Vec<NodeIndex> {
    if cutting_plan.cuts.is_empty() {
        // No cuts: just one boundary loop (degree 1).
        // Walk the single boundary loop and return its VFG node sequence.
        let edge_ref = skeleton
            .edges(node_idx)
            .next()
            .expect("degree >= 1 but no edges");
        let boundary = &edge_ref.weight().boundary_loop;
        return build_boundary_chain(boundary, midpoint_nodes);
    }

    // Build adjacency for the cut spanning tree.
    // Nodes = skeleton edges (boundary loops). Tree edges = cuts.
    let mut tree_adj: HashMap<EdgeIndex, Vec<(usize, EdgeIndex)>> = HashMap::new();
    for (ci, cut) in cutting_plan.cuts.iter().enumerate() {
        tree_adj
            .entry(cut.start_boundary)
            .or_default()
            .push((ci, cut.end_boundary));
        tree_adj
            .entry(cut.end_boundary)
            .or_default()
            .push((ci, cut.start_boundary));
    }

    // DFS traversal of cut tree, building the boundary loop.
    let start_boundary = cutting_plan
        .cuts
        .first()
        .map(|c| c.start_boundary)
        .unwrap_or_else(|| skeleton.edges(node_idx).next().unwrap().id());

    let mut visited: HashSet<EdgeIndex> = HashSet::new();
    let mut result: Vec<NodeIndex> = Vec::new();

    dfs_trace_boundary(
        start_boundary,
        None,
        skeleton,
        cutting_plan,
        cut_vertex_paths,
        cut_duplicates,
        midpoint_nodes,
        cut_endpoint_midpoints,
        &tree_adj,
        &mut visited,
        &mut result,
    );

    result
}

/// DFS on cut spanning tree. At each boundary loop, walks the boundary chain,
/// and at each cut endpoint midpoint that leads to an unvisited boundary,
/// splices in: left cut path → recurse into other boundary → right cut path (reversed).
#[allow(clippy::too_many_arguments)]
fn dfs_trace_boundary(
    boundary_idx: EdgeIndex,
    entry_cut: Option<(usize, bool)>, // (cut_index, entered_via_start_of_cut)
    skeleton: &LabeledCurveSkeleton,
    cutting_plan: &CuttingPlan,
    cut_vertex_paths: &[Vec<VertID>],
    cut_duplicates: &HashMap<VertID, (NodeIndex, NodeIndex)>,
    midpoint_nodes: &HashMap<EdgeID, NodeIndex>,
    cut_endpoint_midpoints: &HashMap<EdgeID, (usize, bool)>,
    tree_adj: &HashMap<EdgeIndex, Vec<(usize, EdgeIndex)>>,
    visited: &mut HashSet<EdgeIndex>,
    result: &mut Vec<NodeIndex>,
) {
    visited.insert(boundary_idx);

    let boundary = &skeleton
        .edge_weight(boundary_idx)
        .expect("skeleton edge missing")
        .boundary_loop;

    // Build the boundary chain: ordered list of (VFG_node, Option<cut to splice>).
    let chain = build_boundary_chain_with_cuts(boundary, midpoint_nodes, cut_endpoint_midpoints);

    // Determine where in the chain to start. If we entered via a cut, start
    // right after that cut's midpoint on this boundary.
    let start_pos = if let Some((entry_ci, entered_via_start)) = entry_cut {
        let cut = &cutting_plan.cuts[entry_ci];
        let entry_midpoint_edge = if entered_via_start {
            // We entered via the start of the cut, so on this boundary the
            // cut's start midpoint is where we arrived.
            match cut.path.points[0] {
                SurfacePoint::OnEdge { edge, .. } => edge,
                _ => unreachable!(),
            }
        } else {
            match *cut.path.points.last().unwrap() {
                SurfacePoint::OnEdge { edge, .. } => edge,
                _ => unreachable!(),
            }
        };
        let entry_mid_node = midpoint_nodes[&entry_midpoint_edge];
        // Find this node in the chain and start AFTER it.
        chain
            .iter()
            .position(|(n, _)| *n == entry_mid_node)
            .map(|p| (p + 1) % chain.len())
            .unwrap_or(0)
    } else {
        0
    };

    let n = chain.len();
    for step in 0..n {
        let idx = (start_pos + step) % n;
        let (vfg_node, ref cut_info) = chain[idx];

        // Check if this node is a cut endpoint midpoint leading to an unvisited boundary.
        if let Some(&(ci, is_start)) = cut_info.as_ref() {
            let cut = &cutting_plan.cuts[ci];
            let other_boundary = if is_start {
                cut.end_boundary
            } else {
                cut.start_boundary
            };

            if !visited.contains(&other_boundary) {
                // Splice: emit left-side cut nodes → recurse → emit right-side (reversed).
                let vpath = &cut_vertex_paths[ci];

                // Left side: from this boundary toward the other.
                if is_start {
                    // Cut goes start → end. Left side forward.
                    result.push(vfg_node); // the midpoint
                    for &cv in vpath {
                        result.push(cut_duplicates[&cv].0); // left
                    }
                } else {
                    // Cut goes start → end, but we are at the end boundary.
                    // Left side goes end → start (reversed direction).
                    result.push(vfg_node); // the midpoint
                    for &cv in vpath.iter().rev() {
                        result.push(cut_duplicates[&cv].0); // left
                    }
                }

                // Recurse into the other boundary.
                let entered_via = (ci, !is_start);
                dfs_trace_boundary(
                    other_boundary,
                    Some(entered_via),
                    skeleton,
                    cutting_plan,
                    cut_vertex_paths,
                    cut_duplicates,
                    midpoint_nodes,
                    cut_endpoint_midpoints,
                    tree_adj,
                    visited,
                    result,
                );

                // Right side: return from the other boundary (reversed).
                if is_start {
                    for &cv in vpath.iter().rev() {
                        result.push(cut_duplicates[&cv].1); // right
                    }
                } else {
                    for &cv in vpath {
                        result.push(cut_duplicates[&cv].1); // right
                    }
                }
                // Do NOT re-emit vfg_node (the midpoint) — it was already emitted above.
                continue;
            }
        }

        // Regular boundary node (or a cut endpoint leading to an already-visited boundary).
        result.push(vfg_node);
    }
}

/// Builds an ordered chain of VFG nodes along a boundary loop, annotating
/// each node with an optional cut splice point.
///
/// Returns Vec of (VFG NodeIndex, Option<(cut_index, is_start)>).
/// Only the boundary midpoint at a cut endpoint gets the annotation.
#[allow(clippy::too_many_arguments)]
fn build_boundary_chain_with_cuts(
    boundary: &BoundaryLoop,
    midpoint_nodes: &HashMap<EdgeID, NodeIndex>,
    cut_endpoint_midpoints: &HashMap<EdgeID, (usize, bool)>,
) -> Vec<(NodeIndex, Option<(usize, bool)>)> {
    boundary
        .edge_midpoints
        .iter()
        .map(|be| {
            let mid_node = midpoint_nodes[be];
            let cut_info = cut_endpoint_midpoints.get(be).copied();
            (mid_node, cut_info)
        })
        .collect()
}

/// Simplified version for degree-1 regions (no cuts to splice).
fn build_boundary_chain(
    boundary: &BoundaryLoop,
    midpoint_nodes: &HashMap<EdgeID, NodeIndex>,
) -> Vec<NodeIndex> {
    boundary
        .edge_midpoints
        .iter()
        .map(|be| midpoint_nodes[be])
        .collect()
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Invariant checks
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Checks structural invariants on the completed VFG.
fn check_invariants(vfg: &VirtualFlatGeometry) {
    let boundary_set: HashSet<NodeIndex> = vfg.boundary_loop.iter().copied().collect();
    let n_nodes = vfg.graph.node_count();

    // 1. Boundary loop is non-empty.
    assert!(!vfg.boundary_loop.is_empty(), "VFG boundary loop is empty");

    // 2. Boundary loop is a simple cycle (no repeated nodes).
    if boundary_set.len() != vfg.boundary_loop.len() {
        // Failure! Now find which nodes appear multiple times.
        let mut counts: HashMap<NodeIndex, usize> = HashMap::new();
        for &n in &vfg.boundary_loop {
            *counts.entry(n).or_insert(0) += 1;
        }
        for (&node, &count) in &counts {
            if count > 1 {
                log::error!(
                    "VFG boundary duplicate: node {:?} appears {} times, origin: {:?}",
                    node,
                    count,
                    vfg.graph[node].origin,
                );
                // Show positions in the loop
                let positions: Vec<usize> = vfg
                    .boundary_loop
                    .iter()
                    .enumerate()
                    .filter(|(_, &n)| n == node)
                    .map(|(i, _)| i)
                    .collect();
                log::error!("  at loop positions: {:?}", positions);
            }
        }
        panic!(
            "VFG boundary loop has repeated nodes: {} unique out of {} total",
            boundary_set.len(),
            vfg.boundary_loop.len(),
        );
    }

    // 3. Every boundary node exists in the graph.
    for &node in &vfg.boundary_loop {
        assert!(
            vfg.graph.node_weight(node).is_some(),
            "VFG boundary references non-existent node {:?}",
            node,
        );
    }

    // 4. Degree checks.
    for node in vfg.graph.node_indices() {
        let degree = vfg.graph.edges(node).count();
        let is_boundary = boundary_set.contains(&node);

        if is_boundary {
            assert!(
                degree >= 1,
                "VFG invariant violated: boundary node {:?} ({:?}) has 0 neighbours",
                node,
                vfg.graph[node].origin,
            );
        } else {
            assert!(
                degree >= 3,
                "VFG invariant violated: interior node {:?} ({:?}) has {} neighbours, expected >= 3",
                node,
                vfg.graph[node].origin,
                degree
            );
        }
    }

    // 5. Every node in the graph is either in the boundary or has degree >= 3.
    //    (Covered by check 4.)

    // 6. Cut duplicate pairs have matching peers.
    for node in vfg.graph.node_indices() {
        if let VirtualNodeOrigin::CutDuplicate {
            peer: Some(peer), ..
        } = vfg.graph[node].origin
        {
            assert!(
                vfg.graph.node_weight(peer).is_some(),
                "Cut duplicate {:?} has peer {:?} that doesn't exist",
                node,
                peer,
            );
            if let VirtualNodeOrigin::CutDuplicate {
                peer: Some(peer_of_peer),
                ..
            } = vfg.graph[peer].origin
            {
                assert_eq!(
                    peer_of_peer, node,
                    "Cut duplicate peer mismatch: {:?} -> {:?} -> {:?}",
                    node, peer, peer_of_peer,
                );
            } else {
                panic!(
                    "Peer {:?} of cut duplicate {:?} is not a CutDuplicate",
                    peer, node
                );
            }
        }
    }

    // 7. vert_to_nodes consistency.
    for (vert, nodes) in &vfg.vert_to_nodes {
        assert!(!nodes.is_empty(), "vert_to_nodes[{:?}] is empty", vert);
        assert!(
            nodes.len() <= 2,
            "vert_to_nodes[{:?}] has {} entries (expected 1 or 2)",
            vert,
            nodes.len()
        );
        for &ni in nodes {
            assert!(
                vfg.graph.node_weight(ni).is_some(),
                "vert_to_nodes[{:?}] references non-existent node {:?}",
                vert,
                ni,
            );
        }
    }

    // 8. All graph nodes are accounted for in vert_to_nodes or midpoint_nodes.
    let tracked_nodes: HashSet<NodeIndex> = vfg
        .vert_to_nodes
        .values()
        .flat_map(|v| v.iter().copied())
        .collect();
    let midpoint_count = vfg
        .graph
        .node_indices()
        .filter(|n| {
            matches!(
                vfg.graph[*n].origin,
                VirtualNodeOrigin::BoundaryMidpoint { .. }
            )
        })
        .count();
    let vertex_node_count = tracked_nodes.len();
    assert_eq!(
        vertex_node_count + midpoint_count,
        n_nodes,
        "Node accounting mismatch: {} vertex nodes + {} midpoint nodes != {} total",
        vertex_node_count,
        midpoint_count,
        n_nodes,
    );

    // TODO: duplicated nodes do not share neighbor vertices?
}
