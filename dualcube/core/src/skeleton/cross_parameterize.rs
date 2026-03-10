use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};

use log::{error, warn};
use mehsh::prelude::{HasNeighbors, HasPosition, Mesh, Vector2D};
use ordered_float::OrderedFloat;
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};

use crate::prelude::{VertID, INPUT};
use crate::skeleton::boundary_loop::BoundaryLoop;
use crate::skeleton::orthogonalize::LabeledCurveSkeleton;

/// The parameterization of a single region (skeleton node) onto a canonical 2D domain.
///
/// Both the input mesh region and the polycube mesh region are mapped to the same
/// canonical domain. The composition of these two maps gives the bijection
/// between the input and polycube surfaces for this region.
///
/// The canonical domains are:
/// - degree 0: a sphere (??? TODO)
/// - degree 1: a square, the single boundary maps to the entire boundary of the square
/// - degree 2+: a regular 4(d-1) gon for degree d.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionParameterization {
    /// For each input mesh vertex in this region, its 2D position in the canonical domain.
    pub input_to_canonical: HashMap<VertID, Vector2D>,

    /// For each polycube mesh vertex in this region, its 2D position in the canonical domain.
    /// NOTE: Keys are VertKey<POLYCUBE> stored as VertID via raw key, same convention as
    /// `SkeletonNode::patch_vertices` on the polycube skeleton.
    pub polycube_to_canonical: HashMap<VertID, Vector2D>,
}

/// A bijection between the input mesh surface and the polycube surface,
/// represented as a collection of per-region parameterizations through canonical domains.
///
/// For any input mesh vertex, the map gives a polycube-surface position (and vice versa)
/// by composing: input surface <-> canonical domain <-> polycube surface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolycubeMap {
    /// Per skeleton-node parameterization. The `NodeIndex` keys are valid for both
    /// the input and polycube `LabeledCurveSkeleton` (skeleton-graphs are isomorphic).
    pub regions: HashMap<NodeIndex, RegionParameterization>,
}

/// Describes how to cut a region with multiple boundary loops to disk topology.
///
/// For degree d ≥ 2, we need d-1 cuts forming a spanning tree over the d boundary loops.
/// Each cut is identified by the pair of skeleton `EdgeIndex`es whose boundary loops
/// it connects. The resulting disk has a single boundary, and the canonical domain
/// is a regular 4(d-1)-gon.
///
/// For degree 0 or 1, no cuts are needed (`cuts` is empty).
#[derive(Debug, Clone)]
pub struct CuttingPlan {
    /// Each entry `(edge_a, edge_b)` means: cut from the boundary loop on skeleton
    /// edge `edge_a` to the one on `edge_b`. There are d-1 such cuts.
    pub cuts: Vec<(EdgeIndex, EdgeIndex)>,
}

impl PolycubeMap {
    /// Constructs a `Mesh<INPUT>` whose vertices are the input mesh vertices repositioned
    /// onto the polycube surface. This is the `triangle_mesh_polycube` needed by `Quad`.
    ///
    /// For each input vertex, looks up its canonical-domain coordinates from the input
    /// parameterization, then finds the corresponding polycube-surface position by
    /// interpolating within the polycube parameterization of the same region.
    pub fn to_triangle_mesh_polycube(
        &self,
        input_mesh: &Mesh<INPUT>,
        polycube_skeleton: &LabeledCurveSkeleton,
    ) -> Mesh<INPUT> {
        // TODO: For each region:
        //   1. Clone the input mesh
        //   2. For each input vertex in the region, get its canonical (u, v) coords
        //   3. Find the triangle in the polycube parameterization containing that (u, v)
        //   4. Interpolate to get the 3D polycube-surface position
        //   5. set_position on the cloned mesh
        // NOTE: Step 3-4 requires triangulating the polycube parameterization and building
        //       a point-location structure. A simple approach is barycentric interpolation
        //       over the polycube triangulation in 2D canonical space.
        Mesh::default()
    }
}

/// Performs cross-parameterization between the input and polycube labeled curve skeletons,
/// by mapping all regions individually to a shared canonical domain.
///
/// Both skeletons must be isomorphic (same graph topology, same `NodeIndex`/`EdgeIndex` values).
/// Each region (node) is independently parameterized onto a shared domain,
/// producing a per-vertex 2D coordinate for both the input and polycube sides.
pub fn cross_parameterize(
    input_skeleton: &LabeledCurveSkeleton,
    polycube_skeleton: &LabeledCurveSkeleton,
    input_mesh: &Mesh<INPUT>,
    polycube_mesh: &Mesh<INPUT>, // Polycube mesh passed as Mesh<INPUT> via raw key conversion
) -> PolycubeMap {
    let mut regions = HashMap::new();

    for node_idx in input_skeleton.node_indices() {
        let degree = input_skeleton.edges(node_idx).count();

        let region = parameterize_region(
            node_idx,
            degree,
            input_skeleton,
            polycube_skeleton,
            input_mesh,
            polycube_mesh,
        );

        regions.insert(node_idx, region);
    }

    PolycubeMap { regions }
}

/// Parameterizes a single region (node) on both the input and polycube side onto a shared canonical domain.
fn parameterize_region(
    node_idx: NodeIndex,
    degree: usize,
    input_skeleton: &LabeledCurveSkeleton,
    polycube_skeleton: &LabeledCurveSkeleton,
    input_mesh: &Mesh<INPUT>,
    polycube_mesh: &Mesh<INPUT>,
) -> RegionParameterization {
    // Compute a shared cutting plan using combined geodesic distances from both sides.
    // This determines *which* boundary loops to connect (the topology of the cuts).
    // The actual cut paths (the *how*) are computed later per-side in parameterize_side.
    let cutting_plan = compute_cutting_plan(
        node_idx,
        input_skeleton,
        polycube_skeleton,
        input_mesh,
        polycube_mesh,
    );

    // TEMP DEBUG LOG
    warn!(
        "Node {:?} (degree {}): cutting plan has {} cuts: {:?}",
        node_idx,
        degree,
        cutting_plan.cuts.len(),
        cutting_plan.cuts
    );

    // Parameterize each side independently using the shared cutting plan.
    let input_to_canonical =
        parameterize_side(node_idx, degree, input_skeleton, input_mesh, &cutting_plan);

    let polycube_to_canonical =
        parameterize_side(node_idx, degree, polycube_skeleton, polycube_mesh, &cutting_plan);

    RegionParameterization {
        input_to_canonical,
        polycube_to_canonical,
    }
}

/// Parameterizes one side (input or polycube) of a region onto the canonical domain.
///
/// Uses the cutting plan to determine cut topology, then:
/// 1. Finds actual cut paths on this mesh surface
/// 2. Assembles the full disk boundary (boundary arcs + cut paths)
/// 3. Arc-length parameterizes boundary onto the 4(d-1)-gon (or square for d=1) // TODO: sphere for d=1?
/// 4. Solves Dirichlet problem for interior vertices
///
/// Returns a map from vertex ID to 2D canonical-domain position.
fn parameterize_side(
    node_idx: NodeIndex,
    degree: usize,
    skeleton: &LabeledCurveSkeleton,
    mesh: &Mesh<INPUT>,
    cutting_plan: &CuttingPlan,
) -> HashMap<VertID, Vector2D> {
    // TODO:
    // 1. Use cutting_plan.cuts to find actual geodesic cut paths on this mesh,
    //    within the region (from one boundary loop to another).
    //    NOTE: endpoint separation — if a boundary has multiple cuts landing on it,
    //    spread the endpoints around the boundary (not all at the same point).
    //
    // 2. Walk the disk boundary: alternate between boundary-loop arcs and cut-path
    //    segments. The boundary-loop arcs are split where cuts land.
    //
    // 3. Map the disk boundary to the canonical 4(d-1)-gon via arc-length
    //    parameterization per segment.
    //
    // 4. Solve 2D Dirichlet (solve_harmonic_2d) for interior vertices.

    HashMap::new()
}

/// Solves a 2D Dirichlet problem on a surface mesh region.
///
/// Given a set of vertices, some with fixed 2D positions (boundary) and the rest free (interior),
/// solves the discrete Laplace equation to find 2D positions for the free vertices.
///
/// Uses uniform graph Laplacian weights and direct Cholesky factorization,
/// same as `solve_harmonic_scalar_field` but for 2D coordinates.
fn solve_harmonic_2d(
    all_vertices: &[VertID],
    boundary_positions: &HashMap<VertID, Vector2D>,
    mesh: &Mesh<INPUT>,
) -> HashMap<VertID, Vector2D> {
    // TODO: Essentially solve_harmonic_scalar_field twice (once for u, once for v)
    // but with boundary values not just 0 or 1, but 2d positions (so 0<=x,y<=1, though depending on the domain shape more can be cut off)
    HashMap::new()
}


/// Computes a shared cutting plan for a region, combining geodesic distances from
/// both the input and polycube sides to select an MST over the boundary loops.
///
/// For degree < 2, returns an empty plan (no cuts needed).
/// For degree >= 2, returns d-1 cuts forming a minimum spanning tree.
pub fn compute_cutting_plan(
    node_idx: NodeIndex,
    input_skeleton: &LabeledCurveSkeleton,
    polycube_skeleton: &LabeledCurveSkeleton,
    input_mesh: &Mesh<INPUT>,
    polycube_mesh: &Mesh<INPUT>,
) -> CuttingPlan {
    let degree = input_skeleton.edges(node_idx).count();

    if degree < 2 {
        return CuttingPlan { cuts: Vec::new() };
    }

    // Compute pairwise shortest-path distances between boundary loops on each side.
    let input_distances = pairwise_boundary_distances(node_idx, input_skeleton, input_mesh);
    let polycube_distances = pairwise_boundary_distances(node_idx, polycube_skeleton, polycube_mesh);

    // Normalize each side by its maximum finite distance, so both contribute on [0, 1].
    let input_max = input_distances
        .values()
        .copied()
        .filter(|d| d.is_finite())
        .fold(0.0_f64, f64::max);
    let polycube_max = polycube_distances
        .values()
        .copied()
        .filter(|d| d.is_finite())
        .fold(0.0_f64, f64::max);

    // Build combined cost for every boundary pair.
    let mut weighted_edges: Vec<(EdgeIndex, EdgeIndex, f64)> = Vec::new();
    for (&(a, b), &d_input) in &input_distances {
        let d_polycube = polycube_distances.get(&(a, b)).copied().unwrap_or(f64::INFINITY);

        let norm_input = if input_max > 0.0 {
            d_input / input_max
        } else {
            0.0
        };
        let norm_polycube = if polycube_max > 0.0 {
            d_polycube / polycube_max
        } else {
            0.0
        };

        weighted_edges.push((a, b, norm_input + norm_polycube));
    }

    let mst = kruskal_mst(&weighted_edges);

    if mst.len() != degree - 1 {
        error!(
            "Cutting plan for node {:?}: expected {} cuts but MST has {}.",
            node_idx,
            degree - 1,
            mst.len()
        );
    }

    CuttingPlan { cuts: mst }
}



/// Returns the set of vertices *within this region* that lie on a specific boundary loop.
///
/// Each half-edge in the boundary loop crosses between two patches; we keep the
/// endpoint that belongs to our patch.
fn boundary_vertices_of_region(
    boundary_loop: &BoundaryLoop,
    patch_set: &HashSet<VertID>,
    mesh: &Mesh<INPUT>,
) -> HashSet<VertID> {
    let mut result = HashSet::new();
    for &e in &boundary_loop.edge_midpoints {
        let r = mesh.root(e);
        let t = mesh.toor(e);
        if patch_set.contains(&r) {
            result.insert(r);
        }
        if patch_set.contains(&t) {
            result.insert(t);
        }
    }
    result
}

/// Computes pairwise shortest geodesic distances between all boundary loops of a
/// region on one mesh side.
///
/// Returns a map from canonically-ordered `(EdgeIndex, EdgeIndex)` pairs (smaller
/// index first) to the minimum vertex-to-vertex distance through the region interior. 
fn pairwise_boundary_distances(
    node_idx: NodeIndex,
    skeleton: &LabeledCurveSkeleton,
    mesh: &Mesh<INPUT>,
) -> HashMap<(EdgeIndex, EdgeIndex), f64> {
    let patch_verts = &skeleton[node_idx].skeleton_node.patch_vertices;
    let patch_set: HashSet<VertID> = patch_verts.iter().copied().collect();

    // Build a vertex-to-index map for the region (used by Dijkstra).
    let vert_to_idx: HashMap<VertID, usize> = patch_verts
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();

    // Collect each incident edge's boundary vertices.
    let mut boundary_info: Vec<(EdgeIndex, HashSet<VertID>)> = Vec::new();
    for edge_ref in skeleton.edges(node_idx) {
        let bverts =
            boundary_vertices_of_region(&edge_ref.weight().boundary_loop, &patch_set, mesh);
        boundary_info.push((edge_ref.id(), bverts));
    }

    // Run one multi-source Dijkstra per boundary.
    let all_dists: Vec<Vec<f64>> = boundary_info
        .iter()
        .map(|(_, bverts)| restricted_dijkstra(bverts, patch_verts, &vert_to_idx, mesh))
        .collect();

    // Extract pairwise minimum distances.
    let mut result = HashMap::new();
    for i in 0..boundary_info.len() {
        for j in (i + 1)..boundary_info.len() {
            let (eid_i, _) = &boundary_info[i];
            let (eid_j, bverts_j) = &boundary_info[j];

            let min_dist = bverts_j
                .iter()
                .filter_map(|v| vert_to_idx.get(v).map(|&idx| all_dists[i][idx]))
                .fold(f64::INFINITY, f64::min);

            // Store with canonical order (smaller EdgeIndex first).
            let key = if *eid_i < *eid_j {
                (*eid_i, *eid_j)
            } else {
                (*eid_j, *eid_i)
            };
            result.insert(key, min_dist);
        }
    }

    result
}

/// Multi-source Dijkstra restricted to vertices within the region.
///
/// Returns a distance vector indexed by position in `region_verts`.
/// Unreachable vertices keep their initial inf value, should not happen.
fn restricted_dijkstra(
    sources: &HashSet<VertID>,
    region_verts: &[VertID],
    vert_to_idx: &HashMap<VertID, usize>,
    mesh: &Mesh<INPUT>,
) -> Vec<f64> {
    let n = region_verts.len();
    let mut dist = vec![f64::INFINITY; n];
    let mut heap: BinaryHeap<Reverse<(OrderedFloat<f64>, usize)>> = BinaryHeap::new();

    for &s in sources {
        if let Some(&idx) = vert_to_idx.get(&s) {
            dist[idx] = 0.0;
            heap.push(Reverse((OrderedFloat(0.0), idx)));
        }
    }

    while let Some(Reverse((OrderedFloat(d), u_idx))) = heap.pop() {
        if d > dist[u_idx] {
            continue; // stale entry
        }
        let u = region_verts[u_idx];
        for nbr in mesh.neighbors(u) {
            // Only traverse edges within the region.
            let Some(&nbr_idx) = vert_to_idx.get(&nbr) else {
                continue;
            };
            let edge_len = (mesh.position(u) - mesh.position(nbr)).norm();
            let new_dist = d + edge_len;
            if new_dist < dist[nbr_idx] {
                dist[nbr_idx] = new_dist;
                heap.push(Reverse((OrderedFloat(new_dist), nbr_idx)));
            }
        }
    }

    dist
}

/// Kruskal's MST on a small complete graph of boundary loops.
///
/// Takes a list of `(node_a, node_b, cost)` triples (where "nodes" are skeleton
/// `EdgeIndex` values identifying boundary loops) and returns the MST edges.
fn kruskal_mst(weighted_edges: &[(EdgeIndex, EdgeIndex, f64)]) -> Vec<(EdgeIndex, EdgeIndex)> {
    let mut sorted: Vec<_> = weighted_edges.to_vec();
    sorted.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));

    // Union-find with path compression.
    let mut parent: HashMap<EdgeIndex, EdgeIndex> = HashMap::new();

    fn find(parent: &mut HashMap<EdgeIndex, EdgeIndex>, x: EdgeIndex) -> EdgeIndex {
        let p = *parent.entry(x).or_insert(x);
        if p == x {
            return x;
        }
        let root = find(parent, p);
        parent.insert(x, root);
        root
    }

    let mut result = Vec::new();
    for &(a, b, cost) in &sorted {
        if !cost.is_finite() {
            continue;
        }
        let ra = find(&mut parent, a);
        let rb = find(&mut parent, b);
        if ra != rb {
            parent.insert(ra, rb);
            result.push((a, b));
        }
    }

    result
}
