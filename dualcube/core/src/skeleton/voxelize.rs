use std::collections::{HashMap, HashSet};

use bimap::BiHashMap;
use log::{error, info, warn};
use mehsh::prelude::{EdgeKey, FaceKey, HasEdges, HasVertices, Mesh, Vector3D, VertKey};
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;

use crate::polycube::POLYCUBE;
use crate::prelude::{Polycube, PrincipalDirection, VertID, EdgeID, INPUT};
use crate::quad::Quad;
use crate::skeleton::boundary_loop::BoundaryLoop;
use crate::skeleton::orthogonalize::{IVector3D, LabeledCurveSkeleton};

/// We keep track of which voxel belongs to which node/edge, to know where the regions belong.
pub enum VoxelOwner {
    /// A voxel that corresponds directly to a graph node.
    Node(NodeIndex),

    /// A voxel that correspond to 1 node entirely.
    Edge(NodeIndex),

    /// A voxel that sits at the midpoint of a graph edge, splitting ownership.
    /// Stores (source node, target node, unit direction from source toward target).
    EdgeMidpoint(NodeIndex, NodeIndex, IVector3D),
}

// TODO: integrate into type?
const ONE: IVector3D = IVector3D::new(1, 1, 1);
const X: IVector3D = IVector3D::new(1, 0, 0);
const Y: IVector3D = IVector3D::new(0, 1, 0);
const Z: IVector3D = IVector3D::new(0, 0, 1);

/// Generates a polycube based on an orthogonalized skeleton, and a labeled skeleton with the same structure,
/// but with the regions on the polycube.
pub fn generate_polycube(skeleton: &LabeledCurveSkeleton, mut omega: usize) -> (Polycube, LabeledCurveSkeleton, Quad) {
    // Map voxels to the node/edge they belong to.
    let mut voxel_owners: HashMap<IVector3D, VoxelOwner> = HashMap::default();

    // We map every node from x,y,z to 2x-1,2y-1,2z-1, such that our voxelization has no faces against each other,
    // i.e., when there are now two voxels next to each other on the grid, we are sure these should be connected.
    for node_idx in skeleton.node_indices() {
        let node_weight = &skeleton[node_idx];
        let pos = node_weight.grid_position;
        let mapped_pos = (pos * 2) - ONE;
        voxel_owners.insert(mapped_pos, VoxelOwner::Node(node_idx));
    }

    // Place Edge Voxels
    for edge in skeleton.edge_references() {
        let source = edge.source();
        let source_pos = skeleton[source].grid_position;

        let target = edge.target();
        let target_pos = skeleton[target].grid_position;

        let edge_weight = edge.weight();
        let dir = edge_weight.direction;
        let original_length = edge_weight.length;

        let start_voxel = (source_pos * 2) - ONE;
        let end_voxel = (target_pos * 2) - ONE;

        let step_base = match dir {
            PrincipalDirection::X => X * 2,
            PrincipalDirection::Y => Y * 2,
            PrincipalDirection::Z => Z * 2,
        };

        let diff = end_voxel - start_voxel;
        let dir_vec = IVector3D::new(
            if diff.x == 0 {
                0
            } else {
                diff.x / diff.x.abs()
            },
            if diff.y == 0 {
                0
            } else {
                diff.y / diff.y.abs()
            },
            if diff.z == 0 {
                0
            } else {
                diff.z / diff.z.abs()
            },
        );

        let step = IVector3D::new(
            step_base.x * dir_vec.x,
            step_base.y * dir_vec.y,
            step_base.z * dir_vec.z,
        );

        // We use *2 scale, so a length 1 edge has 1 intermediate voxel (the bridge). Length 2 has 3 voxels, etc.
        let total_steps = original_length as i32 * 2;
        let mid_point = total_steps / 2;

        let mut pos = start_voxel;

        for i in 1..total_steps {
            pos += step / 2;

            // Determine Owner based on which side of the midpoint we are
            let owner = if i < mid_point {
                VoxelOwner::Edge(source)
            } else if i > mid_point {
                VoxelOwner::Edge(target)
            } else {
                VoxelOwner::EdgeMidpoint(source, target, dir_vec)
            };

            voxel_owners.insert(pos, owner);
        }
    }

    // Build polycube mesh vertices and faces, tracking which node owns each vertex index.
    let mut vertex_map: HashMap<IVector3D, usize> = HashMap::default();
    let mut vertex_owner_map: HashMap<usize, NodeIndex> = HashMap::default();
    let mut positions = Vec::<Vector3D>::new();
    let mut faces = Vec::<Vec<usize>>::new();

    for &voxel_pos in voxel_owners.keys() {
        let center = voxel_pos * 2;
        let owner = &voxel_owners[&voxel_pos];

        for (dir, face_offsets) in &DIRECTIONS {
            let neighbor_pos = voxel_pos + *dir;
            if voxel_owners.contains_key(&neighbor_pos) {
                continue;
            }

            let quad = face_offsets
                .iter()
                .map(|offset| {
                    let grid_pos = center + *offset;
                    let idx = *vertex_map.entry(grid_pos).or_insert_with(|| {
                        let idx = positions.len();
                        positions.push(Vector3D::new(
                            f64::from(grid_pos.x) / 2.,
                            f64::from(grid_pos.y) / 2.,
                            f64::from(grid_pos.z) / 2.,
                        ));
                        idx
                    });
                    // First non-midpoint voxel to touch this vertex claims ownership.
                    if !vertex_owner_map.contains_key(&idx) {
                        match owner {
                            VoxelOwner::Node(node) | VoxelOwner::Edge(node) => {
                                vertex_owner_map.insert(idx, *node);
                            }
                            VoxelOwner::EdgeMidpoint(..) => {}
                        }
                    }
                    idx
                })
                .collect::<Vec<_>>();
            faces.push(quad);
        }
    }

    // Second pass: assign vertices that belong exclusively to EdgeMidpoint voxels.
    // Each such vertex is placed on the source or target side based on its position
    // relative to the midpoint center along the edge direction.
    for (&voxel_pos, owner) in &voxel_owners {
        if let VoxelOwner::EdgeMidpoint(source, target, dir_vec) = owner {
            let center = voxel_pos * 2;
            for (dir, face_offsets) in &DIRECTIONS {
                let neighbor_pos = voxel_pos + *dir;
                if voxel_owners.contains_key(&neighbor_pos) {
                    continue; // internal face, no surface vertices here
                }
                for offset in face_offsets {
                    let grid_pos = center + *offset;
                    if let Some(&idx) = vertex_map.get(&grid_pos) {
                        if !vertex_owner_map.contains_key(&idx) {
                            let d = grid_pos - center;
                            let dot = d.x * dir_vec.x + d.y * dir_vec.y + d.z * dir_vec.z;
                            let node = if dot > 0 { *target } else { *source };
                            vertex_owner_map.insert(idx, node);
                        }
                    }
                }
            }
        }
    }

    let (mesh, vert_id_map, _): (Mesh<POLYCUBE>, _, _) = if faces.is_empty() {
        error!("Polycube mesh has no faces.");
        (Mesh::default(), Default::default(), Default::default())
    } else {
        Mesh::from(&faces, &positions)
            .expect("Failed to build voxel polycube mesh")
    };

    // Generate a quad mesh for the polycube
    if omega % 2 == 0 {
        warn!("Even omega values are currently not supported. Increasing by 1.");
        omega += 1;
    }
    let quad = Quad::from_polycube(&mesh, omega)
        .expect("Failed to generate quad mesh from polycube");

    let polycube_skeleton = generate_labeled_skeleton(skeleton, &mesh, &vert_id_map, vertex_owner_map);

    (
        Polycube {
            structure: mesh,
            region_to_vertex: BiHashMap::new(), // We do not have a dual (yet) so this has to be empty // TODO: create trivial dual
        },
        polycube_skeleton,
        quad,
    )
}

/// Generates an isomorphic skeleton with patches derived from the polycube mesh.
fn generate_labeled_skeleton(
    original: &LabeledCurveSkeleton,
    polycube_mesh: &Mesh<POLYCUBE>,
    vert_id_map: &mehsh::utils::ids::IdMap<mehsh::prelude::VERT, POLYCUBE>,
    vertex_owner_map: HashMap<usize, NodeIndex>,
) -> LabeledCurveSkeleton {
    let mut poly_skeleton = original.clone();

    // Use the IdMap from Mesh::from() for the canonical usize->VertKey mapping.
    let mut vertex_to_node: HashMap<VertKey<POLYCUBE>, NodeIndex> = HashMap::new();
    let mut node_patches: HashMap<NodeIndex, Vec<VertKey<POLYCUBE>>> = HashMap::new();
    for node_idx in poly_skeleton.node_indices() {
        node_patches.insert(node_idx, Vec::new());
    }

    for (&pos_idx, &node) in &vertex_owner_map {
        if let Some(&vert_key) = vert_id_map.key(pos_idx) {
            vertex_to_node.insert(vert_key, node);
            node_patches.entry(node).or_default().push(vert_key);
        }
    }

    // Update each skeleton node with polycube patch vertices and integer grid position.
    for node_idx in poly_skeleton.node_indices() {
        let patch = &node_patches[&node_idx];
        let grid_pos = original[node_idx].grid_position;
        let position = Vector3D::new(
            f64::from(grid_pos.x),
            f64::from(grid_pos.y),
            f64::from(grid_pos.z),
        );

        // Store VertKey<POLYCUBE> as VertID (= VertKey<INPUT>) via raw key conversion.
        let converted_patch: Vec<VertID> = patch.iter()
            .map(|&v| VertKey::<INPUT>::new(v.raw()))
            .collect();

        poly_skeleton[node_idx].skeleton_node.position = position;
        poly_skeleton[node_idx].skeleton_node.patch_vertices = converted_patch;
    }

    // Compute boundary loops for each skeleton edge.
    for edge_idx in poly_skeleton.edge_indices() {
        let (source, target) = poly_skeleton.edge_endpoints(edge_idx).unwrap();
        let boundary_loop = compute_quad_boundary_loop(polycube_mesh, &vertex_to_node, source, target);
        poly_skeleton.edge_weight_mut(edge_idx).unwrap().boundary_loop = boundary_loop;
    }

    // Checks
    let total_assigned: usize = poly_skeleton.node_indices()
        .map(|n| poly_skeleton[n].skeleton_node.patch_vertices.len())
        .sum();
    let total_verts = polycube_mesh.vert_ids().len();
    if total_assigned != total_verts {
        warn!(
            "Polycube skeleton: {} vertices assigned to patches, but mesh has {} vertices.",
            total_assigned, total_verts
        );
    }
    for edge_idx in poly_skeleton.edge_indices() {
        let loop_len = poly_skeleton.edge_weight(edge_idx).unwrap().boundary_loop.edge_midpoints.len();
        if loop_len != 4 {
            warn!(
                "Polycube skeleton edge {:?} has boundary loop of length {} (expected 4).",
                edge_idx, loop_len
            );
        }
    }

    info!(
        "Generated polycube skeleton: {} nodes, {} edges, {} mesh vertices assigned.",
        poly_skeleton.node_count(),
        poly_skeleton.edge_count(),
        total_assigned
    );

    poly_skeleton
}

/// Computes the boundary loop between two adjacent patches on the polycube quad mesh.
///
/// Collects "pure" boundary faces (all vertices belong to source or target, with both
/// present) and walks face-by-face to extract the ordered cycle of crossing half-edges.
fn compute_quad_boundary_loop(
    polycube_mesh: &Mesh<POLYCUBE>,
    vertex_to_node: &HashMap<VertKey<POLYCUBE>, NodeIndex>,
    source: NodeIndex,
    target: NodeIndex,
) -> BoundaryLoop {
    let source_verts: HashSet<VertKey<POLYCUBE>> = vertex_to_node.iter()
        .filter(|(_, &n)| n == source)
        .map(|(&v, _)| v)
        .collect();
    let target_verts: HashSet<VertKey<POLYCUBE>> = vertex_to_node.iter()
        .filter(|(_, &n)| n == target)
        .map(|(&v, _)| v)
        .collect();

    // Collect faces where all vertices are in {source, target} and both are present.
    let boundary_faces: HashSet<FaceKey<POLYCUBE>> = polycube_mesh.face_ids().iter().copied()
        .filter(|&face_id| {
            let mut has_source = false;
            let mut has_target = false;
            for v in polycube_mesh.vertices(face_id) {
                if source_verts.contains(&v) {
                    has_source = true;
                } else if target_verts.contains(&v) {
                    has_target = true;
                } else {
                    return false; // vertex from a third patch or unowned
                }
            }
            has_source && has_target
        })
        .collect();

    if boundary_faces.is_empty() {
        warn!("No boundary faces found between {:?} and {:?}.", source, target);
        return BoundaryLoop { edge_midpoints: Vec::new() };
    }

    // Find a starting crossing half-edge on a pure boundary face.
    let is_crossing = |e: EdgeKey<POLYCUBE>| -> bool {
        let u = polycube_mesh.root(e);
        let v = polycube_mesh.toor(e);
        (source_verts.contains(&u) && target_verts.contains(&v))
            || (target_verts.contains(&u) && source_verts.contains(&v))
    };

    let start = boundary_faces.iter()
        .flat_map(|&fid| polycube_mesh.edges(fid))
        .find(|&e| is_crossing(e));

    let Some(start) = start else {
        warn!("No crossing edge found between {:?} and {:?}.", source, target);
        return BoundaryLoop { edge_midpoints: Vec::new() };
    };

    // Walk the boundary: on each pure boundary face find the two crossing edges,
    // then cross to the adjacent face via twin.
    let mut loop_edges: Vec<EdgeKey<POLYCUBE>> = Vec::new();
    let mut current = start;

    loop {
        loop_edges.push(current);

        // Find the other crossing edge on the same face.
        let mut e = polycube_mesh.next(current);
        let mut found = None;
        for _ in 0..3 {
            if is_crossing(e) {
                found = Some(e);
                break;
            }
            e = polycube_mesh.next(e);
        }

        let Some(other) = found else {
            warn!("Boundary face has only one crossing edge; loop may be incomplete.");
            break;
        };

        let next = polycube_mesh.twin(other);
        if next == start {
            break;
        }

        // If the twin lands on a non-boundary face, stop gracefully.
        if !boundary_faces.contains(&polycube_mesh.face(next)) {
            warn!("Boundary loop left pure boundary faces; loop may be incomplete.");
            break;
        }

        current = next;

        if loop_edges.len() > boundary_faces.len() * 4 {
            error!("Boundary loop walk exceeded limit, aborting.");
            break;
        }
    }

    let converted: Vec<EdgeID> = loop_edges.iter()
        .map(|&e| EdgeKey::<INPUT>::new(e.raw()))
        .collect();

    BoundaryLoop { edge_midpoints: converted }
}

const DIRECTIONS: [(IVector3D, [IVector3D; 4]); 6] = [
    (
        IVector3D::new(1, 0, 0),
        [
            IVector3D::new(1, -1, -1),
            IVector3D::new(1, 1, -1),
            IVector3D::new(1, 1, 1),
            IVector3D::new(1, -1, 1),
        ],
    ),
    (
        IVector3D::new(-1, 0, 0),
        [
            IVector3D::new(-1, -1, -1),
            IVector3D::new(-1, -1, 1),
            IVector3D::new(-1, 1, 1),
            IVector3D::new(-1, 1, -1),
        ],
    ),
    (
        IVector3D::new(0, 1, 0),
        [
            IVector3D::new(-1, 1, -1),
            IVector3D::new(-1, 1, 1),
            IVector3D::new(1, 1, 1),
            IVector3D::new(1, 1, -1),
        ],
    ),
    (
        IVector3D::new(0, -1, 0),
        [
            IVector3D::new(-1, -1, -1),
            IVector3D::new(1, -1, -1),
            IVector3D::new(1, -1, 1),
            IVector3D::new(-1, -1, 1),
        ],
    ),
    (
        IVector3D::new(0, 0, 1),
        [
            IVector3D::new(-1, -1, 1),
            IVector3D::new(1, -1, 1),
            IVector3D::new(1, 1, 1),
            IVector3D::new(-1, 1, 1),
        ],
    ),
    (
        IVector3D::new(0, 0, -1),
        [
            IVector3D::new(-1, -1, -1),
            IVector3D::new(-1, 1, -1),
            IVector3D::new(1, 1, -1),
            IVector3D::new(1, -1, -1),
        ],
    ),
];
