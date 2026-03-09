use std::collections::HashMap;

use bimap::BiHashMap;
use log::{error, info};
use mehsh::prelude::{Mesh, Vector3D};
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;

use crate::polycube::POLYCUBE;
use crate::prelude::{Polycube, PrincipalDirection};
use crate::skeleton::orthogonalize::{IVector3D, LabeledCurveSkeleton};

/// We keep track of which voxel belongs to which node/edge, to know where the regions belong.
pub enum VoxelOwner {
    /// A voxel that corresponds directly to a graph node.
    Node(NodeIndex),

    /// A voxel that correspond to 1 node entirely.
    Edge(NodeIndex),

    /// A voxel that sits at the midpoint of a graph edge, splitting ownership.
    EdgeMidpoint(),
}

// TODO: integrate into type?
const ONE: IVector3D = IVector3D::new(1, 1, 1);
const X: IVector3D = IVector3D::new(1, 0, 0);
const Y: IVector3D = IVector3D::new(0, 1, 0);
const Z: IVector3D = IVector3D::new(0, 0, 1);

/// Generates a polycube based on an orthogonalized skeleton, and a labeled skeleton with the same structure,
/// but with the regions on the polycube.
pub fn generate_polycube(skeleton: &LabeledCurveSkeleton) -> (Polycube, LabeledCurveSkeleton) {
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
                VoxelOwner::EdgeMidpoint()
            };

            voxel_owners.insert(pos, owner);
        }
    }

    // Setup vertices and faces for polycube mesh.
    let mut vertex_map: HashMap<IVector3D, usize> = HashMap::default();
    let mut positions = Vec::<Vector3D>::new();
    let mut faces = Vec::<Vec<usize>>::new();

    let mut intern_vertex = |grid_pos: IVector3D| -> usize {
        *vertex_map.entry(grid_pos).or_insert_with(|| {
            let idx = positions.len();
            positions.push(Vector3D::new(
                f64::from(grid_pos.x) / 2.,
                f64::from(grid_pos.y) / 2.,
                f64::from(grid_pos.z) / 2.,
            ));
            idx
        })
    };

    for &voxel_pos in voxel_owners.keys() {
        let center = voxel_pos * 2;

        for (dir, face_offsets) in &DIRECTIONS {
            let neighbor_pos = voxel_pos + *dir;
            if voxel_owners.contains_key(&neighbor_pos) {
                continue;
            }

            let quad = face_offsets
                .iter()
                .map(|offset| intern_vertex(center + *offset))
                .collect::<Vec<_>>();
            faces.push(quad);
        }
    }

    let mesh: Mesh<POLYCUBE> = if faces.is_empty() {
        error!("Polycube mesh has no faces.");
        Mesh::default()
    } else {
        Mesh::from(&faces, &positions)
            .expect("Failed to build voxel polycube mesh")
            .0
    };

    let polycube_skeleton = generate_labeled_skeleton(skeleton, &mesh, voxel_owners);

    (
        Polycube {
            structure: mesh,
            region_to_vertex: BiHashMap::new(), // We do not have a dual (yet) so this has to be empty // TODO: create trivial dual
        },
        polycube_skeleton,
    )
}

/// For a given polycube based on a skeleton, generates an isomorphic skeleton but with the regions of the polycube.
fn generate_labeled_skeleton(
    original: &LabeledCurveSkeleton,
    mesh: &Mesh<POLYCUBE>,
    voxel_owners: HashMap<IVector3D, VoxelOwner>,
) -> LabeledCurveSkeleton {
    // We want the exact same skeleton structure (including node IDs). We will only change mesh vertices (and centroids).
    let mut poly_skeleton = original.clone();

    // TODO: recompute boundary loops from polycube mesh once patches are remapped.
    for edge_idx in poly_skeleton.edge_indices() {
        // temp debug test
        let len = poly_skeleton
            .edge_weight_mut(edge_idx)
            .unwrap()
            .boundary_loop.edge_midpoints.len();

        info!("Edge {:?} has {} edge-midpoint voxels on its boundary loop.", edge_idx, len);
    }

    // TODO change stuff here actually

    poly_skeleton
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
