use std::collections::HashMap;

use bimap::BiHashMap;
use mehsh::prelude::Mesh;
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;

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

pub fn generate_polycube(
    skeleton: &LabeledCurveSkeleton,
) -> (Polycube, HashMap<IVector3D, VoxelOwner>) {
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
            if diff.x == 0 { 0 } else { diff.x / diff.x.abs() },
            if diff.y == 0 { 0 } else { diff.y / diff.y.abs() },
            if diff.z == 0 { 0 } else { diff.z / diff.z.abs() },
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

    // // Build Mesh Geometry
    // let mut positions: Vec<[f32; 3]> = Vec::new();
    // let mut normals: Vec<[f32; 3]> = Vec::new();
    // let mut indices: Vec<u32> = Vec::new();

    // // Track vertex ownership during generation
    // let mut vertex_to_node: Vec<NodeIndex> = Vec::new();

    // // Deduplication map
    // let mut vertex_map: HashMap<IVector3D, u32> = HashMap::default();

    // let directions = [
    //     (IVector3D::X, Vec3::X),
    //     (IVector3D::NEG_X, -Vec3::X),
    //     (IVector3D::Y, Vec3::Y),
    //     (IVector3D::NEG_Y, -Vec3::Y),
    //     (IVector3D::Z, Vec3::Z),
    //     (IVector3D::NEG_Z, -Vec3::Z),
    // ];

    // // Helper to intern vertices.
    // let mut insert_vertex = |pos: Vec3, normal: Vec3, owner: NodeIndex| {
    //     let grid_pos = (pos / MESH_SCALE * 2.0).round().as_IVector3D();

    //     // Might already exist
    //     *vertex_map.entry(grid_pos).or_insert_with(|| {
    //         let idx = positions.len() as u32;
    //         positions.push(pos.to_array());
    //         normals.push(normal.to_array());
    //         vertex_to_node.push(owner);
    //         idx
    //     })
    // };

    // // For each voxel, if it is fully owned, we generate its vertices
    // // Since we do not generate for shared voxels, we never override ownership.
    // for (&pos, voxel_owner) in &voxel_owners {
    //     let owner = match voxel_owner {
    //         VoxelOwner::Node(n) => n,
    //         VoxelOwner::Edge(n) => n,
    //         VoxelOwner::EdgeMidpoint() => {
    //             continue;
    //         }
    //     };

    //     let center = pos.as_vec3() * MESH_SCALE;

    //     for (dir_ivec, dir_vec) in &directions {
    //         let verts = get_face_vertices(center, *dir_ivec, MESH_SCALE);
    //         insert_vertex(verts.0, *dir_vec, *owner);
    //         insert_vertex(verts.1, *dir_vec, *owner);
    //         insert_vertex(verts.2, *dir_vec, *owner);
    //         insert_vertex(verts.3, *dir_vec, *owner);
    //     }
    // }

    // // All relevant vertices now must exist.
    // // Helper to get vertex (which must exist)
    // let get_vertex = |pos: Vec3| -> u32 {
    //     let grid_pos = (pos / MESH_SCALE * 2.0).round().as_IVector3D();
    //     *vertex_map.get(&grid_pos).unwrap()
    // };

    // let mut push_quad = |verts: (Vec3, Vec3, Vec3, Vec3)| {
    //     let i0 = get_vertex(verts.0);
    //     let i1 = get_vertex(verts.1);
    //     let i2 = get_vertex(verts.2);
    //     let i3 = get_vertex(verts.3);

    //     // Triangle 1 (0-1-2)
    //     indices.extend_from_slice(&[i0, i1, i2]);

    //     // Triangle 2 (0-2-3) -> Diagonal is (i0, i2)
    //     indices.extend_from_slice(&[i0, i2, i3]);
    // };

    // for (&pos, _) in &voxel_owners {
    //     let center = pos.as_vec3() * MESH_SCALE;

    //     for (dir_ivec, _dir_vec) in &directions {
    //         let neighbor_pos = pos + *dir_ivec;

    //         if voxel_owners.contains_key(&neighbor_pos) {
    //             // Do not generate internal faces
    //             continue;
    //         }

    //         let verts = get_face_vertices(center, *dir_ivec, MESH_SCALE);
    //         push_quad(verts);
    //     }
    // }


    
    // temp empty return
    (
        Polycube {
            structure: Mesh::default(),
            region_to_vertex: BiHashMap::new(),
        },
        voxel_owners,
    )
}
