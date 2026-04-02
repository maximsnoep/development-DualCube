use std::collections::HashMap;

use log::{info, warn};
use mehsh::prelude::Vector3D;
use petgraph::visit::{EdgeRef, IntoEdgeReferences};

use crate::{
    layout::Layout,
    prelude::PrincipalDirection,
    skeleton::SkeletonData,
};

fn axis_index(direction: PrincipalDirection) -> usize {
    match direction {
        PrincipalDirection::X => 0,
        PrincipalDirection::Y => 1,
        PrincipalDirection::Z => 2,
    }
}

fn signs_to_corner_index(signs: [i8; 3]) -> usize {
    let sx = usize::from(signs[0] > 0);
    let sy = usize::from(signs[1] > 0);
    let sz = usize::from(signs[2] > 0);
    (sx << 2) | (sy << 1) | sz
}

fn signs_for_direction(direction: PrincipalDirection, along: i8, a: i8, b: i8) -> [i8; 3] {
    match direction {
        PrincipalDirection::X => [along, a, b],
        PrincipalDirection::Y => [a, along, b],
        PrincipalDirection::Z => [a, b, along],
    }
}

pub fn populate_layout_from_skeleton(skeleton_data: &SkeletonData) -> Option<Layout> {
    // Prefer polycube_skeleton when available because it has patch vertices adapted to the polycube.
    let ortho_skeleton = skeleton_data
        .polycube_skeleton()
        .or_else(|| skeleton_data.labeled_skeleton())?;

    // Partial result (for now): seed per-node cube corners and per-edge tube corner correspondences.
    // This is enough to prototype skeleton-driven corner/path intent, but not enough to instantiate Layout.
    let mut node_to_cube_corners: HashMap<usize, Vec<Vector3D>> = HashMap::new();
    let mut node_to_axis_extents: HashMap<usize, [f64; 3]> = HashMap::new();
    let mut node_to_patch_size: HashMap<usize, usize> = HashMap::new();

    for node_id in ortho_skeleton.node_indices() {
        let node = &ortho_skeleton[node_id];
        let center = node.skeleton_node.position;

        // Base extent for isolated nodes; grow extent from incident edge lengths per axis.
        let mut extents = [0.5_f64, 0.5_f64, 0.5_f64];
        for edge_ref in ortho_skeleton.edges(node_id) {
            let edge = edge_ref.weight();
            let axis = axis_index(edge.direction);
            extents[axis] = extents[axis].max(f64::from(edge.length).max(1.0) * 0.5);
        }

        let mut corners = Vec::with_capacity(8);
        for idx in 0..8 {
            let sx = if (idx & 4) != 0 { 1.0 } else { -1.0 };
            let sy = if (idx & 2) != 0 { 1.0 } else { -1.0 };
            let sz = if (idx & 1) != 0 { 1.0 } else { -1.0 };
            let offset = Vector3D::new(sx * extents[0], sy * extents[1], sz * extents[2]);
            corners.push(center + offset);
        }

        node_to_cube_corners.insert(node_id.index(), corners);
        node_to_axis_extents.insert(node_id.index(), extents);
        node_to_patch_size.insert(node_id.index(), node.skeleton_node.patch_vertices.len());
    }

    let mut tube_templates: Vec<([usize; 2], PrincipalDirection, u32, Vec<(usize, usize)>)> =
        Vec::new();

    // For each skeleton edge, derive the four corner-pairs of the tube cross-section.
    let cross_section = [(-1_i8, -1_i8), (-1, 1), (1, 1), (1, -1)];
    for edge_ref in ortho_skeleton.edge_references() {
        let u = edge_ref.source();
        let v = edge_ref.target();
        let edge = edge_ref.weight();

        let delta = ortho_skeleton[v].grid_position - ortho_skeleton[u].grid_position;
        let axis_component = match edge.direction {
            PrincipalDirection::X => delta.x,
            PrincipalDirection::Y => delta.y,
            PrincipalDirection::Z => delta.z,
        };

        let along_u = if axis_component >= 0 { 1_i8 } else { -1_i8 };
        let along_v = -along_u;

        let corner_pairs = cross_section
            .iter()
            .map(|&(a, b)| {
                let idx_u = signs_to_corner_index(signs_for_direction(edge.direction, along_u, a, b));
                let idx_v = signs_to_corner_index(signs_for_direction(edge.direction, along_v, a, b));
                (idx_u, idx_v)
            })
            .collect::<Vec<_>>();

        tube_templates.push(([u.index(), v.index()], edge.direction, edge.length, corner_pairs));
    }

    info!(
        "Skeleton layout seed prepared: {} node cubes, {} edge tubes.",
        node_to_cube_corners.len(),
        tube_templates.len()
    );

    // Keep these around for debugger inspection while implementation is in progress.
    let _partial_layout_seed = (node_to_cube_corners, node_to_axis_extents, node_to_patch_size, tube_templates);

    warn!(
        "populate_layout_from_skeleton currently computes seeds only; returning None because full Layout construction still needs Dual/Polycube mesh-level correspondences and path/patch assignment."
    );
    None
}
