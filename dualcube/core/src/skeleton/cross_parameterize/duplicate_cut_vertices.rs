use std::collections::HashMap;

use petgraph::{graph::EdgeIndex, prelude::StableGraph};

use crate::{
    prelude::{EdgeID, VertID},
    skeleton::cross_parameterize::virtual_mesh::{
        EdgemidpointToVirtual, VertexToVirtual, VirtualEdgeWeight, VirtualNode, VirtualNodeOrigin,
    },
};

pub fn duplicate_cut_endpoint(
    graph: &mut StableGraph<VirtualNode, VirtualEdgeWeight, petgraph::Undirected>,
    edge_midpoint_ids_to_node_indices: &mut HashMap<EdgeID, EdgemidpointToVirtual>,
    cut_index: usize,
    boundary_edge: EdgeIndex,
    midpoint: EdgeID,
) {
    // Remove original from graph
    let midpoint_node_idx = match edge_midpoint_ids_to_node_indices.get(&midpoint) {
        Some(EdgemidpointToVirtual::Unique(idx)) => *idx,
        Some(EdgemidpointToVirtual::CutEndpointPair { .. }) => panic!(
            "Cut endpoint {:?} is shared by multiple cuts, which is not supported.",
            midpoint
        ),
        None => unreachable!(
            "Cut endpoint {:?} does not correspond to any known boundary midpoint",
            midpoint
        ),
    };
    let midpoint_pos = graph.node_weight(midpoint_node_idx).unwrap().position;
    if graph.remove_node(midpoint_node_idx).is_none() {
        panic!(
            "Cut endpoint is reused for multiple cuts, leaving no space between to parameterize."
        );
    };
    edge_midpoint_ids_to_node_indices.remove(&midpoint);

    // Insert two duplicates, one for each side of the cut
    let dup_virtual_1 = VirtualNode {
        position: midpoint_pos,
        origin: VirtualNodeOrigin::CutEndpointMidpointDuplicate {
            edge: midpoint,
            boundary: boundary_edge,
            peer: None, // to be filled in after both copies are created
            cut_index: cut_index,
            side: false,
        },
    };
    let dup_virtual_2 = VirtualNode {
        position: midpoint_pos,
        origin: VirtualNodeOrigin::CutEndpointMidpointDuplicate {
            edge: midpoint,
            boundary: boundary_edge,
            peer: None, // to be filled in after both copies are created
            cut_index: cut_index,
            side: true,
        },
    };
    let dup_node_idx_1 = graph.add_node(dup_virtual_1);
    let dup_node_idx_2 = graph.add_node(dup_virtual_2);

    // Update peer references
    if let VirtualNodeOrigin::CutEndpointMidpointDuplicate { peer, .. } =
        &mut graph[dup_node_idx_1].origin
    {
        *peer = Some(dup_node_idx_2);
    } else {
        unreachable!();
    }
    if let VirtualNodeOrigin::CutEndpointMidpointDuplicate { peer, .. } =
        &mut graph[dup_node_idx_2].origin
    {
        *peer = Some(dup_node_idx_1);
    } else {
        unreachable!();
    }

    // Save as duplicate in map
    edge_midpoint_ids_to_node_indices.insert(
        midpoint,
        EdgemidpointToVirtual::CutEndpointPair {
            left: dup_node_idx_1,
            right: dup_node_idx_2,
        },
    );
}

pub fn duplicate_cut_vertex(
    graph: &mut StableGraph<VirtualNode, VirtualEdgeWeight, petgraph::Undirected>,
    vert_to_nodes: &mut HashMap<VertID, VertexToVirtual>,
    cut_index: usize,
    vertex: VertID,
) {
    // Remove original from graph
    let node_idx = match vert_to_nodes.get(&vertex) {
        Some(VertexToVirtual::Unique(idx)) => *idx,
        Some(VertexToVirtual::CutPair { .. }) => panic!(
            "Cut vertex {:?} is shared by multiple cuts, which is not supported.",
            vertex
        ),
        None => unreachable!(
            "Cut vertex {:?} does not correspond to any known mesh vertex",
            vertex
        ),
    };
    let pos = graph.node_weight(node_idx).unwrap().position;
    if graph.remove_node(node_idx).is_none() {
        panic!("Cut vertex is reused for multiple cuts, leaving no space between to parameterize.");
    };
    vert_to_nodes.remove(&vertex);

    // Insert two duplicates, one for each side of the cut
    let dup_virtual_1 = VirtualNode {
        position: pos,
        origin: VirtualNodeOrigin::CutDuplicate {
            original: vertex,
            peer: None,
            cut_index,
            side: false,
        },
    };
    let dup_virtual_2 = VirtualNode {
        position: pos,
        origin: VirtualNodeOrigin::CutDuplicate {
            original: vertex,
            peer: None,
            cut_index,
            side: true,
        },
    };
    let dup_node_idx_1 = graph.add_node(dup_virtual_1);
    let dup_node_idx_2 = graph.add_node(dup_virtual_2);

    // Update peer references
    if let VirtualNodeOrigin::CutDuplicate { peer, .. } = &mut graph[dup_node_idx_1].origin {
        *peer = Some(dup_node_idx_2);
    } else {
        unreachable!();
    }
    if let VirtualNodeOrigin::CutDuplicate { peer, .. } = &mut graph[dup_node_idx_2].origin {
        *peer = Some(dup_node_idx_1);
    } else {
        unreachable!();
    }

    // Save as duplicate in map
    vert_to_nodes.insert(
        vertex,
        VertexToVirtual::CutPair {
            left: dup_node_idx_1,
            right: dup_node_idx_2,
        },
    );
}
