//! Boundary costs from the flow graphs that loop evolution routes loops by.

use super::Port;
use crate::{Skeleton, SkeletonEdgeID};
use dualcube_dual::FlowGraph;
use dualcube_types::prelude::*;

/// The three per-axis flow graphs of one mesh, with every move kept.
pub struct Flow {
    graphs: [FlowGraph; 3],
}

impl Flow {
    pub fn new(mesh: &Mesh<INPUT>) -> Self {
        let fields = dualcube_flow::Fields::new(mesh);
        let graphs = dualcube_flow::build_flow_graphs(
            mesh,
            &fields,
            dualcube_flow::GraphParams {
                max_deviation: std::f64::consts::PI,
                ..Default::default()
            },
        );
        Self { graphs }
    }

    /// Flow cost per length of `edge`'s boundary drawn as a loop with each label (port from
    /// `nodes[0]` to `nodes[1]`): at least 1, and 1 where the loop follows a confident flow.
    /// A positive label follows the boundary's own orientation, a negative one its reverse.
    pub fn costs(&self, mesh: &Mesh<INPUT>, skeleton: &Skeleton, edge: SkeletonEdgeID) -> [f64; 6] {
        let exits: Vec<EdgeID> = skeleton.edges[edge]
            .boundary
            .crossings
            .iter()
            .map(|&c| skeleton.surface.crossings[c].exit)
            .collect();
        let n = exits.len();
        std::array::from_fn(|p| {
            let port = Port::ALL[p];
            let graph = &self.graphs[port.axis()];
            let (mut cost, mut length) = (0., 0.);
            for i in 0..n {
                // The arc from crossing `i` to `i + 1` lies in the triangle entered through `i`.
                let (from, to) = (mesh.twin(exits[i]), exits[(i + 1) % n]);
                let (a, b) = if port.is_positive() {
                    (from, to)
                } else {
                    (to, from)
                };
                if a == b {
                    continue;
                }
                let step = (mesh.position(b) - mesh.position(a)).norm();
                length += step;
                // Every move within a triangle is kept; a missing one counts as orthogonal.
                cost += graph.get_directed_weight(a, b).unwrap_or(step * 21.);
            }
            if length > 0. { cost / length } else { 1. }
        })
    }
}
