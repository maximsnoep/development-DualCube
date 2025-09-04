use crate::prelude::*;
use itertools::Itertools;
use ordered_float::OrderedFloat;

impl Solution {
    pub fn construct_elastica_loop(
        &self,
        [e1, e2]: [EdgeID; 2],
        direction: PrincipalDirection,
        flow_graph: &grapff::fixed::FixedGraph<EdgeID, f64>,
        measure: impl Fn(f64) -> OrderedFloat<f64>,
    ) -> Option<(Vec<EdgeID>, f64)> {
        // Filter the original flow graph

        let occupied = self.occupied_edgepairs();
        let filter_edges = |edge: (&EdgeID, &EdgeID)| !occupied.contains(&(*edge.0, *edge.1));
        let filter_nodes = |&node: &EdgeID| !self.loops_on_edge(node).iter().any(|&loop_id| self.loops[loop_id].direction == direction);
        let g = flow_graph.filter_edges(filter_edges);
        let g = g.filter_nodes(filter_nodes);

        if let Some((option, cost)) = self.construct_loop([e1, e2], &g, &measure) {
            let mut cleaned_option = vec![];
            for edge_id in &option {
                if cleaned_option.contains(&edge_id) {
                    cleaned_option = cleaned_option.into_iter().take_while(|&x| x != edge_id).collect_vec();
                }
                cleaned_option.push(edge_id);
            }
            Some((option, cost))
        } else {
            None
        }
    }
}
