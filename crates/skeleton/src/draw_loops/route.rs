use super::*;
use dualcube_dual::{CrossingRouteError, FlowGraph};

// TODO: Reconsider earlier accepted routes and tune the candidate budget/flow objective.
const MAX_CANDIDATES: usize = 128;

pub(super) fn find(
    mesh: &Mesh<INPUT>,
    graphs: &[FlowGraph; 3],
    working: &DualLoops,
    drawn: &HashSet<LoopID>,
    id: LoopID,
    crossings: &[LoopID],
    expected: &PartialTopology,
    total: &mut usize,
) -> Result<Loop, Failure> {
    let loops = subset(working, drawn);
    let state = LoopState::from_loops(mesh, loops);
    let axis = working[id].direction;
    let mut anchors = Vec::new();
    if let Some(first) = crossings.first() {
        for &edge in &state.loops[*first].edges {
            for gap in 0..=state.occupied.get(&edge).map_or(0, Vec::len) {
                anchors.push((edge, gap));
            }
        }
    }
    let mut extra = HashMap::new();
    let mut rejected = 0;
    let mut last_reentry = None;
    let mut seen = HashSet::new();
    let mut seed = Vec::new();
    if crossings.is_empty() {
        let sampler = LoopSampler::new(mesh, &state.loops, &state.occupied, Some(graphs));
        seed = sampler.sample_loops(16, axis, OrderedFloat, |(edges, _)| {
            -sampler.separated_area(edges)
        });
    }
    let count = if crossings.is_empty() {
        seed.len()
    } else {
        MAX_CANDIDATES
    };
    for attempt in 0..count {
        *total += 1;
        let sampler = LoopSampler::new(mesh, &state.loops, &state.occupied, Some(graphs))
            .with_extra_proximity(&extra);
        let (edges, gaps) = if crossings.is_empty() {
            let edges = seed[attempt].clone();
            let gaps = sampler.assign_gaps(&edges, axis);
            (edges, gaps)
        } else {
            // Spread starts over the anchor boundary before revisiting any local neighborhood.
            let index = (attempt % anchors.len()) * anchors.len() / count.min(anchors.len());
            let start = anchors[index % anchors.len()];
            match sampler.crossing_candidate(axis, start, crossings, &OrderedFloat) {
                Ok((edges, _, gaps)) => (edges, gaps),
                Err(CrossingRouteError::RepeatedEdge(edge)) => {
                    last_reentry = Some(Reentry::Edge(edge));
                    *extra.entry(mesh.face(edge)).or_insert(0.) += 1.;
                    continue;
                }
                Err(CrossingRouteError::NoRoute) => continue,
            }
        };
        if edges.is_empty() {
            continue;
        }
        let (_, offsets) = sampler.place_loop_in_gaps(&edges, axis, gaps.clone());
        let l = Loop {
            edges,
            offsets,
            direction: axis,
        };
        for &edge in &l.edges {
            *extra.entry(mesh.face(edge)).or_insert(0.) += 0.5;
        }
        if let Some(at) = reentry(mesh, &l) {
            last_reentry = Some(at);
            continue;
        }
        if !seen.insert((l.edges.clone(), gaps.clone())) {
            continue;
        }
        let mut middle = vec![0.; l.edges.len()];
        for (exit, gap) in gaps {
            let edge = l.edges[exit];
            let order = sampler.loops_on_edge(edge);
            let lo = if gap == 0 {
                0.
            } else {
                state.loops[order[gap - 1]].offset(edge).unwrap()
            };
            let hi = if gap == order.len() {
                1.
            } else {
                state.loops[order[gap]].offset(edge).unwrap()
            };
            middle[exit] = (lo + hi) / 2.;
            middle[(exit + 1) % l.edges.len()] = 1. - middle[exit];
        }
        for offsets in [l.offsets.clone(), middle] {
            let mut l = Loop {
                edges: l.edges.clone(),
                offsets,
                direction: axis,
            };
            for _ in 0..2 {
                let mut candidate = working.clone();
                candidate[id] = l.clone();
                let mut selected = drawn.clone();
                selected.insert(id);
                let candidate = subset(&candidate, &selected);
                if PartialTopology::new(mesh, &candidate)
                    .is_ok_and(|topology| topology.equivalent(expected))
                {
                    return Ok(l);
                }
                // The template determines orientation; input-space flow is only a geometric cost.
                l.edges.reverse();
                l.offsets.reverse();
            }
        }
        rejected += 1;
    }
    Err(Failure::SearchExhausted {
        loop_id: id,
        candidates: count,
        topology_rejections: rejected,
        reentry: last_reentry,
    })
}
