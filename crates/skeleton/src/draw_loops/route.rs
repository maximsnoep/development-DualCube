use super::*;
use dualcube_dual::{CrossingRouteError, FlowGraph};

// TODO: Route through arrangement cells once shared dual loops support interior arcs/re-entry.
// TODO: Tune the backtracking, candidate budgets, and flow objective.
const MAX_SEARCHES: usize = 2048;
const MAX_CANDIDATES: usize = 128;

#[allow(clippy::too_many_arguments)]
pub(super) fn find(
    mesh: &Mesh<INPUT>,
    graphs: &[FlowGraph; 3],
    working: &DualLoops,
    drawn: &HashSet<LoopID>,
    id: LoopID,
    crossings: &[LoopID],
    regions: &[usize],
    gap_regions: &HashMap<dualcube_dual::GapNode, usize>,
    expected: &PartialTopology,
    avoid: &[Loop],
    total: &mut usize,
) -> Result<Loop, Failure> {
    let loops = subset(working, drawn);
    let state = LoopState::from_loops(mesh, loops);
    let axis = working[id].direction;
    let mut phases: Vec<_> = (0..crossings.len()).collect();
    phases.sort_by_key(|&i| state.loops[crossings[i]].edges.len());
    let mut sequences = HashSet::new();
    let anchors: Vec<_> = phases
        .into_iter()
        .filter_map(|i| {
            let mut sequence = crossings.to_vec();
            sequence.rotate_left(i);
            let mut after = regions.to_vec();
            after.rotate_left(i);
            if !sequences.insert((sequence.clone(), after.clone())) {
                return None;
            }
            let mut starts = Vec::new();
            for &edge in &state.loops[sequence[0]].edges {
                for gap in 0..=state.occupied.get(&edge).map_or(0, Vec::len) {
                    if gap_regions.get(&(edge, gap)) == after.last() {
                        starts.push((edge, gap));
                    }
                }
            }
            if starts.is_empty() {
                return None;
            }
            Some((sequence, after, starts))
        })
        .take(8)
        .collect();
    let base_sampler = LoopSampler::new(mesh, &state.loops, &state.occupied, Some(graphs));
    let mut extra = HashMap::new();
    for l in avoid {
        for &edge in &l.edges {
            *extra.entry(mesh.face(edge)).or_insert(0.) += 4.;
        }
    }
    let mut failed_starts = HashSet::new();
    let mut stats = SearchStats::default();
    let mut candidates = 0;
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
        if anchors.is_empty() { 0 } else { MAX_SEARCHES }
    };
    for attempt in 0..count {
        if candidates >= MAX_CANDIDATES {
            break;
        }

        let sampler = base_sampler.with_extra_proximity(&extra);
        let (edges, gaps) = if crossings.is_empty() {
            *total += 1;
            stats.searches += 1;
            let edges = seed[attempt].clone();
            let gaps = sampler.assign_gaps(&edges, axis);
            (edges, gaps)
        } else {
            // Cycle through local anchors; progressively cover each anchor's gaps.
            let phase = (attempt / 128) % anchors.len();
            let (sequence, after, starts) = &anchors[phase];
            let round = attempt / (128 * anchors.len());
            let index = round * 128 + attempt % 128;
            let start = starts[spread(index, starts.len())];
            if failed_starts.contains(&(phase, start)) {
                continue;
            }
            *total += 1;
            stats.searches += 1;
            match sampler.crossing_candidate_in_regions(
                axis,
                start,
                sequence,
                Some((after, gap_regions)),
                &OrderedFloat,
            ) {
                Ok((edges, _, gaps)) => (edges, gaps),
                Err(CrossingRouteError::RepeatedEdge(edge)) => {
                    stats.repeated_edge += 1;
                    last_reentry = Some(Reentry::Edge(edge));
                    *extra.entry(mesh.face(edge)).or_insert(0.) += 1.;
                    continue;
                }
                Err(CrossingRouteError::ExpansionLimit) => {
                    stats.expansion_limit += 1;
                    continue;
                }
                Err(CrossingRouteError::NoRoute) => {
                    stats.no_route += 1;
                    failed_starts.insert((phase, start));
                    continue;
                }
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
            stats.repeated_face += 1;
            last_reentry = Some(at);
            continue;
        }
        if !seen.insert((l.edges.clone(), gaps.clone())) {
            stats.duplicates += 1;
            continue;
        }
        candidates += 1;
        let mut arrangement_valid = false;
        let mut topology_valid = false;
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
                if let Ok(topology) = PartialTopology::new(mesh, &candidate) {
                    arrangement_valid = true;
                    if topology.equivalent(expected) {
                        topology_valid = true;
                        if !avoid.iter().any(|old| same_geometry(old, &l)) {
                            tracing::debug!(?id, ?stats, "Accepted loop route");
                            return Ok(l);
                        }
                    }
                }
                // The template determines orientation; input-space flow is only a geometric cost.
                l.edges.reverse();
                l.offsets.reverse();
            }
        }
        if topology_valid {
            stats.duplicates += 1;
        } else if arrangement_valid {
            stats.topology_rejections += 1;
        } else {
            stats.invalid_arrangement += 1;
        }
    }
    Err(Failure::SearchExhausted {
        loop_id: id,
        stats,
        reentry: last_reentry,
    })
}

// A coprime stride visits every start before repeating, while spreading nearby attempts.
fn spread(index: usize, count: usize) -> usize {
    let mut stride = count / 2 + 1;
    while gcd(stride, count) != 1 {
        stride += 1;
    }
    index * stride % count
}
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn same_geometry(a: &Loop, b: &Loop) -> bool {
    if a.edges.len() != b.edges.len() {
        return false;
    }
    let Some(start) = a.edges.iter().position(|&e| e == b.edges[0]) else {
        return false;
    };
    [false, true].into_iter().any(|reverse| {
        (0..b.edges.len()).all(|i| {
            let j = if reverse {
                (start + a.edges.len() - i) % a.edges.len()
            } else {
                (start + i) % a.edges.len()
            };
            a.edges[j] == b.edges[i] && (a.offsets[j] - b.offsets[i]).abs() < 1e-8
        })
    })
}
