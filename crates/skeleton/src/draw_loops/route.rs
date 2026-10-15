use super::*;
use dualcube_dual::{
    FlowGraph,
    insertion::{DrawError, Plan, Step},
    loop_map::LoopMap,
    partial::Correspondence,
};

#[derive(Default, Debug)]
struct Timings {
    setup: std::time::Duration,
    search: std::time::Duration,
    validation: std::time::Duration,
}
impl Drop for Timings {
    fn drop(&mut self) {
        tracing::debug!(?self, "Loop route timings");
    }
}

pub(super) struct Flows {
    vectors: [HashMap<FaceID, Vector3D>; 3],
    graphs: Option<[FlowGraph; 3]>,
}

impl Flows {
    pub(super) fn new(mesh: &Mesh<INPUT>, needs_seed: bool) -> Self {
        let fields = dualcube_flow::Fields::new(mesh);
        let vectors = [&fields.field_x, &fields.field_y, &fields.field_z].map(|field| {
            mesh.face_ids()
                .into_iter()
                .map(|f| {
                    (
                        f,
                        mesh.vertices(f)
                            .filter_map(|v| field.vector_at(v))
                            .sum::<Vector3D>()
                            / 3.,
                    )
                })
                .collect()
        });
        let graphs = needs_seed.then(|| {
            dualcube_flow::build_flow_graphs(
                mesh,
                &fields,
                dualcube_flow::GraphParams {
                    max_deviation: std::f64::consts::PI,
                    ..Default::default()
                },
            )
        });
        Self { vectors, graphs }
    }
}

// TODO: Tune the candidate budget, backtracking choices, and geometric objective.
const MAX_CANDIDATES: usize = 64;

/// Template steps for one correspondence, with the plan that draws them.
pub(super) struct Target<'a> {
    pub plan: &'a Plan,
    pub correspondence: &'a Correspondence,
    pub steps: Vec<Step>,
}

/// Insert loop `id` into `map` so that the result has the `expected` topology. `avoid` lists
/// triangles of earlier drawings to stay away from.
#[allow(clippy::too_many_arguments)]
pub(super) fn find(
    mesh: &Mesh<INPUT>,
    flows: &Flows,
    map: &LoopMap,
    actual: &PartialTopology,
    axis: Direction,
    id: LoopID,
    target: &Target<'_>,
    expected: &PartialTopology,
    avoid: &[HashSet<FaceID>],
    total: &mut usize,
) -> Result<LoopMap, Failure> {
    let mut timing = Timings::default();
    let setup = std::time::Instant::now();
    let steps = &target.steps;
    let k = steps.len();
    let mut stats = SearchStats::default();
    let mut penalties = HashMap::<FaceID, f64>::new();
    for triangles in avoid {
        for &face in triangles {
            *penalties.entry(face).or_default() += 4.;
        }
    }
    let flow = &flows.vectors[match axis {
        Direction::X => 0,
        Direction::Y => 1,
        Direction::Z => 2,
    }];
    // Starting crossings with their anchors, fewest first.
    let mut phases = vec![];
    let mut distinct = HashSet::new();
    for i in 0..k {
        let mut seq = steps.clone();
        seq.rotate_left(i);
        if !distinct.insert(seq) {
            continue;
        }
        let anchors = target.plan.anchors(map, actual, steps, i);
        if !anchors.is_empty() {
            phases.push((i, anchors));
        }
    }
    phases.sort_by_key(|p| p.1.len());
    // A loop crossing nothing starts from sampled seeds, on an empty surface only.
    let seeds = if k == 0 && actual.along.is_empty() {
        seeds(mesh, flows, map, id, axis)
    } else {
        vec![]
    };
    let count = if k == 0 {
        seeds.len()
    } else if phases.is_empty() {
        0
    } else {
        MAX_CANDIDATES
    };
    timing.setup += setup.elapsed();
    for attempt in 0..count {
        stats.searches += 1;
        *total += 1;
        let search = std::time::Instant::now();
        let candidate = if k == 0 {
            Ok(seeds[attempt].clone())
        } else {
            let (start, anchors) = &phases[attempt % phases.len()];
            let index = attempt / phases.len();
            let anchor =
                anchors[(index * anchors.len() / MAX_CANDIDATES.max(1) + index) % anchors.len()];
            target.plan.draw(
                map,
                actual,
                target.correspondence,
                steps,
                *start,
                anchor,
                |face, delta, _| {
                    let length = delta.norm();
                    let f = flow[&face];
                    let confidence = f.norm().min(1.);
                    let alignment = if length > 0. && confidence > 1e-12 {
                        (delta.dot(&f) / (length * f.norm())).clamp(-1., 1.)
                    } else {
                        1.
                    };
                    length
                        * (1.
                            + 5. * confidence * (1. - alignment)
                            + penalties.get(&face).copied().unwrap_or(0.))
                },
            )
        };
        timing.search += search.elapsed();
        let mut drawn = match candidate {
            Ok(drawn) => drawn,
            Err(DrawError::NoRoute(_)) => {
                stats.no_route += 1;
                continue;
            }
            Err(DrawError::ExpansionLimit(_)) => {
                stats.expansion_limit += 1;
                continue;
            }
            Err(DrawError::Mismatch(_)) => {
                stats.arc_mismatches += 1;
                continue;
            }
            Err(DrawError::Revisit(_)) => {
                stats.invalid_arrangement += 1;
                continue;
            }
        };
        let validation = std::time::Instant::now();
        let mut topology = PartialTopology::from_map(&drawn);
        timing.validation += validation.elapsed();
        for reversed in [false, true] {
            if reversed {
                topology.reverse_loop(id);
            }
            if topology.equivalent(expected) {
                if reversed {
                    drawn.reverse_loop(id);
                }
                tracing::debug!(?id, ?stats, "Accepted loop route");
                return Ok(drawn);
            }
        }
        for face in drawn.triangles(id) {
            *penalties.entry(face).or_default() += 0.5;
        }
        stats.topology_rejections += 1;
    }
    Err(Failure::SearchExhausted { loop_id: id, stats })
}

// Loops sampled along the flow field, each added alone to `map`.
fn seeds(
    mesh: &Mesh<INPUT>,
    flows: &Flows,
    map: &LoopMap,
    id: LoopID,
    axis: Direction,
) -> Vec<LoopMap> {
    let empty = DualLoops::with_key();
    let state = LoopState::from_loops(mesh, empty);
    let sampler = LoopSampler::new(mesh, &state.loops, &state.occupied, flows.graphs.as_ref());
    let mut result = vec![];
    for edges in sampler.sample_loops(16, axis, OrderedFloat, |(e, _)| -sampler.separated_area(e)) {
        let len = edges.len();
        let exits: Vec<_> = (0..len)
            .filter(|&i| mesh.twin(edges[i]) == edges[(i + 1) % len])
            .map(|i| edges[i])
            .collect();
        // A sampled loop crosses every edge at most once.
        let order = exits
            .iter()
            .enumerate()
            .map(|(o, &e)| (e, vec![o]))
            .collect();
        let mut seeded = map.clone();
        if seeded
            .add_disjoint(mesh, &[(id, (0..exits.len()).collect())], &exits, &order)
            .is_ok()
        {
            result.push(seeded);
        }
    }
    result
}
