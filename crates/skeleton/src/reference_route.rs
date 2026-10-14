use super::*;
use dualcube_dual::{
    FlowGraph,
    surface_router::{RouteError, SurfaceRouter},
};

// TODO: Tune the candidate budget, backtracking choices, and geometric objective.
const MAX_CANDIDATES: usize = 64;

#[allow(clippy::too_many_arguments)]
pub(super) fn find(
    mesh: &Mesh<INPUT>,
    graphs: &[FlowGraph; 3],
    working: &DualLoops,
    drawn: &HashSet<LoopID>,
    id: LoopID,
    crossings: &[LoopID],
    regions: &[usize],
    router: &SurfaceRouter,
    expected: &PartialTopology,
    avoid: &[Loop],
    total: &mut usize,
) -> Result<Loop, Failure> {
    let loops = subset(working, drawn);
    let axis = working[id].direction;
    let mut stats = SearchStats::default();
    let mut penalties = HashMap::<FaceID, f64>::new();
    for l in avoid {
        for face in l.surface.iter().map(|a| a.face).collect::<HashSet<_>>() {
            *penalties.entry(face).or_default() += 4.;
        }
    }
    let fields = dualcube_flow::Fields::new(mesh);
    let field = match axis {
        Direction::X => &fields.field_x,
        Direction::Y => &fields.field_y,
        Direction::Z => &fields.field_z,
    };
    let flow: HashMap<_, _> = mesh
        .face_ids()
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
        .collect();
    let mut phases = vec![];
    let mut distinct = HashSet::new();
    for i in 0..crossings.len() {
        let mut seq = crossings.to_vec();
        seq.rotate_left(i);
        let mut after = regions.to_vec();
        after.rotate_left(i);
        if !distinct.insert((seq.clone(), after.clone())) {
            continue;
        }
        let anchors = router.anchors(seq[0], *after.last().unwrap(), after[0]);
        if !anchors.is_empty() {
            phases.push((seq, after, anchors));
        }
    }
    phases.sort_by_key(|p| p.2.len());
    let mut seeds = vec![];
    if crossings.is_empty() {
        let state = LoopState::from_loops(mesh, loops.clone());
        let sampler = LoopSampler::new(mesh, &state.loops, &state.occupied, Some(graphs));
        for edges in
            sampler.sample_loops(16, axis, OrderedFloat, |(e, _)| -sampler.separated_area(e))
        {
            let (_, offsets) = sampler.place_loop(&edges, axis);
            let l = Loop {
                edges,
                offsets,
                direction: axis,
                surface: vec![],
            };
            if let Ok(arcs) = l.surface_arcs(mesh) {
                seeds.push(Loop::from_surface(axis, arcs));
            }
        }
    }
    let count = if crossings.is_empty() {
        seeds.len()
    } else if phases.is_empty() {
        0
    } else {
        MAX_CANDIDATES
    };
    let mut seen = Vec::new();
    for attempt in 0..count {
        stats.searches += 1;
        *total += 1;
        let candidate = if crossings.is_empty() {
            Ok(seeds.get(attempt).unwrap().clone())
        } else {
            let phase = attempt % phases.len();
            let (seq, after, anchors) = &phases[phase];
            let index = attempt / phases.len();
            let anchor =
                anchors[(index * anchors.len() / MAX_CANDIDATES.max(1) + index) % anchors.len()];
            router.route(axis, seq, after, anchor, |face, delta| {
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
            })
        };
        let l = match candidate {
            Ok(l) => l,
            Err(RouteError::NoRoute) => {
                stats.no_route += 1;
                continue;
            }
            Err(RouteError::ExpansionLimit) => {
                stats.expansion_limit += 1;
                continue;
            }
            Err(RouteError::Degenerate) => {
                stats.invalid_arrangement += 1;
                continue;
            }
        };
        for face in l.surface.iter().map(|a| a.face).collect::<HashSet<_>>() {
            *penalties.entry(face).or_default() += 0.5;
        }
        let mut valid = false;
        let shorter = shortcut(mesh, &l, &loops);
        for mut l in [shorter, l] {
            if avoid.iter().chain(&seen).any(|old| same_geometry(old, &l)) {
                stats.duplicates += 1;
                continue;
            }
            seen.push(l.clone());
            for _ in 0..2 {
                let mut candidate = working.clone();
                candidate[id] = l.clone();
                let mut selected = drawn.clone();
                selected.insert(id);
                let candidate = subset(&candidate, &selected);
                let Ok(topology) = PartialTopology::new(mesh, &candidate) else {
                    // Reversal cannot repair an invalid geometric arrangement.
                    break;
                };
                valid = true;
                if topology.equivalent(expected) {
                    tracing::debug!(?id, ?stats, "Accepted surface route");
                    return Ok(l);
                }
                l.reverse();
            }
        }
        if valid {
            stats.topology_rejections += 1;
        } else {
            stats.invalid_arrangement += 1;
        }
    }
    Err(Failure::SearchExhausted { loop_id: id, stats })
}

pub(super) fn same_geometry(a: &Loop, b: &Loop) -> bool {
    let n = a.surface.len();
    if n != b.surface.len() || n == 0 {
        return false;
    }
    let close = |p: [f64; 2], q: [f64; 2]| (p[0] - q[0]).abs().max((p[1] - q[1]).abs()) < 1e-12;
    (0..n).any(|start| {
        [false, true].into_iter().any(|reverse| {
            a.surface.iter().enumerate().all(|(i, x)| {
                let y = &b.surface[if reverse {
                    (start + n - i) % n
                } else {
                    (start + i) % n
                }];
                let (from, to) = if reverse {
                    (y.to, y.from)
                } else {
                    (y.from, y.to)
                };
                x.face == y.face && close(x.from, from) && close(x.to, to)
            })
        })
    })
}

// Remove construction bends when the ordered intersections in that triangle stay identical.
fn shortcut(mesh: &Mesh<INPUT>, l: &Loop, loops: &DualLoops) -> Loop {
    let mut obstacles = HashMap::<FaceID, Vec<_>>::new();
    for (id, old) in loops {
        if let Ok(arcs) = old.surface_arcs(mesh) {
            for (i, a) in arcs.into_iter().enumerate() {
                obstacles.entry(a.face).or_default().push((id, i, a));
            }
        }
    }
    let cross = |a: [f64; 2], b: [f64; 2]| a[0] * b[1] - a[1] * b[0];
    let hits = |arcs: &[dualcube_dual::SurfaceArc]| {
        let mut result = Vec::new();
        for a in arcs {
            let d = [a.to[0] - a.from[0], a.to[1] - a.from[1]];
            let mut local = vec![];
            for (id, i, b) in obstacles.get(&a.face).into_iter().flatten() {
                let e = [b.to[0] - b.from[0], b.to[1] - b.from[1]];
                let delta = [b.from[0] - a.from[0], b.from[1] - a.from[1]];
                let det = cross(d, e);
                if det.abs() < 1e-15 {
                    continue;
                }
                let t = cross(delta, e) / det;
                let u = cross(delta, d) / det;
                if t > 0. && t < 1. && u > 0. && u < 1. {
                    local.push((OrderedFloat(t), (*id, *i, det > 0.)));
                }
            }
            local.sort();
            result.extend(local.into_iter().map(|x| x.1));
        }
        result
    };
    let mut arcs = l.surface.clone();
    if let Some(start) =
        (0..arcs.len()).find(|&i| arcs[i].face != arcs[(i + arcs.len() - 1) % arcs.len()].face)
    {
        arcs.rotate_left(start);
    } else {
        return l.clone();
    }
    let mut out = vec![];
    let mut i = 0;
    while i < arcs.len() {
        let end = (i + 1..arcs.len())
            .find(|&j| arcs[j].face != arcs[i].face)
            .unwrap_or(arcs.len());
        let direct = dualcube_dual::SurfaceArc {
            face: arcs[i].face,
            from: arcs[i].from,
            to: arcs[end - 1].to,
        };
        if direct.from != direct.to && hits(std::slice::from_ref(&direct)) == hits(&arcs[i..end]) {
            out.push(direct);
        } else {
            out.extend_from_slice(&arcs[i..end]);
        }
        i = end;
    }
    Loop::from_surface(l.direction, out)
}
