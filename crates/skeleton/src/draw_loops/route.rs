//! Geometry for loop drawing: the cost of a route, and a first loop on an empty surface.

use super::*;
use dualcube_dual::FlowGraph;

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

impl Flows {
    /// Cost of moving by `delta` in a triangle for a loop of `axis`: length, more when it
    /// leaves the flow field's direction.
    pub(super) fn cost(&self, axis: Direction) -> impl Fn(FaceID, Vector3D, usize) -> f64 + '_ {
        let flow = &self.vectors[match axis {
            Direction::X => 0,
            Direction::Y => 1,
            Direction::Z => 2,
        }];
        move |face, delta, _| {
            let length = delta.norm();
            let f = flow[&face];
            let confidence = f.norm().min(1.);
            let alignment = if length > 0. && confidence > 1e-12 {
                (delta.dot(&f) / (length * f.norm())).clamp(-1., 1.)
            } else {
                1.
            };
            length * (1. + 5. * confidence * (1. - alignment))
        }
    }
}

/// A loop sampled along the flow field, added alone to `map`: the first loop of a surface
/// without boundaries (a sphere, where every simple closed curve is valid).
pub(super) fn seed(
    mesh: &Mesh<INPUT>,
    flows: &Flows,
    map: &LoopMap,
    id: LoopID,
    axis: Direction,
) -> Option<LoopMap> {
    let empty = DualLoops::with_key();
    let state = LoopState::from_loops(mesh, empty);
    let sampler = LoopSampler::new(mesh, &state.loops, &state.occupied, flows.graphs.as_ref());
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
            return Some(seeded);
        }
    }
    None
}
