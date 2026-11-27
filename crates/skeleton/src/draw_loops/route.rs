//! Geometry for loop drawing: the cost of a route, and a first loop on an empty surface.

use super::*;
use dualcube_dual::FlowGraph;

pub(super) struct Flows {
    normals: HashMap<FaceID, Vector3D>,
    graphs: Option<[FlowGraph; 3]>,
}

impl Flows {
    pub(super) fn new(mesh: &Mesh<INPUT>, needs_seed: bool) -> Self {
        let normals = mesh
            .face_ids()
            .into_iter()
            .map(|f| (f, mesh.normal(f).normalize()))
            .collect();
        let graphs = needs_seed.then(|| {
            let fields = dualcube_flow::Fields::new(mesh);
            dualcube_flow::build_flow_graphs(
                mesh,
                &fields,
                dualcube_flow::GraphParams {
                    max_deviation: std::f64::consts::PI,
                    ..Default::default()
                },
            )
        });
        Self { normals, graphs }
    }
}

/// The strictness factor alpha of the alignment of moves of DualCube (Snoep et al. 2025).
const STRICTNESS: i32 = 10;

impl Flows {
    /// Cost of moving by `delta` in a triangle for a loop of `axis`: its length, plus its length
    /// times DualCube's weight of a move, `angle(delta × n, axis)^alpha` (Snoep et al. 2025, Sec.
    /// 4.1), which is 0 for a move around the axis and grows steeply as it turns away.
    pub(super) fn cost(&self, axis: Direction) -> impl Fn(FaceID, Vector3D, usize) -> f64 + '_ {
        let target = Vector3D::from(axis);
        move |face, delta, _| {
            let side = delta.cross(&self.normals[&face]);
            let angle = if side.norm() > 1e-12 {
                side.angle(&target)
            } else {
                std::f64::consts::FRAC_PI_2
            };
            delta.norm() * (1. + angle.powi(STRICTNESS))
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
