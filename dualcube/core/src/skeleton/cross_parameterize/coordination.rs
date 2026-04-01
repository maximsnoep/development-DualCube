use std::collections::HashMap;

use petgraph::graph::EdgeIndex;

use crate::prelude::EdgeID;

use super::{CutPath, SurfacePath};

/// For one boundary loop of a region: the oriented, ordered slot list.
///
/// `slots` contains the boundary edge midpoints in CCW order (patch on left),
/// after applying `orientation_reversed` to the raw `BoundaryLoop::edge_midpoints`.
/// `slot_id` values throughout the coordination system are indices into this `slots` vec.
#[derive(Debug, Clone)]
pub struct BoundaryFrame {
    /// The skeleton edge that labels this boundary loop.
    pub skeleton_edge: EdgeIndex,
    /// Whether the raw `BoundaryLoop::edge_midpoints` order was reversed to achieve CCW.
    pub orientation_reversed: bool,
    /// Edge midpoints in CCW orientation order (len = #edges in boundary loop).
    pub slots: Vec<EdgeID>,
}

impl BoundaryFrame {
    /// Returns the edge ID for the given slot index.
    pub fn slot_edge(&self, slot_id: usize) -> EdgeID {
        self.slots[slot_id]
    }

    /// Number of available slots on this boundary.
    pub fn num_slots(&self) -> usize {
        self.slots.len()
    }

    /// Returns true if `slot_a` and `slot_b` are adjacent (distance 1 mod N).
    /// Adjacent endpoints produce a zero-length polygon segment and are infeasible.
    pub fn slots_adjacent(&self, slot_a: usize, slot_b: usize) -> bool {
        let n = self.slots.len();
        let diff = slot_a.abs_diff(slot_b);
        diff == 1 || diff == n - 1
    }
}

/// One endpoint of a cut assigned to a specific slot on a specific boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CutEndpointSpec {
    /// Index into `RegionCoordination::polycube_cuts` / `input_cuts`.
    pub cut_id: usize,
    /// Which boundary loop (skeleton edge) this endpoint lands on.
    pub boundary: EdgeIndex,
    /// Index into `BoundaryFrame::slots` for this boundary.
    pub slot_id: usize,
    /// True = this endpoint is the *start* of the cut; false = the *end*.
    pub is_start: bool,
}

/// The canonical cyclic order of all cut-endpoint events around the opened disk boundary.
///
/// `events[0]` is the **phase anchor**: both sides start their boundary walk here,
/// ensuring identical segment → polygon-side assignment on input and polycube.
///
/// The events are in CCW order around the virtual boundary (after all cuts open the disk).
#[derive(Debug, Clone)]
pub struct CutCycleOrder {
    /// All cut-endpoint events in cyclic CCW order.  Length = 2 × (number of cuts).
    pub events: Vec<CutEndpointSpec>,
}

impl CutCycleOrder {
    /// Returns the polygon segment index for the given `(cut_id, is_start)` pair,
    /// i.e. which polygon side the arc *starting* at that endpoint maps to.
    /// Returns `None` if the spec is not found (should not happen after construction).
    pub fn segment_index_for(&self, cut_id: usize, is_start: bool) -> Option<usize> {
        self.events
            .iter()
            .position(|e| e.cut_id == cut_id && e.is_start == is_start)
    }
}

/// The complete shared coordination artifact for one region.
///
/// Computed once, then consumed by both the input and polycube parameterization paths.
/// Guarantees:
/// 1. Both sides attach cuts at the same canonical boundary slots.
/// 2. Both sides walk the boundary starting from the same phase anchor.
/// 3. Both sides assign boundary arcs to the same polygon sides.
#[derive(Debug, Clone)]
pub struct RegionCoordination {
    /// Oriented boundary frames for the polycube side, keyed by skeleton edge.
    pub polycube_frames: HashMap<EdgeIndex, BoundaryFrame>,
    /// Oriented boundary frames for the input side, keyed by skeleton edge.
    pub input_frames: HashMap<EdgeIndex, BoundaryFrame>,
    /// The 2 × (d-1) endpoint specs (one start + one end per cut).
    /// Ordered: cut 0 start, cut 0 end, cut 1 start, cut 1 end, …
    pub endpoint_specs: Vec<CutEndpointSpec>,
    /// Canonical cyclic endpoint order; drives phase anchor and segment assignment.
    pub cycle_order: CutCycleOrder,
    /// Polycube-side cut paths (computed during candidate selection).
    pub polycube_cuts: Vec<CutPath>,
    /// Input-side cut paths (constrained-realized from shared slot assignment).
    pub input_cuts: Vec<CutPath>,
}

/// One feasible polycube candidate: a specific slot assignment for all cuts,
/// together with the realized polycube paths and their total length score.
#[derive(Debug, Clone)]
pub struct PolycubeCandidate {
    /// For each cut: (start_boundary, start_slot, end_boundary, end_slot).
    pub assignments: Vec<(EdgeIndex, usize, EdgeIndex, usize)>,
    /// Realized polycube cut paths (in cut_topology order).
    pub cuts: Vec<CutPath>,
    /// Sum of polycube path lengths (primary score).
    pub primary_score: f64,
}

/// Computes the total arc-length of a surface path on the given mesh.
pub fn path_length(path: &SurfacePath, mesh: &mehsh::prelude::Mesh<crate::prelude::INPUT>) -> f64 {
    use mehsh::prelude::HasPosition;
    let mut len = 0.0;

    let positions: Vec<mehsh::prelude::Vector3D> = {
        let mut v = Vec::new();
        v.push(super::edge_id_to_midpoint_pos(path.start, mesh));
        for &vert in &path.interior_verts {
            v.push(mesh.position(vert));
        }
        v.push(super::edge_id_to_midpoint_pos(path.end, mesh));
        v
    };

    for w in positions.windows(2) {
        len += (w[1] - w[0]).norm();
    }
    len
}
