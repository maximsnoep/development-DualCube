//! The [`Solution`] type: a compatibility facade over the explicit pipeline
//! phases.
//!
//! A solution owns the current editable state used by the UI/IO layer. The
//! actual pipeline data model is represented by [`InputPhase`], [`FlowPhase`],
//! [`DualPhase`], and [`PrimalPhase`]. A solution lazily derives the downstream
//! representations:
//!
//! 1. flow fields and graphs ([`crate::flow`]),
//! 2. dual loops ([`crate::dual::loops`]),
//! 3. the dual structure ([`crate::dual::dual`]),
//! 4. the layout / embedding ([`crate::primal::layout`]),
//! 5. the polycube ([`crate::primal::polycube`]),
//! 6. the quad mesh ([`Quad`]).
//!
//! The loop bookkeeping and sampling live in [`crate::dual::loops`] and
//! [`crate::dual::sampler`]; the flow graph construction lives in
//! [`crate::flow::flowgraph`].

use crate::prelude::*;

use serde::{Deserialize, Serialize};
use slotmap::SlotMap;
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;

#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum SolutionError {
    #[error("Something is wrong with the DUAL representation: {0}")]
    DualError(#[from] PropertyViolationError),
    #[error("Something is wrong with the PRIMAL representation: {0}")]
    PrimalError(#[from] LayoutError),
    #[error("The DUAL representation is not initialized and can therefore not be modified.")]
    NoDual,
    #[error("The PRIMAL representation is not initialized and can therefore not be modified.")]
    NoPrimal,
    #[error("The POLYCUBE representation is not initialized and can therefore not be modified.")]
    NoPolycube,
    #[error("The QUAD mesh could not be constructed.")]
    QuadFailed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolutionPersistence {
    pub mesh_ref: Arc<Mesh<INPUT>>,
    pub loops: SlotMap<LoopID, Loop>,
    pub dual: Result<Dual, PropertyViolationError>,

    #[serde(default)]
    pub polycube: Option<Polycube>,

    #[serde(default)]
    pub layout: Option<Layout>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Solution {
    pub mesh_ref: Arc<Mesh<INPUT>>,
    pub loops: SlotMap<LoopID, Loop>,

    #[serde(skip)]
    pub(crate) occupied: ids::SecMap<EDGE, INPUT, Vec<LoopID>>,

    pub dual: Result<Dual, PropertyViolationError>,
    pub polycube: Option<Polycube>,
    pub layout: Option<Layout>,
    pub quad: Option<Quad>,

    #[serde(skip)]
    pub flow_graphs: Option<Arc<[grapff::fixed::FixedGraph<EdgeID, f64>; 3]>>,

    #[serde(skip)]
    pub fields: Option<Fields<INPUT>>,

    /// The criterion used to score this solution (see `get_quality`).
    #[serde(skip)]
    pub quality: QualityParams,

    /// Evolution: the misalignment per input face of this solution's layout (kept when the layout itself is dropped
    /// to save memory), used to target badly aligned regions.
    #[serde(skip)]
    pub(crate) targets: Option<Arc<Vec<(FaceID, f64)>>>,

    /// Evolution: the corners of this solution's layout (or of its parent's), kept when the layout itself is dropped;
    /// a solution with similar loops starts its layout from them (see `CornerHint`).
    #[serde(skip)]
    pub(crate) corner_hint: Option<Arc<CornerHint>>,

    /// Evolution: the penalty of every loop of this solution's layout (see `Solution::loop_penalties`), kept when the
    /// layout is dropped, so that the mutations target the loops that need it.
    #[serde(skip)]
    pub(crate) loop_penalties: Option<Arc<HashMap<LoopID, f64>>>,
}

/// The corners of a layout per loop region, to place the corners of a similar solution (e.g., a mutation) at the same
/// positions: every region of the new solution that shares most of its vertices (`HINT_OVERLAP`, in both directions)
/// with a region of this solution gets the position of that region's corner as a target (see
/// `Layout::place_all_corners_near`); the other regions are placed as usual. Optimizations of the layout are thus
/// largely kept by the loops.
#[derive(Debug)]
pub struct CornerHint {
    region_of: HashMap<VertID, usize>,
    sizes: Vec<usize>,
    positions: Vec<Vector3D>,
    // The styles of the paths (by the regions of their corners, both orientations).
    styles: HashMap<(usize, usize), PathStyle>,
}

/// The targets of a layout from a `CornerHint`: positions of the corners per region, and styles of the paths.
#[derive(Debug, Default)]
pub struct LayoutTargets {
    pub corners: HashMap<LoopRegionID, Vector3D>,
    pub styles: HashMap<EdgeKey<POLYCUBE>, PathStyle>,
}

// The fraction of the vertices that two regions must share to be the same region (see `CornerHint`).
const HINT_OVERLAP: f64 = 0.7;

impl CornerHint {
    /// The corners of the layout of a solution (`None` if it has no complete layout).
    #[must_use]
    pub fn of(solution: &Solution) -> Option<Self> {
        let dual = solution.dual.as_ref().ok()?;
        let layout = solution.layout.as_ref()?;
        let mut hint = Self {
            region_of: HashMap::new(),
            sizes: vec![],
            positions: vec![],
            styles: HashMap::new(),
        };
        let mut index_of = HashMap::new();
        for region in dual.loop_structure.face_ids() {
            let Some(corner) = layout.polycube_ref.region_to_vertex.get_by_left(&region) else {
                continue;
            };
            let Some(&vert) = layout.vert_to_corner.get_by_left(corner) else {
                continue;
            };
            let verts = dual.region_to_verts(region);
            if verts.is_empty() {
                continue;
            }
            let index = hint.positions.len();
            index_of.insert(*corner, index);
            hint.sizes.push(verts.len());
            hint.positions.push(layout.granulated_mesh.position(vert));
            for v in verts {
                hint.region_of.insert(v, index);
            }
        }
        let structure = &layout.polycube_ref.structure;
        for (&edge, &style) in &layout.path_styles {
            if let Some([u, v]) = structure.vertices(edge).collect_array::<2>()
                && let (Some(&a), Some(&b)) = (index_of.get(&u), index_of.get(&v))
            {
                hint.styles.insert((a, b), style);
                hint.styles.insert((b, a), style);
            }
        }
        Some(hint)
    }

    /// The targets of the corners of the regions of a (similar) dual structure, and of the styles of the paths
    /// between matched corners.
    #[must_use]
    pub fn targets(&self, dual: &Dual, polycube: &Polycube) -> LayoutTargets {
        let mut targets = LayoutTargets::default();
        let mut matched = HashMap::new();
        for region in dual.loop_structure.face_ids() {
            if polycube.region_to_vertex.get_by_left(&region).is_none() {
                continue;
            }
            let verts = dual.region_to_verts(region);
            let mut counts: HashMap<usize, usize> = HashMap::new();
            for v in &verts {
                if let Some(&index) = self.region_of.get(v) {
                    *counts.entry(index).or_default() += 1;
                }
            }
            if let Some((index, count)) = counts.into_iter().max_by_key(|&(_, count)| count)
                && count as f64 >= HINT_OVERLAP * verts.len() as f64
                && count as f64 >= HINT_OVERLAP * self.sizes[index] as f64
            {
                targets.corners.insert(region, self.positions[index]);
                if let Some(&corner) = polycube.region_to_vertex.get_by_left(&region) {
                    matched.insert(corner, index);
                }
            }
        }
        for edge in polycube.structure.edge_ids() {
            if let Some([u, v]) = polycube.structure.vertices(edge).collect_array::<2>()
                && let (Some(&a), Some(&b)) = (matched.get(&u), matched.get(&v))
                && let Some(&style) = self.styles.get(&(a, b))
            {
                targets.styles.insert(edge, style);
            }
        }
        targets
    }
}

impl Clone for Solution {
    fn clone(&self) -> Self {
        Self {
            mesh_ref: self.mesh_ref.clone(),
            loops: self.loops.clone(),
            occupied: self.occupied.clone(),
            dual: self.dual.clone(),
            polycube: self.polycube.clone(),
            layout: self.layout.clone(),
            quad: self.quad.clone(),
            flow_graphs: self.flow_graphs.clone(),
            fields: self.fields.clone(),
            quality: self.quality,
            targets: self.targets.clone(),
            corner_hint: self.corner_hint.clone(),
            loop_penalties: self.loop_penalties.clone(),
        }
    }
}

impl Solution {
    pub fn to_persistence(&self) -> SolutionPersistence {
        SolutionPersistence {
            mesh_ref: self.mesh_ref.clone(),
            loops: self.loops.clone(),
            dual: self.dual.clone(),
            polycube: self.polycube.clone(),
            layout: self.layout.clone(),
        }
    }

    pub fn from_persistence(data: SolutionPersistence) -> Self {
        // Loops stored without offsets (older files) are ordered along their edges here.
        let LoopState {
            loops, occupied, ..
        } = LoopState::from_loops(&data.mesh_ref, data.loops);
        Self {
            fields: None,

            occupied,
            loops,

            dual: data.dual,

            polycube: data.polycube,
            layout: data.layout,

            quad: None,

            flow_graphs: None,

            quality: QualityParams::default(),
            targets: None,
            corner_hint: None,
            loop_penalties: None,

            mesh_ref: data.mesh_ref,
        }
    }

    pub fn clear(&mut self) {
        self.dual = Err(PropertyViolationError::default());
        self.polycube = None;
        self.layout = None;
        self.quad = None;
    }

    // ***
    // STEPS OF A SOLUTION
    // ***

    /// Create a new (empty) solution from an input mesh.
    pub fn new(mesh_ref: Arc<Mesh<INPUT>>) -> Self {
        Self {
            mesh_ref: mesh_ref.clone(),
            loops: SlotMap::with_key(),
            occupied: ids::SecMap::new(),
            dual: Err(PropertyViolationError::default()),
            polycube: None,
            layout: None,
            quad: None,
            // Flow fields and graphs are computed lazily (see `prepare_flow`), so we
            // don't waste work building them for solutions that are only loaded or
            // reconstructed rather than initialized from scratch.
            fields: None,
            flow_graphs: None,
            quality: QualityParams::default(),
            targets: None,
            corner_hint: None,
            loop_penalties: None,
        }
    }

    /// Ensure the flow fields and per-axis flow graphs exist.
    ///
    /// Loop sampling traces loops through the flow graphs, which are derived from
    /// the flow fields, so these must be computed before any loop can be found.
    /// The fields are axis-guided so that the X/Y/Z directions get a consistent
    /// global meaning, which is what we want for polycube construction.
    pub fn prepare_flow(&mut self) {
        if self.flow_graphs.is_some() {
            return;
        }
        let input = InputPhase::new(self.mesh_ref.clone());
        let flow = input.compute_flow(GraphParams::default());
        self.fields = Some(flow.fields);
        self.flow_graphs = Some(flow.flow_graphs);
    }

    /// Build the three per-axis flow graphs from the current flow fields.
    pub fn set_flow_graphs(&mut self, params: GraphParams) {
        let Some(fields) = &self.fields else {
            warn!("Cannot build flow graphs: vector fields are missing.");
            self.flow_graphs = None;
            return;
        };

        self.flow_graphs = Some(Arc::new(build_flow_graphs(&self.mesh_ref, fields, params)));
        info!("flow graphs set");
    }

    /// Construct the dual structure and the polycube from the current loops.
    pub fn construct_dual_and_polycube(&mut self) -> Result<(), PropertyViolationError> {
        self.layout = None;
        self.polycube = None;

        let dual_phase =
            DualPhase::from_loops(InputPhase::new(self.mesh_ref.clone()), self.loops.clone())?;
        let polycube = dual_phase.compute_polycube();

        if let Err(err) = validate_polycube(&polycube) {
            warn!("construct_dual_and_polycube: invalid polycube: {err:?}");
            return Err(PropertyViolationError::UnknownError);
        }

        self.dual = Ok(dual_phase.dual);
        self.polycube = Some(polycube);

        Ok(())
    }

    /// Place all polycube corners on the input mesh.
    pub fn place_corners(&mut self) -> Result<(), SolutionError> {
        if self.dual.is_err() {
            return Err(SolutionError::NoDual);
        }
        if self.polycube.is_none() {
            return Err(SolutionError::NoPolycube);
        }

        let mut layout = Layout::new(self.dual.as_ref().unwrap(), self.polycube.as_ref().unwrap());
        layout.place_all_corners();
        self.layout = Some(layout);
        Ok(())
    }

    /// Move a single corner to a new mesh vertex.
    pub fn move_corner_to(
        &mut self,
        corner: VertKey<POLYCUBE>,
        new_vertex: VertID,
    ) -> Result<(), SolutionError> {
        if self.layout.is_none() {
            return Err(SolutionError::NoPrimal);
        }

        let mut layout = self.layout.clone().unwrap();
        layout.move_corner(corner, new_vertex)?;
        self.layout = Some(layout);

        Ok(())
    }

    /// Place all polycube edge paths and assign the resulting patches.
    pub fn place_paths(&mut self) -> Result<(), SolutionError> {
        if self.layout.is_none() {
            return Err(SolutionError::NoPrimal);
        }
        let layout = self.layout.as_mut().unwrap();
        layout.place_paths_best(LayoutParams::default().candidates)?;
        Ok(())
    }

    /// Straighten the layout paths (to locally shortest paths, keeping the patch topology).
    pub fn straighten_paths(&mut self) -> Result<StraightenStats, SolutionError> {
        let layout = self
            .layout
            .as_mut()
            .filter(|layout| layout.is_complete())
            .ok_or(SolutionError::NoPrimal)?;
        Ok(layout.straighten_paths()?)
    }

    /// Whether the current loops induce a valid dual structure.
    pub fn dual_is_ok(&self) -> bool {
        Dual::from(self.mesh_ref.clone(), &self.loops).is_ok()
    }

    /// Rebuild the full chain (dual, polycube, layout, quad) from the loops.
    pub fn reconstruct_solution(&mut self, unit: bool) -> Result<(), SolutionError> {
        self.reconstruct_solution_with(unit, LayoutParams::default())
    }

    /// Rebuild the full chain from the loops, with the given parameters for embedding the layout.
    pub fn reconstruct_solution_with(
        &mut self,
        unit: bool,
        params: LayoutParams,
    ) -> Result<(), SolutionError> {
        self.reconstruct_near(unit, params, None)
    }

    /// See `reconstruct_solution_with`; the corners near those of the given hint (see `CornerHint`).
    pub fn reconstruct_near(
        &mut self,
        unit: bool,
        params: LayoutParams,
        hint: Option<&CornerHint>,
    ) -> Result<(), SolutionError> {
        let started_at = Instant::now();
        // Reuse the dual structure if it is up to date (e.g., built when the loops were added during the evolution).
        let current = self
            .dual_is_current()
            .then(|| std::mem::replace(&mut self.dual, Err(PropertyViolationError::default())));
        self.clear();

        if self.loops.len() < 3 {
            return Ok(());
        }

        let dual_timer = Instant::now();
        let input = InputPhase::new(self.mesh_ref.clone());
        let dual_phase = match current {
            Some(Ok(dual)) => DualPhase {
                input,
                loops: self.loops.clone(),
                dual,
            },
            _ => DualPhase::from_loops(input, self.loops.clone())?,
        };
        let dual_ms = dual_timer.elapsed();

        let primal_timer = Instant::now();
        let primal = dual_phase.compute_primal_near(unit, params, hint)?;
        let primal_ms = primal_timer.elapsed();

        self.dual = Ok(primal.dual.dual);
        self.polycube = Some(primal.polycube);
        self.layout = Some(primal.layout);

        debug!(
            "reconstruct_solution: loops={} dual={dual_ms:?} primal={primal_ms:?} total={:?}",
            self.loops.len(),
            started_at.elapsed()
        );

        Ok(())
    }

    pub fn resize_polycube(&mut self, unit: bool) -> Result<(), SolutionError> {
        if self.dual.is_err() {
            return Err(SolutionError::NoDual);
        }
        let dual = self.dual.as_ref().unwrap();
        if self.polycube.is_none() {
            return Err(SolutionError::NoPolycube);
        }
        let polycube = self.polycube.as_mut().unwrap();

        if unit {
            polycube.resize(dual, None);
            Ok(())
        } else {
            if self.layout.is_none() {
                return Err(SolutionError::NoPrimal);
            }
            let layout = self.layout.as_ref().unwrap();
            polycube.resize(dual, Some(layout));
            Ok(())
        }
    }

    /// Constructs the quad mesh of the layout (`self.quad`; `None` on failure).
    pub fn construct_quad(&mut self, density: QuadDensity) -> Result<(), SolutionError> {
        let layout = self.layout.as_ref().ok_or(SolutionError::NoPrimal)?;
        self.quad = build_quad_from_layout(layout, density);
        self.quad.as_ref().ok_or(SolutionError::QuadFailed)?;
        Ok(())
    }

    /// The quality of the solution according to its quality criterion (`self.quality`), or `None` if the solution
    /// has no complete layout. Only the terms needed by the criterion are computed.
    pub fn get_quality(&self) -> Option<f64> {
        let layout = self.layout.as_ref()?;
        let weights = self.quality.weights;
        QualityReport::compute(
            self.loops.len(),
            layout,
            &self.quality,
            QualityTerms::needed(&weights),
        )
        .score(&weights)
    }

    /// All quality terms of the solution (requires a complete layout).
    pub fn quality_report(&self) -> Option<QualityReport> {
        let layout = self.layout.as_ref()?;
        Some(QualityReport::compute(
            self.loops.len(),
            layout,
            &self.quality,
            QualityTerms::all(),
        ))
    }

    /// A cheap estimate of the quality from the dual structure only (see `QualityReport::estimate`), or `None` if
    /// the loops do not form a valid dual structure.
    pub fn estimate_quality(&self) -> Option<f64> {
        let dual = self.current_dual()?;
        let polycube = Polycube::from_dual(&dual);
        validate_polycube(&polycube).ok()?;
        let weights = self.quality.weights;
        Some(
            QualityReport::estimate(
                self.loops.len(),
                &dual,
                &polycube,
                &self.quality,
                QualityTerms::needed(&weights),
            )
            .partial_score(&weights),
        )
    }
}
