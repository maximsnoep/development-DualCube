use crate::prelude::*;
use slotmap::SlotMap;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct InputPhase {
    pub mesh_ref: Arc<Mesh<INPUT>>,
}

impl InputPhase {
    #[must_use]
    pub fn new(mesh_ref: Arc<Mesh<INPUT>>) -> Self {
        Self { mesh_ref }
    }

    #[must_use]
    pub fn compute_flow(&self, graph_params: GraphParams) -> FlowPhase {
        let fields = Fields::new(&self.mesh_ref);
        let flow_graphs = FlowPhase::build_graphs(&self.mesh_ref, &fields, graph_params);
        FlowPhase {
            input: self.clone(),
            fields,
            flow_graphs,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FlowPhase {
    pub input: InputPhase,
    pub fields: Fields<INPUT>,
    pub flow_graphs: Arc<[grapff::fixed::FixedGraph<EdgeID, f64>; 3]>,
}

impl FlowPhase {
    #[must_use]
    pub fn build_graphs(
        mesh_ref: &Mesh<INPUT>,
        fields: &Fields<INPUT>,
        graph_params: GraphParams,
    ) -> Arc<[grapff::fixed::FixedGraph<EdgeID, f64>; 3]> {
        Arc::new(build_flow_graphs(mesh_ref, fields, graph_params))
    }
}

#[derive(Clone, Debug)]
pub struct DualPhase {
    pub input: InputPhase,
    pub loops: SlotMap<LoopID, Loop>,
    pub dual: Dual,
}

impl DualPhase {
    pub fn from_loops(
        input: InputPhase,
        loops: SlotMap<LoopID, Loop>,
    ) -> Result<Self, PropertyViolationError> {
        let dual = Dual::from(input.mesh_ref.clone(), &loops)?;
        Ok(Self { input, loops, dual })
    }

    #[must_use]
    pub fn compute_polycube(&self) -> Polycube {
        Polycube::from_dual(&self.dual)
    }

    /// Compute the primal (polycube and layout) with the given layout parameters; if given, near the corners of a
    /// similar solution (see `CornerHint`).
    pub fn compute_primal_near(
        &self,
        unit: bool,
        params: LayoutParams,
        hint: Option<&CornerHint>,
    ) -> Result<PrimalPhase, SolutionError> {
        let polycube = self.compute_polycube();
        validate_polycube(&polycube)?;
        let targets = hint.map(|hint| hint.targets(&self.dual, &polycube));

        let mut layout = None;
        for attempt in 0..params.attempts {
            match Layout::embed_best_attempt_near(
                &self.dual,
                &polycube,
                params.candidates,
                attempt,
                targets.as_ref().map(|t| (&t.corners, &t.styles)),
            ) {
                Ok(ok_layout) => {
                    layout = Some(ok_layout);
                    break;
                }
                Err(err) => {
                    debug!("compute_primal: layout attempt {attempt} failed: {err:?}");
                }
            }
        }

        let Some(layout) = layout else {
            return Err(SolutionError::NoPrimal);
        };

        let mut primal = PrimalPhase {
            dual: self.clone(),
            polycube,
            layout,
        };
        primal.resize_polycube(unit)?;
        Ok(primal)
    }
}

/// Parameters for embedding the layout.
#[derive(Clone, Copy, Debug)]
pub struct LayoutParams {
    /// Number of path insertion orders that are tried (in parallel); the shortest embedding is kept.
    pub candidates: usize,
    /// Number of times the embedding is retried if all candidates fail.
    pub attempts: usize,
}

impl Default for LayoutParams {
    fn default() -> Self {
        Self {
            candidates: 4,
            attempts: 3,
        }
    }
}

impl LayoutParams {
    /// Cheap parameters, e.g., for scoring many candidate solutions during evolution.
    #[must_use]
    pub fn fast() -> Self {
        Self {
            candidates: 1,
            attempts: 10,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PrimalPhase {
    pub dual: DualPhase,
    pub polycube: Polycube,
    pub layout: Layout,
}

impl PrimalPhase {
    pub fn resize_polycube(&mut self, unit: bool) -> Result<(), SolutionError> {
        if unit {
            self.polycube.resize(&self.dual.dual, None);
        } else {
            self.polycube.resize(&self.dual.dual, Some(&self.layout));
        }
        Ok(())
    }
}

pub fn validate_polycube(polycube: &Polycube) -> Result<(), SolutionError> {
    for face in polycube.structure.face_ids() {
        let normal = polycube.structure.normal(face);
        if normal.x.is_nan() || normal.y.is_nan() || normal.z.is_nan() {
            return Err(SolutionError::NoPolycube);
        }
    }
    Ok(())
}
