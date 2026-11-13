use dualcube::prelude::*;
use dualcube::skeleton::{SkeletonResult, skeletonize::deflate};
use std::sync::{Arc, Mutex};

type ContractedMesh = Mesh<deflate::CONTRACTION>;

/// Positions before the first and after every contraction iteration, in `vert_ids` order.
pub type ContractionStages = Arc<Vec<Vec<Vector3D>>>;

/// One input and iteration limit. Retaining the input Arc makes its identity stable.
#[derive(Clone, Default)]
pub struct ContractionCache(Arc<Mutex<Option<CachedContraction>>>);

struct CachedContraction {
    input: Arc<Mesh<INPUT>>,
    max_iterations: usize,
    step: f64,
    mesh: Arc<ContractedMesh>,
    stages: ContractionStages,
}

impl ContractionCache {
    pub fn get_or_compute(
        &self,
        input: Arc<Mesh<INPUT>>,
        max_iterations: usize,
        step: f64,
    ) -> SkeletonResult<Arc<ContractedMesh>> {
        let mut cached = self
            .0
            .lock()
            .map_err(|_| "Contraction cache lock poisoned")?;
        if let Some(entry) = &*cached
            && Arc::ptr_eq(&entry.input, &input)
            && entry.max_iterations == max_iterations
            && entry.step == step
        {
            info!("Reusing contracted mesh");
            return Ok(entry.mesh.clone());
        }
        let (mesh, stages) = deflate::run_with_stages(&input, max_iterations, step)?;
        let mesh = Arc::new(mesh);
        *cached = Some(CachedContraction {
            input,
            max_iterations,
            step,
            mesh: mesh.clone(),
            stages: Arc::new(stages),
        });
        Ok(mesh)
    }

    /// The stages of the cached contraction.
    pub fn stages(&self) -> Option<ContractionStages> {
        let cached = self.0.lock().ok()?;
        cached.as_ref().map(|entry| entry.stages.clone())
    }
}
