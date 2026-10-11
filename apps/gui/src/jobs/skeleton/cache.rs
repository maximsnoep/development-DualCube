use dualcube::prelude::*;
use dualcube::skeleton::{SkeletonResult, skeletonize::deflate};
use std::sync::{Arc, Mutex};

type ContractedMesh = Mesh<deflate::CONTRACTION>;

/// One input and iteration limit. Retaining the input Arc makes its identity stable.
#[derive(Clone, Default)]
pub struct ContractionCache(Arc<Mutex<Option<CachedContraction>>>);

struct CachedContraction {
    input: Arc<Mesh<INPUT>>,
    max_iterations: usize,
    mesh: Arc<ContractedMesh>,
}

impl ContractionCache {
    pub fn get_or_compute(
        &self,
        input: Arc<Mesh<INPUT>>,
        max_iterations: usize,
    ) -> SkeletonResult<Arc<ContractedMesh>> {
        let mut cached = self
            .0
            .lock()
            .map_err(|_| "Contraction cache lock poisoned")?;
        if let Some(entry) = &*cached
            && Arc::ptr_eq(&entry.input, &input)
            && entry.max_iterations == max_iterations
        {
            info!("Reusing contracted mesh");
            return Ok(entry.mesh.clone());
        }
        let mesh = Arc::new(deflate::run(&input, max_iterations)?);
        *cached = Some(CachedContraction {
            input,
            max_iterations,
            mesh: mesh.clone(),
        });
        Ok(mesh)
    }
}
