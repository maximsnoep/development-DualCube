//! Shared helpers of the integration tests.

use dualcube::prelude::*;
use std::sync::Arc;

/// The (coarse) test mesh.
pub fn blub() -> Arc<Mesh<INPUT>> {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../mehsh/assets/blub001k.obj");
    Arc::new(Mesh::<INPUT>::from_file(&path).unwrap().0)
}
