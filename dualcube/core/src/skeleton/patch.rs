use mehsh::prelude::Mesh;

use crate::{
    prelude::{CurveSkeleton, INPUT},
    skeleton::curve_skeleton::patch_centroid,
};

/// A version of `SurgeryContext::refine_embedding` without the context wrapping.
#[allow(dead_code)]
pub fn refine_embedding(skeleton: &mut CurveSkeleton, original_mesh: &Mesh<INPUT>) {
    for node_idx in skeleton.node_indices() {
        let verts = &skeleton[node_idx].patch_vertices;
        if verts.is_empty() {
            continue;
        }

        let centroid = patch_centroid(verts, original_mesh);

        skeleton[node_idx].position = centroid;
    }
}
