use dualcube::prelude::*;
use mehsh::prelude::*;
use std::collections::HashMap;
use std::io::Write;

pub struct Flag;

impl crate::Export for Flag {
    fn export(
        solution: &dualcube::prelude::Solution,
        path: &std::path::Path,
    ) -> anyhow::Result<()> {
        let Some(layout) = solution.layout.as_ref() else {
            anyhow::bail!("No layout available");
        };

        let face_ids = layout.granulated_mesh.face_ids();
        let face_map = face_ids
            .iter()
            .copied()
            .enumerate()
            .map(|(i, face_id)| (face_id, i))
            .collect::<HashMap<_, _>>();

        let mut labels = vec![-1; face_ids.len()];

        for (&patch_id, patch_faces) in &layout.face_to_patch {
            let label = match to_principal_direction(
                layout.polycube_ref.structure.normal(patch_id).normalize(),
            ) {
                (PrincipalDirection::X, Orientation::Forwards) => 0,
                (PrincipalDirection::X, Orientation::Backwards) => 1,
                (PrincipalDirection::Y, Orientation::Forwards) => 2,
                (PrincipalDirection::Y, Orientation::Backwards) => 3,
                (PrincipalDirection::Z, Orientation::Forwards) => 4,
                (PrincipalDirection::Z, Orientation::Backwards) => 5,
            };

            for &face_id in &patch_faces.faces {
                labels[face_map[&face_id]] = label;
            }
        }

        let path_flag = path.with_extension("flag");

        let mut file_flag =
            anyhow::Context::with_context(std::fs::File::create(&path_flag), || {
                format!("creating FLAG file at {}", path_flag.display())
            })?;

        anyhow::Context::with_context(
            write!(
                file_flag,
                "{}",
                labels
                    .iter()
                    .map(i32::to_string)
                    .collect::<Vec<_>>()
                    .join("\n")
            ),
            || format!("writing FLAG contents to {}", path_flag.display()),
        )?;

        Ok(())
    }
}
