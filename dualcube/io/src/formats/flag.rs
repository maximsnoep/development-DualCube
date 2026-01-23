use crate::Export;
use dualcube::prelude::*;
use log::info;
use mehsh::prelude::*;
use std::collections::HashMap;
use std::io::Write;

pub struct Flag;

impl Export for Flag {
    fn export(
        solution: &dualcube::prelude::Solution,
        path: &std::path::Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let path_flag = path.with_extension("flag");

        if let Some(layout) = &solution.layout {
            let face_map = layout
                .granulated_mesh
                .face_ids()
                .into_iter()
                .enumerate()
                .map(|(i, face_id)| (face_id, i))
                .collect::<HashMap<_, _>>();
            let mut labels = vec![-1; layout.granulated_mesh.face_ids().len()];

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

            info!("Writing FLAG file to {path_flag:?}");
            let mut file_flag = std::fs::File::create(path_flag)?;
            write!(
                file_flag,
                "{}",
                labels
                    .iter()
                    .map(i32::to_string)
                    .collect::<Vec<_>>()
                    .join("\n")
            )?;
            return Ok(());
        }
        Err(Box::new(std::io::Error::other("No layout available")))
    }
}
