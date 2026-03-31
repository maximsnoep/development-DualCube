use crate::{Export, Import};
use dualcube::prelude::Solution;
use log::info;
use serde::{Deserialize, Serialize};
use std::io::Write;

pub struct Dsol;

#[derive(Serialize, Deserialize)]
pub struct DsolSerialization {
    pub solution: Solution,
}

impl Export for Dsol {
    fn export(
        solution: &dualcube::prelude::Solution,
        path: &std::path::Path,
    ) -> anyhow::Result<()> {
        let path_save = path.with_extension("dsol");
        info!("Writing Dsol to {:?}", path_save);

        let mut cloned_solution = solution.clone();
        cloned_solution.quad = None;
        cloned_solution.last_loop = None;

        let serialized = anyhow::Context::with_context(
            rmp_serde::to_vec(&DsolSerialization {
                solution: cloned_solution,
            }),
            || "serializing solution (rmp-serde)".to_string(),
        )?;
        info!("Serialized size (rmp): {} bytes", serialized.len());

        let compressed = anyhow::Context::with_context(
            zstd::stream::encode_all(std::io::Cursor::new(&serialized), 3),
            || "compressing solution (zstd level 3)".to_string(),
        )?;
        info!("Compressed size (zstd): {} bytes", compressed.len());

        let mut file = anyhow::Context::with_context(std::fs::File::create(&path_save), || {
            format!("creating {}", path_save.display())
        })?;

        anyhow::Context::with_context(file.write_all(&compressed), || {
            format!("writing {}", path_save.display())
        })?;

        info!("Successfully written Dsol to {:?}", path_save);
        Ok(())
    }
}

impl Import for Dsol {
    fn import(path: &std::path::Path) -> anyhow::Result<dualcube::prelude::Solution> {
        let file = anyhow::Context::with_context(std::fs::File::open(path), || {
            format!("opening {}", path.display())
        })?;

        let reader = std::io::BufReader::new(file);

        let decompressed = anyhow::Context::with_context(zstd::decode_all(reader), || {
            format!("decompressing {}", path.display())
        })?;

        let serialized: DsolSerialization =
            anyhow::Context::with_context(rmp_serde::from_slice(&decompressed), || {
                format!("deserializing {} (rmp-serde)", path.display())
            })?;

        let solution = serialized.solution;

        Ok(solution)
    }
}
