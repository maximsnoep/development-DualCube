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
    fn export(solution: &dualcube::prelude::Solution, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let path_save = path.with_extension("dsol");
        info!("Writing Dsol to {path_save:?}");
        let mut file = std::fs::File::create(path_save.clone())?;

        let mut cloned_solution = solution.clone();
        cloned_solution.quad = None;
        cloned_solution.last_loop = None;

        let serialized = rmp_serde::to_vec(&DsolSerialization { solution: cloned_solution })?;
        info!("Serialized size (bitcode): {} bytes", serialized.len());

        let compressed = zstd::stream::encode_all(std::io::Cursor::new(&serialized), 3)?;
        info!("Compressed size (zstd): {} bytes", compressed.len());

        file.write_all(&compressed)?;
        info!("Successfully written Dsol to {path_save:?}");

        Ok(())
    }
}

impl Import for Dsol {
    fn import(path: &std::path::Path) -> Result<dualcube::prelude::Solution, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let decompressed = zstd::decode_all(reader)?;
        let serialized: DsolSerialization = rmp_serde::from_slice(&decompressed)?;
        Ok(serialized.solution)
    }
}
