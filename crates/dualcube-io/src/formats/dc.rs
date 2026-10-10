use crate::{Export, Import};
use anyhow::{Context, bail};
use dualcube::prelude::*;
use serde::{Deserialize, Serialize};
use slotmap::SlotMap;
use std::path::Path;
use std::sync::Arc;

pub struct Dc;

const DC_MAGIC: &[u8; 4] = b"DC01";
const DC_FORMAT: &str = "dualcube.dc";
const DC_VERSION: u32 = 1;

#[derive(Debug, Deserialize)]
struct DcVersionProbe {
    pub format: String,
    pub version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DcFileV1 {
    pub format: String,
    pub version: u32,
    pub solution: DcSolutionV1,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DcSolutionV1 {
    pub mesh_ref: Arc<Mesh<INPUT>>,
    pub loops: SlotMap<LoopID, Loop>,
    pub dual: Result<Dual, PropertyViolationError>,

    #[serde(default)]
    pub polycube: Option<Polycube>,

    #[serde(skip)]
    pub layout: Option<Layout>,
}

impl From<SolutionPersistence> for DcSolutionV1 {
    fn from(solution: SolutionPersistence) -> Self {
        Self {
            mesh_ref: solution.mesh_ref,
            loops: solution.loops,
            dual: solution.dual,
            polycube: solution.polycube,
            layout: solution.layout,
        }
    }
}

impl From<DcSolutionV1> for SolutionPersistence {
    fn from(solution: DcSolutionV1) -> Self {
        Self {
            mesh_ref: solution.mesh_ref,
            loops: solution.loops,
            dual: solution.dual,
            polycube: solution.polycube,
            layout: solution.layout,
        }
    }
}

impl Export for Dc {
    fn export(solution: &Solution, path: &Path) -> anyhow::Result<()> {
        let path_save = path.with_extension("dc");
        info!("Writing Dc to {:?}", path_save);

        let persistence = solution.to_persistence();

        let dc_file = DcFileV1 {
            format: DC_FORMAT.to_string(),
            version: DC_VERSION,
            solution: persistence.into(),
        };

        let mut serialized = Vec::new();

        dc_file
            .serialize(&mut rmp_serde::Serializer::new(&mut serialized).with_struct_map())
            .context("serializing Dc file with rmp-serde")?;

        info!("Serialized Dc size: {} bytes", serialized.len());

        let compressed = zstd::stream::encode_all(std::io::Cursor::new(&serialized), 3)
            .context("compressing Dc file with zstd")?;

        info!("Compressed Dc size: {} bytes", compressed.len());

        // Write to a temporary file first and rename it over the target, so a failed
        // write never destroys an existing save.
        let path_tmp = path_save.with_extension("dc.tmp");
        let mut bytes = Vec::with_capacity(DC_MAGIC.len() + compressed.len());
        bytes.extend_from_slice(DC_MAGIC);
        bytes.extend_from_slice(&compressed);
        std::fs::write(&path_tmp, &bytes)
            .with_context(|| format!("writing {}", path_tmp.display()))?;
        std::fs::rename(&path_tmp, &path_save)
            .with_context(|| format!("moving {} to {}", path_tmp.display(), path_save.display()))?;

        info!("Successfully written Dc to {:?}", path_save);
        Ok(())
    }
}

impl Import for Dc {
    fn import(path: &Path) -> anyhow::Result<Solution> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;

        if !bytes.starts_with(DC_MAGIC) {
            bail!("not a Dc file: missing DC01 header");
        }

        let compressed = &bytes[DC_MAGIC.len()..];

        let decompressed = zstd::decode_all(std::io::Cursor::new(compressed))
            .with_context(|| format!("decompressing {}", path.display()))?;

        let probe: DcVersionProbe = rmp_serde::from_slice(&decompressed)
            .with_context(|| format!("reading Dc version from {}", path.display()))?;

        if probe.format != DC_FORMAT {
            bail!("unsupported Dc format: {}", probe.format);
        }

        let persistence = match probe.version {
            1 => {
                let file: DcFileV1 = rmp_serde::from_slice(&decompressed)
                    .with_context(|| format!("deserializing {} as Dc v1", path.display()))?;

                SolutionPersistence::from(file.solution)
            }

            version => bail!("unsupported Dc version: {}", version),
        };

        Ok(Solution::from_persistence(persistence))
    }
}
