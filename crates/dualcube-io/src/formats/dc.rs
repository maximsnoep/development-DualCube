use crate::{Export, Import};
use anyhow::{Context, bail};
use dualcube::prelude::*;
use dualcube::skeleton::skeletonize::{Extraction, deflate::CONTRACTION};
use dualcube::skeleton::{
    DualLoops, LabeledSkeleton, RealizedSkeleton, draw_loops, label, pipeline,
};
use serde::{Deserialize, Serialize};
use slotmap::SlotMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub struct Dc;

const DC_MAGIC: &[u8; 4] = b"DC01";
const DC_FORMAT: &str = "dualcube.dc";
const DC_VERSION: u32 = 2;

#[derive(Debug, Deserialize)]
struct DcVersionProbe {
    pub format: String,
    pub version: u32,
}

#[derive(Debug, Deserialize)]
struct DcFileV1 {
    pub solution: DcSolutionV1,
}

#[derive(Debug, Deserialize)]
struct DcSolutionV1 {
    pub mesh_ref: Arc<Mesh<INPUT>>,
    pub loops: SlotMap<LoopID, Loop>,
    pub dual: Result<Dual, PropertyViolationError>,

    #[serde(default)]
    pub polycube: Option<Polycube>,

    #[serde(skip)]
    pub layout: Option<Layout>,
}

// v2 adds the layout (v1 skipped it), the skeleton stages, and the source of the model.
#[derive(Serialize)]
struct DcFileV2<'a> {
    format: String,
    version: u32,
    solution: SolutionRef<'a>,
    skeleton: Option<&'a SkeletonSave>,
    source: Option<PathBuf>,
}

// A solution serialized as its `SolutionPersistence`, without copying it.
#[derive(Serialize)]
struct SolutionRef<'a> {
    mesh_ref: &'a Arc<Mesh<INPUT>>,
    loops: &'a SlotMap<LoopID, Loop>,
    dual: &'a Result<Dual, PropertyViolationError>,
    polycube: &'a Option<Polycube>,
    layout: &'a Option<Layout>,
}

#[derive(Deserialize)]
struct DcFileV2Owned {
    solution: SolutionPersistence,
    #[serde(default)]
    skeleton: Option<SkeletonSave>,
    #[serde(default)]
    source: Option<PathBuf>,
}

/// The extraction state: deflated only, or extracted.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SkeletonExtraction {
    Deflated(Mesh<CONTRACTION>),
    Extracted(Extraction),
}

/// The skeleton initialization's stages as the GUI retains them; a failure keeps its message only.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SkeletonSave {
    pub extraction: Option<SkeletonExtraction>,
    pub labeling: Option<label::Attempt>,
    pub labeled: Option<LabeledSkeleton>,
    pub realized: Option<RealizedSkeleton>,
    pub optimized: Option<RealizedSkeleton>,
    pub drawing: Option<draw_loops::Attempt>,
    pub loops: Option<DualLoops>,
    pub failure: Option<(pipeline::Stage, String)>,
    /// The solution the initialization reconstructed, before any evolution.
    #[serde(default)]
    pub output: Option<SolutionParts>,
}

/// A solution without its mesh (it shares the mesh of the solution it is saved with).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SolutionParts {
    pub loops: SlotMap<LoopID, Loop>,
    pub dual: Result<Dual, PropertyViolationError>,
    pub polycube: Option<Polycube>,
    pub layout: Option<Layout>,
}

impl SolutionParts {
    #[must_use]
    pub fn of(solution: &Solution) -> Self {
        let persistence = solution.to_persistence();
        Self {
            loops: persistence.loops,
            dual: persistence.dual,
            polycube: persistence.polycube,
            layout: persistence.layout,
        }
    }

    #[must_use]
    pub fn with_mesh(self, mesh_ref: Arc<Mesh<INPUT>>) -> Solution {
        Solution::from_persistence(SolutionPersistence {
            mesh_ref,
            loops: self.loops,
            dual: self.dual,
            polycube: self.polycube,
            layout: self.layout,
        })
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
        write(solution, None, None, path)
    }
}

impl Import for Dc {
    fn import(path: &Path) -> anyhow::Result<Solution> {
        Ok(read(path)?.solution)
    }
}

/// The contents of a Dc file.
pub struct Save {
    pub solution: Solution,
    /// Present in v2 files saved with skeleton state.
    pub skeleton: Option<SkeletonSave>,
    /// The file the model was first read from (as the saving program knew it).
    pub source: Option<PathBuf>,
}

/// Writes a v2 Dc file: the solution with its layout, the skeleton stages, and the source of the model.
pub fn write(
    solution: &Solution,
    skeleton: Option<&SkeletonSave>,
    source: Option<&Path>,
    path: &Path,
) -> anyhow::Result<()> {
    let path_save = path.with_extension("dc");
    info!("Writing Dc to {:?}", path_save);

    let dc_file = DcFileV2 {
        format: DC_FORMAT.to_string(),
        version: DC_VERSION,
        solution: SolutionRef {
            mesh_ref: &solution.mesh_ref,
            loops: &solution.loops,
            dual: &solution.dual,
            polycube: &solution.polycube,
            layout: &solution.layout,
        },
        skeleton,
        source: source.map(Path::to_path_buf),
    };

    // Streamed through zstd into a temporary file (nothing held in memory), renamed over the target afterwards, so a
    // failed write never destroys an existing save.
    let path_tmp = path_save.with_extension("dc.tmp");
    let write = || -> anyhow::Result<()> {
        let mut file = std::io::BufWriter::new(std::fs::File::create(&path_tmp)?);
        file.write_all(DC_MAGIC)?;
        let mut encoder = zstd::stream::Encoder::new(file, 3)?;
        dc_file
            .serialize(&mut rmp_serde::Serializer::new(&mut encoder).with_struct_map())
            .context("serializing Dc file with rmp-serde")?;
        encoder.finish()?.flush()?;
        Ok(())
    };
    write().with_context(|| format!("writing {}", path_tmp.display()))?;
    std::fs::rename(&path_tmp, &path_save)
        .with_context(|| format!("moving {} to {}", path_tmp.display(), path_save.display()))?;

    info!("Successfully written Dc to {:?}", path_save);
    Ok(())
}

/// Reads a v1 or v2 Dc file.
pub fn read(path: &Path) -> anyhow::Result<Save> {
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

    match probe.version {
        1 => {
            let file: DcFileV1 = rmp_serde::from_slice(&decompressed)
                .with_context(|| format!("deserializing {} as Dc v1", path.display()))?;
            Ok(Save {
                solution: Solution::from_persistence(file.solution.into()),
                skeleton: None,
                source: None,
            })
        }
        2 => {
            let file: DcFileV2Owned = rmp_serde::from_slice(&decompressed)
                .with_context(|| format!("deserializing {} as Dc v2", path.display()))?;
            Ok(Save {
                solution: Solution::from_persistence(file.solution),
                skeleton: file.skeleton,
                source: file.source,
            })
        }
        version => bail!("unsupported Dc version: {}", version),
    }
}
