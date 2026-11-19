pub mod figure;
pub mod formats {
    pub mod apg;
    pub mod dc;
    pub mod flag;
    pub mod hex;
    pub mod loops;
    pub mod nlr;
    pub mod obj;
}

pub trait Export {
    fn export(solution: &Solution, path: &Path) -> anyhow::Result<()>;
}

pub trait Import {
    fn import(path: &Path) -> anyhow::Result<Solution>;
}

pub use crate::formats::{
    apg::APG,
    dc::{Dc, Save, SkeletonExtraction, SkeletonSave, SolutionParts},
    flag::Flag,
    hex::HEX,
    loops::Loops,
    nlr::NLR,
    obj::OBJ,
};
use dualcube::prelude::*;
use std::{path::Path, sync::Arc};

fn extension_of(path: &Path) -> anyhow::Result<String> {
    path.extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .ok_or_else(|| anyhow::anyhow!("{} has no file extension", path.display()))
}

pub fn import_solution(path: &Path) -> anyhow::Result<Solution> {
    match extension_of(path)?.as_str() {
        "obj" | "stl" => Ok(Solution::new(Arc::new(import_mesh(path)?.0))),
        "dc" | "dsol" => Dc::import(path),
        "loops" => Loops::import(path),
        ext => anyhow::bail!("unsupported file extension `{ext}` for {}", path.display()),
    }
}

/// Like [`import_solution`], with the skeleton state and source a Dc file keeps.
pub fn import_save(path: &Path) -> anyhow::Result<Save> {
    match extension_of(path)?.as_str() {
        "dc" | "dsol" => formats::dc::read(path),
        _ => Ok(Save {
            solution: import_solution(path)?,
            skeleton: None,
            source: None,
        }),
    }
}

/// Read a surface mesh, for positive genus replaced by its TetGen surface when TetGen needed
/// boundary Steiner points (also returned: how many). A TetGen failure keeps the mesh as read;
/// the skeleton pipeline reports it again.
pub fn import_mesh(path: &Path) -> anyhow::Result<(Mesh<INPUT>, usize)> {
    let (mesh, ..) = Mesh::from_file(path)
        .map_err(|err| anyhow::anyhow!("reading mesh {}: {err}", path.display()))?;
    match dualcube::skeleton::validate::conform_input(&mesh) {
        Ok(Some((refined, added))) => {
            info!("{}: TetGen added {added} boundary vertices", path.display());
            Ok((refined, added))
        }
        Ok(None) => Ok((mesh, 0)),
        Err(error) => {
            warn!("{}: input kept as read ({error})", path.display());
            Ok((mesh, 0))
        }
    }
}

pub fn export_solution(sol: &Solution, path: &Path) -> anyhow::Result<()> {
    match extension_of(path)?.as_str() {
        "obj" => OBJ::export(sol, path),
        "dc" => Dc::export(sol, path),
        "loops" => Loops::export(sol, path),
        "nlr" => NLR::export(sol, path),
        "apg" => APG::export(sol, path),
        ext => anyhow::bail!("unsupported file extension `{ext}` for {}", path.display()),
    }
}
