use log::info;
use std::io::Error;
use std::path::PathBuf;
use std::process::Command;

pub struct HEX;

const HEXMESH_PIPELINE: &str = "~/polycube-to-hexmesh/pipeline.sh";
const DATA: &str = "\\\\wsl.localhost\\Debian\\home\\snoep\\polycube-to-hexmesh\\data";
const DAT2A: &str = "~/polycube-to-hexmesh/data/";

impl crate::Export for HEX {
    fn export(
        solution: &dualcube::prelude::Solution,
        path: &std::path::Path,
    ) -> anyhow::Result<()> {
        let Some(layout) = solution.layout.as_ref() else {
            anyhow::bail!("No layout available");
        };

        let path_hex = path.with_extension("hex.mesh");
        let path_obj = path.with_extension("obj");
        let path_flag = path.with_extension("flag");
        let wsl_path_hex = PathBuf::from(DATA).join(path_hex.file_name().unwrap());
        let wsl_path_obj = PathBuf::from(DATA).join(path_obj.file_name().unwrap());
        let wsl_path_flag = PathBuf::from(DATA).join(path_flag.file_name().unwrap());

        // Output current solution to obj and flag
        info!(
            "Exporting OBJ and FLAG files to {} and {}",
            path_obj.display(),
            path_flag.display()
        );
        crate::OBJ::export(&solution, &path_obj)?;
        info!("Exporting FLAG file to {}", path_flag.display());
        crate::Flag::export(&solution, &path_flag)?;

        // Move the files to the WSL data directory
        info!(
            "Copying OBJ and FLAG files to WSL data directory at {}",
            DATA
        );
        std::fs::copy(&path_obj, &wsl_path_obj)?;
        info!("Copying FLAG file to WSL data directory at {}", DATA);
        std::fs::copy(&path_flag, &wsl_path_flag)?;

        let wsl_path_obj2 = PathBuf::from(DAT2A).join(path_obj.file_name().unwrap());
        let wsl_path_hex2 = PathBuf::from(DAT2A).join(path_hex.file_name().unwrap());
        let wsl_path_flag2 = PathBuf::from(DAT2A).join(path_flag.file_name().unwrap());

        info!("Running hexmesh pipeline on WSL with OBJ and FLAG files");
        let out = run(
            HEXMESH_PIPELINE,
            &[
                wsl_path_obj2.to_str().unwrap(),
                "-out",
                wsl_path_hex2.to_str().unwrap(),
                "-algo",
                wsl_path_flag2.to_str().unwrap(),
            ],
        )?;

        println!("Pipeline output: {:?}", out);

        anyhow::Context::with_context(layout.granulated_mesh.to_obj(&path_obj), || {
            format!("writing HEX to {}", path_obj.display())
        })?;

        Ok(())
    }
}

fn run(script: &str, args: &[&str]) -> Result<std::process::Output, Error> {
    let mut wsl = Command::new("wsl");
    let wsl_args = args.into_iter().map(|arg| arg.replace("C:", "/mnt/c"));
    let command = wsl.args([script]).args(wsl_args);
    info!("Running command: {:?}", command);
    command.output().map_err(|e| Error::from(e))
}
