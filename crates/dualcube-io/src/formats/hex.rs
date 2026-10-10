use dualcube::prelude::*;
use std::io::Error;
use std::process::Command;

pub struct HEX;

/// Default location of the external hexmeshing pipeline; override with `DUALCUBE_HEXMESH_PIPELINE`.
const HEXMESH_PIPELINE: &str = "~/polycube-to-hexmesh/pipeline.sh";

/// Resolves the pipeline script path, expanding a leading `~` (which `Command` does not do).
fn pipeline_path() -> String {
    let raw =
        std::env::var("DUALCUBE_HEXMESH_PIPELINE").unwrap_or_else(|_| HEXMESH_PIPELINE.to_owned());
    match raw.strip_prefix("~/") {
        Some(rest) => match std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
            Some(home) => std::path::Path::new(&home).join(rest).display().to_string(),
            None => raw,
        },
        None => raw,
    }
}

impl crate::Export for HEX {
    fn export(solution: &Solution, path: &std::path::Path) -> anyhow::Result<()> {
        if solution.layout.is_none() {
            anyhow::bail!("No layout available");
        }

        let path_hex = path.with_extension("hex.mesh");
        let path_obj = path.with_extension("obj");
        let path_flag = path.with_extension("flag");

        info!(
            "HEX export paths: requested={} obj={} flag={} hex={}",
            path.display(),
            path_obj.display(),
            path_flag.display(),
            path_hex.display()
        );

        info!("Exporting OBJ file to {}", path_obj.display());
        crate::OBJ::export(solution, &path_obj)?;
        info!("Exporting FLAG file to {}", path_flag.display());
        crate::Flag::export(solution, &path_flag)?;

        let obj_arg = path_obj.display().to_string();
        let hex_arg = path_hex.display().to_string();
        let flag_arg = path_flag.display().to_string();

        let pipeline = pipeline_path();
        info!(
            "Running hexmesh pipeline with raw paths: script={} obj={} out={} flag={}",
            pipeline, obj_arg, hex_arg, flag_arg
        );
        let out = anyhow::Context::with_context(
            run(&pipeline, &[&obj_arg, "-out", &hex_arg, "-algo", &flag_arg]),
            || format!("running hexmesh pipeline {pipeline}"),
        )?;

        info!(
            "hex pipeline finished: status={} stdout_bytes={} stderr_bytes={}",
            out.status,
            out.stdout.len(),
            out.stderr.len()
        );
        info!(
            "hex pipeline stdout: {}",
            String::from_utf8_lossy(&out.stdout)
        );
        info!(
            "hex pipeline stderr: {}",
            String::from_utf8_lossy(&out.stderr)
        );

        if !out.status.success() {
            anyhow::bail!(
                "hexmesh pipeline failed ({}): {}",
                out.status,
                String::from_utf8_lossy(&out.stderr)
            );
        }
        if !path_hex.exists() {
            anyhow::bail!("hexmesh pipeline did not produce {}", path_hex.display());
        }

        Ok(())
    }
}

fn run(command: &str, args: &[&str]) -> Result<std::process::Output, Error> {
    if command.trim().is_empty() {
        return Err(Error::other("hexmesh pipeline command is empty"));
    }

    let mut process = Command::new(command);
    process.args(args);

    info!("Running command without path conversion: {:?}", process);
    process.output()
}
