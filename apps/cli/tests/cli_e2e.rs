use cli::cli_main;
use std::fs;
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is <repo>/apps/cli
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .to_path_buf()
}

fn test_output_dir(name: &str) -> PathBuf {
    let dir = repo_root().join("target").join("cli-e2e-tests").join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// A small triangle mesh that is tracked in the repository.
const TEST_MODEL: &str = "blub001k.obj";

fn example_model(name: &str) -> PathBuf {
    // Prefer local (untracked) example models, fall back to the tracked mehsh test assets.
    let local = repo_root().join("out").join("examples").join(name);
    if local.exists() {
        return local;
    }
    repo_root()
        .join("crates")
        .join("mehsh")
        .join("assets")
        .join(name)
}

fn maybe_models() -> Vec<PathBuf> {
    ["cube.obj", "plane.obj", "bunny.obj", TEST_MODEL]
        .into_iter()
        .map(example_model)
        .filter(|p| p.exists())
        .collect()
}

fn run_cli(args: &[&str]) {
    let argv = std::iter::once("cli".to_owned())
        .chain(args.iter().map(|s| (*s).to_owned()))
        .collect::<Vec<_>>();
    cli_main(argv).unwrap();
}

#[test]
fn import_command_runs_on_example_model() {
    let input = example_model(TEST_MODEL);
    assert!(input.exists(), "missing test model: {}", input.display());

    run_cli(&["import", &input.display().to_string()]);
}

#[test]
fn initialize_then_export_dc_end_to_end() {
    let input = example_model(TEST_MODEL);
    assert!(input.exists(), "missing test model: {}", input.display());

    let out_dir = test_output_dir("initialize_then_export_dc_end_to_end");
    let output_base = out_dir.join("cube_initialized");

    run_cli(&[
        "initialize",
        "--input",
        &input.display().to_string(),
        "--output",
        &output_base.display().to_string(),
    ]);

    let output_file = output_base.with_extension("dc");
    assert!(
        output_file.exists(),
        "expected output file to exist: {}",
        output_file.display()
    );
}

#[test]
fn reconstruct_command_runs_from_initialized_solution() {
    let input = example_model(TEST_MODEL);
    assert!(input.exists(), "missing test model: {}", input.display());

    let out_dir = test_output_dir("reconstruct_command_runs_from_initialized_solution");
    let initialized_base = out_dir.join("cube_initialized");
    let reconstructed_base = out_dir.join("cube_reconstructed");

    run_cli(&[
        "initialize",
        "--input",
        &input.display().to_string(),
        "--output",
        &initialized_base.display().to_string(),
    ]);

    let initialized_file = initialized_base.with_extension("dc");
    assert!(
        initialized_file.exists(),
        "expected initialized file to exist: {}",
        initialized_file.display()
    );

    run_cli(&[
        "reconstruct",
        "--input",
        &initialized_file.display().to_string(),
        "--output",
        &reconstructed_base.display().to_string(),
        "--unit",
    ]);

    let reconstructed_file = reconstructed_base.with_extension("dc");
    assert!(
        reconstructed_file.exists(),
        "expected reconstructed file to exist: {}",
        reconstructed_file.display()
    );
}

#[test]
fn initialize_smoke_runs_across_available_models() {
    let inputs = maybe_models();
    assert!(
        !inputs.is_empty(),
        "expected at least one example model to exist"
    );

    let out_dir = test_output_dir("initialize_smoke_runs_across_available_models");

    for input in inputs {
        let output_base = out_dir.join(input.file_stem().unwrap());
        run_cli(&[
            "initialize",
            "--input",
            &input.display().to_string(),
            "--output",
            &output_base.display().to_string(),
        ]);

        let output_file = output_base.with_extension("dc");
        assert!(
            output_file.exists(),
            "expected output file to exist: {}",
            output_file.display()
        );
    }
}

#[test]
fn stages_reports_a_full_run_of_the_test_model() {
    let input = example_model(TEST_MODEL);
    let dir = test_output_dir("stages");
    let report = dir.join("stages.csv");
    run_cli(&[
        "stages",
        "--input",
        &input.display().to_string(),
        "--report",
        &report.display().to_string(),
    ]);
    let text = fs::read_to_string(&report).unwrap();
    let lines: Vec<_> = text.lines().collect();
    assert_eq!(lines.len(), 2, "{text}");
    let header: Vec<_> = lines[0].split(',').collect();
    let row: Vec<_> = lines[1].split(',').collect();
    let column = |name: &str| row[header.iter().position(|&h| h == name).unwrap()];
    assert_eq!(column("status"), "ok", "{text}");
    assert_eq!(column("genus"), "0");
    assert_eq!(column("loops_drawn"), column("loops_planned"));
    assert!(!column("s_reconstruction").is_empty());
}
