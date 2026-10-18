use dualcube::prelude::QualityParams;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Args {
    pub command: Command,
}

#[derive(Debug, Clone)]
pub enum Command {
    Help,
    Import {
        input: PathBuf,
    },
    Initialize {
        input: PathBuf,
        output: Option<PathBuf>,
        quality: QualityParams,
        reconstruct: bool,
        unit: bool,
    },
    Anneal {
        input: PathBuf,
        output: Option<PathBuf>,
        steps: Option<usize>,
        offspring: Option<usize>,
        quality: QualityParams,
        reconstruct: bool,
        unit: bool,
    },
    Reconstruct {
        input: PathBuf,
        output: Option<PathBuf>,
        unit: bool,
    },
    Score {
        input: PathBuf,
        quality: QualityParams,
        csv: Option<PathBuf>,
    },
    Export {
        input: PathBuf,
        output: Option<PathBuf>,
        format: String,
    },
    Benchmark {
        /// Models (obj/stl files), and folders of models.
        inputs: Vec<PathBuf>,
        output: PathBuf,
        name: String,
        runs: usize,
        seconds: f64,
        quality: QualityParams,
        /// A session folder to continue (the other options are those of the session).
        resume: Option<PathBuf>,
    },
}

impl Args {
    pub fn parse(argv: Vec<String>) -> anyhow::Result<Self> {
        if argv.len() <= 1 {
            return Ok(Self {
                command: Command::Help,
            });
        }

        let mut it = argv.into_iter();
        let _bin = it.next();
        let cmd = it.next().unwrap_or_else(|| "help".to_owned());

        let command = match cmd.as_str() {
            "help" | "--help" | "-h" => Command::Help,
            "import" => {
                let input = take_positional_path(&mut it, "input")?;
                Command::Import { input }
            }
            "initialize" | "init" => {
                let mut input = None;
                let mut output = None;
                let mut quality = QualityParams::default();
                let mut reconstruct = false;
                let mut unit = true;

                parse_kv_flags(it.collect(), |flag, value| {
                    if parse_quality_flag(flag, value, &mut quality)? {
                        return Ok(());
                    }
                    match flag {
                        "--input" | "-i" => {
                            input = Some(PathBuf::from(require_value(flag, value)?));
                        }
                        "--output" | "-o" => {
                            output = Some(PathBuf::from(require_value(flag, value)?));
                        }
                        "--reconstruct" => reconstruct = true,
                        "--no-unit" => unit = false,
                        "--unit" => unit = true,
                        other => anyhow::bail!("unknown flag for initialize: {other}"),
                    }
                    Ok(())
                })?;

                let input = input.ok_or_else(|| anyhow::anyhow!("missing required --input"))?;
                Command::Initialize {
                    input,
                    output,
                    quality,
                    reconstruct,
                    unit,
                }
            }
            "benchmark" => {
                let mut inputs = vec![];
                let mut output = PathBuf::from("benchmarks");
                let mut name = "benchmark".to_owned();
                let mut runs = 1;
                let mut seconds = 60.;
                let mut quality = QualityParams::default();
                let mut resume = None;

                parse_kv_flags(it.collect(), |flag, value| {
                    if parse_quality_flag(flag, value, &mut quality)? {
                        return Ok(());
                    }
                    match flag {
                        "--resume" => resume = Some(PathBuf::from(require_value(flag, value)?)),
                        "--input" | "-i" => inputs.push(PathBuf::from(require_value(flag, value)?)),
                        "--output" | "-o" => output = PathBuf::from(require_value(flag, value)?),
                        "--name" => name = require_value(flag, value)?.to_owned(),
                        "--runs" => runs = require_value(flag, value)?.parse::<usize>()?,
                        "--seconds" => seconds = require_value(flag, value)?.parse::<f64>()?,
                        other => anyhow::bail!("unknown flag for benchmark: {other}"),
                    }
                    Ok(())
                })?;

                anyhow::ensure!(
                    !inputs.is_empty() || resume.is_some(),
                    "missing required --input (a model or a folder), or --resume (a session folder)"
                );
                Command::Benchmark {
                    inputs,
                    output,
                    name,
                    runs,
                    seconds,
                    quality,
                    resume,
                }
            }
            "anneal" => {
                let mut input = None;
                let mut output = None;
                let mut steps = None;
                let mut offspring = None;
                let mut quality = QualityParams::default();
                let mut reconstruct = false;
                let mut unit = true;

                parse_kv_flags(it.collect(), |flag, value| {
                    if parse_quality_flag(flag, value, &mut quality)? {
                        return Ok(());
                    }
                    match flag {
                        "--input" | "-i" => {
                            input = Some(PathBuf::from(require_value(flag, value)?));
                        }
                        "--output" | "-o" => {
                            output = Some(PathBuf::from(require_value(flag, value)?));
                        }
                        "--steps" => {
                            steps = Some(require_value(flag, value)?.parse::<usize>()?);
                        }
                        "--offspring" => {
                            offspring = Some(require_value(flag, value)?.parse::<usize>()?);
                        }
                        "--reconstruct" => reconstruct = true,
                        "--no-unit" => unit = false,
                        "--unit" => unit = true,
                        other => anyhow::bail!("unknown flag for anneal: {other}"),
                    }
                    Ok(())
                })?;

                let input = input.ok_or_else(|| anyhow::anyhow!("missing required --input"))?;
                Command::Anneal {
                    input,
                    output,
                    steps,
                    offspring,
                    quality,
                    reconstruct,
                    unit,
                }
            }
            "reconstruct" => {
                let mut input = None;
                let mut output = None;
                let mut unit = true;

                parse_kv_flags(it.collect(), |flag, value| {
                    match flag {
                        "--input" | "-i" => {
                            input = Some(PathBuf::from(require_value(flag, value)?));
                        }
                        "--output" | "-o" => {
                            output = Some(PathBuf::from(require_value(flag, value)?));
                        }
                        "--no-unit" => unit = false,
                        "--unit" => unit = true,
                        other => anyhow::bail!("unknown flag for reconstruct: {other}"),
                    }
                    Ok(())
                })?;

                let input = input.ok_or_else(|| anyhow::anyhow!("missing required --input"))?;
                Command::Reconstruct {
                    input,
                    output,
                    unit,
                }
            }
            "score" => {
                let mut input = None;
                let mut quality = QualityParams::default();
                let mut csv = None;

                parse_kv_flags(it.collect(), |flag, value| {
                    if parse_quality_flag(flag, value, &mut quality)? {
                        return Ok(());
                    }
                    match flag {
                        "--input" | "-i" => {
                            input = Some(PathBuf::from(require_value(flag, value)?));
                        }
                        "--csv" => csv = Some(PathBuf::from(require_value(flag, value)?)),
                        other => anyhow::bail!("unknown flag for score: {other}"),
                    }
                    Ok(())
                })?;

                let input = input.ok_or_else(|| anyhow::anyhow!("missing required --input"))?;
                Command::Score {
                    input,
                    quality,
                    csv,
                }
            }
            "export" => {
                let mut input = None;
                let mut output = None;
                let mut format = None;

                parse_kv_flags(it.collect(), |flag, value| {
                    match flag {
                        "--input" | "-i" => {
                            input = Some(PathBuf::from(require_value(flag, value)?));
                        }
                        "--output" | "-o" => {
                            output = Some(PathBuf::from(require_value(flag, value)?));
                        }
                        "--format" | "-f" => {
                            format = Some(require_value(flag, value)?.to_owned());
                        }
                        other => anyhow::bail!("unknown flag for export: {other}"),
                    }
                    Ok(())
                })?;

                let input = input.ok_or_else(|| anyhow::anyhow!("missing required --input"))?;
                let format = format.ok_or_else(|| anyhow::anyhow!("missing required --format"))?;
                Command::Export {
                    input,
                    output,
                    format,
                }
            }
            other => anyhow::bail!("unknown command: {other}"),
        };

        Ok(Self { command })
    }
}

/// Parse the flags of the quality criterion. Returns whether the flag was one of them.
///
/// - `--beta <f64>` (complexity: the penalty per loop)
fn parse_quality_flag(
    flag: &str,
    value: Option<&str>,
    quality: &mut QualityParams,
) -> anyhow::Result<bool> {
    match flag {
        "--beta" => quality.weights.complexity = require_value(flag, value)?.parse::<f64>()?,
        _ => return Ok(false),
    }
    Ok(true)
}

fn require_value<'a>(flag: &str, value: Option<&'a str>) -> anyhow::Result<&'a str> {
    value.ok_or_else(|| anyhow::anyhow!("missing value for {flag}"))
}

fn take_positional_path<I>(it: &mut I, name: &str) -> anyhow::Result<PathBuf>
where
    I: Iterator<Item = String>,
{
    it.next()
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("missing required positional argument `{name}`"))
}

fn parse_kv_flags<F>(args: Vec<String>, mut f: F) -> anyhow::Result<()>
where
    F: FnMut(&str, Option<&str>) -> anyhow::Result<()>,
{
    let mut i = 0usize;
    while i < args.len() {
        let flag = &args[i];
        if !flag.starts_with('-') {
            anyhow::bail!("unexpected positional argument: {}", args[i]);
        }

        let takes_value = matches!(
            flag.as_str(),
            "--input"
                | "-i"
                | "--output"
                | "-o"
                | "--samples"
                | "-s"
                | "--method"
                | "--iterations"
                | "--pool1"
                | "--pool2"
                | "--patience"
                | "--format"
                | "-f"
                | "--beta"
                | "--csv"
                | "--steps"
                | "--offspring"
                | "--name"
                | "--runs"
                | "--seconds"
                | "--resume"
        );

        let value = if takes_value {
            i += 1;
            Some(
                args.get(i)
                    .ok_or_else(|| anyhow::anyhow!("missing value for {}", flag))?
                    .as_str(),
            )
        } else {
            None
        };

        f(flag, value)?;
        i += 1;
    }
    Ok(())
}
