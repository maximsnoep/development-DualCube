pub mod commands;
pub mod parser;
pub mod stages;

pub fn cli_main(argv: Vec<String>) -> anyhow::Result<()> {
    commands::run(parser::Args::parse(argv))
}
