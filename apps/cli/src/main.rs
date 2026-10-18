use std::env;

// A faster allocator than that of the system, also with many threads allocating at once.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    // Log events (of all crates, through `tracing`) from INFO on, to stderr: level, target, and message.
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_writer(std::io::stderr)
        .without_time()
        .init();

    if let Err(err) = cli::cli_main(env::args().collect()) {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}
