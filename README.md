# DualCube: Robust Polycube Maps
> Maxim Snoep, via [algo](https://algo.win.tue.nl) at [(TU/e)](https://tue.nl)

**DualCube** is a research-driven implementation of robust algorithms for polycube map construction[^1][^2].

It provides:
- robust polycube segmentations on triangle meshes,
- an interactive GUI for visualization and manual interaction,
- and a lightweight CLI for batch processing.

[^1]: SNOEP M., SPECKMANN B., VERBEEK K.: Polycubes via dual loops. In _Proceedings of the 2025 SIAM International Meshing Roundtable (IMR)_ (2025). [doi:10.1137/1.9781611978575.7](https://doi.org/10.1137/1.9781611978575.7)
[^2]: SNOEP M., SPECKMANN B., VERBEEK K.: Robust construction of polycube segmentations via dual loops. In _Proceedings of the 2025 Symposium on Geometry Processing (SGP)_ (2025). [doi:10.1111/cgf.70195](https://doi.org/10.1111/cgf.70195)

## Getting Started

### Prerequisites

DualCube requires the Rust programming language and the Cargo package manager. The recommended installation method is [rustup](https://rustup.rs/). To download or update the repository, you will also want [git](https://git-scm.com/install/).

### Clone the project

```
git clone https://www.github.com/maximsnoep/DualCube
```

### Update to the latest version

```
git pull
```

## How to Run

This workspace contains separate `gui` and `cli` packages, so the recommended way to run them is with package-based Cargo commands (`-p ...`). By default, `cargo run` uses the **dev** profile. For optimized runs, use `--release`.

### Run the GUI

```
cargo run -p gui
```

### Run the CLI

```
cargo run -p cli
```

## Status

This is an active research codebase. It is suitable for experimentation, visualization, and batch testing, but interfaces, workflows, and file formats will continue to evolve.
