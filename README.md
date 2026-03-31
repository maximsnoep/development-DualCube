# DualCube: Robust Polycube Maps
> Maxim Snoep, via [algo](https://algo.win.tue.nl) at [TU/e](https://tue.nl)

**DualCube** is a research-oriented implementation of robust algorithms for polycube map construction.[^1][^2]

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

To clone the development repository:
```
git clone https://www.github.com/maximsnoep/DualCube
cd DualCube
```

To clone the official stable repository:
```
git clone https://www.github.com/tue-alga/DualCube
cd DualCube
```

### Update to the latest version

```
git pull
```

## How to Run

This workspace contains separate `gui` and `cli` packages, so the recommended way to run them is with package-based Cargo commands using `-p ...`. By default, `cargo run` uses the `dev` profile. For optimized runs, use `--release`.

### Run the GUI

```
cargo run -p gui
```

Release build:
```
cargo run -p gui --release
```

### Run the CLI

```
cargo run -p cli
```

Release build:
```
cargo run -p cli --release
```

## Status

DualCube is an active research codebase intended for experimentation, visualization, and batch testing. Interfaces, workflows, and file formats may continue to evolve.
