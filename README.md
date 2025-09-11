# DualCube: Robust Polycube Maps
> Maxim Snoep, via [algo](https://algo.win.tue.nl) at [(TU/e)](https://tue.nl)

**DualCube** is a research-driven implementation of robust algorithms for polycube map construction[^1][^2].

[^1]: SNOEP M., SPECKMANN B., VERBEEK K.: Polycubes via dual loops. In _Proceedings of the 2025 SIAM International Meshing Roundtable (IMR)_ (2025). [doi:10.1137/1.9781611978575.7](https://doi.org/10.1137/1.9781611978575.7)
[^2]: SNOEP M., SPECKMANN B., VERBEEK K.: Robust construction of polycube segmentations via dual loops. In _Proceedings of the 2025 Symposium on Geometry Processing (SGP)_ (2025). [doi:10.1111/cgf.70195](https://doi.org/10.1111/cgf.70195)

## 🚀 Getting Started

### Prerequisites

Install [Rust and Cargo](https://rustup.rs/) using the official installer.

### Download

```bash
git clone https://www.github.com/maximsnoep/DualCube
```

### Update

```bash
git pull
```

### Build and run

```bash
cargo run -p gui
```

## 🛠️ To-Do

Planned features and improvements:

- [ ] Compound state handling  (dual -> seg -> quad -> hex)
- [ ] Manual loop editing tools  
- [ ] Realizability under parameter constraints  
- [ ] Automatic loop initialization for high-genus surfaces  
- [ ] Loop computation using direction fields  
- [ ] Camera position export/import
- [ ] PNG export for renders (to replace screenshots)  
- [ ] CLI frontend (using `ratatui`)



