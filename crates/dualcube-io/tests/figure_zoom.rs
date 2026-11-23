use dualcube::prelude::*;
use io::figure::{CameraZoom, Figure, FigureParams, FigureStyle, figure_svg, rasterize};
use std::path::Path;
use std::sync::Arc;

fn solution() -> Solution {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../mehsh/assets/blub001k.obj");
    Solution::new(Arc::new(io::import_mesh(&path).unwrap().0))
}

// The width and height of the figure.
fn size(svg: &str) -> [f64; 2] {
    let attribute = |name: &str| {
        let start = svg.find(&format!(r#"{name}=""#)).unwrap() + name.len() + 2;
        let end = start + svg[start..].find('"').unwrap();
        svg[start..end].parse::<f64>().unwrap()
    };
    [attribute("width"), attribute("height")]
}

// The coordinates of the paths.
fn coordinates(svg: &str) -> Vec<f64> {
    svg.split(r#" d=""#)
        .skip(1)
        .flat_map(|rest| {
            rest[..rest.find('"').unwrap()]
                .split(|c: char| c == 'M' || c == 'L' || c == 'Z' || c.is_whitespace())
                .filter(|t| !t.is_empty())
                .map(|t| t.parse::<f64>().unwrap())
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
fn a_zoomed_figure_is_framed_as_the_window() {
    let solution = solution();
    let style = FigureStyle::default();
    let fitted = FigureParams::default();
    let svg = figure_svg(&solution, None, Figure::InputMesh, &fitted, &style).unwrap();
    assert_eq!(size(&svg), [300., 300.]);
    assert!(!svg.contains("clipPath"));

    // Far away along the view, a window twice as wide as high.
    let mesh = &solution.mesh_ref;
    let center = mesh.center();
    let far = FigureParams {
        zoom: Some(CameraZoom {
            eye: center - fitted.view.normalize() * 10. * mesh.max_dim(),
            fov: 0.5,
            aspect: 2.,
        }),
        ..fitted.clone()
    };
    let svg = figure_svg(&solution, None, Figure::InputMesh, &far, &style).unwrap();
    assert_eq!(size(&svg), [300., 150.]);
    assert!(svg.contains("clipPath"));
    // The whole mesh, small in the middle.
    let xs = coordinates(&svg);
    assert!(xs.iter().all(|&x| (0. ..=300.).contains(&x)), "{xs:?}");

    // From inside the mesh: only what lies in front, and nothing far off the page.
    let inside = FigureParams {
        zoom: Some(CameraZoom {
            eye: center,
            fov: 0.8,
            aspect: 1.5,
        }),
        ..fitted
    };
    let svg = figure_svg(&solution, None, Figure::InputMesh, &inside, &style).unwrap();
    assert_eq!(size(&svg), [300., 200.]);
    let xs = coordinates(&svg);
    assert!(!xs.is_empty());
    assert!(xs.iter().all(|x| x.is_finite() && x.abs() < 1e5), "{xs:?}");
    let ([width, height], pixels) = rasterize(&svg, 2.).unwrap();
    assert_eq!([width, height], [600, 400]);
    assert_eq!(pixels.len(), 600 * 400 * 4);
}
