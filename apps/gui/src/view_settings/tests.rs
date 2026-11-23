use super::*;

fn settings() -> ViewSettings {
    ViewSettings {
        eye: Vec3::new(25.125, -3.5, 1e-3),
        target: Vec3::new(0.1, 0.2, 0.3),
        up: Vec3::Z,
        loop_width: 0.0075,
        format: FigureFormat::Svg,
        label: true,
        smooth_paths: false,
        zoom: true,
        fov: 0.625,
        aspect: 1.25,
    }
}

fn other() -> ViewSettings {
    ViewSettings {
        eye: Vec3::ONE,
        target: Vec3::ZERO,
        up: Vec3::Y,
        loop_width: 0.006,
        format: FigureFormat::Pdf,
        label: false,
        smooth_paths: true,
        zoom: false,
        fov: 0.75,
        aspect: 1.5,
    }
}

#[test]
fn line_round_trips_exactly() {
    let settings = settings();
    assert_eq!(ViewSettings::parse(&settings.line(), other()), Ok(settings));
}

#[test]
fn report_round_trips_through_its_last_line() {
    let settings = settings();
    let configuration = Configuration::default();
    let report = settings.report(&Report {
        name: "model",
        source: Some(std::path::Path::new("/models/model.obj")),
        save: Some(std::path::Path::new("/out/model.dc")),
        figures: &[std::path::PathBuf::from("/out/model_mesh.svg")],
        vertices: 4,
        faces: 4,
        date: "2026-11-19",
        configuration: &configuration,
    });
    assert!(report.contains("save: model.dc"));
    assert_eq!(ViewSettings::parse(&report, other()), Ok(settings));
}

#[test]
fn omitted_keys_keep_their_value() {
    let parsed = ViewSettings::parse("eye=(1, 2, 3)", other());
    assert_eq!(
        parsed,
        Ok(ViewSettings {
            eye: Vec3::new(1., 2., 3.),
            ..other()
        })
    );
    let parsed = ViewSettings::parse("target=[1 , 2,3 ] label=1", other()).unwrap();
    assert_eq!(parsed.target, Vec3::new(1., 2., 3.));
    assert!(parsed.label);
    let parsed = ViewSettings::parse("view: format=PNG smooth=no", other()).unwrap();
    assert_eq!(parsed.format, FigureFormat::Png);
    assert!(!parsed.smooth_paths);
}

#[test]
fn malformed_text_is_rejected() {
    for text in [
        "",
        "eye=1,2",
        "eye=1,2,x",
        "eye=1,2,inf",
        "zoom=2",
        "eye",
        "format=jpg",
        "label=maybe",
        "fov=0",
        "fov=4",
        "aspect=-1",
        "eye=0,0,0 target=0,0,0",
        "up=0,0,0",
    ] {
        assert!(ViewSettings::parse(text, other()).is_err(), "{text:?}");
    }
}

#[test]
fn figure_camera_looks_from_eye_to_target() {
    let (view, up) = other().figure_camera();
    let expected = -Vector3D::new(1., 1., 1.).normalize();
    assert!((view - expected).norm() < 1e-6);
    assert_eq!(up, Vector3D::new(0., 1., 0.));
}

#[test]
fn the_window_is_only_written_when_zoomed() {
    let line = other().line();
    assert!(
        line.contains("zoom=false") && !line.contains("fov="),
        "{line}"
    );
    let line = settings().line();
    assert!(line.contains("zoom=true fov=0.625 aspect=1.25"), "{line}");
}
