use super::*;

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/cli-stages-tests")
        .join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn stages_parse_by_name_in_pipeline_order() {
    for (i, stage) in Stage::ALL.into_iter().enumerate() {
        assert_eq!(Stage::parse(stage.name()).unwrap(), stage);
        assert_eq!(stage as usize, i);
    }
    assert!(Stage::Extraction < Stage::Reconstruction);
    assert!(Stage::parse("cubes").is_err());
}

#[test]
fn folders_list_meshes_and_sets_resolve_relative_paths() {
    let dir = scratch("inputs");
    for name in ["b.obj", "a.STL", "notes.txt", "c.dc"] {
        std::fs::write(dir.join(name), "").unwrap();
    }
    assert_eq!(
        meshes(&dir).unwrap(),
        [dir.join("a.STL"), dir.join("b.obj")]
    );
    let set = dir.join("set.txt");
    std::fs::write(&set, "# comment\n\nb.obj  # trailing\n../x/y.obj\n").unwrap();
    assert_eq!(
        meshes(&set).unwrap(),
        [dir.join("b.obj"), dir.join("../x/y.obj")]
    );
    assert_eq!(meshes(&dir.join("b.obj")).unwrap(), [dir.join("b.obj")]);
}

#[test]
fn csv_fields_quote_separators_quotes_and_line_breaks() {
    assert_eq!(csv_field("plain"), "plain");
    assert_eq!(csv_field("a, b"), "\"a, b\"");
    assert_eq!(csv_field("say \"x\""), "\"say \"\"x\"\"\"");
    assert_eq!(csv_field("two\nlines"), "\"two\nlines\"");
}

#[test]
fn unreadable_mesh_fails_at_extraction_with_full_row() {
    let row = run_mesh(Path::new("does/not/exist.obj"), Stage::Reconstruction);
    assert_eq!(row.status(), "failed at extraction");
    assert!(row.seconds.iter().all(Option::is_none));
    assert_eq!(row.fields().len(), Row::default().fields().len());
}

#[test]
fn report_has_a_header_and_one_line_per_row() {
    let dir = scratch("report");
    let rows = [
        Row {
            input: "a.obj".into(),
            failure: Some((Stage::Drawing, "x, \"y\"".to_owned())),
            ..Row::default()
        },
        Row::default(),
    ];
    let path = dir.join("report.csv");
    write_report(&path, &rows).unwrap();
    let text = std::fs::read_to_string(path).unwrap();
    let lines: Vec<_> = text.lines().collect();
    assert_eq!(lines.len(), 3);
    assert!(lines[0].starts_with("input,genus,faces,status,error,"));
    assert!(lines[1].starts_with("a.obj,,0,failed at drawing,\"x, \"\"y\"\"\","));
    assert!(lines[2].contains(",ok,"));
}
