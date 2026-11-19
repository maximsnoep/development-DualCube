use dualcube::prelude::*;
use dualcube::skeleton::pipeline::{Progress, Stage};
use dualcube::skeleton::skeletonize::{self, Options};
use io::{SkeletonExtraction, SkeletonSave};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("dc_save");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn blub() -> Mesh<INPUT> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../mehsh/assets/blub001k.obj");
    io::import_mesh(&path).unwrap().0
}

#[test]
fn v2_keeps_skeleton_stages_layout_and_source() {
    let mesh = blub();
    let extraction = skeletonize::extract(&mesh, Options::default()).unwrap();
    let mut progress = Progress::from_skeleton(extraction.skeleton.clone());
    progress.finish(&mesh).unwrap();

    let mut solution = Solution::new(Arc::new(mesh));
    solution.replace_loops(progress.loops.clone().unwrap());
    solution.construct_dual_and_polycube().unwrap();
    solution.place_corners().unwrap();
    solution.place_paths().unwrap();
    assert!(solution.layout.is_some());

    let save = SkeletonSave {
        extraction: Some(SkeletonExtraction::Extracted(extraction.clone())),
        labeling: progress.labeling.clone(),
        labeled: progress.labeled.clone(),
        realized: progress.realized.clone(),
        optimized: progress.optimized.clone(),
        drawing: progress.drawing.clone(),
        loops: progress.loops.clone(),
        failure: Some((Stage::LoopDrawing, "kept as text".to_owned())),
    };
    let path = scratch("blub.dc");
    let source = Path::new("/models/blub001k.obj");
    io::formats::dc::write(&solution, Some(&save), Some(source), &path).unwrap();
    let read = io::import_save(&path).unwrap();

    assert_eq!(read.source.as_deref(), Some(source));
    let layout = read.solution.layout.as_ref().expect("v2 keeps the layout");
    let original = solution.layout.as_ref().unwrap();
    assert_eq!(
        layout.granulated_mesh.nr_verts(),
        original.granulated_mesh.nr_verts()
    );
    assert_eq!(layout.face_to_patch.len(), original.face_to_patch.len());
    assert_eq!(read.solution.loops.len(), solution.loops.len());

    let skeleton = read.skeleton.expect("v2 keeps the skeleton");
    let Some(SkeletonExtraction::Extracted(read_extraction)) = &skeleton.extraction else {
        panic!("extraction lost");
    };
    assert_eq!(read_extraction.genus, extraction.genus);
    assert_eq!(
        read_extraction.contracted_mesh.nr_verts(),
        extraction.contracted_mesh.nr_verts()
    );
    // Keys survive, so the stages still refer to the same skeleton elements.
    assert_eq!(
        read_extraction.skeleton.nodes.keys().collect::<Vec<_>>(),
        extraction.skeleton.nodes.keys().collect::<Vec<_>>()
    );
    let (labeled, original) = (skeleton.labeled.unwrap(), progress.labeled.unwrap());
    assert_eq!(
        labeled
            .labels
            .iter()
            .map(|(e, l)| (e, l.direction, l.sign))
            .collect::<Vec<_>>(),
        original
            .labels
            .iter()
            .map(|(e, l)| (e, l.direction, l.sign))
            .collect::<Vec<_>>()
    );
    let (optimized, original) = (skeleton.optimized.unwrap(), progress.optimized.unwrap());
    assert_eq!(optimized.node_positions, original.node_positions);
    assert_eq!(optimized.cuboids.connections, original.cuboids.connections);
    assert!(skeleton.realized.is_some() && skeleton.labeling.is_some());
    assert_eq!(
        skeleton.drawing.unwrap().planned,
        progress.drawing.unwrap().planned
    );
    assert_eq!(skeleton.loops.unwrap().len(), progress.loops.unwrap().len());
    assert_eq!(
        skeleton.failure,
        Some((Stage::LoopDrawing, "kept as text".to_owned()))
    );
}

#[derive(Serialize)]
struct V1<'a> {
    format: &'a str,
    version: u32,
    solution: V1Solution,
}

#[derive(Serialize)]
struct V1Solution {
    mesh_ref: Arc<Mesh<INPUT>>,
    loops: slotmap::SlotMap<LoopID, Loop>,
    dual: Result<Dual, PropertyViolationError>,
    polycube: Option<Polycube>,
}

#[test]
fn v1_files_still_load_without_skeleton() {
    let solution = Solution::new(Arc::new(blub()));
    let file = V1 {
        format: "dualcube.dc",
        version: 1,
        solution: V1Solution {
            mesh_ref: solution.mesh_ref.clone(),
            loops: solution.loops.clone(),
            dual: solution.dual.clone(),
            polycube: None,
        },
    };
    let mut bytes = Vec::new();
    file.serialize(&mut rmp_serde::Serializer::new(&mut bytes).with_struct_map())
        .unwrap();
    let mut contents = b"DC01".to_vec();
    contents.extend(zstd::stream::encode_all(std::io::Cursor::new(&bytes), 3).unwrap());
    let path = scratch("v1.dc");
    std::fs::write(&path, contents).unwrap();

    let read = io::import_save(&path).unwrap();
    assert!(read.skeleton.is_none() && read.source.is_none());
    assert_eq!(
        read.solution.mesh_ref.nr_verts(),
        solution.mesh_ref.nr_verts()
    );
}
