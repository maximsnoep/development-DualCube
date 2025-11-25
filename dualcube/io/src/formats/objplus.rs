use crate::{Export, Import};
use itertools::Itertools;
use log::info;
use std::{fs::OpenOptions, io::Write};

pub struct ObjPlus;

impl Export for ObjPlus {
    fn export(solution: &dualcube::prelude::Solution, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let path_obj = path.with_extension("lines.obj");

        if let Some(layout) = &solution.layout {
            info!("Writing OBJ+ file to {path_obj:?}");
            let vert_ids = layout.granulated_mesh.to_obj(&path_obj)?;
            // open the file and write the lines to it
            let mut file = OpenOptions::new().append(true).open(path_obj.clone()).unwrap();

            for path in layout.edge_to_path.values() {
                let line = path.iter().map(|vert_id| format!("{}", vert_ids.id(vert_id).unwrap())).join(" ");
                writeln!(file, "l {}", line)?;
            }
            return Ok(());
        }
        Err(Box::new(std::io::Error::other("No layout available")))
    }
}
