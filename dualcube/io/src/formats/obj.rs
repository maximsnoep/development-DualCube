use log::info;

use crate::Export;

pub struct Obj;

impl Export for Obj {
    fn export(
        solution: &dualcube::prelude::Solution,
        path: &std::path::Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let path_obj = path.with_extension("obj");

        if let Some(layout) = &solution.layout {
            info!("Writing OBJ file to {path_obj:?}");
            layout.granulated_mesh.to_obj(&path_obj)?;
            return Ok(());
        }
        Err(Box::new(std::io::Error::other("No layout available")))
    }
}
