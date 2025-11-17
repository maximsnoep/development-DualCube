use dualcube::prelude::Polycube;

use crate::Export;

pub struct Dotgraph;

impl Export for Dotgraph {
    fn export(solution: &dualcube::prelude::Solution, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let path_obj = path.with_extension("graph");

        if let (Ok(dual), Ok(layout)) = (&solution.dual, &solution.layout) {
            Polycube::to_dotgraph(dual, layout, &path_obj)?;
            return Ok(());
        }
        Err(Box::new(std::io::Error::other("No dual or layout available")))
    }
}
