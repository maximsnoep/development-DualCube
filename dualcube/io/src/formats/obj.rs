pub struct OBJ;

impl crate::Export for OBJ {
    fn export(
        solution: &dualcube::prelude::Solution,
        path: &std::path::Path,
    ) -> anyhow::Result<()> {
        let path_obj = path.with_extension("obj");

        let Some(layout) = solution.layout.as_ref() else {
            anyhow::bail!("No layout available");
        };

        anyhow::Context::with_context(layout.granulated_mesh.to_obj(&path_obj), || {
            format!("writing OBJ to {}", path_obj.display())
        })?;

        Ok(())
    }
}
