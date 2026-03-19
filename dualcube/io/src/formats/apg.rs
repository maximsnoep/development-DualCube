use crate::Export;
use dualcube::polycube::POLYCUBE;
use dualcube::prelude::*;
use mehsh::prelude::*;
use std::io::Write;

// Abstract Polycube Graph
pub struct APG;

impl Export for APG {
    fn export(
        solution: &dualcube::prelude::Solution,
        path: &std::path::Path,
    ) -> anyhow::Result<()> {
        let path = path.with_extension("apg");

        let dual = solution
            .dual
            .as_ref()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let layout = solution
            .layout
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No layout available"))?;

        let mut file = std::fs::File::create(path)?;

        let mut polycube = Polycube::from_dual(&dual);
        polycube.resize(&dual, Some(layout));

        let mut vert_ids = ids::IdMap::<VERT, POLYCUBE>::new();
        for (i, vert_id) in polycube.structure.vert_ids().into_iter().enumerate() {
            vert_ids.insert(i, vert_id);
        }

        writeln!(
                file,
                "/ comments are lines starting with a slash (/), they should be ignored when parsing the file"
            )?;
        writeln!(file, "/ ")?;
        writeln!(
            file,
            "/ number of faces, number of edges, and number of vertices:"
        )?;
        writeln!(
            file,
            "{} {} {}",
            polycube.structure.face_ids().len(),
            polycube.structure.edge_ids().len() / 2,
            polycube.structure.vert_ids().len(),
        )?;

        writeln!(file, "/ faces in format of:")?;
        writeln!(file, "/ <VERT_ID> <VERT_ID> <VERT_ID> <VERT_ID>")?;
        for face_id in polycube.structure.face_ids() {
            let vertices = polycube.structure.vertices(face_id);
            writeln!(
                file,
                "{}",
                vertices
                    .map(|vert_id| format!("{}", vert_ids.id(&vert_id).unwrap()))
                    .collect_vec()
                    .into_iter()
                    .rev()
                    .join(" ")
            )?;
        }

        writeln!(file, "/ edges in format of:")?;
        writeln!(file, "/ <VERT_ID> <VERT_ID> <AXIS_LABEL> <TARGET_LENGTH>")?;
        let mut edge_strings = vec![];
        let mut edge_lengths = vec![];
        for edge_id in polycube.structure.edge_ids() {
            let direction_vector = polycube.structure.vector(edge_id).normalize();
            let (direction, orientation) = to_principal_direction(direction_vector);
            if orientation == Orientation::Backwards {
                continue;
            }
            let label = match direction {
                PrincipalDirection::X => "X",
                PrincipalDirection::Y => "Y",
                PrincipalDirection::Z => "Z",
            };
            let Some([v1, v2]) = polycube.structure.vertices(edge_id).collect_array::<2>() else {
                panic!()
            };
            edge_strings.push(format!(
                "{} {} {label}",
                vert_ids.id(&v1).unwrap(),
                vert_ids.id(&v2).unwrap()
            ));

            let path = layout.edge_to_path.get(&edge_id).unwrap();
            let length_of_path = path
                .windows(2)
                .map(|w| layout.granulated_mesh.distance(w[0], w[1]))
                .sum::<f64>();
            edge_lengths.push(length_of_path);
        }

        let min_edge_length = edge_lengths.iter().cloned().fold(f64::MAX, f64::min);
        let edge_lengths_int = edge_lengths
            .into_iter()
            .map(|length: f64| (length / min_edge_length).ceil() as u32)
            .collect::<Vec<_>>();

        for i in 0..edge_strings.len() {
            writeln!(file, "{} {}", edge_strings[i], edge_lengths_int[i])?;
        }

        Ok(())
    }
}
