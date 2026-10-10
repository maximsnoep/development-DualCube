//! Render object for the polycube.

use super::super::gizmos::{add_band, add_quad, axis_index, band_colors, uniform_color_map};
use super::super::store::{RenderAsset, RenderObject};
use crate::{colors, resources::Configuration};
use bevy::prelude::*;
use dualcube::prelude::*;
use std::collections::HashMap;

/// The POLYCUBE render object, it has:
/// - meshes with black / gray / colored faces
/// - the dual loops per principal direction
/// - the polycube edges (flat and non-flat)
pub(in crate::render) fn build(
    solution: &Solution,
    configuration: &Configuration,
) -> Option<RenderObject> {
    let polycube = solution.polycube.as_ref()?;

    let structure = &polycube.structure;
    let (scale, translation) = structure.scale_translation();

    let mut colored_color_map = HashMap::new();
    // The loops as bands (as on the model): their width relative to the diagonal of the polycube, lifted slightly off
    // its faces.
    let corners = structure
        .vert_ids()
        .into_iter()
        .map(|v| structure.position(v))
        .collect::<Vec<_>>();
    let (min, max) = corners.iter().fold(
        (
            Vector3D::repeat(f64::INFINITY),
            Vector3D::repeat(f64::NEG_INFINITY),
        ),
        |(min, max), p| (min.inf(p), max.sup(p)),
    );
    let half = 0.5 * configuration.loop_width * (max - min).norm().max(1e-12);
    let mut loops = DIRECTIONS.map(|_| mehsh_bevy::MeshBuilder::new());

    for face_id in structure.face_ids() {
        colored_color_map.insert(
            face_id,
            colors::from_direction(
                to_principal_direction(structure.normal(face_id)).0,
                Some(Perspective::Primal),
                None,
            ),
        );

        // Each quad face carries two loop segments, connecting the midpoints of
        // opposite edges, colored by the dual direction they run along.
        let Some(edges) = structure.edges(face_id).collect_array::<4>() else {
            panic!("Expected four edges for face {face_id:?}");
        };
        let positions = edges.map(|edge_id| structure.position(edge_id));
        let normal = structure.normal(face_id).normalize();

        for (from, to, across) in [
            (0, 2, positions[1] - positions[3]),
            (1, 3, positions[0] - positions[2]),
        ] {
            // The loop separates the polycube along the axis across the segment: its negative side is toward the
            // negative end of that axis.
            let dir = to_principal_direction(across).0;
            let axis = axis_index(dir);
            let mut negative = Vector3D::zeros();
            negative[axis] = -1.;
            let frame = |position| LoopFrame {
                position,
                normal,
                negative,
            };
            let lift = (0.3 + 0.03 * axis as f64) * half;
            add_band(
                &mut loops[axis],
                [(frame(positions[from]), frame(positions[to]))],
                half,
                lift,
                band_colors(dir),
            );
        }
    }

    let [x_loops, y_loops, z_loops] = loops.map(|mut builder| {
        builder.normalize(scale, translation);
        builder.build()
    });

    // The polycube's edges as strips on its faces (no lines; as the paths on the model): every edge a strip into each
    // of its two faces, lifted slightly above them (and the loops). "paths": the edges between faces with different
    // normals; "flat paths": all edges (thinner).
    let mut edges = [(); 2].map(|()| mehsh_bevy::MeshBuilder::new());
    let lift = 0.45 * half;
    for pedge_id in structure.edge_ids() {
        let twin = structure.twin(pedge_id);
        if twin < pedge_id {
            continue;
        }
        let Some([u, v]) = structure.vertices(pedge_id).collect_array::<2>() else {
            continue;
        };
        let (a, b) = (structure.position(u), structure.position(v));
        let Some(along) = (b - a).try_normalize(1e-12) else {
            continue;
        };
        let faces = [structure.face(pedge_id), structure.face(twin)];
        let flat = structure.normal(faces[0]) == structure.normal(faces[1]);
        for (builder, factor, shown) in [(0, 0.6, !flat), (1, 0.35, true)] {
            if !shown {
                continue;
            }
            let width = factor * half;
            for face in faces {
                let normal = structure.normal(face).normalize();
                let corners = structure
                    .vertices(face)
                    .map(|w| structure.position(w))
                    .collect::<Vec<_>>();
                let centroid = corners.iter().sum::<Vector3D>() / corners.len().max(1) as f64;
                // Into the face.
                let mut inward = normal.cross(&along);
                if inward.dot(&(centroid - a)) < 0. {
                    inward = -inward;
                }
                let (p, q) = (a + normal * lift, b + normal * lift);
                add_quad(
                    &mut edges[builder],
                    [p, q, q + inward * width, p + inward * width],
                    normal,
                    colors::GRAY,
                );
            }
        }
    }
    let [paths, flat_paths] = edges.map(|mut builder| {
        builder.normalize(scale, translation);
        builder.build()
    });

    Some(
        RenderObject::default()
            .mesh(
                structure,
                &uniform_color_map(structure, colors::BLACK),
                "black",
            )
            .mesh(
                structure,
                &uniform_color_map(structure, colors::LIGHT_GRAY),
                "gray",
            )
            .mesh(structure, &colored_color_map, "colored")
            .add("x-loops", RenderAsset::Mesh(x_loops))
            .add("y-loops", RenderAsset::Mesh(y_loops))
            .add("z-loops", RenderAsset::Mesh(z_loops))
            .add("paths", RenderAsset::Mesh(paths))
            .add("flat paths", RenderAsset::Mesh(flat_paths))
            .to_owned(),
    )
}
