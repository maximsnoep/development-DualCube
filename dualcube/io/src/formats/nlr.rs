use bimap::BiHashMap;
use dualcube::polycube::POLYCUBE;
use dualcube::prelude::*;
use itertools::Itertools;
use log::info;
use mehsh::prelude::*;
use ordered_float::OrderedFloat;
use std::io::Write;

use crate::Export;

pub struct Nlr;

impl Export for Nlr {
    fn export(
        solution: &Solution,
        path: &std::path::Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let path_topol = path.with_extension("topol");
        let path_geom = path.with_extension("geom");
        let path_cdim = path.with_extension("cdim");
        let path_bcdat = path.with_extension("bcdat");
        let path_xloops = path.with_extension("xloops.seg");
        let path_yloops = path.with_extension("yloops.seg");
        let path_zloops = path.with_extension("zloops.seg");
        let path_xpatches = path.with_extension("xpatches.seg");
        let path_ypatches = path.with_extension("ypatches.seg");
        let path_zpatches = path.with_extension("zpatches.seg");

        if let (Ok(dual), Some(layout), Some(polycube), Some(quad)) = (
            &solution.dual,
            &solution.layout,
            &solution.polycube,
            &solution.quad,
        ) {
            let signature = " -- automatically generated via DualCube (Maxim Snoep)";

            // Minimum dimension of smallest edge in polycube (cartesian representation)
            let min_dim = 5;
            // Scale S of farfield box. Given that polycube is W x H x D, and let the largest dimension by MAX (of W, H, D), then the farfield box will have dimensions 2*S*MAX x 2*S*MAX x 2*S*MAX.
            let s = 10.;
            let ff_mult = min_dim as f64 * s;

            // Center coordinate of the input (real)
            let real_center = dual.mesh_ref.center();
            // Scale of the input (real)
            let real_scale = dual.mesh_ref.max_dim();

            // Center coordinate of the polycube (cartesian)
            let cartesian_center = polycube.structure.center();
            // Scale of the polycube (cartesian)
            let cartesian_scale = polycube.structure.max_dim();

            // Find a reference vertex. We use the vertex with the smallest x-coordinate
            let refv = polycube
                .structure
                .vert_ids()
                .into_iter()
                .min_by_key(|&vert| OrderedFloat(polycube.structure.position(vert).x))
                .unwrap();

            let v1 = real_center
                - Vector3D::new(
                    ff_mult * real_scale,
                    ff_mult * real_scale,
                    ff_mult * real_scale,
                );
            let v2 = v1 + 2. * Vector3D::new(0., 0., ff_mult * real_scale);
            let v3 = v2 + 2. * Vector3D::new(0., ff_mult * real_scale, 0.);
            let v4 = v3 - 2. * Vector3D::new(0., 0., ff_mult * real_scale);

            let v5 = v1 + 2. * Vector3D::new(ff_mult * real_scale, 0., 0.);
            let v6 = v5 + 2. * Vector3D::new(0., 0., ff_mult * real_scale);
            let v7 = v6 + 2. * Vector3D::new(0., ff_mult * real_scale, 0.);
            let v8 = v7 - 2. * Vector3D::new(0., 0., ff_mult * real_scale);

            assert!(polycube.structure.verts.len() < 10000);
            let vert_to_id: BiHashMap<VertKey<POLYCUBE>, usize> = polycube
                .structure
                .vert_ids()
                .iter()
                .enumerate()
                .map(|(i, &id)| (id, 10001 + i))
                .collect();

            assert!(polycube.structure.edges.len() < 10000);
            let edge_to_id: BiHashMap<EdgeKey<POLYCUBE>, usize> = polycube
                .structure
                .edge_ids()
                .iter()
                .filter(|&&edge_id| edge_id < polycube.structure.twin(edge_id))
                .enumerate()
                .map(|(i, &id)| (id, 20001 + i))
                .collect();

            assert!(polycube.structure.faces.len() < 10000);
            let face_to_id: BiHashMap<FaceKey<POLYCUBE>, usize> = polycube
                .structure
                .face_ids()
                .iter()
                .enumerate()
                .map(|(i, &id)| (id, 30001 + i))
                .collect();

            // ------------------------
            // --- WRITE TOPOL FILE ---
            //
            //
            //
            info!("Writing TOPOL file to {path_topol:?}");
            let mut file_topol = std::fs::File::create(path_topol)?;

            // Write info (should apparently be 5 lines)
            write!(
                file_topol,
                "'topol file <> {signature}'\n'2nd line'\n'3rd line'\n'4th line'\n'5th line'"
            )?;

            // Write blocks and compound blocks (we do not define any)
            write!(
                file_topol,
                "\n NUMBER OF BLOCKS:\n       0\n NUMBER OF COMPOUND BLOCKS:\n       0"
            )?;

            // Write faces
            write!(
                file_topol,
                "\n NUMBER OF ELEMENTARY FACES:\n       {}\n        FACE       EDGE1       EDGE2       EDGE3       EDGE4       IDENT\n",
                polycube.structure.face_ids().len() + 6
            )?;
            write!(
                file_topol,
                "{}",
                polycube
                    .structure
                    .face_ids()
                    .iter()
                    .map(|face_id| {
                        let face_int = face_to_id.get_by_left(face_id).unwrap();
                        let Some([e0, e1, e2, e3]) = polycube.structure.edges(*face_id).collect_array::<4>() else {
                            panic!("Expecting face {face_id:?} to have exactly four edges");
                        };
                        let edge_int1 = edge_to_id
                            .get_by_left(&e0)
                            .or_else(|| edge_to_id.get_by_left(&polycube.structure.twin(e0)))
                            .unwrap();
                        let edge_int2 = edge_to_id
                            .get_by_left(&e1)
                            .or_else(|| edge_to_id.get_by_left(&polycube.structure.twin(e1)))
                            .unwrap();
                        let edge_int3 = edge_to_id
                            .get_by_left(&e2)
                            .or_else(|| edge_to_id.get_by_left(&polycube.structure.twin(e2)))
                            .unwrap();
                        let edge_int4 = edge_to_id
                            .get_by_left(&e3)
                            .or_else(|| edge_to_id.get_by_left(&polycube.structure.twin(e3)))
                            .unwrap();
                        format!("       {face_int}       {edge_int1}       {edge_int2}       {edge_int3}       {edge_int4}       'FACE'")
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            )?;
            // Add the bounding box faces

            write!(
                file_topol,
                "\n       91234       80102       80304       80203       80401       'FACE'"
            )?;
            write!(
                file_topol,
                "\n       97658       80506       80708       80805       80607       'FACE'"
            )?;
            write!(
                file_topol,
                "\n       91485       80105       80408       80401       80805       'FACE'"
            )?;
            write!(
                file_topol,
                "\n       97326       80206       80307       80607       80203       'FACE'"
            )?;
            write!(
                file_topol,
                "\n       91562       80102       80506       80105       80206       'FACE'"
            )?;
            write!(
                file_topol,
                "\n       97843       80304       80708       80307       80408       'FACE'"
            )?;

            write!(file_topol, "\n NUMBER OF COMPOUND FACES:\n       0\n")?;

            // Write edges
            write!(
                file_topol,
                " NUMBER OF ELEMENTARY EDGES:\n       {}\n        EDGE       VERT1       VERT2       IDENT\n",
                (polycube.structure.edge_ids().len() / 2) + 12
            )?;
            write!(
                file_topol,
                "{}",
                polycube
                    .structure
                    .edge_ids()
                    .iter()
                    .filter_map(|edge_id| {
                        edge_to_id.get_by_left(edge_id).map(|edge_int| {
                            let Some([v0, v1]) = polycube.structure.vertices(*edge_id).collect_array::<2>() else {
                                panic!("Expecting edge {edge_id:?} to have exactly two vertices");
                            };
                            let vert_int1 = vert_to_id.get_by_left(&v0).unwrap();
                            let vert_int2 = vert_to_id.get_by_left(&v1).unwrap();
                            format!("       {edge_int}       {vert_int1}       {vert_int2}       'EDGE'")
                        })
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            )?;
            // Add the bounding box edges
            write!(
                file_topol,
                "\n       80102       70001       70002       'EDGE'"
            )?;
            write!(
                file_topol,
                "\n       80203       70002       70003       'EDGE'"
            )?;
            write!(
                file_topol,
                "\n       80304       70003       70004       'EDGE'"
            )?;
            write!(
                file_topol,
                "\n       80401       70004       70001       'EDGE'"
            )?;

            write!(
                file_topol,
                "\n       80506       70005       70006       'EDGE'"
            )?;
            write!(
                file_topol,
                "\n       80607       70006       70007       'EDGE'"
            )?;
            write!(
                file_topol,
                "\n       80708       70007       70008       'EDGE'"
            )?;
            write!(
                file_topol,
                "\n       80805       70008       70005       'EDGE'"
            )?;

            write!(
                file_topol,
                "\n       80105       70001       70005       'EDGE'"
            )?;
            write!(
                file_topol,
                "\n       80206       70002       70006       'EDGE'"
            )?;
            write!(
                file_topol,
                "\n       80307       70003       70007       'EDGE'"
            )?;
            write!(
                file_topol,
                "\n       80408       70004       70008       'EDGE'"
            )?;

            write!(file_topol, "\n NUMBER OF COMPOUND EDGES:\n       0")?;

            info!("Finished writing TOPOL file");

            // -----------------------
            // --- WRITE GEOM FILE ---
            //
            //
            //
            info!("Writing GEOM file to {path_geom:?}");
            let mut file_geom = std::fs::File::create(path_geom)?;
            write!(file_geom, "'geom file <> {signature}'")?;

            // Write all verts
            write!(
                file_geom,
                "\n NUMBER OF VERTICES:\n       {}\n        VERT       X Y Z                     IDENT\n",
                polycube.structure.nr_verts() + 8
            )?;
            write!(
                file_geom,
                "{}",
                polycube
                    .structure
                    .vert_ids()
                    .iter()
                    .map(|vert_id| {
                        let edge_id = polycube.structure.edges(*vert_id).next().unwrap();
                        let path = layout.edge_to_path.get(&edge_id).unwrap();
                        let first_vertex = path[0];
                        let vert_int = vert_to_id.get_by_left(vert_id).unwrap();
                        let pos = layout.granulated_mesh.position(first_vertex);
                        format!(
                            "       {}       {}  {}  {}       'VERTEX'",
                            vert_int,
                            ryu::Buffer::new().format(pos.x),
                            ryu::Buffer::new().format(pos.y),
                            ryu::Buffer::new().format(pos.z)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            )?;
            write!(
                file_geom,
                "\n       70001       {}  {}  {}       'VERTEX'",
                ryu::Buffer::new().format(v1.x),
                ryu::Buffer::new().format(v1.y),
                ryu::Buffer::new().format(v1.z)
            )?;
            write!(
                file_geom,
                "\n       70002       {}  {}  {}       'VERTEX'",
                ryu::Buffer::new().format(v2.x),
                ryu::Buffer::new().format(v2.y),
                ryu::Buffer::new().format(v2.z)
            )?;
            write!(
                file_geom,
                "\n       70003       {}  {}  {}       'VERTEX'",
                ryu::Buffer::new().format(v3.x),
                ryu::Buffer::new().format(v3.y),
                ryu::Buffer::new().format(v3.z)
            )?;
            write!(
                file_geom,
                "\n       70004       {}  {}  {}       'VERTEX'",
                ryu::Buffer::new().format(v4.x),
                ryu::Buffer::new().format(v4.y),
                ryu::Buffer::new().format(v4.z)
            )?;
            write!(
                file_geom,
                "\n       70005       {}  {}  {}       'VERTEX'",
                ryu::Buffer::new().format(v5.x),
                ryu::Buffer::new().format(v5.y),
                ryu::Buffer::new().format(v5.z)
            )?;
            write!(
                file_geom,
                "\n       70006       {}  {}  {}       'VERTEX'",
                ryu::Buffer::new().format(v6.x),
                ryu::Buffer::new().format(v6.y),
                ryu::Buffer::new().format(v6.z)
            )?;
            write!(
                file_geom,
                "\n       70007       {}  {}  {}       'VERTEX'",
                ryu::Buffer::new().format(v7.x),
                ryu::Buffer::new().format(v7.y),
                ryu::Buffer::new().format(v7.z)
            )?;
            write!(
                file_geom,
                "\n       70008       {}  {}  {}       'VERTEX'",
                ryu::Buffer::new().format(v8.x),
                ryu::Buffer::new().format(v8.y),
                ryu::Buffer::new().format(v8.z)
            )?;

            // Write all edges
            write!(
                file_geom,
                "\n NUMBER OF EDGES:\n       {}\n        EDGE       CONTROL POINTS\n",
                polycube.structure.nr_edges() / 2
            )?;
            write!(
                file_geom,
                "{}",
                polycube
                    .structure
                    .edge_ids()
                    .iter()
                    .filter_map(|edge_id| {
                        edge_to_id.get_by_left(edge_id).map(|edge_int| {
                            let verts = quad.edge_to_verts.get(edge_id).unwrap();
                            let mut lines = vec![];
                            let width = verts.len();
                            lines.push(format!("       {}       {}", edge_int, width));
                            for &vert_id in verts {
                                let pos = quad.quad_mesh.position(vert_id);
                                lines.push(format!(
                                    "  {}  {}  {}",
                                    ryu::Buffer::new().format(pos.x),
                                    ryu::Buffer::new().format(pos.y),
                                    ryu::Buffer::new().format(pos.z)
                                ));
                            }
                            lines.join("\n")
                        })
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            )?;

            // Write all faces
            write!(
                file_geom,
                "\n NUMBER OF FACES:\n       {}",
                polycube.structure.nr_faces()
            )?;

            write!(file_geom, "\n FACE        X  Y\n")?;
            write!(
                file_geom,
                "{}",
                polycube
                    .structure
                    .face_ids()
                    .iter()
                    .map(|face_id| {
                        let verts = quad.face_to_verts.get(face_id).unwrap();
                        let mut lines = vec![];
                        let width = verts.len();
                        let height = verts[0].len();
                        lines.push(format!(
                            "       {}       {}  {}",
                            face_to_id.get_by_left(face_id).unwrap(),
                            width,
                            height
                        ));
                        for j in 0..height {
                            #[allow(clippy::needless_range_loop)]
                            for i in 0..width {
                                let vert_id = verts[i][j];
                                let pos = quad.quad_mesh.position(vert_id);
                                let line = format!(
                                    "  {}  {}  {}",
                                    ryu::Buffer::new().format(pos.x),
                                    ryu::Buffer::new().format(pos.y),
                                    ryu::Buffer::new().format(pos.z)
                                );
                                lines.push(line);
                            }
                        }
                        lines.join("\n")
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            )?;

            info!("Finished writing GEOM file");

            // -----------------------
            // --- WRITE CDIM FILE ---
            //
            //
            //
            info!("Writing CDIM file to {path_cdim:?}");
            let mut file_cdim = std::fs::File::create(path_cdim)?;
            write!(file_cdim, "'cdim file <> {signature}'")?;

            // Write all edge lengths
            let loops = solution.loops.keys();
            let edge_per_loop = loops
                .map(|loop_id| {
                    dual.loop_structure
                        .edge_ids()
                        .into_iter()
                        .find(|&segment_id| dual.segment_to_loop(segment_id) == loop_id)
                        .unwrap()
                })
                .flat_map(|segment_id| dual.loop_structure.faces(segment_id).collect_array::<2>())
                .map(|[face1, face2]| {
                    (
                        polycube
                            .region_to_vertex
                            .get_by_left(&face1)
                            .unwrap()
                            .to_owned(),
                        polycube
                            .region_to_vertex
                            .get_by_left(&face2)
                            .unwrap()
                            .to_owned(),
                    )
                })
                .map(|(vertex1, vertex2)| {
                    polycube
                        .structure
                        .edge_between_verts(vertex1, vertex2)
                        .unwrap()
                        .0
                });
            write!(
                file_cdim,
                "\n NUMBER OF USER SPECIFIED EDGE DIMENSIONS:\n       {}\n        EDGE       DIM\n",
                edge_per_loop.clone().count() + 3
            )?;
            write!(
                file_cdim,
                "{}",
                edge_per_loop
                    .clone()
                    .map(|edge_id| {
                        let edge_int = edge_to_id
                            .get_by_left(&edge_id)
                            .or_else(|| edge_to_id.get_by_left(&polycube.structure.twin(edge_id)))
                            .unwrap();
                        let length = polycube.structure.size(edge_id) as usize * min_dim;
                        format!("       {edge_int}       {length}")
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            )?;

            write!(
                file_cdim,
                "\n       80105       {}",
                cartesian_scale * ff_mult * 2.
            )?;
            write!(
                file_cdim,
                "\n       80401       {}",
                cartesian_scale * ff_mult * 2.
            )?;
            write!(
                file_cdim,
                "\n       80102       {}",
                cartesian_scale * ff_mult * 2.
            )?;

            // Write grid levels
            write!(
                file_cdim,
                "\n GRID LEVEL OF BASIC GRID AND COMPUTATIONAL GRID:\n       1 1"
            )?;

            // Write refinement
            write!(
                file_cdim,
                "\n NUMBER OF BLOCKS WITH LOCAL GRID REFINEMENT:\n       0"
            )?;

            // Write edges in x (i) direction
            let x_edges = edge_per_loop
                .clone()
                .filter(|&edge_id| {
                    to_principal_direction(polycube.structure.vector(edge_id)).0
                        == PrincipalDirection::X
                })
                .map(|edge_id| {
                    edge_to_id
                        .get_by_left(&edge_id)
                        .or_else(|| edge_to_id.get_by_left(&polycube.structure.twin(edge_id)))
                        .unwrap()
                })
                .collect_vec();
            write!(
                file_cdim,
                "\n NUMBER OF EDGES IN I-DIRECTION IN CARTESIAN SPACE:\n       {}\n  EDGES:\n  ",
                x_edges.len() + 1
            )?;
            write!(
                file_cdim,
                "{}",
                x_edges
                    .iter()
                    .map(|edge_id| format!("  {edge_id}"))
                    .collect::<Vec<_>>()
                    .join("  ")
            )?;
            write!(file_cdim, "  80105")?;

            // Write edges in y (j) direction
            let y_edges = edge_per_loop
                .clone()
                .filter(|&edge_id| {
                    to_principal_direction(polycube.structure.vector(edge_id)).0
                        == PrincipalDirection::Y
                })
                .map(|edge_id| {
                    edge_to_id
                        .get_by_left(&edge_id)
                        .or_else(|| edge_to_id.get_by_left(&polycube.structure.twin(edge_id)))
                        .unwrap()
                })
                .collect_vec();
            write!(
                file_cdim,
                "\n NUMBER OF EDGES IN J-DIRECTION IN CARTESIAN SPACE:\n       {}\n  EDGES:\n  ",
                y_edges.len() + 1
            )?;
            write!(
                file_cdim,
                "{}",
                y_edges
                    .iter()
                    .map(|edge_id| format!("  {edge_id}"))
                    .collect::<Vec<_>>()
                    .join("  ")
            )?;
            write!(file_cdim, "  80401")?;

            // Write edges in z (k) direction
            let z_edges = edge_per_loop
                .clone()
                .filter(|&edge_id| {
                    to_principal_direction(polycube.structure.vector(edge_id)).0
                        == PrincipalDirection::Z
                })
                .map(|edge_id| {
                    edge_to_id
                        .get_by_left(&edge_id)
                        .or_else(|| edge_to_id.get_by_left(&polycube.structure.twin(edge_id)))
                        .unwrap()
                })
                .collect_vec();
            write!(
                file_cdim,
                "\n NUMBER OF EDGES IN K-DIRECTION IN CARTESIAN SPACE:\n       {}\n  EDGES:\n  ",
                z_edges.len() + 1
            )?;
            write!(
                file_cdim,
                "{}",
                z_edges
                    .iter()
                    .map(|edge_id| format!("  {edge_id}"))
                    .collect::<Vec<_>>()
                    .join("  ")
            )?;
            write!(file_cdim, "  80102")?;

            // Write reference (origin) vertex
            write!(
                file_cdim,
                "\n NUMBER OF VERTICES WITH CARTESIAN COORDINATES:\n       2\n        VERT i j k\n"
            )?;
            let center_to_ref = polycube.structure.position(refv) - cartesian_center;

            let i = vert_to_id.get_by_left(&refv).unwrap();
            let x = center_to_ref.x as i32;
            let y = center_to_ref.y as i32;
            let z = center_to_ref.z as i32;
            let w = (cartesian_scale * ff_mult) as i32;

            // Write symmetry and orientation
            write!(
                file_cdim,
                "       {i} {x} {y} {z}\n       70001 -{w} -{w} -{w}\nSYMMETRY\n       0\nORIENTATION\n       0"
            )?;
            info!("Finished writing CDIM file");

            // ------------------------
            // --- WRITE BCDAT FILE ---
            //
            //
            //
            info!("Writing BCDAT file to {path_bcdat:?}");
            let mut file_bcdat = std::fs::File::create(path_bcdat)?;
            write!(
                file_bcdat,
                "'cdim file <> {signature}'\n\n        15  'NaS solid wall'        {}\n",
                polycube.structure.faces.len()
            )?;
            write!(
                file_bcdat,
                "{}",
                polycube
                    .structure
                    .face_ids()
                    .into_iter()
                    .map(|face_id| format!("  {}", face_to_id.get_by_left(&face_id).unwrap()))
                    .collect::<Vec<_>>()
                    .join("  ")
            )?;
            info!("Finished writing BCDAT file");

            // ------------------------
            // --- WRITE SEG FILE ---
            //
            //
            //

            for (path, dir) in [
                (path_xloops, PrincipalDirection::X),
                (path_yloops, PrincipalDirection::Y),
                (path_zloops, PrincipalDirection::Z),
            ] {
                info!("Writing SEG file to {path:?} for direction {dir:?}");

                let mut file_seg = std::fs::File::create(path)?;

                for xloop in solution.get_loops_in_direction(dir) {
                    let edges_through_loop = &solution.loops.get(xloop).unwrap().edges;
                    let mut positions_on_loop = edges_through_loop
                        .iter()
                        .map(|&edge_id| solution.get_coordinates_of_loop_in_edge(xloop, edge_id))
                        .collect_vec();
                    // repeat first position to make a loop
                    positions_on_loop.push(positions_on_loop[0]);
                    let lines = positions_on_loop
                        .iter()
                        .map(|pos| {
                            format!(
                                "    {}    {}    {}    ",
                                ryu::Buffer::new().format(pos.x),
                                ryu::Buffer::new().format(pos.y),
                                ryu::Buffer::new().format(pos.z)
                            )
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    write!(
                        file_seg,
                        "    1\n    {}\n    1D SEGMENT\n{}\n",
                        positions_on_loop.len(),
                        lines
                    )?;
                }
            }
        }

        Ok(())
    }
}
