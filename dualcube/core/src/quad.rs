use crate::layout::Layout;
use crate::polycube::{Polycube, POLYCUBE};
use crate::prelude::*;
use bimap::BiHashMap;
use faer::sparse::{SparseColMat, Triplet};
use faer::Mat;
use faer_gmres::gmres;
use itertools::Itertools;
use mehsh::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::{HashMap, HashSet};

mehsh::prelude::define_tag!(QUAD);

#[derive(Clone, Debug, Default)]
pub struct Quad {
    pub triangle_mesh_polycube: Mesh<INPUT>,
    pub quad_mesh_polycube: Mesh<QUAD>,
    pub quad_mesh: Mesh<QUAD>,
    pub face_to_verts: HashMap<FaceKey<POLYCUBE>, Vec<Vec<VertKey<QUAD>>>>,
    pub edge_to_verts: HashMap<EdgeKey<POLYCUBE>, Vec<VertKey<QUAD>>>,
    pub frozen: HashSet<VertKey<QUAD>>,
}

// serialize Quad to nothing
impl Serialize for Quad {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_unit()
    }
}

// deserialize Quad
impl<'de> Deserialize<'de> for Quad {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Quad::default())
    }
}

impl Quad {
    fn arc_length_parameterization(verts: &[Vector3D]) -> Vec<f64> {
        // Arc-length parameterization of list of points to [0, 1] interval
        let distances = std::iter::once(0.0).chain(verts.windows(2).map(|w| (w[1] - w[0]).norm())).collect_vec();
        let total_length = distances.iter().sum::<f64>();

        distances
            .into_iter()
            .scan(0.0, |acc, d| {
                *acc += d;
                Some(*acc / total_length)
            })
            .collect_vec()
    }

    #[must_use]
    pub fn from_layout(layout: &Layout, polycube: &Polycube, omega: usize) -> Option<Self> {
        let mut triangle_mesh_polycube = layout.granulated_mesh.clone();

        let mut edges_done: HashMap<EdgeKey<POLYCUBE>, Vec<usize>> = HashMap::new();
        let mut corners_done: HashMap<VertKey<POLYCUBE>, usize> = HashMap::new();

        let mut faces = vec![];
        let mut vertex_positions = vec![];

        // set of frozen vertices. Should be vertices that lie on important boundaries or features.
        let mut frozen = HashSet::new();

        // For every patch in the layout (corresponding to a face in the polycube), we map this patch to a unit square
        // 1. Map the boundary of the patch to the boundary of a unit square via arc-length parameterization
        // 2. Map the interior of the patch to the interior of the unit square via mean-value coordinates (MVC)
        // FOR SOME REASON IMPORTANT TO HAVE THE POLYCUBE WITH EDGELENGTHS = 1 FOR THE MAPPING, AND USE A SEPERATE SCALED ONE TO COMPUTE THE PROPER QUAD DISTRIBUTIONS
        let polycube = Polycube::from_dual(&layout.dual_ref);
        let mut polycube_resized = polycube.clone();
        polycube_resized.resize(&layout.dual_ref, Some(layout));
        let polycube = polycube.structure;

        let mut queue = vec![];
        queue.push(polycube.face_ids()[0]);

        let mut patches_done = HashSet::new();

        let mut face_to_verts_usize: HashMap<FaceKey<POLYCUBE>, Vec<Vec<usize>>> = HashMap::new();
        let mut face_to_verts: HashMap<FaceKey<POLYCUBE>, Vec<Vec<VertKey<QUAD>>>> = HashMap::new();

        let mut edge_to_verts_usize: HashMap<EdgeKey<POLYCUBE>, Vec<usize>> = HashMap::new();
        let mut edge_to_verts: HashMap<EdgeKey<POLYCUBE>, Vec<VertKey<QUAD>>> = HashMap::new();

        while let Some(patch_id) = queue.pop() {
            if patches_done.contains(&patch_id) {
                continue;
            }
            patches_done.insert(patch_id);
            let neighboring_patches = polycube.neighbors(patch_id);
            for &neighbor in &neighboring_patches {
                if !patches_done.contains(&neighbor) {
                    queue.push(neighbor);
                }
            }

            // A map for each vertex in the patch to its corresponding 2D coordinate in the unit square
            let mut map_to_2d = HashMap::new();

            let patch_edges = polycube.edges(patch_id);

            // Edge1 is mapped to unit edge (0,1) -> (1,1)
            let edge1 = patch_edges[0];
            let boundary1 = layout.edge_to_path.get(&edge1).unwrap();
            let corner1 = polycube.vertices(edge1)[0];

            // Edge2 is mapped to unit edge (1,1) -> (1,0)
            let edge2 = patch_edges[1];
            let boundary2 = layout.edge_to_path.get(&edge2).unwrap();
            let corner2 = polycube.vertices(edge2)[0];

            // Edge3 is mapped to unit edge (1,0) -> (0,0)
            let edge3 = patch_edges[2];
            let boundary3 = layout.edge_to_path.get(&edge3).unwrap();
            let corner3 = polycube.vertices(edge3)[0];

            // Edge4 is mapped to unit edge (0,0) -> (0,1)
            let edge4 = patch_edges[3];
            let boundary4 = layout.edge_to_path.get(&edge4).unwrap();
            let corner4 = polycube.vertices(edge4)[0];

            // d3p1 = (x1, y1, z1)
            // d3p2 = (x2, y2, z2)
            // d3p3 = (x3, y3, z3)
            // d3p4 = (x4, y4, z4)
            // Figure out which coordinate is constant.
            // Then map
            // d2p1 = (u1, v1)
            // d2p2 = (u2, v1)
            // d2p3 = (u2, v2)
            // d2p4 = (u1, v2)
            let p1 = polycube.vertices(edge1)[0];
            let d3p1 = polycube.position(p1);
            let p2 = polycube.vertices(edge2)[0];
            let d3p2 = polycube.position(p2);
            let p3 = polycube.vertices(edge3)[0];
            let d3p3 = polycube.position(p3);
            let p4 = polycube.vertices(edge4)[0];
            let d3p4 = polycube.position(p4);

            let coordinates = if d3p1.x == d3p2.x && d3p1.x == d3p3.x && d3p1.x == d3p4.x {
                (1, 2, 0)
            } else if d3p1.y == d3p2.y && d3p1.y == d3p3.y && d3p1.y == d3p4.y {
                (0, 2, 1)
            } else if d3p1.z == d3p2.z && d3p1.z == d3p3.z && d3p1.z == d3p4.z {
                (0, 1, 2)
            } else {
                panic!("The face is not axis-aligned!")
            };

            let d2p1 = Vector2D::new(d3p1[coordinates.0], d3p1[coordinates.1]);
            let d2p2 = Vector2D::new(d3p2[coordinates.0], d3p2[coordinates.1]);
            let d2p3 = Vector2D::new(d3p3[coordinates.0], d3p3[coordinates.1]);
            let d2p4 = Vector2D::new(d3p4[coordinates.0], d3p4[coordinates.1]);

            // Interpolate the vertices on the edges (paths) between p1 and p2, p2 and p3, p3 and p4, p4 and p1
            // Parameterize (using arc-length) the vertices
            // From p1 to p2
            let interpolation1 = Self::arc_length_parameterization(&boundary1.iter().map(|&v| layout.granulated_mesh.position(v)).collect_vec());
            for (i, &v) in boundary1.iter().enumerate() {
                let mapped_pos = d2p1 * (1.0 - interpolation1[i]) + d2p2 * interpolation1[i];
                map_to_2d.insert(v, mapped_pos);
            }
            // From p2 to p3
            let interpolation2 = Self::arc_length_parameterization(&boundary2.iter().map(|&v| layout.granulated_mesh.position(v)).collect_vec());
            for (i, &v) in boundary2.iter().enumerate() {
                let mapped_pos = d2p2 * (1.0 - interpolation2[i]) + d2p3 * interpolation2[i];
                map_to_2d.insert(v, mapped_pos);
            }
            // From p3 to p4
            let interpolation3 = Self::arc_length_parameterization(&boundary3.iter().map(|&v| layout.granulated_mesh.position(v)).collect_vec());
            for (i, &v) in boundary3.iter().enumerate() {
                let mapped_pos = d2p3 * (1.0 - interpolation3[i]) + d2p4 * interpolation3[i];
                map_to_2d.insert(v, mapped_pos);
            }
            // From p4 to p1
            let interpolation4 = Self::arc_length_parameterization(&boundary4.iter().map(|&v| layout.granulated_mesh.position(v)).collect_vec());
            for (i, &v) in boundary4.iter().enumerate() {
                let mapped_pos = d2p4 * (1.0 - interpolation4[i]) + d2p1 * interpolation4[i];
                map_to_2d.insert(v, mapped_pos);
            }

            // Now we have the boundary of the patch mapped to the unit square
            // We need to map the interior of the patch to the interior of the unit square
            // We can use mean-value coordinates (MVC) to do this

            let all_verts = layout
                .face_to_patch
                .get(&patch_id)
                .unwrap()
                .faces
                .iter()
                .flat_map(|&face_id| layout.granulated_mesh.vertices(face_id))
                .collect::<HashSet<_>>();

            let interior_verts = all_verts.iter().filter(|&&v| !map_to_2d.contains_key(&v)).copied().collect_vec();

            let mut vert_to_id = BiHashMap::new();
            for (i, &v) in interior_verts.iter().enumerate() {
                vert_to_id.insert(v, i);
            }

            let n = interior_verts.len();
            let mut triplets = Vec::new();
            let mut bu = vec![0.0; n];
            let mut bv = vec![0.0; n];
            for i in 0..n {
                triplets.push((i, i, 1.0));
            }

            for &v0 in &interior_verts {
                let row = vert_to_id.get_by_left(&v0).unwrap().to_owned();
                // For vi (all neighbors of v0), we calculate weight wi, where wi = tan(alpha_{i-1} / 2) + tan(alpha_i / 2) / || vi - v0 ||

                let neighbors = layout.granulated_mesh.neighbors(v0);
                let k = neighbors.len();

                let w = (0..k)
                    .map(|i| {
                        let vip1 = neighbors[(i + 1) % k];
                        let eip1 = layout.granulated_mesh.edge_between_verts(v0, vip1).unwrap().0;
                        let vi = neighbors[i];
                        let ei = layout.granulated_mesh.edge_between_verts(v0, vi).unwrap().0;
                        let vim1 = neighbors[(i + k - 1) % k];
                        let eim1 = layout.granulated_mesh.edge_between_verts(v0, vim1).unwrap().0;

                        let alpha_im1 = layout.granulated_mesh.angle(eim1, ei);
                        let alpha_i = layout.granulated_mesh.angle(ei, eip1);
                        let len_ei = (layout.granulated_mesh.position(v0) - layout.granulated_mesh.position(vi)).norm();

                        ((alpha_im1 / 2.0).tan() + (alpha_i / 2.0).tan()) / len_ei
                    })
                    .collect_vec();

                let sum_w = w.iter().sum::<f64>();
                let weights = w.iter().map(|&wi| wi / sum_w).collect_vec();

                for i in 0..k {
                    let vi = neighbors[i];
                    let is_boundary = map_to_2d.contains_key(&vi);

                    if is_boundary {
                        let mapped_pos = map_to_2d.get(&vi).unwrap().to_owned();
                        bu[row] += weights[i] * mapped_pos.x;
                        bv[row] += weights[i] * mapped_pos.y;
                    } else {
                        let col = vert_to_id.get_by_left(&vi).unwrap().to_owned();
                        triplets.push((row, col, -weights[i]));
                    }
                }
            }

            let b_u = Mat::from_fn(n, 1, |i, _| bu[i]);
            let b_v = Mat::from_fn(n, 1, |i, _| bv[i]);

            assert!(bu.iter().all(|v| v.is_finite()), "bu contains NaN or Inf");
            assert!(bv.iter().all(|v| v.is_finite()), "bv contains NaN or Inf");

            let mut x_u = Mat::from_fn(n, 1, |_, _| 0.0);
            let mut x_v = Mat::from_fn(n, 1, |_, _| 0.0);

            let faer_triplets = triplets.into_iter().map(|(i, j, v)| Triplet::new(i, j, v)).collect::<Vec<_>>();
            let a = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &faer_triplets).unwrap();
            if !interior_verts.is_empty() {
                if gmres(a.as_ref(), b_u.as_ref(), x_u.as_mut(), 1000, 1e-8, None).is_err() {
                    return None;
                }
                if gmres(a.as_ref(), b_v.as_ref(), x_v.as_mut(), 1000, 1e-8, None).is_err() {
                    return None;
                }
            }

            for &v in &all_verts {
                let is_boundary = map_to_2d.contains_key(&v);
                let position = if is_boundary {
                    let mapped_pos = map_to_2d.get(&v).unwrap().to_owned();
                    match coordinates {
                        (1, 2, 0) => Vector3D::new(d3p1.x, mapped_pos.x, mapped_pos.y),
                        (0, 2, 1) => Vector3D::new(mapped_pos.x, d3p1.y, mapped_pos.y),
                        (0, 1, 2) => Vector3D::new(mapped_pos.x, mapped_pos.y, d3p1.z),
                        _ => panic!("Invalid coordinates"),
                    }
                } else {
                    let mapped_pos = Vector2D::new(
                        x_u[(vert_to_id.get_by_left(&v).unwrap().to_owned(), 0)],
                        x_v[(vert_to_id.get_by_left(&v).unwrap().to_owned(), 0)],
                    );
                    match coordinates {
                        (1, 2, 0) => Vector3D::new(d3p1.x, mapped_pos.x, mapped_pos.y),
                        (0, 2, 1) => Vector3D::new(mapped_pos.x, d3p1.y, mapped_pos.y),
                        (0, 1, 2) => Vector3D::new(mapped_pos.x, mapped_pos.y, d3p1.z),
                        _ => panic!("Invalid coordinates"),
                    }
                };

                triangle_mesh_polycube.set_position(v, position);
            }

            let grid_width = polycube_resized.structure.size(edge1) * 2.;
            // assert!(polycube.size(edge3) == grid_width);
            let grid_n = grid_width as usize * omega + 1;

            let grid_height = polycube_resized.structure.size(edge2) * 2.;
            // assert!(polycube.size(edge4) == grid_height);
            let grid_m = grid_height as usize * omega + 1;

            let to_pos = |i: usize, j: usize| {
                let u = i as f64 / (grid_m - 1) as f64;
                let v = j as f64 / (grid_n - 1) as f64;
                let e1_vector = d3p2 - d3p1;
                let e2_vector = d3p3 - d3p2;
                d3p1 + e1_vector * v + e2_vector * u
            };

            let mut vert_map = vec![vec![None; grid_n]; grid_m];
            // Fill the vert_map with Some(v) for all boundary vertices that were already created

            // First check if the 4 corners are already done
            if corners_done.contains_key(&corner1) {
                let corner1_pos = corners_done.get(&corner1).unwrap().to_owned();
                vert_map[0][0] = Some(corner1_pos);
            }
            if corners_done.contains_key(&corner2) {
                let corner2_pos = corners_done.get(&corner2).unwrap().to_owned();
                vert_map[0][grid_n - 1] = Some(corner2_pos);
            }
            if corners_done.contains_key(&corner3) {
                let corner3_pos = corners_done.get(&corner3).unwrap().to_owned();
                vert_map[grid_m - 1][grid_n - 1] = Some(corner3_pos);
            }
            if corners_done.contains_key(&corner4) {
                let corner4_pos = corners_done.get(&corner4).unwrap().to_owned();
                vert_map[grid_m - 1][0] = Some(corner4_pos);
            }

            // Edge1 which is i=0 and j=0 to j=grid_n-1
            if edges_done.contains_key(&edge1) {
                let edge_verts = edges_done.get(&edge1).unwrap().to_owned();
                assert!(edge_verts.len() == grid_n);
                for (j, &vert) in edge_verts.iter().enumerate() {
                    vert_map[0][j] = Some(vert);
                }
                edge_to_verts_usize.insert(edge1, edge_verts.clone());
            }

            // Edge2 which is j=grid_n-1 and i=0 to i=grid_m-1
            if edges_done.contains_key(&edge2) {
                let edge_verts = edges_done.get(&edge2).unwrap().to_owned();
                assert!(edge_verts.len() == grid_m);
                for (i, &vert) in edge_verts.iter().enumerate() {
                    vert_map[i][grid_n - 1] = Some(vert);
                }
                edge_to_verts_usize.insert(edge2, edge_verts.clone());
            }

            // Edge3 which is i=grid_m-1 and j=grid_n-1 to j=0
            if edges_done.contains_key(&edge3) {
                let edge_verts = edges_done.get(&edge3).unwrap().to_owned();
                assert!(edge_verts.len() == grid_n);
                for (j, &vert) in edge_verts.iter().enumerate() {
                    vert_map[grid_m - 1][grid_n - 1 - j] = Some(vert);
                }
                edge_to_verts_usize.insert(edge3, edge_verts.clone());
            }

            // Edge4 which is j=0 and i=grid_m-1 to i=0
            if edges_done.contains_key(&edge4) {
                let edge_verts = edges_done.get(&edge4).unwrap().to_owned();
                assert!(edge_verts.len() == grid_m);
                for (i, &vert) in edge_verts.iter().enumerate() {
                    vert_map[grid_m - 1 - i][0] = Some(vert);
                }
                edge_to_verts_usize.insert(edge4, edge_verts.clone());
            }

            for i in 0..grid_m - 1 {
                for j in 0..grid_n - 1 {
                    // Define a quadrilateral face with vertices i,j, i,j+1, i+1,j+1, i+1,j
                    // First check if a vertex already exists at this position, if not, create a new vertex (the counter is increased)
                    if vert_map[i][j].is_none() {
                        vert_map[i][j] = Some(vertex_positions.len());
                        vertex_positions.push(to_pos(i, j));
                    }

                    if vert_map[i][j + 1].is_none() {
                        vert_map[i][j + 1] = Some(vertex_positions.len());
                        vertex_positions.push(to_pos(i, j + 1));
                    }

                    if vert_map[i + 1][j + 1].is_none() {
                        vert_map[i + 1][j + 1] = Some(vertex_positions.len());
                        vertex_positions.push(to_pos(i + 1, j + 1));
                    }

                    if vert_map[i + 1][j].is_none() {
                        vert_map[i + 1][j] = Some(vertex_positions.len());
                        vertex_positions.push(to_pos(i + 1, j));
                    }

                    // Now we have 4 vertices, we can create a face
                    let (v0, v1, v2, v3) = (
                        vert_map[i][j].unwrap(),
                        vert_map[i][j + 1].unwrap(),
                        vert_map[i + 1][j + 1].unwrap(),
                        vert_map[i + 1][j].unwrap(),
                    );

                    faces.push(vec![v0, v1, v2, v3]);
                }
            }

            face_to_verts_usize.insert(patch_id, vert_map.iter().map(|row| row.iter().filter_map(|&v| v).collect_vec()).collect_vec());

            // Add the boundary vertices to the edges_done map
            corners_done.insert(corner1, vert_map[0][0].unwrap());
            corners_done.insert(corner2, vert_map[0][grid_n - 1].unwrap());
            corners_done.insert(corner3, vert_map[grid_m - 1][grid_n - 1].unwrap());
            corners_done.insert(corner4, vert_map[grid_m - 1][0].unwrap());

            // Edge1 which is i=0 and j=0 to j=grid_n-1
            for i in [0] {
                let mut vs = vec![];
                // REVERSE
                for j in (0..grid_n).rev() {
                    vs.push(vert_map[i][j].unwrap());
                }
                // ADD FOR TWIN
                edges_done.insert(polycube.twin(edge1), vs);
            }
            // Edge2 which is j=grid_n-1 and i=0 to i=grid_m-1
            for j in [grid_n - 1] {
                let mut vs = vec![];

                // REVERSE
                for i in (0..grid_m).rev() {
                    vs.push(vert_map[i][j].unwrap());
                }
                // ADD FOR TWIN
                edges_done.insert(polycube.twin(edge2), vs);
            }
            // Edge3 which is i=grid_m-1 and j=grid_n-1 to j=0
            for i in [grid_m - 1] {
                let mut vs = vec![];

                // REVERSE
                for j in 0..grid_n {
                    vs.push(vert_map[i][j].unwrap());
                }
                // ADD FOR TWIN
                edges_done.insert(polycube.twin(edge3), vs);
            }
            // Edge4 which is j=0 and i=grid_m-1 to i=0
            for j in [0] {
                let mut vs = vec![];

                #[allow(clippy::needless_range_loop)]
                for i in 0..grid_m {
                    vs.push(vert_map[i][j].unwrap());
                }
                // ADD FOR TWIN
                edges_done.insert(polycube.twin(edge4), vs);
            }
        }

        assert!(patches_done.len() == polycube.face_ids().len(), "Not all patches were done!");

        // Create the polycube quad mesh:
        if let Ok((quad_mesh_polycube, vert_id_map, _)) = Mesh::<QUAD>::from(&faces, &vertex_positions) {
            for (face_id, vert_ids) in &face_to_verts_usize {
                // Convert usize to VertKey<QUAD>
                let vert_keys = vert_ids
                    .iter()
                    .map(|row| row.iter().map(|&v| vert_id_map.key(v).unwrap().to_owned()).collect_vec())
                    .collect_vec();
                face_to_verts.insert(face_id.to_owned(), vert_keys);
            }

            for (edge_id, vert_ids) in &edge_to_verts_usize {
                // Convert usize to VertKey<QUAD>
                let vert_keys = vert_ids.iter().map(|&v| vert_id_map.key(v).unwrap().to_owned()).collect_vec();
                edge_to_verts.insert(edge_id.to_owned(), vert_keys.clone());
                let twin_id = polycube.twin(*edge_id);
                let rev_vert_keys = vert_keys.iter().rev().cloned().collect_vec();
                edge_to_verts.insert(twin_id, rev_vert_keys);
            }

            // For each vert_id in quad_mesh_polycube, check the surrounding normals. If surrounding normals are all equal, then this vertex is NOT frozen. Else it is frozen (its a boundary vertex)
            for vert_id in quad_mesh_polycube.vert_ids() {
                let mut surrounding_normals = HashSet::new();
                for neighbor in quad_mesh_polycube.neighbors(vert_id) {
                    surrounding_normals.insert(to_principal_direction(quad_mesh_polycube.normal(neighbor)));
                }
                if surrounding_normals.len() > 1 {
                    frozen.insert(vert_id);
                }
            }

            // Create the quad mesh
            // First, create copy of quad_mesh_polycube
            let mut quad_mesh = quad_mesh_polycube.clone();
            let triangle_lookup = triangle_mesh_polycube.bvh();

            // Now, we need to set the positions of the vertices in the quad mesh based on barycentric coordinates of the mapped triangles
            for vert_id in quad_mesh.vert_ids() {
                // Get the nearest triangle in the triangle mesh (polycube map)
                let point = quad_mesh_polycube.position(vert_id);
                let triangle = triangle_lookup.nearest(&[point.x, point.y, point.z]);
                let corners = triangle_mesh_polycube.vertices(triangle);

                // check distance from point to triangle
                let distance1 = mehsh::utils::geom::distance_to_triangle(
                    point,
                    (
                        triangle_mesh_polycube.position(corners[0]),
                        triangle_mesh_polycube.position(corners[1]),
                        triangle_mesh_polycube.position(corners[2]),
                    ),
                );

                if distance1 > 0.001 {
                    log::warn!("Distance from point to triangle is very large: {distance1} (> 0.001)");
                }

                // Calculate the barycentric coordinates of the point in the triangle
                let (u, v, w) = mehsh::utils::geom::calculate_barycentric_coordinates(
                    point,
                    (
                        triangle_mesh_polycube.position(corners[0]),
                        triangle_mesh_polycube.position(corners[1]),
                        triangle_mesh_polycube.position(corners[2]),
                    ),
                );

                // Do the inverse of the barycentric coordinates, with the original triangle in granulated mesh
                let new_position = mehsh::utils::geom::inverse_barycentric_coordinates(
                    u,
                    v,
                    w,
                    (
                        layout.granulated_mesh.position(corners[0]),
                        layout.granulated_mesh.position(corners[1]),
                        layout.granulated_mesh.position(corners[2]),
                    ),
                );

                // quad_mesh.set_position(vert_id, new_position);

                if distance1 < 0.001 {
                    quad_mesh.set_position(vert_id, new_position);
                }
            }

            Some(Self {
                triangle_mesh_polycube,
                quad_mesh_polycube,
                quad_mesh,
                face_to_verts,
                edge_to_verts,
                frozen,
            })
        } else {
            panic!("Failed to create quad mesh from faces and vertex positions");
        }
    }

    pub fn smoothing(&mut self, _iterations: usize, reference: &Mesh<INPUT>, _fix_sharp: bool) {
        log::info!("Performing 200 iterations of Laplacian smoothing on the quad mesh.");

        for i in 0..200 {
            log::info!("Smoothing iteration {}", i);
            let new_positions = self
                .quad_mesh
                .vert_ids()
                .into_par_iter()
                .filter(|&vert_id| !self.frozen.contains(&vert_id))
                .flat_map(|vert_id| {
                    let mut p_i = self.quad_mesh.position(vert_id);

                    let neighbors = self.quad_mesh.neighbors(vert_id);

                    let sum = neighbors.iter().map(|&n| self.quad_mesh.position(n)).sum::<Vector3D>();
                    let avg = sum / neighbors.len() as f64;
                    let delta_p_i = avg - p_i;

                    // inflate factor, > 0
                    let lambda = 0.35;
                    // shrink factor, < 0
                    let mu = -0.34;

                    if i % 2 == 0 {
                        // shrink
                        p_i += lambda * delta_p_i;
                    } else {
                        // inflate
                        p_i += mu * delta_p_i;
                    }

                    Some((vert_id, p_i))
                })
                .collect::<Vec<_>>();

            for (vert_id, closest_point) in new_positions {
                self.quad_mesh.set_position(vert_id, closest_point);
            }
        }
    }
}
