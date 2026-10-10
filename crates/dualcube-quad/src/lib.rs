use bimap::BiHashMap;
use dualcube_primal::prelude::*;
use dualcube_types::prelude::*;
use faer::Mat;
use faer::prelude::Solve;
use faer::sparse::{SparseColMat, Triplet};
use orx_parallel::*;

fn arc_length_parameterization(verts: &[Vector3D]) -> Vec<f64> {
    // Arc-length parameterization of list of points to [0, 1] interval
    let distances = std::iter::once(0.0)
        .chain(verts.windows(2).map(|w| (w[1] - w[0]).norm()))
        .collect_vec();
    let total_length = distances.iter().sum::<f64>();

    if !(total_length > 0.0 && total_length.is_finite()) {
        // Degenerate path (coincident points): fall back to uniform spacing.
        let last = verts.len().saturating_sub(1).max(1) as f64;
        return (0..verts.len()).map(|i| i as f64 / last).collect_vec();
    }

    distances
        .into_iter()
        .scan(0.0, |acc, d| {
            *acc += d;
            Some(*acc / total_length)
        })
        .collect_vec()
}

/// The density of a quad mesh.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QuadDensity {
    /// Quads with edges of about this factor times the mean edge length of the input mesh: the density follows the
    /// resolution of the input mesh (quads much smaller than its triangles add nothing).
    Auto(f64),
    /// This many quads along a polycube edge of average length (and proportionally more or fewer along the others).
    Fixed(usize),
}

/// The default factor of `QuadDensity::Auto`.
pub const DEFAULT_QUAD_FACTOR: f64 = 1.5;

impl Default for QuadDensity {
    fn default() -> Self {
        Self::Auto(DEFAULT_QUAD_FACTOR)
    }
}

// The target edge length of the quads (`Auto`), or the number of quads per unit edge of the polycube (`Fixed`).
#[derive(Clone, Copy)]
enum Resolution {
    Length(f64),
    PerUnit(f64),
}

impl Resolution {
    fn of(density: QuadDensity, mesh: &Mesh<INPUT>) -> Self {
        match density {
            QuadDensity::Fixed(omega) => Self::PerUnit(omega.max(1) as f64),
            QuadDensity::Auto(factor) => {
                let edges = mesh.edge_ids();
                let mean =
                    edges.iter().map(|&e| mesh.size(e)).sum::<f64>() / edges.len().max(1) as f64;
                Self::Length((factor * mean).max(f64::MIN_POSITIVE))
            }
        }
    }
}

// The number of quads along every polycube edge, in proportion to the geometric lengths of the edges (from the layout):
// either a number per unit edge of the polycube (with unit edge lengths), so the density does not depend on the scale of
// the mesh, or a target length of the quad edges. Twin edges and opposite edges of a face (a rectangle) get the same
// count, so the grids of neighboring faces match.
fn edge_resolutions(
    polycube: &Mesh<POLYCUBE>,
    geometric: &Mesh<POLYCUBE>,
    resolution: Resolution,
) -> Option<HashMap<EdgeKey<POLYCUBE>, usize>> {
    let edges = polycube.edge_ids();
    let unit_total: f64 = edges.iter().map(|&e| polycube.size(e)).sum();
    let geometric_total: f64 = edges.iter().map(|&e| geometric.size(e)).sum();
    if !(unit_total > 0. && geometric_total > 0.) {
        return None;
    }
    let scale = unit_total / geometric_total;

    // Classes of edges that need the same count: twins, and opposite edges of a face.
    let index: HashMap<EdgeKey<POLYCUBE>, usize> =
        edges.iter().enumerate().map(|(i, &e)| (e, i)).collect();
    let mut parent = (0..edges.len()).collect_vec();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    let mut union = |a: EdgeKey<POLYCUBE>, b: EdgeKey<POLYCUBE>| {
        let (ra, rb) = (find(&mut parent, index[&a]), find(&mut parent, index[&b]));
        parent[ra] = rb;
    };
    for &edge in &edges {
        union(edge, polycube.twin(edge));
    }
    for face in polycube.face_ids() {
        let [e1, e2, e3, e4] = polycube.edges(face).collect_array::<4>()?;
        union(e1, e3);
        union(e2, e4);
    }
    let mut lengths: HashMap<usize, (f64, usize)> = HashMap::new();
    for &edge in &edges {
        let class = find(&mut parent, index[&edge]);
        let entry = lengths.entry(class).or_default();
        entry.0 += geometric.size(edge) * scale;
        entry.1 += 1;
    }
    Some(
        edges
            .iter()
            .map(|&edge| {
                let (sum, count) = lengths[&find(&mut parent, index[&edge])];
                let length = sum / count as f64;
                let cells = match resolution {
                    Resolution::PerUnit(omega) => length * omega,
                    // The lengths are scaled to the unit polycube.
                    Resolution::Length(target) => length / scale / target,
                };
                let cells = cells.round().max(1.) as usize;
                (edge, cells)
            })
            .collect(),
    )
}

#[must_use]
pub fn build_quad_from_layout(layout: &Layout, density: QuadDensity) -> Option<Quad> {
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
    let resolution = Resolution::of(density, &layout.dual_ref.mesh_ref);
    let Some(resolution) = edge_resolutions(&polycube, &polycube_resized.structure, resolution)
    else {
        warn!("Cannot construct quad mesh: the polycube is not made of quadrilaterals");
        return None;
    };

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
        for neighbor in polycube.neighbors(patch_id) {
            if !patches_done.contains(&neighbor) {
                queue.push(neighbor);
            }
        }

        // A map for each vertex in the patch to its corresponding 2D coordinate in the unit square
        let mut map_to_2d = HashMap::new();

        let Some([edge1, edge2, edge3, edge4]) = polycube.edges(patch_id).collect_array::<4>()
        else {
            warn!("Cannot construct quad mesh: a polycube face is not a quadrilateral");
            return None;
        };
        let (Some(boundary1), Some(boundary2), Some(boundary3), Some(boundary4)) = (
            layout.edge_to_path.get(&edge1),
            layout.edge_to_path.get(&edge2),
            layout.edge_to_path.get(&edge3),
            layout.edge_to_path.get(&edge4),
        ) else {
            warn!("Cannot construct quad mesh: the layout is incomplete");
            return None;
        };

        // Edge1 is mapped to unit edge (0,1) -> (1,1), edge2 to (1,1) -> (1,0), edge3 to (1,0) -> (0,0), and edge4 to
        // (0,0) -> (0,1).
        let corner1 = polycube.vertices(edge1).next()?;
        let corner2 = polycube.vertices(edge2).next()?;
        let corner3 = polycube.vertices(edge3).next()?;
        let corner4 = polycube.vertices(edge4).next()?;

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
        let p1 = polycube.vertices(edge1).next().unwrap();
        let d3p1 = polycube.position(p1);
        let p2 = polycube.vertices(edge2).next().unwrap();
        let d3p2 = polycube.position(p2);
        let p3 = polycube.vertices(edge3).next().unwrap();
        let d3p3 = polycube.position(p3);
        let p4 = polycube.vertices(edge4).next().unwrap();
        let d3p4 = polycube.position(p4);

        // The constant coordinate of the (axis-aligned) face.
        let constant = |axis: usize| {
            let values = [d3p1[axis], d3p2[axis], d3p3[axis], d3p4[axis]];
            let spread = values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - values.iter().copied().fold(f64::INFINITY, f64::min);
            spread <= 1e-9 * (1. + d3p1.norm())
        };
        let coordinates = if constant(0) {
            (1, 2, 0)
        } else if constant(1) {
            (0, 2, 1)
        } else if constant(2) {
            (0, 1, 2)
        } else {
            warn!("Cannot construct quad mesh: a polycube face is not axis-aligned");
            return None;
        };

        let d2p1 = Vector2D::new(d3p1[coordinates.0], d3p1[coordinates.1]);
        let d2p2 = Vector2D::new(d3p2[coordinates.0], d3p2[coordinates.1]);
        let d2p3 = Vector2D::new(d3p3[coordinates.0], d3p3[coordinates.1]);
        let d2p4 = Vector2D::new(d3p4[coordinates.0], d3p4[coordinates.1]);

        // Interpolate the vertices on the edges (paths) between p1 and p2, p2 and p3, p3 and p4, p4 and p1
        // Parameterize (using arc-length) the vertices
        // From p1 to p2
        let interpolation1 = arc_length_parameterization(
            &boundary1
                .iter()
                .map(|&v| layout.granulated_mesh.position(v))
                .collect_vec(),
        );
        for (i, &v) in boundary1.iter().enumerate() {
            let mapped_pos = d2p1 * (1.0 - interpolation1[i]) + d2p2 * interpolation1[i];
            map_to_2d.insert(v, mapped_pos);
        }
        // From p2 to p3
        let interpolation2 = arc_length_parameterization(
            &boundary2
                .iter()
                .map(|&v| layout.granulated_mesh.position(v))
                .collect_vec(),
        );
        for (i, &v) in boundary2.iter().enumerate() {
            let mapped_pos = d2p2 * (1.0 - interpolation2[i]) + d2p3 * interpolation2[i];
            map_to_2d.insert(v, mapped_pos);
        }
        // From p3 to p4
        let interpolation3 = arc_length_parameterization(
            &boundary3
                .iter()
                .map(|&v| layout.granulated_mesh.position(v))
                .collect_vec(),
        );
        for (i, &v) in boundary3.iter().enumerate() {
            let mapped_pos = d2p3 * (1.0 - interpolation3[i]) + d2p4 * interpolation3[i];
            map_to_2d.insert(v, mapped_pos);
        }
        // From p4 to p1
        let interpolation4 = arc_length_parameterization(
            &boundary4
                .iter()
                .map(|&v| layout.granulated_mesh.position(v))
                .collect_vec(),
        );
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

        let interior_verts = all_verts
            .iter()
            .filter(|&&v| !map_to_2d.contains_key(&v))
            .copied()
            .collect_vec();

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

            let neighbors = layout.granulated_mesh.neighbors(v0).collect_vec();
            let k = neighbors.len();

            let w = (0..k)
                .map(|i| {
                    let vip1 = neighbors[(i + 1) % k];
                    let eip1 = layout
                        .granulated_mesh
                        .edge_between_verts(v0, vip1)
                        .unwrap()
                        .0;
                    let vi = neighbors[i];
                    let ei = layout.granulated_mesh.edge_between_verts(v0, vi).unwrap().0;
                    let vim1 = neighbors[(i + k - 1) % k];
                    let eim1 = layout
                        .granulated_mesh
                        .edge_between_verts(v0, vim1)
                        .unwrap()
                        .0;

                    let alpha_im1 = layout.granulated_mesh.angle(eim1, ei);
                    let alpha_i = layout.granulated_mesh.angle(ei, eip1);
                    let len_ei = (layout.granulated_mesh.position(v0)
                        - layout.granulated_mesh.position(vi))
                    .norm();

                    ((alpha_im1 / 2.0).tan() + (alpha_i / 2.0).tan()) / len_ei
                })
                .collect_vec();

            // A degenerate one-ring (e.g., coincident vertices, of zero-area triangles) has no mean value weights:
            // then the vertex is the average of its neighbors (Tutte weights, always a valid embedding).
            let sum_w = w.iter().sum::<f64>();
            let weights = if w.iter().all(|wi| wi.is_finite()) && sum_w.is_finite() && sum_w > 0.0 {
                w.iter().map(|&wi| wi / sum_w).collect_vec()
            } else {
                debug!("Degenerate one-ring around {v0:?}: uniform weights");
                vec![1.0 / k as f64; k]
            };

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

        if !(bu.iter().chain(&bv).all(|v| v.is_finite())) {
            warn!("Right-hand side of the patch parameterization contains NaN or Inf");
            return None;
        }

        // Both coordinates share the same system matrix: factor it once (sparse LU) and solve
        // for u and v together. This replaces two unpreconditioned, unrestarted GMRES runs that
        // allocated O(n * 1000) dense memory each and could fail to converge on large patches.
        let mut x_uv = Mat::from_fn(n, 2, |i, j| if j == 0 { bu[i] } else { bv[i] });
        if !interior_verts.is_empty() {
            let faer_triplets = triplets
                .into_iter()
                .map(|(i, j, v)| Triplet::new(i, j, v))
                .collect::<Vec<_>>();
            let a = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &faer_triplets).ok()?;
            let Ok(lu) = a.sp_lu() else {
                warn!("Sparse LU factorization of the patch parameterization failed");
                return None;
            };
            lu.solve_in_place(x_uv.as_mut());
            if !(0..n).all(|i| x_uv[(i, 0)].is_finite() && x_uv[(i, 1)].is_finite()) {
                warn!("Patch parameterization produced non-finite coordinates");
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
                    x_uv[(vert_to_id.get_by_left(&v).unwrap().to_owned(), 0)],
                    x_uv[(vert_to_id.get_by_left(&v).unwrap().to_owned(), 1)],
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

        let grid_n = resolution[&edge1] + 1;
        let grid_m = resolution[&edge2] + 1;

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
            if edge_verts.len() != grid_n {
                warn!("Cannot construct quad mesh: the grids of neighboring faces do not match");
                return None;
            }
            for (j, &vert) in edge_verts.iter().enumerate() {
                vert_map[0][j] = Some(vert);
            }
            edge_to_verts_usize.insert(edge1, edge_verts.clone());
        }

        // Edge2 which is j=grid_n-1 and i=0 to i=grid_m-1
        if edges_done.contains_key(&edge2) {
            let edge_verts = edges_done.get(&edge2).unwrap().to_owned();
            if edge_verts.len() != grid_m {
                warn!("Cannot construct quad mesh: the grids of neighboring faces do not match");
                return None;
            }
            for (i, &vert) in edge_verts.iter().enumerate() {
                vert_map[i][grid_n - 1] = Some(vert);
            }
            edge_to_verts_usize.insert(edge2, edge_verts.clone());
        }

        // Edge3 which is i=grid_m-1 and j=grid_n-1 to j=0
        if edges_done.contains_key(&edge3) {
            let edge_verts = edges_done.get(&edge3).unwrap().to_owned();
            if edge_verts.len() != grid_n {
                warn!("Cannot construct quad mesh: the grids of neighboring faces do not match");
                return None;
            }
            for (j, &vert) in edge_verts.iter().enumerate() {
                vert_map[grid_m - 1][grid_n - 1 - j] = Some(vert);
            }
            edge_to_verts_usize.insert(edge3, edge_verts.clone());
        }

        // Edge4 which is j=0 and i=grid_m-1 to i=0
        if edges_done.contains_key(&edge4) {
            let edge_verts = edges_done.get(&edge4).unwrap().to_owned();
            if edge_verts.len() != grid_m {
                warn!("Cannot construct quad mesh: the grids of neighboring faces do not match");
                return None;
            }
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

        face_to_verts_usize.insert(
            patch_id,
            vert_map
                .iter()
                .map(|row| row.iter().filter_map(|&v| v).collect_vec())
                .collect_vec(),
        );

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

    if patches_done.len() != polycube.face_ids().len() {
        warn!("Cannot construct quad mesh: not all faces of the polycube are connected");
        return None;
    }

    // Create the polycube quad mesh:
    if let Ok((quad_mesh_polycube, vert_id_map, _)) = Mesh::<QUAD>::from(&faces, &vertex_positions)
    {
        for (face_id, vert_ids) in &face_to_verts_usize {
            // Convert usize to VertKey<QUAD>
            let vert_keys = vert_ids
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|&v| vert_id_map.key(v).unwrap().to_owned())
                        .collect_vec()
                })
                .collect_vec();
            face_to_verts.insert(face_id.to_owned(), vert_keys);
        }

        for (edge_id, vert_ids) in &edge_to_verts_usize {
            // Convert usize to VertKey<QUAD>
            let vert_keys = vert_ids
                .iter()
                .map(|&v| vert_id_map.key(v).unwrap().to_owned())
                .collect_vec();
            edge_to_verts.insert(edge_id.to_owned(), vert_keys.clone());
            let twin_id = polycube.twin(*edge_id);
            let rev_vert_keys = vert_keys.iter().rev().cloned().collect_vec();
            edge_to_verts.insert(twin_id, rev_vert_keys);
        }

        // A vertex is frozen if its faces do not all have the same label (it lies on a polycube edge or corner).
        for vert_id in quad_mesh_polycube.vert_ids() {
            let mut labels = quad_mesh_polycube
                .faces(vert_id)
                .map(|face| to_principal_direction(quad_mesh_polycube.normal(face)));
            if let Some(first) = labels.next()
                && labels.any(|label| label != first)
            {
                frozen.insert(vert_id);
            }
        }

        // Create the quad mesh
        // First, create copy of quad_mesh_polycube
        let mut quad_mesh = quad_mesh_polycube.clone();
        let triangle_lookup = triangle_mesh_polycube.bvh();

        // The position of every vertex on the input surface: the barycentric coordinates of the nearest point on the
        // polycube map, applied to the same triangle of the granulated mesh (in parallel).
        let mapped = quad_mesh
            .vert_ids()
            .into_par()
            .map(|vert_id| {
                let point = quad_mesh_polycube.position(vert_id);
                let triangle = triangle_lookup.nearest(&[point.x, point.y, point.z]);
                let [a, b, c] = triangle_mesh_polycube
                    .vertices(triangle)
                    .collect_array::<3>()?;
                let on_polycube = (
                    triangle_mesh_polycube.position(a),
                    triangle_mesh_polycube.position(b),
                    triangle_mesh_polycube.position(c),
                );
                let distance = geom::distance_to_triangle(point, on_polycube);
                // Barycentric coordinates of the closest point on the triangle (always within [0, 1], even if the BVH
                // returned a slightly-off triangle).
                let closest = geom::point_on_triangle(point, on_polycube);
                let (u, v, w) = geom::calculate_barycentric_coordinates(closest, on_polycube);
                let position = geom::inverse_barycentric_coordinates(
                    u,
                    v,
                    w,
                    (
                        layout.granulated_mesh.position(a),
                        layout.granulated_mesh.position(b),
                        layout.granulated_mesh.position(c),
                    ),
                );
                Some((vert_id, position, distance))
            })
            .collect::<Vec<_>>();
        let mut far = 0;
        for (vert_id, position, distance) in mapped.into_iter().flatten() {
            if distance > 0.001 {
                far += 1;
            }
            // Always map back: skipping a vertex would leave it in polycube coordinates while the rest of the quad
            // mesh lives in input-mesh coordinates.
            quad_mesh.set_position(vert_id, position);
        }
        if far > 0 {
            warn!("{far} quad vertices lie far (> 0.001) from the polycube map");
        }

        Some(Quad {
            triangle_mesh_polycube,
            quad_mesh_polycube,
            quad_mesh,
            face_to_verts,
            edge_to_verts,
            frozen,
        })
    } else {
        warn!("Failed to create quad mesh from faces and vertex positions");
        None
    }
}
