use mehsh::prelude::*;
use orx_parallel::*;

#[derive(Default)]
pub struct MeshBuilder {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    colors: Vec<[f32; 4]>,
    uvs: Vec<[f32; 2]>,
}

impl MeshBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self {
            positions: Vec::new(),
            normals: Vec::new(),
            colors: Vec::new(),
            uvs: Vec::new(),
        }
    }

    /// Adds a triangle (in the coordinates of a mesh, see `normalize`) with vertex normals and a flat color (sRGB).
    pub fn add_triangle(&mut self, corners: [Vector3D; 3], normals: [Vector3D; 3], color: [f32; 3]) {
        let color = srgb_to_linear(color);
        for (corner, normal) in corners.into_iter().zip(normals) {
            self.positions.push(v3d_to_slice(corner));
            self.normals.push(v3d_to_slice(normal));
            self.colors.push(color);
            self.uvs.push([0., 0.]);
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn normalize(&mut self, scale: f64, translation: Vector3D) {
        for position in &mut self.positions {
            *position = [
                position[0].mul_add(scale as f32, translation.x as f32),
                position[1].mul_add(scale as f32, translation.y as f32),
                position[2].mul_add(scale as f32, translation.z as f32),
            ];
        }
    }

    #[must_use]
    pub fn build(self) -> bevy::mesh::Mesh {
        bevy::mesh::Mesh::new(
            bevy::mesh::PrimitiveTopology::TriangleList,
            bevy::asset::RenderAssetUsages::RENDER_WORLD
                | bevy::asset::RenderAssetUsages::MAIN_WORLD,
        )
        .with_inserted_indices(bevy::mesh::Indices::U32(
            (0..self.positions.len() as u32).collect::<Vec<_>>(),
        ))
        .with_inserted_attribute(bevy::mesh::Mesh::ATTRIBUTE_POSITION, self.positions)
        .with_inserted_attribute(bevy::mesh::Mesh::ATTRIBUTE_NORMAL, self.normals)
        .with_inserted_attribute(bevy::mesh::Mesh::ATTRIBUTE_COLOR, self.colors)
        .with_inserted_attribute(bevy::mesh::Mesh::ATTRIBUTE_UV_0, self.uvs)
    }
}

/// Construct a Bevy mesh object (one that can be rendered using Bevy).
/// Requires a `color_map` to assign colors to faces. If no color is assigned to a face, it will default to black.
#[must_use]
pub fn to_bevy<M: Tag>(
    mesh: &Mesh<M>,
    color_map: &HashMap<FaceKey<M>, [f32; 3]>,
) -> (bevy::mesh::Mesh, Vector3D, f64) {
    if mesh.faces.is_empty() {
        return (MeshBuilder::new().build(), Vector3D::new(0., 0., 0.), 1.);
    }
    let mut bevy_mesh_builder = bevy_builder(mesh, color_map);
    let (scale, translation) = mesh.scale_translation();
    bevy_mesh_builder.normalize(scale, translation);
    (bevy_mesh_builder.build(), translation, scale)
}

fn bevy_builder<M: Tag>(mesh: &Mesh<M>, color_map: &HashMap<FaceKey<M>, [f32; 3]>) -> MeshBuilder {
    // Compute every vertex normal exactly once, instead of re-walking the one-ring (and
    // recomputing all incident face normals) for every face corner.
    let vert_normals: HashMap<VertKey<M>, [f32; 3]> = mesh
        .vert_ids()
        .par()
        .map(|&v| (v, v3d_to_slice(mesh.normal(v))))
        .collect::<Vec<_>>()
        .into_iter()
        .collect();

    let triangulated_faces = mesh
        .face_ids()
        .par()
        .map(|&id| {
            let verts = mesh.vertices(id).collect_vec();
            let c = srgb_to_linear(color_map.get(&id).copied().unwrap_or([0., 0., 0.]));
            // Triangles as-is; quads split along v1-v3 (as before); larger polygons fan from v0.
            let corners: Vec<VertKey<M>> = match verts.as_slice() {
                &[v0, v1, v2] => vec![v0, v1, v2],
                &[v0, v1, v2, v3] => vec![v3, v0, v1, v1, v2, v3],
                vs if vs.len() > 4 => (1..vs.len() - 1)
                    .flat_map(|i| [vs[0], vs[i], vs[i + 1]])
                    .collect(),
                _ => vec![],
            };
            let p = corners
                .iter()
                .map(|&v| v3d_to_slice(mesh.position(v)))
                .collect_vec();
            let n = corners.iter().map(|v| vert_normals[v]).collect_vec();
            let c = vec![c; corners.len()];
            (p, n, c)
        })
        .collect::<Vec<_>>();

    let total_len: usize = triangulated_faces.iter().map(|(p, _, _)| p.len()).sum();
    info!("Building mesh for Bevy with {total_len} vertices.",);

    let mut positions = Vec::with_capacity(total_len);
    let mut normals = Vec::with_capacity(total_len);
    let mut colors = Vec::with_capacity(total_len);

    for (p, n, c) in triangulated_faces {
        positions.extend(p);
        normals.extend(n);
        colors.extend(c);
    }
    let uvs = vec![[0., 0.]; positions.len()];

    MeshBuilder {
        positions,
        normals,
        colors,
        uvs,
    }
}

// Construct a Bevy gizmos object of the wireframe (one that can be rendered using Bevy)
#[must_use]
pub fn gizmos<M: Tag>(mesh: &Mesh<M>, color: [f32; 3]) -> bevy::gizmos::GizmoAsset {
    let mut gizmo = bevy::gizmos::GizmoAsset::new();
    let (scale, translation) = mesh.scale_translation();
    // Half-edges come in twin pairs; draw each undirected edge once.
    for e in mesh.edge_ids_iter().filter(|&e| e < mesh.twin(e)) {
        let u = mesh.position(mesh.root(e));
        let v = mesh.position(mesh.toor(e));
        gizmo.line(
            v3d_to_bevy(&(u * scale + translation)),
            v3d_to_bevy(&(v * scale + translation)),
            srgb_to_bevy(color),
        );
    }
    gizmo
}

fn v3d_to_slice(vec: Vector3D) -> [f32; 3] {
    [vec.x as f32, vec.y as f32, vec.z as f32]
}

fn srgb_to_linear(color: [f32; 3]) -> [f32; 4] {
    bevy::color::ColorToComponents::to_f32_array(
        bevy::color::Color::srgb_from_array(color).to_linear(),
    )
}

fn v3d_to_bevy(vec: &Vector3D) -> bevy::math::Vec3 {
    bevy::math::Vec3::new(vec.x as f32, vec.y as f32, vec.z as f32)
}

fn srgb_to_bevy(color: [f32; 3]) -> bevy::color::Color {
    bevy::color::Color::srgb_from_array(color)
}
