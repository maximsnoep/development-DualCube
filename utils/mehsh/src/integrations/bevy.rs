use crate::prelude::*;
use bevy_math::Vec3;
use std::collections::HashMap;

#[derive(Default)]
pub struct MeshBuilder {
    positions: Vec<Vec3>,
    normals: Vec<Vec3>,
    colors: Vec<[f32; 4]>,
    uvs: Vec<[f32; 2]>,
}

impl MeshBuilder {
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            positions: Vec::with_capacity(capacity),
            normals: Vec::with_capacity(capacity),
            colors: Vec::with_capacity(capacity),
            uvs: Vec::with_capacity(capacity),
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn add_vertex(&mut self, position: &Vector3D, normal: &Vector3D, color: &Color) {
        self.positions.push(to_bevy_vec(position));
        self.normals.push(to_bevy_vec(normal));
        self.colors.push(bevy_color::ColorToComponents::to_f32_array(
            bevy_color::Color::srgb(color[0], color[1], color[2]).to_linear(),
        ));
        self.uvs.push([0., 0.]);
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn normalize(&mut self, scale: f64, translation: Vector3D) {
        for position in &mut self.positions {
            *position = Vec3::new(
                position.x.mul_add(scale as f32, translation.x as f32),
                position.y.mul_add(scale as f32, translation.y as f32),
                position.z.mul_add(scale as f32, translation.z as f32),
            );
        }
    }

    #[must_use]
    pub fn build(self) -> bevy_render::mesh::Mesh {
        bevy_render::mesh::Mesh::new(
            bevy_render::mesh::PrimitiveTopology::TriangleList,
            bevy_render::render_asset::RenderAssetUsages::RENDER_WORLD | bevy_render::render_asset::RenderAssetUsages::MAIN_WORLD,
        )
        .with_inserted_indices(bevy_render::mesh::Indices::U32((0..u32::try_from(self.positions.len()).unwrap()).collect()))
        .with_inserted_attribute(bevy_render::mesh::Mesh::ATTRIBUTE_POSITION, self.positions)
        .with_inserted_attribute(bevy_render::mesh::Mesh::ATTRIBUTE_NORMAL, self.normals)
        .with_inserted_attribute(bevy_render::mesh::Mesh::ATTRIBUTE_COLOR, self.colors)
        .with_inserted_attribute(bevy_render::mesh::Mesh::ATTRIBUTE_UV_0, self.uvs)
    }
}

/// Construct a Bevy mesh object (one that can be rendered using Bevy).
/// Requires a `color_map` to assign colors to faces. If no color is assigned to a face, it will default to black.
impl<M: Tag> Mesh<M>
where
    M: std::default::Default + std::cmp::Eq + std::hash::Hash + Copy + Clone,
{
    #[must_use]
    pub fn bevy(&self, color_map: &HashMap<FaceKey<M>, [f32; 3]>) -> (bevy_render::mesh::Mesh, Vector3D, f64) {
        if self.faces.is_empty() {
            return (MeshBuilder::with_capacity(0).build(), Vector3D::new(0., 0., 0.), 1.);
        }

        let k = self.vertices(self.faces.ids().next().unwrap()).len();

        let mut bevy_mesh_builder = MeshBuilder::with_capacity(self.faces.len() * (k - 2) * 3);
        let (triangle_mesh, face_map) = self.triangulate().unwrap();

        for face_id in triangle_mesh.faces.ids() {
            let corners = triangle_mesh.vertices(face_id);
            let original_face = match face_map.get(&face_id) {
                Some(&original_face) => original_face,
                None => face_id,
            };
            let triangle = [corners[0], corners[1], corners[2]];
            for vertex_id in triangle {
                bevy_mesh_builder.add_vertex(
                    &triangle_mesh.position(vertex_id),
                    &triangle_mesh.normal(vertex_id),
                    color_map.get(&original_face).unwrap_or(&[0., 0., 0.]),
                );
            }
        }

        let (scale, translation) = self.scale_translation();
        bevy_mesh_builder.normalize(scale, translation);

        (bevy_mesh_builder.build(), translation, scale)
    }

    // Construct a Bevy gizmos object of the wireframe (one that can be rendered using Bevy)
    #[must_use]
    pub fn gizmos(&self, color: [f32; 3]) -> bevy_gizmos::GizmoAsset {
        let mut gizmo = bevy_gizmos::GizmoAsset::new();
        let (scale, translation) = self.scale_translation();

        for &(u, v) in &self.edges_positions() {
            let ut = u * scale + translation;
            let vt = v * scale + translation;
            gizmo.line(to_bevy_vec(&ut), to_bevy_vec(&vt), bevy_color::Color::srgb(color[0], color[1], color[2]));
        }

        gizmo
    }

    #[must_use]
    pub fn scale_translation(&self) -> (f64, Vector3D) {
        let scale = 20. * (1. / self.max_dim());
        let center = self.center();
        (scale, -scale * center)
    }
}

#[allow(clippy::cast_possible_truncation)]
fn to_bevy_vec(vec: &Vector3D) -> Vec3 {
    Vec3::new(vec.x as f32, vec.y as f32, vec.z as f32)
}
