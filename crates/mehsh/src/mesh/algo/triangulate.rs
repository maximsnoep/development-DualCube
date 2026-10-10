use crate::prelude::*;
use std::collections::HashMap;

const TRIANGULATION_EPS: f64 = 1.0e-10;

// Given a polygonal mesh, triangulate all polygonal faces to obtain a triangular mesh.
// This is useful before using algorithms that assume every face is a triangle.
impl<M: Tag> Mesh<M> {
    pub fn triangulate(&self) -> Result<(Self, HashMap<FaceKey<M>, FaceKey<M>>), MeshError<M>> {
        let (mesh, face_sources) = self.triangulate_selected(None)?;

        info!(
            "Triangulated {} faces into {} triangular faces",
            self.faces.len(),
            mesh.faces.len()
        );

        Ok((mesh, face_sources))
    }

    // Rebuild the mesh with `face` triangulated. Other faces keep their current polygonal shape.
    // The rebuild keeps this operation transactional: if triangulation fails, `self` is untouched.
    pub fn triangulate_face(&mut self, face: FaceKey<M>) -> Result<Vec<FaceKey<M>>, MeshError<M>> {
        if !self.faces.contains(face) {
            return Err(MeshError::Unknown(format!(
                "Cannot triangulate non-existing face {face:?}"
            )));
        }

        if self.vertices(face).count() == 3 {
            return Ok(vec![face]);
        }

        let (new_mesh, face_sources) = self.triangulate_selected(Some(face))?;
        let new_faces = face_sources
            .iter()
            .filter_map(|(&new_face, &source_face)| (source_face == face).then_some(new_face))
            .collect_vec();

        *self = new_mesh;
        Ok(new_faces)
    }

    fn triangulate_selected(
        &self,
        selected_face: Option<FaceKey<M>>,
    ) -> Result<(Self, HashMap<FaceKey<M>, FaceKey<M>>), MeshError<M>> {
        let old_verts = self.vert_ids();
        let positions = old_verts
            .iter()
            .map(|&vert| self.position(vert))
            .collect_vec();
        let old_vert_to_index = old_verts
            .into_iter()
            .enumerate()
            .map(|(index, vert)| (vert, index))
            .collect::<HashMap<_, _>>();

        let mut new_faces = Vec::<Vec<usize>>::new();
        let mut new_face_sources = Vec::<FaceKey<M>>::new();

        for face in self.face_ids() {
            let face_verts = self.vertices(face).collect_vec();
            let should_triangulate =
                selected_face.map_or(face_verts.len() > 3, |selected| selected == face);
            let pieces = if should_triangulate {
                self.triangulate_face_vertices(face, &face_verts)?
            } else {
                vec![face_verts]
            };

            for piece in pieces {
                new_faces.push(
                    piece
                        .into_iter()
                        .map(|vert| old_vert_to_index[&vert])
                        .collect_vec(),
                );
                new_face_sources.push(face);
            }
        }

        let (mesh, _, face_map) = Self::from(&new_faces, &positions)?;
        let face_sources = new_face_sources
            .into_iter()
            .enumerate()
            .filter_map(|(index, source_face)| {
                face_map
                    .key(index)
                    .copied()
                    .map(|new_face| (new_face, source_face))
            })
            .collect::<HashMap<_, _>>();

        Ok((mesh, face_sources))
    }

    fn triangulate_face_vertices(
        &self,
        face: FaceKey<M>,
        verts: &[VertKey<M>],
    ) -> Result<Vec<Vec<VertKey<M>>>, MeshError<M>> {
        match verts.len() {
            0..=2 => return Err(MeshError::FaceNotPolygon(face)),
            3 => return Ok(vec![verts.to_vec()]),
            _ => {}
        }

        let points = verts.iter().map(|&vert| self.position(vert)).collect_vec();
        let projected = project_polygon_to_2d(&points).ok_or(MeshError::FaceNotPlanar(face))?;
        let area = signed_area(&projected);
        if area.abs() < TRIANGULATION_EPS {
            return Err(MeshError::FaceNotPlanar(face));
        }

        let orientation = area.signum();
        let mut remaining = (0..verts.len()).collect_vec();
        let mut triangles = Vec::with_capacity(verts.len() - 2);

        while remaining.len() > 3 {
            let mut ear_index = None;

            for i in 0..remaining.len() {
                let prev = remaining[(i + remaining.len() - 1) % remaining.len()];
                let curr = remaining[i];
                let next = remaining[(i + 1) % remaining.len()];

                if !is_convex(
                    projected[prev],
                    projected[curr],
                    projected[next],
                    orientation,
                ) {
                    continue;
                }

                if contains_other_vertex(&projected, &remaining, [prev, curr, next]) {
                    continue;
                }

                ear_index = Some(i);
                triangles.push(vec![verts[prev], verts[curr], verts[next]]);
                break;
            }

            let Some(i) = ear_index else {
                return Err(MeshError::FaceNotSimple(face));
            };
            remaining.remove(i);
        }

        triangles.push(remaining.iter().map(|&index| verts[index]).collect_vec());
        Ok(triangles)
    }
}

fn project_polygon_to_2d(points: &[Vector3D]) -> Option<Vec<Vector2D>> {
    let normal = newell_normal(points)?;
    let origin = points[0];
    let x_axis = points
        .windows(2)
        .map(|edge| edge[1] - edge[0])
        .chain(std::iter::once(points[0] - points[points.len() - 1]))
        .find(|edge| edge.norm() > TRIANGULATION_EPS)?
        .normalize();
    let y_axis = x_axis.cross(&normal).normalize();

    Some(
        points
            .iter()
            .map(|&point| geom::project_point_onto_plane(point, (x_axis, y_axis), origin))
            .collect_vec(),
    )
}

fn newell_normal(points: &[Vector3D]) -> Option<Vector3D> {
    let mut normal = Vector3D::zeros();
    for i in 0..points.len() {
        let current = points[i];
        let next = points[(i + 1) % points.len()];
        normal.x += (current.y - next.y) * (current.z + next.z);
        normal.y += (current.z - next.z) * (current.x + next.x);
        normal.z += (current.x - next.x) * (current.y + next.y);
    }

    (normal.norm() > TRIANGULATION_EPS).then(|| normal.normalize())
}

fn signed_area(points: &[Vector2D]) -> f64 {
    points
        .iter()
        .zip(points.iter().cycle().skip(1))
        .map(|(a, b)| a.x * b.y - b.x * a.y)
        .sum::<f64>()
        * 0.5
}

fn is_convex(prev: Vector2D, curr: Vector2D, next: Vector2D, orientation: f64) -> bool {
    cross_2d(curr - prev, next - curr) * orientation > TRIANGULATION_EPS
}

fn contains_other_vertex(points: &[Vector2D], remaining: &[usize], triangle: [usize; 3]) -> bool {
    remaining.iter().copied().any(|index| {
        !triangle.contains(&index)
            && point_in_triangle(
                points[index],
                points[triangle[0]],
                points[triangle[1]],
                points[triangle[2]],
            )
    })
}

fn point_in_triangle(point: Vector2D, a: Vector2D, b: Vector2D, c: Vector2D) -> bool {
    let coord = |v: Vector2D| robust::Coord { x: v.x, y: v.y };
    let o1 = robust::orient2d(coord(a), coord(b), coord(point));
    let o2 = robust::orient2d(coord(b), coord(c), coord(point));
    let o3 = robust::orient2d(coord(c), coord(a), coord(point));

    (o1 >= 0.0 && o2 >= 0.0 && o3 >= 0.0) || (o1 <= 0.0 && o2 <= 0.0 && o3 <= 0.0)
}

fn cross_2d(a: Vector2D, b: Vector2D) -> f64 {
    a.x * b.y - a.y * b.x
}
