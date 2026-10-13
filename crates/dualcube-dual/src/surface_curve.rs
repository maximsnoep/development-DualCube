//! Piecewise linear curves on immutable input triangles.

use crate::{Loop, PropertyViolationError};
use dualcube_types::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(thiserror::Error, Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SurfaceCurveError {
    #[error("point outside triangle")]
    PointOutsideTriangle,
    #[error("curve through mesh vertex")]
    MeshVertex,
    #[error("zero segment")]
    ZeroSegment,
    #[error("disconnected segments")]
    DisconnectedSegments,
    #[error("interior face jump")]
    InteriorFaceJump,
    #[error("disconnected face transition")]
    DisconnectedTransition,
    #[error("overlapping adjacent segments")]
    OverlappingAdjacentSegments,
    #[error("touching or overlapping curves")]
    TouchingCurves,
    #[error("self intersection")]
    SelfIntersection,
    #[error("triple intersection")]
    TripleIntersection,
    #[error("collapsed cell edge")]
    CollapsedCellEdge,
    #[error("inconsistent cell barrier")]
    InconsistentBarrier,
    #[error("nonconforming cell boundary")]
    NonconformingBoundary,
}

/// Coordinates `(u, v)` mean barycentric weights `(1-u-v, u, v)` in face vertex order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurfaceArc {
    pub face: FaceID,
    pub from: [f64; 2],
    pub to: [f64; 2],
}

impl SurfaceArc {
    /// Map triangle coordinates to the immutable input surface.
    pub fn position(mesh: &Mesh<INPUT>, face: FaceID, p: [f64; 2]) -> Vector3D {
        let corners: Vec<_> = mesh.vertices(face).map(|v| mesh.position(v)).collect();
        corners[0] * (1. - p[0] - p[1]) + corners[1] * p[0] + corners[2] * p[1]
    }
}

impl Loop {
    pub fn from_surface(direction: Direction, surface: Vec<SurfaceArc>) -> Self {
        Self {
            direction,
            surface,
            edges: vec![],
            offsets: vec![],
        }
    }

    /// Every segment in traversal order, including repeated triangle and edge visits.
    pub fn surface_arcs(
        &self,
        mesh: &Mesh<INPUT>,
    ) -> Result<Vec<SurfaceArc>, PropertyViolationError> {
        if !self.surface.is_empty() {
            return Ok(self.surface.clone());
        }
        let mut arcs = Vec::new();
        let len = self.edges.len();
        for i in 0..len {
            let j = (i + 1) % len;
            let a = self.edges[i];
            let b = self.edges[j];
            if mesh.twin(a) == b {
                continue;
            }
            let face = mesh.face(a);
            if mesh.face(b) != face {
                return Err(PropertyViolationError::UnknownError);
            }
            let sides: Vec<_> = mesh.edges(face).collect();
            if sides.len() != 3 {
                return Err(PropertyViolationError::UnknownError);
            }
            let point = |edge, t: f64| {
                let corners = [[0., 0.], [1., 0.], [0., 1.]];
                let side = sides.iter().position(|&e| e == edge).unwrap();
                let [a, b] = [corners[side], corners[(side + 1) % 3]];
                [a[0] * (1. - t) + b[0] * t, a[1] * (1. - t) + b[1] * t]
            };
            arcs.push(SurfaceArc {
                face,
                from: point(a, self.offsets.get(i).copied().unwrap_or(0.5)),
                to: point(b, self.offsets.get(j).copied().unwrap_or(0.5)),
            });
        }
        if arcs.is_empty() {
            return Err(PropertyViolationError::PathEmpty);
        }
        Ok(arcs)
    }

    pub fn reverse(&mut self) {
        self.edges.reverse();
        self.offsets.reverse();
        self.surface.reverse();
        for arc in &mut self.surface {
            std::mem::swap(&mut arc.from, &mut arc.to);
        }
    }
}

#[cfg(test)]
mod tests;
