use nalgebra::DMatrix;

use crate::utils::primitives::{EPS, Vector2D, Vector3D};

/// Represents the orientation of three points in 3D space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    C,   // Collinear
    CW,  // Clockwise
    CCW, // Counterclockwise
}

/// Represents the type of intersection between line segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntersectionType {
    Proper,
    Endpoint,
}

#[must_use]
pub fn calculate_triangle_area(t: (Vector3D, Vector3D, Vector3D)) -> f64 {
    (t.1 - t.0).cross(&(t.2 - t.0)).magnitude() * 0.5
}

#[must_use]
pub fn are_points_coplanar(a: Vector3D, b: Vector3D, c: Vector3D, d: Vector3D) -> bool {
    (b - a).cross(&(c - a)).dot(&(d - a)) == 0.
}

#[must_use]
pub fn calculate_orientation(a: Vector3D, b: Vector3D, c: Vector3D, n: Vector3D) -> Orientation {
    let orientation = (b - a).cross(&(c - a)).dot(&n);
    if orientation > 0. {
        Orientation::CCW
    } else if orientation < 0. {
        Orientation::CW
    } else {
        Orientation::C
    }
}

#[must_use]
pub fn calculate_clockwise_angle(a: Vector3D, b: Vector3D, c: Vector3D, n: Vector3D) -> f64 {
    let ab = (b - a).normalize();
    let ac = (c - a).normalize();
    let angle = ab.angle(&ac);
    if calculate_orientation(a, b, c, n) == Orientation::CCW {
        2.0f64.mul_add(std::f64::consts::PI, -angle)
    } else {
        angle
    }
}

#[must_use]
pub fn project_point_onto_plane(point: Vector3D, plane: (Vector3D, Vector3D), reference: Vector3D) -> Vector2D {
    Vector2D::new((point - reference).dot(&plane.0), (point - reference).dot(&plane.1))
}

#[must_use]
pub fn is_point_inside_triangle(p: Vector3D, t: (Vector3D, Vector3D, Vector3D)) -> bool {
    let s1 = calculate_triangle_area((t.0, t.1, p));
    let s2 = calculate_triangle_area((t.1, t.2, p));
    let s3 = calculate_triangle_area((t.2, t.0, p));
    let st = calculate_triangle_area(t);
    (s1 + s2 + s3 - st).abs() < EPS && (0.0 - EPS..=st + EPS).contains(&s1) && (0.0 - EPS..=st + EPS).contains(&s2) && (0.0 - EPS..=st + EPS).contains(&s3)
}

#[must_use]
pub fn is_within_inclusive_range(a: f64, b: f64, c: f64) -> bool {
    if b < c { (b..=c).contains(&a) } else { (c..=b).contains(&a) }
}

#[must_use]
pub fn calculate_2d_lineseg_intersection(p_u: Vector2D, p_v: Vector2D, q_u: Vector2D, q_v: Vector2D) -> Option<(Vector2D, IntersectionType)> {
    let (x1, x2, x3, x4, y1, y2, y3, y4) = (p_u.x, p_v.x, q_u.x, q_v.x, p_u.y, p_v.y, q_u.y, q_v.y);

    let t_numerator = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4);
    let u_numerator = (x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2);
    let denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);

    if denominator.abs() < EPS {
        return None;
    }

    if is_within_inclusive_range(t_numerator, 0.0, denominator) {
        let t = t_numerator / denominator;
        if t.abs() < EPS {
            return Some((p_u, IntersectionType::Endpoint));
        }
        if (t - 1.0).abs() < EPS {
            return Some((p_v, IntersectionType::Endpoint));
        }
        let sx_t = t.mul_add(x2 - x1, x1);
        let sy_t = t.mul_add(y2 - y1, y1);
        let s_t = Vector2D::new(sx_t, sy_t);

        Some((s_t, IntersectionType::Proper))
    } else if is_within_inclusive_range(u_numerator, 0.0, denominator) {
        let u = u_numerator / denominator;
        if u.abs() < EPS {
            return Some((q_u, IntersectionType::Endpoint));
        }
        if (u - 1.0).abs() < EPS {
            return Some((q_v, IntersectionType::Endpoint));
        }
        let sx_u = u.mul_add(x4 - x3, x3);
        let sy_u = u.mul_add(y4 - y3, y3);
        let s_u = Vector2D::new(sx_u, sy_u);

        Some((s_u, IntersectionType::Proper))
    } else {
        None
    }
}

#[must_use]
pub fn calculate_3d_lineseg_intersection(p_u: Vector3D, p_v: Vector3D, q_u: Vector3D, q_v: Vector3D) -> Option<(Vector3D, IntersectionType)> {
    if !are_points_coplanar(p_u, p_v, q_u, q_v) {
        return None;
    }

    let p = p_v - p_u;
    let q = q_v - q_u;
    let normal_vector = p.cross(&q).normalize();
    let reference_point = p_u;
    let plane = (p.normalize(), p.cross(&normal_vector).normalize());

    calculate_2d_lineseg_intersection(
        project_point_onto_plane(p_u, plane, reference_point),
        project_point_onto_plane(p_v, plane, reference_point),
        project_point_onto_plane(q_u, plane, reference_point),
        project_point_onto_plane(q_v, plane, reference_point),
    )
    .map(|(point_in_2d, intersection_type)| {
        let point_in_3d = reference_point + (plane.0 * point_in_2d.x) + (plane.1 * point_in_2d.y);
        (point_in_3d, intersection_type)
    })
}

/// Calculates the closest point on triangle `t`, given a point `p`
#[must_use]
pub fn point_on_triangle(p: Vector3D, t: (Vector3D, Vector3D, Vector3D)) -> Vector3D {
    let (a, b, c) = t;

    let ab = b - a;
    let ac = c - a;
    let ap = p - a;

    // Normal vector of the triangle plane
    let n = ab.cross(&ac);
    let n_norm = n.norm();
    let n_unit = n / n_norm;

    // Distance from p to the plane
    let dist_to_plane = ap.dot(&n_unit);
    let proj = p - dist_to_plane * n_unit;

    if n_norm != 0.0 && is_point_inside_triangle(proj, (a, b, c)) {
        proj // Perpendicular projection onto the triangle plane
    } else {
        // Closest point to one of the triangle’s edges
        let d1 = distance_to_segment(p, a, b);
        let d2 = distance_to_segment(p, b, c);
        let d3 = distance_to_segment(p, c, a);
        if d1 <= d2 && d1 <= d3 {
            point_on_edge(p, a, b)
        } else if d2 <= d1 && d2 <= d3 {
            point_on_edge(p, b, c)
        } else {
            point_on_edge(p, c, a)
        }
    }
}

/// Calculates the closest point on the edge defined by points `a` and `b` to point `p`
#[must_use]
pub fn point_on_edge(p: Vector3D, a: Vector3D, b: Vector3D) -> Vector3D {
    let ab = b - a;
    let t = (p - a).dot(&ab) / ab.dot(&ab);
    let t_clamped = t.clamp(0.0, 1.0);
    a + ab * t_clamped
}

/// Calculates the distance of point `p` to triangle `t`
#[must_use]
pub fn distance_to_triangle(p: Vector3D, t: (Vector3D, Vector3D, Vector3D)) -> f64 {
    let closest_point = point_on_triangle(p, t);
    (p - closest_point).norm()
}

fn distance_to_segment(p: Vector3D, a: Vector3D, b: Vector3D) -> f64 {
    let closest = point_on_edge(p, a, b);
    (p - closest).norm()
}

// Calculate the barycentric coordinates of point `p` with respect to triangle `t`.
#[must_use]
#[inline]
pub fn calculate_barycentric_coordinates(p: Vector3D, t: (Vector3D, Vector3D, Vector3D)) -> (f64, f64, f64) {
    let (a, b, c) = t;
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;

    let d00 = v0.dot(&v0);
    let d01 = v0.dot(&v1);
    let d11 = v1.dot(&v1);
    let d20 = v2.dot(&v0);
    let d21 = v2.dot(&v1);

    let denom = d00 * d11 - d01 * d01;

    if denom.abs() < 1e-12 {
        // Degenerate case: the triangle is a line or a point
        // We'll fall back to 1D parameterization along the longest edge

        // Try AB first
        let ab = b - a;
        let ab_len2 = ab.dot(&ab);
        if ab_len2 > 1e-12 {
            let t = (p - a).dot(&ab) / ab_len2;
            return (1.0 - t, t, 0.0);
        }

        // Try BC
        let bc = c - b;
        let bc_len2 = bc.dot(&bc);
        if bc_len2 > 1e-12 {
            let t = (p - b).dot(&bc) / bc_len2;
            return (0.0, 1.0 - t, t);
        }

        // All points are the same
        return (1.0, 0.0, 0.0);
    }

    let inv_denom = 1.0 / denom;
    let v = (d11 * d20 - d01 * d21) * inv_denom;
    let w = (d00 * d21 - d01 * d20) * inv_denom;
    let u = 1.0 - v - w;

    (u, v, w)
}

// Inverse barycentric coordinates: given barycentric coordinates (u, v, w), find the point p in triangle t.
#[must_use]
#[inline]
pub fn inverse_barycentric_coordinates(u: f64, v: f64, w: f64, t: (Vector3D, Vector3D, Vector3D)) -> Vector3D {
    let p1 = t.0 * u;
    let p2 = t.1 * v;
    let p3 = t.2 * w;
    p1 + p2 + p3
}

// PCA to fit plane
pub fn fit_plane(points: &[Vector3D]) -> (Vector3D, f64) {
    let n = points.len();

    // 1. Compute centroid
    let centroid = points.iter().cloned().sum::<Vector3D>() / n as f64;

    // 2. Build covariance matrix
    let mut covariance = DMatrix::zeros(3, 3);
    for p in points {
        let diff = p - centroid;
        covariance += diff * diff.transpose();
    }
    covariance /= n as f64;

    // 3. Eigen decomposition
    let eigen = covariance.symmetric_eigen();
    let idx = eigen.eigenvalues.imin();
    let eigenvector = eigen.eigenvectors.column(idx);
    let rms = eigen.eigenvalues[idx].sqrt();
    let diameter = diameter(points);
    (Vector3D::new(eigenvector[0], eigenvector[1], eigenvector[2]), rms / diameter)
}

// Diameter of a set of points (max distance between two points)
pub fn diameter(points: &[Vector3D]) -> f64 {
    let mut max_dist = 0.0;
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let dist = (points[i] - points[j]).norm();
            if dist > max_dist {
                max_dist = dist;
            }
        }
    }
    max_dist
}

/// Project a 3D triangle into its local 2D coordinates
/// Returns a tuple (q0, q1, q2) in 2D
pub fn triangle_to_2d(p0: Vector3D, p1: Vector3D, p2: Vector3D) -> Option<(Vector2D, Vector2D, Vector2D)> {
    // 1. Compute edges
    let v1 = p1 - p0;
    let v2 = p2 - p0;

    // 3. Construct orthonormal basis (u_x, u_y, u_z)
    let ux = v1.normalize();
    let uz = v1.cross(&v2).normalize();
    let uy = uz.cross(&ux); // guaranteed orthonormal

    // 4. Project each vertex into 2D plane coordinates
    let q0 = Vector2D::new(p0.dot(&ux), p0.dot(&uy));
    let q1 = Vector2D::new(p1.dot(&ux), p1.dot(&uy));
    let q2 = Vector2D::new(p2.dot(&ux), p2.dot(&uy));

    Some((q0, q1, q2))
}
