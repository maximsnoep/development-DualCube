use crate::utils::primitives::{EPS, Vector2D, Vector3D};
use nalgebra::DMatrix;

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
    let coord = |v: Vector3D| robust::Coord3D {
        x: v.x,
        y: v.y,
        z: v.z,
    };
    robust::orient3d(coord(a), coord(b), coord(c), coord(d)) == 0.0
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
pub fn project_point_onto_plane(
    point: Vector3D,
    plane: (Vector3D, Vector3D),
    reference: Vector3D,
) -> Vector2D {
    Vector2D::new(
        (point - reference).dot(&plane.0),
        (point - reference).dot(&plane.1),
    )
}

#[must_use]
pub fn is_point_inside_triangle(p: Vector3D, t: (Vector3D, Vector3D, Vector3D)) -> bool {
    let s1 = calculate_triangle_area((t.0, t.1, p));
    let s2 = calculate_triangle_area((t.1, t.2, p));
    let s3 = calculate_triangle_area((t.2, t.0, p));
    let st = calculate_triangle_area(t);
    // Tolerance relative to the triangle's size: an absolute epsilon on areas fails both for
    // large coordinates (rounding error exceeds it) and for tiny triangles (it exceeds the area).
    let tol = 1e-9 * st.max(EPS);
    (s1 + s2 + s3 - st).abs() <= tol
        && (-tol..=st + tol).contains(&s1)
        && (-tol..=st + tol).contains(&s2)
        && (-tol..=st + tol).contains(&s3)
}

#[must_use]
pub fn calculate_2d_lineseg_intersection(
    p_u: Vector2D,
    p_v: Vector2D,
    q_u: Vector2D,
    q_v: Vector2D,
) -> Option<(Vector2D, IntersectionType)> {
    let (x1, x2, x3, x4, y1, y2, y3, y4) = (p_u.x, p_v.x, q_u.x, q_v.x, p_u.y, p_v.y, q_u.y, q_v.y);

    let t_numerator = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4);
    let u_numerator = (x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2);
    let denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);

    // `t` and `u` are dimensionless, but the denominator scales with |p| * |q|: test parallelism relative to that.
    const INTERSECTION_EPS: f64 = 1.0e-12;
    let scale = (p_v - p_u).norm() * (q_v - q_u).norm();
    if denominator.abs() <= INTERSECTION_EPS * scale {
        return None;
    }

    let t = t_numerator / denominator;
    let u = u_numerator / denominator;
    if !(-INTERSECTION_EPS..=1.0 + INTERSECTION_EPS).contains(&t)
        || !(-INTERSECTION_EPS..=1.0 + INTERSECTION_EPS).contains(&u)
    {
        return None;
    }

    let point = Vector2D::new(t.mul_add(x2 - x1, x1), t.mul_add(y2 - y1, y1));
    let intersection_type = if t.abs() < INTERSECTION_EPS
        || (t - 1.0).abs() < INTERSECTION_EPS
        || u.abs() < INTERSECTION_EPS
        || (u - 1.0).abs() < INTERSECTION_EPS
    {
        IntersectionType::Endpoint
    } else {
        IntersectionType::Proper
    };

    Some((point, intersection_type))
}

#[must_use]
pub fn calculate_3d_lineseg_intersection(
    p_u: Vector3D,
    p_v: Vector3D,
    q_u: Vector3D,
    q_v: Vector3D,
) -> Option<(Vector3D, IntersectionType)> {
    if !are_points_coplanar(p_u, p_v, q_u, q_v) {
        return None;
    }

    let p = p_v - p_u;
    let q = q_v - q_u;
    if p.norm() < 1.0e-12 || q.norm() < 1.0e-12 {
        return None;
    }
    let normal = p.cross(&q);
    if normal.norm() <= 1.0e-12 * p.norm() * q.norm() {
        return None;
    }
    let normal_vector = normal.normalize();
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
    // Ericson, "Real-Time Collision Detection", 5.1.5: classify `p` against the Voronoi regions
    // of the triangle. No epsilons, and well-defined for degenerate triangles.
    let (a, b, c) = t;

    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(&ap);
    let d2 = ac.dot(&ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }

    let bp = p - b;
    let d3 = ab.dot(&bp);
    let d4 = ac.dot(&bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + ab * (d1 / (d1 - d3));
    }

    let cp = p - c;
    let d5 = ab.dot(&cp);
    let d6 = ac.dot(&cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + ac * (d2 / (d2 - d6));
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }

    let denom = va + vb + vc;
    if denom == 0.0 {
        // Fully degenerate (all points coincide or collinear in a way that fell through).
        return point_on_edge(p, a, b);
    }
    let v = vb / denom;
    let w = vc / denom;
    a + ab * v + ac * w
}

/// Calculates the closest point on the edge defined by points `a` and `b` to point `p`
#[must_use]
pub fn point_on_edge(p: Vector3D, a: Vector3D, b: Vector3D) -> Vector3D {
    let ab = b - a;
    let len_sq = ab.dot(&ab);
    if len_sq == 0.0 {
        return a;
    }
    let t = (p - a).dot(&ab) / len_sq;
    let t_clamped = t.clamp(0.0, 1.0);
    a + ab * t_clamped
}

/// Calculates the distance of point `p` to triangle `t`
#[must_use]
pub fn distance_to_triangle(p: Vector3D, t: (Vector3D, Vector3D, Vector3D)) -> f64 {
    let closest_point = point_on_triangle(p, t);
    (p - closest_point).norm()
}

// Calculate the barycentric coordinates of point `p` with respect to triangle `t`.
#[must_use]
#[inline]
pub fn calculate_barycentric_coordinates(
    p: Vector3D,
    t: (Vector3D, Vector3D, Vector3D),
) -> (f64, f64, f64) {
    let (a, b, c) = t;
    let ab = b - a;
    let ac = c - a;
    let d00 = ab.dot(&ab);
    let d01 = ab.dot(&ac);
    let d11 = ac.dot(&ac);
    let denom = d00 * d11 - d01 * d01;

    let ap2 = p - a;
    let d20 = ap2.dot(&ab);
    let d21 = ap2.dot(&ac);

    // `denom` = |ab|^2 |ac|^2 sin^2(angle) scales with length^4, so the degeneracy test must be relative:
    // an absolute threshold misclassifies every small (but valid) triangle as degenerate.
    if denom <= f64::EPSILON * d00 * d11 {
        // Degenerate case: the triangle is a line or a point. Fall back to a 1D parameterization
        // along its longest edge.
        let bc = c - b;
        let ca = a - c;
        let candidates = [(ab.dot(&ab), 0), (bc.dot(&bc), 1), (ca.dot(&ca), 2)];
        let (len2, which) = candidates
            .into_iter()
            .max_by(|x, y| x.0.total_cmp(&y.0))
            .unwrap();
        if len2 <= 0.0 {
            // All points are the same.
            return (1.0, 0.0, 0.0);
        }
        return match which {
            0 => {
                let t = (p - a).dot(&ab) / len2;
                (1.0 - t, t, 0.0)
            }
            1 => {
                let t = (p - b).dot(&bc) / len2;
                (0.0, 1.0 - t, t)
            }
            _ => {
                let t = (p - c).dot(&ca) / len2;
                (t, 0.0, 1.0 - t)
            }
        };
    }

    let bar_b = (d11 * d20 - d01 * d21) / denom;
    let bar_c = (d00 * d21 - d01 * d20) / denom;
    let bar_a = 1.0 - bar_b - bar_c;

    (bar_a, bar_b, bar_c)
}

// Inverse barycentric coordinates: given barycentric coordinates (u, v, w), find the point p in triangle t.
#[must_use]
#[inline]
pub fn inverse_barycentric_coordinates(
    u: f64,
    v: f64,
    w: f64,
    t: (Vector3D, Vector3D, Vector3D),
) -> Vector3D {
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
    (
        Vector3D::new(eigenvector[0], eigenvector[1], eigenvector[2]),
        rms / diameter,
    )
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
pub fn triangle_to_2d(
    p0: Vector3D,
    p1: Vector3D,
    p2: Vector3D,
) -> Option<(Vector2D, Vector2D, Vector2D)> {
    // 1. Compute edges
    let v1 = p1 - p0;
    let v2 = p2 - p0;

    // Reject degenerate triangles
    if v1.norm() < 1e-12 || v2.norm() < 1e-12 {
        return None;
    }
    let cross = v1.cross(&v2);
    if cross.norm() < 1e-12 {
        return None;
    }

    // 3. Construct orthonormal basis (u_x, u_y, u_z)
    let ux = v1.normalize();
    let uz = cross.normalize();
    let uy = uz.cross(&ux); // guaranteed orthonormal

    // 4. Project each vertex into 2D plane coordinates
    let q0 = Vector2D::new(p0.dot(&ux), p0.dot(&uy));
    let q1 = Vector2D::new(p1.dot(&ux), p1.dot(&uy));
    let q2 = Vector2D::new(p2.dot(&ux), p2.dot(&uy));

    Some((q0, q1, q2))
}
