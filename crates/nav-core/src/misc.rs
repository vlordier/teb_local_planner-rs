#![allow(unused_variables, unused_imports, dead_code, unused_mut, unused_assignments, unreachable_patterns)]
//! Math helpers ported from `misc.h` and `distance_calculations.h`.

/// epsilon for SMALL_NUM comparisons
pub const SMALL_NUM: f64 = 1e-8;

/// Normalize angle to [-pi, pi]. Same as g2o::normalize_theta (C++).
pub fn normalize_angle(mut angle: f64) -> f64 {
    // NaN and Inf pass through unchanged — intentional for CBMC
    // compatibility and matches g2o behavior where NaN angles are
    // undefined and the caller must handle them.
    if angle.is_nan() || angle.is_infinite() {
        return angle;
    }
    // Use compile-time constant for CBMC floatbv reasoning.
    // CBMC can prove x % immediate_constant properties for concrete values,
    // but not x % symbolic_f64_divisor.
    const TWO_PI: f64 = 2.0 * std::f64::consts::PI;
    // Match g2o::normalize_theta: fmod first, then adjust
    // fmod returns result with same sign as dividend
    angle = angle % TWO_PI;
    if angle < -std::f64::consts::PI {
        angle += TWO_PI;
    } else if angle >= std::f64::consts::PI {
        angle -= TWO_PI;
    }
    angle
}

/// Alias for normalize_angle.
pub fn normalize_theta(angle: f64) -> f64 {
    normalize_angle(angle)
}

/// Average angle using mean direction vector.
pub fn average_angles(angles: &[f64]) -> f64 {
    let (s_sin, s_cos): (f64, f64) =
        angles.iter().fold((0.0, 0.0), |(s, c), &a| (s + a.sin(), c + a.cos()));
    if s_sin.abs() < SMALL_NUM && s_cos.abs() < SMALL_NUM {
        0.0
    } else {
        s_sin.atan2(s_cos)
    }
}

/// Fast sigmoid: x / (1 + |x|).
pub fn fast_sigmoid(x: f64) -> f64 {
    x / (1.0 + x.abs())
}

/// Check |a - b| < epsilon.
pub fn is_close(a: f64, b: f64, epsilon: f64) -> bool {
    (a - b).abs() < epsilon
}

/// Euclidean distance between two Vector2.
pub fn distance_points2d(a: &nalgebra::Vector2<f64>, b: &nalgebra::Vector2<f64>) -> f64 {
    (b - a).norm()
}

/// 2D cross product (z-component).
pub fn cross2d(v1: &nalgebra::Vector2<f64>, v2: &nalgebra::Vector2<f64>) -> f64 {
    v1.x * v2.y - v1.y * v2.x
}

/// Closest point on segment.
pub fn closest_point_on_line_segment_2d(
    point: &nalgebra::Vector2<f64>,
    line_start: &nalgebra::Vector2<f64>,
    line_end: &nalgebra::Vector2<f64>,
) -> nalgebra::Vector2<f64> {
    let diff = *line_end - line_start;
    let sq_norm = diff.dot(&diff);
    if sq_norm == 0.0 {
        return *line_start;
    }
    let u = ((point.x - line_start.x) * diff.x + (point.y - line_start.y) * diff.y) / sq_norm;
    if u <= 0.0 { *line_start } else if u >= 1.0 { *line_end } else { *line_start + diff * u }
}

/// Distance point to segment.
pub fn distance_point_to_segment_2d(
    point: &nalgebra::Vector2<f64>,
    line_start: &nalgebra::Vector2<f64>,
    line_end: &nalgebra::Vector2<f64>,
) -> f64 {
    (point - closest_point_on_line_segment_2d(point, line_start, line_end)).norm()
}

/// Check line segment intersection.
pub fn check_line_segments_intersection_2d(
    l1s: &nalgebra::Vector2<f64>, l1e: &nalgebra::Vector2<f64>,
    l2s: &nalgebra::Vector2<f64>, l2e: &nalgebra::Vector2<f64>,
) -> (bool, nalgebra::Vector2<f64>) {
    let l1 = *l1e - l1s;
    let l2 = *l2e - l2s;
    let d = l1.x * l2.y - l2.x * l1.y;
    if d.abs() < SMALL_NUM { return (false, nalgebra::Vector2::zeros()); }
    let dp = d > 0.0;
    let aux = *l1s - l2s;
    let sn = l1.x * aux.y - l1.y * aux.x;
    if (sn < 0.0) != dp { return (false, nalgebra::Vector2::zeros()); }
    let tn = l2.x * aux.y - l2.y * aux.x;
    if (tn < 0.0) != dp { return (false, nalgebra::Vector2::zeros()); }
    if (sn > d) != dp || (tn > d) != dp { return (false, nalgebra::Vector2::zeros()); }
    (true, *l1s + l1 * (tn / d))
}

/// Distance between two segments.
pub fn distance_segment_to_segment_2d(
    s1s: &nalgebra::Vector2<f64>, s1e: &nalgebra::Vector2<f64>,
    s2s: &nalgebra::Vector2<f64>, s2e: &nalgebra::Vector2<f64>,
) -> f64 {
    if check_line_segments_intersection_2d(s1s, s1e, s2s, s2e).0 { return 0.0; }
    let d = [
        distance_point_to_segment_2d(s1s, s2s, s2e),
        distance_point_to_segment_2d(s1e, s2s, s2e),
        distance_point_to_segment_2d(s2s, s1s, s1e),
        distance_point_to_segment_2d(s2e, s1s, s1e),
    ];
    d[0].min(d[1]).min(d[2]).min(d[3])
}

/// Distance point to polygon.
pub fn distance_point_to_polygon_2d(p: &nalgebra::Vector2<f64>, verts: &[nalgebra::Vector2<f64>]) -> f64 {
    let mut dist = f64::INFINITY;
    if verts.is_empty() { return dist; }
    if verts.len() == 1 { return (p - verts[0]).norm(); }
    for i in 0..verts.len() - 1 {
        let d = distance_point_to_segment_2d(p, &verts[i], &verts[i+1]);
        if d < dist { dist = d; }
    }
    if verts.len() > 2 {
        let d = distance_point_to_segment_2d(p, verts.last().unwrap(), &verts[0]);
        if d < dist { dist = d; }
    }
    dist
}

/// Distance segment to polygon.
pub fn distance_segment_to_polygon_2d(
    ls: &nalgebra::Vector2<f64>, le: &nalgebra::Vector2<f64>, verts: &[nalgebra::Vector2<f64>],
) -> f64 {
    let mut dist = f64::INFINITY;
    if verts.is_empty() { return dist; }
    if verts.len() == 1 { return distance_point_to_segment_2d(&verts[0], ls, le); }
    for i in 0..verts.len() - 1 {
        let d = distance_segment_to_segment_2d(ls, le, &verts[i], &verts[i+1]);
        if d < dist { dist = d; }
    }
    if verts.len() > 2 {
        let d = distance_segment_to_segment_2d(ls, le, verts.last().unwrap(), &verts[0]);
        if d < dist { dist = d; }
    }
    dist
}

/// Distance polygon to polygon.
pub fn distance_polygon_to_polygon_2d(v1: &[nalgebra::Vector2<f64>], v2: &[nalgebra::Vector2<f64>]) -> f64 {
    let mut dist = f64::INFINITY;
    if v1.is_empty() || v2.is_empty() { return dist; }
    if v1.len() == 1 { return distance_point_to_polygon_2d(&v1[0], v2); }
    for i in 0..v1.len() - 1 {
        let d = distance_segment_to_polygon_2d(&v1[i], &v1[i+1], v2);
        if d < dist { dist = d; }
    }
    if v1.len() > 2 {
        let d = distance_segment_to_polygon_2d(v1.last().unwrap(), &v1[0], v2);
        if d < dist { dist = d; }
    }
    dist
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_angle() {
        assert_eq!(normalize_angle(0.0), 0.0);
        // g2o::normalize_theta(3*PI): fmod -> PI >= PI -> PI - 2*PI = -PI
        assert_eq!(normalize_angle(3.0 * std::f64::consts::PI), -std::f64::consts::PI);
        // g2o::normalize_theta(-3*PI): fmod -> -PI, no adjustment -> -PI
        assert_eq!(normalize_angle(-3.0 * std::f64::consts::PI), -std::f64::consts::PI);
        assert_eq!(normalize_angle(std::f64::consts::PI / 2.0), std::f64::consts::PI / 2.0);
    }

    #[test]
    fn test_average_angles() {
        assert_eq!(average_angles(&[0.0, 0.0]), 0.0);
    }

    #[test]
    fn test_fast_sigmoid() {
        assert_eq!(fast_sigmoid(0.0), 0.0);
        assert_eq!(fast_sigmoid(-1.0), -0.5);
    }

    #[test]
    fn test_cross2d() {
        let a = nalgebra::Vector2::new(1.0, 0.0);
        let b = nalgebra::Vector2::new(0.0, 1.0);
        assert_eq!(cross2d(&a, &b), 1.0);
    }
}
