#![allow(unused_variables, unused_imports, dead_code, unused_mut, unused_assignments, unreachable_patterns)]
//! Math utilities for TEB planner.
//!
//! Ported from C++ `misc.h`, `penalties.h`, and `distance_calculations.h`.

use approx::AbsDiffEq;
use nalgebra::{ComplexField, DVector, OMatrix, Scalar};

/// Maximum value for double (approximate).
pub const HUGE_VAL: f64 = f64::INFINITY;
/// Small number threshold.
pub const SMALL_NUM: f64 = 1e-8;

#[inline]
pub fn normalize_theta(theta: f64) -> f64 {
    let pi2 = 2.0 * std::f64::consts::PI;
    let mut t = theta % pi2;
    if t < -std::f64::consts::PI { t += pi2; }
    else if t >= std::f64::consts::PI { t -= pi2; }
    t
}

/// Compute the mean/average angle of multiple angles.
///
/// Uses the circular mean: atan2(sum(sin), sum(cos)).
#[inline]
pub fn average_angles(angles: &[f64]) -> f64 {
    let (sum_sin, sum_cos): (f64, f64) = angles.iter().fold((0.0, 0.0), |(s, c), &a| {
        (s + a.sin(), c + a.cos())
    });
    if sum_sin.abs() < SMALL_NUM && sum_cos.abs() < SMALL_NUM {
        0.0
    } else {
        sum_sin.atan2(sum_cos)
    }
}

/// Alias for average_angles — circular mean of angles, already normalized to [-π, π].
pub fn normalize_angles(angles: &[f64]) -> f64 {
    average_angles(angles)
}

/// Fast sigmoid approximation: x / (1 + |x|).
///
/// Used instead of std::sign() for direction detection in velocity edges.
#[inline]
pub fn fast_sigmoid(x: f64) -> f64 {
    x / (1.0 + x.abs())
}

/// 2D cross product (returns z-component of 3D cross product).
///
/// Positive if v2 is counter-clockwise from v1.
#[inline]
pub fn cross2d<T: Scalar + ComplexField + Copy>(v1: &OMatrix<T, nalgebra::Const<2>, nalgebra::Const<1>>, v2: &OMatrix<T, nalgebra::Const<2>, nalgebra::Const<1>>) -> T {
    v1[0] * v2[1] - v2[0] * v1[1]
}

/// Check if two doubles are close.
#[inline]
pub fn is_close(a: f64, b: f64, epsilon: f64) -> bool {
    (a - b).abs() < epsilon
}

/// Sign function (returns -1, 0, or 1).
#[inline]
pub fn sign(x: f64) -> f64 {
    if x < 0.0 { -1.0 } else if x > 0.0 { 1.0 } else { 0.0 }
}

// ===== Penalty functions =====

/// Linear penalty for bounding `var` to the interval `-a < var < a`.
///
/// C++ equivalent: `penaltyBoundToInterval(var, -a, a, epsilon)`
#[inline]
pub fn penalty_bound_to_interval_symmetric(var: f64, a: f64, epsilon: f64) -> f64 {
    if var < -a + epsilon {
        (-var - (a - epsilon)).abs()
    } else if var <= a - epsilon {
        0.0
    } else {
        var - (a - epsilon)
    }
}

/// Linear penalty for bounding `var` to the interval `a < var < b`.
#[inline]
pub fn penalty_bound_to_interval(var: f64, a: f64, b: f64, epsilon: f64) -> f64 {
    if var < a + epsilon {
        (-var + (a + epsilon)).abs()
    } else if var <= b - epsilon {
        0.0
    } else {
        var - (b - epsilon)
    }
}

/// Linear penalty for bounding `var` from below: `a < var`.
///
/// C++ equivalent: `penaltyBoundFromBelow(var, a, epsilon)`
#[inline]
pub fn penalty_bound_from_below(var: f64, a: f64, epsilon: f64) -> f64 {
    if var >= a + epsilon {
        0.0
    } else {
        (-var + (a + epsilon)).abs()
    }
}

/// Derivative of `penalty_bound_to_interval_symmetric` w.r.t. `var`.
#[inline]
pub fn penalty_bound_to_interval_deriv_symmetric(var: f64, a: f64, epsilon: f64) -> f64 {
    if var < -a + epsilon {
        -1.0
    } else if var <= a - epsilon {
        0.0
    } else {
        1.0
    }
}

/// Derivative of `penalty_bound_to_interval` w.r.t. `var`.
#[inline]
pub fn penalty_bound_to_interval_deriv(var: f64, a: f64, b: f64, epsilon: f64) -> f64 {
    if var < a + epsilon {
        -1.0
    } else if var <= b - epsilon {
        0.0
    } else {
        1.0
    }
}

/// Derivative of `penalty_bound_from_below` w.r.t. `var`.
///
/// C++ equivalent: `penaltyBoundFromBelowDerivative(var, a, epsilon)`
#[inline]
pub fn penalty_bound_from_below_deriv(var: f64, a: f64, epsilon: f64) -> f64 {
    if var >= a + epsilon {
        0.0
    } else {
        -1.0
    }
}

// ===== Distance calculations =====

/// Find the closest point on a line segment to a reference point.
#[inline]
pub fn closest_point_on_line_segment_2d(
    point: &nalgebra::Vector2<f64>,
    line_start: &nalgebra::Vector2<f64>,
    line_end: &nalgebra::Vector2<f64>,
) -> nalgebra::Vector2<f64> {
    let diff = *line_end - line_start;
    let sq_norm = diff.norm_squared();

    if sq_norm == 0.0 {
        return *line_start;
    }

    let u = ((point.x - line_start.x) * diff.x + (point.y - line_start.y) * diff.y) / sq_norm;

    if u <= 0.0 {
        *line_start
    } else if u >= 1.0 {
        *line_end
    } else {
        line_start + diff * u
    }
}

/// Distance from a point to a line segment.
#[inline]
pub fn distance_point_to_segment_2d(
    point: &nalgebra::Vector2<f64>,
    line_start: &nalgebra::Vector2<f64>,
    line_end: &nalgebra::Vector2<f64>,
) -> f64 {
    (point - closest_point_on_line_segment_2d(point, line_start, line_end)).norm()
}

/// Check if two line segments intersect.
///
/// Returns true if they intersect; optionally writes the intersection point.
pub fn check_line_segments_intersect_2d(
    line1_start: &nalgebra::Vector2<f64>,
    line1_end: &nalgebra::Vector2<f64>,
    line2_start: &nalgebra::Vector2<f64>,
    line2_end: &nalgebra::Vector2<f64>,
    intersection: &mut Option<nalgebra::Vector2<f64>>,
) -> bool {
    let line1 = *line1_end - line1_start;
    let line2 = *line2_end - line2_start;

    let denom = line1.x * line2.y - line2.x * line1.y;
    if denom.abs() < SMALL_NUM {
        return false; // Collinear
    }
    let denom_positive = denom > 0.0;

    let aux = *line1_start - line2_start;

    let s_numer = line1.x * aux.y - line1.y * aux.x;
    if (s_numer < 0.0) == denom_positive {
        return false;
    }

    let t_numer = line2.x * aux.y - line2.y * aux.x;
    if (t_numer < 0.0) == denom_positive {
        return false;
    }

    if (s_numer > denom) == denom_positive || (t_numer > denom) == denom_positive {
        return false;
    }

    if let Some(pt) = intersection {
        *pt = *line1_start + (t_numer / denom) * line1;
    }

    true
}

/// Smallest distance between two line segments.
pub fn distance_segment_to_segment_2d(
    line1_start: &nalgebra::Vector2<f64>,
    line1_end: &nalgebra::Vector2<f64>,
    line2_start: &nalgebra::Vector2<f64>,
    line2_end: &nalgebra::Vector2<f64>,
) -> f64 {
    let mut intersection = None;
    if check_line_segments_intersect_2d(line1_start, line1_end, line2_start, line2_end, &mut intersection) {
        return 0.0;
    }

    let d0 = distance_point_to_segment_2d(line1_start, line2_start, line2_end);
    let d1 = distance_point_to_segment_2d(line1_end, line2_start, line2_end);
    let d2 = distance_point_to_segment_2d(line2_start, line1_start, line1_end);
    let d3 = distance_point_to_segment_2d(line2_end, line1_start, line1_end);

    d0.min(d1).min(d2).min(d3)
}

/// Smallest distance from a point to a closed polygon.
pub fn distance_point_to_polygon_2d(
    point: &nalgebra::Vector2<f64>,
    vertices: &[nalgebra::Vector2<f64>],
) -> f64 {
    if vertices.is_empty() {
        return f64::MAX;
    }

    if vertices.len() == 1 {
        return (point - vertices[0]).norm();
    }

    let mut dist = f64::MAX;

    // Check each edge
    for i in 0..vertices.len() - 1 {
        let d = distance_point_to_segment_2d(point, &vertices[i], &vertices[i + 1]);
        if d < dist {
            dist = d;
        }
    }

    // Close the polygon (last vertex to first)
    if vertices.len() > 2 {
        let d = distance_point_to_segment_2d(point, &vertices[vertices.len() - 1], &vertices[0]);
        if d < dist {
            dist = d;
        }
    }

    dist
}

/// Smallest distance from a line segment to a closed polygon.
pub fn distance_segment_to_polygon_2d(
    line_start: &nalgebra::Vector2<f64>,
    line_end: &nalgebra::Vector2<f64>,
    vertices: &[nalgebra::Vector2<f64>],
) -> f64 {
    if vertices.is_empty() {
        return f64::MAX;
    }

    if vertices.len() == 1 {
        return distance_point_to_segment_2d(&vertices[0], line_start, line_end);
    }

    let mut dist = f64::MAX;

    for i in 0..vertices.len() - 1 {
        let d = distance_segment_to_segment_2d(line_start, line_end, &vertices[i], &vertices[i + 1]);
        if d < dist {
            dist = d;
        }
    }

    if vertices.len() > 2 {
        let d = distance_segment_to_segment_2d(
            line_start,
            line_end,
            &vertices[vertices.len() - 1],
            &vertices[0],
        );
        if d < dist {
            dist = d;
        }
    }

    dist
}

/// Smallest distance between two closed polygons.
pub fn distance_polygon_to_polygon_2d(
    vertices1: &[nalgebra::Vector2<f64>],
    vertices2: &[nalgebra::Vector2<f64>],
) -> f64 {
    if vertices1.is_empty() || vertices2.is_empty() {
        return f64::MAX;
    }

    if vertices1.len() == 1 {
        return distance_point_to_polygon_2d(&vertices1[0], vertices2);
    }

    let mut dist = f64::MAX;

    for i in 0..vertices1.len() - 1 {
        let d = distance_segment_to_polygon_2d(&vertices1[i], &vertices1[i + 1], vertices2);
        if d < dist {
            dist = d;
        }
    }

    if vertices1.len() > 2 {
        let d = distance_segment_to_polygon_2d(
            &vertices1[vertices1.len() - 1],
            &vertices1[0],
            vertices2,
        );
        if d < dist {
            dist = d;
        }
    }

    dist
}

/// Euclidean distance between two 2D points.
#[inline]
pub fn distance_points2d(a: &nalgebra::Vector2<f64>, b: &nalgebra::Vector2<f64>) -> f64 {
    (a - b).norm()
}

/// Absolute value for f64.
#[inline]
pub fn fabs(x: f64) -> f64 {
    x.abs()
}

/// Check if |a| < |b|.
#[inline]
pub fn smaller_than_abs(i: f64, j: f64) -> bool {
    i.abs() < j.abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_theta() {
        assert!((normalize_theta(0.0) - 0.0).abs() < 1e-10);
        assert!((normalize_theta(std::f64::consts::PI) + std::f64::consts::PI).abs() < 1e-10);
        assert!((normalize_theta(3.0 * std::f64::consts::PI) + std::f64::consts::PI).abs() < 1e-10);
        assert!((normalize_theta(-3.0 * std::f64::consts::PI) + std::f64::consts::PI).abs() < 1e-10);
        assert!((normalize_theta(5.0 * std::f64::consts::PI / 2.0) - std::f64::consts::PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_average_angles() {
        let angles = vec![0.0, 0.1, -0.1];
        let avg = average_angles(&angles);
        assert!((avg - 0.0).abs() < 1e-10);

        let angles = vec![std::f64::consts::PI - 0.01, std::f64::consts::PI + 0.01];
        let avg = average_angles(&angles);
        assert!((avg - std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn test_fast_sigmoid() {
        assert!((fast_sigmoid(0.0) - 0.0).abs() < 1e-10);
        assert!((fast_sigmoid(1.0) - 0.5).abs() < 1e-10);
        assert!((fast_sigmoid(-1.0) - (-0.5)).abs() < 1e-10);
        assert!((fast_sigmoid(100.0) - 1.0).abs() < 0.02); // approaches 1.0
        assert!((fast_sigmoid(-100.0) - (-1.0)).abs() < 0.02); // approaches -1.0
    }

    #[test]
    fn test_cross2d() {
        let v1 = nalgebra::Vector2::new(1.0, 0.0);
        let v2 = nalgebra::Vector2::new(0.0, 1.0);
        assert!((cross2d(&v1, &v2) - 1.0).abs() < 1e-10);

        let v3 = nalgebra::Vector2::new(0.0, 1.0);
        assert!((cross2d(&v1, &v3) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_penalty_bound_from_below() {
        // Within bounds
        assert!((penalty_bound_from_below(5.0, 3.0, 0.1) - 0.0).abs() < 1e-10);
        // At boundary with epsilon
        assert!((penalty_bound_from_below(3.1, 3.0, 0.1) - 0.0).abs() < 1e-10);
        // Below boundary
        assert!((penalty_bound_from_below(2.5, 3.0, 0.1) - 0.6).abs() < 1e-10);
        // Way below
        assert!((penalty_bound_from_below(0.0, 3.0, 0.1) - 3.1).abs() < 1e-10);
    }

    #[test]
    fn test_penalty_bound_from_below_deriv() {
        assert!((penalty_bound_from_below_deriv(5.0, 3.0, 0.1) - 0.0).abs() < 1e-10);
        assert!((penalty_bound_from_below_deriv(2.5, 3.0, 0.1) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_distance_point_to_segment() {
        let point = nalgebra::Vector2::new(0.0, 1.0);
        let start = nalgebra::Vector2::new(0.0, 0.0);
        let end = nalgebra::Vector2::new(1.0, 0.0);
        assert!((distance_point_to_segment_2d(&point, &start, &end) - 1.0).abs() < 1e-10);

        // Point beyond segment end
        let point2 = nalgebra::Vector2::new(2.0, 1.0);
        let dist = distance_point_to_segment_2d(&point2, &start, &end);
        let expected = ((2.0 - 1.0).powi(2) + 1.0).sqrt();
        assert!((dist - expected).abs() < 1e-10);
    }

    #[test]
    fn test_distance_point_to_polygon() {
        // Square centered at origin with side 2
        let vertices = vec![
            nalgebra::Vector2::new(-1.0, -1.0),
            nalgebra::Vector2::new(1.0, -1.0),
            nalgebra::Vector2::new(1.0, 1.0),
            nalgebra::Vector2::new(-1.0, 1.0),
        ];
        // Point outside
        let outside = nalgebra::Vector2::new(2.0, 0.0);
        assert!((distance_point_to_polygon_2d(&outside, &vertices) - 1.0).abs() < 1e-10);
        // Point inside
        let inside = nalgebra::Vector2::new(0.0, 0.0);
        assert!((distance_point_to_polygon_2d(&inside, &vertices) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_closest_point_on_line_segment() {
        let point = nalgebra::Vector2::new(0.5, 0.5);
        let start = nalgebra::Vector2::new(0.0, 0.0);
        let end = nalgebra::Vector2::new(1.0, 0.0);
        let closest = closest_point_on_line_segment_2d(&point, &start, &end);
        assert!((closest.x - 0.5).abs() < 1e-10);
        assert!((closest.y - 0.0).abs() < 1e-10);
    }
}
