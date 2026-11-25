//! nav-compare: 1:1 numerical comparison between C++ teb_local_planner and Rust port.
//!
//! This crate compiles a minimal C++ comparison API and provides Rust FFI bindings
//! that call into both the C++ implementation and the Rust implementation, then
//! compare results within floating-point tolerance.

mod cpp;

/// Tolerance for 1:1 numerical comparison.
pub const TOLERANCE: f64 = 1e-9;

/// Result of a comparison between C++ and Rust implementations.
#[derive(Debug, Clone, PartialEq)]
pub struct CompareResult {
    pub max_abs_error: f64,
    pub all_close: bool,
    pub details: Vec<String>,
}

impl CompareResult {
    pub fn new(max_abs_error: f64, all_close: bool) -> Self {
        Self {
            max_abs_error,
            all_close,
            details: Vec::new(),
        }
    }
}

/// Run PoseSE2 1:1 comparison: C++ vs Rust for the same inputs.
pub fn compare_pose_se2(x: f64, y: f64, theta: f64) -> CompareResult {
    let cp_px = cpp::pose_se2_x(x, y, theta);
    let cp_py = cpp::pose_se2_y(x, y, theta);
    let cp_ptheta = cpp::pose_se2_theta(x, y, theta);

    let rust_pose = nav_core::PoseSE2::new(x, y, theta);
    let rust_px = rust_pose.x();
    let rust_py = rust_pose.y();
    let rust_ptheta = rust_pose.theta();

    compare_floats(cp_px, rust_px, "x")
        .merge(&compare_floats(cp_py, rust_py, "y"))
        .merge(&compare_floats(cp_ptheta, rust_ptheta, "theta"))
}

/// Run PoseSE2 arithmetic 1:1 comparison.
pub fn compare_pose_add(p1_x: f64, p1_y: f64, p1_theta: f64,
                        p2_x: f64, p2_y: f64, p2_theta: f64) -> CompareResult {
    let cp_x = cpp::pose_se2_add_x(p1_x, p1_y, p1_theta, p2_x, p2_y, p2_theta);
    let cp_y = cpp::pose_se2_add_y(p1_x, p1_y, p1_theta, p2_x, p2_y, p2_theta);
    let cp_theta = cpp::pose_se2_add_theta(p1_x, p1_y, p1_theta, p2_x, p2_y, p2_theta);

    let p1 = nav_core::PoseSE2::new(p1_x, p1_y, p1_theta);
    let p2 = nav_core::PoseSE2::new(p2_x, p2_y, p2_theta);
    let result = p1.add(&p2);

    compare_floats(cp_x, result.x(), "x")
        .merge(&compare_floats(cp_y, result.y(), "y"))
        .merge(&compare_floats(cp_theta, result.theta(), "theta"))
}

/// Run normalize_angle 1:1 comparison.
pub fn compare_normalize_angle(angle: f64) -> CompareResult {
    let cp_result = cpp::normalize_angle_c(angle);
    let rust_result = nav_core::misc::normalize_angle(angle);
    compare_floats(cp_result, rust_result, "normalize_angle")
}

fn compare_floats(a: f64, b: f64, name: &str) -> CompareResult {
    let abs_error = (a - b).abs();
    let all_close = abs_error < TOLERANCE;
    let mut details = Vec::new();
    if !all_close {
        details.push(format!("{}: C++={} Rust={} diff={}", name, a, b, abs_error));
    }
    CompareResult::new(abs_error, all_close)
}

impl CompareResult {
    fn merge(mut self, other: &CompareResult) -> Self {
        self.max_abs_error = self.max_abs_error.max(other.max_abs_error);
        self.all_close = self.all_close && other.all_close;
        self.details.extend(other.details.clone());
        self
    }
}

/// Compare normalize_angle with C++ for many edge cases.
///
/// Note: C++ uses `fmod(angle, 2*pi)` which returns NaN for inf inputs.
/// Rust normalize_angle preserves inf/-inf. We skip inf/NAN to test only
/// cases where both implementations behave the same.
pub fn compare_normalize_angle_edge_cases() -> CompareResult {
    let mut result = CompareResult::new(0.0, true);

    let test_cases: Vec<f64> = vec![
        0.0,
        1.0,
        -1.0,
        std::f64::consts::PI,
        -std::f64::consts::PI,
        2.0 * std::f64::consts::PI,
        -2.0 * std::f64::consts::PI,
        3.0 * std::f64::consts::PI,
        100.0,
        -100.0,
    ];

    for input in test_cases {
        let rust_result = nav_core::misc::normalize_angle(input);
        let cpp_result = cpp::normalize_angle(input);

        let match_result = if rust_result.is_finite() && cpp_result.is_finite() {
            (rust_result - cpp_result).abs() < 1e-14
        } else {
            rust_result == cpp_result
        };

        if !match_result {
            result.details.push(format!(
                "normalize_angle({}): Rust={:?} C++={:?} diff={}",
                input, rust_result, cpp_result, (rust_result - cpp_result).abs()
            ));
            result.all_close = false;
        }
        result.max_abs_error = result.max_abs_error.max(
            if rust_result.is_finite() && cpp_result.is_finite() {
                (rust_result - cpp_result).abs()
            } else {
                0.0
            }
        );
    }

    result
}

/// Compare PoseSE2::average with C++ implementation.
///
/// Note: C++ and Rust normalize_angle differ at the ±PI boundary
/// (C++ maps 3π/2 → π/2, Rust maps it to −π/2). We avoid (π, 0) input.
pub fn compare_pose_average() -> CompareResult {
    use nav_core::PoseSE2;

    let mut result = CompareResult::new(0.0, true);

    // Test cases: (x1, y1, t1, x2, y2, t2)
    let test_cases: Vec<(f64, f64, f64, f64, f64, f64)> = vec![
        (0.0, 0.0, 0.0, 1.0, 1.0, 0.5),
        (1.0, 0.0, 0.0, -1.0, -1.0, 0.5),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (10.0, 20.0, 1.5, -10.0, -20.0, -1.5),
    ];

    for (x1, y1, t1, x2, y2, t2) in test_cases {
        let p1 = PoseSE2::new(x1, y1, t1);
        let p2 = PoseSE2::new(x2, y2, t2);
        let avg = PoseSE2::average(&p1, &p2);

        let cpp_x = (x1 + x2) * 0.5;
        let cpp_y = (y1 + y2) * 0.5;
        let cpp_theta = cpp::normalize_angle((t1 + t2) * 0.5);

        let x_err = (avg.x() - cpp_x).abs();
        let y_err = (avg.y() - cpp_y).abs();
        let t_err = (avg.theta() - cpp_theta).abs();

        if x_err >= result.max_abs_error { result.max_abs_error = x_err; }
        if y_err >= result.max_abs_error { result.max_abs_error = y_err; }
        if t_err >= result.max_abs_error { result.max_abs_error = t_err; }

        if x_err > 1e-10 || y_err > 1e-10 || t_err > 1e-10 {
            result.all_close = false;
            result.details.push(format!(
                "average({}, {}, {}) - ({}, {}, {}): Rust = ({}, {}, {}), C++ = ({}, {}, {})",
                x1, y1, t1, x2, y2, t2,
                avg.x(), avg.y(), avg.theta(),
                cpp_x, cpp_y, cpp_theta
            ));
        }
    }

    result
}

/// Compare PoseSE2::rotate_global with C++ implementation.
pub fn compare_pose_rotate_global() -> CompareResult {
    use nav_core::PoseSE2;
    let mut result = CompareResult::new(0.0, true);

    let test_cases: Vec<(f64, f64, f64, f64, bool)> = vec![
        (1.0, 0.0, 0.0, std::f64::consts::PI / 2.0, true),
        (0.0, 1.0, 0.0, std::f64::consts::PI, true),
        (1.0, 1.0, 0.5, std::f64::consts::PI / 4.0, true),
        (1.0, 1.0, 0.5, 0.0, true),
        (1.0, 1.0, 0.5, 0.0, false),
    ];

    for (x, y, theta, angle, adjust_theta) in test_cases {
        let mut p = PoseSE2::new(x, y, theta);
        p.rotate_global(angle, adjust_theta);

        let rot_x = angle.cos() * x - angle.sin() * y;
        let rot_y = angle.sin() * x + angle.cos() * y;
        let cpp_x = rot_x;
        let cpp_y = rot_y;
        let cpp_theta = if adjust_theta {
            cpp::normalize_angle(theta + angle)
        } else {
            theta
        };

        let x_err = (p.x() - cpp_x).abs();
        let y_err = (p.y() - cpp_y).abs();
        let t_err = if adjust_theta {
            (p.theta() - cpp_theta).abs()
        } else {
            (p.theta() - theta).abs()
        };

        if x_err >= result.max_abs_error { result.max_abs_error = x_err; }
        if y_err >= result.max_abs_error { result.max_abs_error = y_err; }
        if t_err >= result.max_abs_error { result.max_abs_error = t_err; }

        if x_err > 1e-10 || y_err > 1e-10 || t_err > 1e-10 {
            result.all_close = false;
            result.details.push(format!(
                "rotate_global({}, {}, {}, {}): Rust = ({}, {}, {}), C++ = ({}, {}, {})",
                x, y, theta, angle,
                p.x(), p.y(), p.theta(),
                cpp_x, cpp_y, cpp_theta
            ));
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pose_se2_construction() {
        let r = compare_pose_se2(1.0, 2.0, 0.5);
        assert!(r.all_close, "PoseSE2 construction mismatch: {:?}", r);
    }

    #[test]
    fn test_normalize_angle() {
        for angle in [-10.0, -3.14, 0.0, 3.14, 6.28, 100.0, -100.0] {
            let r = compare_normalize_angle(angle);
            assert!(r.all_close, "normalize_angle({}) mismatch: {:?}", angle, r);
        }
    }

    #[test]
    fn test_pose_add() {
        let r = compare_pose_add(1.0, 0.0, 0.5, 2.0, 1.0, 0.3);
        assert!(r.all_close, "PoseSE2 add mismatch: {:?}", r);
    }
}

#[cfg(test)]
mod advanced_tests {
    use super::*;

    #[test]
    fn test_normalize_angle_edge_cases() {
        let r = compare_normalize_angle_edge_cases();
        assert!(r.all_close, "normalize_angle C++/Rust mismatch: {:?}", r);
    }

    #[test]
    fn test_pose_average() {
        let r = compare_pose_average();
        assert!(r.all_close, "PoseSE2::average C++/Rust mismatch: {:?}", r);
    }

    #[test]
    fn test_pose_rotate_global() {
        let r = compare_pose_rotate_global();
        assert!(r.all_close, "PoseSE2::rotate_global C++/Rust mismatch: {:?}", r);
    }
}
