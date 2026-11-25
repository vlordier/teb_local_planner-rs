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
    // Call C++ PoseSE2 via FFI
    let cp_px = cpp::pose_se2_x(x, y, theta);
    let cp_py = cpp::pose_se2_y(x, y, theta);
    let cp_ptheta = cpp::pose_se2_theta(x, y, theta);
    
    // Call Rust PoseSE2
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
    // C++ addition
    let cp_x = cpp::pose_se2_add_x(p1_x, p1_y, p1_theta, p2_x, p2_y, p2_theta);
    let cp_y = cpp::pose_se2_add_y(p1_x, p1_y, p1_theta, p2_x, p2_y, p2_theta);
    let cp_theta = cpp::pose_se2_add_theta(p1_x, p1_y, p1_theta, p2_x, p2_y, p2_theta);
    
    // Rust addition
    let p1 = nav_core::PoseSE2::new(p1_x, p1_y, p1_theta);
    let p2 = nav_core::PoseSE2::new(p2_x, p2_y, p2_theta);
    let result = p1.add(&p2);
    let rust_x = result.x();
    let rust_y = result.y();
    let rust_theta = result.theta();
    
    compare_floats(cp_x, rust_x, "x")
        .merge(&compare_floats(cp_y, rust_y, "y"))
        .merge(&compare_floats(cp_theta, rust_theta, "theta"))
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
