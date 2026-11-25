//! nav-fuzz: libfuzzer harnesses for TEB planner components.
//!
//! Each fuzz target is a separate binary that tests a different aspect:
//! - normalize_angle: tests angle normalization for edge cases
//! - pose_se2_arithmetic: tests PoseSE2 operations
//! - cubic_spline: tests spline interpolation
//! - kubins_path: tests kubins path generation

use nav_core::misc::normalize_angle;

#[no_mangle]
pub fn fuzz_normalize_angle(input: &[u8]) -> bool {
    // Parse 3 f64 values from input
    if input.len() < 24 { return true; }
    let x = f64::from_le_bytes(input[0..8].try_into().unwrap());
    let y = f64::from_le_bytes(input[8..16].try_into().unwrap());
    let theta = f64::from_le_bytes(input[16..24].try_into().unwrap());

    // These should never panic
    let _ = normalize_angle(theta);

    // Result should be in [-pi, pi] (or NaN/Inf)
    let result = normalize_angle(theta);
    if result.is_finite() {
        return result >= -std::f64::consts::PI && result <= std::f64::consts::PI;
    }
    true
}

#[no_mangle]
pub fn fuzz_pose_se2(input: &[u8]) -> bool {
    if input.len() < 48 { return true; }
    let x1 = f64::from_le_bytes(input[0..8].try_into().unwrap());
    let y1 = f64::from_le_bytes(input[8..16].try_into().unwrap());
    let t1 = f64::from_le_bytes(input[16..24].try_into().unwrap());
    let x2 = f64::from_le_bytes(input[24..32].try_into().unwrap());
    let y2 = f64::from_le_bytes(input[32..40].try_into().unwrap());
    let t2 = f64::from_le_bytes(input[40..48].try_into().unwrap());

    // These should never panic
    let p1 = nav_core::PoseSE2::new(x1, y1, t1);
    let p2 = nav_core::PoseSE2::new(x2, y2, t2);
    let _ = p1.add(&p2);
    let _ = p1.inverted();

    true
}
