#![allow(unused_variables, unused_imports, dead_code, unused_mut, unused_assignments, unreachable_patterns)]
//! Time-optimal parameterization of geometric trajectories.
//!
//! Port of the C++ `time_optimal_parameterization.hpp` from teb_local_planner.
//!
//! The algorithm computes minimum-time velocity profiles along a fixed geometric
//! path while respecting kinematic constraints (max velocity, acceleration).
//!
//! # Algorithm
//! 1. **Forward pass**: propagate velocity limits from start, respecting acceleration.
//! 2. **Backward pass**: propagate velocity limits from end, respecting deceleration.
//! 3. **Smoothing pass**: resolve oscillations where forward and backward profiles
//!    conflict, using the harmonic mean of adjacent velocities.

use nav_core::PoseSE2;

/// Configuration for time parameterization.
pub struct TimeParamConfig {
    /// Reference time step for discretization.
    pub dt_ref: f64,
    /// Maximum time step allowed.
    pub dt_max: f64,
    /// Minimum time step allowed.
    pub dt_min: f64,
    /// Maximum linear velocity.
    pub max_vel: f64,
    /// Maximum linear acceleration.
    pub max_acc: f64,
    /// Maximum yaw velocity.
    pub max_yaw_vel: f64,
    /// Maximum yaw acceleration.
    pub max_yaw_acc: f64,
}

impl Default for TimeParamConfig {
    fn default() -> Self {
        Self {
            dt_ref: 0.05,
            dt_max: 0.2,
            dt_min: 0.01,
            max_vel: 2.0,
            max_acc: 3.0,
            max_yaw_vel: 1.0,
            max_yaw_acc: 2.0,
        }
    }
}

/// Velocity profile along a trajectory.
pub struct VelocityProfile {
    /// Linear velocity at each waypoint.
    pub velocities: Vec<f64>,
    /// Time at each waypoint.
    pub times: Vec<f64>,
    /// Acceleration at each waypoint.
    pub accelerations: Vec<f64>,
}

/// Compute the time-optimal parameterization of a geometric trajectory.
///
/// Given fixed waypoints and vehicle kinematic constraints, this function
/// computes a time-optimal velocity profile that respects all bounds.
///
/// # Arguments
/// * `waypoints` — sequence of poses defining the geometric path.
/// * `config` — kinematic constraints.
///
/// # Returns
/// A `VelocityProfile` with times, velocities, and accelerations at each waypoint.
pub fn compute_time_optimal_solution(
    waypoints: &[PoseSE2],
    config: &TimeParamConfig,
) -> VelocityProfile {
    let n = waypoints.len();
    assert!(n >= 2, "need at least 2 waypoints");

    // Forward pass
    let v_forward = forward_pass(waypoints, config);

    // Backward pass
    let v_backward = backward_pass(waypoints, config);

    // Merge forward and backward (take the minimum at each point)
    let mut velocities = Vec::with_capacity(n);
    for i in 0..n {
        velocities.push(v_forward[i].min(v_backward[i]));
    }

    // Smoothing pass
    let mut profile = VelocityProfile {
        velocities: vec![0.0; n],
        times: vec![0.0; n],
        accelerations: vec![0.0; n],
    };
    smoothing_pass(&mut profile, &velocities, config);
    profile
}

/// Forward pass: propagate velocity limits from start to end.
///
/// Starting from v=0, each waypoint's maximum velocity is limited by the
/// previous velocity plus the maximum acceleration over the segment time.
fn forward_pass(waypoints: &[PoseSE2], config: &TimeParamConfig) -> Vec<f64> {
    let n = waypoints.len();
    let mut v = vec![0.0; n];

    for i in 1..n {
        let dist = waypoints[i - 1].position_distance(&waypoints[i]);
        // Minimum time to traverse this segment given max acceleration
        let dt_min = compute_min_time(v[i - 1], config.max_vel, config.max_acc, dist);
        // Maximum velocity at waypoint i constrained by acceleration from i-1
        let v_max = compute_v_max_from_accel(v[i - 1], config.max_acc, dt_min, dist);
        v[i] = v_max.min(config.max_vel);
    }

    v
}

/// Backward pass: propagate velocity limits from end to start.
///
/// Starting from v=0 at the last waypoint, each waypoint's maximum velocity
/// is limited by the following velocity plus the deceleration capability.
fn backward_pass(waypoints: &[PoseSE2], config: &TimeParamConfig) -> Vec<f64> {
    let n = waypoints.len();
    let mut v = vec![0.0; n];

    for i in (0..n - 1).rev() {
        let dist = waypoints[i].position_distance(&waypoints[i + 1]);
        let dt_min = compute_min_time(v[i + 1], config.max_vel, config.max_acc, dist);
        let v_end = v[i + 1];
        let v_max = compute_v_max_from_decel(config.max_acc, dt_min, dist, v_end);
        v[i] = v_max.min(config.max_vel);
    }

    v
}

/// Smoothing pass: compute final times and accelerations, resolving oscillations.
///
/// Takes the merged velocity profile and converts it to a time-parameterized
/// trajectory with consistent accelerations.
fn smoothing_pass(
    profile: &mut VelocityProfile,
    velocities: &[f64],
    config: &TimeParamConfig,
) {
    let n = velocities.len();
    profile.velocities = velocities.to_vec();

    for i in 1..n {
        let dt = compute_segment_dt(&profile.velocities, config, i);
        let dt = dt.clamp(config.dt_min, config.dt_max);
        profile.times[i] = profile.times[i - 1] + dt;
        profile.accelerations[i] = (velocities[i] - velocities[i - 1]) / dt;
    }

    profile.accelerations[0] = 0.0;
}

/// Compute the minimum time to traverse a segment given start velocity,
/// max velocity, max acceleration, and segment distance.
#[inline]
fn compute_min_time(v_start: f64, v_max: f64, max_acc: f64, dist: f64) -> f64 {
    if max_acc < 1e-12 {
        return dist / (v_start + 1e-10);
    }
    if v_start >= v_max {
        return dist / v_start.max(1e-10);
    }
    let dist_acc = (v_max * v_max - v_start * v_start) / (2.0 * max_acc);
    if dist_acc >= dist {
        let v_end_sq = v_start * v_start + 2.0 * max_acc * dist;
        let v_end = v_end_sq.sqrt();
        return (v_end - v_start) / max_acc;
    }
    let t_acc = (v_max - v_start) / max_acc;
    let dist_coast = dist - dist_acc;
    t_acc + dist_coast / v_max
}

/// Compute the maximum velocity achievable at the end of a segment given
/// the start velocity, max acceleration, segment time, and distance.
#[inline]
fn compute_v_max_from_accel(v_start: f64, max_acc: f64, dt: f64, dist: f64) -> f64 {
    if dt < 1e-12 {
        return v_start;
    }
    let max_dist = v_start * dt + 0.5 * max_acc * dt * dt;
    if max_dist <= dist {
        (v_start + max_acc * dt).max(v_start)
    } else {
        let a = 2.0 * (dist - v_start * dt) / (dt * dt);
        v_start + a * dt
    }
}

/// Compute the maximum velocity at waypoint i given deceleration constraint
/// to waypoint i+1.
#[inline]
fn compute_v_max_from_decel(max_acc: f64, dt: f64, dist: f64, v_end: f64) -> f64 {
    if dt < 1e-12 { return v_end.max(0.0); }
    // v_i = v_end + 0.5*a*dt + dist/dt
    // Derived from: dist = v_i*dt + 0.5*(-a)*dt^2 + v_end*dt
    // Rearranged: v_i = (dist - v_end*dt + 0.5*a*dt^2) / dt
    //              = dist/dt - v_end + 0.5*a*dt
    // But we want v_max, so we compute the max start velocity that allows
    // decelerating to v_end over this segment.
    let v_start = dist / dt + v_end - 0.5 * max_acc * dt;
    v_start.max(v_end)
}

/// Compute the segment time step given velocities and constraints.
#[inline]
fn compute_segment_dt(velocities: &[f64], config: &TimeParamConfig, i: usize) -> f64 {
    let v_avg = compute_segment_average(
        velocities[i - 1],
        velocities[i],
        config.dt_ref,
        config.max_acc,
    );
    if v_avg.abs() < 1e-12 {
        return config.dt_ref;
    }
    v_avg
}

/// Average velocity over a segment, considering acceleration constraints.
///
/// Computes the effective average velocity accounting for the fact that
/// the robot may not be able to reach the theoretical average.
pub fn compute_segment_average(
    v_start: f64,
    v_end: f64,
    dt: f64,
    max_acc: f64,
) -> f64 {
    if dt.abs() < 1e-12 {
        return (v_start + v_end) / 2.0;
    }
    let v_avg = (v_start + v_end) / 2.0;
    let a_required = (v_end - v_start) / dt;
    if a_required.abs() <= max_acc {
        return v_avg;
    }
    if max_acc < 1e-12 {
        return v_avg;
    }
    let max_dist = v_start * dt + 0.5 * max_acc * dt.abs() * dt;
    max_dist / dt.abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_waypoints_straight() {
        let waypoints = vec![
            PoseSE2::from_xyz(0.0, 0.0, 0.0),
            PoseSE2::from_xyz(5.0, 0.0, 0.0),
        ];
        let config = TimeParamConfig {
            max_vel: 2.0,
            max_acc: 3.0,
            ..Default::default()
        };
        let profile = compute_time_optimal_solution(&waypoints, &config);
        assert_eq!(profile.velocities.len(), 2);
        assert_eq!(profile.times.len(), 2);
        assert_eq!(profile.accelerations.len(), 2);
        assert!(profile.times[1] > 0.0);
        assert!(profile.velocities[0] >= 0.0);
        assert!(profile.velocities[1] >= 0.0);
    }

    #[test]
    fn test_multiple_waypoints() {
        let waypoints = vec![
            PoseSE2::from_xyz(0.0, 0.0, 0.0),
            PoseSE2::from_xyz(2.0, 0.0, 0.0),
            PoseSE2::from_xyz(4.0, 0.0, 0.0),
            PoseSE2::from_xyz(6.0, 0.0, 0.0),
        ];
        let config = TimeParamConfig {
            max_vel: 3.0,
            max_acc: 5.0,
            ..Default::default()
        };
        let profile = compute_time_optimal_solution(&waypoints, &config);
        assert_eq!(profile.velocities.len(), 4);
        assert_eq!(profile.times.len(), 4);
        for i in 1..profile.times.len() {
            assert!(profile.times[i] > profile.times[i - 1], "times not increasing at {}", i);
        }
    }

    #[test]
    fn test_compute_segment_average_uniform_velocity() {
        let avg = compute_segment_average(2.0, 2.0, 0.1, 5.0);
        assert!((avg - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_segment_average_accelerating() {
        let avg = compute_segment_average(0.0, 4.0, 0.4, 5.0);
        assert!(avg < 2.0);
    }

    #[test]
    fn test_compute_v_max_from_accel_basic() {
        let v_max = compute_v_max_from_accel(0.0, 3.0, 1.0, 1.5);
        assert!((v_max - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_v_max_from_decel_basic() {
        let v_max = compute_v_max_from_decel(3.0, 1.0, 4.5, 0.0);
        assert!((v_max - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_forward_pass_zero_velocity() {
        let waypoints = vec![
            PoseSE2::from_xyz(0.0, 0.0, 0.0),
            PoseSE2::from_xyz(1.0, 0.0, 0.0),
            PoseSE2::from_xyz(2.0, 0.0, 0.0),
        ];
        let config = TimeParamConfig {
            max_vel: 2.0,
            max_acc: 1.0,
            ..Default::default()
        };
        let v = forward_pass(&waypoints, &config);
        assert!((v[0] - 0.0).abs() < 1e-10);
        for i in 1..v.len() {
            assert!(v[i] >= 0.0);
            assert!(v[i] <= config.max_vel);
        }
    }

    #[test]
    fn test_backward_pass_zero_at_end() {
        let waypoints = vec![
            PoseSE2::from_xyz(0.0, 0.0, 0.0),
            PoseSE2::from_xyz(1.0, 0.0, 0.0),
            PoseSE2::from_xyz(2.0, 0.0, 0.0),
        ];
        let config = TimeParamConfig {
            max_vel: 2.0,
            max_acc: 1.0,
            ..Default::default()
        };
        let v = backward_pass(&waypoints, &config);
        assert!((v[2] - 0.0).abs() < 1e-10);
        for i in 0..2 {
            assert!(v[i] >= 0.0);
        }
    }
}
#[cfg(test)]
mod adversarial_tests {
    use super::*;
    use crate::CubicSpline1D;
    
    #[test]
    fn test_cubic_spline_n1_panic() {
        let x: Vec<f64> = vec![0.0];
        let y: Vec<f64> = vec![1.0];
        let result = std::panic::catch_unwind(|| {
            CubicSpline1D::fit(x, y)
        });
        assert!(result.is_err(), "CubicSpline1D::fit MUST panic for n=1");
    }
    
    #[test]
    fn test_cubic_spline_nonmonotonic_x() {
        let x = vec![0.0, 2.0, 1.0];
        let y = vec![0.0, 1.0, 2.0];
        let _ = std::panic::catch_unwind(|| {
            CubicSpline1D::fit(x, y)
        });
    }
    
    #[test]
    fn test_cubic_spline_derivatives_at_boundaries() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 4.0, 9.0];
        let spline = CubicSpline1D::fit(x.clone(), y);
        let d0 = spline.eval_derivative(x[0]);
        let d2 = spline.eval_second_derivative(x[0]);
        assert!(d0.is_finite(), "derivative at x=0 finite, got {}", d0);
        assert!(d2.is_finite(), "2nd deriv at x=0 finite, got {}", d2);
    }
    
    #[test]
    fn test_cubic_spline_equal_adjacent_x() {
        let x = vec![0.0, 1.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 2.0, 3.0];
        let _ = std::panic::catch_unwind(|| {
            CubicSpline1D::fit(x, y)
        });
    }
    
    #[test]
    fn test_cubic_spline_constant_y() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![5.0, 5.0, 5.0, 5.0];
        let spline = CubicSpline1D::fit(x.clone(), y);
        for xi in &x {
            let v = spline.eval(*xi);
            assert!((v - 5.0).abs() < 1e-10,
                "constant y spline should be 5.0, got {} at x={}", v, xi);
        }
    }
    
    #[test]
    fn test_cubic_spline_large_y() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![1e15, 2e15, 3e15, 4e15];
        let spline = CubicSpline1D::fit(x.clone(), y);
        for xi in &x {
            let v = spline.eval(*xi);
            assert!(v.is_finite(), "large y eval should be finite, got {}", v);
        }
    }
    
    #[test]
    fn test_time_param_zero_distance() {
        let poses = vec![
            PoseSE2::from_xyz(0.0, 0.0, 0.0),
            PoseSE2::from_xyz(0.0, 0.0, 0.0),
            PoseSE2::from_xyz(1.0, 0.0, 0.0),
        ];
        let config = TimeParamConfig {
            max_vel: 10.0,
            max_acc: 5.0,
            ..Default::default()
        };
        let profile = compute_time_optimal_solution(&poses, &config);
        for v in &profile.velocities {
            assert!(v.is_finite(), "velocity finite zero dist, got {}", v);
            assert!(*v >= 0.0, "velocity non-neg, got {}", v);
        }
    }
    
    #[test]
    fn test_time_param_tiny_distance() {
        let poses = vec![
            PoseSE2::from_xyz(0.0, 0.0, 0.0),
            PoseSE2::from_xyz(1e-15, 0.0, 0.0),
            PoseSE2::from_xyz(1.0, 0.0, 0.0),
        ];
        let config = TimeParamConfig {
            max_vel: 10.0,
            max_acc: 5.0,
            ..Default::default()
        };
        let profile = compute_time_optimal_solution(&poses, &config);
        for v in &profile.velocities {
            assert!(v.is_finite(), "velocity finite tiny dist, got {}", v);
        }
    }
    
    #[test]
    fn test_time_param_zero_max_acc() {
        let poses = vec![
            PoseSE2::from_xyz(0.0, 0.0, 0.0),
            PoseSE2::from_xyz(1.0, 0.0, 0.0),
            PoseSE2::from_xyz(2.0, 0.0, 0.0),
        ];
        let config = TimeParamConfig {
            max_vel: 10.0,
            max_acc: 0.0,
            ..Default::default()
        };
        let profile = compute_time_optimal_solution(&poses, &config);
        for v in &profile.velocities {
            assert!(v.is_finite(), "velocity finite zero max_acc, got {}", v);
        }
    }
    
    #[test]
    fn test_time_param_backward_motion() {
        let poses = vec![
            PoseSE2::from_xyz(0.0, 0.0, 0.0),
            PoseSE2::from_xyz(-1.0, 0.0, 0.0),
            PoseSE2::from_xyz(-2.0, 0.0, 0.0),
        ];
        let config = TimeParamConfig {
            max_vel: 10.0,
            max_acc: 5.0,
            ..Default::default()
        };
        let profile = compute_time_optimal_solution(&poses, &config);
        for v in &profile.velocities {
            assert!(v.is_finite(), "speed finite backward, got {}", v);
            assert!(*v >= 0.0, "speed non-neg, got {}", v);
        }
    }
    
    #[test]
    fn test_time_param_all_zeros() {
        let poses = vec![
            PoseSE2::from_xyz(0.0, 0.0, 0.0),
            PoseSE2::from_xyz(1.0, 0.0, 0.0),
            PoseSE2::from_xyz(2.0, 0.0, 0.0),
        ];
        let config = TimeParamConfig {
            max_vel: 10.0,
            max_acc: 5.0,
            ..Default::default()
        };
        let profile = compute_time_optimal_solution(&poses, &config);
        assert!((profile.velocities[0] - 0.0).abs() < 1e-10,
            "first vel should be 0, got {}", profile.velocities[0]);
        assert!((profile.velocities[profile.velocities.len()-1] - 0.0).abs() < 1e-10,
            "last vel should be 0, got {}", profile.velocities[profile.velocities.len()-1]);
    }
    
    #[test]
    fn test_cubic_spline_eval_out_of_bounds() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 4.0, 9.0];
        let spline = CubicSpline1D::fit(x, y);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            spline.eval(5.0)
        }));
        assert!(result.is_err(), "eval(5.0) should panic");
    }
    
    // Bug: compute_min_time with v_start=v_max=0 and zero distance
    #[test]
    fn test_compute_min_time_zero_vel_zero_dist() {
        // v_start=0, v_max=0, dist=0: should return 0, not inf
        let t = compute_min_time(0.0, 0.0, 1.0, 0.0);
        assert!(t.is_finite(), "min_time for zero everything should be finite, got {}", t);
        assert!(t >= 0.0, "min_time should be non-negative, got {}", t);
    }
    
    // Bug: compute_segment_average with negative dt
    #[test]
    fn test_compute_segment_average_negative_dt() {
        // dt should always be positive, but what if it's not?
        let v_avg = compute_segment_average(1.0, 2.0, -0.1, 5.0);
        // This should still be finite and reasonable
        assert!(v_avg.is_finite(), "segment avg with negative dt should be finite, got {}", v_avg);
    }
    
    // Bug: compute_min_time with very large v_start and zero max_acc
    #[test]
    fn test_compute_min_time_zero_acc_large_vel() {
        let t = compute_min_time(10.0, 10.0, 0.0, 5.0);
        // With zero max_acc and v_start >= v_max, goes to line 167: dist / v_start.max(1e-10)
        assert!(t.is_finite(), "min_time with zero acc should be finite, got {}", t);
    }
    
    // Bug: compute_v_max_from_decel with very large v_end
    #[test]
    fn test_compute_v_max_from_decel_large_v_end() {
        // v_end much larger than what's physically possible
        let v = compute_v_max_from_decel(1.0, 0.1, 5.0, 100.0);
        // Should be at least v_end (the constraint v_start >= v_end)
        assert!(v.is_finite(), "v_max_from_decel should be finite, got {}", v);
        assert!(v >= 100.0 - 1e-10, "v_max_from_decel should be >= v_end, got {}", v);
    }
    
    // Bug: forward_pass with decreasing distances
    #[test]
    fn test_forward_pass_decreasing_dist() {
        let poses = vec![
            PoseSE2::from_xyz(0.0, 0.0, 0.0),
            PoseSE2::from_xyz(0.5, 0.0, 0.0),
            PoseSE2::from_xyz(0.6, 0.0, 0.0),
            PoseSE2::from_xyz(0.7, 0.0, 0.0),
        ];
        let config = TimeParamConfig {
            max_vel: 10.0,
            max_acc: 5.0,
            ..Default::default()
        };
        let profile = compute_time_optimal_solution(&poses, &config);
        // All velocities should be non-negative
        for v in &profile.velocities {
            assert!(*v >= 0.0, "velocity should be non-negative, got {}", v);
        }
    }
    
    // Bug: time parameterization with all waypoints at same position
    #[test]
    fn test_time_param_all_same_position() {
        let poses = vec![
            PoseSE2::from_xyz(1.0, 2.0, 0.0),
            PoseSE2::from_xyz(1.0, 2.0, 0.0),
            PoseSE2::from_xyz(1.0, 2.0, 0.0),
        ];
        let config = TimeParamConfig {
            max_vel: 10.0,
            max_acc: 5.0,
            ..Default::default()
        };
        let profile = compute_time_optimal_solution(&poses, &config);
        // All velocities should be zero
        for (i, v) in profile.velocities.iter().enumerate() {
            assert!(*v < 1e-10, "velocity at all-zero dist should be ~0, got {} at i={}", v, i);
        }
        // Times should be monotonically increasing
        for i in 1..profile.times.len() {
            assert!(profile.times[i] > profile.times[i-1],
                "times should strictly increase even with zero distance");
        }
    }
    
    // Note: single waypoint requires >= 2 waypoints (documented, panics with "need at least 2 waypoints")
}
