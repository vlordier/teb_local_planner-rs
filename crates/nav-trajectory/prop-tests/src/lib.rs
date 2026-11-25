//! Proptest properties for nav-trajectory algorithms.
//!
//! - Dubins path length non-negative, segments valid
//! - Reeds-Shepp segments valid
//! - TrajectorySpline pose_at(t) interpolation
//! - Time parameterization velocity/acceleration constraints

mod dubins_prop {
    use nav_core::PoseSE2;
    use nav_trajectory::{dubins_path_from_poses, sample_dubins};
    use proptest::prelude::*;

    /// Dubins path length is always non-negative.
    #[test]
    fn dubins_path_length_non_negative() {
        proptest! {
            fn dubins_path_length_non_negative(
                p1_x: f64, p1_y: f64, p1_t: f64,
                p2_x: f64, p2_y: f64, p2_t: f64,
                radius: f64,
            ) {
                let radius = radius.abs().max(0.1);
                let p1 = PoseSE2::from_xyz(p1_x, p1_y, p1_t);
                let p2 = PoseSE2::from_xyz(p2_x, p2_y, p2_t);
                if let Some(path) = dubins_path_from_poses(&p1, &p2, radius) {
                    assert!(path.length >= 0.0);
                }
            }
        }
    }

    /// Every Dubins segment has non-negative length.
    #[test]
    fn dubins_segments_non_negative() {
        proptest! {
            fn dubins_segments_non_negative(
                p1_x: f64, p1_y: f64, p1_t: f64,
                p2_x: f64, p2_y: f64, p2_t: f64,
                radius: f64,
            ) {
                let radius = radius.abs().max(0.1);
                let p1 = PoseSE2::from_xyz(p1_x, p1_y, p1_t);
                let p2 = PoseSE2::from_xyz(p2_x, p2_y, p2_t);
                if let Some(path) = dubins_path_from_poses(&p1, &p2, radius) {
                    assert!(path.segments.iter().all(|s| s.length >= 0.0));
                }
            }
        }
    }

    /// Sampling at distance 0 returns start pose.
    #[test]
    fn dubins_sample_at_zero_returns_start() {
        proptest! {
            fn dubins_sample_at_zero_returns_start(
                p1_x: f64, p1_y: f64, p1_t: f64,
                p2_x: f64, p2_y: f64, p2_t: f64,
                radius: f64,
            ) {
                let radius = radius.abs().max(0.1);
                let p1 = PoseSE2::from_xyz(p1_x, p1_y, p1_t);
                let p2 = PoseSE2::from_xyz(p2_x, p2_y, p2_t);
                if let Some(path) = dubins_path_from_poses(&p1, &p2, radius) {
                    let sampled = sample_dubins(&path, &p1, 0.0);
                    assert!((sampled.x() - p1.x()).abs() < 1e-6
                        && (sampled.y() - p1.y()).abs() < 1e-6
                        && (sampled.theta() - p1.theta()).abs() < 1e-6);
                }
            }
        }
    }
}

mod reeds_shepp_prop {
    use nav_trajectory::reeds_shepp_path_from_poses;
    use proptest::prelude::*;

    /// Reeds-Shepp segment lengths are valid (non-negative magnitude).
    #[test]
    fn reeds_shepp_segments_valid() {
        proptest! {
            fn reeds_shepp_segments_valid(
                p1_x: f64, p1_y: f64, p1_t: f64,
                p2_x: f64, p2_y: f64, p2_t: f64,
                radius: f64,
            ) {
                let radius = radius.abs().max(0.1);
                if let Some(params) = reeds_shepp_path_from_poses(
                    p1_x, p1_y, p1_t, p2_x, p2_y, p2_t, radius,
                ) {
                    assert!(params.segments.iter().all(|s| s.length >= -1e-10));
                }
            }
        }
    }

    /// Reeds-Shepp total length is non-negative.
    #[test]
    fn reeds_shepp_total_length_non_negative() {
        proptest! {
            fn reeds_shepp_total_length_non_negative(
                p1_x: f64, p1_y: f64, p1_t: f64,
                p2_x: f64, p2_y: f64, p2_t: f64,
                radius: f64,
            ) {
                let radius = radius.abs().max(0.1);
                if let Some(params) = reeds_shepp_path_from_poses(
                    p1_x, p1_y, p1_t, p2_x, p2_y, p2_t, radius,
                ) {
                    assert!(params.total_length >= 0.0);
                }
            }
        }
    }
}

mod spline_prop {
    use nav_trajectory::TrajectorySpline;
    use proptest::prelude::*;

    /// TrajectorySpline::pose_at at t=0 returns start pose.
    #[test]
    fn trajectory_spline_pose_at_zero_returns_start() {
        proptest! {
            fn trajectory_spline_pose_at_zero_returns_start(
                px: Vec<f64>, py: Vec<f64>, pyaw: Vec<f64>,
            ) {
                let n = px.len();
                if n < 2 { return Ok(()); }
                let spline = TrajectorySpline::fit(
                    px[..n].to_vec(),
                    py[..n].to_vec(),
                    pyaw[..n].to_vec(),
                    (0..n).map(|i| i as f64).collect(),
                );
                let (x0, y0, yaw0) = spline.pose_at(0.0);
                assert!((x0 - px[0]).abs() < 1e-10
                    && (y0 - py[0]).abs() < 1e-10
                    && (yaw0 - pyaw[0]).abs() < 1e-10);
            }
        }
    }

    /// TrajectorySpline interpolates all data points.
    #[test]
    fn trajectory_spline_interpolates_all_points() {
        proptest! {
            fn trajectory_spline_interpolates_all_points(
                px: Vec<f64>, py: Vec<f64>, pyaw: Vec<f64>,
            ) {
                let n = px.len();
                if n < 2 { return Ok(()); }
                let spline = TrajectorySpline::fit(
                    px[..n].to_vec(),
                    py[..n].to_vec(),
                    pyaw[..n].to_vec(),
                    (0..n).map(|i| i as f64).collect(),
                );
                let tol = 1e-10;
                let passes = (0..n).all(|i| {
                    let (x, y, yaw) = spline.pose_at(i as f64);
                    (x - px[i]).abs() < tol && (y - py[i]).abs() < tol && (yaw - pyaw[i]).abs() < tol
                });
                assert!(passes);
            }
        }
    }
}

mod time_param_prop {
    use nav_core::PoseSE2;
    use nav_trajectory::{compute_time_optimal_solution, TimeParamConfig};
    use proptest::prelude::*;

    /// Time parameterization respects max velocity.
    #[test]
    fn time_param_respects_max_velocity() {
        proptest! {
            fn time_param_respects_max_velocity(
                waypoints: Vec<(f64, f64, f64)>,
            ) {
                let ws: Vec<PoseSE2> = waypoints.into_iter()
                    .map(|(x, y, t)| PoseSE2::from_xyz(x, y, t))
                    .collect();
                if ws.len() < 2 { return Ok(()); }
                let config = TimeParamConfig::default();
                let profile = compute_time_optimal_solution(&ws, &config);
                assert!(profile.velocities.iter().all(|&v| v.abs() <= config.max_vel + 1e-6));
            }
        }
    }

    /// Time parameterization respects max acceleration.
    #[test]
    fn time_param_respects_max_acceleration() {
        proptest! {
            fn time_param_respects_max_acceleration(
                waypoints: Vec<(f64, f64, f64)>,
            ) {
                let ws: Vec<PoseSE2> = waypoints.into_iter()
                    .map(|(x, y, t)| PoseSE2::from_xyz(x, y, t))
                    .collect();
                if ws.len() < 2 { return Ok(()); }
                let config = TimeParamConfig::default();
                let profile = compute_time_optimal_solution(&ws, &config);
                assert!(profile.accelerations.iter().all(|&a| a.abs() <= config.max_acc + 1e-6));
            }
        }
    }

    /// Time parameterization produces monotonically increasing times.
    #[test]
    fn time_param_times_monotonic() {
        proptest! {
            fn time_param_times_monotonic(
                waypoints: Vec<(f64, f64, f64)>,
            ) {
                let ws: Vec<PoseSE2> = waypoints.into_iter()
                    .map(|(x, y, t)| PoseSE2::from_xyz(x, y, t))
                    .collect();
                if ws.len() < 2 { return Ok(()); }
                let config = TimeParamConfig::default();
                let profile = compute_time_optimal_solution(&ws, &config);
                assert!(profile.times.windows(2).all(|w| w[0] <= w[1] + 1e-10));
            }
        }
    }
}
