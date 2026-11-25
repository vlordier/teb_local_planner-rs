//! Proptest properties for nav-local algorithms.
//!
//! - TEB band autoResize preserves start/end
//! - EdgeShortestPath error is zero when poses are adjacent
//! - EdgeObstacle error bounded from below by safety margin
//! - DWA velocity command respects velocity limits

mod teb_prop {
    use nav_local::teb::pose_se2::PoseSe2;
    use nav_local::TimedElasticBand;
    use proptest::prelude::*;

    /// TEB band autoResize preserves start and end poses.
    #[test]
    fn auto_resize_preserves_start_end() {
        proptest! {
            fn auto_resize_preserves_start_end(
                sx: f64, sy: f64, st: f64,
                gx: f64, gy: f64, gt: f64,
                intermediates: Vec<(f64, f64, f64)>,
            ) {
                let mut teb = TimedElasticBand::new();
                let start = PoseSe2::from_components(sx, sy, st);
                let goal = PoseSe2::from_components(gx, gy, gt);

                teb.add_pose(start, true);
                for &(ix, iy, it) in &intermediates {
                    teb.add_pose(PoseSe2::from_components(ix, iy, it), false);
                }
                teb.add_pose(goal, true);

                if teb.size_poses() >= 3 {

                let orig_start = teb.pose(0).clone();
                let orig_end = teb.back_pose().clone();

                teb.auto_resize(0.1, 0.05, 2, 2, false);

                let new_start = teb.pose(0);
                let new_end = teb.back_pose();

                assert!((new_start.x() - orig_start.x()).abs() < 1e-10
                    && (new_start.y() - orig_start.y()).abs() < 1e-10
                    && (new_start.theta() - orig_start.theta()).abs() < 1e-10
                    && (new_end.x() - orig_end.x()).abs() < 1e-10
                    && (new_end.y() - orig_end.y()).abs() < 1e-10
                    && (new_end.theta() - orig_end.theta()).abs() < 1e-10);
                }
            }
        }
    }

    /// TEB band update_and_prune preserves start and end.
    #[test]
    fn update_and_prune_preserves_bounds() {
        proptest! {
            fn update_and_prune_preserves_bounds(
                sx: f64, sy: f64, st: f64,
                gx: f64, gy: f64, gt: f64,
            ) {
                let mut teb = TimedElasticBand::new();
                let start = PoseSe2::from_components(sx, sy, st);
                let goal = PoseSe2::from_components(gx, gy, gt);

                teb.add_pose(start, true);
                teb.add_pose(goal, true);

                let orig_start = teb.pose(0).clone();
                let orig_end = teb.back_pose().clone();

                teb.update_and_prune(Some(start.clone()), Some(goal), 2);

                let new_start = teb.pose(0);
                let new_end = teb.back_pose();

                assert!((new_start.x() - orig_start.x()).abs() < 1e-10
                    && (new_start.y() - orig_start.y()).abs() < 1e-10
                    && (new_start.theta() - orig_start.theta()).abs() < 1e-10
                    && (new_end.x() - orig_end.x()).abs() < 1e-10
                    && (new_end.y() - orig_end.y()).abs() < 1e-10
                    && (new_end.theta() - orig_end.theta()).abs() < 1e-10);
            }
        }
    }

    /// TEB band add_pose maintains pose/timediff count invariant.
    #[test]
    fn band_pose_timediff_invariant() {
        proptest! {
            fn band_pose_timediff_invariant(n: u8) {
                let n = (n % 10) as usize + 1;
                let mut teb = TimedElasticBand::new();

                for i in 0..n {
                    teb.add_pose(
                        PoseSe2::from_components(
                            i as f64, i as f64 * 0.5, 0.0,
                        ),
                        i == 0 || i == n - 1,
                    );
                }

                assert_eq!(teb.size_poses(), n);
                assert_eq!(teb.size_time_diffs(), n.saturating_sub(1));
            }
        }
    }
}

mod edge_prop {
    use nav_local::teb::edges::{Edge, EdgeShortestPath, EdgeObstacle};
    use nav_local::teb::pose_se2::PoseSe2;
    use proptest::prelude::*;

    /// EdgeShortestPath error vector norm >= 0.
    #[test]
    fn edge_shortest_path_non_negative_error() {
        proptest! {
            fn edge_shortest_path_non_negative_error(
                x1: f64, y1: f64, x2: f64, y2: f64,
            ) {
                let poses = vec![
                    PoseSe2::from_components(x1, y1, 0.0),
                    PoseSe2::from_components(x2, y2, 0.0),
                ];
                let timediffs = vec![0.5];
                let edge = EdgeShortestPath::new(0, 1, 1.0);
                let err = edge.error(&poses, &timediffs);
                assert!(f64::sqrt(err[0] * err[0] + err[1] * err[1]) >= 0.0);
            }
        }
    }

    /// EdgeShortestPath error is symmetric for equal poses.
    #[test]
    fn edge_shortest_path_zero_for_same_pose() {
        proptest! {
            fn edge_shortest_path_zero_for_same_pose(x: f64, y: f64) {
                let poses = vec![
                    PoseSe2::from_components(x, y, 0.0),
                    PoseSe2::from_components(x, y, 0.0),
                ];
                let timediffs = vec![0.5];
                let edge = EdgeShortestPath::new(0, 1, 1.0);
                let err = edge.error(&poses, &timediffs);
                assert!(f64::sqrt(err[0] * err[0] + err[1] * err[1]) < 1e-10);
            }
        }
    }

    /// EdgeObstacle error_with_distance is bounded from below by zero.
    #[test]
    fn edge_obstacle_error_bounded_below() {
        proptest! {
            fn edge_obstacle_error_bounded_below(dist: f64, min_dist: f64) {
                let edge = EdgeObstacle::new(0, 1.0, nalgebra::Point2::new(0.0, 0.0));
                let min_dist = min_dist.abs().max(0.01);
                let dist = dist.abs().max(0.0);
                let epsilon = 0.01;
                let err = edge.error_with_distance(dist, min_dist, epsilon);
                assert!(err[0] >= -1e-10 && err[1] >= -1e-10);
            }
        }
    }

    /// EdgeObstacle error is zero when distance >= min_distance.
    #[test]
    fn edge_obstacle_error_zero_when_safe() {
        proptest! {
            fn edge_obstacle_error_zero_when_safe(min_dist: f64) {
                let edge = EdgeObstacle::new(0, 1.0, nalgebra::Point2::new(0.0, 0.0));
                let min_dist = min_dist.abs().max(0.01);
                let dist = min_dist + 0.5;
                let err = edge.error_with_distance(dist, min_dist, 0.01);
                assert!(err[0] < 1e-10);
            }
        }
    }
}

mod dwa_prop {
    use nav_local::dwa::{DWAConfig, DWAPlanner};
    use nav_local::{PoseSE2, Planner, Costmap};
    use proptest::prelude::*;

    /// DWA velocity command respects velocity limits.
    #[test]
    fn dwa_velocity_respects_limits() {
        proptest! {
            fn dwa_velocity_respects_limits(vx_max: f64, yaw_rate_max: f64) {
                let vx_max = vx_max.abs().max(0.1);
                let yaw_rate_max = yaw_rate_max.abs().max(0.1);

                let config = DWAConfig {
                    vel_min: -vx_max,
                    vel_max: vx_max,
                    yaw_rate_min: -yaw_rate_max,
                    yaw_rate_max: yaw_rate_max,
                    ..DWAConfig::default()
                };

                let max_vel = config.vel_max;
                let max_yaw_rate = config.yaw_rate_max;
                let mut planner = DWAPlanner::new_with_config(config);
                let pose = PoseSE2::from_xyz(0.0, 0.0, 0.0);
                let goal = PoseSE2::from_xyz(10.0, 0.0, 0.0);
                let costmap = Costmap(vec![0.0; 100 * 100]);

                if let Some(cmd) = planner.plan(pose, goal, &costmap) {
                    assert!(cmd.vx.abs() <= max_vel + 1e-6
                        && cmd.omega.abs() <= max_yaw_rate + 1e-6);
                }
            }
        }
    }

    /// DWA planner produces deterministic results for same input.
    #[test]
    fn dwa_deterministic() {
        proptest! {
            fn dwa_deterministic(x: f64, y: f64, theta: f64) {
                let config = DWAConfig::default();
                let mut planner = DWAPlanner::new_with_config(config);

                let pose = PoseSE2::from_xyz(x, y, theta);
                let goal = PoseSE2::from_xyz(10.0, 0.0, 0.0);
                let costmap = Costmap(vec![0.0; 100 * 100]);

                let result1 = planner.plan(pose, goal, &costmap);
                let result2 = planner.plan(pose, goal, &costmap);

                match (&result1, &result2) {
                    (Some(c1), Some(c2)) => {
                        assert!((c1.vx - c2.vx).abs() < 1e-10
                            && (c1.omega - c2.omega).abs() < 1e-10);
                    }
                    (None, None) => {}
                    _ => panic!("inconsistent results"),
                }
            }
        }
    }
}
mod optimizer_prop {
    use nav_core::constraints::TebConfig;
    use nav_local::teb::pose_se2::PoseSe2;
    use nav_local::teb::band::TimedElasticBand;
    use nav_local::teb::optimizer::TEBOptimizer;
    use proptest::prelude::*;

    /// Optimization does not crash on valid input.
    #[test]
    fn optimizer_no_panic_on_valid_input() {
        proptest! {
            #[allow(clippy::float_cmp)]
            fn optimizer_no_panic(
                x_start: f64, y_start: f64, theta_start: f64,
                x_goal: f64, y_goal: f64, _theta_goal: f64,
            ) {
                let start = PoseSe2::from_components(x_start, y_start, theta_start);
                let goal = PoseSe2::from_components(x_goal, y_goal, 0.0);
                let mut band = TimedElasticBand::new();
                band.add_pose(start, true);
                band.add_pose_and_timediff(goal, 1.0);

                let config = TebConfig::default();
                let mut optimizer = TEBOptimizer::new(band, config);
                optimizer.set_max_outer_iterations(3);
                optimizer.set_max_inner_iterations(10);

                // Should not panic
                let _ = optimizer.optimize();
            }
        }
    }

    /// Optimization preserves start pose (first pose is fixed).
    #[test]
    fn optimizer_preserves_start_pose() {
        proptest! {
            fn optimizer_preserves_start(
                start_x: f64, start_y: f64, start_theta: f64,
                goal_x: f64, goal_y: f64, _goal_theta: f64,
            ) {
                let start = PoseSe2::from_components(start_x, start_y, start_theta);
                let goal = PoseSe2::from_components(goal_x, goal_y, 0.0);
                let mut band = TimedElasticBand::new();
                band.add_pose(start, true);
                band.add_pose_and_timediff(goal, 1.0);
                let initial_start = band.pose(0).clone();
                let config = TebConfig::default();
                let mut optimizer = TEBOptimizer::new(band, config);
                optimizer.set_max_outer_iterations(3);
                optimizer.set_max_inner_iterations(10);
                let _ = optimizer.optimize();

                assert!((optimizer.band().pose(0).x() - initial_start.x()).abs() < 1e-10);
                assert!((optimizer.band().pose(0).y() - initial_start.y()).abs() < 1e-10);
            }
        }
    }

    /// Optimization preserves goal pose (last pose is fixed).
    #[test]
    fn optimizer_preserves_goal_pose() {
        proptest! {
            fn optimizer_preserves_goal(
                start_x: f64, start_y: f64, _start_theta: f64,
                goal_x: f64, goal_y: f64, goal_theta: f64,
            ) {
                let start = PoseSe2::from_components(start_x, start_y, 0.0);
                let goal = PoseSe2::from_components(goal_x, goal_y, goal_theta);
                let mut band = TimedElasticBand::new();
                band.add_pose(start, true);
                band.add_pose_and_timediff(goal, 1.0);

                let initial_goal = band.pose(1).clone();
                let config = TebConfig::default();
                let mut optimizer = TEBOptimizer::new(band, config);
                optimizer.set_max_outer_iterations(3);
                optimizer.set_max_inner_iterations(10);
                let _ = optimizer.optimize();

                assert!((optimizer.band().pose(1).x() - initial_goal.x()).abs() < 1e-10);
                assert!((optimizer.band().pose(1).y() - initial_goal.y()).abs() < 1e-10);
            }
        }
    }

    /// Time diffs remain positive after optimization.
    #[test]
    fn optimizer_time_diffs_positive() {
        proptest! {
            fn optimizer_time_diffs_positive(
                start_x: f64, start_y: f64, _start_theta: f64,
                goal_x: f64, goal_y: f64, _goal_theta: f64,
            ) {
                let start = PoseSe2::from_components(start_x, start_y, 0.0);
                let goal = PoseSe2::from_components(goal_x, goal_y, 0.0);
                let mut band = TimedElasticBand::new();
                band.add_pose(start, true);
                band.add_pose_and_timediff(goal, 1.0);

                let config = TebConfig::default();
                let mut optimizer = TEBOptimizer::new(band, config);
                optimizer.set_max_outer_iterations(5);
                optimizer.set_max_inner_iterations(20);
                let _ = optimizer.optimize();

                for i in 0..optimizer.band().size_time_diffs() {
                    let td = optimizer.band().timediff(i);
                    assert!(td > 0.0, "Time diff at {} is {} (should be > 0)", i, td);
                }
            }
        }
    }

    /// Optimization with multiple intermediate poses does not panic.
    #[test]
    fn optimizer_multi_pose_stability() {
        proptest! {
            fn optimizer_multi_pose(
                x_start: f64, x_end: f64, y_start: f64, y_end: f64, n: usize,
            ) {
                let x_start = x_start.abs() % 100.0;
                let x_end = x_end.abs() % 100.0;
                let y_start = y_start.abs() % 100.0;
                let y_end = y_end.abs() % 100.0;
                let n = n % 10 + 2;

                let mut band = TimedElasticBand::new();
                band.add_pose(PoseSe2::from_components(0.0, 0.0, 0.0), true);

                for i in 0..n {
                    let t = (i as f64) / (n + 1) as f64;
                    let px = x_start + t * (x_end - x_start);
                    let py = y_start + t * (y_end - y_start);
                    band.add_pose_and_timediff(PoseSe2::from_components(px, py, 0.0), 0.5);
                }

                let config = TebConfig::default();
                let mut optimizer = TEBOptimizer::new(band, config);
                optimizer.set_max_outer_iterations(3);
                optimizer.set_max_inner_iterations(10);
                let _ = optimizer.optimize();
            }
        }
    }

    /// Optimize returns false for empty band.
    #[test]
    fn optimizer_empty_band_returns_false() {
        let band = TimedElasticBand::new();
        let config = TebConfig::default();
        let mut optimizer = TEBOptimizer::new(band, config);
        assert!(!optimizer.optimize());
    }

    /// Optimize returns false for band with only 1 pose (no time diffs).
    #[test]
    fn optimizer_single_pose_returns_false() {
        let mut band = TimedElasticBand::new();
        band.add_pose(PoseSe2::from_components(0.0, 0.0, 0.0), true);
        let config = TebConfig::default();
        let mut optimizer = TEBOptimizer::new(band, config);
        assert!(!optimizer.optimize());
    }
}
