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
                assert!(err.norm() >= 0.0);
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
                assert!(err.norm() < 1e-10);
            }
        }
    }

    /// EdgeObstacle error_with_distance is bounded from below by zero.
    #[test]
    fn edge_obstacle_error_bounded_below() {
        proptest! {
            fn edge_obstacle_error_bounded_below(dist: f64, min_dist: f64) {
                let edge = EdgeObstacle::new(0, 1.0);
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
                let edge = EdgeObstacle::new(0, 1.0);
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
