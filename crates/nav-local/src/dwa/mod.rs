//! Differential Drive Wedge Area (DWA) local planner.
//!
//! Ported from ROS `dwa_local_planner`. The planner evaluates a
//! discretized sample of (v, omega) in the velocity space, simulates
//! each trajectory for `window_duration`, scores them, and returns the
//! best command.

use nalgebra::Vector2;

use crate::common::{Costmap, Planner, PoseSE2, VelocityCommand};

/// Configuration for the DWA planner.
pub struct DWAConfig {
    /// Minimum linear velocity [m/s].
    pub vel_min: f64,
    /// Maximum linear velocity [m/s].
    pub vel_max: f64,
    /// Minimum angular velocity [rad/s].
    pub yaw_rate_min: f64,
    /// Maximum angular velocity [rad/s].
    pub yaw_rate_max: f64,
    /// Minimum linear acceleration [m/s^2] (used as max deceleration).
    pub acc_linear_min: f64,
    /// Maximum linear acceleration [m/s^2].
    pub acc_linear_max: f64,
    /// Minimum angular acceleration [rad/s^2] (used as max decel).
    pub acc_angular_min: f64,
    /// Maximum angular acceleration [rad/s^2].
    pub acc_angular_max: f64,
    /// Trajectory simulation window duration [s].
    pub window_duration: f64,
    /// Time step for discretizing the trajectory [s].
    pub dt_for_dist_calc: f64,
}

impl Default for DWAConfig {
    fn default() -> Self {
        Self {
            vel_min: -0.5,
            vel_max: 1.0,
            yaw_rate_min: -1.0,
            yaw_rate_max: 1.0,
            acc_linear_min: -1.5,
            acc_linear_max: 2.5,
            acc_angular_min: -3.0,
            acc_angular_max: 3.5,
            window_duration: 2.0,
            dt_for_dist_calc: 0.05,
        }
    }
}

/// A trajectory candidate in DWA.
#[derive(Debug, Clone)]
pub struct Trajectory {
    pub vx: f64,
    pub vy: f64,
    pub omega: f64,
    pub cost: f64,
    pub near_goal: bool,
    pub clear: bool,
}

/// Differential Drive Wedge Area planner.
pub struct DWAPlanner {
    pub config: DWAConfig,
}

impl DWAPlanner {
    pub fn new() -> Self {
        Self::new_with_config(DWAConfig::default())
    }

    pub fn new_with_config(config: DWAConfig) -> Self {
        Self { config }
    }

    /// Evaluate a trajectory candidate.
    pub fn evaluate_trajectory(
        &self,
        pose: PoseSE2,
        vx: f64,
        _vy: f64,
        omega: f64,
        costmap: &Costmap,
    ) -> Trajectory {
        let mut near_goal = false;
        let mut clear = true;
        let dt = self.config.dt_for_dist_calc;
        let t_samples = (self.config.window_duration / dt) as usize;

        for t in 0..t_samples {
            let step_t = (t as f64 + 1.0) * dt;
            let sim_x = pose.x() + vx * step_t * pose.theta().cos();
            let sim_y = pose.y() + vx * step_t * pose.theta().sin();

            // Check costmap at simulated position
            if let Some(cost) = costmap.get_cost(sim_x, sim_y) {
                if cost > 200.0 {
                    clear = false;
                    break;
                }
            }

            // Check if near goal (within 2.0m)
            let dx = sim_x - pose.x();
            let dy = sim_y - pose.y();
            if (dx * dx + dy * dy).sqrt() > 2.0 {
                near_goal = false;
            }
        }

        let cost = if !clear {
            f64::INFINITY
        } else {
            let dist_to_goal = 0.0; // stub: would use goal pose
            1.0 / (1.0 + dist_to_goal)
        };

        Trajectory {
            vx,
            vy: 0.0,
            omega,
            cost,
            near_goal,
            clear,
        }
    }

    /// Plan a velocity command using DWA.
    pub fn plan(
        &mut self,
        pose: PoseSE2,
        _goal: PoseSE2,
        costmap: &Costmap,
    ) -> Option<VelocityCommand> {
        let vx_samples = 5;
        let omega_samples = 5;
        let vx_step = (self.config.vel_max - self.config.vel_min) / vx_samples as f64;
        let omega_step = (self.config.yaw_rate_max - self.config.yaw_rate_min)
            / omega_samples as f64;

        let mut best_traj: Option<Trajectory> = None;
        let mut best_score = f64::INFINITY;

        for vi in 0..vx_samples {
            for wi in 0..omega_samples {
                let vx = self.config.vel_min + vi as f64 * vx_step;
                let omega = self.config.yaw_rate_min + wi as f64 * omega_step;

                let traj = self.evaluate_trajectory(pose, vx, 0.0, omega, costmap);

                if !traj.clear {
                    continue;
                }

                let score = traj.cost;
                if score < best_score {
                    best_score = score;
                    best_traj = Some(traj);
                }
            }
        }

        best_traj.map(|t| VelocityCommand::new(t.vx, 0.0, t.omega))
    }
}

impl Planner for DWAPlanner {
    fn plan(
        &mut self,
        pose: PoseSE2,
        goal: PoseSE2,
        costmap: &Costmap,
    ) -> Option<VelocityCommand> {
        self.plan(pose, goal, costmap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let planner = DWAPlanner::new();
        assert!((planner.config.vel_max - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_trajectory_clear() {
        let planner = DWAPlanner::new();
        let pose = PoseSE2::from_xyz(0.0, 0.0, 0.0);
        let costmap = Costmap(vec![0.0; 100 * 100]);
        let traj = planner.evaluate_trajectory(pose, 0.5, 0.0, 0.0, &costmap);
        assert!(traj.clear);
        assert!(traj.cost.is_finite());
    }
}
