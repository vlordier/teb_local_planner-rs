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
    pub heading_score: f64,
    pub vel_score: f64,
}

pub struct DWAPlanner {
    pub config: DWAConfig,
    goal: Option<PoseSE2>,
}

impl DWAPlanner {
    pub fn new() -> Self {
        Self::new_with_config(DWAConfig::default())
    }

    pub fn new_with_config(config: DWAConfig) -> Self {
        Self { config, goal: None }
    }

    /// Set the current goal pose.
    pub fn set_goal(&mut self, goal: PoseSE2) {
        self.goal = Some(goal);
    }

    /// Check if a velocity command is dynamically feasible.
    fn is_feasible(&self, current_vx: f64, current_omega: f64,
                   new_vx: f64, new_omega: f64, dt: f64) -> bool {
        // Linear acceleration check
        let acc_x = (new_vx - current_vx).abs() / dt;
        let max_acc = self.config.acc_linear_max.abs().max(-self.config.acc_linear_min).max(0.01);
        if acc_x > max_acc {
            let stop_dist = current_vx * current_vx / (2.0 * max_acc);
            if stop_dist > self.config.window_duration * current_vx.abs() * 2.0 {
                return false;
            }
        }

        // Angular acceleration check
        let acc_theta = (new_omega - current_omega).abs() / dt;
        if acc_theta > self.config.acc_angular_max.abs() {
            return false;
        }

        // Velocity bounds check
        if new_vx < self.config.vel_min || new_vx > self.config.vel_max {
            return false;
        }
        if new_omega < self.config.yaw_rate_min || new_omega > self.config.yaw_rate_max {
            return false;
        }

        true
    }

    /// Evaluate a trajectory candidate with proper scoring.
    pub fn evaluate_trajectory(
        &self,
        pose: PoseSE2,
        vx: f64,
        _vy: f64,
        omega: f64,
        costmap: &Costmap,
        goal: &PoseSE2,
    ) -> Trajectory {
        let dt = self.config.dt_for_dist_calc;
        let t_samples = (self.config.window_duration / dt).max(1.0) as usize;

        let mut clear = true;
        let mut min_dist_to_goal = f64::INFINITY;
        let mut final_x = pose.x();
        let mut final_y = pose.y();
        let mut final_theta = pose.theta();

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

            // Track minimum distance to goal
            let dx = sim_x - goal.x();
            let dy = sim_y - goal.y();
            let dist_to_goal = (dx * dx + dy * dy).sqrt();
            if dist_to_goal < min_dist_to_goal {
                min_dist_to_goal = dist_to_goal;
            }

            // Track final state
            final_x = sim_x;
            final_y = sim_y;
            final_theta = pose.theta();
        }

        let near_goal = min_dist_to_goal < 2.0;

        // Heading score: alignment between final heading and goal direction
        let heading_target = (final_y - pose.y()).atan2(final_x - pose.x());
        let mut heading_diff = (heading_target - final_theta).abs();
        if heading_diff > std::f64::consts::PI {
            heading_diff = 2.0 * std::f64::consts::PI - heading_diff;
        }
        let heading_score = heading_diff / std::f64::consts::PI;

        // Velocity score: prefer higher velocities, normalized to [0, 1]
        let vel_score = if vx >= 0.0 {
            vx / self.config.vel_max.max(0.01)
        } else {
            -vx / (-self.config.vel_min).max(0.01)
        }.min(1.0);

        let cost = if !clear {
            f64::INFINITY
        } else {
            let dist_score = 1.0 / (1.0 + min_dist_to_goal);
            3.0 * dist_score + 1.0 * (1.0 - heading_score) + 1.0 * vel_score
        };

        Trajectory {
            vx,
            vy: 0.0,
            omega,
            cost,
            near_goal,
            clear,
            heading_score,
            vel_score,
        }
    }

    /// Plan a velocity command using DWA.
    pub fn plan(
        &mut self,
        pose: PoseSE2,
        goal: PoseSE2,
        costmap: &Costmap,
    ) -> Option<VelocityCommand> {
        self.set_goal(goal);

        let vx_samples = 8;
        let omega_samples = 8;
        let vx_step = (self.config.vel_max - self.config.vel_min) / vx_samples as f64;
        let omega_step = (self.config.yaw_rate_max - self.config.yaw_rate_min)
            / omega_samples as f64;

        let mut best_traj: Option<Trajectory> = None;
        let mut best_score = f64::INFINITY;

        for vi in 0..=vx_samples {
            for wi in 0..=omega_samples {
                let vx = self.config.vel_min + vi as f64 * vx_step;
                let omega = self.config.yaw_rate_min + wi as f64 * omega_step;

                if !self.is_feasible(vx, 0.0, vx, omega, self.config.dt_for_dist_calc) {
                    continue;
                }

                let traj = self.evaluate_trajectory(pose, vx, 0.0, omega, costmap, &goal);

                if !traj.clear {
                    continue;
                }

                if traj.cost < best_score {
                    best_score = traj.cost;
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
        let goal = PoseSE2::from_xyz(5.0, 0.0, 0.0);
        let costmap = Costmap(vec![0.0; 100 * 100]);
        let traj = planner.evaluate_trajectory(pose, 0.5, 0.0, 0.0, &costmap, &goal);
        assert!(traj.clear);
        assert!(traj.cost.is_finite());
        assert!(traj.heading_score >= 0.0 && traj.heading_score <= 1.0);
        assert!(traj.vel_score >= 0.0 && traj.vel_score <= 1.0);
    }
}
