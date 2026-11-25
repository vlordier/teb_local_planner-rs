//! Model Predictive Control (MPC) local planner.
//!
//! Solves a QP at each step: minimize
//!   sum_k ||x_k - x_ref||_Q + ||u_k||_R
//! subject to vehicle dynamics and constraints.

use crate::common::{Costmap, Planner, PoseSE2, VelocityCommand};

/// Vehicle kinematic constraints for MPC.
pub struct VehicleConstraints {
    /// Max linear velocity [m/s].
    pub max_vel_x: f64,
    /// Max angular velocity [rad/s].
    pub max_theta: f64,
    /// Min linear velocity [m/s] (backwards).
    pub min_vel_x: f64,
    /// Max linear acceleration [m/s^2].
    pub max_acc_x: f64,
    /// Max angular acceleration [rad/s^2].
    pub max_acc_theta: f64,
}

/// Configuration for the MPC planner.
pub struct MPCConfig {
    /// Prediction horizon (number of steps).
    pub prediction_horizon: usize,
    /// Discretization timestep [s].
    pub dt: f64,
    /// State cost weights: [x_pos, y_pos, theta].
    pub q: Vec<f64>,
    /// Control cost weights: [vx, omega].
    pub r: Vec<f64>,
    /// Vehicle kinematic constraints.
    pub constraints: VehicleConstraints,
}

impl Default for MPCConfig {
    fn default() -> Self {
        Self {
            prediction_horizon: 20,
            dt: 0.1,
            q: vec![1.0, 1.0, 0.5],
            r: vec![0.1, 0.1],
            constraints: VehicleConstraints {
                max_vel_x: 1.0,
                max_theta: 1.0,
                min_vel_x: -0.5,
                max_acc_x: 2.5,
                max_acc_theta: 3.5,
            },
        }
    }
}

/// Simple QP solver stub.
pub struct Solver {
    pub max_iters: usize,
    pub tol: f64,
}

impl Default for Solver {
    fn default() -> Self {
        Self {
            max_iters: 50,
            tol: 1e-6,
        }
    }
}

/// MPC planner stub.
pub struct MPCPlanner {
    pub config: MPCConfig,
    pub solver: Solver,
}

impl MPCPlanner {
    pub fn new(config: MPCConfig) -> Self {
        Self {
            config,
            solver: Solver::default(),
        }
    }

    /// Plan a velocity command using MPC.
    ///
    /// In a full implementation, this solves the QP problem.
    /// This stub returns a simple straight-line velocity command.
    pub fn plan(
        &mut self,
        pose: PoseSE2,
        goal: PoseSE2,
        _costmap: &Costmap,
    ) -> Option<VelocityCommand> {
        let dx = goal.x() - pose.x();
        let dy = goal.y() - pose.y();
        let dist = (dx * dx + dy * dy).sqrt();

        if dist < 0.05 {
            return Some(VelocityCommand::new(0.0, 0.0, 0.0));
        }

        // Simple proportional controller (stub for full QP)
        let vx = (self.config.constraints.max_vel_x * dist)
            .min(self.config.constraints.max_vel_x);
        let target_theta = dy.atan2(dx);
        let theta_err = self.normalize_angle(target_theta - pose.theta());
        let omega = 2.0 * theta_err;

        let vx = vx.max(self.config.constraints.min_vel_x)
            .min(self.config.constraints.max_vel_x);

        Some(VelocityCommand::new(vx, 0.0, omega))
    }

    fn normalize_angle(&self, angle: f64) -> f64 {
        let mut a = angle;
        while a > std::f64::consts::PI {
            a -= 2.0 * std::f64::consts::PI;
        }
        while a < -std::f64::consts::PI {
            a += 2.0 * std::f64::consts::PI;
        }
        a
    }
}

impl Planner for MPCPlanner {
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
        let planner = MPCPlanner::new(MPCConfig::default());
        assert_eq!(planner.config.prediction_horizon, 20);
    }

    #[test]
    fn test_plan_near_goal() {
        let mut planner = MPCPlanner::new(MPCConfig::default());
        let pose = PoseSE2::from_xyz(0.0, 0.0, 0.0);
        let goal = PoseSE2::from_xyz(0.01, 0.01, 0.0);
        let costmap = Costmap(vec![0.0; 10 * 10]);
        let cmd = planner.plan(pose, goal, &costmap);
        assert!(cmd.is_some());
    }
}
