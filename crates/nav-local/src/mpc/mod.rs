//! Model Predictive Control (MPC) local planner.
//!
//! Solves a QP at each step: minimize
//!   sum_k ||x_k - x_ref||_Q + ||u_k||_R
//! subject to vehicle dynamics and constraints.

use crate::common::{Costmap, Planner, PoseSE2, VelocityCommand};
use nalgebra::{DMatrix, DVector, LU};

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

    /// Plan a velocity command using MPC with QP optimization.
    pub fn plan(
        &mut self,
        pose: PoseSE2,
        goal: PoseSE2,
        _costmap: &Costmap,
    ) -> Option<VelocityCommand> {
        let N = self.config.prediction_horizon;
        let dt = self.config.dt;
        let max_vel = self.config.constraints.max_vel_x;
        let min_vel = self.config.constraints.min_vel_x;
        let max_omega = self.config.constraints.max_theta;
        let max_acc = self.config.constraints.max_acc_x;

        // Reference trajectory: straight line from current pose to goal
        let dx = goal.x() - pose.x();
        let dy = goal.y() - pose.y();
        let dist = (dx * dx + dy * dy).sqrt();

        if dist < 0.05 {
            return Some(VelocityCommand::new(0.0, 0.0, 0.0));
        }

        // Reference trajectory parameters
        let ref_vx = max_vel.min(dist / (N as f64 * dt));
        let ref_theta = dy.atan2(dx);

        // Build reference state trajectory
        let mut ref_states: Vec<[f64; 3]> = Vec::with_capacity(N);
        for k in 0..N {
            let t = (k as f64 + 1.0) * dt;
            ref_states.push([
                pose.x() + dx * (t / dist).min(1.0),
                pose.y() + dy * (t / dist).min(1.0),
                ref_theta,
            ]);
        }

        // Solve QP: min 0.5 * u^T H u + f^T u
        // Variables: u = [vx_0, omega_0, vx_1, omega_1, ..., vx_{N-1}, omega_{N-1}]
        let n_u = 2 * N;

        // Build Hessian H (diagonal from R weights)
        let mut H = DMatrix::zeros(n_u, n_u);
        for k in 0..N {
            let q0 = self.config.q[0] + self.config.q[1];  // position weight
            let q1 = self.config.q[2];  // theta weight
            // Approximate state cost contribution to control Hessian
            H[(2*k, 2*k)] = self.config.r[0] + q0 * dt * dt;
            H[(2*k+1, 2*k+1)] = self.config.r[1] + q1 * dt * dt;
        }

        // Build gradient f
        let mut f = DVector::zeros(n_u);
        for k in 0..N {
            let state = &ref_states[k];
            let pos_err = pose.x() - state[0];
            let vel_err = 0.0 - ref_vx;  // target velocity
            f[2*k] = (pos_err + vel_err) * self.config.r[0];
            f[2*k+1] = (pose.theta() - state[2]) * self.config.r[1];
        }

        // Solve using normal equations: (H + lambda*I) u = -f
        let lambda = 1e-6;
        for i in 0..n_u {
            H[(i, i)] += lambda;
        }

        let lu = nalgebra::LU::new(H);
        let solution = lu.solve(&f).unwrap_or(DVector::zeros(n_u));

        // Extract first control and apply bounds
        let vx = solution[0].clamp(min_vel, max_vel);
        let omega = solution[1].clamp(-max_omega, max_omega);

        // Limit acceleration from zero initial velocity
        let max_delta_v = max_acc * dt;
        let vx = (0.0 + max_delta_v).max(vx).min(0.0 + max_delta_v);

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
