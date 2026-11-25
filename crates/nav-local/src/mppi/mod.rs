//! Model Predictive Path Integral (MPPI) control.
//!
//! MPPI is a sampling-based MPC variant that uses particle filtering
//! over control sequences. At each step:
//! 1. Sample N control trajectories from a distribution
//! 2. Score each trajectory using a cost function
//! 3. Compute weighted average of control sequences
//! 4. Apply the first control action

use nalgebra::Vector2;

use crate::common::{Costmap, Planner, PoseSE2, VelocityCommand};

/// Configuration for the MPPI planner.
pub struct MPPIConfig {
    /// Number of particles (control trajectory samples).
    pub num_particles: usize,
    /// Temperature parameter for softmax weighting (lower = more greedy).
    pub temperature: f64,
    /// Prediction horizon (number of steps per particle).
    pub prediction_horizon: usize,
    /// Discretization timestep [s].
    pub dt: f64,
    /// Weight for control cost (regularization).
    pub control_cost_weight: f64,
    /// Standard deviation of the control noise distribution.
    pub control_std: Vector2<f64>, // [sigma_v, sigma_omega]
}

impl Default for MPPIConfig {
    fn default() -> Self {
        Self {
            num_particles: 200,
            temperature: 0.1,
            prediction_horizon: 20,
            dt: 0.1,
            control_cost_weight: 0.01,
            control_std: Vector2::new(0.2, 0.3),
        }
    }
}

/// A single control trajectory (sequence of [v, omega] commands).
#[derive(Debug, Clone)]
struct ControlTrajectory {
    controls: Vec<Vector2<f64>>,
    cost: f64,
}

/// MPPI planner.
pub struct MPPIPlanner {
    pub config: MPPIConfig,
    /// Previous control trajectory for perturbation (lambda effect).
    prev_controls: Vec<Vector2<f64>>,
}

impl MPPIPlanner {
    pub fn new() -> Self {
        Self::new_with_config(MPPIConfig::default())
    }

    pub fn new_with_config(config: MPPIConfig) -> Self {
        Self {
            config,
            prev_controls: Vec::new(),
        }
    }

    /// Sample a single control trajectory.
    fn sample_trajectory(&self, horizon: usize) -> Vec<Vector2<f64>> {
        let mut controls = Vec::with_capacity(horizon);
        for _ in 0..horizon {
            let v = self.sample_gaussian(0.0, self.config.control_std[0]);
            let omega = self.sample_gaussian(0.0, self.config.control_std[1]);
            controls.push(Vector2::new(v, omega));
        }
        controls
    }

    /// Sample from a 1D Gaussian (Box-Muller).
    fn sample_gaussian(&self, mean: f64, std: f64) -> f64 {
        let u1: f64 = rand::random::<f64>().max(1e-10);
        let u2: f64 = rand::random::<f64>();
        mean + std * (-2.0 * u1.ln()).sqrt() * u2.cos()
    }

    /// Evaluate the cost of a control trajectory.
    fn evaluate_trajectory(
        &self,
        pose: PoseSE2,
        controls: &[Vector2<f64>],
        goal: &PoseSE2,
        costmap: &Costmap,
    ) -> f64 {
        let mut cost = 0.0;
        let mut current = pose;

        for (k, ctrl) in controls.iter().enumerate() {
            let t = (k as f64 + 1.0) * self.config.dt;
            let sim_x = current.x() + ctrl[0] * t * current.theta().cos();
            let sim_y = current.y() + ctrl[0] * t * current.theta().sin();

            // Goal cost: distance to goal at final timestep
            if k == controls.len() - 1 {
                let goal_dist = ((sim_x - goal.x()).powi(2)
                    + (sim_y - goal.y()).powi(2))
                    .sqrt();
                cost += goal_dist;
            }

            // Obstacle cost: check costmap
            if let Some(obs_cost) = costmap.get_cost(sim_x, sim_y) {
                if obs_cost > 0.0 {
                    cost += obs_cost as f64 * 10.0;
                }
            }

            // Control cost: penalize large control effort
            cost += self.config.control_cost_weight
                * (ctrl[0].powi(2) + ctrl[1].powi(2));
        }

        cost
    }

    /// Plan a velocity command using MPPI.
    pub fn plan(
        &mut self,
        pose: PoseSE2,
        goal: PoseSE2,
        costmap: &Costmap,
    ) -> Option<VelocityCommand> {
        let N = self.config.num_particles;
        let horizon = self.config.prediction_horizon;

        // Sample particles
        let mut particles: Vec<ControlTrajectory> = Vec::with_capacity(N);
        for _ in 0..N {
            let controls = if self.prev_controls.is_empty()
                || self.config.temperature > 10.0
            {
                self.sample_trajectory(horizon)
            } else {
                // Perturb previous controls
                let mut perturbed = Vec::with_capacity(horizon);
                for ctrl in &self.prev_controls {
                    perturbed.push(Vector2::new(
                        ctrl[0] + self.sample_gaussian(0.0, self.config.control_std[0]),
                        ctrl[1] + self.sample_gaussian(0.0, self.config.control_std[1]),
                    ));
                }
                perturbed
            };

            let cost = self.evaluate_trajectory(pose, &controls, &goal, costmap);
            particles.push(ControlTrajectory { controls, cost });
        }

        // Compute softmax weights
        let min_cost = particles.iter().map(|p| p.cost).fold(f64::INFINITY, f64::min);
        let mut weights: Vec<f64> = particles
            .iter()
            .map(|p| ((min_cost - p.cost) / self.config.temperature).exp())
            .collect();

        let total_weight: f64 = weights.iter().sum();
        if total_weight < 1e-10 {
            weights.iter_mut().for_each(|w| *w = 1.0 / N as f64);
        } else {
            weights.iter_mut().for_each(|w| *w /= total_weight);
        }

        // Compute weighted average of first control action
        let mut avg_v = 0.0;
        let mut avg_omega = 0.0;
        for (i, particle) in particles.iter().enumerate() {
            if let Some(ctrl) = particle.controls.first() {
                avg_v += weights[i] * ctrl[0];
                avg_omega += weights[i] * ctrl[1];
            }
        }

        // Store best controls for next iteration
        let best_idx = particles
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.cost.partial_cmp(&b.1.cost).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.prev_controls = particles[best_idx].controls.clone();

        Some(VelocityCommand::new(avg_v, 0.0, avg_omega))
    }
}

impl Planner for MPPIPlanner {
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
        let planner = MPPIPlanner::new();
        assert_eq!(planner.config.num_particles, 200);
    }

    #[test]
    fn test_sample_trajectory() {
        let planner = MPPIPlanner::new();
        let traj = planner.sample_trajectory(5);
        assert_eq!(traj.len(), 5);
        for ctrl in traj {
            assert!(ctrl[0].abs() < 2.0);
            assert!(ctrl[1].abs() < 3.0);
        }
    }

    #[test]
    fn test_plan() {
        let mut planner = MPPIPlanner::new();
        let pose = PoseSE2::from_xyz(0.0, 0.0, 0.0);
        let goal = PoseSE2::from_xyz(1.0, 0.0, 0.0);
        let costmap = Costmap(vec![0.0; 10 * 10]);
        let cmd = planner.plan(pose, goal, &costmap);
        assert!(cmd.is_some());
    }
}
