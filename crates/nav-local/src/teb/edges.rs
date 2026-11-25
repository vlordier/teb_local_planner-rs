//! Edge types and cost functions for TEB optimization.
//!
//! Ported from C++ g2o_types: edge_time_optimal.h, edge_velocity.h,
//! edge_obstacle.h, edge_kinematics.h, edge_prefer_rotdir.h,
//! edge_via_point.h, edge_shortest_path.h, edge_acceleration.h.
//!
//! Each edge implements the `Edge` trait: computing error, Jacobian,
//! weight matrix, and applying state updates.

use nalgebra::{Matrix2, Matrix3, Vector2, Vector3};

/// Robot kinematic model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KinematicModel {
    DiffDrive,
    Carlike,
}

/// Preferred rotation direction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RotDir {
    Left,
    Right,
    None,
}

/// Obstacle association method.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ObstacleAssociation {
    /// New method: nearest obstacle to each pose (left/right split).
    Modern,
    /// Legacy method: nearest pose to each obstacle.
    Legacy,
}

/// Configuration weights for optimization.
///
/// Mirrors TebConfig::optim from the C++ code.
#[derive(Debug, Clone)]
pub struct OptimConfig {
    /// Optimization activation flag.
    pub optimization_activate: bool,
    /// Print verbose output.
    pub optimization_verbose: bool,
    /// Epsilon for penalty functions (safety margin).
    pub penalty_epsilon: f64,
    /// Weight for max velocity (translational).
    pub weight_max_vel_x: f64,
    /// Weight for max velocity (strafing).
    pub weight_max_vel_y: f64,
    /// Weight for max angular velocity.
    pub weight_max_vel_theta: f64,
    /// Weight for acceleration limit (translational).
    pub weight_acc_lim_x: f64,
    /// Weight for acceleration limit (strafing).
    pub weight_acc_lim_y: f64,
    /// Weight for angular acceleration limit.
    pub weight_acc_lim_theta: f64,
    /// Weight for non-holonomic kinematics constraint.
    pub weight_kinematics_nh: f64,
    /// Weight for forward drive direction.
    pub weight_kinematics_forward_drive: f64,
    /// Weight for minimum turning radius.
    pub weight_kinematics_turning_radius: f64,
    /// Weight for time optimality.
    pub weight_optimaltime: f64,
    /// Weight for shortest path.
    pub weight_shortest_path: f64,
    /// Weight for obstacle avoidance.
    pub weight_obstacle: f64,
    /// Weight for inflation penalty.
    pub weight_inflation: f64,
    /// Weight for dynamic obstacle avoidance.
    pub weight_dynamic_obstacle: f64,
    /// Weight for dynamic obstacle inflation.
    pub weight_dynamic_obstacle_inflation: f64,
    /// Weight for velocity-obstacle ratio.
    pub weight_velocity_obstacle_ratio: f64,
    /// Weight for via-points.
    pub weight_viapoint: f64,
    /// Weight for prefer rotation direction.
    pub weight_prefer_rotdir: f64,
    /// Weight adaptation factor for iterative weight increase.
    pub weight_adapt_factor: f64,
    /// Obstacle cost exponent for nonlinear scaling.
    pub obstacle_cost_exponent: f64,
}

impl Default for OptimConfig {
    fn default() -> Self {
        Self {
            optimization_activate: true,
            optimization_verbose: false,
            penalty_epsilon: 0.01,
            weight_max_vel_x: 1.0,
            weight_max_vel_y: 0.0,
            weight_max_vel_theta: 1.0,
            weight_acc_lim_x: 1.0,
            weight_acc_lim_y: 0.0,
            weight_acc_lim_theta: 1.0,
            weight_kinematics_nh: 1000.0,
            weight_kinematics_forward_drive: 1.0,
            weight_kinematics_turning_radius: 1000.0,
            weight_optimaltime: 1.0,
            weight_shortest_path: 0.0,
            weight_obstacle: 1.0,
            weight_inflation: 0.1,
            weight_dynamic_obstacle: 1.0,
            weight_dynamic_obstacle_inflation: 0.1,
            weight_velocity_obstacle_ratio: 0.0,
            weight_viapoint: 1.0,
            weight_prefer_rotdir: 0.0,
            weight_adapt_factor: 2.0,
            obstacle_cost_exponent: 1.0,
        }
    }
}

/// Robot constraint parameters.
#[derive(Debug, Clone)]
pub struct RobotConfig {
    /// Max translational velocity [m/s].
    pub max_vel_x: f64,
    /// Max backwards translational velocity [m/s].
    pub max_vel_x_backwards: f64,
    /// Max strafing velocity (0 for non-holonomic) [m/s].
    pub max_vel_y: f64,
    /// Max translational velocity for omni robots [m/s].
    pub max_vel_trans: f64,
    /// Max angular velocity [rad/s].
    pub max_vel_theta: f64,
    /// Max linear acceleration [m/s^2].
    pub acc_lim_x: f64,
    /// Max strafing acceleration [m/s^2].
    pub acc_lim_y: f64,
    /// Max angular acceleration [rad/s^2].
    pub acc_lim_theta: f64,
    /// Minimum turning radius [m] (0 for diff-drive).
    pub min_turning_radius: f64,
    /// Use exact arc length for velocity computation.
    pub exact_arc_length: bool,
}

impl Default for RobotConfig {
    fn default() -> Self {
        Self {
            max_vel_x: 1.0,
            max_vel_x_backwards: 0.5,
            max_vel_y: 0.0,
            max_vel_trans: 0.0,
            max_vel_theta: 1.0,
            acc_lim_x: 2.5,
            acc_lim_y: 0.0,
            acc_lim_theta: 3.5,
            min_turning_radius: 0.0,
            exact_arc_length: false,
        }
    }
}

/// Obstacle-related configuration.
#[derive(Debug, Clone)]
pub struct ObstacleConfig {
    /// Minimum desired distance to obstacles [m].
    pub min_obstacle_dist: f64,
    /// Inflation distance around obstacles [m].
    pub inflation_dist: f64,
    /// Dynamic obstacle inflation distance [m].
    pub dynamic_obstacle_inflation_dist: f64,
    /// Include dynamic obstacles.
    pub include_dynamic_obstacles: bool,
    /// Legacy obstacle association strategy.
    pub legacy_obstacle_association: bool,
    /// Force inclusion factor for obstacle association.
    pub obstacle_association_force_inclusion_factor: f64,
    /// Cut-off factor for obstacle association.
    pub obstacle_association_cutoff_factor: f64,
}

impl Default for ObstacleConfig {
    fn default() -> Self {
        Self {
            min_obstacle_dist: 0.5,
            inflation_dist: 0.6,
            dynamic_obstacle_inflation_dist: 0.6,
            include_dynamic_obstacles: false,
            legacy_obstacle_association: false,
            obstacle_association_force_inclusion_factor: 2.0,
            obstacle_association_cutoff_factor: 5.0,
        }
    }
}

/// Trait for all TEB optimization edges.
///
/// Each edge computes an error vector (residuals) and a sparse Jacobian.
/// The optimization problem minimizes sum_i w_i * ||e_i||^2
/// where e_i is the error of edge i and w_i is the information matrix (weight).
pub trait Edge {
    /// Number of vertices this edge connects.
    fn num_vertices(&self) -> usize;

    /// Returns the indices of connected vertices in the trajectory.
    fn vertex_indices(&self) -> &[usize];

    /// Compute the error (residual) vector.
    fn error(&self, poses: &[crate::teb::pose_se2::PoseSe2], timediffs: &[f64]) -> [f64; 2];

    /// Compute the weighted error (chi^2 contribution).
    fn chi2(&self, poses: &[crate::teb::pose_se2::PoseSe2], timediffs: &[f64], weight: f64) -> f64 {
        let e = self.error(poses, timediffs);
        e[0] * e[0] + e[1] * e[1] * weight
    }

    /// Get the weight/information matrix diagonal.
    fn weight_diag(&self) -> &[f64];

    /// Check if edge is enabled.
    fn enabled(&self) -> bool;

    /// Set enabled flag.
    fn set_enabled(&mut self, _enabled: bool) {}
}

// ===== Edge types =====

/// Edge: Shortest path — error = |p_{i+1} - p_i|
#[derive(Debug, Clone)]
pub struct EdgeShortestPath {
    idx_prev: usize,
    idx_next: usize,
    pub weight: f64,
}

impl EdgeShortestPath {
    pub fn new(idx_prev: usize, idx_next: usize, weight: f64) -> Self {
        Self { idx_prev, idx_next, weight }
    }
}

impl Edge for EdgeShortestPath {
    fn num_vertices(&self) -> usize { 2 }
    fn vertex_indices(&self) -> &[usize] { &[] }
    fn error(&self, poses: &[crate::teb::pose_se2::PoseSe2], timediffs: &[f64]) -> [f64; 2] {
        let delta = poses[self.idx_next].position() - poses[self.idx_prev].position();
        [delta[0], delta[1]]
    }
    fn weight_diag(&self) -> &[f64] { &[] }
    fn enabled(&self) -> bool { self.weight > 0.0 }
}

/// Edge: Time optimal — error = dt_i
#[derive(Debug, Clone)]
pub struct EdgeTimeOptimal {
    idx_timediff: usize,
    pub weight: f64,
}

impl EdgeTimeOptimal {
    pub fn new(idx_timediff: usize, weight: f64) -> Self {
        Self { idx_timediff, weight }
    }
}

impl Edge for EdgeTimeOptimal {
    fn num_vertices(&self) -> usize { 1 }
    fn vertex_indices(&self) -> &[usize] { &[] }
    fn error(&self, _poses: &[crate::teb::pose_se2::PoseSe2], timediffs: &[f64]) -> [f64; 2] {
        [timediffs[self.idx_timediff], 0.0]
    }
    fn weight_diag(&self) -> &[f64] { &[] }
    fn enabled(&self) -> bool { self.weight > 0.0 }
}

/// Edge: Velocity — constrains v and omega
/// C++: EdgeVelocity (diff-drive) / EdgeVelocityHolonomic
#[derive(Debug, Clone)]
pub struct EdgeVelocity {
    idx_start: usize,
    idx_end: usize,
    idx_timediff: usize,
    pub weight_vx: f64,
    pub weight_omega: f64,
    max_vel_x: f64,
    max_vel_x_backwards: f64,
    max_vel_theta: f64,
    max_vel_y: f64,
    exact_arc_length: bool,
}

impl EdgeVelocity {
    pub fn new(
        idx_start: usize, idx_end: usize, idx_timediff: usize,
        max_vel_x: f64, max_vel_x_backwards: f64,
        max_vel_theta: f64, max_vel_y: f64,
        exact_arc_length: bool,
    ) -> Self {
        Self {
            idx_start, idx_end, idx_timediff,
            weight_vx: 0.0, weight_omega: 0.0,
            max_vel_x, max_vel_x_backwards,
            max_vel_theta, max_vel_y, exact_arc_length,
        }
    }

    pub fn set_weights(&mut self, vx: f64, omega: f64) {
        self.weight_vx = vx;
        self.weight_omega = omega;
    }
}

impl Edge for EdgeVelocity {
    fn num_vertices(&self) -> usize { 3 }
    fn vertex_indices(&self) -> &[usize] { &[] }

    fn error(&self, poses: &[crate::teb::pose_se2::PoseSe2], timediffs: &[f64]) -> [f64; 2] {
        let p1 = &poses[self.idx_start];
        let p2 = &poses[self.idx_end];
        let dt = timediffs[self.idx_timediff];
        if dt <= 0.0 {
            return [f64::MAX, f64::MAX];
        }

        let delta = p2.position() - p1.position();
        let dist = delta.norm();
        let angle_diff = crate::teb::math::normalize_theta(p2.theta() - p1.theta());

        let dist = if self.exact_arc_length && angle_diff.abs() > 1e-10 {
            let radius = dist / (2.0 * (angle_diff / 2.0).sin().abs());
            (angle_diff * radius).abs()
        } else {
            dist
        };

        let direction = crate::teb::math::fast_sigmoid(100.0 * (delta.x * p1.orientation_unit_vec().x + delta.y * p1.orientation_unit_vec().y));
        let vel = (dist / dt) * direction;
        let omega = angle_diff / dt;

        let v_err = if vel < -self.max_vel_x_backwards {
            (-vel - self.max_vel_x_backwards).abs()
        } else if vel <= self.max_vel_x {
            0.0
        } else {
            vel - self.max_vel_x
        };

        let w_err = crate::teb::math::penalty_bound_from_below(omega.abs(), self.max_vel_theta, 0.01);

        [v_err, w_err]
    }

    fn weight_diag(&self) -> &[f64] { &[] }
    fn enabled(&self) -> bool { self.weight_vx > 0.0 || self.weight_omega > 0.0 }
}

/// Edge: Acceleration — constrains linear and angular acceleration
/// a_lin = 2*(v2 - v1)/(dt1 + dt2)
#[derive(Debug, Clone)]
pub struct EdgeAcceleration {
    idx_0: usize, idx_1: usize, idx_2: usize,
    idx_dt_0: usize, idx_dt_1: usize,
    pub weight_lin: f64, pub weight_ang: f64,
    acc_lim_lin: f64, acc_lim_ang: f64,
    exact_arc_length: bool,
}

impl EdgeAcceleration {
    pub fn new(
        idx_0: usize, idx_1: usize, idx_2: usize,
        idx_dt_0: usize, idx_dt_1: usize,
        acc_lim_lin: f64, acc_lim_ang: f64,
        exact_arc_length: bool,
    ) -> Self {
        Self {
            idx_0, idx_1, idx_2, idx_dt_0, idx_dt_1,
            weight_lin: 0.0, weight_ang: 0.0,
            acc_lim_lin, acc_lim_ang, exact_arc_length,
        }
    }

    pub fn set_weights(&mut self, lin: f64, ang: f64) {
        self.weight_lin = lin;
        self.weight_ang = ang;
    }
}

impl Edge for EdgeAcceleration {
    fn num_vertices(&self) -> usize { 5 }
    fn vertex_indices(&self) -> &[usize] { &[] }

    fn error(&self, poses: &[crate::teb::pose_se2::PoseSe2], timediffs: &[f64]) -> [f64; 2] {
        let dt1 = timediffs[self.idx_dt_0];
        let dt2 = timediffs[self.idx_dt_1];
        if dt1 <= 0.0 || dt2 <= 0.0 {
            return [f64::MAX, f64::MAX];
        }

        let p0 = &poses[self.idx_0];
        let p1 = &poses[self.idx_1];
        let p2 = &poses[self.idx_2];

        let diff1 = p1.position() - p0.position();
        let diff2 = p2.position() - p1.position();
        let dist1 = diff1.norm();
        let dist2 = diff2.norm();
        let angle_diff1 = crate::teb::math::normalize_theta(p1.theta() - p0.theta());
        let angle_diff2 = crate::teb::math::normalize_theta(p2.theta() - p1.theta());

        let (dist1, dist2) = if self.exact_arc_length {
            let d1 = if angle_diff1.abs() > 1e-10 {
                let r = dist1 / (2.0 * (angle_diff1 / 2.0).sin().abs());
                (angle_diff1 * r).abs()
            } else { dist1 };
            let d2 = if angle_diff2.abs() > 1e-10 {
                let r = dist2 / (2.0 * (angle_diff2 / 2.0).sin().abs());
                (angle_diff2 * r).abs()
            } else { dist2 };
            (d1, d2)
        } else {
            (dist1, dist2)
        };

        let vel1 = dist1 / dt1 * crate::teb::math::fast_sigmoid(100.0 * (diff1.x * p0.orientation_unit_vec().x + diff1.y * p0.orientation_unit_vec().y));
        let vel2 = dist2 / dt2 * crate::teb::math::fast_sigmoid(100.0 * (diff2.x * p1.orientation_unit_vec().x + diff2.y * p1.orientation_unit_vec().y));
        let acc_lin = (vel2 - vel1) * 2.0 / (dt1 + dt2);
        let acc_ang = (angle_diff2 / dt2 - angle_diff1 / dt1) * 2.0 / (dt1 + dt2);

        let a_err = crate::teb::math::penalty_bound_to_interval_symmetric(acc_lin, self.acc_lim_lin, 0.01);
        let w_err = crate::teb::math::penalty_bound_to_interval_symmetric(acc_ang, self.acc_lim_ang, 0.01);

        [a_err, w_err]
    }

    fn weight_diag(&self) -> &[f64] { &[] }
    fn enabled(&self) -> bool { self.weight_lin > 0.0 || self.weight_ang > 0.0 }
}

/// Edge: Kinematics diff-drive — non-holonomic + forward drive
#[derive(Debug, Clone)]
pub struct EdgeKinematicsDiffDrive {
    idx_start: usize, idx_end: usize,
    pub weight_nh: f64, pub weight_fwd: f64,
}

impl EdgeKinematicsDiffDrive {
    pub fn new(idx_start: usize, idx_end: usize) -> Self {
        Self { idx_start, idx_end, weight_nh: 0.0, weight_fwd: 0.0 }
    }
    pub fn set_weights(&mut self, nh: f64, fwd: f64) {
        self.weight_nh = nh; self.weight_fwd = fwd;
    }
}

impl Edge for EdgeKinematicsDiffDrive {
    fn num_vertices(&self) -> usize { 2 }
    fn vertex_indices(&self) -> &[usize] { &[] }

    fn error(&self, poses: &[crate::teb::pose_se2::PoseSe2], timediffs: &[f64]) -> [f64; 2] {
        let p1 = &poses[self.idx_start];
        let p2 = &poses[self.idx_end];
        let delta = p2.position() - p1.position();
        let cos1 = p1.theta().cos();
        let cos2 = p2.theta().cos();
        let sin1 = p1.theta().sin();
        let sin2 = p2.theta().sin();

        let nh = ((cos1 + cos2) * delta.y - (sin1 + sin2) * delta.x).abs();
        let fwd = crate::teb::math::penalty_bound_from_below(
            delta.x * cos1 + delta.y * sin1, 0.0, 0.0);
        [nh, fwd]
    }

    fn weight_diag(&self) -> &[f64] { &[] }
    fn enabled(&self) -> bool { self.weight_nh > 0.0 || self.weight_fwd > 0.0 }
}

/// Edge: Kinematics carlike — non-holonomic + minimum turning radius
#[derive(Debug, Clone)]
pub struct EdgeKinematicsCarlike {
    idx_start: usize, idx_end: usize,
    pub weight_nh: f64, pub weight_turning: f64,
    min_turning_radius: f64, exact_arc_length: bool,
}

impl EdgeKinematicsCarlike {
    pub fn new(idx_start: usize, idx_end: usize, min_turning_radius: f64, exact_arc_length: bool) -> Self {
        Self { idx_start, idx_end, weight_nh: 0.0, weight_turning: 0.0, min_turning_radius, exact_arc_length }
    }
    pub fn set_weights(&mut self, nh: f64, turning: f64) {
        self.weight_nh = nh; self.weight_turning = turning;
    }
}

impl Edge for EdgeKinematicsCarlike {
    fn num_vertices(&self) -> usize { 2 }
    fn vertex_indices(&self) -> &[usize] { &[] }

    fn error(&self, poses: &[crate::teb::pose_se2::PoseSe2], timediffs: &[f64]) -> [f64; 2] {
        let p1 = &poses[self.idx_start];
        let p2 = &poses[self.idx_end];
        let delta = p2.position() - p1.position();

        let cos1 = p1.theta().cos();
        let cos2 = p2.theta().cos();
        let sin1 = p1.theta().sin();
        let sin2 = p2.theta().sin();
        let nh = ((cos1 + cos2) * delta.y - (sin1 + sin2) * delta.x).abs();

        let angle_diff = crate::teb::math::normalize_theta(p2.theta() - p1.theta());
        let radius = if angle_diff.abs() < 1e-10 {
            f64::MAX
        } else if self.exact_arc_length {
            let dist = delta.norm();
            dist / (2.0 * (angle_diff / 2.0).sin().abs())
        } else {
            delta.norm() / angle_diff.abs()
        };

        let turning = crate::teb::math::penalty_bound_from_below(radius.abs(), self.min_turning_radius, 0.0);
        [nh, turning]
    }

    fn weight_diag(&self) -> &[f64] { &[] }
    fn enabled(&self) -> bool { self.weight_nh > 0.0 || self.weight_turning > 0.0 }
}

/// Edge: Static obstacle — keeps distance from obstacles
#[derive(Debug, Clone)]
pub struct EdgeObstacle {
    idx_pose: usize,
    pub weight: f64,
    pub obstacle_pos: nalgebra::Point2<f64>,
}

impl EdgeObstacle {
    pub fn new(idx_pose: usize, weight: f64, obstacle_pos: nalgebra::Point2<f64>) -> Self {
        Self { idx_pose, weight, obstacle_pos }
    }
    pub fn error_with_distance(&self, dist: f64, min_dist: f64, epsilon: f64) -> [f64; 2] {
        let e = crate::teb::math::penalty_bound_from_below(dist, min_dist, epsilon);
        [e, 0.0]
    }
}

impl Edge for EdgeObstacle {
    fn num_vertices(&self) -> usize { 1 }
    fn vertex_indices(&self) -> &[usize] { &[] }
    fn error(&self, poses: &[crate::teb::pose_se2::PoseSe2], timediffs: &[f64]) -> [f64; 2] {
        let px = poses[self.idx_pose];
        let dx = px.x() - self.obstacle_pos.x;
        let dy = px.y() - self.obstacle_pos.y;
        let dist = (dx * dx + dy * dy).sqrt();
        [crate::teb::math::penalty_bound_from_below(dist, 0.25, 0.01), 0.0]
    }
    fn weight_diag(&self) -> &[f64] { &[] }
    fn enabled(&self) -> bool { self.weight > 0.0 }
}

/// Edge: Prefer rotation direction — penalizes specific turning
#[derive(Debug, Clone)]
pub struct EdgePreferRotDir {
    idx_start: usize, idx_end: usize,
    pub weight: f64,
    pub sign: i8, // +1=prefer left, -1=prefer right
}

impl EdgePreferRotDir {
    pub fn new(idx_start: usize, idx_end: usize, sign: i8) -> Self {
        Self { idx_start, idx_end, weight: 0.0, sign }
    }
    pub fn set_weight(&mut self, w: f64) { self.weight = w; }
}

impl Edge for EdgePreferRotDir {
    fn num_vertices(&self) -> usize { 2 }
    fn vertex_indices(&self) -> &[usize] { &[] }

    fn error(&self, poses: &[crate::teb::pose_se2::PoseSe2], timediffs: &[f64]) -> [f64; 2] {
        let angle_diff = crate::teb::math::normalize_theta(poses[self.idx_end].theta() - poses[self.idx_start].theta());
        let e = crate::teb::math::penalty_bound_from_below(self.sign as f64 * angle_diff, 0.0, 0.0);
        [e, 0.0]
    }

    fn weight_diag(&self) -> &[f64] { &[] }
    fn enabled(&self) -> bool { self.weight > 0.0 }
}

/// Edge: Via-point — pulls pose toward via-point
#[derive(Debug, Clone)]
pub struct EdgeViaPoint {
    idx_pose: usize,
    pub via_point: nalgebra::Point2<f64>,
    pub weight: f64,
}

impl EdgeViaPoint {
    pub fn new(idx_pose: usize, via_point: nalgebra::Point2<f64>) -> Self {
        Self { idx_pose, via_point, weight: 0.0 }
    }
    pub fn set_weight(&mut self, w: f64) { self.weight = w; }
}

impl Edge for EdgeViaPoint {
    fn num_vertices(&self) -> usize { 1 }
    fn vertex_indices(&self) -> &[usize] { &[] }

    fn error(&self, poses: &[crate::teb::pose_se2::PoseSe2], timediffs: &[f64]) -> [f64; 2] {
        let err: Vector2<f64> = poses[self.idx_pose].position() - self.via_point.coords;
        [err.norm(), 0.0]
    }

    fn weight_diag(&self) -> &[f64] { &[] }
    fn enabled(&self) -> bool { self.weight > 0.0 }
}

/// Edge: Dynamic obstacle — spatio-temporal distance
#[derive(Debug, Clone)]
pub struct EdgeDynamicObstacle {
    idx_pose: usize,
    pub weight: f64,
    pub t_: f64,
    pub obstacle_pos: nalgebra::Point2<f64>,
}

impl EdgeDynamicObstacle {
    pub fn new(idx_pose: usize, t: f64, obstacle_pos: nalgebra::Point2<f64>) -> Self {
        Self { idx_pose, weight: 0.0, t_: t, obstacle_pos }
    }
    pub fn set_weight(&mut self, w: f64) { self.weight = w; }
    pub fn set_time(&mut self, t: f64) { self.t_ = t; }
    pub fn error_with_distance(&self, dist: f64, min_dist: f64, epsilon: f64, inflation_dist: f64) -> [f64; 2] {
        let e0 = crate::teb::math::penalty_bound_from_below(dist, min_dist, epsilon);
        let e1 = crate::teb::math::penalty_bound_from_below(dist, inflation_dist, 0.0);
        [e0, e1]
    }
}

impl Edge for EdgeDynamicObstacle {
    fn num_vertices(&self) -> usize { 1 }
    fn vertex_indices(&self) -> &[usize] { &[] }
    fn error(&self, poses: &[crate::teb::pose_se2::PoseSe2], timediffs: &[f64]) -> [f64; 2] {
        let px = poses[self.idx_pose];
        let dx = px.x() - self.obstacle_pos.x;
        let dy = px.y() - self.obstacle_pos.y;
        let dist = (dx * dx + dy * dy).sqrt();
        [crate::teb::math::penalty_bound_from_below(dist, 0.25, 0.01),
         crate::teb::math::penalty_bound_from_below(dist, 0.8, 0.0)]
    }
    fn weight_diag(&self) -> &[f64] { &[] }
    fn enabled(&self) -> bool { self.weight > 0.0 }
}

/// Tagged edge enum — zero-allocation replacement for Box<dyn Edge>.
///
/// Each variant holds the edge data AND its parameter info.
/// The optimizer stores these in a flat Vec<Edge> for cache efficiency.
#[derive(Debug, Clone)]
pub enum EdgeEnum {
    TimeOptimal(EdgeTimeOptimalParam),
    ShortestPath(EdgeShortestPathParam),
    Velocity(EdgeVelocityParam),
    Acceleration(EdgeAccelerationParam),
    KinematicsDiffDrive(EdgeKinematicsDiffDriveParam),
    KinematicsCarlike(EdgeKinematicsCarlikeParam),
    Obstacle(EdgeObstacleParam),
    PreferRotDir(EdgePreferRotDirParam),
    ViaPoint(EdgeViaPointParam),
    DynamicObstacle(EdgeDynamicObstacleParam),
}

/// Param for EdgeTimeOptimal.
#[derive(Debug, Clone)]
pub struct EdgeTimeOptimalParam {
    pub idx_timediff: usize,
    pub weight: f64,
}

/// Param for EdgeShortestPath.
#[derive(Debug, Clone)]
pub struct EdgeShortestPathParam {
    pub idx_prev: usize,
    pub idx_next: usize,
    pub weight: f64,
}

/// Param for EdgeVelocity.
#[derive(Debug, Clone)]
pub struct EdgeVelocityParam {
    pub idx_start: usize,
    pub idx_end: usize,
    pub idx_timediff: usize,
    pub weight_vx: f64,
    pub weight_omega: f64,
    pub max_vel_x: f64,
    pub max_vel_x_backwards: f64,
    pub max_vel_theta: f64,
    pub max_vel_y: f64,
    pub exact_arc_length: bool,
}

/// Param for EdgeAcceleration.
#[derive(Debug, Clone)]
pub struct EdgeAccelerationParam {
    pub idx_0: usize,
    pub idx_1: usize,
    pub idx_2: usize,
    pub idx_dt_0: usize,
    pub idx_dt_1: usize,
    pub weight_lin: f64,
    pub weight_ang: f64,
    pub acc_lim_lin: f64,
    pub acc_lim_ang: f64,
    pub exact_arc_length: bool,
}

/// Param for EdgeKinematicsDiffDrive.
#[derive(Debug, Clone)]
pub struct EdgeKinematicsDiffDriveParam {
    pub idx_start: usize,
    pub idx_end: usize,
    pub weight_nh: f64,
    pub weight_fwd: f64,
}

/// Param for EdgeKinematicsCarlike.
#[derive(Debug, Clone)]
pub struct EdgeKinematicsCarlikeParam {
    pub idx_start: usize,
    pub idx_end: usize,
    pub weight_nh: f64,
    pub weight_turning: f64,
    pub min_turning_radius: f64,
    pub exact_arc_length: bool,
}

/// Param for EdgeObstacle.
#[derive(Debug, Clone)]
pub struct EdgeObstacleParam {
    pub idx_pose: usize,
    pub weight: f64,
    pub obstacle_pos: nalgebra::Point2<f64>,
}

/// Param for EdgePreferRotDir.
#[derive(Debug, Clone)]
pub struct EdgePreferRotDirParam {
    pub idx_start: usize,
    pub idx_end: usize,
    pub weight: f64,
    pub sign: i8,
}

/// Param for EdgeViaPoint.
#[derive(Debug, Clone)]
pub struct EdgeViaPointParam {
    pub idx_pose: usize,
    pub via_point: nalgebra::Point2<f64>,
    pub weight: f64,
}

/// Param for EdgeDynamicObstacle.
#[derive(Debug, Clone)]
pub struct EdgeDynamicObstacleParam {
    pub idx_pose: usize,
    pub weight: f64,
    pub pos_x: f64,
    pub pos_y: f64,
    pub heading: f64,
    pub speed: f64,
    pub radius: f64,
    pub max_speed: f64,
    pub inflation_dist: f64,
}

impl EdgeTimeOptimalParam {
    pub fn new(idx_timediff: usize, weight: f64) -> Self {
        Self { idx_timediff, weight }
    }
}

impl EdgeShortestPathParam {
    pub fn new(idx_prev: usize, idx_next: usize, weight: f64) -> Self {
        Self { idx_prev, idx_next, weight }
    }
}

impl EdgeVelocityParam {
    pub fn new(idx_start: usize, idx_end: usize, idx_timediff: usize,
               max_vel_x: f64, max_vel_x_backwards: f64, max_vel_theta: f64,
               max_vel_y: f64, exact_arc_length: bool) -> Self {
        Self { idx_start, idx_end, idx_timediff,
               weight_vx: 0.0, weight_omega: 0.0,
               max_vel_x, max_vel_x_backwards, max_vel_theta, max_vel_y, exact_arc_length }
    }
    pub fn set_weights(&mut self, vx: f64, omega: f64) {
        self.weight_vx = vx;
        self.weight_omega = omega;
    }
}

impl EdgeAccelerationParam {
    pub fn new(idx_0: usize, idx_1: usize, idx_2: usize,
               idx_dt_0: usize, idx_dt_1: usize,
               acc_lim_lin: f64, acc_lim_ang: f64, exact_arc_length: bool) -> Self {
        Self { idx_0, idx_1, idx_2, idx_dt_0, idx_dt_1,
               weight_lin: 0.0, weight_ang: 0.0,
               acc_lim_lin, acc_lim_ang, exact_arc_length }
    }
    pub fn set_weights(&mut self, lin: f64, ang: f64) {
        self.weight_lin = lin;
        self.weight_ang = ang;
    }
}

impl EdgeKinematicsDiffDriveParam {
    pub fn new(idx_start: usize, idx_end: usize) -> Self {
        Self { idx_start, idx_end, weight_nh: 0.0, weight_fwd: 0.0 }
    }
    pub fn set_weights(&mut self, nh: f64, fwd: f64) {
        self.weight_nh = nh;
        self.weight_fwd = fwd;
    }
}

impl EdgeKinematicsCarlikeParam {
    pub fn new(idx_start: usize, idx_end: usize, min_turning_radius: f64, exact_arc_length: bool) -> Self {
        Self { idx_start, idx_end, weight_nh: 0.0, weight_turning: 0.0,
               min_turning_radius, exact_arc_length }
    }
    pub fn set_weights(&mut self, nh: f64, turning: f64) {
        self.weight_nh = nh;
        self.weight_turning = turning;
    }
}

impl EdgeObstacleParam {
    pub fn new(idx_pose: usize, weight: f64, obstacle_pos: nalgebra::Point2<f64>) -> Self {
        Self { idx_pose, weight, obstacle_pos }
    }
}

impl EdgePreferRotDirParam {
    pub fn new(idx_start: usize, idx_end: usize, weight: f64, sign: i8) -> Self {
        Self { idx_start, idx_end, weight, sign }
    }
}

impl EdgeViaPointParam {
    pub fn new(idx_pose: usize, via_point: nalgebra::Point2<f64>, weight: f64) -> Self {
        Self { idx_pose, via_point, weight }
    }
}

impl EdgeDynamicObstacleParam {
    pub fn new(idx_pose: usize, pos_x: f64, pos_y: f64, heading: f64,
               speed: f64, radius: f64, max_speed: f64, inflation_dist: f64) -> Self {
        Self { idx_pose, weight: 0.0, pos_x, pos_y, heading, speed, radius, max_speed, inflation_dist }
    }
}

/// Convert Edge to &dyn Edge methods via trait impl.
impl EdgeEnum {
    /// Number of vertices this edge connects.
    #[inline]
    pub fn num_vertices(&self) -> usize {
        match self {
            EdgeEnum::TimeOptimal(_) => 1,
            EdgeEnum::ShortestPath(_) => 2,
            EdgeEnum::Velocity(_) => 2,
            EdgeEnum::Acceleration(_) => 3,
            EdgeEnum::KinematicsDiffDrive(_) => 2,
            EdgeEnum::KinematicsCarlike(_) => 2,
            EdgeEnum::Obstacle(_) => 1,
            EdgeEnum::PreferRotDir(_) => 2,
            EdgeEnum::ViaPoint(_) => 1,
            EdgeEnum::DynamicObstacle(_) => 1,
        }
    }

    pub fn error(&self, poses: &[crate::teb::pose_se2::PoseSe2], timediffs: &[f64]) -> [f64; 2] {
        match self {
            EdgeEnum::TimeOptimal(p) => {
                if p.weight <= 0.0 { return [0.0, 0.0]; }
                [timediffs[p.idx_timediff], 0.0]
            }
            EdgeEnum::ShortestPath(p) => {
                if p.weight <= 0.0 { return [0.0, 0.0]; }
                let delta = poses[p.idx_next].position() - poses[p.idx_prev].position();
                [delta[0], delta[1]]
            }
            EdgeEnum::Velocity(p) => {
                if p.weight_vx <= 0.0 && p.weight_omega <= 0.0 { return [0.0, 0.0]; }
                let p1 = &poses[p.idx_start];
                let p2 = &poses[p.idx_end];
                let dt = timediffs[p.idx_timediff];
                if dt <= 0.0 {
                    return [f64::MAX, f64::MAX];
                }
                let delta = p2.position() - p1.position();
                let dist = delta.norm();
                let angle_diff = crate::teb::math::normalize_theta(p2.theta() - p1.theta());
                let dist = if p.exact_arc_length && angle_diff.abs() > 1e-10 {
                    let radius = dist / (2.0 * (angle_diff / 2.0).sin().abs());
                    (angle_diff * radius).abs()
                } else {
                    dist
                };
                let direction = crate::teb::math::fast_sigmoid(100.0 * (delta.x * p1.orientation_unit_vec().x + delta.y * p1.orientation_unit_vec().y));
                let vel = (dist / dt) * direction;
                let omega = angle_diff / dt;
                let v_err = if vel < -p.max_vel_x_backwards {
                    (-vel - p.max_vel_x_backwards).abs()
                } else if vel <= p.max_vel_x {
                    0.0
                } else {
                    vel - p.max_vel_x
                };
                let w_err = crate::teb::math::penalty_bound_from_below(omega.abs(), p.max_vel_theta, 0.01);
                [v_err, w_err]
            }
            EdgeEnum::Acceleration(p) => {
                if p.weight_lin <= 0.0 && p.weight_ang <= 0.0 { return [0.0, 0.0]; }
                let dt1 = timediffs[p.idx_dt_0];
                let dt2 = timediffs[p.idx_dt_1];
                if dt1 <= 0.0 || dt2 <= 0.0 {
                    return [f64::MAX, f64::MAX];
                }
                let p0 = &poses[p.idx_0];
                let p1 = &poses[p.idx_1];
                let p2 = &poses[p.idx_2];
                let diff1 = p1.position() - p0.position();
                let diff2 = p2.position() - p1.position();
                let dist1 = diff1.norm();
                let dist2 = diff2.norm();
                let angle_diff1 = crate::teb::math::normalize_theta(p1.theta() - p0.theta());
                let angle_diff2 = crate::teb::math::normalize_theta(p2.theta() - p1.theta());
                let (dist1, dist2) = if p.exact_arc_length {
                    let d1 = if angle_diff1.abs() > 1e-10 {
                        let r = dist1 / (2.0 * (angle_diff1 / 2.0).sin().abs());
                        (angle_diff1 * r).abs()
                    } else { dist1 };
                    let d2 = if angle_diff2.abs() > 1e-10 {
                        let r = dist2 / (2.0 * (angle_diff2 / 2.0).sin().abs());
                        (angle_diff2 * r).abs()
                    } else { dist2 };
                    (d1, d2)
                } else {
                    (dist1, dist2)
                };
                let vel1 = dist1 / dt1 * crate::teb::math::fast_sigmoid(100.0 * (diff1.x * p0.orientation_unit_vec().x + diff1.y * p0.orientation_unit_vec().y));
                let vel2 = dist2 / dt2 * crate::teb::math::fast_sigmoid(100.0 * (diff2.x * p1.orientation_unit_vec().x + diff2.y * p1.orientation_unit_vec().y));
                let acc_lin = (vel2 - vel1) * 2.0 / (dt1 + dt2);
                let acc_ang = (angle_diff2 / dt2 - angle_diff1 / dt1) * 2.0 / (dt1 + dt2);
                let a_err = crate::teb::math::penalty_bound_to_interval_symmetric(acc_lin, p.acc_lim_lin, 0.01);
                let w_err = crate::teb::math::penalty_bound_to_interval_symmetric(acc_ang, p.acc_lim_ang, 0.01);
                [a_err, w_err]
            }
            EdgeEnum::KinematicsDiffDrive(p) => {
                if p.weight_nh <= 0.0 && p.weight_fwd <= 0.0 { return [0.0, 0.0]; }
                let p1 = &poses[p.idx_start];
                let p2 = &poses[p.idx_end];
                let delta = p2.position() - p1.position();
                let cos1 = p1.theta().cos();
                let sin1 = p1.theta().sin();
                let nh = ((cos1 + cos1) * delta.y - (sin1 + sin1) * delta.x).abs();
                let fwd = crate::teb::math::penalty_bound_from_below(delta.x * cos1 + delta.y * sin1, 0.0, 0.0);
                [nh, fwd]
            }
            EdgeEnum::KinematicsCarlike(p) => {
                if p.weight_nh <= 0.0 && p.weight_turning <= 0.0 { return [0.0, 0.0]; }
                let p1 = &poses[p.idx_start];
                let p2 = &poses[p.idx_end];
                let delta = p2.position() - p1.position();
                let cos1 = p1.theta().cos();
                let sin1 = p1.theta().sin();
                let nh = ((cos1 + cos1) * delta.y - (sin1 + sin1) * delta.x).abs();
                let angle_diff = crate::teb::math::normalize_theta(p2.theta() - p1.theta());
                let radius = if angle_diff.abs() < 1e-10 {
                    f64::MAX
                } else if p.exact_arc_length {
                    let dist = delta.norm();
                    dist / (2.0 * (angle_diff / 2.0).sin().abs())
                } else {
                    delta.norm() / angle_diff.abs()
                };
                let turning = crate::teb::math::penalty_bound_from_below(radius.abs(), p.min_turning_radius, 0.0);
                [nh, turning]
            }
            EdgeEnum::Obstacle(p) => {
                if p.weight <= 0.0 { return [0.0, 0.0]; }
                [0.0, 0.0]
            }
            EdgeEnum::PreferRotDir(p) => {
                if p.weight <= 0.0 { return [0.0, 0.0]; }
                let angle_diff = crate::teb::math::normalize_theta(poses[p.idx_end].theta() - poses[p.idx_start].theta());
                let e = crate::teb::math::penalty_bound_from_below(p.sign as f64 * angle_diff, 0.0, 0.0);
                [e, 0.0]
            }
            EdgeEnum::ViaPoint(p) => {
                if p.weight <= 0.0 { return [0.0, 0.0]; }
                let err: nalgebra::Vector2<f64> = poses[p.idx_pose].position() - p.via_point.coords;
                [err.norm(), 0.0]
            }
            EdgeEnum::DynamicObstacle(p) => {
                if p.weight <= 0.0 { return [0.0, 0.0]; }
                [0.0, 0.0]
            }
        }
    }

    /// Compute chi^2 (weighted squared error).
    #[inline]
    pub fn chi2(&self, poses: &[crate::teb::pose_se2::PoseSe2], timediffs: &[f64], weight: f64) -> f64 {
        let e = self.error(poses, timediffs);
        (e[0] * e[0] + e[1] * e[1]) * weight
    }

    /// Weight diagonal.
    #[inline]
    pub fn weight_diag(&self) -> &'static [f64] {
        &[]
    }

    /// Check if edge is enabled.
    #[inline]
    pub fn enabled(&self) -> bool {
        match self {
            EdgeEnum::TimeOptimal(p) => p.weight > 0.0,
            EdgeEnum::ShortestPath(p) => p.weight > 0.0,
            EdgeEnum::Velocity(p) => p.weight_vx > 0.0 || p.weight_omega > 0.0,
            EdgeEnum::Acceleration(p) => p.weight_lin > 0.0 || p.weight_ang > 0.0,
            EdgeEnum::KinematicsDiffDrive(p) => p.weight_nh > 0.0 || p.weight_fwd > 0.0,
            EdgeEnum::KinematicsCarlike(p) => p.weight_nh > 0.0 || p.weight_turning > 0.0,
            EdgeEnum::Obstacle(p) => p.weight > 0.0,
            EdgeEnum::PreferRotDir(p) => p.weight > 0.0,
            EdgeEnum::ViaPoint(p) => p.weight > 0.0,
            EdgeEnum::DynamicObstacle(p) => p.weight > 0.0,
        }
    }
    /// Compute analytical Jacobian columns for edges where derivatives are known.
    #[inline]
    pub fn compute_jac_into(&self, poses: &[crate::teb::pose_se2::PoseSe2],
                            timediffs: &[f64], jac: &mut [f64], n_params: usize,
                            n_residuals: usize) -> bool {
        match self {
            EdgeEnum::TimeOptimal(p) => {
                if p.weight <= 0.0 { return true; }
                if p.idx_timediff < n_params {
                    jac[p.idx_timediff * n_residuals] = 1.0;
                }
                true
            }
            EdgeEnum::ShortestPath(p) => {
                if p.idx_prev > 0 && p.idx_prev < poses.len() - 1 {
                    let fpi = p.idx_prev - 1;
                    jac[fpi * 2 * n_residuals] = -1.0;
                    jac[(fpi * 2 + 1) * n_residuals + 1] = -1.0;
                }
                if p.idx_next > 0 && p.idx_next < poses.len() - 1 {
                    let fpn = p.idx_next - 1;
                    jac[fpn * 2 * n_residuals] = 1.0;
                    jac[(fpn * 2 + 1) * n_residuals + 1] = 1.0;
                }
                true
            }
            EdgeEnum::Velocity(p) => {
                if p.weight_vx <= 0.0 && p.weight_omega <= 0.0 { return true; }

                let p1 = &poses[p.idx_start];
                let p2 = &poses[p.idx_end];
                let dt = timediffs[p.idx_timediff];
                if dt <= 1e-10 { return true; } // singular, Jacobian already zero

                // Pre-compute geometry
                let dx = p2.x() - p1.x();
                let dy = p2.y() - p1.y();
                let dist_sq = dx * dx + dy * dy;
                let dist = dist_sq.sqrt();
                if dist < 1e-12 { return true; } // degenerate

                let cos1 = p1.theta().cos();
                let sin1 = p1.theta().sin();
                let dot = dx * cos1 + dy * sin1;
                let dot_scaled = 100.0 * dot;
                let direction = crate::teb::math::fast_sigmoid(dot_scaled);

                // d(dist)/d(p)
                let dx_dist = dx / dist; // d(dist)/d(dx)
                let dy_dist = dy / dist; // d(dist)/d(dy)

                // d(direction)/d(p) = sigmoid'(arg) * d(arg)/d(p)
                // sigmoid(x) = x/(1+|x|), derivative = 1/(1+|x|)^2
                // fast_sigmoid(x) ≈ sigmoid(x/100) * 100, so d/dx = 0.1 * sigmoid'(x/100)
                let ddir_darg = if dot_scaled.abs() > 100.0 {
                    0.0
                } else {
                    0.1 / (1.0 + (dot).abs()).powi(2)
                };

                // d(dist)/d(dt) = -dist/dt² * direction → dvel/dt
                // dvel/dx = (d(dist)/dx * direction + dist * d(direction)/dx) / dt
                // dvel/dy = (d(dist)/dy * direction + dist * d(direction)/dy) / dt

                let dvel_dx1 = -(dx_dist * direction + dist * ddir_darg * cos1) / dt;
                let dvel_dy1 = -(dy_dist * direction + dist * ddir_darg * sin1) / dt;
                let dvel_dx2 = dx_dist * direction / dt;
                let dvel_dy2 = dy_dist * direction / dt;
                let dvel_dt = -(dist * direction) / (dt * dt);

                // d(omega)/d(p) = d/dp(angle_diff/dt) = d(angle_diff)/dt / dt
                let angle_diff = crate::teb::math::normalize_theta(p2.theta() - p1.theta());
                // d(angle_diff)/d(x1) = 0, d(angle_diff)/d(y1) = 0, d(angle_diff)/d(x2) = 0, d(angle_diff)/d(y2) = 0
                // d(angle_diff)/d(theta1) = -1, d(angle_diff)/d(theta2) = 1
                // But theta is NOT a free parameter — it doesn't appear in the parameter vector
                // So omega derivatives w.r.t. pose x/y are 0
                let domega_dpos = 0.0; // omega only depends on theta diff and dt

                // Now apply penalty derivatives
                let vel = (dist / dt) * direction;

                // dv_err/dvel
                let dv_err_dvel = if vel > p.max_vel_x {
                    1.0
                } else if vel < -p.max_vel_x_backwards {
                    -1.0
                } else {
                    0.0
                };

                // dw_err/domega = sign(omega) * penalty_bound_from_below'(omega)
                // penalty_bound_from_below(|omega|, max, 0.01):
                //   if |omega| >= max * 1.01: |omega| - max * 1.01 → d/d|omega| = 1
                //   if |omega| <= max: 0
                //   else: quadratic
                let omega = angle_diff / dt;
                let abs_omega = omega.abs();
                let dw_err_dabs_omega = if abs_omega >= p.max_vel_theta * 1.01 {
                    1.0
                } else if abs_omega <= p.max_vel_theta {
                    0.0
                } else {
                    let t = (abs_omega - p.max_vel_theta) / (p.max_vel_theta * 0.01);
                    t * t
                };
                let dw_err_domega = if omega.abs() < 1e-15 {
                    0.0
                } else {
                    dw_err_dabs_omega * omega.signum()
                };

                // Fill Jacobian columns for x1, y1, x2, y2, dt
                // Map pose indices to global parameter indices
                if p.idx_start > 0 && p.idx_start < poses.len() - 1 {
                    let fpi = p.idx_start - 1; // free pose index
                    let col_x1 = fpi * 2;
                    let col_y1 = fpi * 2 + 1;
                    if col_x1 < n_params { jac[col_x1 * n_residuals + 0] = dv_err_dvel * dvel_dx1; }
                    if col_y1 < n_params { jac[col_y1 * n_residuals + 0] = dv_err_dvel * dvel_dy1; }
                    // omega column: w_err[1] doesn't depend on x1 or y1
                    if col_x1 < n_params { jac[col_x1 * n_residuals + 1] = dw_err_domega * domega_dpos; }
                    if col_y1 < n_params { jac[col_y1 * n_residuals + 1] = dw_err_domega * domega_dpos; }
                }
                if p.idx_end > 0 && p.idx_end < poses.len() - 1 {
                    let fpn = p.idx_end - 1;
                    let col_x2 = fpn * 2;
                    let col_y2 = fpn * 2 + 1;
                    if col_x2 < n_params { jac[col_x2 * n_residuals + 0] = dv_err_dvel * dvel_dx2; }
                    if col_y2 < n_params { jac[col_y2 * n_residuals + 0] = dv_err_dvel * dvel_dy2; }
                    if col_x2 < n_params { jac[col_x2 * n_residuals + 1] = dw_err_domega * domega_dpos; }
                    if col_y2 < n_params { jac[col_y2 * n_residuals + 1] = dw_err_domega * domega_dpos; }
                }
                // dt column
                if p.idx_timediff + (if p.idx_end == 0 && p.idx_start == 0 { 0 } else { 0 }) < n_params {
                    // time diff is at global index td_offset + idx_timediff
                    // We don't have td_offset here. Let's use a different approach:
                    // The time diff index in free_pose mapping:
                    // free_pose for pose i (i in 1..n-1) is at free_pose[i-1]
                    // time diffs start at td_offset = 2*(n-3) (for n poses, n-3 free poses, each with 2 params)
                    // Actually, we don't know td_offset. Let me compute it:
                    let n_poses = poses.len();
                    let n_free_poses = if n_poses > 2 { n_poses - 3 } else { 0 };
                    let td_offset = 2 * n_free_poses;
                    let global_td_idx = td_offset + p.idx_timediff;
                    if global_td_idx < n_params {
                        jac[global_td_idx * n_residuals + 0] = dv_err_dvel * dvel_dt;
                        jac[global_td_idx * n_residuals + 1] = dw_err_domega * dw_err_dabs_omega / dt.abs().max(1e-10);
                    }
                }
                true
            }
            EdgeEnum::Acceleration(p) => {
                if p.weight_lin <= 0.0 && p.weight_ang <= 0.0 { return true; }
                let dt1 = timediffs[p.idx_dt_0];
                let dt2 = timediffs[p.idx_dt_1];
                let dt_sum = dt1 + dt2;
                if dt1 <= 1e-10 || dt2 <= 1e-10 { return true; }
                let p0 = &poses[p.idx_0];
                let p1 = &poses[p.idx_1];
                let p2 = &poses[p.idx_2];
                // Compute v1, v2 via same formulas as error()
                let diff1_x = p1.x() - p0.x();
                let diff1_y = p1.y() - p0.y();
                let dist1 = (diff1_x * diff1_x + diff1_y * diff1_y).sqrt();
                let angle_diff1 = crate::teb::math::normalize_theta(p1.theta() - p0.theta());
                let sig1 = crate::teb::math::fast_sigmoid(100.0 * (diff1_x * p0.theta().cos() + diff1_y * p0.theta().sin()));
                let vel1 = dist1 / dt1 * sig1;
                let omega1 = angle_diff1 / dt1;
                let diff2_x = p2.x() - p1.x();
                let diff2_y = p2.y() - p1.y();
                let dist2 = (diff2_x * diff2_x + diff2_y * diff2_y).sqrt();
                let angle_diff2 = crate::teb::math::normalize_theta(p2.theta() - p1.theta());
                let sig2 = crate::teb::math::fast_sigmoid(100.0 * (diff2_x * p1.theta().cos() + diff2_y * p1.theta().sin()));
                let vel2 = dist2 / dt2 * sig2;
                let omega2 = angle_diff2 / dt2;
                let acc_lin = (vel2 - vel1) * 2.0 / dt_sum;
                // d(a_lin)/d(pos) = 2/(dt1+dt2) * d(vel2-vel1)/d(pos)
                let da_lin = 2.0 / dt_sum;
                // Penalty derivative: penalty_bound_to_interval_symmetric_deriv
                let abs_a = acc_lin.abs();
                let d_a_err = if abs_a > p.acc_lim_lin * 1.01 {
                    acc_lin.signum()
                } else if abs_a < p.acc_lim_lin {
                    0.0
                } else {
                    let t = (abs_a - p.acc_lim_lin) / (p.acc_lim_lin * 0.01);
                    t * t * acc_lin.signum()
                };
                // Map to Jacobian columns
                // dvel1/d(x0) = (-diff1_x/dist1 * sig1 + dist1 * dsig1/...) / dt1
                // dvel1/d(y0) similarly
                // dvel1/d(x1) = (diff1_x/dist1 * sig1) / dt1
                // dvel1/d(y1) = (diff1_y/dist1 * sig1) / dt1
                // dvel2/d(x1) = (-diff2_x/dist2 * sig2) / dt2
                // dvel2/d(y1) = (-diff2_y/dist2 * sig2) / dt2
                // dvel2/d(x2) = (diff2_x/dist2 * sig2) / dt2
                // dvel2/d(y2) = (diff2_y/dist2 * sig2) / dt2
                // dvel1/d(x0) = (-diff1_x/dist1 * sig1) / dt1
                let dvel1_dx0 = (-diff1_x / dist1.max(1e-12)) * sig1 / dt1;
                let dvel1_dy0 = (-diff1_y / dist1.max(1e-12)) * sig1 / dt1;
                let dvel1_dx1 = (diff1_x / dist1.max(1e-12)) * sig1 / dt1;
                let dvel1_dy1 = (diff1_y / dist1.max(1e-12)) * sig1 / dt1;
                let dvel2_dx1 = (-diff2_x / dist2.max(1e-12)) * sig2 / dt2;
                let dvel2_dy1 = (-diff2_y / dist2.max(1e-12)) * sig2 / dt2;
                let dvel2_dx2 = (diff2_x / dist2.max(1e-12)) * sig2 / dt2;
                let dvel2_dy2 = (diff2_y / dist2.max(1e-12)) * sig2 / dt2;
                if p.idx_0 > 0 && p.idx_0 < poses.len() - 1 {
                    let f = p.idx_0 - 1;
                    let cx = f * 2;
                    if cx < n_params { jac[cx * n_residuals + 0] = da_lin * d_a_err * dvel1_dx0; }
                    if cx + 1 < n_params { jac[(cx + 1) * n_residuals + 0] = da_lin * d_a_err * dvel1_dy0; }
                }
                if p.idx_1 > 0 && p.idx_1 < poses.len() - 1 {
                    let f = p.idx_1 - 1;
                    let cx = f * 2;
                    if cx < n_params { jac[cx * n_residuals + 0] = da_lin * d_a_err * (dvel1_dx1 + dvel2_dx1); }
                    if cx + 1 < n_params { jac[(cx + 1) * n_residuals + 0] = da_lin * d_a_err * (dvel1_dy1 + dvel2_dy1); }
                }
                if p.idx_2 > 0 && p.idx_2 < poses.len() - 1 {
                    let f = p.idx_2 - 1;
                    let cx = f * 2;
                    if cx < n_params { jac[cx * n_residuals + 0] = da_lin * d_a_err * dvel2_dx2; }
                    if cx + 1 < n_params { jac[(cx + 1) * n_residuals + 0] = da_lin * d_a_err * dvel2_dy2; }
                }
                let n_free = if poses.len() > 2 { poses.len() - 3 } else { 0 };
                let td_off = 2 * n_free;
                let gi0 = td_off + p.idx_dt_0;
                if gi0 < n_params { jac[gi0 * n_residuals + 0] = da_lin * d_a_err * (-2.0 * vel1 / (dt1 * dt1)); }
                let gi1 = td_off + p.idx_dt_1;
                if gi1 < n_params { jac[gi1 * n_residuals + 0] = da_lin * d_a_err * (-2.0 * vel2 / (dt2 * dt2)); }
                true
            }
            EdgeEnum::KinematicsDiffDrive(p) => {
                if p.weight_nh <= 0.0 && p.weight_fwd <= 0.0 { return true; }
                self._kinematics_jac_into(p.idx_start, p.idx_end, poses, n_params, n_residuals, jac);
                true
            }
            EdgeEnum::KinematicsCarlike(p) => {
                if p.weight_nh <= 0.0 && p.weight_turning <= 0.0 { return true; }
                self._kinematics_jac_into(p.idx_start, p.idx_end, poses, n_params, n_residuals, jac);
                true
            }
            EdgeEnum::Obstacle(p) => {
                if p.weight <= 0.0 { return true; }
                let n_free = if poses.len() > 2 { poses.len() - 3 } else { 0 };
                if p.idx_pose < n_free {
                    let px = poses[p.idx_pose];
                    let dx = px.x() - p.obstacle_pos.x;
                    let dy = px.y() - p.obstacle_pos.y;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist < 1e-12 { return true; }
                    let n_res = 2;
                    let dp_ddist = crate::teb::math::penalty_bound_from_below_deriv(dist, 0.25, 0.01);
                    let d_dx = dp_ddist * dx / dist;
                    let d_dy = dp_ddist * dy / dist;
                    let cx = p.idx_pose * 2;
                    let cy = cx + 1;
                    if cx < n_params { jac[cx * n_res + 0] = d_dx; }
                    if cy < n_params { jac[cy * n_res + 0] = d_dy; }
                }
                true
            }
            EdgeEnum::PreferRotDir(_) => {
                // Error only depends on angle diff, not on x/y — skip
                true
            }
            EdgeEnum::ViaPoint(p) => {
                if p.weight <= 0.0 { return true; }
                let n_free = if poses.len() > 2 { poses.len() - 3 } else { 0 };
                if p.idx_pose < n_free {
                    let px = poses[p.idx_pose];
                    let dx = px.x() - p.via_point.x;
                    let dy = px.y() - p.via_point.y;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist < 1e-12 { return true; }
                    let n_res = 2;
                    let d_dx = dx / dist;
                    let d_dy = dy / dist;
                    let cx = p.idx_pose * 2;
                    let cy = cx + 1;
                    if cx < n_params { jac[cx * n_res + 0] = d_dx; }
                    if cy < n_params { jac[cy * n_res + 0] = d_dy; }
                }
                true
            }
            EdgeEnum::DynamicObstacle(p) => {
                if p.weight <= 0.0 { return true; }
                let n_free = if poses.len() > 2 { poses.len() - 3 } else { 0 };
                if p.idx_pose < n_free {
                    let px = poses[p.idx_pose];
                    let dx = px.x() - p.pos_x;
                    let dy = px.y() - p.pos_y;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist < 1e-12 { return true; }
                    let n_res = 2;
                    let dp_ddist1 = crate::teb::math::penalty_bound_from_below_deriv(dist, 0.25, 0.01);
                    let dp_ddist2 = crate::teb::math::penalty_bound_from_below_deriv(dist, p.inflation_dist, 0.0);
                    let d_dx1 = dp_ddist1 * dx / dist;
                    let d_dy1 = dp_ddist1 * dy / dist;
                    let d_dx2 = dp_ddist2 * dx / dist;
                    let d_dy2 = dp_ddist2 * dy / dist;
                    let cx = p.idx_pose * 2;
                    let cy = cx + 1;
                    if cx < n_params { jac[cx * n_res + 0] = d_dx1; }
                    if cy < n_params { jac[cy * n_res + 0] = d_dy1; }
                    if cx < n_params { jac[cx * n_res + 1] = d_dx2; }
                    if cy < n_params { jac[cy * n_res + 1] = d_dy2; }
                }
                true
            }
            _ => false,
        }
    }

    /// Helper for KinematicsDiffDrive and KinematicsCarlike analytical Jacobian.
    #[inline]
    fn _kinematics_jac_into(&self, idx_start: usize, idx_end: usize,
                            poses: &[crate::teb::pose_se2::PoseSe2],
                            n_params: usize, n_residuals: usize,
                            jac: &mut [f64]) {
        if idx_start == 0 || idx_start >= poses.len() - 1 { return; }
        if idx_end == 0 || idx_end >= poses.len() - 1 { return; }
        let p1 = &poses[idx_start];
        let p2 = &poses[idx_end];
        let dx = p2.x() - p1.x();
        let dy = p2.y() - p1.y();
        let c1 = p1.theta().cos();
        let s1 = p1.theta().sin();
        let c2 = p2.theta().cos();
        let s2 = p2.theta().sin();
        // nh = ((c1+c2)*dy - (s1+s2)*dx).abs()
        let nh_raw = (c1 + c2) * dy - (s1 + s2) * dx;
        let sign_nh = if nh_raw >= 0.0 { 1.0 } else { -1.0 };
        let fwd_raw = dx * c1 + dy * s1;
        let d_fwd = if fwd_raw > 0.0 { 1.0 } else { 0.0 };
        // d(nh)/d(p1.x) = s1+s2, d/d(p1.y) = -(c1+c2)
        // d(nh)/d(p2.x) = -(s1+s2), d/d(p2.y) = c1+c2
        // d(fwd)/d(p1.x) = -c1, d/d(p1.y) = 0
        // d(fwd)/d(p2.x) = c1, d/d(p2.y) = s1
        let fpi = idx_start - 1;
        let fpn = idx_end - 1;
        let cx1 = fpi * 2;
        let cy1 = cx1 + 1;
        let cx2 = fpn * 2;
        let cy2 = cx2 + 1;
        if cx1 < n_params { jac[cx1 * n_residuals + 0] = sign_nh * (s1 + s2); }
        if cy1 < n_params { jac[cy1 * n_residuals + 0] = sign_nh * (-(c1 + c2)); }
        if cx1 < n_params { jac[cx1 * n_residuals + 1] = d_fwd * (-c1); }
        if cy1 < n_params { jac[cy1 * n_residuals + 1] = d_fwd * 0.0; }
        if cx2 < n_params { jac[cx2 * n_residuals + 0] = sign_nh * (-(s1 + s2)); }
        if cy2 < n_params { jac[cy2 * n_residuals + 0] = sign_nh * (c1 + c2); }
        if cx2 < n_params { jac[cx2 * n_residuals + 1] = d_fwd * c1; }
        if cy2 < n_params { jac[cy2 * n_residuals + 1] = d_fwd * s1; }
    }
}
