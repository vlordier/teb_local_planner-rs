//! TEB optimizer: Levenberg-Marquardt loop with iterative weight adaptation.
//!
//! Ported from C++ `optimal_planner.cpp`: optimizeTEB, optimizeGraph, buildGraph.

use nalgebra::{DMatrix, DVector, LU};
use nav_core::constraints::TebConfig;

use crate::teb::band::TimedElasticBand;
use crate::teb::pose_se2::PoseSe2;
use crate::teb::edges::{
    Edge, EdgeAcceleration, EdgeKinematicsCarlike, EdgeKinematicsDiffDrive,
    EdgeObstacle, EdgePreferRotDir, EdgeShortestPath, EdgeTimeOptimal, EdgeVelocity,
    EdgeViaPoint, EdgeDynamicObstacle,
};

// ===== Parameter types =====

/// Parameter vertex type: which state variable is this?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamVertex {
    /// Pose parameter: 0 = x, 1 = y
    Pose { pose_idx: usize, comp: usize },
    /// Time difference parameter
    TimeDiff { td_idx: usize },
}

// ===== Edge parameter description =====

/// Describes which vertices an edge uses.
#[derive(Debug, Clone)]
enum EdgeParamInfo {
    TimeOptimal { td_idx: usize },
    ShortestPath { prev: usize, next: usize },
    Velocity { start: usize, end: usize, td_idx: usize },
    Acceleration { p0: usize, p1: usize, p2: usize, td0: usize, td1: usize },
    Kinematics { start: usize, end: usize },
    Obstacle { pose_idx: usize },
    ViaPoint { pose_idx: usize },
    PreferRotDir { start: usize, end: usize },
}

impl EdgeParamInfo {
    fn to_global_params(&self, free_pose: &[usize], td_offset: usize) -> Vec<usize> {
        let mut params = Vec::new();
        for pv in self.param_types() {
            match pv {
                ParamVertex::Pose { pose_idx, comp } => {
                    if let Some(&base) = free_pose.get(pose_idx) {
                        params.push(base + comp);
                    } else {
                        params.push(usize::MAX);
                    }
                }
                ParamVertex::TimeDiff { td_idx } => {
                    params.push(td_offset + td_idx);
                }
            }
        }
        params
    }

    fn param_types(&self) -> Vec<ParamVertex> {
        match self {
            EdgeParamInfo::TimeOptimal { td_idx } => {
                vec![ParamVertex::TimeDiff { td_idx: *td_idx }]
            }
            EdgeParamInfo::ShortestPath { prev, next } => {
                vec![
                    ParamVertex::Pose { pose_idx: *prev, comp: 0 },
                    ParamVertex::Pose { pose_idx: *prev, comp: 1 },
                    ParamVertex::Pose { pose_idx: *next, comp: 0 },
                    ParamVertex::Pose { pose_idx: *next, comp: 1 },
                ]
            }
            EdgeParamInfo::Velocity { start, end, td_idx } => {
                vec![
                    ParamVertex::Pose { pose_idx: *start, comp: 0 },
                    ParamVertex::Pose { pose_idx: *start, comp: 1 },
                    ParamVertex::Pose { pose_idx: *end, comp: 0 },
                    ParamVertex::Pose { pose_idx: *end, comp: 1 },
                    ParamVertex::TimeDiff { td_idx: *td_idx },
                ]
            }
            EdgeParamInfo::Acceleration { p0, p1, p2, td0, td1 } => {
                vec![
                    ParamVertex::Pose { pose_idx: *p0, comp: 0 },
                    ParamVertex::Pose { pose_idx: *p0, comp: 1 },
                    ParamVertex::Pose { pose_idx: *p1, comp: 0 },
                    ParamVertex::Pose { pose_idx: *p1, comp: 1 },
                    ParamVertex::Pose { pose_idx: *p2, comp: 0 },
                    ParamVertex::Pose { pose_idx: *p2, comp: 1 },
                    ParamVertex::TimeDiff { td_idx: *td0 },
                    ParamVertex::TimeDiff { td_idx: *td1 },
                ]
            }
            EdgeParamInfo::Kinematics { start, end } => {
                vec![
                    ParamVertex::Pose { pose_idx: *start, comp: 0 },
                    ParamVertex::Pose { pose_idx: *start, comp: 1 },
                    ParamVertex::Pose { pose_idx: *end, comp: 0 },
                    ParamVertex::Pose { pose_idx: *end, comp: 1 },
                ]
            }
            EdgeParamInfo::Obstacle { pose_idx } => {
                vec![ParamVertex::Pose { pose_idx: *pose_idx, comp: 0 }]
            }
            EdgeParamInfo::ViaPoint { pose_idx } => {
                vec![ParamVertex::Pose { pose_idx: *pose_idx, comp: 0 }]
            }
            EdgeParamInfo::PreferRotDir { start, end } => {
                vec![
                    ParamVertex::Pose { pose_idx: *start, comp: 0 },
                    ParamVertex::Pose { pose_idx: *start, comp: 1 },
                    ParamVertex::Pose { pose_idx: *end, comp: 0 },
                    ParamVertex::Pose { pose_idx: *end, comp: 1 },
                ]
            }
        }
    }
}

/// TEB Optimizer — runs the Levenberg-Marquardt optimization loop.
pub struct TEBOptimizer {
    band: TimedElasticBand,
    config: TebConfig,
    max_outer_iterations: usize,
    max_inner_iterations: usize,
}

impl TEBOptimizer {
    /// Create a new optimizer.
    pub fn new(band: TimedElasticBand, config: TebConfig) -> Self {
        Self {
            band,
            config,
            max_outer_iterations: 5,
            max_inner_iterations: 3,
        }
    }

    pub fn set_max_outer_iterations(&mut self, n: usize) {
        self.max_outer_iterations = n;
    }

    pub fn set_max_inner_iterations(&mut self, n: usize) {
        self.max_inner_iterations = n;
    }

    /// Get a reference to the internal band (for inspection after optimization).
    pub fn band(&self) -> &TimedElasticBand {
        &self.band
    }

    /// Run the full optimization (mirrors C++ optimizeTEB).
    pub fn optimize(&mut self) -> bool {
        let weight_adapt_factor = self.config.optim.weight_adapt_factor;
        if !self.config.optim.optimization_activate {
            return false;
        }

        let mut success = false;
        let mut weight_multiplier = 1.0;

        for _outer in 0..self.max_outer_iterations {
            if self.config.trajectory.teb_autosize {
                self.band.auto_resize(
                    self.config.trajectory.dt_ref,
                    self.config.trajectory.dt_hysteresis,
                    self.config.trajectory.min_samples,
                    self.config.trajectory.max_samples,
                    true,
                );
            }

            let edges = self.build_edges(weight_multiplier);
            if edges.is_empty() {
                return false;
            }

            if !self.optimize_graph(&edges) {
                return false;
            }

            weight_multiplier *= weight_adapt_factor;
        }

        success
    }

    /// Build all edges (mirrors C++ buildGraph + AddEdges*).
    fn build_edges(&self, weight_multiplier: f64) -> Vec<(Box<dyn Edge>, EdgeParamInfo)> {
        let n = self.band.size_poses();
        if n < 2 {
            return Vec::new();
        }

        let mut edges: Vec<(Box<dyn Edge>, EdgeParamInfo)> = Vec::new();
        let weights = &self.config.optim;
        let robot = &self.config.robot;

        // Time-optimal edges
        for i in 0..n - 1 {
            edges.push((Box::new(EdgeTimeOptimal::new(i, weights.weight_optimaltime)),
                EdgeParamInfo::TimeOptimal { td_idx: i }));
        }

        // Shortest-path edges
        for i in 0..n - 1 {
            edges.push((Box::new(EdgeShortestPath::new(i, i + 1, weights.weight_shortest_path)),
                EdgeParamInfo::ShortestPath { prev: i, next: i + 1 }));
        }

        // Velocity edges
        for i in 0..n - 1 {
            let mut edge = EdgeVelocity::new(
                i, i + 1, i,
                robot.max_vel_x, robot.max_vel_x_backwards,
                robot.max_vel_theta, robot.max_vel_y,
                robot.exact_arc_length,
            );
            edge.set_weights(
                weights.weight_max_vel_x * weight_multiplier,
                weights.weight_max_vel_theta * weight_multiplier,
            );
            edges.push((Box::new(edge), EdgeParamInfo::Velocity {
                start: i, end: i + 1, td_idx: i,
            }));
        }

        // Acceleration edges
        for i in 0..n - 2 {
            let mut edge = EdgeAcceleration::new(
                i, i + 1, i + 2,
                i, i + 1,
                robot.acc_lim_x, robot.acc_lim_theta,
                robot.exact_arc_length,
            );
            edge.set_weights(
                weights.weight_acc_lim_x * weight_multiplier,
                weights.weight_acc_lim_theta * weight_multiplier,
            );
            edges.push((Box::new(edge), EdgeParamInfo::Acceleration {
                p0: i, p1: i + 1, p2: i + 2,
                td0: i, td1: i + 1,
            }));
        }

        // Kinematics edges
        if robot.min_turning_radius == 0.0 || weights.weight_kinematics_turning_radius == 0.0 {
            for i in 0..n - 1 {
                let mut edge = EdgeKinematicsDiffDrive::new(i, i + 1);
                edge.set_weights(
                    weights.weight_kinematics_nh * weight_multiplier,
                    weights.weight_kinematics_forward_drive * weight_multiplier,
                );
                edges.push((Box::new(edge), EdgeParamInfo::Kinematics {
                    start: i, end: i + 1,
                }));
            }
        } else {
            for i in 0..n - 1 {
                let mut edge = EdgeKinematicsCarlike::new(
                    i, i + 1, robot.min_turning_radius,
                    robot.exact_arc_length,
                );
                edge.set_weights(
                    weights.weight_kinematics_nh * weight_multiplier,
                    weights.weight_kinematics_turning_radius * weight_multiplier,
                );
                edges.push((Box::new(edge), EdgeParamInfo::Kinematics {
                    start: i, end: i + 1,
                }));
            }
        }

        edges
    }

    /// Inner LM optimization (mirrors C++ optimizeGraph).
    fn optimize_graph(&mut self, edges: &[(Box<dyn Edge>, EdgeParamInfo)]) -> bool {
        let n_poses = self.band.size_poses();
        let n_td = self.band.size_time_diffs();

        if n_poses < 2 || n_td < 1 {
            return false;
        }

        if self.config.robot.max_vel_x < 0.01 {
            return false;
        }

        // Build free pose mapping: pose_idx -> param_start
        let mut free_pose: Vec<usize> = vec![usize::MAX; n_poses];
        let mut param_count = 0usize;
        for i in 0..n_poses {
            let is_start = i == 0;
            let is_goal = i == n_poses - 1;
            if !is_start && !is_goal {
                free_pose[i] = param_count;
                param_count += 2;
            }
        }

        let td_offset = param_count;
        let n_params = td_offset + n_td;

        if n_params == 0 {
            return true;
        }

        let edge_params: Vec<Vec<usize>> = edges.iter()
            .map(|(_, info)| info.to_global_params(&free_pose, td_offset))
            .collect();

        // Initialize parameter vector
        let mut x: Vec<f64> = vec![0.0; n_params];
        for pi in 1..n_poses - 1 {
            let base = free_pose[pi];
            if base < usize::MAX {
                x[base] = self.band.pose(pi).x();
                x[base + 1] = self.band.pose(pi).y();
            }
        }
        for td_i in 0..n_td {
            x[td_offset + td_i] = self.band.timediff(td_i);
        }

        // LM iterations
        let mut lambda = 1e-3;

        for _iter in 0..self.max_inner_iterations {
            let (e, J_cols) = self.evaluate_residuals_jacobian(edges, &free_pose, td_offset);
            let n_e = e.len();
            if n_e == 0 {
                return true;
            }

            let n = n_params;
            let (AtA, Atb) = self.build_normal_eq(&J_cols, &e, n);

            let mut Ad = AtA.clone();
            for i in 0..n {
                Ad[i * n + i] += AtA[i * n + i] * lambda;
            }

            let dx = self.solve_system(&Ad, n, &Atb);

            let dx_norm = dx.iter().map(|v| v * v).sum::<f64>().sqrt();
            if dx_norm < 1e-6 {
                break;
            }

            let x_new: Vec<f64> = (0..n).map(|i| x[i] - dx[i]).collect();
            let cost_new = self.compute_cost(&x_new, edges, &free_pose, td_offset);
            let cost_old = self.compute_cost(&x, edges, &free_pose, td_offset);

            if cost_new < cost_old {
                x = x_new;
                let denom = 0.5 * dx.iter().zip(Atb.iter()).map(|(a, b)| a * b).sum::<f64>();
                let rho = if denom.abs() > 1e-20 {
                    (cost_old - cost_new) / denom
                } else { 0.0 };
                let factor = if rho > 0.0 { (1.0 - 2.0 * rho).max(0.333) } else { 2.0 };
                lambda = (lambda * factor).max(1e-10);
            } else {
                lambda = (lambda * 3.0).min(1e10);
                if lambda >= 1e10 { break; }
            }

            if cost_new < 1e-12 { break; }
        }

        self.apply_solution(&x, &free_pose, td_offset);
        true
    }

    fn evaluate_residuals_jacobian(
        &self,
        edges: &[(Box<dyn Edge>, EdgeParamInfo)],
        free_pose: &[usize],
        td_offset: usize,
    ) -> (Vec<f64>, Vec<Vec<f64>>) {
        let n_params = td_offset + self.band.size_time_diffs();
        let poses: Vec<PoseSe2> = (0..self.band.size_poses()).map(|i| *self.band.pose(i)).collect();
        let timediffs: Vec<f64> = (0..self.band.size_time_diffs()).map(|i| self.band.timediff(i)).collect();
        let mut residuals: Vec<f64> = Vec::new();
        let mut jacobian_cols: Vec<Vec<f64>> = vec![vec![0.0f64; 0]; n_params];

        for (edge, _info) in edges.iter() {
            if !edge.enabled() { continue; }
            let e = edge.error(&poses, &timediffs);
            let n_res = e.len();
            for i in 0..n_res {
                residuals.push(e[i]);
            }
            for pi in 0..n_params {
                let fd = self.finite_diff_residual_at(&**edge, pi, &poses, &timediffs, free_pose, td_offset, 1e-7);
                for r in 0..n_res {
                    let col = &mut jacobian_cols[pi];
                    if col.len() <= r { col.resize(n_res, 0.0); }
                    col[r] = fd[r];
                }
            }
        }

        (residuals, jacobian_cols)
    }

    fn finite_diff_residual_at(
        &self,
        edge: &dyn Edge,
        param_idx: usize,
        poses: &[PoseSe2],
        timediffs: &[f64],
        free_pose: &[usize],
        td_offset: usize,
        eps: f64,
    ) -> Vec<f64> {
        let n_v = edge.num_vertices();
        let mut pos = poses.to_vec();
        let mut td = timediffs.to_vec();

        let max_free_base = free_pose.iter().filter(|&&b| b != usize::MAX).copied().max().unwrap_or(0);

        if param_idx <= max_free_base + 1 {
            let mut pose_idx = 0usize;
            let mut comp = 0usize;
            for (j, &base) in free_pose.iter().enumerate() {
                if base != usize::MAX && base <= param_idx && param_idx < base + 2 {
                    pose_idx = j;
                    comp = param_idx - base;
                    break;
                }
            }
            if pose_idx < pos.len() {
                let p = pos[pose_idx];
                let mut p2 = p;
                let new_val = if comp == 0 { p.x() + eps } else { p.y() + eps };
                if comp == 0 { p2.set_x(new_val); } else { p2.set_y(new_val); }
                pos[pose_idx] = p2;
            }
        } else {
            let td_idx = param_idx - td_offset;
            if td_idx >= 0 && td_idx < td.len() {
                td[td_idx] = timediffs[td_idx] + eps;
            }
        }

        let e_p = edge.error(&pos, &td);
        let e_o = edge.error(poses, timediffs);

        let mut result = Vec::with_capacity(n_v);
        for i in 0..n_v {
            result.push(e_p[i] - e_o[i]);
        }
        result
    }

    fn build_normal_eq(&self, J_cols: &[Vec<f64>], e: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut AtA = vec![0.0f64; n * n];
        let mut Atb = vec![0.0f64; n];

        for (col, j_col) in J_cols.iter().enumerate() {
            for (row, &j_val) in j_col.iter().enumerate() {
                if row < e.len() { Atb[col] += j_val * e[row]; }
            }
        }

        for ci in 0..n {
            for cj in ci..n {
                let mut sum = 0.0;
                for (ri, &jv) in J_cols[ci].iter().enumerate() {
                    if ri < J_cols[cj].len() { sum += jv * J_cols[cj][ri]; }
                }
                AtA[ci * n + cj] = sum;
                if cj != ci { AtA[cj * n + ci] = sum; }
            }
        }

        (AtA, Atb)
    }

    fn solve_system(&self, A: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
        if n == 0 { return Vec::new(); }
        let mat = DMatrix::from_row_slice(n, n, A);
        let b_vec = DVector::from_vec(b.to_vec());
        let lu = LU::new(mat);
        lu.solve(&b_vec).map(|x| (0..n).map(|i| x[i]).collect()).unwrap_or_else(|| vec![0.0; n])
    }

    fn compute_cost(
        &self,
        x: &[f64],
        edges: &[(Box<dyn Edge>, EdgeParamInfo)],
        free_pose: &[usize],
        td_offset: usize,
    ) -> f64 {
        let n_poses = free_pose.len();
        let mut pos: Vec<PoseSe2> = (0..self.band.size_poses()).map(|i| *self.band.pose(i)).collect();
        let mut td: Vec<f64> = (0..self.band.size_time_diffs()).map(|i| self.band.timediff(i)).collect();

        let max_free_base = free_pose.iter().filter(|&&b| b != usize::MAX).copied().max().unwrap_or(0);

        for pi in 0..x.len() {
            if pi <= max_free_base + 1 {
                let mut pose_idx = 0usize;
                let mut comp = 0usize;
                for (j, &base) in free_pose.iter().enumerate() {
                    if base != usize::MAX && base <= pi && pi < base + 2 {
                        pose_idx = j;
                        comp = pi - base;
                        break;
                    }
                }
                if pose_idx < pos.len() {
                    let p = pos[pose_idx];
                    let mut p2 = p;
                    if comp == 0 { p2.set_x(x[pi]); } else { p2.set_y(x[pi]); }
                    pos[pose_idx] = p2;
                }
            } else {
                let td_idx = pi - td_offset;
                if td_idx >= 0 && td_idx < td.len() {
                    td[td_idx] = x[pi];
                }
            }
        }

        let mut total = 0.0;
        for (edge, _) in edges.iter() {
            if !edge.enabled() { continue; }
            let e = edge.error(&pos, &td);
            for &v in &e { total += v * v; }
        }
        total
    }

    fn apply_solution(&mut self, x: &[f64], free_pose: &[usize], td_offset: usize) {
        let max_free_base = free_pose.iter().filter(|&&b| b != usize::MAX).copied().max().unwrap_or(0);

        for pi in 0..x.len() {
            if pi <= max_free_base + 1 {
                let mut pose_idx = 0usize;
                let mut comp = 0usize;
                for (j, &base) in free_pose.iter().enumerate() {
                    if base != usize::MAX && base <= pi && pi < base + 2 {
                        pose_idx = j;
                        comp = pi - base;
                        break;
                    }
                }
                if pose_idx < self.band.size_poses() {
                    let p = self.band.pose(pose_idx);
                    let mut p2 = *p;
                    if comp == 0 { p2.set_x(x[pi]); } else { p2.set_y(x[pi]); }
                    *self.band.pose_mut(pose_idx).x_mut() = p2.x();
                    *self.band.pose_mut(pose_idx).y_mut() = p2.y();
                }
            } else {
                let td_idx = pi - td_offset;
                if td_idx >= 0 && td_idx < self.band.size_time_diffs() {
                    *self.band.timediff_mut(td_idx) = x[pi];
                }
            }
        }
    }
}
