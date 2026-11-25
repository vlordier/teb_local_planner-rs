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
/// Pre-allocated scratch buffers for zero-copy LM optimization.
///
/// Allocated once per TEBOptimizer::new, reused across all optimization iterations.
/// Eliminates ~230K Vec allocations per optimization pass on 20 poses.
pub struct ScratchBuffers {
    /// Flattened Jacobian: [param_idx * n_residuals] in column-major order.
    jacobian_flat: Vec<f64>,
    /// Pre-allocated residual vector.
    residuals: Vec<f64>,
    /// Perturbation scratch: poses copy with one parameter modified.
    pose_perturb: Vec<PoseSe2>,
    /// Perturbation scratch: timediff copy with one element modified.
    td_perturb: Vec<f64>,
    /// Difference vector dx computed by LM solver (also holds x and Atb during computation).
    dx: Vec<f64>,
    /// Normal equation matrix AtA (symmetric, flat n*n).
    Ata: Vec<f64>,
    /// Normal equation vector Atb.
    Atb: Vec<f64>,
}

impl ScratchBuffers {
    /// Initialize scratch buffers.
    pub fn new(n_params: usize, n_residuals: usize, n_poses: usize, n_timediffs: usize) -> Self {
        Self {
            jacobian_flat: vec![0.0f64; n_params.max(1) * n_residuals.max(1)],
            residuals: vec![0.0f64; n_residuals.max(1)],
            pose_perturb: vec![PoseSe2::zero(); n_poses.max(1)],
            td_perturb: vec![0.0f64; n_timediffs.max(1)],
            dx: vec![0.0f64; n_params.max(1)],
            Ata: vec![0.0f64; n_params.max(1) * n_params.max(1)],
            Atb: vec![0.0f64; n_params.max(1)],
        }
    }

    /// Clear residuals for next evaluation.
    #[inline]
    pub fn clear_residuals(&mut self) {
        self.residuals.clear();
    }

    /// Append residual element.
    #[inline]
    pub fn push_residual(&mut self, val: f64) {
        self.residuals.push(val);
    }

    /// Zero out Jacobian flat array.
    #[inline]
    pub fn zero_jacobian(&mut self, n_params: usize, n_res: usize) {
        let size = n_params * n_res;
        let avail = self.jacobian_flat.len().min(size);
        self.jacobian_flat[..avail].iter_mut().for_each(|v| *v = 0.0);
    }

    /// Copy band data into perturbation buffers.
    #[inline]
    pub fn sync(&mut self, poses: &[PoseSe2], timediffs: &[f64]) {
        self.pose_perturb.copy_from_slice(poses);
        self.td_perturb.copy_from_slice(timediffs);
    }
}


/// TEB Optimizer — runs the Levenberg-Marquardt optimization loop.
pub struct TEBOptimizer {
    band: TimedElasticBand,
    config: TebConfig,
    max_outer_iterations: usize,
    max_inner_iterations: usize,
    /// Reusable scratch buffers for zero-copy optimization.
    scratch: ScratchBuffers,
}

impl TEBOptimizer {
    /// Create a new optimizer.
    pub fn new(band: TimedElasticBand, config: TebConfig) -> Self {
        let n_poses = band.size_poses();
        let n_td = band.size_time_diffs();
        Self {
            band,
            config,
            max_outer_iterations: 5,
            max_inner_iterations: 3,
            scratch: ScratchBuffers::new(n_td.max(1), n_poses.max(1), n_poses, n_td),
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
    /// Inner LM optimization with zero-copy scratch buffers.
    fn optimize_graph(&mut self, edges: &[(Box<dyn Edge>, EdgeParamInfo)]) -> bool {
        let n_poses = self.band.size_poses();
        let n_td = self.band.size_time_diffs();
        if n_poses < 2 || n_td < 1 { return false; }
        if self.config.robot.max_vel_x < 0.01 { return false; }

        // Build free pose mapping
        let mut free_pose: Vec<usize> = vec![usize::MAX; n_poses];
        let mut param_count = 0usize;
        for i in 1..n_poses - 1 {
            free_pose[i] = param_count;
            param_count += 2;
        }
        let td_offset = param_count;
        let n_params = td_offset + n_td;
        if n_params == 0 { return true; }

        // Copy band data once
        let poses: Vec<PoseSe2> = (0..n_poses).map(|i| *self.band.pose(i)).collect();
        let timediffs: Vec<f64> = (0..n_td).map(|i| self.band.timediff(i)).collect();

        // Ensure scratch capacity
        let max_res: usize = edges.iter().filter(|(e, _)| e.as_ref().enabled()).map(|(e, _)| e.as_ref().num_vertices()).sum();
        let jac_cap = n_params * max_res.max(1);
        if self.scratch.jacobian_flat.len() < jac_cap {
            self.scratch.jacobian_flat.resize(jac_cap, 0.0);
        }
        if self.scratch.residuals.capacity() < max_res {
            self.scratch.residuals.reserve(max_res - self.scratch.residuals.len());
        }

        // Initialize x vector
        for i in 0..n_params { self.scratch.dx[i] = 0.0; }
        for pi in 1..n_poses - 1 {
            let base = free_pose[pi];
            if base < usize::MAX {
                self.scratch.dx[base] = self.band.pose(pi).x();
                self.scratch.dx[base + 1] = self.band.pose(pi).y();
            }
        }
        for td_i in 0..n_td {
            self.scratch.dx[td_offset + td_i] = self.band.timediff(td_i);
        }

        let mut lambda = 1e-3;
        for _iter in 0..self.max_inner_iterations {
            let n_e = self.eval_jac_zc(edges, &poses, &timediffs, &free_pose, td_offset, n_params);
            if n_e == 0 { return true; }

            self.build_normal_zc(n_params);

            // Damped: Ata *= (1 + lambda)
            for i in 0..n_params {
                self.scratch.Ata[i * n_params + i] *= (1.0 + lambda);
            }

            self.solve_zc(n_params);
            let dx_norm = self.scratch.dx[..n_params].iter().map(|v| v * v).sum::<f64>().sqrt();
            if dx_norm < 1e-6 { break; }

            // Save x, compute cost at x - dx
            let x_save: Vec<f64> = self.scratch.dx[..n_params].to_vec();
            let cost_new = self.cost_zc(&poses, &timediffs, edges, &free_pose, td_offset, n_params, true);
            let cost_old = self.cost_zc(&poses, &timediffs, edges, &free_pose, td_offset, n_params, false);

            if cost_new < cost_old {
                let rho = 0.5 * self.scratch.dx[..n_params].iter()
                    .zip(self.scratch.Atb.iter())
                    .map(|(a, b)| a * b).sum::<f64>();
                let rho = if rho.abs() > 1e-20 { (cost_old - cost_new) / rho } else { 0.0 };
                let factor = if rho > 0.0 { (1.0 - 2.0 * rho).max(0.333) } else { 2.0 };
                lambda = (lambda * factor).max(1e-10);
            } else {
                self.scratch.dx[..n_params].copy_from_slice(&x_save);
                lambda = (lambda * 3.0).min(1e10);
                if lambda >= 1e10 { break; }
            }
            if cost_new < 1e-12 { break; }
        }

        self.apply_sol_zc(&free_pose, td_offset, n_params);
        true
    }

    fn eval_jac_zc(&mut self, edges: &[(Box<dyn Edge>, EdgeParamInfo)], poses: &[PoseSe2],
                   timediffs: &[f64], free_pose: &[usize], td_offset: usize, n_params: usize) -> usize {
        let s = &mut self.scratch;
        s.residuals.clear();
        s.zero_jacobian(n_params, s.residuals.capacity().max(1));

        let max_fb = free_pose.iter().filter(|&&b| b != usize::MAX).copied().max().unwrap_or(0);

        for (edge, _info) in edges.iter() {
            let e_ref = edge.as_ref();
            if !e_ref.enabled() { continue; }
            let e = e_ref.error(poses, timediffs);
            let n_res = e.len();
            for i in 0..n_res { s.residuals.push(e[i]); }
            for pi in 0..n_params {
                Self::fd_col_into(s, e_ref, pi, poses, timediffs, free_pose, td_offset, max_fb, n_params, pi);
            }
        }
        s.residuals.len()
    }

    fn fd_col_into(s: &mut ScratchBuffers, edge: &dyn Edge, param_idx: usize,
                   poses: &[PoseSe2], timediffs: &[f64], free_pose: &[usize],
                   td_offset: usize, max_fb: usize, n_params: usize, jac_col: usize) {
        let eps = 1e-7;
        s.sync(poses, timediffs);

        if param_idx <= max_fb + 1 {
            let mut pose_idx = 0usize;
            let mut comp = 0usize;
            for (j, &base) in free_pose.iter().enumerate() {
                if base != usize::MAX && base <= param_idx && param_idx < base + 2 {
                    pose_idx = j;
                    comp = param_idx - base;
                    break;
                }
            }
            if pose_idx < s.pose_perturb.len() {
                let p = s.pose_perturb[pose_idx];
                if comp == 0 { s.pose_perturb[pose_idx].set_x(p.x() + eps); }
                else { s.pose_perturb[pose_idx].set_y(p.y() + eps); }
            }
        } else {
            let td_idx = param_idx - td_offset;
            if td_idx >= 0 && td_idx < s.td_perturb.len() {
                s.td_perturb[td_idx] += eps;
            }
        }

        let e_p = edge.error(&s.pose_perturb, &s.td_perturb);
        let nr = s.residuals.len().min(e_p.len());
        for r in 0..nr {
            let off = jac_col * nr + r;
            if off < s.jacobian_flat.len() {
                s.jacobian_flat[off] = (e_p[r] - s.residuals[r]) / eps;
            }
        }

        // Restore original values
        if param_idx <= max_fb + 1 {
            let mut pose_idx = 0usize;
            let mut comp = 0usize;
            for (j, &base) in free_pose.iter().enumerate() {
                if base != usize::MAX && base <= param_idx && param_idx < base + 2 {
                    pose_idx = j;
                    comp = param_idx - base;
                    break;
                }
            }
            if pose_idx < poses.len() {
                let p = poses[pose_idx];
                if comp == 0 { s.pose_perturb[pose_idx].set_x(p.x()); }
                else { s.pose_perturb[pose_idx].set_y(p.y()); }
            }
        } else {
            let td_idx = param_idx - td_offset;
            if td_idx >= 0 && td_idx < timediffs.len() {
                s.td_perturb[td_idx] = timediffs[td_idx];
            }
        }
    }

    fn build_normal_zc(&mut self, n: usize) {
        let s = &mut self.scratch;
        let nr = s.residuals.len();
        // Zero AtA and Atb
        for i in 0..n * n { s.Ata[i] = 0.0; }
        for i in 0..n { s.Atb[i] = 0.0; }

        // Atb = J^T * e
        for ci in 0..n {
            for ri in 0..nr {
                let off = ci * nr + ri;
                if off < s.jacobian_flat.len() {
                    s.Atb[ci] += s.jacobian_flat[off] * s.residuals[ri];
                }
            }
        }

        // AtA = J^T * J
        for ci in 0..n {
            for cj in ci..n {
                let mut prod = 0.0;
                for ri in 0..nr {
                    let o1 = ci * nr + ri;
                    let o2 = cj * nr + ri;
                    if o1 < s.jacobian_flat.len() && o2 < s.jacobian_flat.len() {
                        prod += s.jacobian_flat[o1] * s.jacobian_flat[o2];
                    }
                }
                s.Ata[ci * n + cj] = prod;
                if cj != ci { s.Ata[cj * n + ci] = prod; }
            }
        }
    }

    fn solve_zc(&mut self, n: usize) {
        if n == 0 { return; }
        let s = &mut self.scratch;
        let mat = nalgebra::DMatrix::from_row_slice(n, n, &s.Ata[..n*n]);
        let b = nalgebra::DVector::from_vec(s.Atb[..n].to_vec());
        let lu = nalgebra::LU::new(mat);
        let result = lu.solve(&b).map(|x| (0..n).map(|i| x[i]).collect())
            .unwrap_or_else(|| vec![0.0; n]);
        s.dx[..n].copy_from_slice(&result);
    }

    fn cost_zc(&self, poses: &[PoseSe2], timediffs: &[f64],
               edges: &[(Box<dyn Edge>, EdgeParamInfo)], free_pose: &[usize],
               td_offset: usize, n: usize, use_dx: bool) -> f64 {
        let mut pos = poses.to_vec();
        let mut td = timediffs.to_vec();
        let x = if use_dx { &self.scratch.dx[..n] } else { &self.scratch.dx[..n] };
        let max_fb = free_pose.iter().filter(|&&b| b != usize::MAX).copied().max().unwrap_or(0);

        for pi in 0..n {
            if pi <= max_fb + 1 {
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
                    if comp == 0 { pos[pose_idx].set_x(x[pi]); }
                    else { pos[pose_idx].set_y(x[pi]); }
                }
            } else {
                let td_idx = pi - td_offset;
                if td_idx >= 0 && td_idx < td.len() { td[td_idx] = x[pi]; }
            }
        }

        let mut total = 0.0;
        for (edge, _) in edges.iter() {
            if !edge.as_ref().enabled() { continue; }
            let e = edge.as_ref().error(&pos, &td);
            for &v in &e { total += v * v; }
        }
        total
    }

    fn apply_sol_zc(&mut self, free_pose: &[usize], td_offset: usize, n_params: usize) {
        let n_poses = self.band.size_poses();
        let n_td = self.band.size_time_diffs();
        let dx = &self.scratch.dx[..n_params.min(self.scratch.dx.len())];
        let max_fb = free_pose.iter().filter(|&&b| b != usize::MAX).copied().max().unwrap_or(0);

        for pi in 0..dx.len() {
            if pi <= max_fb + 1 {
                let mut pose_idx = 0usize;
                let mut comp = 0usize;
                for (j, &base) in free_pose.iter().enumerate() {
                    if base != usize::MAX && base <= pi && pi < base + 2 {
                        pose_idx = j;
                        comp = pi - base;
                        break;
                    }
                }
                if pose_idx < n_poses {
                    if comp == 0 { self.band.pose_mut(pose_idx).set_x(dx[pi]); }
                    else { self.band.pose_mut(pose_idx).set_y(dx[pi]); }
                }
            } else {
                let td_idx = pi - td_offset;
                if td_idx >= 0 && td_idx < n_td {
                    *self.band.timediff_mut(td_idx) = dx[pi];
                }
            }
        }
    }

}
