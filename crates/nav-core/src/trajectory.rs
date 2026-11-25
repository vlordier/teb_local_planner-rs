//! Trajectory type for TEB: a sequence of poses with time differences.
//!
//! Ported from `timed_elastic_band.h` and the trajectory-related sections
//! of the TEB planner.

use crate::state::PoseSE2;

/// A trajectory modeled as a Timed Elastic Band (TEB).
///
/// The trajectory consists of:
/// - A **pose sequence**: the spatial part (x, y, θ) for each waypoint.
/// - A **time difference sequence**: Δtᵢ between consecutive poses.
///
/// Let Q = {sᵢ}ᵢ₌₀ⁿ be a sequence of poses, where sᵢ = [xᵢ, yᵢ, βᵢ]ᵀ.
/// The TEB augments this with time intervals ΔTᵢ forming:
/// τ = {ΔTᵢ}ᵢ₌₀ⁿ⁻¹.
///
/// The tuple of both sequences defines the underlying trajectory.
#[derive(Debug, Clone)]
pub struct Trajectory {
    /// The pose sequence.
    poses: Vec<PoseSE2>,
    /// The time difference sequence.
    /// `time_diffs[i]` is the time to transit from `poses[i]` to `poses[i+1]`.
    /// Length is `poses.len() - 1`.
    time_diffs: Vec<f64>,
}

impl Trajectory {
    // ── Constructors ──────────────────────────────────────────────

    /// Create an empty trajectory.
    pub fn new() -> Self {
        Self {
            poses: Vec::new(),
            time_diffs: Vec::new(),
        }
    }

    /// Create a trajectory with pre-allocated capacity.
    pub fn with_capacity(poses: usize, time_diffs: usize) -> Self {
        Self {
            poses: Vec::with_capacity(poses),
            time_diffs: Vec::with_capacity(time_diffs),
        }
    }

    /// Check if the trajectory is initialized (non-empty).
    #[inline]
    pub fn is_init(&self) -> bool {
        !self.poses.is_empty()
    }

    // ── Accessors ─────────────────────────────────────────────────

    /// Number of poses.
    #[inline]
    pub fn size_poses(&self) -> usize {
        self.poses.len()
    }

    /// Number of time differences.
    #[inline]
    pub fn size_time_diffs(&self) -> usize {
        self.time_diffs.len()
    }

    /// Get a pose by index (const).
    #[inline]
    pub fn pose(&self, index: usize) -> &PoseSE2 {
        &self.poses[index]
    }

    /// Get a mutable reference to a pose by index.
    #[inline]
    pub fn pose_mut(&mut self, index: usize) -> &mut PoseSE2 {
        &mut self.poses[index]
    }

    /// Get the last pose (mutable).
    #[inline]
    pub fn back_pose(&mut self) -> &mut PoseSE2 {
        self.poses.last_mut().unwrap()
    }

    /// Get the last pose (mutable). Alias for back_pose.
    #[inline]
    pub fn back_pose_mut(&mut self) -> &mut PoseSE2 {
        self.back_pose()
    }

    /// Get the last pose (const).
    #[inline]
    pub fn back_pose_ref(&self) -> &PoseSE2 {
        self.poses.last().unwrap()
    }

    /// Get the last time difference (mutable).
    #[inline]
    pub fn back_time_diff(&mut self) -> &mut f64 {
        self.time_diffs.last_mut().unwrap()
    }

    /// Get the last time difference (const).
    #[inline]
    pub fn back_time_diff_ref(&self) -> &f64 {
        self.time_diffs.last().unwrap()
    }

    /// Get time difference by index.
    #[inline]
    pub fn time_diff(&self, index: usize) -> &f64 {
        &self.time_diffs[index]
    }

    /// Mutable reference to time difference by index.
    #[inline]
    pub fn time_diff_mut(&mut self, index: usize) -> &mut f64 {
        &mut self.time_diffs[index]
    }

    /// Access the full pose sequence.
    #[inline]
    pub fn poses(&self) -> &[PoseSE2] {
        &self.poses
    }

    /// Mutable access to the full pose sequence.
    #[inline]
    pub fn poses_mut(&mut self) -> &mut Vec<PoseSE2> {
        &mut self.poses
    }

    /// Access the full time difference sequence.
    #[inline]
    pub fn time_diffs(&self) -> &[f64] {
        &self.time_diffs
    }

    /// Mutable access to the full time difference sequence.
    #[inline]
    pub fn time_diffs_mut(&mut self) -> &mut Vec<f64> {
        &mut self.time_diffs
    }

    /// Total transition time.
    #[inline]
    pub fn total_time(&self) -> f64 {
        self.time_diffs.iter().sum()
    }

    /// Find the index of the closest pose to a given position.
    pub fn find_closest_pose(&self, position: &nalgebra::Vector2<f64>) -> usize {
        let mut min_dist = f64::INFINITY;
        let mut min_idx = 0;
        for (i, pose) in self.poses.iter().enumerate() {
            let d = (position - pose.position()).norm();
            if d < min_dist {
                min_dist = d;
                min_idx = i;
            }
        }
        min_idx
    }

    // ── Modifications ─────────────────────────────────────────────

    /// Append a new pose.
    /// If the trajectory is non-empty, a default time diff of 0.3 is added.
    pub fn add_pose(&mut self, pose: PoseSE2) {
        if !self.poses.is_empty() {
            // Default time resolution
            self.time_diffs.push(0.3);
        }
        self.poses.push(pose);
    }

    /// Append a new time difference.
    pub fn add_time_diff(&mut self, dt: f64) {
        self.time_diffs.push(dt);
    }

    /// Append a (pose, time_diff) pair.
    /// Requires that there are already n poses and n-1 time diffs.
    pub fn add_pose_and_time_diff(&mut self, pose: PoseSE2, dt: f64) {
        self.poses.push(pose);
        self.time_diffs.push(dt);
    }

    /// Insert a pose at the given index.
    /// The time_diffs array is not adjusted — caller must manage.
    pub fn insert_pose(&mut self, index: usize, pose: PoseSE2) {
        self.poses.insert(index, pose);
    }

    /// Remove the pose at the given index.
    /// Also removes the corresponding time diff (if valid).
    pub fn remove_pose(&mut self, index: usize) {
        self.poses.remove(index);
        // Remove the time diff after this pose if it exists
        if index < self.time_diffs.len() {
            self.time_diffs.remove(index);
        }
    }

    /// Resize the trajectory by removing old poses from the back.
    pub fn resize_poses(&mut self, new_size: usize) {
        let dt_to_remove = self.poses.len().saturating_sub(new_size);
        self.poses.truncate(new_size);
        // Each removed pose corresponds to a time diff after it
        // Remove time_diffs from the back up to dt_to_remove
        self.time_diffs.truncate(self.time_diffs.len().saturating_sub(dt_to_remove));
    }

    /// Auto-resize the trajectory based on desired time resolution.
    ///
    /// If `dt < dt_ref - hysteresis`, insert new poses.
    /// If `dt > dt_ref + hysteresis`, remove poses.
    ///
    /// This is the TEB autosize logic from the C++ implementation.
    pub fn auto_resize(&mut self, dt_ref: f64, dt_hysteresis: f64, min_samples: usize) {
        if self.poses.len() <= 1 {
            return;
        }

        // Compute current average temporal resolution
        let total_time: f64 = self.time_diffs.iter().sum();
        let avg_dt = total_time / self.time_diffs.len() as f64;

        if avg_dt > dt_ref + dt_hysteresis && self.poses.len() > min_samples {
            // Remove every other pose (coarsen)
            let mut to_remove = Vec::new();
            for i in (0..self.poses.len() - 1).step_by(2) {
                to_remove.push(i + 1);
            }
            // Remove in reverse order
            for &i in to_remove.iter().rev() {
                self.remove_pose(i);
            }
        } else if avg_dt < dt_ref - dt_hysteresis {
            // Insert poses (refine): double the resolution
            // Insert midpoint poses between consecutive poses
            let mut new_poses = Vec::with_capacity(self.poses.len() * 2);
            let mut new_time_diffs = Vec::with_capacity(self.time_diffs.len());
            for i in 0..self.poses.len() {
                new_poses.push(self.poses[i]);
                if i < self.time_diffs.len() {
                    new_time_diffs.push(self.time_diffs[i] / 2.0);
                    if i < self.poses.len() - 1 {
                        // Insert intermediate pose
                        let mid = PoseSE2::average(&self.poses[i], &self.poses[i + 1]);
                        new_poses.push(mid);
                    }
                }
            }
            self.poses = new_poses;
            self.time_diffs = new_time_diffs;
        }
    }

    /// Clear the trajectory.
    pub fn clear(&mut self) {
        self.poses.clear();
        self.time_diffs.clear();
    }

    /// Initialize trajectory between start and goal poses with N intermediate samples.
    /// This is a placeholder; full init_from_plan is future work.
    pub(crate) fn init_to_goal(
        _start: &PoseSE2,
        _goal: &PoseSE2,
        _intermediate_samples: usize,
        _max_vel_x: f64,
        _min_samples: usize,
    ) {
        // Full implementation extracts poses from global plan, linearly interpolates
        // positions and angles between start and goal.
    }

    /// Update and prune TEB for hot-starting from a previous solution.
    ///
    /// If the goal hasn't moved much, update existing poses;
    /// otherwise reinitialize.
    pub fn update_and_prune(
        &mut self,
        start: &PoseSE2,
        goal: &PoseSE2,
        min_samples: usize,
        force_reinit_dist: f64,
        force_reinit_angular: f64,
    ) {
        if self.poses.len() >= 2 {
            let goal_pos_dist =
                (goal.position() - self.back_pose_ref().position()).norm();
            let goal_ang_diff = (goal.theta() - self.back_pose_ref().theta()).abs();
            // Angular distance needs wrapping
            let angular_dist = {
                let diff = goal.theta() - self.back_pose_ref().theta();
                diff.abs().min((diff - 2.0 * std::f64::consts::PI).abs())
                    .min((diff + 2.0 * std::f64::consts::PI).abs())
            };

            if goal_pos_dist < force_reinit_dist && angular_dist < force_reinit_angular {
                // Warm start: update existing
                self.back_pose_mut().copy_from(goal);
                // Also update start
                if let Some(first) = self.poses.first_mut() {
                    first.copy_from(start);
                }
            } else {
                // Cold start: reinit
                self.clear();
                self.poses.push(*start);
                self.poses.push(*goal);
            }
        }
    }
}

impl Default for Trajectory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::PoseSE2;

    #[test]
    fn test_empty_trajectory() {
        let t = Trajectory::new();
        assert!(!t.is_init());
        assert_eq!(t.size_poses(), 0);
        assert_eq!(t.size_time_diffs(), 0);
    }

    #[test]
    fn test_add_pose() {
        let mut t = Trajectory::new();
        let p1 = PoseSE2::from_xyz(0.0, 0.0, 0.0);
        let p2 = PoseSE2::from_xyz(1.0, 0.0, 0.0);
        let p3 = PoseSE2::from_xyz(2.0, 0.0, 0.0);

        t.add_pose(p1);
        assert_eq!(t.size_poses(), 1);
        assert_eq!(t.size_time_diffs(), 0);

        t.add_pose(p2);
        assert_eq!(t.size_poses(), 2);
        assert_eq!(t.size_time_diffs(), 1);
        assert!((t.time_diff(0) - 0.3).abs() < 1e-12);

        t.add_pose(p3);
        assert_eq!(t.size_poses(), 3);
        assert_eq!(t.size_time_diffs(), 2);
    }

    #[test]
    fn test_accessors() {
        let mut t = Trajectory::new();
        t.add_pose(PoseSE2::from_xyz(1.0, 2.0, 0.5));
        t.add_pose(PoseSE2::from_xyz(3.0, 4.0, 0.5));

        assert!((t.pose(0).x() - 1.0).abs() < 1e-12);
        assert!((t.pose(0).y() - 2.0).abs() < 1e-12);

        assert!((t.back_pose_ref().x() - 3.0).abs() < 1e-12);

        assert!((t.back_time_diff_ref() - 0.3).abs() < 1e-12);
    }

    #[test]
    fn test_remove_pose() {
        let mut t = Trajectory::new();
        t.add_pose(PoseSE2::from_xyz(0.0, 0.0, 0.0));
        t.add_pose(PoseSE2::from_xyz(1.0, 0.0, 0.0));
        t.add_pose(PoseSE2::from_xyz(2.0, 0.0, 0.0));

        t.remove_pose(1);
        assert_eq!(t.size_poses(), 2);
        assert_eq!(t.size_time_diffs(), 1);
        assert!((t.pose(0).x() - 0.0).abs() < 1e-12);
        assert!((t.pose(1).x() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_find_closest_pose() {
        let mut t = Trajectory::new();
        t.add_pose(PoseSE2::from_xyz(0.0, 0.0, 0.0));
        t.add_pose(PoseSE2::from_xyz(1.0, 0.0, 0.0));
        t.add_pose(PoseSE2::from_xyz(2.0, 0.0, 0.0));

        let pos = nalgebra::Vector2::new(0.9, 0.1);
        let idx = t.find_closest_pose(&pos);
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_total_time() {
        let mut t = Trajectory::new();
        t.add_pose(PoseSE2::from_xyz(0.0, 0.0, 0.0));
        t.add_pose(PoseSE2::from_xyz(1.0, 0.0, 0.0));
        t.add_pose(PoseSE2::from_xyz(2.0, 0.0, 0.0));

        t.time_diffs_mut()[0] = 0.5;
        t.time_diffs_mut()[1] = 0.3;

        assert!((t.total_time() - 0.8).abs() < 1e-12);
    }

    #[test]
    fn test_update_and_prune_cold_start() {
        let start = PoseSE2::from_xyz(0.0, 0.0, 0.0);
        let goal = PoseSE2::from_xyz(100.0, 0.0, 0.0);  // far away -> cold start
        let mut t = Trajectory::new();
        // First add poses so update_and_prune has something to work with
        t.add_pose(start.clone());
        t.add_pose(PoseSE2::from_xyz(1.0, 0.0, 0.0));
        t.add_pose(PoseSE2::from_xyz(2.0, 0.0, 0.0));

        // Goal is far -> cold start should reinit
        t.update_and_prune(&start, &goal, 3, 0.5, 0.5);
        assert_eq!(t.size_poses(), 2);
    }

    #[test]
    fn test_resize_poses() {
        let mut t = Trajectory::new();
        t.add_pose(PoseSE2::from_xyz(0.0, 0.0, 0.0));
        t.add_pose(PoseSE2::from_xyz(1.0, 0.0, 0.0));
        t.add_pose(PoseSE2::from_xyz(2.0, 0.0, 0.0));
        t.add_pose(PoseSE2::from_xyz(3.0, 0.0, 0.0));

        t.resize_poses(2);
        assert_eq!(t.size_poses(), 2);
        assert_eq!(t.size_time_diffs(), 1);
    }
}
