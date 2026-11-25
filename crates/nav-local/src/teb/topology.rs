//! Topology handling and homotopy class planning.
//!
//! Ported from C++ `homotopy_class_planner.cpp`, `h_signature.h`, and `graph_search.h`.
//!
//! Key concepts:
//! - H-signature: complex-valued signature encoding obstacle-relative ordering along a trajectory
//! - Left/right classification: each obstacle is classified as left (-1) or right (+1) of the trajectory
//! - H-cfd (Homotopy Class Field Diagram): explores alternative topologies via graph search

use nalgebra::{Complex, Vector2};
use std::sync::Arc;

use crate::teb::band::TimedElasticBand;
use crate::teb::pose_se2::PoseSe2;
use crate::teb::obstacles::{Obstacle, ObstaclePtr};
use crate::teb::math::normalize_theta;

/// H-signature for 2D obstacle classification.
///
/// Computes the complex-valued homotopy signature of a trajectory relative to obstacles.
/// Two trajectories are in the same homotopy class if their H-signatures are approximately equal.
///
/// Ported from `calculateEquivalenceClass` in homotopy_class_planner.cpp.
#[derive(Debug, Clone)]
pub struct HSignature {
    /// Complex-valued signature: sum over obstacles of h_i / |h_i|
    /// where h_i = sum_j (p_j - o_i) / |p_j - o_i| (normalized relative position vectors)
    signature: Complex<f64>,
}

impl HSignature {
    /// Compute H-signature for a trajectory relative to obstacles.
    pub fn compute(poses: &[PoseSe2], obstacles: &[&ObstaclePtr], h_scale: f64) -> Self {
        let mut sig = Complex::new(0.0, 0.0);

        for obs in obstacles {
            let h = Self::obstacle_h_signature(obs.as_ref(), poses, h_scale);
            if h.norm() > 1e-10 {
                // Normalize: h / |h|
                let norm = h.norm();
                sig += h / norm;
            }
        }

        Self { signature: sig }
    }

    /// Compute h-signature for a single obstacle.
    fn obstacle_h_signature(obs: &dyn Obstacle, poses: &[PoseSe2], h_scale: f64) -> Complex<f64> {
        let obs_pos = obs.centroid();
        let mut h = Complex::new(0.0, 0.0);

        for pose in poses {
            let diff = obs_pos - pose.position();
            if diff.norm() > 1e-10 {
                // h_i_j = normalize((p_j - o_i) / |p_j - o_i|)
                let normalized = diff / diff.norm();
                let complex_val = Complex::new(normalized.x, normalized.y);
                h += complex_val;
            }
        }

        h * h_scale
    }

    /// Check if two H-signatures are approximately equal.
    pub fn is_equal(&self, other: &Self, threshold: f64) -> bool {
        let diff = self.signature - other.signature;
        diff.re.abs() < threshold && diff.im.abs() < threshold
    }

    /// Get the raw signature value.
    pub fn value(&self) -> Complex<f64> {
        self.signature
    }
}

/// Classification result for each obstacle.
///
/// Each obstacle is classified as:
/// - +1 (right) if the trajectory passes to the left of it
/// - -1 (left) if the trajectory passes to the right of it
/// - 0 (unknown/uncertain) if distance is too large
#[derive(Debug, Clone)]
pub struct EdgeClassification {
    pub signs: Vec<i8>,  // +1=right, -1=left, 0=unknown
    pub distances: Vec<f64>,
}

impl EdgeClassification {
    fn new(n: usize) -> Self {
        Self {
            signs: vec![0i8; n],
            distances: vec![0.0; n],
        }
    }
}

/// Topology manager for homotopy class planning.
///
/// Manages multiple TEB trajectories in different homotopy classes,
/// classifies obstacles relative to each trajectory, and selects the best trajectory.
///
/// Ported from `HomotopyClassPlanner`.
pub struct TopologyManager {
    /// Classifications for each known homotopy class.
    classifications: Vec<EdgeClassification>,
    /// Index of the currently selected (best) class.
    selected_index: usize,
    /// H-signature threshold for equivalence.
    h_threshold: f64,
    /// H-signature scaling factor.
    h_scale: f64,
}

impl Default for TopologyManager {
    fn default() -> Self {
        Self::new()
    }
}

impl TopologyManager {
    /// Create a new topology manager.
    pub fn new() -> Self {
        Self {
            classifications: Vec::new(),
            selected_index: 0,
            h_threshold: 0.1,
            h_scale: 0.2,
        }
    }

    /// Set the H-signature equivalence threshold.
    pub fn set_h_threshold(&mut self, threshold: f64) {
        self.h_threshold = threshold;
    }

    /// Set the H-signature scaling factor.
    pub fn set_h_scale(&mut self, scale: f64) {
        self.h_scale = scale;
    }

    /// Classify obstacles relative to a trajectory.
    ///
    /// For each obstacle, computes whether the trajectory passes to its left or right.
    /// Uses the signed distance from the obstacle to each segment of the trajectory.
    pub fn classify_obstacles(
        &self,
        trajectory: &[PoseSe2],
        obstacles: &[&ObstaclePtr],
    ) -> EdgeClassification {
        let mut classification = EdgeClassification::new(obstacles.len());

        for (i, obs_ptr) in obstacles.iter().enumerate() {
            let obs_pos = obs_ptr.centroid();

            // Check distance to first pose
            let dist_to_start = (obs_pos - trajectory[0].position()).norm();
            if dist_to_start > 100.0 {
                classification.signs[i] = 0;
                classification.distances[i] = dist_to_start;
                continue;
            }

            // Find closest segment and compute side
            let (closest_side, min_dist) = Self::find_side_and_distance(&obs_pos, trajectory);
            classification.distances[i] = min_dist;
            classification.signs[i] = closest_side;
        }

        classification
    }

    /// Find which side of the trajectory an obstacle is on.
    fn find_side_and_distance(
        obs_pos: &Vector2<f64>,
        trajectory: &[PoseSe2],
    ) -> (i8, f64) {
        let mut min_dist = f64::MAX;
        let mut side: i8 = 0;

        for i in 0..trajectory.len().saturating_sub(1) {
            let p1 = trajectory[i].position();
            let p2 = trajectory[i + 1].position();
            // Compute distance from point to segment
            let dist = Self::point_segment_distance(obs_pos, p1, p2);
            if dist < min_dist {
                min_dist = dist;

                // Determine side: +1 if obstacle is to the right, -1 if left
                let seg = p2 - p1;
                let to_obs = obs_pos - p1;
                // Cross product in 2D: seg.x * to_obs.y - seg.y * to_obs.x
                let cross = seg.x * to_obs.y - seg.y * to_obs.x;
                side = if cross > 0.0 { 1 } else { -1 };
            }
        }

        (side, min_dist)
    }

    /// Compute distance from a point to a line segment.
    fn point_segment_distance(
        p: &Vector2<f64>,
        a: &Vector2<f64>,
        b: &Vector2<f64>,
    ) -> f64 {
        let ab = b - a;
        let ap = p - a;
        let ab_sq = ab.x * ab.x + ab.y * ab.y;

        if ab_sq < 1e-10 {
            return ap.norm(); // degenerate segment
        }

        let t = (ap.x * ab.x + ap.y * ab.y) / ab_sq;
        let t = t.max(0.0).min(1.0);

        let closest = a + ab * t;
        (p - closest).norm()
    }

    /// Detect if the topology has changed compared to a previous classification.
    pub fn detect_topology_change(&self, new_classification: &EdgeClassification) -> bool {
        if self.classifications.is_empty() {
            return false;
        }

        let current = &self.classifications[self.selected_index];

        // Check if signs are the same
        if current.signs.len() != new_classification.signs.len() {
            return true;
        }

        for (a, b) in current.signs.iter().zip(new_classification.signs.iter()) {
            if a != b && *a != 0 && *b != 0 {
                return true;
            }
        }

        false
    }

    /// Add a new classification for a homotopy class.
    ///
    /// Returns true if this is a new (non-equivalent) class.
    pub fn add_classification(&mut self, classification: EdgeClassification) -> bool {
        // Compute H-signature and check equivalence with existing classes
        let h_sig = HSignature::compute(
            &vec![PoseSe2::zero(); classification.signs.len()], // placeholder
            &[], // placeholder
            self.h_scale,
        );

        for existing in self.classifications.iter() {
            if existing.signs.len() == classification.signs.len()
                && existing.signs == classification.signs
            {
                return false; // Already have this class
            }
        }

        self.classifications.push(classification);
        true
    }

    /// Select the best TEB from multiple candidates.
    ///
    /// Returns the index of the best TEB in the input slice.
    pub fn select_best(&self, tebs: &[&mut TimedElasticBand]) -> usize {
        // Select the TEB with the lowest cost
        let mut best_idx = 0;
        let mut best_cost = f64::MAX;

        for (i, teb) in tebs.iter().enumerate() {
            let cost = Self::teb_cost(teb);
            if cost < best_cost {
                best_cost = cost;
                best_idx = i;
            }
        }

        best_idx
    }

    /// Estimate the cost of a TEB.
    fn teb_cost(teb: &TimedElasticBand) -> f64 {
        let mut cost = 0.0;

        // Sum of squared time differences (prefer shorter times)
        for i in 0..teb.size_time_diffs() {
            let dt = teb.timediff(i);
            cost += dt * dt;
        }

        // Sum of squared path lengths (prefer shorter paths)
        for i in 1..teb.size_poses() { let delta = teb.pose(i).position() - teb.pose(i - 1).position(); cost += delta.norm() * delta.norm(); }

        cost
    }

    /// Explore homotopy classes by generating candidate paths.
    ///
    /// For each obstacle near the start-goal line, generates paths passing to
    /// the left and right of the obstacle.
    pub fn explore_homotopy_classes(
        &self,
        start: &PoseSe2,
        goal: &PoseSe2,
        obstacles: &[&ObstaclePtr],
    ) -> Vec<Vec<PoseSe2>> {
        let mut candidates: Vec<Vec<PoseSe2>> = Vec::new();

        // Always include the direct path
        candidates.push(vec![*start, *goal]);

        // For each obstacle, generate left and right paths
        for obs in obstacles {
            let obs_pos = obs.centroid();

            // Path passing to the right of the obstacle
            let right_via = Self::offset_point(&obs_pos, start, goal, 1.0, 0.5);
            candidates.push(vec![*start, right_via, *goal]);

            // Path passing to the left of the obstacle
            let left_via = Self::offset_point(&obs_pos, start, goal, -1.0, 0.5);
            candidates.push(vec![*start, left_via, *goal]);
        }

        candidates
    }

    /// Generate a via-point offset from an obstacle relative to start-goal direction.
    fn offset_point(
        obs_pos: &Vector2<f64>,
        start: &PoseSe2,
        goal: &PoseSe2,
        side: f64,
        frac: f64,
    ) -> PoseSe2 {
        let sg = goal.position() - start.position();
        let frac_pos = start.position() + sg * frac;

        // Perpendicular offset
        let perp = Vector2::new(-sg.y, sg.x).normalize();
        let offset = perp * (sg.norm() * 0.3 * side);

        PoseSe2::new(frac_pos + offset, goal.theta())
    }

    /// Get the number of known homotopy classes.
    pub fn num_classes(&self) -> usize {
        self.classifications.len()
    }

    /// Get the selected class index.
    pub fn selected_index(&self) -> usize {
        self.selected_index
    }

    /// Set the selected class.
    pub fn set_selected(&mut self, idx: usize) {
        if idx < self.classifications.len() {
            self.selected_index = idx;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use super::*;

    #[test]
    fn test_h_signature_equality() {
        let sig1 = HSignature {
            signature: Complex::new(0.5, 0.3),
        };
        let sig2 = HSignature {
            signature: Complex::new(0.51, 0.31),
        };
        assert!(sig1.is_equal(&sig2, 0.1));

        let sig3 = HSignature {
            signature: Complex::new(1.0, 1.0),
        };
        assert!(!sig1.is_equal(&sig3, 0.1));
    }

    #[test]
    fn test_topology_classification() {
        let mut manager = TopologyManager::new();
        let start = PoseSe2::from_components(0.0, 0.0, 0.0);
        let mid = PoseSe2::from_components(1.0, 0.0, 0.0);
        let goal = PoseSe2::from_components(2.0, 0.0, 0.0);
        let trajectory = vec![start, mid, goal];

        // Obstacle to the right
        let obs_right: ObstaclePtr = Arc::new(crate::teb::obstacles::CircularObstacle::new(Vector2::new(1.0, 0.5), 0.1));
        let obs_left: ObstaclePtr = Arc::new(crate::teb::obstacles::CircularObstacle::new(Vector2::new(1.0, -0.5), 0.1));
        let obstacles = vec![&obs_right, &obs_left];

        let classification = manager.classify_obstacles(&trajectory, &obstacles);
        assert_eq!(classification.signs.len(), 2);
    }

    #[test]
    fn test_point_segment_distance() {
        let p = Vector2::new(0.0, 1.0);
        let a = Vector2::new(-1.0, 0.0);
        let b = Vector2::new(1.0, 0.0);
        let obs: Arc<dyn Obstacle + Send + Sync> = Arc::new(crate::teb::obstacles::PointObstacle::new(Vector2::new(0.0, 1.0)));
        let dist = TopologyManager::new().classify_obstacles(
            &[PoseSe2::from_components(-1.0, 0.0, 0.0), PoseSe2::from_components(1.0, 0.0, 0.0)],
            &[&obs],
        );
        assert!(dist.distances[0] <= 1.001);
    }

    #[test]
    fn test_h_signature_scale() {
        let mut manager = TopologyManager::new();
        manager.set_h_scale(0.5);
        assert_eq!(manager.h_scale, 0.5);
    }
}
