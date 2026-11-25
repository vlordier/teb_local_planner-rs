#![allow(unused_variables, unused_imports, dead_code, unused_mut, unused_assignments, unreachable_patterns)]
//! Obstacle hierarchy ported from `obstacles.h`.
//!
//! - `Obstacle` — abstract base trait.
//! - `PointObstacle` — 2D point obstacle.
//! - `CircularObstacle` — circle obstacle.
//! - `LineObstacle` — line segment obstacle.
//! - `PolygonObstacle` — arbitrary polygon.
//! - `PillObstacle` — line + radius (capsule).

use crate::misc::*;

/// Container type for obstacle references.
pub type ObstContainer = Vec<Box<dyn Obstacle>>;

/// Abstract obstacle trait.
pub trait Obstacle {
    /// Get centroid coordinates.
    fn centroid(&self) -> &nalgebra::Vector2<f64>;

    /// Check collision at a point with minimum distance threshold.
    fn check_collision(&self, pos: &nalgebra::Vector2<f64>, min_dist: f64) -> bool;

    /// Check line segment intersection.
    fn check_line_intersection(
        &self,
        line_start: &nalgebra::Vector2<f64>,
        line_end: &nalgebra::Vector2<f64>,
        min_dist: f64,
    ) -> bool;

    /// Minimum distance from a point.
    fn minimum_distance_to_point(&self, pos: &nalgebra::Vector2<f64>) -> f64;

    /// Minimum distance from a line segment.
    fn minimum_distance_to_line(
        &self,
        line_start: &nalgebra::Vector2<f64>,
        line_end: &nalgebra::Vector2<f64>,
    ) -> f64;

    /// Minimum distance to a polygon.
    fn minimum_distance_to_polygon(&self, polygon: &[nalgebra::Vector2<f64>]) -> f64;

    /// Closest point on boundary.
    fn closest_point(&self, pos: &nalgebra::Vector2<f64>) -> nalgebra::Vector2<f64>;

    /// Spatiotemporal distance at time t.
    fn minimum_spatio_temporal_distance(
        &self,
        pos: &nalgebra::Vector2<f64>,
        t: f64,
    ) -> f64;

    /// Check if obstacle is dynamic (moving).
    fn is_dynamic(&self) -> bool;

    /// Get centroid velocity (zero for static).
    fn centroid_velocity(&self) -> nalgebra::Vector2<f64>;

    /// Set centroid velocity and mark dynamic if significant.
    fn set_centroid_velocity(&mut self, vel: &nalgebra::Vector2<f64>) {
        self.set_centroid_velocity_internal(vel);
    }

    fn set_centroid_velocity_internal(&mut self, _vel: &nalgebra::Vector2<f64>);

    /// Predict centroid at time t using constant velocity model.
    fn predict_centroid_constant_velocity(&self, t: f64) -> nalgebra::Vector2<f64> {
        *self.centroid() + t * self.centroid_velocity()
    }
}

// ── Point Obstacle ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PointObstacle {
    position: nalgebra::Vector2<f64>,
    dynamic: bool,
    centroid_velocity: nalgebra::Vector2<f64>,
}

impl PointObstacle {
    pub fn new() -> Self {
        Self {
            position: nalgebra::Vector2::zeros(),
            dynamic: false,
            centroid_velocity: nalgebra::Vector2::zeros(),
        }
    }

    pub fn with_position(x: f64, y: f64) -> Self {
        Self {
            position: nalgebra::Vector2::new(x, y),
            dynamic: false,
            centroid_velocity: nalgebra::Vector2::zeros(),
        }
    }

    pub fn from_vector(pos: nalgebra::Vector2<f64>) -> Self {
        Self {
            position: pos,
            dynamic: false,
            centroid_velocity: nalgebra::Vector2::zeros(),
        }
    }

    pub fn position(&self) -> &nalgebra::Vector2<f64> {
        &self.position
    }
}

impl Default for PointObstacle {
    fn default() -> Self {
        Self::new()
    }
}

impl Obstacle for PointObstacle {
    fn centroid(&self) -> &nalgebra::Vector2<f64> {
        &self.position
    }

    fn check_collision(&self, pos: &nalgebra::Vector2<f64>, min_dist: f64) -> bool {
        self.minimum_distance_to_point(pos) < min_dist
    }

    fn check_line_intersection(
        &self,
        line_start: &nalgebra::Vector2<f64>,
        line_end: &nalgebra::Vector2<f64>,
        min_dist: f64,
    ) -> bool {
        let a = *line_end - line_start;
        let b = self.position - line_start;
        let t = a.dot(&b) / a.dot(&a);
        let t = t.clamp(0.0, 1.0);
        let nearest = *line_start + a * t;
        self.check_collision(&nearest, min_dist)
    }

    fn minimum_distance_to_point(&self, pos: &nalgebra::Vector2<f64>) -> f64 {
        (pos - self.position).norm()
    }

    fn minimum_distance_to_line(
        &self,
        line_start: &nalgebra::Vector2<f64>,
        line_end: &nalgebra::Vector2<f64>,
    ) -> f64 {
        distance_point_to_segment_2d(&self.position, line_start, line_end)
    }

    fn minimum_distance_to_polygon(&self, polygon: &[nalgebra::Vector2<f64>]) -> f64 {
        distance_point_to_polygon_2d(&self.position, polygon)
    }

    fn closest_point(&self, _pos: &nalgebra::Vector2<f64>) -> nalgebra::Vector2<f64> {
        self.position
    }

    fn minimum_spatio_temporal_distance(
        &self,
        pos: &nalgebra::Vector2<f64>,
        t: f64,
    ) -> f64 {
        let predicted = self.position + t * self.centroid_velocity;
        (predicted - pos).norm()
    }

    fn is_dynamic(&self) -> bool {
        self.dynamic
    }

    fn centroid_velocity(&self) -> nalgebra::Vector2<f64> {
        self.centroid_velocity
    }

    fn set_centroid_velocity_internal(&mut self, vel: &nalgebra::Vector2<f64>) {
        if vel.norm() >= 0.001 {
            self.centroid_velocity = *vel;
            self.dynamic = true;
        }
    }
}

// ── Circular Obstacle ──────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CircularObstacle {
    position: nalgebra::Vector2<f64>,
    radius: f64,
    dynamic: bool,
    centroid_velocity: nalgebra::Vector2<f64>,
}

impl CircularObstacle {
    pub fn new(radius: f64) -> Self {
        Self {
            position: nalgebra::Vector2::zeros(),
            radius,
            dynamic: false,
            centroid_velocity: nalgebra::Vector2::zeros(),
        }
    }

    pub fn with_position(x: f64, y: f64, radius: f64) -> Self {
        Self {
            position: nalgebra::Vector2::new(x, y),
            radius,
            dynamic: false,
            centroid_velocity: nalgebra::Vector2::zeros(),
        }
    }

    pub fn radius(&self) -> f64 {
        self.radius
    }
}

impl Obstacle for CircularObstacle {
    fn centroid(&self) -> &nalgebra::Vector2<f64> {
        &self.position
    }

    fn check_collision(&self, pos: &nalgebra::Vector2<f64>, min_dist: f64) -> bool {
        self.minimum_distance_to_point(pos) < min_dist
    }

    fn check_line_intersection(
        &self,
        line_start: &nalgebra::Vector2<f64>,
        line_end: &nalgebra::Vector2<f64>,
        min_dist: f64,
    ) -> bool {
        let a = *line_end - line_start;
        let b = self.position - line_start;
        let t = a.dot(&b) / a.dot(&a);
        let t = t.clamp(0.0, 1.0);
        let nearest = *line_start + a * t;
        self.check_collision(&nearest, min_dist)
    }

    fn minimum_distance_to_point(&self, pos: &nalgebra::Vector2<f64>) -> f64 {
        (pos - self.position).norm() - self.radius
    }

    fn minimum_distance_to_line(
        &self,
        line_start: &nalgebra::Vector2<f64>,
        line_end: &nalgebra::Vector2<f64>,
    ) -> f64 {
        distance_point_to_segment_2d(&self.position, line_start, line_end) - self.radius
    }

    fn minimum_distance_to_polygon(&self, polygon: &[nalgebra::Vector2<f64>]) -> f64 {
        distance_point_to_polygon_2d(&self.position, polygon) - self.radius
    }

    fn closest_point(&self, pos: &nalgebra::Vector2<f64>) -> nalgebra::Vector2<f64> {
        let dir = (pos - self.position).try_normalize(1e-10).unwrap_or(nalgebra::Vector2::zeros());
        self.position + self.radius * dir
    }

    fn minimum_spatio_temporal_distance(
        &self,
        pos: &nalgebra::Vector2<f64>,
        t: f64,
    ) -> f64 {
        let predicted = self.position + t * self.centroid_velocity;
        (predicted - pos).norm() - self.radius
    }

    fn is_dynamic(&self) -> bool {
        self.dynamic
    }

    fn centroid_velocity(&self) -> nalgebra::Vector2<f64> {
        self.centroid_velocity
    }

    fn set_centroid_velocity_internal(&mut self, vel: &nalgebra::Vector2<f64>) {
        if vel.norm() >= 0.001 {
            self.centroid_velocity = *vel;
            self.dynamic = true;
        }
    }
}

// ── Line Obstacle ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct LineObstacle {
    start: nalgebra::Vector2<f64>,
    end: nalgebra::Vector2<f64>,
    centroid: nalgebra::Vector2<f64>,
    dynamic: bool,
    centroid_velocity: nalgebra::Vector2<f64>,
}

impl LineObstacle {
    pub fn from_points(x1: f64, y1: f64, x2: f64, y2: f64) -> Self {
        let start = nalgebra::Vector2::new(x1, y1);
        let end = nalgebra::Vector2::new(x2, y2);
        let centroid = 0.5 * (start + end);
        Self {
            start,
            end,
            centroid,
            dynamic: false,
            centroid_velocity: nalgebra::Vector2::zeros(),
        }
    }

    pub fn from_vectors(start: nalgebra::Vector2<f64>, end: nalgebra::Vector2<f64>) -> Self {
        let centroid = 0.5 * (start + end);
        Self {
            start,
            end,
            centroid,
            dynamic: false,
            centroid_velocity: nalgebra::Vector2::zeros(),
        }
    }

    pub fn start(&self) -> &nalgebra::Vector2<f64> {
        &self.start
    }

    pub fn end(&self) -> &nalgebra::Vector2<f64> {
        &self.end
    }
}

impl Obstacle for LineObstacle {
    fn centroid(&self) -> &nalgebra::Vector2<f64> {
        &self.centroid
    }

    fn check_collision(&self, pos: &nalgebra::Vector2<f64>, min_dist: f64) -> bool {
        self.minimum_distance_to_point(pos) <= min_dist
    }

    fn check_line_intersection(
        &self,
        line_start: &nalgebra::Vector2<f64>,
        line_end: &nalgebra::Vector2<f64>,
        _min_dist: f64,
    ) -> bool {
        let (intersects, _) = check_line_segments_intersection_2d(
            line_start,
            line_end,
            &self.start,
            &self.end,
        );
        intersects
    }

    fn minimum_distance_to_point(&self, pos: &nalgebra::Vector2<f64>) -> f64 {
        distance_point_to_segment_2d(pos, &self.start, &self.end)
    }

    fn minimum_distance_to_line(
        &self,
        line_start: &nalgebra::Vector2<f64>,
        line_end: &nalgebra::Vector2<f64>,
    ) -> f64 {
        distance_segment_to_segment_2d(&self.start, &self.end, line_start, line_end)
    }

    fn minimum_distance_to_polygon(&self, polygon: &[nalgebra::Vector2<f64>]) -> f64 {
        distance_segment_to_polygon_2d(&self.start, &self.end, polygon)
    }

    fn closest_point(&self, pos: &nalgebra::Vector2<f64>) -> nalgebra::Vector2<f64> {
        closest_point_on_line_segment_2d(pos, &self.start, &self.end)
    }

    fn minimum_spatio_temporal_distance(
        &self,
        pos: &nalgebra::Vector2<f64>,
        t: f64,
    ) -> f64 {
        let offset = t * self.centroid_velocity;
        distance_point_to_segment_2d(pos, &(self.start + offset), &(self.end + offset))
    }

    fn is_dynamic(&self) -> bool {
        self.dynamic
    }

    fn centroid_velocity(&self) -> nalgebra::Vector2<f64> {
        self.centroid_velocity
    }

    fn set_centroid_velocity_internal(&mut self, vel: &nalgebra::Vector2<f64>) {
        if vel.norm() >= 0.001 {
            self.centroid_velocity = *vel;
            self.dynamic = true;
        }
    }
}

// ── Polygon Obstacle ───────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PolygonObstacle {
    vertices: Vec<nalgebra::Vector2<f64>>,
    centroid: nalgebra::Vector2<f64>,
    dynamic: bool,
    centroid_velocity: nalgebra::Vector2<f64>,
}

impl PolygonObstacle {
    pub fn from_vertices(vertices: Vec<nalgebra::Vector2<f64>>) -> Self {
        let centroid = if vertices.is_empty() {
            nalgebra::Vector2::zeros()
        } else {
            let sum: nalgebra::Vector2<f64> = vertices.iter().sum();
            sum / vertices.len() as f64
        };
        Self {
            vertices,
            centroid,
            dynamic: false,
            centroid_velocity: nalgebra::Vector2::zeros(),
        }
    }

    pub fn vertices(&self) -> &[nalgebra::Vector2<f64>] {
        &self.vertices
    }

    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }
}

impl Obstacle for PolygonObstacle {
    fn centroid(&self) -> &nalgebra::Vector2<f64> {
        &self.centroid
    }

    fn check_collision(&self, pos: &nalgebra::Vector2<f64>, min_dist: f64) -> bool {
        self.minimum_distance_to_point(pos) <= min_dist
    }

    fn check_line_intersection(
        &self,
        line_start: &nalgebra::Vector2<f64>,
        line_end: &nalgebra::Vector2<f64>,
        _min_dist: f64,
    ) -> bool {
        let count = self.vertices.len();
        if count == 0 {
            return false;
        }
        for i in 0..count {
            let next = if i + 1 < count { i + 1 } else { 0 };
            let (intersects, _) = check_line_segments_intersection_2d(
                line_start,
                line_end,
                &self.vertices[i],
                &self.vertices[next],
            );
            if intersects {
                return true;
            }
        }
        false
    }

    fn minimum_distance_to_point(&self, pos: &nalgebra::Vector2<f64>) -> f64 {
        distance_point_to_polygon_2d(pos, &self.vertices)
    }

    fn minimum_distance_to_line(
        &self,
        line_start: &nalgebra::Vector2<f64>,
        line_end: &nalgebra::Vector2<f64>,
    ) -> f64 {
        distance_segment_to_polygon_2d(line_start, line_end, &self.vertices)
    }

    fn minimum_distance_to_polygon(&self, polygon: &[nalgebra::Vector2<f64>]) -> f64 {
        distance_polygon_to_polygon_2d(&self.vertices, polygon)
    }

    fn closest_point(&self, pos: &nalgebra::Vector2<f64>) -> nalgebra::Vector2<f64> {
        let mut min_dist = f64::INFINITY;
        let mut closest = pos.clone();
        let count = self.vertices.len();
        if count == 0 {
            return closest;
        }
        for i in 0..count {
            let next = if i + 1 < count { i + 1 } else { 0 };
            let cp = closest_point_on_line_segment_2d(
                pos, &self.vertices[i], &self.vertices[next],
            );
            let d = (cp - pos).norm();
            if d < min_dist {
                min_dist = d;
                closest = cp;
            }
        }
        closest
    }

    fn minimum_spatio_temporal_distance(
        &self,
        pos: &nalgebra::Vector2<f64>,
        t: f64,
    ) -> f64 {
        let offset = t * self.centroid_velocity;
        let shifted: Vec<_> = self.vertices.iter().map(|v| v + offset).collect();
        distance_point_to_polygon_2d(pos, &shifted)
    }

    fn is_dynamic(&self) -> bool {
        self.dynamic
    }

    fn centroid_velocity(&self) -> nalgebra::Vector2<f64> {
        self.centroid_velocity
    }

    fn set_centroid_velocity_internal(&mut self, vel: &nalgebra::Vector2<f64>) {
        if vel.norm() >= 0.001 {
            self.centroid_velocity = *vel;
            self.dynamic = true;
        }
    }
}

// ── Pill (Capsule) Obstacle ────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PillObstacle {
    start: nalgebra::Vector2<f64>,
    end: nalgebra::Vector2<f64>,
    centroid: nalgebra::Vector2<f64>,
    radius: f64,
    dynamic: bool,
    centroid_velocity: nalgebra::Vector2<f64>,
}

impl PillObstacle {
    pub fn from_points(x1: f64, y1: f64, x2: f64, y2: f64, radius: f64) -> Self {
        let start = nalgebra::Vector2::new(x1, y1);
        let end = nalgebra::Vector2::new(x2, y2);
        let centroid = 0.5 * (start + end);
        Self {
            start,
            end,
            centroid,
            radius,
            dynamic: false,
            centroid_velocity: nalgebra::Vector2::zeros(),
        }
    }

    pub fn radius(&self) -> f64 {
        self.radius
    }
}

impl Obstacle for PillObstacle {
    fn centroid(&self) -> &nalgebra::Vector2<f64> {
        &self.centroid
    }

    fn check_collision(&self, pos: &nalgebra::Vector2<f64>, min_dist: f64) -> bool {
        self.minimum_distance_to_point(pos) <= min_dist
    }

    fn check_line_intersection(
        &self,
        line_start: &nalgebra::Vector2<f64>,
        line_end: &nalgebra::Vector2<f64>,
        _min_dist: f64,
    ) -> bool {
        let (intersects, _) = check_line_segments_intersection_2d(
            line_start, line_end, &self.start, &self.end,
        );
        intersects
    }

    fn minimum_distance_to_point(&self, pos: &nalgebra::Vector2<f64>) -> f64 {
        distance_point_to_segment_2d(pos, &self.start, &self.end) - self.radius
    }

    fn minimum_distance_to_line(
        &self,
        line_start: &nalgebra::Vector2<f64>,
        line_end: &nalgebra::Vector2<f64>,
    ) -> f64 {
        distance_segment_to_segment_2d(&self.start, &self.end, line_start, line_end) - self.radius
    }

    fn minimum_distance_to_polygon(&self, polygon: &[nalgebra::Vector2<f64>]) -> f64 {
        distance_segment_to_polygon_2d(&self.start, &self.end, polygon) - self.radius
    }

    fn closest_point(&self, pos: &nalgebra::Vector2<f64>) -> nalgebra::Vector2<f64> {
        let closed = closest_point_on_line_segment_2d(pos, &self.start, &self.end);
        let dir = (pos - &closed).try_normalize(1e-10).unwrap_or(nalgebra::Vector2::zeros());
        closed + self.radius * dir
    }

    fn minimum_spatio_temporal_distance(
        &self,
        pos: &nalgebra::Vector2<f64>,
        t: f64,
    ) -> f64 {
        let offset = t * self.centroid_velocity;
        distance_point_to_segment_2d(pos, &(self.start + offset), &(self.end + offset)) - self.radius
    }

    fn is_dynamic(&self) -> bool {
        self.dynamic
    }

    fn centroid_velocity(&self) -> nalgebra::Vector2<f64> {
        self.centroid_velocity
    }

    fn set_centroid_velocity_internal(&mut self, vel: &nalgebra::Vector2<f64>) {
        if vel.norm() >= 0.001 {
            self.centroid_velocity = *vel;
            self.dynamic = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_obstacle_distance() {
        let obs = PointObstacle::with_position(0.0, 0.0);
        let pos = nalgebra::Vector2::new(3.0, 4.0);
        assert!((obs.minimum_distance_to_point(&pos) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_circular_obstacle_distance() {
        let obs = CircularObstacle::with_position(0.0, 0.0, 1.0);
        let pos = nalgebra::Vector2::new(3.0, 0.0);
        assert!((obs.minimum_distance_to_point(&pos) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_line_obstacle_distance() {
        let obs = LineObstacle::from_points(0.0, 0.0, 1.0, 0.0);
        let pos = nalgebra::Vector2::new(0.5, 1.0);
        assert!((obs.minimum_distance_to_point(&pos) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_pill_obstacle_distance() {
        let obs = PillObstacle::from_points(0.0, 0.0, 1.0, 0.0, 0.5);
        let pos = nalgebra::Vector2::new(0.5, 1.0);
        assert!((obs.minimum_distance_to_point(&pos) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_dynamic_flag() {
        let mut obs = PointObstacle::with_position(0.0, 0.0);
        assert!(!obs.is_dynamic());
        let vel = nalgebra::Vector2::new(0.5, 0.0);
        obs.set_centroid_velocity(&vel);
        assert!(obs.is_dynamic());
    }

    #[test]
    fn test_obst_container() {
        let mut c: ObstContainer = Vec::new();
        c.push(Box::new(PointObstacle::with_position(0.0, 0.0)));
        c.push(Box::new(CircularObstacle::with_position(1.0, 1.0, 0.5)));
        assert_eq!(c.len(), 2);
    }
}
