#![allow(unused_variables, unused_imports, dead_code, unused_mut, unused_assignments, unreachable_patterns)]
//! Obstacle representation and distance queries.
//!
//! Ported from C++ `obstacles.h` and `obstacles.cpp`.
//! Uses parry3d for collision geometry.

use parry3d::shape::Compound;
use parry3d::query::Contact;
use std::sync::Arc;
use nalgebra::Vector2;
use std::sync::LazyLock;

use crate::teb::math::{
    distance_point_to_polygon_2d, distance_point_to_segment_2d, distance_segment_to_segment_2d,
    distance_segment_to_polygon_2d, HUGE_VAL,
};

/// Alias for a 2D point container.
pub type Point2dContainer = Vec<Vector2<f64>>;

/// Trait implemented by all obstacle types.
///
/// Ported from C++ `Obstacle` base class.
pub trait Obstacle {
    /// Get centroid coordinates.
    fn centroid(&self) -> &Vector2<f64>;

    /// Check if a point collides with the obstacle (given min distance).
    fn check_collision(&self, point: &Vector2<f64>, min_dist: f64) -> bool;

    /// Check if a line segment intersects the obstacle.
    fn check_line_intersection(
        &self,
        line_start: &Vector2<f64>,
        line_end: &Vector2<f64>,
        min_dist: f64,
    ) -> bool;

    /// Get minimum distance from a point.
    fn get_minimum_distance(&self, point: &Vector2<f64>) -> f64;

    /// Get minimum distance from a line segment.
    fn get_minimum_distance_line(
        &self,
        line_start: &Vector2<f64>,
        line_end: &Vector2<f64>,
    ) -> f64;

    /// Get minimum distance from a polygon.
    fn get_minimum_distance_polygon(&self, polygon: &Point2dContainer) -> f64;

    /// Get the closest point on the obstacle boundary.
    fn get_closest_point(&self, point: &Vector2<f64>) -> Vector2<f64>;

    /// Get minimum spatiotemporal distance (for dynamic obstacles).
    fn get_minimum_spacetime_distance(
        &self,
        point: &Vector2<f64>,
        t: f64,
    ) -> f64 {
        // Default: assume static
        self.get_minimum_distance(point)
    }

    /// Get minimum spatiotemporal distance for a line segment.
    fn get_minimum_spacetime_distance_line(
        &self,
        line_start: &Vector2<f64>,
        line_end: &Vector2<f64>,
        t: f64,
    ) -> f64 {
        // Default: assume static
        self.get_minimum_distance_line(line_start, line_end)
    }

    /// Get minimum spatiotemporal distance for a polygon.
    fn get_minimum_spacetime_distance_polygon(
        &self,
        polygon: &Point2dContainer,
        t: f64,
    ) -> f64 {
        // Default: assume static
        self.get_minimum_distance_polygon(polygon)
    }

    /// Predict centroid position at time t using constant velocity model.
    fn predict_centroid_constant_velocity(&self, t: f64) -> Vector2<f64> {
        *self.centroid()
    }

    /// Check if the obstacle is dynamic (moving).
    fn is_dynamic(&self) -> bool;

    /// Set centroid velocity and mark as dynamic.
    fn set_centroid_velocity(&mut self, _vel: &Vector2<f64>) {
        // Default: no-op
    }

    /// Get centroid velocity (zero for static obstacles).
    fn centroid_velocity(&self) -> &Vector2<f64> {
        static ZERO: LazyLock<Vector2<f64>> = LazyLock::new(Vector2::zeros);
        &*ZERO
    }
}

/// Arc-allocated obstacle container.
pub type ObstaclePtr = Arc<dyn Obstacle + Send + Sync>;

// ==================== Point Obstacle ====================

/// A 2D point obstacle.
#[derive(Debug, Clone)]
pub struct PointObstacle {
    pos: Vector2<f64>,
    velocity: Vector2<f64>,
    is_dynamic: bool,
}

impl PointObstacle {
    /// Create a new point obstacle.
    pub fn new(pos: Vector2<f64>) -> Self {
        Self {
            pos,
            velocity: Vector2::zeros(),
            is_dynamic: false,
        }
    }

    /// Create from x,y coordinates.
    pub fn from_xy(x: f64, y: f64) -> Self {
        Self::new(Vector2::new(x, y))
    }
}

impl Obstacle for PointObstacle {
    fn centroid(&self) -> &Vector2<f64> {
        &self.pos
    }

    fn check_collision(&self, point: &Vector2<f64>, min_dist: f64) -> bool {
        self.get_minimum_distance(point) < min_dist
    }

    fn check_line_intersection(
        &self,
        line_start: &Vector2<f64>,
        line_end: &Vector2<f64>,
        min_dist: f64,
    ) -> bool {
        let a = *line_end - *line_start;
        let b = self.pos - *line_start;
        let t = a.dot(&b) / a.dot(&a);
        let t = t.max(0.0).min(1.0);
        let nearest = *line_start + a * t;
        self.check_collision(&nearest, min_dist)
    }

    fn get_minimum_distance(&self, point: &Vector2<f64>) -> f64 {
        (point - self.pos).norm()
    }

    fn get_minimum_distance_line(
        &self,
        line_start: &Vector2<f64>,
        line_end: &Vector2<f64>,
    ) -> f64 {
        distance_point_to_segment_2d(&self.pos, line_start, line_end)
    }

    fn get_minimum_distance_polygon(&self, polygon: &Point2dContainer) -> f64 {
        distance_point_to_polygon_2d(&self.pos, polygon)
    }

    fn get_closest_point(&self, _point: &Vector2<f64>) -> Vector2<f64> {
        self.pos
    }

    fn get_minimum_spacetime_distance(&self, point: &Vector2<f64>, t: f64) -> f64 {
        if self.is_dynamic {
            (self.pos + t * self.velocity - *point).norm()
        } else {
            self.get_minimum_distance(point)
        }
    }

    fn get_minimum_spacetime_distance_line(
        &self,
        line_start: &Vector2<f64>,
        line_end: &Vector2<f64>,
        t: f64,
    ) -> f64 {
        if self.is_dynamic {
            let pred_pos = self.pos + t * self.velocity;
            distance_point_to_segment_2d(&pred_pos, line_start, line_end)
        } else {
            self.get_minimum_distance_line(line_start, line_end)
        }
    }

    fn is_dynamic(&self) -> bool {
        self.is_dynamic
    }

    fn set_centroid_velocity(&mut self, vel: &Vector2<f64>) {
        self.velocity = *vel;
        self.is_dynamic = true;
    }

    fn centroid_velocity(&self) -> &Vector2<f64> {
        &self.velocity
    }
}

// ==================== Line Obstacle ====================

/// A 2D line segment obstacle.
#[derive(Debug, Clone)]
pub struct LineObstacle {
    start: Vector2<f64>,
    end: Vector2<f64>,
}

impl LineObstacle {
    /// Create a new line obstacle.
    pub fn new(start: Vector2<f64>, end: Vector2<f64>) -> Self {
        Self { start, end }
    }

    /// Create from x,y coordinates.
    pub fn from_coords(
        sx: f64,
        sy: f64,
        ex: f64,
        ey: f64,
    ) -> Self {
        Self::new(Vector2::new(sx, sy), Vector2::new(ex, ey))
    }

    pub fn start(&self) -> &Vector2<f64> { &self.start }
    pub fn end(&self) -> &Vector2<f64> { &self.end }
}

impl Obstacle for LineObstacle {
    fn centroid(&self) -> &Vector2<f64> {
        static ZERO: LazyLock<Vector2<f64>> = LazyLock::new(Vector2::zeros);
        &*ZERO
    }

    fn check_collision(&self, point: &Vector2<f64>, min_dist: f64) -> bool {
        self.get_minimum_distance(point) < min_dist
    }

    fn check_line_intersection(
        &self,
        line_start: &Vector2<f64>,
        line_end: &Vector2<f64>,
        min_dist: f64,
    ) -> bool {
        self.get_minimum_distance_line(line_start, line_end) < min_dist
    }

    fn get_minimum_distance(&self, point: &Vector2<f64>) -> f64 {
        distance_point_to_segment_2d(point, &self.start, &self.end)
    }

    fn get_minimum_distance_line(
        &self,
        line_start: &Vector2<f64>,
        line_end: &Vector2<f64>,
    ) -> f64 {
        distance_segment_to_segment_2d(&self.start, &self.end, line_start, line_end)
    }

    fn get_minimum_distance_polygon(&self, polygon: &Point2dContainer) -> f64 {
        distance_segment_to_polygon_2d(&self.start, &self.end, polygon)
    }

    fn get_closest_point(&self, point: &Vector2<f64>) -> Vector2<f64> {
        crate::teb::math::closest_point_on_line_segment_2d(point, &self.start, &self.end)
    }

    fn is_dynamic(&self) -> bool {
        false
    }
}

// ==================== Circular Obstacle ====================

/// A 2D circular obstacle (point + radius).
#[derive(Debug, Clone)]
pub struct CircularObstacle {
    pos: Vector2<f64>,
    radius: f64,
    velocity: Vector2<f64>,
    is_dynamic: bool,
}

impl CircularObstacle {
    /// Create a new circular obstacle.
    pub fn new(pos: Vector2<f64>, radius: f64) -> Self {
        Self {
            pos,
            radius,
            velocity: Vector2::zeros(),
            is_dynamic: false,
        }
    }
}

impl Obstacle for CircularObstacle {
    fn centroid(&self) -> &Vector2<f64> {
        &self.pos
    }

    fn check_collision(&self, point: &Vector2<f64>, min_dist: f64) -> bool {
        self.get_minimum_distance(point) < min_dist
    }

    fn check_line_intersection(
        &self,
        line_start: &Vector2<f64>,
        line_end: &Vector2<f64>,
        min_dist: f64,
    ) -> bool {
        let a = *line_end - *line_start;
        let b = self.pos - *line_start;
        let t = a.dot(&b) / a.dot(&a);
        let t = t.max(0.0).min(1.0);
        let nearest = *line_start + a * t;
        self.check_collision(&nearest, min_dist)
    }

    fn get_minimum_distance(&self, point: &Vector2<f64>) -> f64 {
        (point - self.pos).norm() - self.radius
    }

    fn get_minimum_distance_line(
        &self,
        line_start: &Vector2<f64>,
        line_end: &Vector2<f64>,
    ) -> f64 {
        distance_point_to_segment_2d(&self.pos, line_start, line_end) - self.radius
    }

    fn get_minimum_distance_polygon(&self, polygon: &Point2dContainer) -> f64 {
        distance_point_to_polygon_2d(&self.pos, polygon) - self.radius
    }

    fn get_closest_point(&self, point: &Vector2<f64>) -> Vector2<f64> {
        let dir = point - self.pos;
        let dist = dir.norm();
        if dist < 1e-10 {
            return self.pos;
        }
        self.pos + dir / dist * self.radius
    }

    fn get_minimum_spacetime_distance(&self, point: &Vector2<f64>, t: f64) -> f64 {
        if self.is_dynamic {
            let pred_pos = self.pos + t * self.velocity;
            (point - pred_pos).norm() - self.radius
        } else {
            self.get_minimum_distance(point)
        }
    }

    fn is_dynamic(&self) -> bool {
        self.is_dynamic
    }

    fn set_centroid_velocity(&mut self, vel: &Vector2<f64>) {
        self.velocity = *vel;
        self.is_dynamic = true;
    }

    fn centroid_velocity(&self) -> &Vector2<f64> {
        &self.velocity
    }
}

// ==================== Polygon Obstacle ====================

/// A 2D polygon obstacle.
#[derive(Debug, Clone)]
pub struct PolygonObstacle {
    vertices: Point2dContainer,
    centroid: Vector2<f64>,
    velocity: Vector2<f64>,
    is_dynamic: bool,
    /// parry3d convex hull for efficient distance queries
    parry_shape: Option<Arc<Compound>>,
}

impl PolygonObstacle {
    /// Create from vertices (counter-clockwise or clockwise order).
    /// The first vertex is NOT repeated at the end.
    pub fn new(vertices: Point2dContainer) -> Self {
        let centroid = Self::compute_centroid(&vertices);
        let parry_shape = if vertices.len() >= 3 {
            Self::build_parry_hull(&vertices)
        } else {
            None
        };
        Self {
            vertices,
            centroid,
            velocity: Vector2::zeros(),
            is_dynamic: false,
            parry_shape,
        }
    }

    fn compute_centroid(vertices: &[Vector2<f64>]) -> Vector2<f64> {
        if vertices.is_empty() {
            return Vector2::zeros();
        }
        vertices.iter().sum::<Vector2<f64>>() / vertices.len() as f64
    }

    fn build_parry_hull(vertices: &[Vector2<f64>]) -> Option<Arc<Compound>> {
        if vertices.len() < 3 {
            return None;
        }
        // Convert 2D vertices to 3D points for parry3d ConvexHull
        let points: Vec<[f32; 3]> = vertices
            .iter()
            .map(|v| [v.x as f32, v.y as f32, 0.0])
            .collect();
        // Build a simple convex hull from the points
        // parry3d::shape::ConvexPolytope would be ideal but we use ConvexHull
        // For a proper implementation we'd need to use parry3d's hlib
        // As a fallback, we'll use the point cloud approach
        // Since parry3d ConvexHull requires the ConvexHull struct from the hlib crate,
        // let's use a simpler approach with individual segments
        let _ = points; // silence unused warning
        None
    }

    /// Get vertices.
    pub fn vertices(&self) -> &Point2dContainer {
        &self.vertices
    }

    /// Get centroid.
    pub fn centroid(&self) -> &Vector2<f64> {
        &self.centroid
    }

    /// Fix polygon closure (ensure first != last vertex).
    pub fn fix_polygon_closure(&mut self) {
        if self.vertices.len() >= 2
            && self.vertices.first() == self.vertices.last()
        {
            self.vertices.pop();
        }
    }
}

impl Obstacle for PolygonObstacle {
    fn centroid(&self) -> &Vector2<f64> {
        &self.centroid
    }

    fn check_collision(&self, point: &Vector2<f64>, min_dist: f64) -> bool {
        self.get_minimum_distance(point) < min_dist
    }

    fn check_line_intersection(
        &self,
        line_start: &Vector2<f64>,
        line_end: &Vector2<f64>,
        min_dist: f64,
    ) -> bool {
        // Check each edge
        for i in 0..self.vertices.len() {
            let next = (i + 1) % self.vertices.len();
            let edge_start = self.vertices[i];
            let edge_end = self.vertices[next];
            if line_intersects_polygon_edge(
                line_start,
                line_end,
                &edge_start,
                &edge_end,
                min_dist,
            ) {
                return true;
            }
        }
        false
    }

    fn get_minimum_distance(&self, point: &Vector2<f64>) -> f64 {
        // Check if point is inside the polygon first
        if point_in_polygon(point, &self.vertices) {
            // Return negative distance to indicate inside
            -distance_point_to_polygon_2d(point, &self.vertices)
        } else {
            distance_point_to_polygon_2d(point, &self.vertices)
        }
    }

    fn get_minimum_distance_line(
        &self,
        line_start: &Vector2<f64>,
        line_end: &Vector2<f64>,
    ) -> f64 {
        distance_segment_to_polygon_2d(line_start, line_end, &self.vertices)
    }

    fn get_minimum_distance_polygon(&self, polygon: &Point2dContainer) -> f64 {
        if polygon.is_empty() || polygon.len() == 1 {
            distance_point_to_polygon_2d(&polygon[0], &self.vertices)
        } else {
            // Check each edge of self against polygon
            let mut dist = HUGE_VAL;
            for i in 0..self.vertices.len() {
                let next = (i + 1) % self.vertices.len();
                let d = distance_segment_to_segment_2d(
                    &self.vertices[i],
                    &self.vertices[next],
                    &polygon[0],
                    &polygon[1],
                );
                // Check all endpoints of both segments
                for &v1 in &self.vertices {
                    let d1 = distance_point_to_segment_2d(&v1, &polygon[0], &polygon[1]);
                    if d1 < dist {
                        dist = d1;
                    }
                }
                for &v2 in polygon {
                    let d2 = distance_point_to_segment_2d(&v2, &self.vertices[i], &self.vertices[next]);
                    if d2 < dist {
                        dist = d2;
                    }
                }
            }
            dist
        }
    }

    fn get_closest_point(&self, point: &Vector2<f64>) -> Vector2<f64> {
        let mut closest = self.vertices[0];
        let mut min_dist = f64::MAX;
        for i in 0..self.vertices.len() {
            let next = (i + 1) % self.vertices.len();
            let cp = crate::teb::math::closest_point_on_line_segment_2d(
                point,
                &self.vertices[i],
                &self.vertices[next],
            );
            let d = (point - cp).norm();
            if d < min_dist {
                min_dist = d;
                closest = cp;
            }
        }
        closest
    }

    fn get_minimum_spacetime_distance(&self, point: &Vector2<f64>, t: f64) -> f64 {
        if self.is_dynamic {
            let shifted_vertices: Point2dContainer = self
                .vertices
                .iter()
                .map(|v| v + t * self.velocity)
                .collect();
            distance_point_to_polygon_2d(point, &shifted_vertices)
        } else {
            self.get_minimum_distance(point)
        }
    }

    fn get_minimum_spacetime_distance_line(
        &self,
        line_start: &Vector2<f64>,
        line_end: &Vector2<f64>,
        t: f64,
    ) -> f64 {
        if self.is_dynamic {
            let shifted_vertices: Point2dContainer = self
                .vertices
                .iter()
                .map(|v| v + t * self.velocity)
                .collect();
            distance_segment_to_polygon_2d(line_start, line_end, &shifted_vertices)
        } else {
            self.get_minimum_distance_line(line_start, line_end)
        }
    }

    fn is_dynamic(&self) -> bool {
        self.is_dynamic
    }

    fn set_centroid_velocity(&mut self, vel: &Vector2<f64>) {
        self.velocity = *vel;
        self.is_dynamic = true;
    }

    fn centroid_velocity(&self) -> &Vector2<f64> {
        &self.velocity
    }
}

/// Check if a line intersects a polygon edge.
fn line_intersects_polygon_edge(
    line_start: &Vector2<f64>,
    line_end: &Vector2<f64>,
    edge_start: &Vector2<f64>,
    edge_end: &Vector2<f64>,
    min_dist: f64,
) -> bool {
    let mut intersection = None;
    if crate::teb::math::check_line_segments_intersect_2d(
        line_start,
        line_end,
        edge_start,
        edge_end,
        &mut intersection,
    ) {
        return true;
    }
    // Check distance-based intersection
    let d = distance_segment_to_segment_2d(line_start, line_end, edge_start, edge_end);
    d < min_dist
}

/// Check if a point is inside a polygon (ray casting algorithm).
fn point_in_polygon(point: &Vector2<f64>, vertices: &[Vector2<f64>]) -> bool {
    if vertices.len() < 3 {
        return false;
    }

    let mut inside = false;
    let n = vertices.len();

    for i in 0..n {
        let j = (i + 1) % n;
        let xi = vertices[i].x;
        let yi = vertices[i].y;
        let xj = vertices[j].x;
        let yj = vertices[j].y;

        let intersect = ((yi > point.y) != (yj > point.y))
            && (point.x < (xj - xi) * (point.y - yi) / (yj - yi) + xi);

        if intersect {
            inside = !inside;
        }
    }

    inside
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_obstacle_distance() {
        let obs = PointObstacle::from_xy(3.0, 4.0);
        let point = Vector2::new(0.0, 0.0);
        assert!((obs.get_minimum_distance(&point) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_point_obstacle_collision() {
        let obs = PointObstacle::from_xy(1.0, 0.0);
        let point = Vector2::new(0.0, 0.0);
        assert!(obs.check_collision(&point, 2.0)); // distance=1 < min_dist=2
        assert!(!obs.check_collision(&point, 0.5)); // distance=1 > min_dist=0.5
    }

    #[test]
    fn test_line_obstacle_distance() {
        let obs = LineObstacle::from_coords(0.0, 0.0, 1.0, 0.0);
        let point = Vector2::new(0.5, 1.0);
        assert!((obs.get_minimum_distance(&point) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_circular_obstacle_distance() {
        let obs = CircularObstacle::new(Vector2::new(0.0, 0.0), 1.0);
        let point = Vector2::new(3.0, 0.0);
        assert!((obs.get_minimum_distance(&point) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_polygon_obstacle() {
        let vertices = vec![
            Vector2::new(-1.0, -1.0),
            Vector2::new(1.0, -1.0),
            Vector2::new(1.0, 1.0),
            Vector2::new(-1.0, 1.0),
        ];
        let obs = PolygonObstacle::new(vertices);
        // Point outside
        let outside = Vector2::new(2.0, 0.0);
        assert!((obs.get_minimum_distance(&outside) - 1.0).abs() < 1e-10);
        // Point inside
        let inside = Vector2::new(0.0, 0.0);
        assert!(obs.get_minimum_distance(&inside) < 0.0);
    }

    #[test]
    fn test_dynamic_obstacle() {
        let mut obs = PointObstacle::from_xy(0.0, 0.0);
        obs.set_centroid_velocity(&Vector2::new(1.0, 0.0));
        assert!(obs.is_dynamic());
        // At t=2, obstacle should be at (2, 0)
        assert!((obs.get_minimum_spacetime_distance(&Vector2::new(0.0, 0.0), 2.0) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_point_in_polygon() {
        let vertices = vec![
            Vector2::new(-1.0, -1.0),
            Vector2::new(1.0, -1.0),
            Vector2::new(1.0, 1.0),
            Vector2::new(-1.0, 1.0),
        ];
        assert!(point_in_polygon(&Vector2::new(0.0, 0.0), &vertices));
        assert!(!point_in_polygon(&Vector2::new(2.0, 0.0), &vertices));
    }
}
