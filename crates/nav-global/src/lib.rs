//! Global path planning algorithms ported from teb_local_planner C++.
//!
//! Provides A*, Hybrid A*, D* Lite, and Informed RRT* search algorithms
//! for 2D robot navigation with support for vehicle kinematics and dynamic replanning.

pub mod astar;
pub mod hybrid_astar;
pub mod dstar_lite;
pub mod rrt_star;

// Re-export A* types
pub use astar::{AStarConfig, AStarNode, AStarResult};
pub use astar::astar_search;

// Re-export Hybrid A* types
pub use hybrid_astar::{
    HybridAStarConfig, HybridAStarNode, HybridAStarResult, HybridStateKey,
    MotionPrimitive, MotionDirection,
};
pub use hybrid_astar::hybrid_astar_search;

// Re-export D* Lite types
pub use dstar_lite::{DStarLiteConfig, DStarLiteNode, DStarLiteResult};
pub use dstar_lite::{dstar_lite_new, dstar_lite_replan};

// Re-export RRT* types
pub use rrt_star::{RRTStarConfig, RRTStarNode, RRTStarResult};
pub use rrt_star::rrt_star_search;

use nalgebra::Vector2;

/// Costmap-based collision checking interface.
///
/// Implement this trait to provide costmap access for all search algorithms.
/// The costmap is a 2D grid where each cell stores a cost value representing
/// traversal difficulty (e.g., distance to nearest obstacle).
pub trait Costmap2d {
    /// Cell size in meters (costmap resolution).
    fn resolution(&self) -> f64;

    /// Width of the costmap in cells.
    fn width(&self) -> usize;

    /// Height of the costmap in cells.
    fn height(&self) -> usize;

    /// Convert world coordinates to costmap cell indices.
    /// Returns `None` if the coordinates are out of bounds.
    fn world_to_cell(&self, x: f64, y: f64) -> Option<(usize, usize)> {
        let res = self.resolution();
        let col = (x / res).floor() as usize;
        let row = (y / res).floor() as usize;
        if col < self.width() && row < self.height() {
            Some((col, row))
        } else {
            None
        }
    }

    /// Get the cost at world coordinates.
    /// Returns the cost of the containing cell, or `f64::MAX` if out of bounds.
    fn cost_at(&self, x: f64, y: f64) -> f64 {
        if let Some((col, row)) = self.world_to_cell(x, y) {
            self.cost_at_cell(col, row)
        } else {
            f64::MAX
        }
    }

    /// Get the cost at cell indices.
    fn cost_at_cell(&self, col: usize, row: usize) -> f64;

    /// Check if a cell is free (traversable).
    fn is_free(&self, x: f64, y: f64) -> bool {
        self.cost_at(x, y) < f64::MAX
    }

    /// Check if a line segment between two points is free.
    /// Uses Bresenham-style sampling for efficient line traversal.
    fn check_line_free(&self, x0: f64, y0: f64, x1: f64, y1: f64, min_dist: f64) -> bool {
        let dx = x1 - x0;
        let dy = y1 - y0;
        let dist = (dx * dx + dy * dy).sqrt();
        if dist < f64::EPSILON {
            return self.is_free(x0, y0);
        }
        let steps = (dist / min_dist).ceil().max(1.0) as usize;
        for i in 0..=steps {
            let t = i as f64 / steps as f64;
            let x = x0 + t * dx;
            let y = y0 + t * dy;
            if !self.is_free(x, y) {
                return false;
            }
        }
        true
    }

    /// Get the cost of a cell, clamped to [0, 1] for normalized queries.
    fn cost_normalized(&self, col: usize, row: usize) -> f64 {
        let c = self.cost_at_cell(col, row);
        c.clamp(0.0, 1.0)
    }
}

/// A 2D point alias for convenience.
pub type Point2 = Vector2<f64>;

/// Normalize an angle to [-pi, pi].
#[inline]
pub fn normalize_theta(theta: f64) -> f64 {
    let mut t = theta;
    while t > std::f64::consts::PI {
        t -= 2.0 * std::f64::consts::PI;
    }
    while t < -std::f64::consts::PI {
        t += 2.0 * std::f64::consts::PI;
    }
    t
}
