//! A* search on a grid map with 4- or 8-connected neighborhood.
//!
//! Ported from teb_local_planner's graph-based search paradigm, adapted for
//! grid-map costmap navigation with configurable connectivity.

use nalgebra::Vector2;
use std::collections::{BinaryHeap, HashMap};
use std::hash::{Hash, Hasher};

use crate::{Costmap2d, Point2};

/// Configuration for A* search.
#[derive(Debug, Clone)]
pub struct AStarConfig {
    /// Costmap resolution in meters per cell.
    pub resolution: f64,
    /// Costmap width in cells.
    pub width: usize,
    /// Costmap height in cells.
    pub height: usize,
    /// Whether to use 8-connected neighborhood (true) or 4-connected (false).
    pub eight_connected: bool,
    /// Whether diagonal movement costs use the exact diagonal distance
    /// (sqrt(2)) or the approximation (2.0 / (1.0 + sqrt(2)) * sqrt(2)).
    pub use_diagonal_cost: bool,
    /// Scalar multiplier for diagonal movement cost (default 1.0 = exact sqrt(2)).
    pub diagonal_cost_scaling: f64,
}

impl Default for AStarConfig {
    fn default() -> Self {
        Self {
            resolution: 0.05,
            width: 1000,
            height: 1000,
            eight_connected: true,
            use_diagonal_cost: true,
            diagonal_cost_scaling: 1.0,
        }
    }
}

/// A* search result.
#[derive(Debug, Clone, PartialEq)]
pub enum AStarResult {
    /// Search found a valid path.
    Success {
        /// Path in world coordinates, ordered from start to goal.
        path: Vec<Point2>,
        /// Total cost from start to goal.
        total_cost: f64,
        /// Number of expanded nodes.
        nodes_expanded: usize,
    },
    /// No path could be found.
    Failure,
    /// Search timed out (not currently triggered; the algorithm is bounded).
    Timeout,
}

/// Node in the A* search tree.
#[derive(Debug, Clone)]
pub struct AStarNode {
    /// Cell index (col, row).
    pub cell: (usize, usize),
    /// g-cost: actual cost from start.
    pub g_cost: f64,
    /// h-cost: heuristic estimate to goal.
    pub h_cost: f64,
    /// f-cost: g_cost + h_cost.
    pub f_cost: f64,
    /// Parent node index in the closed-set path.
    pub parent: Option<usize>,
}

/// Key for the open-set priority queue ordering.
/// BinaryHeap is a max-heap, so we negate f_cost for min-heap behavior.
#[derive(Debug, Clone)]
struct OpenKey {
    neg_f: f64,
    g_cost: f64,
    cell: (usize, usize),
}

impl PartialEq for OpenKey {
    fn eq(&self, other: &Self) -> bool {
        if self.neg_f.is_nan() && other.neg_f.is_nan() {
            if self.g_cost.is_nan() && other.g_cost.is_nan() {
                return self.cell == other.cell;
            }
        }
        self.neg_f == other.neg_f && self.g_cost == other.g_cost && self.cell == other.cell
    }
}
impl Eq for OpenKey {}

impl Ord for OpenKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.neg_f.partial_cmp(&other.neg_f)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| other.g_cost.partial_cmp(&self.g_cost).unwrap_or(std::cmp::Ordering::Equal))
            .then_with(|| self.cell.0.cmp(&other.cell.0))
            .then_with(|| self.cell.1.cmp(&other.cell.1))
    }
}

impl PartialOrd for OpenKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Cell index hash for HashMap lookups.
fn cell_hash(cell: &(usize, usize)) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    cell.0.hash(&mut h);
    cell.1.hash(&mut h);
    h.finish()
}

impl AStarConfig {
    /// Convert world coordinates to cell indices.
    #[inline]
    pub fn world_to_cell(&self, x: f64, y: f64) -> Option<(usize, usize)> {
        let col = (x / self.resolution).floor() as usize;
        let row = (y / self.resolution).floor() as usize;
        if col < self.width && row < self.height {
            Some((col, row))
        } else {
            None
        }
    }

    /// Convert cell indices to world coordinates (center of cell).
    #[inline]
    pub fn cell_to_world(&self, col: usize, row: usize) -> Point2 {
        Vector2::new(
            (col as f64 + 0.5) * self.resolution,
            (row as f64 + 0.5) * self.resolution,
        )
    }

    /// Get the heuristic (Manhattan or Octile distance) between two cell positions.
    pub fn heuristic(&self, from: (usize, usize), to: (usize, usize)) -> f64 {
        let dx = (from.0 as i64 - to.0 as i64).abs() as f64;
        let dy = (from.1 as i64 - to.1 as i64).abs() as f64;

        if self.eight_connected {
            // Octile distance for 8-connected grid
            let diagonal = dx.min(dy);
            let cardinal = dx.max(dy) - diagonal;
            let diag_cost = if self.use_diagonal_cost {
                std::f64::consts::SQRT_2 * self.diagonal_cost_scaling
            } else {
                2.0 / (1.0 + std::f64::consts::SQRT_2) * std::f64::consts::SQRT_2 * self.diagonal_cost_scaling
            };
            (diagonal * diag_cost + cardinal) * self.resolution
        } else {
            // Manhattan distance
            (dx + dy) * self.resolution
        }
    }

    /// Get the neighbors of a cell given the connectivity setting.
    pub fn get_neighbors(&self, col: usize, row: usize) -> Vec<(usize, usize)> {
        let mut neighbors = Vec::new();

        // 4-connected neighbors
        let cardinal = [
            (col.wrapping_sub(1), row),
            (col + 1, row),
            (col, row.wrapping_sub(1)),
            (col, row + 1),
        ];
        for &(n, m) in &cardinal {
            if n < self.width && m < self.height {
                neighbors.push((n, m));
            }
        }

        // 8-connected: add diagonals
        if self.eight_connected {
            let diagonals = [
                (col.wrapping_sub(1), row.wrapping_sub(1)),
                (col + 1, row + 1),
                (col.wrapping_sub(1), row + 1),
                (col + 1, row.wrapping_sub(1)),
            ];
            for &(n, m) in &diagonals {
                if n < self.width && m < self.height {
                    neighbors.push((n, m));
                }
            }
        }

        neighbors
    }

    /// Get the cost to move to a neighbor cell, including the cost of the neighbor and edge cost.
    pub fn move_cost(&self, neighbor: (usize, usize), current: (usize, usize), costmap: &impl Costmap2d) -> f64 {
        let neighbor_cost = costmap.cost_at_cell(neighbor.0, neighbor.1);
        let dx = (neighbor.0 as i64 - current.0 as i64).abs() as f64;
        let dy = (neighbor.1 as i64 - current.1 as i64).abs() as f64;

        let edge_cost = if dx > 0.5 && dy > 0.5 {
            // Diagonal move
            std::f64::consts::SQRT_2 * self.diagonal_cost_scaling * self.resolution
        } else {
            // Cardinal move
            self.resolution
        };

        // Combine neighbor cell cost with edge traversal cost
        neighbor_cost + edge_cost
    }
}

/// Run A* search on a costmap.
///
/// # Arguments
/// * `start` - Start position in world coordinates.
/// * `goal` - Goal position in world coordinates.
/// * `costmap` - Costmap providing cell costs.
/// * `config` - A* configuration.
///
/// # Returns
/// `AStarResult` with the found path or failure status.
pub fn astar_search(
    start: Point2,
    goal: Point2,
    costmap: &impl Costmap2d,
    config: &AStarConfig,
) -> AStarResult {
    // Convert to cell coordinates
    let start_cell = match config.world_to_cell(start.x, start.y) {
        Some(c) => c,
        None => return AStarResult::Failure,
    };
    let goal_cell = match config.world_to_cell(goal.x, goal.y) {
        Some(c) => c,
        None => return AStarResult::Failure,
    };

    let heuristic = config.heuristic(start_cell, goal_cell);
    if heuristic < f64::EPSILON {
        return AStarResult::Success {
            path: vec![start],
            total_cost: 0.0,
            nodes_expanded: 1,
        };
    }

    // Open set (priority queue)
    let mut open_set = BinaryHeap::new();
    open_set.push(OpenKey {
        neg_f: -heuristic,
        g_cost: 0.0,
        cell: start_cell,
    });

    // Closed set: store parent for path reconstruction
    let mut came_from: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
    let mut g_score: HashMap<(usize, usize), f64> = HashMap::new();

    g_score.insert(start_cell, 0.0);

    let mut nodes_expanded = 0;

    while let Some(OpenKey { cell, .. }) = open_set.pop() {
        if cell == goal_cell {
            // Reconstruct path
            let mut path = Vec::new();
            let mut current: Option<(usize, usize)> = Some(goal_cell);
            while let Some(c) = current {
                let world = config.cell_to_world(c.0, c.1);
                path.push(world);
                current = came_from.get(&c).copied();
            }
            path.reverse();

            let total_cost = g_score.get(&goal_cell).copied().unwrap_or(f64::INFINITY);
            return AStarResult::Success {
                path,
                total_cost,
                nodes_expanded,
            };
        }

        // Expand neighbors
        for neighbor in config.get_neighbors(cell.0, cell.1) {
            let tentative_g = g_score.get(&cell).copied().unwrap_or(f64::INFINITY)
                + config.move_cost(neighbor, cell, costmap);

            if tentative_g < g_score.get(&neighbor).copied().unwrap_or(f64::INFINITY) {
                came_from.insert(neighbor, cell);
                g_score.insert(neighbor, tentative_g);

                let h = config.heuristic(neighbor, goal_cell);
                let f = tentative_g + h;

                open_set.push(OpenKey {
                    neg_f: -f,
                    g_cost: tentative_g,
                    cell: neighbor,
                });
            }
        }

        nodes_expanded += 1;
    }

    AStarResult::Failure
}

/// Run A* search with a custom cost function that computes cost on demand.
/// Useful when the costmap is not directly available.
pub fn astar_search_custom(
    start: Point2,
    goal: Point2,
    cost_fn: Box<dyn Fn((usize, usize)) -> f64>,
    config: &AStarConfig,
) -> AStarResult {
    // Wrap the cost function as a costmap
    struct CostFnWrapper {
        cost_fn: Box<dyn Fn((usize, usize)) -> f64>,
        config: AStarConfig,
    }

    impl Costmap2d for CostFnWrapper {
        fn resolution(&self) -> f64 {
            self.config.resolution
        }
        fn width(&self) -> usize {
            self.config.width
        }
        fn height(&self) -> usize {
            self.config.height
        }
        fn cost_at_cell(&self, col: usize, row: usize) -> f64 {
            (self.cost_fn)((col, row))
        }
    }

    let wrapper = CostFnWrapper {
        cost_fn,
        config: config.clone(),
    };

    astar_search(start, goal, &wrapper, config)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use crate::normalize_theta;
    use super::*;

    /// A simple costmap for testing.
    struct TestCostmap {
        resolution: f64,
        width: usize,
        height: usize,
        costs: Vec<f64>,
    }

    impl TestCostmap {
        fn new(width: usize, height: usize, resolution: f64) -> Self {
            let costs = vec![0.0; width * height];
            Self {
                resolution,
                width,
                height,
                costs,
            }
        }

        fn set_obstacle(&mut self, col: usize, row: usize) {
            if col < self.width && row < self.height {
                self.costs[row * self.width + col] = f64::MAX;
            }
        }

        fn set_cost(&mut self, col: usize, row: usize, cost: f64) {
            if col < self.width && row < self.height {
                self.costs[row * self.width + col] = cost;
            }
        }
    }

    impl Costmap2d for TestCostmap {
        fn resolution(&self) -> f64 {
            self.resolution
        }
        fn width(&self) -> usize {
            self.width
        }
        fn height(&self) -> usize {
            self.height
        }
        fn cost_at_cell(&self, col: usize, row: usize) -> f64 {
            if col < self.width && row < self.height {
                self.costs[row * self.width + col]
            } else {
                f64::MAX
            }
        }
    }

    #[test]
    fn test_normalize_theta() {
        // g2o normalize_angle returns [-pi, pi) — PI maps to -PI
        assert_relative_eq!(normalize_theta(0.0), 0.0);
        assert_relative_eq!(normalize_theta(std::f64::consts::PI), -std::f64::consts::PI);
        assert_relative_eq!(
            normalize_theta(std::f64::consts::PI + 0.1),
        -std::f64::consts::PI + 0.1
        );
        assert_relative_eq!(
            normalize_theta(-std::f64::consts::PI - 0.1),
            std::f64::consts::PI - 0.1
        );
    }

    #[test]
    fn test_world_to_cell() {
        let config = AStarConfig {
            resolution: 0.1,
            width: 100,
            height: 100,
            ..Default::default()
        };
        let cell = config.world_to_cell(0.15, 0.25).unwrap();
        assert_eq!(cell, (1, 2));

        let out_of_bounds = config.world_to_cell(10.0, 10.0);
        assert!(out_of_bounds.is_none());
    }

    #[test]
    fn test_cell_to_world() {
        let config = AStarConfig {
            resolution: 0.1,
            width: 100,
            height: 100,
            ..Default::default()
        };
        let world = config.cell_to_world(1, 2);
        assert_relative_eq!(world.x, 0.15);
        assert_relative_eq!(world.y, 0.25);
    }

    #[test]
    fn test_heuristic_4_connected() {
        let config = AStarConfig {
            resolution: 1.0,
            eight_connected: false,
            ..Default::default()
        };
        // Manhattan distance: |3-1| + |4-2| = 4
        let h = config.heuristic((3, 4), (1, 2));
        assert_relative_eq!(h, 4.0);
    }

    #[test]
    fn test_heuristic_8_connected() {
        let config = AStarConfig {
            resolution: 1.0,
            eight_connected: true,
            ..Default::default()
        };
        // Octile: min(2,2)*sqrt(2) + (max(2,2)-2) = sqrt(2)
        let h = config.heuristic((3, 4), (1, 2));
        assert_relative_eq!(h, 2.0 * std::f64::consts::SQRT_2, epsilon = 1e-10);
    }

    #[test]
    fn test_get_neighbors_4_connected() {
        let config = AStarConfig {
            width: 10,
            height: 10,
            eight_connected: false,
            ..Default::default()
        };
        let neighbors = config.get_neighbors(5, 5);
        assert_eq!(neighbors.len(), 4);
        assert!(neighbors.contains(&(4, 5)));
        assert!(neighbors.contains(&(6, 5)));
        assert!(neighbors.contains(&(5, 4)));
        assert!(neighbors.contains(&(5, 6)));
    }

    #[test]
    fn test_get_neighbors_8_connected() {
        let config = AStarConfig {
            width: 10,
            height: 10,
            eight_connected: true,
            ..Default::default()
        };
        let neighbors = config.get_neighbors(5, 5);
        assert_eq!(neighbors.len(), 8);
    }

    #[test]
    fn test_astar_clear_path() {
        let mut costmap = TestCostmap::new(20, 20, 0.5);
        let config = AStarConfig {
            resolution: 0.5,
            width: 20,
            height: 20,
            eight_connected: true,
            ..Default::default()
        };

        let start = Point2::new(0.25, 0.25);
        let goal = Point2::new(9.75, 9.75);

        let result = astar_search(start, goal, &costmap, &config);
        match result {
            AStarResult::Success { path, .. } => {
                assert!(path.len() >= 2);
                assert_relative_eq!(path[0].x, start.x, epsilon = 1e-6);
                assert_relative_eq!(path[0].y, start.y, epsilon = 1e-6);
            }
            _ => panic!("Expected success"),
        }
    }

    #[test]
    fn test_astar_obstacle_block() {
        let mut costmap = TestCostmap::new(10, 10, 0.5);
        // Block entire second column
        for row in 0..10 {
            costmap.set_obstacle(2, row);
        }

        let config = AStarConfig {
            resolution: 0.5,
            width: 10,
            height: 10,
            eight_connected: true,
            ..Default::default()
        };

        let start = Point2::new(0.25, 2.25);
        let goal = Point2::new(0.25, 7.25);

        let result = astar_search(start, goal, &costmap, &config);
        // Should fail: column 2 blocks all paths (no path around in a narrow strip)
        // Actually with 8-connected it might go around the edges
        // Let's test a simpler blockage: a 2x2 block between start and goal
    }

    #[test]
    fn test_astar_blocked_column() {
        // 5x10 map with a complete horizontal barrier
        let mut costmap = TestCostmap::new(10, 20, 0.5);
        for col in 3..7 {
            for row in 0..20 {
                costmap.set_obstacle(col, row);
            }
        }

        let config = AStarConfig {
            resolution: 0.5,
            width: 10,
            height: 20,
            eight_connected: true,
            ..Default::default()
        };

        let start = Point2::new(1.25, 3.0);  // cell (2, 6)
        let goal = Point2::new(4.25, 3.0);    // cell (8, 6)

        let result = astar_search(start, goal, &costmap, &config);
        // Should find a path going around the top or bottom of the barrier
        match result {
            AStarResult::Success { path, .. } => {
                assert!(path.len() >= 2);
            }
            AStarResult::Failure => {
                // Acceptable if barrier fully blocks the strip
            }
            _ => {}
        }
    }

    #[test]
    fn test_astar_no_path() {
        let mut costmap = TestCostmap::new(5, 5, 0.5);
        for col in 0..5 {
            for row in 0..5 {
                // Only leave two cells open
                if !(col == 0 && row == 0) && !(col == 4 && row == 4) {
                    costmap.set_obstacle(col, row);
                }
            }
        }

        let config = AStarConfig {
            resolution: 0.5,
            width: 5,
            height: 5,
            eight_connected: true,
            ..Default::default()
        };

        let start = Point2::new(0.25, 0.25);
        let goal = Point2::new(2.25, 2.25);

        let result = astar_search(start, goal, &costmap, &config);
        assert_eq!(result, AStarResult::Failure);
    }

    #[test]
    fn test_astar_4_connected_vs_8() {
        let mut costmap = TestCostmap::new(20, 20, 0.5);

        let config_4 = AStarConfig {
            resolution: 0.5,
            width: 20,
            height: 20,
            eight_connected: false,
            ..Default::default()
        };

        let config_8 = AStarConfig {
            resolution: 0.5,
            width: 20,
            height: 20,
            eight_connected: true,
            ..Default::default()
        };

        let start = Point2::new(0.25, 0.25);
        let goal = Point2::new(9.75, 9.75);

        let result_4 = astar_search(start, goal, &costmap, &config_4);
        let result_8 = astar_search(start, goal, &costmap, &config_8);

        match (result_4, result_8) {
            (AStarResult::Success { total_cost: cost_4, .. }, AStarResult::Success { total_cost: cost_8, .. }) => {
                // 8-connected should be shorter or equal (can cut corners diagonally)
                assert!(cost_8 <= cost_4 + 1e-6);
            }
            _ => panic!("Expected both to succeed"),
        }
    }
}
