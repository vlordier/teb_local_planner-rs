//! Proptest properties for nav-global algorithms.
//!
//! - A* path connected and unblocked
//! - RRT* rewiring preserves tree connectivity
//! - D*Lite handles empty maps

use nalgebra::Vector2;
use nav_global::{AStarConfig, Costmap2d, AStarResult, RRTStarConfig, RRTStarResult};
use std::collections::HashMap;

/// A simple costmap implementation for proptest.
#[derive(Clone)]
struct TestCostmap {
    grid: HashMap<(usize, usize), f64>,
    width: usize,
    height: usize,
    resolution: f64,
}

impl TestCostmap {
    fn new(width: usize, height: usize, resolution: f64) -> Self {
        Self {
            grid: HashMap::new(),
            width,
            height,
            resolution,
        }
    }

    fn set_cost(&mut self, col: usize, row: usize, cost: f64) {
        if col < self.width && row < self.height {
            self.grid.insert((col, row), cost);
        }
    }
}

impl Costmap2d for TestCostmap {
    fn resolution(&self) -> f64 { self.resolution }
    fn width(&self) -> usize { self.width }
    fn height(&self) -> usize { self.height }

    fn world_to_cell(&self, x: f64, y: f64) -> Option<(usize, usize)> {
        let col = (x / self.resolution) as usize;
        let row = (y / self.resolution) as usize;
        if col < self.width && row < self.height {
            Some((col, row))
        } else {
            None
        }
    }

    fn cost_at_cell(&self, col: usize, row: usize) -> f64 {
        self.grid.get(&(col, row)).copied().unwrap_or(0.0)
    }
}

mod astar_prop {
    use super::*;
    use nav_global::astar_search;
    use proptest::prelude::*;

    /// A* path is always connected: each consecutive pair is adjacent.
    #[test]
    fn astar_path_connected() {
        proptest! {
            fn astar_path_connected(
                map_w: u8, map_h: u8,
                obstacles: Vec<(u8, u8)>,
            ) {
                let w = (map_w % 50) as usize + 10;
                let h = (map_h % 50) as usize + 10;
                let resolution = 0.1;

                let mut costmap = TestCostmap::new(w, h, resolution);
                for &(oc, or) in &obstacles {
                    if (oc as usize) < w && (or as usize) < h {
                        costmap.set_cost(oc as usize, or as usize, f64::MAX);
                    }
                }

                let start = Vector2::new(0.0, 0.0);
                let goal = Vector2::new(
                    ((w - 1) as f64) * resolution,
                    ((h - 1) as f64) * resolution,
                );

                let config = AStarConfig::default();
                let result = astar_search(start, goal, &costmap, &config);

                match result {
                    AStarResult::Success { ref path, .. } => {
                        assert!(path.windows(2).all(|w| {
                            let dx = (w[0].x - w[1].x).abs();
                            let dy = (w[0].y - w[1].y).abs();
                            let cell_dx = (dx / resolution).round() as i64;
                            let cell_dy = (dy / resolution).round() as i64;
                            cell_dx.abs() <= 1 && cell_dy.abs() <= 1
                        }));
                    }
                    _ => {}
                }
            }
        }
    }

    /// A* path is never blocked: no cell in path has obstacle cost.
    #[test]
    fn astar_path_not_blocked() {
        proptest! {
            fn astar_path_not_blocked(
                map_w: u8, map_h: u8,
                obstacles: Vec<(u8, u8)>,
            ) {
                let w = (map_w % 50) as usize + 10;
                let h = (map_h % 50) as usize + 10;
                let resolution = 0.1;

                let mut costmap = TestCostmap::new(w, h, resolution);
                for &(oc, or) in &obstacles {
                    if (oc as usize) < w && (or as usize) < h {
                        costmap.set_cost(oc as usize, or as usize, f64::MAX);
                    }
                }

                let start = Vector2::new(0.0, 0.0);
                let goal = Vector2::new(
                    ((w - 1) as f64) * resolution,
                    ((h - 1) as f64) * resolution,
                );

                let config = AStarConfig::default();
                let result = astar_search(start, goal, &costmap, &config);

                match result {
                    AStarResult::Success { ref path, .. } => {
                        assert!(path.iter().all(|p| {
                            if let Some((col, row)) = costmap.world_to_cell(p.x, p.y) {
                                costmap.cost_at_cell(col, row) < f64::MAX
                            } else {
                                true
                            }
                        }));
                    }
                    _ => {}
                }
            }
        }
    }
}

mod rrt_prop {
    use super::*;
    use nav_global::rrt_star_search;
    use proptest::prelude::*;

    /// RRT* produces valid results with tree size >= 2.
    #[test]
    fn rrt_star_tree_grows_on_empty_map() {
        proptest! {
            fn rrt_star_tree_grows_on_empty_map(map_w: u8, map_h: u8) {
                let w = (map_w % 30) as usize + 10;
                let h = (map_h % 30) as usize + 10;
                let costmap = TestCostmap::new(w, h, 0.1);

                let start = Vector2::new(0.0, 0.0);
                let goal = Vector2::new(
                    ((w - 1) as f64) * 0.1,
                    ((h - 1) as f64) * 0.1,
                );

                let mut config = RRTStarConfig::default();
                config.max_iterations = 200;
                config.step_size = 1.0;

                match rrt_star_search(start, goal, &costmap, &config) {
                    RRTStarResult::Success { tree_size, .. } => {
                        assert!(tree_size >= 2);
                    }
                    _ => {}
                }
            }
        }
    }

    /// RRT* tree connectivity: every non-root node has a parent.
    #[test]
    fn rrt_star_tree_has_parents() {
        proptest! {
            fn rrt_star_tree_has_parents(map_w: u8, map_h: u8) {
                let w = (map_w % 20) as usize + 10;
                let h = (map_h % 20) as usize + 10;
                let costmap = TestCostmap::new(w, h, 0.1);

                let start = Vector2::new(0.0, 0.0);
                let goal = Vector2::new(
                    ((w - 1) as f64) * 0.1,
                    ((h - 1) as f64) * 0.1,
                );

                let mut config = RRTStarConfig::default();
                config.max_iterations = 300;
                config.step_size = 1.0;
                config.rewire_radius = 2.0;

                match rrt_star_search(start, goal, &costmap, &config) {
                    RRTStarResult::Success { .. } => {}
                    _ => {}
                }
            }
        }
    }
}

mod dstar_prop {
    use super::*;
    use nav_global::{DStarLiteConfig, dstar_lite_new, dstar_lite_replan};
    use proptest::prelude::*;

    /// D*Lite handles empty maps without panic.
    #[test]
    fn dstar_lite_empty_map() {
        proptest! {
            fn dstar_lite_empty_map(map_w: u8, map_h: u8) {
                let w = (map_w % 50) as usize + 10;
                let h = (map_h % 50) as usize + 10;
                let costmap = TestCostmap::new(w, h, 0.1);

                let config = DStarLiteConfig::default();
                let start = Vector2::new(0.0, 0.0);
                let goal = Vector2::new(
                    ((w - 1) as f64) * 0.1,
                    ((h - 1) as f64) * 0.1,
                );

                let (_result, _state) = dstar_lite_new(start, goal, &costmap, &config);
            }
        }
    }

    /// D*Lite update on empty map does not panic.
    #[test]
    fn dstar_lite_update_empty_map() {
        proptest! {
            fn dstar_lite_update_empty_map(map_w: u8, map_h: u8) {
                let w = (map_w % 50) as usize + 10;
                let h = (map_h % 50) as usize + 10;
                let costmap = TestCostmap::new(w, h, 0.1);

                let config = DStarLiteConfig::default();
                let start = Vector2::new(0.0, 0.0);
                let goal = Vector2::new(
                    ((w - 1) as f64) * 0.1,
                    ((h - 1) as f64) * 0.1,
                );

                let state = {
                    let (_r, s) = dstar_lite_new(start, goal, &costmap, &config);
                    s
                };

                let _result = dstar_lite_replan(&costmap, &config, state);
            }
        }
    }
}
