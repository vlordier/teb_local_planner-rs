//! nav-bevy: Bevy-based visualization and simulation for navigation planners.
//!
//! Modules:
//! - [`visualization`]: render TEB trajectories, obstacles, robot footprint
//! - [`simulation`]: simple 2D physics simulation for testing
//!
//! # Feature flags
//! - `bevy` — enables the Bevy visualization and simulation systems
//!
//! # Example
//! ```no_run
//! use bevy::prelude::*;
//! use nav_bevy::{visualization, simulation};
//!
//! fn main() {
//!     App::new()
//!         .add_plugins(DefaultPlugins)
//!         .add_systems(Update, (
//!             visualization::visualize_trajectory,
//!             visualization::visualize_obstacles,
//!             visualization::update_robot_footprint,
//!             simulation::simulate_robot,
//!             simulation::run_teb_planning,
//!             simulation::update_sensor_data,
//!         ))
//!         .run();
//! }
//! ```

#[cfg(feature = "bevy")]
pub mod visualization;

#[cfg(feature = "bevy")]
pub mod simulation;

#[cfg(feature = "bevy")]
pub use visualization::*;

#[cfg(feature = "bevy")]
pub use simulation::*;
