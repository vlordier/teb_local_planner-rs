#![allow(unused_variables, unused_imports, dead_code, unused_mut, unused_assignments, unreachable_patterns)]
//! nav-core: Core types for the TEB (Timed Elastic Band) local planner.
//!
//! Ported from the C++ `teb_local_planner` by Christoph Rösmann.

pub mod misc;
pub mod state;
pub mod trajectory;
pub mod vehicle_model;
pub mod constraints;
pub mod obstacles;

pub use state::{PoseSE2, VelocityPair};
pub use trajectory::Trajectory;
pub use vehicle_model::{VehicleModel, AckermannModel, DifferentialModel};
pub use constraints::TebConfig;
pub use obstacles::{
    Obstacle, PointObstacle, CircularObstacle, LineObstacle, PolygonObstacle, PillObstacle,
    ObstContainer,
};
pub use misc::normalize_angle;
