#![allow(unused_variables, unused_imports, dead_code, unused_mut, unused_assignments, unreachable_patterns)]
//! nav-local: local planners — TEB, DWA, MPC, MPPI.
//!
//! TEB implementation (port of teb_local_planner C++):
//! - band.rs: TimedElasticBand (trajectories, vertex management)
//! - optimizer.rs: TebOptimalPlanner (graph build, LM optimization)
//! - edges.rs: cost function edges (kinematics, velocity, obstacle, acceleration, etc.)
//! - obstacles.rs: obstacle association and edge creation
//! - topology.rs: homotopy class planning (H-signatures, graph search)
//!
//! Future local planners (stubbed):
//! - dwa: Dynamic Window Approach
//! - mpc: Model Predictive Control
//! - mppi: Model Predictive Path Integral

pub mod common;
pub mod teb;
pub mod dwa;
pub mod mpc;
pub mod mppi;

pub use teb::*;
pub use dwa::*;
pub use mpc::*;
pub use mppi::*;

// Re-export common types needed by DWA and external consumers
pub use common::{Costmap, PoseSE2, Planner, VelocityCommand};
