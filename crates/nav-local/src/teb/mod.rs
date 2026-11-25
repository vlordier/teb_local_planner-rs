//! TEB local planner module.
//!
//! Sub-modules mirror the C++ header organization:
//! - band.rs      → timed_elastic_band.h (TimedElasticBand, PoseVertex, TimeDiffVertex)
//! - optimizer.rs → optimal_planner.h (TebOptimalPlanner, graph optimization)
//! - edges.rs     → g2o_types/* (all edge types: kinematics, velocity, obstacle, etc.)
//! - obstacles.rs → obstacles.cpp + edge creation (obstacle association)
//! - topology.rs  → homotopy_class_planner.h + h_signature.h + graph_search.h

pub mod band;
pub mod pose_se2;
pub mod math;
pub mod optimizer;
pub mod edges;
pub mod obstacles;
pub mod topology;

pub use band::*;
pub use optimizer::*;
pub use edges::*;
pub use obstacles::*;
pub use topology::*;
