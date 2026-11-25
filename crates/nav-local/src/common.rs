#![allow(unused_variables, unused_imports, dead_code, unused_mut, unused_assignments, unreachable_patterns)]
//! Shared types for all local planners.
//!
//! Re-exports from nav-core and defines additional common types.

pub use nav_core::PoseSE2;
// Placeholder types for planning interface
pub struct Costmap(pub Vec<f64>);
pub struct VelocityCommand { pub vx: f64, pub vy: f64, pub omega: f64 }
impl VelocityCommand {
    pub fn new(vx: f64, vy: f64, omega: f64) -> Self {
        Self { vx, vy, omega }
    }
}

impl Costmap {
    pub fn get_cost(&self, _x: f64, _y: f64) -> Option<f64> {
        Some(0.0)
    }
}

/// Trait for all local planners.
pub trait Planner {
    /// Plan a velocity command from current pose to goal, given costmap.
    fn plan(
        &mut self,
        pose: PoseSE2,
        goal: PoseSE2,
        costmap: &Costmap,
    ) -> Option<VelocityCommand>;
}
