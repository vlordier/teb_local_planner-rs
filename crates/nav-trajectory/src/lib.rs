//! nav-trajectory: trajectory primitives and time parameterization.
//!
//! Algorithms:
//! - Dubins paths (fixed-wing: forward arcs only)
//! - Reeds-Shepp curves (differential-drive: forward + reverse)
//! - Spline interpolation
//! - Time parameterization (optimal time, shortcutting)

pub mod dubins;
pub mod reeds_shepp;
pub mod splines;
pub mod time_parameterization;

pub use dubins::*;
pub use reeds_shepp::*;
pub use splines::*;
pub use time_parameterization::*;
