//! SE2 pose type for the Timed Elastic Band.
//!
//! Ported from C++ `pose_se2.h`.

use nalgebra::Vector2;

use crate::teb::math::{average_angles, normalize_angles};
use num_traits::identities::Zero;

/// A pose in SE2: position (x, y) + orientation (theta).
///
/// Ported from C++ `PoseSE2`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoseSe2 {
    position: Vector2<f64>,
    theta: f64,
}

impl Default for PoseSe2 {
    fn default() -> Self {
        Self::zero()
    }
}

impl PoseSe2 {
    /// Create a zero pose.
    #[inline]
    pub fn zero() -> Self {
        Self {
            position: Vector2::zeros(),
            theta: 0.0,
        }
    }

    /// Create from position and angle.
    #[inline]
    pub fn new(position: Vector2<f64>, theta: f64) -> Self {
        Self { position, theta }
    }

    /// Create from individual components.
    #[inline]
    pub fn from_components(x: f64, y: f64, theta: f64) -> Self {
        Self {
            position: Vector2::new(x, y),
            theta,
        }
    }

    /// Access position.
    #[inline]
    pub fn position(&self) -> &Vector2<f64> {
        &self.position
    }

    /// Access position (mutable).
    #[inline]
    pub fn position_mut(&mut self) -> &mut Vector2<f64> {
        &mut self.position
    }

    /// Access x coordinate.
    #[inline]
    pub fn x(&self) -> f64 {
        self.position.x
    }

    /// Access x coordinate (mutable).
    #[inline]
    pub fn x_mut(&mut self) -> &mut f64 {
        &mut self.position.x
    }

    /// Access y coordinate.
    #[inline]
    pub fn y(&self) -> f64 {
        self.position.y
    }

    /// Access y coordinate (mutable).
    #[inline]
    pub fn y_mut(&mut self) -> &mut f64 {
        &mut self.position.y
    }

    /// Access theta (orientation).
    #[inline]
    pub fn theta(&self) -> f64 {
        self.theta
    }

    /// Access theta (mutable).
    #[inline]
    pub fn theta_mut(&mut self) -> &mut f64 {
        &mut self.theta
    }

    /// Set all components to zero.
    #[inline]
    pub fn set_zero(&mut self) {
        self.position.set_zero();
        self.theta = 0.0;
    }

    /// Return the unit direction vector: [cos(theta), sin(theta)].
    #[inline]
    pub fn orientation_unit_vec(&self) -> Vector2<f64> {
        Vector2::new(self.theta.cos(), self.theta.sin())
    }

    /// Scale all components. Theta normalized afterwards.
    #[inline]
    pub fn scale(&mut self, factor: f64) {
        self.position *= factor;
        self.theta = normalize_angles(&[self.theta * factor]);
    }

    /// Add another pose (position-wise, theta normalized).
    #[inline]
    pub fn plus(&mut self, other: &PoseSe2) {
        self.position += other.position;
        self.theta = normalize_angles(&[self.theta, other.theta]);
    }

    /// Compute the mean of two poses.
    #[inline]
    pub fn average(p1: &PoseSe2, p2: &PoseSe2) -> Self {
        Self {
            position: (p1.position + p2.position) / 2.0,
            theta: normalize_angles(&[p1.theta, p2.theta]),
        }
    }

    /// Rotate globally by angle. If adjust_theta, also rotate theta.
    #[inline]
    pub fn rotate_global(&mut self, angle: f64, adjust_theta: bool) {
        let new_x = angle.cos() * self.position.x - angle.sin() * self.position.y;
        let new_y = angle.sin() * self.position.x + angle.cos() * self.position.y;
        self.position.x = new_x;
        self.position.y = new_y;
        if adjust_theta {
            self.theta = normalize_angles(&[self.theta + angle]);
        }
    }

    /// Subtract another pose.
    #[inline]
    pub fn subtract(&self, other: &PoseSe2) -> PoseSe2 {
        Self {
            position: self.position - other.position,
            theta: normalize_angles(&[self.theta, other.theta]),
        }
    }

    /// Add another pose (returns new pose).
    #[inline]
    pub fn add(&self, other: &PoseSe2) -> PoseSe2 {
        Self {
            position: self.position + other.position,
            theta: normalize_angles(&[self.theta, other.theta]),
        }
    }

    /// Multiply by scalar (theta NOT normalized - for velocity computation).
    #[inline]
    pub fn scalar_multiply(&self, scalar: f64) -> PoseSe2 {
        Self {
            position: self.position * scalar,
            theta: self.theta * scalar,
        }
    }

    /// Set x coordinate.
    #[inline]
    pub fn set_x(&mut self, x: f64) {
        self.position.x = x;
    }

    /// Set y coordinate.
    #[inline]
    pub fn set_y(&mut self, y: f64) {
        self.position.y = y;
    }
}

impl std::ops::Add for PoseSe2 {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self {
        self.plus(&rhs);
        self
    }
}

impl std::ops::AddAssign for PoseSe2 {
    fn add_assign(&mut self, rhs: Self) {
        self.plus(&rhs);
    }
}

impl std::ops::Sub for PoseSe2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self.subtract(&rhs)
    }
}

impl std::ops::Mul<f64> for PoseSe2 {
    type Output = Self;
    fn mul(self, scalar: f64) -> Self {
        self.scalar_multiply(scalar)
    }
}
