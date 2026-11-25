//! PoseSE2 and VelocityPair — ported from `pose_se2.h` and `velocity.h`.
//!
//! PoseSE2: a pose in SE2 = R² × S¹ (x, y, theta).
//! VelocityPair: translational (vx, vy) + rotational (omega) velocity.

use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use nalgebra::Vector2;

use crate::misc::normalize_angle;

/// A pose in SE2: x, y position + yaw angle theta ∈ [-π, π].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoseSE2 {
    position: Vector2<f64>,
    theta: f64,
}

impl PoseSE2 {
    // ── Constructors ──────────────────────────────────────────────

    /// Default: [0, 0, 0].
    pub fn zero() -> Self {
        Self::from_xyz(0.0, 0.0, 0.0)
    }

    /// Construct from x, y, theta.
    #[inline]
    pub fn new(x: f64, y: f64, theta: f64) -> Self {
        Self::from_xyz(x, y, theta)
    }

    /// Construct from position vector and theta.
    pub fn from_position(position: Vector2<f64>, theta: f64) -> Self {
        Self {
            position,
            theta: normalize_angle(theta),
        }
    }

    /// Construct from x, y, theta.
    #[inline]
    pub fn from_xyz(x: f64, y: f64, theta: f64) -> Self {
        Self {
            position: Vector2::new(x, y),
            theta: normalize_angle(theta),
        }
    }


    // ── Accessors ─────────────────────────────────────────────────

    /// 2D position reference.
    pub fn position(&self) -> &Vector2<f64> {
        &self.position
    }

    /// Mutable 2D position reference.
    pub fn position_mut(&mut self) -> &mut Vector2<f64> {
        &mut self.position
    }

    /// x coordinate.
    pub fn x(&self) -> f64 {
        self.position.x
    }

    /// Mutable x.
    pub fn x_mut(&mut self) -> &mut f64 {
        &mut self.position.x
    }

    /// y coordinate.
    pub fn y(&self) -> f64 {
        self.position.y
    }

    /// Mutable y.
    pub fn y_mut(&mut self) -> &mut f64 {
        &mut self.position.y
    }

    /// Yaw angle.
    pub fn theta(&self) -> f64 {
        self.theta
    }

    /// Mutable yaw.
    pub fn theta_mut(&mut self) -> &mut f64 {
        &mut self.theta
    }

    /// Set all to zero.
    pub fn set_zero(&mut self) {
        self.position = nalgebra::Vector2::zeros();
        self.theta = 0.0;
    }

    /// Copy pose from another.
    pub fn copy_from(&mut self, other: &Self) {
        self.position = other.position;
        self.theta = other.theta;
    }

    /// Unit direction vector [cos(theta), sin(theta)].
    pub fn orientation_unit_vec(&self) -> Vector2<f64> {
        Vector2::new(self.theta.cos(), self.theta.sin())
    }

    // ── Arithmetic operations ─────────────────────────────────────

    /// Scale all SE2 components. Theta normalized to [-π, π].
    pub fn scale(&mut self, factor: f64) {
        self.position *= factor;
        self.theta = normalize_angle(self.theta * factor);
    }

    /// In-place addition. Theta normalized.
    pub fn add_assign(&mut self, rhs: &Self) {
        self.position += &rhs.position;
        self.theta = normalize_angle(self.theta + rhs.theta);
    }

    /// In-place subtraction. Theta normalized.
    pub fn sub_assign(&mut self, rhs: &Self) {
        self.position -= &rhs.position;
        self.theta = normalize_angle(self.theta - rhs.theta);
    }

    /// Add two poses, returning new PoseSE2. Theta normalized.
    pub fn add(&self, rhs: &Self) -> Self {
        let mut result = *self;
        result.add_assign(rhs);
        result
    }

    /// Subtract two poses, returning new PoseSE2. Theta normalized.
    pub fn sub(&self, rhs: &Self) -> Self {
        let mut result = *self;
        result.sub_assign(rhs);
        result
    }

    /// Multiply pose with scalar — does NOT normalize theta.
    pub fn scalar_mul(self, scalar: f64) -> Self {
        Self {
            position: self.position * scalar,
            theta: self.theta * scalar,
        }
    }

    /// Multiply from the left.
    pub fn scalar_mul_left(scalar: f64, pose: Self) -> Self {
        pose.scalar_mul(scalar)
    }

    /// Mean of two poses.
    pub fn average(pose1: &Self, pose2: &Self) -> Self {
        let pos = (&pose1.position + &pose2.position) * 0.5;
        let theta = average_angle(pose1.theta, pose2.theta);
        Self { position: pos, theta }
    }

    /// Rotate pose globally: [x, y] = Rot(angle) * [x, y].
    pub fn rotate_global(&mut self, angle: f64, adjust_theta: bool) {
        let new_x = angle.cos() * self.position.x - angle.sin() * self.position.y;
        let new_y = angle.sin() * self.position.x + angle.cos() * self.position.y;
        self.position.x = new_x;
        self.position.y = new_y;
        if adjust_theta {
            self.theta = normalize_angle(self.theta + angle);
        }
    }

    /// Transform by rigid transform T: T * PoseSE2.
    pub fn transform_by(&self, t: &Self) -> Self {
        let rot = t.theta();
        let rotated_x = rot.cos() * self.x() - rot.sin() * self.y();
        let rotated_y = rot.sin() * self.x() + rot.cos() * self.y();
        Self {
            position: Vector2::new(t.x() + rotated_x, t.y() + rotated_y),
            theta: normalize_angle(t.theta() + self.theta),
        }
    }

    /// Inverse in SE2.
    pub fn inverted(&self) -> Self {
        let rot = -self.theta;
        let rx = rot.cos() * (-self.x()) - rot.sin() * (-self.y());
        let ry = rot.sin() * (-self.x()) + rot.cos() * (-self.y());
        Self {
            position: Vector2::new(rx, ry),
            theta: normalize_angle(-self.theta),
        }
    }

    /// Euclidean distance in R².
    pub fn position_distance(&self, other: &Self) -> f64 {
        (self.position - other.position).norm()
    }

    /// Angular distance.
    pub fn angular_distance(&self, other: &Self) -> f64 {
        normalize_angle(self.theta - other.theta).abs()
    }
}

impl Default for PoseSE2 {
    fn default() -> Self {
        Self::zero()
    }
}

impl AbsDiffEq for PoseSE2 {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        f64::abs_diff_eq(&self.position.x, &other.position.x, epsilon)
            && f64::abs_diff_eq(&self.position.y, &other.position.y, epsilon)
            && f64::abs_diff_eq(&self.theta, &other.theta, epsilon)
    }
}

impl RelativeEq for PoseSE2 {
    fn default_max_relative() -> Self::Epsilon {
        f64::default_max_relative()
    }

    fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool {
        f64::relative_eq(&self.position.x, &other.position.x, epsilon, max_relative)
            && f64::relative_eq(&self.position.y, &other.position.y, epsilon, max_relative)
            && f64::relative_eq(&self.theta, &other.theta, epsilon, max_relative)
    }
}

impl UlpsEq for PoseSE2 {
    fn default_max_ulps() -> u32 {
        f64::default_max_ulps()
    }

    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        f64::ulps_eq(&self.position.x, &other.position.x, epsilon, max_ulps)
            && f64::ulps_eq(&self.position.y, &other.position.y, epsilon, max_ulps)
            && f64::ulps_eq(&self.theta, &other.theta, epsilon, max_ulps)
    }
}

/// Average angle of two angles, normalized to [-π, π].
fn average_angle(a: f64, b: f64) -> f64 {
    let (da, db) = if a <= b { (a, b) } else { (b, a) };
    let diff = db - da;
    if diff > std::f64::consts::PI {
        a + b + 2.0 * std::f64::consts::PI - da - db
    } else if diff < -std::f64::consts::PI {
        a + b - 2.0 * std::f64::consts::PI + db - da
    } else {
        (a + b) / 2.0
    }
}

/// Translational velocities (vx, vy) + angular velocity (omega).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VelocityPair {
    pub vx: f64,
    pub vy: f64,
    pub omega: f64,
}

impl VelocityPair {
    pub fn zero() -> Self {
        Self {
            vx: 0.0,
            vy: 0.0,
            omega: 0.0,
        }
    }

    pub fn speed(&self) -> f64 {
        (self.vx * self.vx + self.vy * self.vy).sqrt()
    }
}

impl Default for VelocityPair {
    fn default() -> Self {
        Self::zero()
    }
}

impl AbsDiffEq for VelocityPair {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        f64::abs_diff_eq(&self.vx, &other.vx, epsilon)
            && f64::abs_diff_eq(&self.vy, &other.vy, epsilon)
            && f64::abs_diff_eq(&self.omega, &other.omega, epsilon)
    }
}

impl RelativeEq for VelocityPair {
    fn default_max_relative() -> Self::Epsilon {
        f64::default_max_relative()
    }

    fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool {
        f64::relative_eq(&self.vx, &other.vx, epsilon, max_relative)
            && f64::relative_eq(&self.vy, &other.vy, epsilon, max_relative)
            && f64::relative_eq(&self.omega, &other.omega, epsilon, max_relative)
    }
}

impl UlpsEq for VelocityPair {
    fn default_max_ulps() -> u32 {
        f64::default_max_ulps()
    }

    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        f64::ulps_eq(&self.vx, &other.vx, epsilon, max_ulps)
            && f64::ulps_eq(&self.vy, &other.vy, epsilon, max_ulps)
            && f64::ulps_eq(&self.omega, &other.omega, epsilon, max_ulps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pose_se2_default() {
        let p = PoseSE2::default();
        assert_eq!(p.x(), 0.0);
        assert_eq!(p.y(), 0.0);
        assert_eq!(p.theta(), 0.0);
    }

    #[test]
    fn test_pose_se2_constructors() {
        let p = PoseSE2::from_xyz(1.0, 2.0, std::f64::consts::FRAC_PI_4);
        assert!((p.x() - 1.0).abs() < 1e-12);
        assert!((p.y() - 2.0).abs() < 1e-12);
        assert!((p.theta() - std::f64::consts::FRAC_PI_4).abs() < 1e-12);

        let pos = Vector2::new(3.0, 4.0);
        let p2 = PoseSE2::from_position(pos, std::f64::consts::PI);
        assert!((p2.x() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_normalize_theta_in_constructor() {
        let p = PoseSE2::from_xyz(0.0, 0.0, 3.0 * std::f64::consts::PI);
        assert_eq!(p.theta(), -std::f64::consts::PI);
    }

    #[test]
    fn test_add_assign() {
        let mut p = PoseSE2::from_xyz(1.0, 2.0, 0.5);
        let q = PoseSE2::from_xyz(0.5, -1.0, 0.3);
        p.add_assign(&q);
        assert!((p.x() - 1.5).abs() < 1e-12);
        assert!((p.y() - 1.0).abs() < 1e-12);
        assert!((p.theta() - 0.8).abs() < 1e-12);
    }

    #[test]
    fn test_sub_assign() {
        let mut p = PoseSE2::from_xyz(2.0, 3.0, 1.0);
        let q = PoseSE2::from_xyz(0.5, 1.0, 0.3);
        p.sub_assign(&q);
        assert!((p.x() - 1.5).abs() < 1e-12);
        assert!((p.y() - 2.0).abs() < 1e-12);
        assert!((p.theta() - 0.7).abs() < 1e-12);
    }

    #[test]
    fn test_average() {
        let p1 = PoseSE2::from_xyz(0.0, 0.0, 0.0);
        let p2 = PoseSE2::from_xyz(2.0, 0.0, std::f64::consts::FRAC_PI_2);
        let avg = PoseSE2::average(&p1, &p2);
        assert!((avg.x() - 1.0).abs() < 1e-12);
        assert!((avg.y() - 0.0).abs() < 1e-12);
        assert!((avg.theta() - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
    }

    #[test]
    fn test_rotate_global() {
        let mut p = PoseSE2::from_xyz(1.0, 0.0, 0.0);
        p.rotate_global(std::f64::consts::FRAC_PI_2, true);
        assert!((p.x()).abs() < 1e-12);
        assert!((p.y() - 1.0).abs() < 1e-12);
        assert!((p.theta() - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    }

    #[test]
    fn test_scale() {
        let mut p = PoseSE2::from_xyz(2.0, 4.0, 1.0);
        p.scale(0.5);
        assert!((p.x() - 1.0).abs() < 1e-12);
        assert!((p.y() - 2.0).abs() < 1e-12);
        assert!((p.theta() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_orientation_unit_vec() {
        let p = PoseSE2::from_xyz(0.0, 0.0, std::f64::consts::FRAC_PI_4);
        let v = p.orientation_unit_vec();
        assert!((v.x - std::f64::consts::FRAC_PI_4.cos()).abs() < 1e-12);
        assert!((v.y - std::f64::consts::FRAC_PI_4.sin()).abs() < 1e-12);
    }

    #[test]
    fn test_transform_by() {
        let p = PoseSE2::from_xyz(1.0, 2.0, 0.5);
        let identity = PoseSE2::default();
        let result = p.transform_by(&identity);
        assert!((result.x() - p.x()).abs() < 1e-12);
        assert!((result.y() - p.y()).abs() < 1e-12);
        assert!((result.theta() - p.theta()).abs() < 1e-12);
    }

    #[test]
    fn test_inverted() {
        let p = PoseSE2::from_xyz(1.0, 2.0, 0.5);
        let inv = p.inverted();
        let result = p.transform_by(&inv);
        assert!((result.x()).abs() < 1e-12);
        assert!((result.y()).abs() < 1e-12);
        assert!((result.theta()).abs() < 1e-12);
    }

    #[test]
    fn test_velocity_pair() {
        let v = VelocityPair {
            vx: 1.0,
            vy: 0.5,
            omega: 0.3,
        };
        assert!((v.speed() - 1.118033988749895).abs() < 1e-10);
    }

    #[test]
    fn test_scalar_mul() {
        let p = PoseSE2::from_xyz(1.0, 2.0, 3.0);
        let scaled = p.scalar_mul(2.0);
        assert!((scaled.x() - 2.0).abs() < 1e-12);
        assert!((scaled.y() - 4.0).abs() < 1e-12);
        assert!((scaled.theta() - 6.0).abs() < 1e-12);
    }
}
