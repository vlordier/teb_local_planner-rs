#![allow(unused_variables, unused_imports, dead_code, unused_mut, unused_assignments, unreachable_patterns)]
//! VehicleModel trait and implementations.
//!
//! Ported from `vehicle_model.h` concepts in teb_local_planner.
//! The vehicle model encodes kinematic constraints: minimum turning radius,
//! max velocities, and max accelerations.

/// Trait for robot kinematic models.
///
/// Models the physical constraints of the robot's motion.
pub trait VehicleModel {
    /// Type name identifier.
    fn name(&self) -> &str;

    /// Minimum turning radius [m].
    /// Zero for differential-drive robots; positive for carlike/ackermann.
    fn min_turning_radius(&self) -> f64;

    /// Maximum translational velocity [m/s].
    fn max_velocity(&self) -> f64;

    /// Maximum translational acceleration [m/s²].
    fn max_acceleration(&self) -> f64;

    /// Maximum angular velocity [rad/s].
    fn max_angular_velocity(&self) -> f64;

    /// Maximum angular acceleration [rad/s²].
    fn max_angular_acceleration(&self) -> f64;

    /// Maximum strafing (lateral) velocity [m/s].
    /// Nonzero for holonomic (omnidirectional) robots.
    fn max_strafing_velocity(&self) -> f64;

    /// Maximum strafing acceleration [m/s²].
    fn max_strafing_acceleration(&self) -> f64;

    /// Maximum backwards translational velocity [m/s].
    fn max_velocity_backwards(&self) -> f64;

    /// Minimum turning radius check: returns true if the turning radius
    /// required for a given velocity and angular rate is feasible.
    fn is_turning_radius_feasible(&self, vx: f64, omega: f64) -> bool {
        if self.min_turning_radius() <= 0.0 {
            return true; // no constraint for diff-drive
        }
        if omega.abs() < 1e-10 {
            return true; // straight line
        }
        let r = (vx / omega).abs();
        r >= self.min_turning_radius()
    }

    /// Check if the given velocity command is kinematically feasible.
    fn is_velocity_feasible(&self, vx: f64, vy: f64, omega: f64) -> bool {
        if vx.abs() > self.max_velocity()
            && vx.abs() > self.max_velocity_backwards()
        {
            return false;
        }
        if vy.abs() > self.max_strafing_velocity() {
            return false;
        }
        if omega.abs() > self.max_angular_velocity() {
            return false;
        }
        self.is_turning_radius_feasible(vx, omega)
    }
}

/// Differential drive model.
///
/// Two independently driven wheels. Minimum turning radius = 0 (can spin in place).
#[derive(Debug, Clone)]
pub struct DifferentialModel {
    /// Maximum translational velocity [m/s].
    max_vel_x: f64,
    /// Maximum backwards translational velocity [m/s].
    max_vel_x_backwards: f64,
    /// Maximum angular velocity [rad/s].
    max_vel_theta: f64,
    /// Maximum translational acceleration [m/s²].
    acc_lim_x: f64,
    /// Maximum angular acceleration [rad/s²].
    acc_lim_theta: f64,
}

impl DifferentialModel {
    /// Create a new differential drive model with default realistic values.
    pub fn default() -> Self {
        Self {
            max_vel_x: 0.4,
            max_vel_x_backwards: 0.2,
            max_vel_theta: 0.3,
            acc_lim_x: 0.5,
            acc_lim_theta: 0.5,
        }
    }

    /// Create with custom parameters.
    pub fn new(
        max_vel_x: f64,
        max_vel_x_backwards: f64,
        max_vel_theta: f64,
        acc_lim_x: f64,
        acc_lim_theta: f64,
    ) -> Self {
        Self {
            max_vel_x,
            max_vel_x_backwards,
            max_vel_theta,
            acc_lim_x,
            acc_lim_theta,
        }
    }
}

impl Default for DifferentialModel {
    fn default() -> Self {
        Self::default()
    }
}

impl VehicleModel for DifferentialModel {
    fn name(&self) -> &str {
        "DifferentialDrive"
    }

    fn min_turning_radius(&self) -> f64 {
        0.0
    }

    fn max_velocity(&self) -> f64 {
        self.max_vel_x
    }

    fn max_acceleration(&self) -> f64 {
        self.acc_lim_x
    }

    fn max_angular_velocity(&self) -> f64 {
        self.max_vel_theta
    }

    fn max_angular_acceleration(&self) -> f64 {
        self.acc_lim_theta
    }

    fn max_strafing_velocity(&self) -> f64 {
        0.0
    }

    fn max_strafing_acceleration(&self) -> f64 {
        0.0
    }

    fn max_velocity_backwards(&self) -> f64 {
        self.max_vel_x_backwards
    }
}

/// Ackermann (carlike) drive model.
///
/// Single steering angle controls the turning radius.
/// Requires a non-zero minimum turning radius.
#[derive(Debug, Clone)]
pub struct AckermannModel {
    /// Minimum turning radius [m].
    min_turning_radius: f64,
    /// Maximum translational velocity [m/s].
    max_vel_x: f64,
    /// Maximum backwards translational velocity [m/s].
    max_vel_x_backwards: f64,
    /// Maximum angular velocity [rad/s].
    max_vel_theta: f64,
    /// Maximum translational acceleration [m/s²].
    acc_lim_x: f64,
    /// Maximum angular acceleration [rad/s²].
    acc_lim_theta: f64,
    /// Wheelbase distance (distance between drive shaft and steering axle) [m].
    /// Used for converting steering angle to angular velocity.
    wheelbase: f64,
    /// If true, the commanded message uses steering angle instead of rotational velocity.
    cmd_angle_instead_rotvel: bool,
}

impl AckermannModel {
    /// Create with default values.
    pub fn default() -> Self {
        Self {
            min_turning_radius: 0.5,
            max_vel_x: 0.4,
            max_vel_x_backwards: 0.1,
            max_vel_theta: 0.5,
            acc_lim_x: 0.5,
            acc_lim_theta: 0.5,
            wheelbase: 1.0,
            cmd_angle_instead_rotvel: false,
        }
    }

    /// Compute the steering angle from a turning radius.
    /// angle = atan(wheelbase / radius).
    #[inline]
    #[must_use]
    pub fn steering_angle_from_radius(&self, radius: f64) -> f64 {
        self.wheelbase.atan2(radius)
    }

    /// Convert a commanded steering angle [rad] to angular velocity [rad/s].
    /// omega = vx / (wheelbase / tan(angle)) = vx * tan(angle) / wheelbase.
    #[inline]
    #[must_use]
    pub fn rotvel_from_steering_angle(&self, vx: f64, steering_angle: f64) -> f64 {
        if self.wheelbase.abs() < 1e-10 {
            return 0.0;
        }
        vx * steering_angle.tan() / self.wheelbase
    }

    /// Convert an angular velocity [rad/s] to a steering angle [rad].
    #[inline]
    #[must_use]
    pub fn steering_angle_from_rotvel(&self, vx: f64, omega: f64) -> f64 {
        if vx.abs() < 1e-10 {
            return 0.0;
        }
        let radius = vx / omega;
        self.wheelbase.atan2(radius)
    }
}

impl Default for AckermannModel {
    fn default() -> Self {
        Self::default()
    }
}

impl VehicleModel for AckermannModel {
    fn name(&self) -> &str {
        "AckermannDrive"
    }

    fn min_turning_radius(&self) -> f64 {
        self.min_turning_radius
    }

    fn max_velocity(&self) -> f64 {
        self.max_vel_x
    }

    fn max_acceleration(&self) -> f64 {
        self.acc_lim_x
    }

    fn max_angular_velocity(&self) -> f64 {
        self.max_vel_theta
    }

    fn max_angular_acceleration(&self) -> f64 {
        self.acc_lim_theta
    }

    fn max_strafing_velocity(&self) -> f64 {
        0.0
    }

    fn max_strafing_acceleration(&self) -> f64 {
        0.0
    }

    fn max_velocity_backwards(&self) -> f64 {
        self.max_vel_x_backwards
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_differential_model_defaults() {
        let m = DifferentialModel::default();
        assert_eq!(m.name(), "DifferentialDrive");
        assert_eq!(m.min_turning_radius(), 0.0);
        assert_eq!(m.max_velocity(), 0.4);
        assert_eq!(m.max_angular_velocity(), 0.3);
        assert_eq!(m.max_velocity_backwards(), 0.2);
        assert_eq!(m.max_strafing_velocity(), 0.0);
    }

    #[test]
    fn test_ackermann_model_defaults() {
        let m = AckermannModel::default();
        assert_eq!(m.name(), "AckermannDrive");
        assert_eq!(m.min_turning_radius(), 0.5);
        assert_eq!(m.max_velocity(), 0.4);
        assert_eq!(m.wheelbase, 1.0);
    }

    #[test]
    fn test_ackermann_steering_angle() {
        let m = AckermannModel::default();
        // radius = wheelbase → angle = 45°
        let angle = m.steering_angle_from_radius(1.0);
        assert!((angle - std::f64::consts::FRAC_PI_4).abs() < 1e-10);

        // Large radius → small angle
        let angle = m.steering_angle_from_radius(10.0);
        assert!(angle.abs() < 0.1);
    }

    #[test]
    fn test_ackermann_rotvel_conversion() {
        let m = AckermannModel::default();
        let vx = 1.0;
        let steering = std::f64::consts::FRAC_PI_4; // 45°
        let omega = m.rotvel_from_steering_angle(vx, steering);
        // omega = 1.0 * tan(π/4) / 1.0 = 1.0
        assert!((omega - 1.0).abs() < 1e-10);

        // Back-conversion
        let back = m.steering_angle_from_rotvel(vx, omega);
        assert!((back - steering).abs() < 1e-10);
    }

    #[test]
    fn test_turning_radius_feasible_diffdrive() {
        let m = DifferentialModel::default();
        // diff-drive: always feasible (min_turning_radius = 0)
        assert!(m.is_turning_radius_feasible(1.0, 10.0));
        assert!(m.is_turning_radius_feasible(0.0, 0.0));
    }

    #[test]
    fn test_turning_radius_feasible_ackermann() {
        let m = AckermannModel::default();
        // vx = 1.0, omega = 1.0 → radius = 1.0, min = 0.5 → feasible
        assert!(m.is_turning_radius_feasible(1.0, 1.0));
        // vx = 1.0, omega = 3.0 → radius = 0.33, min = 0.5 → NOT feasible
        assert!(!m.is_turning_radius_feasible(1.0, 3.0));
        // zero omega (straight line) → always feasible
        assert!(m.is_turning_radius_feasible(1.0, 0.0));
    }

    #[test]
    fn test_velocity_feasible() {
        let m = DifferentialModel::default();
        // within bounds
        assert!(m.is_velocity_feasible(0.3, 0.0, 0.2));
        // vx exceeds max
        assert!(!m.is_velocity_feasible(1.0, 0.0, 0.0));
        // omega exceeds max
        assert!(!m.is_velocity_feasible(0.0, 0.0, 1.0));
        // vy nonzero on diff-drive
        assert!(!m.is_velocity_feasible(0.0, 0.1, 0.0));
    }
}
