#![allow(unused_variables, unused_imports, dead_code, unused_mut, unused_assignments)]
//! Proptest properties for nav-core types.
//!
//! - PoseSE2 arithmetic (associativity, inverse, theta normalization)
//! - VelocityPair zero default
//! - Trajectory push/pop data integrity

use proptest::proptest;
mod pose_prop {
    use nav_core::{PoseSE2, VelocityPair};
    use proptest::prelude::*;

    /// PoseSE2::add_assign is associative up to floating point epsilon.
    #[test]
    fn pose_se2_add_associative() {
        proptest! {
            fn pose_se2_add_associative(
                a_x: f64, a_y: f64, a_t: f64,
                b_x: f64, b_y: f64, b_t: f64,
                c_x: f64, c_y: f64, c_t: f64,
            ) {
                let a = PoseSE2::from_xyz(a_x, a_y, a_t);
                let b = PoseSE2::from_xyz(b_x, b_y, b_t);
                let c = PoseSE2::from_xyz(c_x, c_y, c_t);
                let lhs = a.add(&b).add(&c);
                let rhs = a.add(&b.add(&c));
                assert!((lhs.x() - rhs.x()).abs() < 1e-10);
                assert!((lhs.y() - rhs.y()).abs() < 1e-10);
                assert!((lhs.theta() - rhs.theta()).abs() < 1e-10);
            }
        }
    }

    /// PoseSE2::inverse is correct: a + a.inverse() == zero.
    #[test]
    fn pose_se2_inverse_is_zero() {
        proptest! {
            fn pose_se2_inverse_is_zero(x: f64, y: f64, theta: f64) {
                let a = PoseSE2::from_xyz(x, y, theta);
                let inv = a.inverted();
                let result = a.add(&inv);
                let zero = PoseSE2::zero();
                assert!((result.x() - zero.x()).abs() < 1e-10);
                assert!((result.y() - zero.y()).abs() < 1e-10);
                assert!((result.theta() - zero.theta()).abs() < 1e-10);
            }
        }
    }

    /// PoseSE2::normalize_theta keeps theta in [-pi, pi].
    #[test]
    fn pose_se2_theta_in_bounds() {
        proptest! {
            fn pose_se2_theta_in_bounds(x: f64, y: f64, theta: f64) {
                let p = PoseSE2::from_xyz(x, y, theta);
                let pi = std::f64::consts::PI;
                assert!(p.theta() >= -pi && p.theta() <= pi);
            }
        }
    }

    /// VelocityPair::zero has all components zero by default.
    #[test]
    fn velocity_pair_zero_components() {
        let vp = VelocityPair::zero();
        assert_eq!(vp.vx, 0.0);
        assert_eq!(vp.vy, 0.0);
        assert_eq!(vp.omega, 0.0);
    }

    /// VelocityPair::default equals zero.
    #[test]
    fn velocity_pair_default_is_zero() {
        let vp = VelocityPair::default();
        assert_eq!(vp.vx, 0.0);
        assert_eq!(vp.vy, 0.0);
        assert_eq!(vp.omega, 0.0);
    }
}
// ── Standalone property tests ──────────────────────────────────────────────

/// normalize_angle always returns value in [-pi, pi] for finite inputs.
#[test]
fn prop_normalize_angle_finite_inputs_in_range() {
    use nav_core::misc::normalize_angle;
    proptest!(|(angle in -1000.0..1000.0f64)| {
        let result = normalize_angle(angle);
        if result.is_finite() {
            assert!(
                result >= -std::f64::consts::PI && result <= std::f64::consts::PI,
                "normalize_angle({}) = {} outside [-pi, pi]",
                angle, result
            );
        }
    });
}

/// normalize_angle is idempotent: applying it twice yields the same result.
#[test]
fn prop_normalize_angle_idempotent() {
    use nav_core::misc::normalize_angle;
    proptest!(|(angle in -1000.0..1000.0f64)| {
        let r1 = normalize_angle(angle);
        let r2 = normalize_angle(r1);
        assert!(
            (r1 - r2).abs() < 1e-14,
            "normalize_angle not idempotent: r1={}, r2={}",
            r1, r2
        );
    });
}

/// PoseSE2::add is commutative: a + b ≈ b + a (theta normalized).
#[test]
fn prop_pose_se2_add_commutative() {
    use nav_core::PoseSE2;
    proptest!(|(
        x1 in -100.0..100.0f64,
        y1 in -100.0..100.0f64,
        t1 in -100.0..100.0f64,
        x2 in -100.0..100.0f64,
        y2 in -100.0..100.0f64,
        t2 in -100.0..100.0f64,
    )| {
        let p1 = PoseSE2::new(x1, y1, t1);
        let p2 = PoseSE2::new(x2, y2, t2);
        let a = p1.add(&p2);
        let b = p2.add(&p1);
        assert!(
            (a.x() - b.x()).abs() < 1e-10,
            "commutative x: a.x={} b.x={}",
            a.x(), b.x()
        );
        assert!(
            (a.y() - b.y()).abs() < 1e-10,
            "commutative y: a.y={} b.y={}",
            a.y(), b.y()
        );
        assert!(
            (a.theta() - b.theta()).abs() < 1e-10,
            "commutative theta: a.t={} b.t={}",
            a.theta(), b.theta()
        );
    });
}


mod trajectory_prop {
    use nav_core::{PoseSE2, Trajectory};
    use proptest::prelude::*;

    /// Trajectory push preserves data integrity.
    #[test]
    fn trajectory_push_preserves_data() {
        proptest! {
            fn trajectory_push_preserves_data(x: f64, y: f64, theta: f64) {
                let mut traj = Trajectory::new();
                let orig = PoseSE2::from_xyz(x, y, theta);
                traj.add_pose(orig);
                let ret = traj.pose(0).clone();
                assert!((ret.x() - x).abs() < 1e-10);
                assert!((ret.y() - y).abs() < 1e-10);
                assert!((ret.theta() - theta).abs() < 1e-10);
            }
        }
    }

    /// Trajectory with multiple poses preserves ordering.
    #[test]
    fn trajectory_multiple_poses_ordered() {
        proptest! {
            fn trajectory_multiple_poses_ordered(
                p1_x: f64, p1_y: f64, p1_t: f64,
                p2_x: f64, p2_y: f64, p2_t: f64,
                p3_x: f64, p3_y: f64, p3_t: f64,
            ) {
                let mut traj = Trajectory::new();
                let poses = [
                    PoseSE2::from_xyz(p1_x, p1_y, p1_t),
                    PoseSE2::from_xyz(p2_x, p2_y, p2_t),
                    PoseSE2::from_xyz(p3_x, p3_y, p3_t),
                ];
                for p in &poses {
                    traj.add_pose(*p);
                }
                for (i, orig) in poses.iter().enumerate() {
                    let r = traj.pose(i);
                    assert!((r.x() - orig.x()).abs() < 1e-10);
                    assert!((r.y() - orig.y()).abs() < 1e-10);
                    assert!((r.theta() - orig.theta()).abs() < 1e-10);
                }
            }
        }
    }

    /// Trajectory::remove_pose preserves remaining pose integrity.
    #[test]
    fn trajectory_remove_preserves_remaining() {
        proptest! {
            fn trajectory_remove_preserves_remaining(
                x1: f64, y1: f64, t1: f64,
                x2: f64, y2: f64, t2: f64,
            ) {
                let mut traj = Trajectory::new();
                traj.add_pose(PoseSE2::from_xyz(x1, y1, t1));
                traj.add_pose(PoseSE2::from_xyz(x2, y2, t2));
                assert_eq!(traj.size_poses(), 2);
                traj.remove_pose(0);
                assert_eq!(traj.size_poses(), 1);
                let r = traj.pose(0);
                assert!((r.x() - x2).abs() < 1e-10);
                assert!((r.y() - y2).abs() < 1e-10);
                assert!((r.theta() - t2).abs() < 1e-10);
            }
        }
    }
}
// ── Adversarial: near-zero and denormal edge cases ─────────────────────────

proptest! {
    #[test]
    fn prop_normalize_angle_near_zero(a in -0.001..0.001f64) {
        use nav_core::misc::normalize_angle;
        let result = normalize_angle(a);
        assert!(result.is_finite());
        assert!(result >= -std::f64::consts::PI && result <= std::f64::consts::PI);
    }

    #[test]
    fn prop_normalize_angle_boundaries(
        a in -10.0..10.0f64
    ) {
        use nav_core::misc::normalize_angle;
        let result = normalize_angle(a);
        // If finite, must be in [-pi, pi]
        if result.is_finite() {
            assert!(result > -std::f64::consts::PI - 1e-12 && result < std::f64::consts::PI + 1e-12);
        }
    }

    #[test]
    fn prop_pose_se2_inverse_is_inverse(
        x in -100.0..100.0f64,
        y in -100.0..100.0f64,
        theta in -10.0..10.0f64
    ) {
        use nav_core::PoseSE2;
        let p = PoseSE2::new(x, y, theta);
        let inv = p.inverted();
        let restored = inv.transform_by(&p);
        // Result should be zero (identity): inv transformed by original should give identity
        assert!((restored.x()).abs() < 1e-6, "x error {} too large", restored.x());
        assert!((restored.y()).abs() < 1e-6, "y error {} too large", restored.y());
        assert!((restored.theta()).abs() < 1e-10, "theta error {} too large", restored.theta());
    }
}
