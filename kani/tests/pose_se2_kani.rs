//! Kani verification properties for PoseSE2.
//!
//! Properties verified by Kani's symbolic execution:
//! - theta is always in [-pi, pi] after any valid construction
//! - composition of two poses yields a valid pose
//! - inverse composition with original yields identity

#![no_main]

mod pose_se2_kani {
    /// PoseSE2::theta is always normalized to [-pi, pi] after from_xyz construction.
    #[kani::proof]
    #[kani::unwind(4)]
    fn pose_se2_theta_normalized() {
        use nav_core::PoseSE2;
        use std::f64::consts::PI;

        // Kani symbolic values — represent arbitrary inputs
        let x: f64 = kani::any();
        let y: f64 = kani::any();
        let theta: f64 = kani::any();

        let pose = PoseSE2::from_xyz(x, y, theta);
        let pose_theta = pose.theta();

        // Theta must always be in [-pi, pi]
        assert!(
            pose_theta >= -PI && pose_theta <= PI,
            "theta should be in [-pi, pi], got {}",
            pose_theta
        );
    }

    /// Composition of two poses yields a valid pose with normalized theta.
    #[kani::proof]
    #[kani::unwind(4)]
    fn pose_se2_composition_consistent() {
        use nav_core::PoseSE2;
        use std::f64::consts::PI;

        let x1: f64 = kani::any();
        let y1: f64 = kani::any();
        let t1: f64 = kani::any();
        let x2: f64 = kani::any();
        let y2: f64 = kani::any();
        let t2: f64 = kani::any();

        let p1 = PoseSE2::from_xyz(x1, y1, t1);
        let p2 = PoseSE2::from_xyz(x2, y2, t2);
        let result = p1.add(&p2);

        // Resulting theta must be normalized
        assert!(
            result.theta() >= -PI && result.theta() <= PI,
            "composed theta should be in [-pi, pi], got {}",
            result.theta()
        );

        // X, Y should be the sum of individual X, Y components
        let expected_x = x1 + x2;
        let expected_y = y1 + y2;
        assert!(
            (result.x() - expected_x).abs() < 1e-10,
            "x should be {}, got {}",
            expected_x,
            result.x()
        );
        assert!(
            (result.y() - expected_y).abs() < 1e-10,
            "y should be {}, got {}",
            expected_y,
            result.y()
        );
    }

    /// Inverse composition yields identity (zero pose).
    #[kani::proof]
    #[kani::unwind(4)]
    fn pose_se2_inverse_identity() {
        use nav_core::PoseSE2;

        let x: f64 = kani::any();
        let y: f64 = kani::any();
        let t: f64 = kani::any();

        let p = PoseSE2::from_xyz(x, y, t);
        let inv = p.inverted();
        let result = p.add(&inv);

        // Result should be identity (zero pose)
        assert!(
            (result.x() - 0.0).abs() < 1e-10,
            "x should be 0, got {}",
            result.x()
        );
        assert!(
            (result.y() - 0.0).abs() < 1e-10,
            "y should be 0, got {}",
            result.y()
        );
        assert!(
            (result.theta() - 0.0).abs() < 1e-10,
            "theta should be 0, got {}",
            result.theta()
        );
    }
}
