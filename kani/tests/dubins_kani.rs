//! Kani verification properties for Dubins path generation.
//!
//! CBMC cannot verify floatbv properties for symbolic inputs and the
//! kani runtime's fmax declaration triggers assertions. We use conditional
//! expressions instead of .max() to avoid C fmax calls.

#![no_main]

mod dubins_kani {
    /// Dubins path segment lengths are within bounds (non-negative).
    #[kani::proof]
    #[kani::unwind(4)]
    fn dubins_path_bounds() {
        use nav_core::PoseSE2;
        use nav_trajectory::dubins_path_from_poses;

        let x1: f64 = kani::any();
        let y1: f64 = kani::any();
        let t1: f64 = kani::any();
        let x2: f64 = kani::any();
        let y2: f64 = kani::any();
        let t2: f64 = kani::any();
        let radius: f64 = kani::any();

        kani::assume(x1 >= -100.0 && x1 <= 100.0);
        kani::assume(y1 >= -100.0 && y1 <= 100.0);
        kani::assume(t1 >= -100.0 && t1 <= 100.0);
        kani::assume(x2 >= -100.0 && x2 <= 100.0);
        kani::assume(y2 >= -100.0 && y2 <= 100.0);
        kani::assume(t2 >= -100.0 && t2 <= 100.0);
        kani::assume(radius >= -100.0 && radius <= 100.0);

        // Avoid f64::max() which maps to C fmax (kani runtime assertion)
        let radius = if radius.abs() < 0.1 { 0.1 } else { radius.abs() };
        let p1 = PoseSE2::from_xyz(x1, y1, t1);
        let p2 = PoseSE2::from_xyz(x2, y2, t2);

        if let Some(path) = dubins_path_from_poses(&p1, &p2, radius) {
            // Verify: path length is finite and non-negative
            assert!(path.length.is_finite(), "path length should be finite");
            assert!(path.length >= 0.0, "Dubins path length must be non-negative, got {}", path.length);

            for seg in &path.segments {
                assert!(seg.length.is_finite(), "segment length should be finite");
                assert!(seg.length >= 0.0, "Dubins segment length must be non-negative, got {}", seg.length);
            }
        }
    }

    /// Dubins path length scales monotonically with distance.
    #[kani::proof]
    #[kani::unwind(4)]
    fn dubins_length_non_decreasing() {
        use nav_core::PoseSE2;
        use nav_trajectory::dubins_path_from_poses;

        let x1: f64 = kani::any();
        let y1: f64 = kani::any();
        let t1: f64 = kani::any();
        let radius: f64 = kani::any();

        kani::assume(x1 >= -100.0 && x1 <= 100.0);
        kani::assume(y1 >= -100.0 && y1 <= 100.0);
        kani::assume(t1 >= -100.0 && t1 <= 100.0);
        kani::assume(radius >= -100.0 && radius <= 100.0);

        // Avoid f64::max() which maps to C fmax
        let radius = if radius.abs() < 0.1 { 0.1 } else { radius.abs() };

        let p1 = PoseSE2::from_xyz(x1, y1, t1);
        let p_close = PoseSE2::from_xyz(x1 + 1.0, y1, t1);
        let p_far = PoseSE2::from_xyz(x1 + 10.0, y1, t1);

        let path_close = dubins_path_from_poses(&p1, &p_close, radius);
        let path_far = dubins_path_from_poses(&p1, &p_far, radius);

        match (path_close, path_far) {
            (Some(close), Some(far)) => {
                // Only assert if lengths are finite (CBMC can't prove f64 ops)
                assert!(far.length.is_finite(), "far length should be finite");
                assert!(close.length.is_finite(), "close length should be finite");
                assert!(
                    far.length >= close.length,
                    "Far path length {} should be >= close path length {}",
                    far.length,
                    close.length
                );
            }
            _ => {}
        }
    }
}
