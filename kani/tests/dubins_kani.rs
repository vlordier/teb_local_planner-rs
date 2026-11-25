//! Kani verification properties for Dubins path generation.
//!
//! Uses concrete f64 values to avoid CBMC floatbv limitations.
//! Parallel-angle poses trigger the straight-line special case, avoiding
//! transcendental function calls (hypot, sin, cos, atan2) that CBMC cannot model.

#![no_main]

mod kubins_kani {
    use nav_core::PoseSE2;
    use nav_trajectory::dubins_path_from_poses;

    /// Dubins path segment lengths are within bounds (non-negative).
    /// Concrete values: parallel poses with small radius → straight-line path.
    #[kani::proof]
    #[kani::unwind(4)]
    fn kubins_path_bounds() {
        let p1 = PoseSE2::new(0.0, 0.0, 0.0);
        let p2 = PoseSE2::new(3.0, 0.0, 0.0);
        let radius = 1.0;

        if let Some(path) = dubins_path_from_poses(&p1, &p2, radius) {
            assert!(path.length.is_finite(), "path length should be finite, got {}", path.length);
            assert!(path.length >= 0.0, "length must be non-negative, got {}", path.length);

            for seg in &path.segments {
                assert!(seg.length.is_finite(), "segment length should be finite, got {}", seg.length);
                assert!(seg.length >= 0.0, "segment must be non-negative, got {}", seg.length);
            }
        }
    }

    /// Dubins path length scales monotonically with distance.
    /// Concrete: same start, two different end distances (parallel angles).
    #[kani::proof]
    #[kani::unwind(4)]
    fn kubins_length_non_decreasing() {
        let p1 = PoseSE2::new(0.0, 0.0, 0.0);
        let p_close = PoseSE2::new(1.0, 0.0, 0.0);
        let p_far = PoseSE2::new(10.0, 0.0, 0.0);
        let radius = 1.0;

        let path_close = dubins_path_from_poses(&p1, &p_close, radius);
        let path_far = dubins_path_from_poses(&p1, &p_far, radius);

        match (path_close, path_far) {
            (Some(close), Some(far)) => {
                assert!(far.length.is_finite(), "far length should be finite, got {}", far.length);
                assert!(close.length.is_finite(), "close length should be finite, got {}", close.length);
                assert!(
                    far.length >= close.length,
                    "Far path ({}) should be >= close path ({})",
                    far.length, close.length
                );
            }
            _ => {}
        }
    }
}
