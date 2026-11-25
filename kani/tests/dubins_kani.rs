//! Kani verification properties for Dubins path generation.
//!
//! Properties verified by Kani's symbolic execution:
//! - Segment lengths are bounded within expected ranges
//! - Total path length is non-negative

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

        // Must use a positive radius
        let radius = radius.abs().max(0.1);
        let p1 = PoseSE2::from_xyz(x1, y1, t1);
        let p2 = PoseSE2::from_xyz(x2, y2, t2);

        if let Some(path) = dubins_path_from_poses(&p1, &p2, radius) {
            // Total length must be non-negative
            assert!(
                path.length >= 0.0,
                "Dubins path length must be non-negative, got {}",
                path.length
            );

            // Each segment length must be non-negative
            for seg in &path.segments {
                assert!(
                    seg.length >= 0.0,
                    "Dubins segment length must be non-negative, got {}",
                    seg.length
                );
            }

            // Total length should be sum of segment lengths
            let total_segment_len: f64 = path.segments.iter().map(|s| s.length).sum();
            assert!(
                (path.length - total_segment_len).abs() < 1e-10,
                "Total length {} should equal sum of segments {}",
                path.length,
                total_segment_len
            );
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
        let radius = radius.abs().max(0.1);

        let p1 = PoseSE2::from_xyz(x1, y1, t1);

        // Two goals: close (offset 1.0) and far (offset 10.0)
        let p_close = PoseSE2::from_xyz(x1 + 1.0, y1, t1);
        let p_far = PoseSE2::from_xyz(x1 + 10.0, y1, t1);

        let path_close = dubins_path_from_poses(&p1, &p_close, radius);
        let path_far = dubins_path_from_poses(&p1, &p_far, radius);

        match (path_close, path_far) {
            (Some(close), Some(far)) => {
                assert!(
                    far.length >= close.length,
                    "Far path length {} should be >= close path length {}",
                    far.length,
                    close.length
                );
            }
            _ => {} // If either fails, we skip the assertion
        }
    }
}
