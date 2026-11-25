//! Kani verification properties for Reeds-Shepp path generation.
//!
//! NOTE: The Reeds-Shepp implementation calls FFI math functions (`hypot`,
//! `atan2`, `cos`, `sin`) that Kani/CBMC cannot model. These harnesses use
//! concrete stub paths (via `RSParameters::new`) to verify pure-Rust code
//! paths: segment structure, total length computation, and backward-segment
//! handling.
//!
//! Unsupported FFI functions: `hypot`, `atan2`, `cos`, `sin` —
//! see https://github.com/model-checking/kani/issues/2423

#![no_main]

mod reeds_shepp_kani {
    use nav_trajectory::{RSParameters, RSPathType, RSSegment, SegmentType};

    /// Verify RS path segment structure: total_length equals sum of abs(segment.length).
    #[kani::proof]
    #[kani::unwind(4)]
    fn reeds_shepp_total_length_consistency() {
        let segments = vec![
            RSSegment { kind: SegmentType::LeftTurn, length: 2.0 },
            RSSegment { kind: SegmentType::Straight, length: 1.5 },
            RSSegment { kind: SegmentType::RightTurn, length: 2.0 },
        ];
        let expected = segments.iter().map(|s| s.length.abs()).sum::<f64>();
        let path = RSParameters::new(segments, RSPathType::LSR);

        assert!(
            (path.total_length - expected).abs() < 1e-12,
            "total_length ({}) should equal sum of abs lengths ({})",
            path.total_length,
            expected
        );
    }

    /// Verify RS path with backward (negative-length) segments: total_length is always non-negative.
    #[kani::proof]
    #[kani::unwind(4)]
    fn reeds_shepp_backward_segments_consistency() {
        let segments = vec![
            RSSegment { kind: SegmentType::LeftTurn, length: 1.0 },
            RSSegment { kind: SegmentType::Straight, length: -0.5 },
            RSSegment { kind: SegmentType::RightTurn, length: 1.0 },
        ];
        let path = RSParameters::new(segments, RSPathType::LSR);

        assert!(path.total_length >= 0.0);
        assert!(path.total_length.is_finite());
    }
}
