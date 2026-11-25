
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

extern crate nav_core;
extern crate nav_trajectory;

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
    /// Verify RS path with empty segments: total_length = 0.
    #[kani::proof]
    #[kani::unwind(4)]
    fn reeds_shepp_empty_path_total_length() {
        let segments: Vec<RSSegment> = vec![];
        let path = RSParameters::new(segments, RSPathType::LSLS);

        assert!(
            (path.total_length - 0.0).abs() < 1e-12,
            "empty path total_length should be 0, got {}",
            path.total_length
        );
        assert!(path.segments.is_empty(), "empty path should have no segments");
    }

    /// Verify RS path with a single segment: total_length equals abs of that segment's length.
    #[kani::proof]
    #[kani::unwind(4)]
    fn reeds_shepp_single_segment_total_length() {
        let segments = vec![
            RSSegment { kind: SegmentType::LeftTurn, length: -3.14159 },
        ];
        let path = RSParameters::new(segments, RSPathType::LSLS);

        assert!(
            (path.total_length - 3.14159).abs() < 1e-10,
            "single segment total_length should be 3.14159, got {}",
            path.total_length
        );
        assert_eq!(path.segments.len(), 1, "single segment path should have exactly 1 segment");
    }

    /// Verify RS path with mixed positive/negative segments: total_length is sum of absolute values.
    #[kani::proof]
    #[kani::unwind(8)]
    fn reeds_shepp_mixed_segments_total_length() {
        let segments = vec![
            RSSegment { kind: SegmentType::LeftTurn, length: 2.5 },
            RSSegment { kind: SegmentType::Straight, length: -1.0 },
            RSSegment { kind: SegmentType::RightTurn, length: -0.5 },
            RSSegment { kind: SegmentType::Straight, length: 3.0 },
            RSSegment { kind: SegmentType::LeftTurn, length: 1.5 },
        ];
        let path = RSParameters::new(segments, RSPathType::LSR);

        // total_length = 2.5 + 1.0 + 0.5 + 3.0 + 1.5 = 8.5
        let expected = 8.5;
        assert!(
            (path.total_length - expected).abs() < 1e-10,
            "total_length should be 8.5, got {}",
            path.total_length
        );
        assert!(path.total_length > 0.0, "total_length should be positive");
        assert!(path.total_length.is_finite(), "total_length should be finite");
    }

    /// Verify RS path segment types are preserved.
    #[kani::proof]
    #[kani::unwind(4)]
    fn reeds_shepp_segment_types_preserved() {
        let segments = vec![
            RSSegment { kind: SegmentType::LeftTurn, length: 1.0 },
            RSSegment { kind: SegmentType::Straight, length: 2.0 },
            RSSegment { kind: SegmentType::RightTurn, length: 0.5 },
            RSSegment { kind: SegmentType::Straight, length: -1.5 },
        ];
        let path = RSParameters::new(segments.clone(), RSPathType::LSR);

        assert_eq!(path.segments[0].kind, SegmentType::LeftTurn);
        assert_eq!(path.segments[1].kind, SegmentType::Straight);
        assert_eq!(path.segments[2].kind, SegmentType::RightTurn);
        assert_eq!(path.segments[3].kind, SegmentType::Straight);

        // Verify signed lengths preserved
        assert!((path.segments[0].length - 1.0).abs() < 1e-12);
        assert!((path.segments[1].length - 2.0).abs() < 1e-12);
        assert!((path.segments[2].length - 0.5).abs() < 1e-12);
        assert!((path.segments[3].length - (-1.5)).abs() < 1e-12);
    }

    /// Verify RS path with only straight segments.
    #[kani::proof]
    #[kani::unwind(4)]
    fn reeds_shepp_all_straight_segments() {
        let segments = vec![
            RSSegment { kind: SegmentType::Straight, length: 5.0 },
            RSSegment { kind: SegmentType::Straight, length: 3.0 },
        ];
        let path = RSParameters::new(segments, RSPathType::LSLS);

        assert!(
            (path.total_length - 8.0).abs() < 1e-12,
            "all straight total_length should be 8.0, got {}",
            path.total_length
        );
        assert!(path.segments.iter().all(|s| matches!(s.kind, SegmentType::Straight)));
    }

    /// Verify RS path with only turning segments.
    #[kani::proof]
    #[kani::unwind(4)]
    fn reeds_shepp_all_turn_segments() {
        let segments = vec![
            RSSegment { kind: SegmentType::LeftTurn, length: 1.5708 },
            RSSegment { kind: SegmentType::RightTurn, length: -1.5708 },
            RSSegment { kind: SegmentType::LeftTurn, length: 0.7854 },
        ];
        let path = RSParameters::new(segments, RSPathType::LSLS);

        let expected = 1.5708 + 1.5708 + 0.7854;
        assert!(
            (path.total_length - expected).abs() < 1e-8,
            "total_length should be ~{}, got {}",
            expected,
            path.total_length
        );
    }

    /// Verify RS path segment count matches construction.
    #[kani::proof]
    #[kani::unwind(16)]
    fn reeds_shepp_segment_count_preserved() {
        for count in 0..10 {
            let segments: Vec<RSSegment> = (0..count)
                .map(|i| RSSegment {
                    kind: match i % 3 {
                        0 => SegmentType::LeftTurn,
                        1 => SegmentType::RightTurn,
                        _ => SegmentType::Straight,
                    },
                    length: (i + 1) as f64,
                })
                .collect();
            let path = RSParameters::new(segments, RSPathType::LSR);
            assert_eq!(
                path.segments.len(),
                count,
                "segment count should be {}, got {}",
                count,
                path.segments.len()
            );
        }
    }
}
