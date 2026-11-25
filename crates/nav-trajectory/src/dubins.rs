//! Dubins path generation for forward-only vehicles.
//!
//! Port of the C++ teb_local_planner geometric path primitives.
//!
//! A Dubins path consists of circular arcs (L = left turn, R = right turn)
//! and straight-line segments (S), forming six possible path types:
//! LSL, RSR, LSR, RSL, RLR, LRL.
//!
//! The model assumes a vehicle with maximum curvature κ = 1/R (turning radius R).

use nav_core::normalize_angle;
// Port of C++ g2o::normalize_theta (now normalize_angle from nav_core::misc)
use nav_core::PoseSE2;


/// A Dubins path: sequence of segments connecting two poses with bounded curvature.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DubinsPath {
    /// The three segments (at most 3 non-zero segments).
    pub segments: [DubinsSegment; 3],
    /// Total path length.
    pub length: f64,
}

impl DubinsPath {
    pub fn new(segments: [DubinsSegment; 3], length: f64) -> Self {
        Self { segments, length }
    }

    /// Returns the number of non-zero segments.
    pub fn segment_count(&self) -> usize {
        self.segments.iter().filter(|s| s.length > 1e-10).count()
    }
}

/// A single Dubins segment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DubinsSegment {
    /// Segment type.
    pub kind: DubinsSegmentKind,
    /// Length of this segment (distance units).
    pub length: f64,
}

/// Segment type: left arc, right arc, or straight line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DubinsSegmentKind {
    Left,
    Right,
    Straight,
}

impl std::fmt::Display for DubinsSegmentKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DubinsSegmentKind::Left => write!(f, "L"),
            DubinsSegmentKind::Right => write!(f, "R"),
            DubinsSegmentKind::Straight => write!(f, "S"),
        }
    }
}

/// Normalized path (before scaling by radius).
#[derive(Debug, Clone, Copy)]
pub struct NormPath {
    pub parts: [NormSeg; 3],
    pub total: f64,
}

/// Normalized segment (kind, length).
#[derive(Debug, Clone, Copy)]
pub struct NormSeg(pub DubinsSegmentKind, pub f64);

/// Compute the shortest Dubins path between two poses.
///
/// # Arguments
/// * `p1` - Start pose (x, y, yaw)
/// * `p2` - Goal pose (x, y, yaw)
/// * `radius` - Minimum turning radius (must be > 0). Curvature κ = 1/radius.
///
/// Returns the shortest Dubins path among the six possible types,
/// or `None` if the poses are identical or radius <= 0.
pub fn dubins_path_from_poses(
    p1: &PoseSE2,
    p2: &PoseSE2,
    radius: f64,
) -> Option<DubinsPath> {
    if radius <= 0.0 {
        return None;
    }

    // Normalize: translate p1 to origin, rotate by -theta1, scale by 1/radius.
    let dx = p2.x() - p1.x();
    let dy = p2.y() - p1.y();
    let theta1 = p1.theta();
    let theta2 = p2.theta();
    // Identical poses: no Dubins path exists.
    if dx == 0.0 && dy == 0.0 {
        return None;
    }

    let c = theta1.cos();
    let s = theta1.sin();
    let x = (dx * c + dy * s) / radius;
    let y = (-dx * s + dy * c) / radius;
    let phi = normalize_angle(theta2 - theta1);


    // Special case: straight line between parallel poses
    if phi.abs() < 1e-10 && y.abs() < 1e-10 && x.abs() > 1e-10 {
        let l = x.abs();
        return Some(DubinsPath::new([
            DubinsSegment { kind: DubinsSegmentKind::Straight, length: l * radius },
            DubinsSegment { kind: DubinsSegmentKind::Straight, length: 0.0 },
            DubinsSegment { kind: DubinsSegmentKind::Straight, length: 0.0 },
        ], l * radius));
    }

    // Try all six path types, pick shortest
    let candidates: [Option<NormPath>; 6] = [
        lsl(x, y, phi),
        rsr(x, y, phi),
        lsr(x, y, phi),
        rsl(x, y, phi),
        rlr(x, y, phi),
        lrl(x, y, phi),
    ];

    let best = candidates
        .into_iter()
        .filter_map(|c| c)
        .min_by(|a, b| a.total.total_cmp(&b.total));

    best.map(|np| {
        // Clamp tiny negatives to zero (numerical precision from acos/sqrt)
        let s0 = np.parts[0].1.max(0.0);
        let s1 = np.parts[1].1.max(0.0);
        let s2 = np.parts[2].1.max(0.0);
        let total = s0 + s1 + s2;
        let segments = [
            DubinsSegment { kind: np.parts[0].0, length: s0 * radius },
            DubinsSegment { kind: np.parts[1].0, length: s1 * radius },
            DubinsSegment { kind: np.parts[2].0, length: s2 * radius },
        ];
        DubinsPath::new(segments, total * radius)
    })
}

/// Sample a pose along the Dubins path at distance `d` from start.
///
/// # Panics
/// Panics if `d` is outside [0, path.length].
pub fn sample_dubins(path: &DubinsPath, start: &PoseSE2, d: f64) -> PoseSE2 {
    assert!(
        d >= -1e-10 && d <= path.length + 1e-10,
        "d={} must be in [0, path.length={}]",
        d,
        path.length
    );

    let mut pose = *start;
    let mut remaining = d;

    for seg in &path.segments {
        if seg.length < 1e-10 || remaining <= 1e-10 {
            break;
        }

        if remaining >= seg.length - 1e-10 {
            pose = advance_segment(pose, seg.kind, seg.length);
            remaining -= seg.length;
        } else {
            return advance_segment(pose, seg.kind, remaining);
            break;
        }
    }


    pose
}

/// Advance a pose along a segment by distance `dist`.
fn advance_segment(pose: PoseSE2, kind: DubinsSegmentKind, dist: f64) -> PoseSE2 {
    match kind {
        DubinsSegmentKind::Left => {
            let cx = pose.x() - pose.theta().sin();
            let cy = pose.y() + pose.theta().cos();
            let start_angle = f64::atan2(pose.y() - cy, pose.x() - cx);
            let angle = dist;
            let current_angle = start_angle + angle;
            PoseSE2::new(
                cx + current_angle.cos(),
                cy + current_angle.sin(),
                normalize_angle(pose.theta() + angle),
            )
        }
        DubinsSegmentKind::Right => {
            let cx = pose.x() + pose.theta().sin();
            let cy = pose.y() - pose.theta().cos();
            let start_angle = f64::atan2(pose.y() - cy, pose.x() - cx);
            let angle = dist;
            let current_angle = start_angle - angle;
            PoseSE2::new(
                cx + current_angle.cos(),
                cy + current_angle.sin(),
                normalize_angle(pose.theta() - angle),
            )
        }
        DubinsSegmentKind::Straight => {
            let dir = pose.orientation_unit_vec();
            PoseSE2::new(
                pose.x() + dir.x * dist,
                pose.y() + dir.y * dist,
                pose.theta(),
            )
        }
    }
}

// ---- Normalized path computations (radius = 1) ----

fn lsl(x: f64, y: f64, phi: f64) -> Option<NormPath> {
    let p = [x - phi.sin() + y.sin(), y - 1.0 + phi.cos() - y.cos()];
    let p2 = p[0].hypot(p[1]);
    if p2 > 200.0 {
        return None;
    }

    let t = normalize_angle(
        f64::atan2(p[1], p[0]) - phi + f64::atan2(-2.0, p2),
    );
    let r = normalize_angle(phi - f64::atan2(2.0, p2));
    if t < -EPS || r < -EPS {
        return None;
    }
    let l = p2 - 2.0 * (phi.sin() - p[1].sin());
    if l < -EPS {
        return None;
    }

    Some(NormPath {
        parts: [
            NormSeg(DubinsSegmentKind::Left, t),
            NormSeg(DubinsSegmentKind::Straight, l),
            NormSeg(DubinsSegmentKind::Left, r),
        ],
        total: (t + l + r).abs(),
    })
}

fn rsr(x: f64, y: f64, phi: f64) -> Option<NormPath> {
    let p = [x + phi.sin() - y.sin(), y - 1.0 + phi.cos() - y.cos()];
    let p2 = p[0].hypot(p[1]);
    if p2 > 200.0 {
        return None;
    }

    let t = normalize_angle(
        f64::atan2(p[1], p[0]) - phi - f64::atan2(2.0, p2),
    );
    let r = normalize_angle(-phi + f64::atan2(2.0, p2));
    if t < -EPS || r < -EPS {
        return None;
    }
    let l = p2 - 2.0 * (phi.sin() - p[1].sin());
    if l < -EPS {
        return None;
    }

    Some(NormPath {
        parts: [
            NormSeg(DubinsSegmentKind::Right, t),
            NormSeg(DubinsSegmentKind::Straight, l),
            NormSeg(DubinsSegmentKind::Right, r),
        ],
        total: (t + l + r).abs(),
    })
}

fn lsr(x: f64, y: f64, phi: f64) -> Option<NormPath> {
    let p = [x + phi.sin() - y.sin(), y - 1.0 + phi.cos() - y.cos()];
    let p2 = p[0].hypot(p[1]);
    if p2 > 200.0 {
        return None;
    }

    let r = f64::atan2(-2.0, p2);
    let t = normalize_angle(
        f64::atan2(p[1], p[0]) + f64::atan2(2.0, p2) - phi,
    );
    let l = -(2.0 - p2 * (phi.cos() - p[0].cos()) - 2.0 * f64::atan2(p[1], p[0])
        + 2.0 * f64::atan2(-2.0, p2));

    if l < -EPS || r < -EPS || t < -EPS {
        return None;
    }

    Some(NormPath {
        parts: [
            NormSeg(DubinsSegmentKind::Left, t),
            NormSeg(DubinsSegmentKind::Straight, l),
            NormSeg(DubinsSegmentKind::Right, r),
        ],
        total: (t + l + r).abs(),
    })
}

fn rsl(x: f64, y: f64, phi: f64) -> Option<NormPath> {
    let p = [x - phi.sin() + y.sin(), y - 1.0 + phi.cos() - y.cos()];
    let p2 = p[0].hypot(p[1]);
    if p2 > 200.0 {
        return None;
    }

    let l = f64::atan2(-2.0, p2);
    let t = normalize_angle(
        -phi + f64::atan2(p[1], p[0]) + f64::atan2(2.0, p2),
    );
    let r = -(2.0 - p2 * (phi.cos() - p[0].cos()) + 2.0 * f64::atan2(p[1], p[0])
        - 2.0 * f64::atan2(-2.0, p2));

    if l < -EPS || r < -EPS || t < -EPS {
        return None;
    }

    Some(NormPath {
        parts: [
            NormSeg(DubinsSegmentKind::Right, t),
            NormSeg(DubinsSegmentKind::Straight, r),
            NormSeg(DubinsSegmentKind::Left, l),
        ],
        total: (t + l + r).abs(),
    })
}

fn rlr(x: f64, y: f64, phi: f64) -> Option<NormPath> {
    let delta = normalize_angle(-phi);
    let s_sq = (x - phi.sin() + y.sin()).powi(2)
        + (y - 1.0 + phi.cos() - y.cos()).powi(2);

    if s_sq > 4.0 + EPS {
        return None;
    }

    let l = normalize_angle(
        2.0 * f64::atan2(y - 1.0 + phi.cos() - y.cos(), x - phi.sin() + y.sin())
            - phi
            + f64::atan2(2.0 * (x - phi.sin() + y.sin()) - 4.0 * (y - 1.0 + phi.cos() - y.cos()).sin(),
                         2.0 * (y - 1.0 + phi.cos() - y.cos()) - 4.0 * (x - phi.sin() + y.sin()).sin())
            + delta,
    );

    let seg = (l.abs() - s_sq + 4.0 * (y - 1.0 + phi.cos() - y.cos()).sin().powi(2) / s_sq).abs();
    let total = l.abs() + seg + l.abs();

    Some(NormPath {
        parts: [
            NormSeg(DubinsSegmentKind::Right, l),
            NormSeg(DubinsSegmentKind::Straight, seg),
            NormSeg(DubinsSegmentKind::Right, l),
        ],
        total,
    })
}

fn lrl(x: f64, y: f64, phi: f64) -> Option<NormPath> {
    let delta = normalize_angle(-phi);
    let s_sq = (x + phi.sin() - y.sin()).powi(2)
        + (y + 1.0 - phi.cos() + y.cos()).powi(2);

    if s_sq > 4.0 + EPS {
        return None;
    }

    let l = normalize_angle(
        -2.0 * f64::atan2(y + 1.0 - phi.cos() + y.cos(), x + phi.sin() - y.sin())
            + phi
            - f64::atan2(2.0 * (x + phi.sin() - y.sin()) + 4.0 * (y + 1.0 - phi.cos() + y.cos()).sin(),
                         2.0 * (y + 1.0 - phi.cos() + y.cos()) + 4.0 * (x + phi.sin() - y.sin()).sin())
            - delta,
    );

    let seg = (l.abs() - s_sq + 4.0 * (y + 1.0 - phi.cos() + y.cos()).sin().powi(2) / s_sq).abs();
    let total = l.abs() + seg + l.abs();

    Some(NormPath {
        parts: [
            NormSeg(DubinsSegmentKind::Left, l),
            NormSeg(DubinsSegmentKind::Straight, seg),
            NormSeg(DubinsSegmentKind::Left, l),
        ],
        total,
    })
}

const EPS: f64 = 1e-10;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dubins_path_same_pose() {
        let p1 = PoseSE2::zero();
        let p2 = PoseSE2::zero();
        let path = dubins_path_from_poses(&p1, &p2, 1.0);
        assert!(path.is_none());
    }

    #[test]
    fn test_dubins_path_simple_straight() {
        let p1 = PoseSE2::zero();
        let p2 = PoseSE2::from_xyz(10.0, 0.0, 0.0);
        let path = dubins_path_from_poses(&p1, &p2, 5.0).unwrap();
        assert!(path.length > 0.0);
    }

    #[test]
    fn test_dubins_path_zero_radius() {
        let p1 = PoseSE2::zero();
        let p2 = PoseSE2::from_xyz(5.0, 0.0, 0.0);
        let path = dubins_path_from_poses(&p1, &p2, 0.0);
        assert!(path.is_none());
    }

    #[test]
    fn test_dubins_sample_end() {
        let p1 = PoseSE2::zero();
        let p2 = PoseSE2::from_xyz(10.0, 0.0, 0.0);
        let path = dubins_path_from_poses(&p1, &p2, 5.0).unwrap();
        let end = sample_dubins(&path, &p1, path.length);
        assert!((end.x() - p2.x()).abs() < 0.1);
        assert!((end.y() - p2.y()).abs() < 0.1);
    }
}
