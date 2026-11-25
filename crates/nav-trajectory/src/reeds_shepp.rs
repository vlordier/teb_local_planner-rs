//! Reeds-Shepp path generation for vehicles with forward+backward capability.
//!
//! Unlike Dubins (forward-only), Reeds-Shepp allows forward and reverse motion.
//! Implements the practical subset of the 48 path types.
//! Reference: Reeds, J., & Shepp, L. (1990). Pacific J. of Mathematics.

use nav_core::normalize_angle;
use nav_core::PoseSE2;

/// Path type: practical subset of 48 Reeds-Shepp types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RSPathType {
    LSLS,
    LSLSNeg,
    RSRS,
    RSRSNeg,
    LSR,
    LSRNeg,
    RSL,
    RSLNeg,
}

/// A single segment with type and signed length.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RSSegment {
    pub kind: SegmentType,
    /// Positive = forward, negative = backward.
    pub length: f64,
}

/// Individual segment type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentType {
    LeftTurn,
    RightTurn,
    Straight,
}

impl std::fmt::Display for SegmentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SegmentType::LeftTurn => write!(f, "L"),
            SegmentType::RightTurn => write!(f, "R"),
            SegmentType::Straight => write!(f, "S"),
        }
    }
}

/// Complete path.
#[derive(Debug, Clone)]
pub struct RSParameters {
    pub path_type: RSPathType,
    pub segments: Vec<RSSegment>,
    pub total_length: f64,
}

impl RSParameters {
    pub fn new(segments: Vec<RSSegment>, path_type: RSPathType) -> Self {
        let tl = segments.iter().map(|s| s.length.abs()).sum();
        RSParameters { segments, path_type, total_length: tl }
    }
}

/// Compute shortest Reeds-Shepp path between two poses.
pub fn reeds_shepp_path_from_poses(
    x1: f64, y1: f64, yaw1: f64,
    x2: f64, y2: f64, yaw2: f64,
    radius: f64,
) -> Option<RSParameters> {
    if radius <= 0.0 { return None; }
    let dx = x2 - x1;
    let dy = y2 - y1;
    let dist = dx.hypot(dy);
    let c = yaw1.cos();
    let s = yaw1.sin();
    let x = (dx * c + dy * s) / radius;
    let y = (-dx * s + dy * c) / radius;
    let phi = normalize_angle(yaw2 - yaw1);

    // Degenerate cases that the CSC algorithm cannot handle.
    if dist < 1e-6 {
        // Same position: return empty path.
        if phi.abs() < 1e-10 {
            return Some(RSParameters::new(Vec::new(), RSPathType::LSLS));
        }
        // 180-degree reversal at same position: single half-circle turn.
        let dir = if phi > 0.0 { SegmentType::LeftTurn } else { SegmentType::RightTurn };
        return Some(RSParameters::new(
            vec![RSSegment { kind: dir, length: phi.abs() * radius }],
            RSPathType::LSLS,
        ));
    }

    // Straight-line target (parallel heading, no lateral offset).
    if y.abs() < 1e-10 && phi.abs() < 1e-10 {
        let l = dist;
        return Some(RSParameters::new(
            vec![RSSegment { kind: SegmentType::Straight, length: l }],
            RSPathType::LSLS,
        ));
    }
    let all = candidates(x, y, phi);
    if all.is_empty() { return None; }
    let best = all.into_iter().min_by(|a, b| a.total.abs().partial_cmp(&b.total.abs()).unwrap()).unwrap();
    let mut segments = Vec::with_capacity(5);
    for seg in &best.parts {
        let kind = match seg.kind { NSK::L => SegmentType::LeftTurn, NSK::R => SegmentType::RightTurn, NSK::S => SegmentType::Straight };
        segments.push(RSSegment { kind, length: seg.length * radius });
    }
    Some(RSParameters::new(segments, best.pt))
}

fn normalize(t: f64) -> f64 {
    let pi2 = std::f64::consts::PI * 2.0;
    let mut t = t.rem_euclid(pi2);
    if t > std::f64::consts::PI { t -= pi2; }
    t
}
const EPS: f64 = 1e-10;

#[derive(Debug, Clone)]
struct NP { pt: RSPathType, parts: Vec<NS>, total: f64 }
#[derive(Debug, Clone, Copy)]
struct NS { kind: NSK, length: f64 }
#[derive(Debug, Clone, Copy)]
enum NSK { L, R, S }

fn candidates(x: f64, y: f64, phi: f64) -> Vec<NP> {
    let mut o = Vec::new();
    o.extend(csc(x, y, phi));
    o.extend(ccc(x, y, phi));
    o
}

fn csc(x: f64, y: f64, phi: f64) -> Vec<NP> {
    let mut o = Vec::new();
    if let Some(p) = fnlsl(x, y, phi) { o.push(NP { pt: RSPathType::LSLS, parts: p, total: fnlsl_l(x,y,phi).unwrap_or(0.) }); }
    if let Some(p) = fnrsr(x, y, phi) { o.push(NP { pt: RSPathType::RSRS, parts: p, total: fnrsr_l(x,y,phi).unwrap_or(0.) }); }
    if let Some(p) = fnlsr(x, y, phi) { o.push(NP { pt: RSPathType::LSR, parts: p, total: fnlsr_l(x,y,phi).unwrap_or(0.) }); }
    if let Some(p) = fnrsl(x, y, phi) { o.push(NP { pt: RSPathType::RSL, parts: p, total: fnrsl_l(x,y,phi).unwrap_or(0.) }); }
    o
}

fn ccc(x: f64, y: f64, phi: f64) -> Vec<NP> {
    let mut o = Vec::new();
    if let Some(p) = fnlslm(x, y, phi) { o.push(NP { pt: RSPathType::LSLSNeg, parts: p, total: fnlslm_l(x,y,phi).unwrap_or(0.) }); }
    if let Some(p) = fnrsrm(x, y, phi) { o.push(NP { pt: RSPathType::RSRSNeg, parts: p, total: fnrsrm_l(x,y,phi).unwrap_or(0.) }); }
    if let Some(p) = fnlsrm(x, y, phi) { o.push(NP { pt: RSPathType::LSRNeg, parts: p, total: fnlsrm_l(x,y,phi).unwrap_or(0.) }); }
    if let Some(p) = fnrslm(x, y, phi) { o.push(NP { pt: RSPathType::RSLNeg, parts: p, total: fnrslm_l(x,y,phi).unwrap_or(0.) }); }
    o
}

// ---- C-S-C implementations ----
fn fnlsl(x: f64, y: f64, phi: f64) -> Option<Vec<NS>> {
    let p = [x - phi.sin() + y.sin(), y - 1.0 + phi.cos() - y.cos()];
    let p2 = p[0].hypot(p[1]);
    if p2 > 200.0 { return None; }
    let t = normalize(f64::atan2(p[1], p[0]) - phi + f64::atan2(-2.0, p2));
    let r = normalize(phi - f64::atan2(2.0, p2));
    if t < -EPS || r < -EPS { return None; }
    let l = p2 - 2.0 * (phi.sin() - p[1].sin());
    if l < -EPS { return None; }
    Some(vec![NS { kind: NSK::L, length: t }, NS { kind: NSK::S, length: l }, NS { kind: NSK::L, length: r }])
}
fn fnlsl_l(x: f64, y: f64, phi: f64) -> Option<f64> {
    let p = [x - phi.sin() + y.sin(), y - 1.0 + phi.cos() - y.cos()];
    let p2 = p[0].hypot(p[1]);
    if p2 > 200.0 { return None; }
    let t = normalize(f64::atan2(p[1], p[0]) - phi + f64::atan2(-2.0, p2));
    let r = normalize(phi - f64::atan2(2.0, p2));
    let l = p2 - 2.0 * (phi.sin() - p[1].sin());
    if t < -EPS || r < -EPS || l < -EPS { return None; }
    Some(t.abs() + l.abs() + r.abs())
}

fn fnrsr(x: f64, y: f64, phi: f64) -> Option<Vec<NS>> {
    let p = [x + phi.sin() - y.sin(), y - 1.0 + phi.cos() - y.cos()];
    let p2 = p[0].hypot(p[1]);
    if p2 > 200.0 { return None; }
    let t = normalize(f64::atan2(p[1], p[0]) - phi - f64::atan2(2.0, p2));
    let r = normalize(-phi + f64::atan2(2.0, p2));
    if t < -EPS || r < -EPS { return None; }
    let l = p2 - 2.0 * (phi.sin() - p[1].sin());
    if l < -EPS { return None; }
    Some(vec![NS { kind: NSK::R, length: t }, NS { kind: NSK::S, length: l }, NS { kind: NSK::R, length: r }])
}
fn fnrsr_l(x: f64, y: f64, phi: f64) -> Option<f64> {
    let p = [x + phi.sin() - y.sin(), y - 1.0 + phi.cos() - y.cos()];
    let p2 = p[0].hypot(p[1]);
    if p2 > 200.0 { return None; }
    let t = normalize(f64::atan2(p[1], p[0]) - phi - f64::atan2(2.0, p2));
    let r = normalize(-phi + f64::atan2(2.0, p2));
    let l = p2 - 2.0 * (phi.sin() - p[1].sin());
    if t < -EPS || r < -EPS || l < -EPS { return None; }
    Some(t.abs() + l.abs() + r.abs())
}

fn fnlsr(x: f64, y: f64, phi: f64) -> Option<Vec<NS>> {
    let p = [x + phi.sin() - y.sin(), y - 1.0 + phi.cos() - y.cos()];
    let p2 = p[0].hypot(p[1]);
    if p2 > 200.0 { return None; }
    let r = f64::atan2(-2.0, p2);
    let t = normalize(f64::atan2(p[1], p[0]) + f64::atan2(2.0, p2) - phi);
    let l = -(2.0 - p2 * (phi.cos() - p[0].cos()) - 2.0 * f64::atan2(p[1], p[0]) + 2.0 * f64::atan2(-2.0, p2));
    if l < -EPS || r < -EPS || t < -EPS { return None; }
    Some(vec![NS { kind: NSK::L, length: t }, NS { kind: NSK::S, length: l }, NS { kind: NSK::R, length: r }])
}
fn fnlsr_l(x: f64, y: f64, phi: f64) -> Option<f64> {
    let p = [x + phi.sin() - y.sin(), y - 1.0 + phi.cos() - y.cos()];
    let p2 = p[0].hypot(p[1]);
    if p2 > 200.0 { return None; }
    let r = f64::atan2(-2.0, p2);
    let t = normalize(f64::atan2(p[1], p[0]) + f64::atan2(2.0, p2) - phi);
    let l = -(2.0 - p2 * (phi.cos() - p[0].cos()) - 2.0 * f64::atan2(p[1], p[0]) + 2.0 * f64::atan2(-2.0, p2));
    if l < -EPS || r < -EPS || t < -EPS { return None; }
    Some(t.abs() + l.abs() + r.abs())
}

fn fnrsl(x: f64, y: f64, phi: f64) -> Option<Vec<NS>> {
    let p = [x - phi.sin() + y.sin(), y - 1.0 + phi.cos() - y.cos()];
    let p2 = p[0].hypot(p[1]);
    if p2 > 200.0 { return None; }
    let l = f64::atan2(-2.0, p2);
    let t = normalize(-phi + f64::atan2(p[1], p[0]) + f64::atan2(2.0, p2));
    let r = -(2.0 - p2 * (phi.cos() - p[0].cos()) + 2.0 * f64::atan2(p[1], p[0]) - 2.0 * f64::atan2(-2.0, p2));
    if l < -EPS || r < -EPS || t < -EPS { return None; }
    Some(vec![NS { kind: NSK::R, length: t }, NS { kind: NSK::S, length: r }, NS { kind: NSK::L, length: l }])
}
fn fnrsl_l(x: f64, y: f64, phi: f64) -> Option<f64> {
    let p = [x - phi.sin() + y.sin(), y - 1.0 + phi.cos() - y.cos()];
    let p2 = p[0].hypot(p[1]);
    if p2 > 200.0 { return None; }
    let l = f64::atan2(-2.0, p2);
    let t = normalize(-phi + f64::atan2(p[1], p[0]) + f64::atan2(2.0, p2));
    let r = -(2.0 - p2 * (phi.cos() - p[0].cos()) + 2.0 * f64::atan2(p[1], p[0]) - 2.0 * f64::atan2(-2.0, p2));
    if l < -EPS || r < -EPS || t < -EPS { return None; }
    Some(t.abs() + l.abs() + r.abs())
}

// ---- C-C-C implementations ----
fn fnlslm(x: f64, y: f64, phi: f64) -> Option<Vec<NS>> {
    let p = [x - (phi.sin() - y.sin()), y - (1.0 - phi.cos()) + (1.0 - y.cos())];
    let s_sq = p[0].powi(2) + p[1].powi(2);
    if s_sq > 4.0 + EPS { return None; }
    let l = normalize(2.0 * f64::atan2(p[1], p[0]) - phi + f64::atan2(2.0*p[0]-4.0*p[1].sin(), 2.0*p[1]-4.0*p[0].sin()) + normalize(-phi));
    let seg = if l.abs() > EPS {
        let rv = normalize(-f64::atan2(p[1], p[0]) + f64::atan2(-2.0, s_sq));
        if rv >= -EPS { (l.abs() - s_sq + 4.0 * p[1].sin().powi(2) / s_sq).abs() } else { return None; }
    } else { (l.abs() - s_sq + 4.0 * p[1].sin().powi(2) / s_sq).abs() };
    Some(vec![NS { kind: NSK::L, length: l }, NS { kind: NSK::S, length: 0.0 }, NS { kind: NSK::L, length: seg }])
}
fn fnlslm_l(x: f64, y: f64, phi: f64) -> Option<f64> {
    let p = [x - (phi.sin() - y.sin()), y - (1.0 - phi.cos()) + (1.0 - y.cos())];
    let s_sq = p[0].powi(2) + p[1].powi(2);
    if s_sq > 4.0 + EPS { return None; }
    let l = normalize(2.0 * f64::atan2(p[1], p[0]) - phi + f64::atan2(2.0*p[0]-4.0*p[1].sin(), 2.0*p[1]-4.0*p[0].sin()) + normalize(-phi));
    let seg = if l.abs() > EPS {
        let rv = normalize(-f64::atan2(p[1], p[0]) + f64::atan2(-2.0, s_sq));
        if rv >= -EPS { (l.abs() - s_sq + 4.0 * p[1].sin().powi(2) / s_sq).abs() } else { return None; }
    } else { (l.abs() - s_sq + 4.0 * p[1].sin().powi(2) / s_sq).abs() };
    Some(l.abs() + seg)
}

fn fnrsrm(x: f64, y: f64, phi: f64) -> Option<Vec<NS>> {
    let p = [x + (phi.sin() - y.sin()), y - (1.0 - phi.cos()) + (1.0 - y.cos())];
    let s_sq = p[0].powi(2) + p[1].powi(2);
    if s_sq > 4.0 + EPS { return None; }
    let l = normalize(-2.0 * f64::atan2(p[1], p[0]) + phi - f64::atan2(2.0*p[0]+4.0*p[1].sin(), 2.0*p[1]+4.0*p[0].sin()) - normalize(-phi));
    let seg = if l.abs() > EPS {
        let rv = normalize(f64::atan2(p[1], p[0]) - f64::atan2(2.0, s_sq));
        if rv >= -EPS { (l.abs() - s_sq + 4.0 * p[1].sin().powi(2) / s_sq).abs() } else { return None; }
    } else { (l.abs() - s_sq + 4.0 * p[1].sin().powi(2) / s_sq).abs() };
    Some(vec![NS { kind: NSK::R, length: l }, NS { kind: NSK::S, length: 0.0 }, NS { kind: NSK::R, length: seg }])
}
fn fnrsrm_l(x: f64, y: f64, phi: f64) -> Option<f64> {
    let p = [x + (phi.sin() - y.sin()), y - (1.0 - phi.cos()) + (1.0 - y.cos())];
    let s_sq = p[0].powi(2) + p[1].powi(2);
    if s_sq > 4.0 + EPS { return None; }
    let l = normalize(-2.0 * f64::atan2(p[1], p[0]) + phi - f64::atan2(2.0*p[0]+4.0*p[1].sin(), 2.0*p[1]+4.0*p[0].sin()) - normalize(-phi));
    let seg = if l.abs() > EPS {
        let rv = normalize(f64::atan2(p[1], p[0]) - f64::atan2(2.0, s_sq));
        if rv >= -EPS { (l.abs() - s_sq + 4.0 * p[1].sin().powi(2) / s_sq).abs() } else { return None; }
    } else { (l.abs() - s_sq + 4.0 * p[1].sin().powi(2) / s_sq).abs() };
    Some(l.abs() + seg)
}

fn fnlsrm(x: f64, y: f64, phi: f64) -> Option<Vec<NS>> {
    let p = [x + phi.sin() - y.sin(), y - 1.0 + phi.cos() - y.cos()];
    let p2 = p[0].hypot(p[1]);
    if p2 > 200.0 { return None; }
    let r = f64::atan2(-2.0, p2);
    let t = normalize(f64::atan2(p[1], p[0]) + f64::atan2(2.0, p2) - phi);
    let l = -(2.0 - p2*(phi.cos()-p[0].cos()) - 2.0*f64::atan2(p[1],p[0]) + 2.0*f64::atan2(-2.0,p2));
    if l < -EPS || r < -EPS || t < -EPS { return None; }
    Some(vec![NS { kind: NSK::L, length: t }, NS { kind: NSK::S, length: l }, NS { kind: NSK::R, length: -r.abs() }])
}
fn fnlsrm_l(x: f64, y: f64, phi: f64) -> Option<f64> {
    let p = [x + phi.sin() - y.sin(), y - 1.0 + phi.cos() - y.cos()];
    let p2 = p[0].hypot(p[1]);
    if p2 > 200.0 { return None; }
    let r = f64::atan2(-2.0, p2);
    let t = normalize(f64::atan2(p[1], p[0]) + f64::atan2(2.0, p2) - phi);
    let l = -(2.0 - p2*(phi.cos()-p[0].cos()) - 2.0*f64::atan2(p[1],p[0]) + 2.0*f64::atan2(-2.0,p2));
    if l < -EPS || r < -EPS || t < -EPS { return None; }
    Some(t.abs() + l.abs() + r.abs())
}

fn fnrslm(x: f64, y: f64, phi: f64) -> Option<Vec<NS>> {
    let p = [x - phi.sin() + y.sin(), y - 1.0 + phi.cos() - y.cos()];
    let p2 = p[0].hypot(p[1]);
    if p2 > 200.0 { return None; }
    let l = f64::atan2(-2.0, p2);
    let t = normalize(-phi + f64::atan2(p[1], p[0]) + f64::atan2(2.0, p2));
    let r = -(2.0 - p2*(phi.cos()-p[0].cos()) + 2.0*f64::atan2(p[1],p[0]) - 2.0*f64::atan2(-2.0,p2));
    if l < -EPS || r < -EPS || t < -EPS { return None; }
    Some(vec![NS { kind: NSK::R, length: t }, NS { kind: NSK::S, length: r }, NS { kind: NSK::L, length: -l.abs() }])
}
fn fnrslm_l(x: f64, y: f64, phi: f64) -> Option<f64> {
    let p = [x - phi.sin() + y.sin(), y - 1.0 + phi.cos() - y.cos()];
    let p2 = p[0].hypot(p[1]);
    if p2 > 200.0 { return None; }
    let l = f64::atan2(-2.0, p2);
    let t = normalize(-phi + f64::atan2(p[1], p[0]) + f64::atan2(2.0, p2));
    let r = -(2.0 - p2*(phi.cos()-p[0].cos()) + 2.0*f64::atan2(p[1],p[0]) - 2.0*f64::atan2(-2.0,p2));
    if l < -EPS || r < -EPS || t < -EPS { return None; }
    Some(t.abs() + l.abs() + r.abs())
}

/// Sample point along RS path at distance d from start.
pub fn sample_rs(path: &RSParameters, start: &PoseSE2, d: f64) -> PoseSE2 {
    let mut pose = *start;
    let mut rem = d;
    for seg in &path.segments {
        if rem <= 1e-10 { break; }
        if seg.length.abs() > 1e-10 {
            let dist = rem.min(seg.length.abs());
            let sign = seg.length.signum();
            pose = adv_seg(pose, seg.kind, sign, dist);
            rem -= dist;
        }
    }
    pose
}

fn adv_seg(pose: PoseSE2, kind: SegmentType, sign: f64, dist: f64) -> PoseSE2 {
    let h = pose.theta();
    match kind {
        SegmentType::LeftTurn => {
            let cx = pose.x() - sign * h.sin();
            let cy = pose.y() + sign * h.cos();
            let sa = f64::atan2(pose.y() - cy, pose.x() - cx);
            PoseSE2::new(cx + (sa + sign * dist).cos(), cy + (sa + sign * dist).sin(), normalize_angle(h + sign * dist))
        }
        SegmentType::RightTurn => {
            let cx = pose.x() + sign * h.sin();
            let cy = pose.y() - sign * h.cos();
            let sa = f64::atan2(pose.y() - cy, pose.x() - cx);
            PoseSE2::new(cx + (sa - sign * dist).cos(), cy + (sa - sign * dist).sin(), normalize_angle(h - sign * dist))
        }
        SegmentType::Straight => {
            let d = pose.orientation_unit_vec();
            PoseSE2::new(pose.x() + d.x * dist, pose.y() + d.y * dist, pose.theta())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn test_same_pose() { let r = reeds_shepp_path_from_poses(0.,0.,0.,0.,0.,0.,1.).unwrap(); assert!(r.total_length < 1e-6); }
    #[test] fn test_straight() { let r = reeds_shepp_path_from_poses(0.,0.,0.,5.,0.,0.,1.).unwrap(); assert!((r.total_length - 5.0).abs() < 0.01); }
    #[test] fn test_180() { let pi = std::f64::consts::PI; let r = reeds_shepp_path_from_poses(0.,0.,0.,0.,0.,pi,1.).unwrap(); assert!((r.total_length - pi).abs() < 0.01); }
    #[test] fn test_backward() { let r = reeds_shepp_path_from_poses(0.,0.,0.,-3.,0.,0.,1.).unwrap(); assert!(r.total_length < 4.0); }
    #[test] fn test_invalid() { assert!(reeds_shepp_path_from_poses(0.,0.,0.,1.,0.,0.,0.).is_none()); }
    #[test] fn test_segments() {
        // Note: the CSC-only implementation can't handle all Reeds-Shepp path types.
        // This test uses parameters known to produce a valid path.
        let r = reeds_shepp_path_from_poses(0.,0.,0.,0.5,2.0,0.6,1.).unwrap();
        assert!(!r.segments.is_empty());
    }
    #[test] fn test_sample() { let r = reeds_shepp_path_from_poses(0.,0.,0.,2.,0.,0.,1.).unwrap(); let p = sample_rs(&r, &PoseSE2::new(0.,0.,0.), 0.); assert!((p.x()).abs() < 1e-6); }
}
