extern "C" {
    fn pose_se2_new(x: f64, y: f64, theta: f64, px: *mut f64, py: *mut f64, ptheta: *mut f64);
    fn pose_se2_add(x1: f64, y1: f64, t1: f64, x2: f64, y2: f64, t2: f64, px: *mut f64, py: *mut f64, ptheta: *mut f64);
    fn normalize_angle(angle: f64) -> f64;
}

pub fn pose_se2_x(x: f64, y: f64, theta: f64) -> f64 {
    let mut px: f64 = 0.0;
    let mut py: f64 = 0.0;
    let mut ptheta: f64 = 0.0;
    unsafe { pose_se2_new(x, y, theta, &mut px, &mut py, &mut ptheta) }
    px
}

pub fn pose_se2_y(x: f64, y: f64, theta: f64) -> f64 {
    let mut px: f64 = 0.0;
    let mut py: f64 = 0.0;
    let mut ptheta: f64 = 0.0;
    unsafe { pose_se2_new(x, y, theta, &mut px, &mut py, &mut ptheta) }
    py
}

pub fn pose_se2_theta(x: f64, y: f64, theta: f64) -> f64 {
    let mut px: f64 = 0.0;
    let mut py: f64 = 0.0;
    let mut ptheta: f64 = 0.0;
    unsafe { pose_se2_new(x, y, theta, &mut px, &mut py, &mut ptheta) }
    ptheta
}

pub fn pose_se2_add_x(x1: f64, y1: f64, t1: f64, x2: f64, y2: f64, t2: f64) -> f64 {
    let mut px: f64 = 0.0;
    let mut py: f64 = 0.0;
    let mut ptheta: f64 = 0.0;
    unsafe { pose_se2_add(x1, y1, t1, x2, y2, t2, &mut px, &mut py, &mut ptheta) }
    px
}

pub fn pose_se2_add_y(x1: f64, y1: f64, t1: f64, x2: f64, y2: f64, t2: f64) -> f64 {
    let mut px: f64 = 0.0;
    let mut py: f64 = 0.0;
    let mut ptheta: f64 = 0.0;
    unsafe { pose_se2_add(x1, y1, t1, x2, y2, t2, &mut px, &mut py, &mut ptheta) }
    py
}

pub fn pose_se2_add_theta(x1: f64, y1: f64, t1: f64, x2: f64, y2: f64, t2: f64) -> f64 {
    let mut px: f64 = 0.0;
    let mut py: f64 = 0.0;
    let mut ptheta: f64 = 0.0;
    unsafe { pose_se2_add(x1, y1, t1, x2, y2, t2, &mut px, &mut py, &mut ptheta) }
    ptheta
}


pub fn normalize_angle_c(angle: f64) -> f64 {
    unsafe { normalize_angle(angle) }
}