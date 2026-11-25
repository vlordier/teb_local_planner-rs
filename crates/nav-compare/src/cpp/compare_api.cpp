#include "compare_api.h"
#include <cmath>
#include <algorithm>

// Minimal g2o::normalize_theta implementation (from teb_local_planner misc.h)
double normalize_angle(double angle) {
    double two_pi = 2.0 * M_PI;
    angle = fmod(angle, two_pi);
    if (angle < -M_PI) {
        angle += two_pi;
    } else if (angle >= M_PI) {
        angle -= two_pi;
    }
    return angle;
}

void pose_se2_new(double x, double y, double theta,
                  double* px, double* py, double* ptheta) {
    *px = x;
    *py = y;
    *ptheta = normalize_angle(theta);
}

void pose_se2_add(double x1, double y1, double t1,
                  double x2, double y2, double t2,
                  double* px, double* py, double* ptheta) {
    *px = x1 + x2;
    *py = y1 + y2;
    *ptheta = normalize_angle(t1 + t2);
}
