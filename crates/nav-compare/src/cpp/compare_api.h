#ifndef COMPARE_API_H
#define COMPARE_API_H

#ifdef __cplusplus
extern "C" {
#endif

// PoseSE2 constructor: returns (px, py, theta) via output params
void pose_se2_new(double x, double y, double theta,
                  double* px, double* py, double* ptheta);

// PoseSE2 addition
void pose_se2_add(double x1, double y1, double t1,
                  double x2, double y2, double t2,
                  double* px, double* py, double* ptheta);

// normalize_angle (mimics g2o::normalize_theta)
double c_normalize_angle(double angle);

#ifdef __cplusplus
}
#endif

#endif
